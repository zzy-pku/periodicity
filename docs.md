# 项目说明：序列化 `sin(x)` 实验与统一评测使用手册

## 1. 项目目标

本项目用于复现论文附录 H 中的 hidden periodicity 实验，研究在“把实数写成固定长度数字串后再建模 `sin(x)`”的设定下，模型是否能够：

* 在训练区间内正确拟合 `sin(x)`
* 在训练区间外保持正确的周期外推

当前实现已经扩展为一套完整流程：

* 训练脚本
* 单实验评测脚本
* 双实验对比脚本
* logit probe 分析
* 第二阶段 periodicity analysis

为了方便使用，所有评测逻辑现在统一整理在目录 [evaluation](/home/zzy/periodicity_generalization/evaluation:1) 下，并通过统一入口脚本 [run_serialized_sin_evaluations.py](/home/zzy/periodicity_generalization/run_serialized_sin_evaluations.py:1) 调用。

## 2. 任务定义

目标函数为：

`y = sin(x)`

模型学习的是：

* 输入：`x` 的固定长度数字串
* 输出：`sin(x)` 的固定长度数字串

默认区间定义为：

* 训练集：`x ∈ [-3π, 3π]`
* ID 测试集：同样在 `[-3π, 3π]`，但采样点与训练集不同
* OOD 测试集：`x ∈ [-6π, -3π) ∪ (3π, 6π]`

训练和评测脚本都支持通过命令行覆盖这些边界。

## 3. 输入输出表示

### 3.1 固定 10 位格式

输入和输出都写成固定 10 字符格式：

* `+00.000000`
* `-03.141593`
* `+18.849556`

格式为：

* 第 1 位：符号位 `+` / `-`
* 第 2-3 位：整数位
* 第 4 位：小数点 `.`
* 第 5-10 位：6 位小数

即：

`±DD.dddddd`

接近 0 的值统一规范为 `+00.000000`。

### 3.2 Tokenization

输入输出都使用 **Qwen2.5 tokenizer** 编码，并且：

* 不添加 special tokens
* embedding 来自 Qwen2.5 预训练模型
* embedding 冻结

### 3.3 输出类别

输出层并不预测整个 Qwen 词表，而只预测 13 个合法字符：

* `+`
* `-`
* `.`
* `0-9`

同时通过位置 mask 约束每一位只能输出合法字符，因此解码结果一定满足：

`±DD.dddddd`

## 4. 模型实现

目前支持两种模型：

* `Qwen2.5Embedding-FANformer`
* `Qwen2.5Embedding-Transformer`

两者都：

* 复用 Qwen2.5 input embedding
* 冻结 embedding
* 使用固定长度输入串到固定长度输出串的映射
* 默认关闭 causal mask

当前任务不是自回归 next-token 预测，而是固定长度监督映射，所以 `causal=False` 更符合任务定义。

## 5. 数据生成逻辑

### 5.1 数值采样

先在指定区间采样 `x`，再计算：

`y = sin(x)`

### 5.2 序列化

每条样本都会保存：

* `x_value`
* `y_value`
* `input_text`
* `target_text`

其中：

* `input_text = format(x)`
* `target_text = format(sin(x))`

### 5.3 ID/OOD 切分

训练脚本和评测脚本都使用同一套切分逻辑：

* 训练集在 ID 区间均匀采样
* ID 测试集在同区间上使用 offset 网格，避免与训练点重合
* OOD 测试集在左右两侧区间采样

## 6. 训练目标与优化器

训练目标是 token-level Cross Entropy，而不是直接数值回归。

原因：

* 输出是数字字符串序列
* 每个位置都是离散字符分类任务
* 最后再把字符串解码成数值进行评估

默认优化器与超参数：

* Optimizer：`AdamW`
* `lr=1e-5`
* `weight_decay=0.01`

## 7. 训练脚本使用说明

训练脚本是 [appendix_h_fanformer_sin.py](/home/zzy/periodicity_generalization/appendix_h_fanformer_sin.py:1)。

### 7.1 常用命令

最基础的训练：

```bash
python appendix_h_fanformer_sin.py \
  --output_dir ./outputs/appendix_h_run
```

一个较完整的训练示例：

```bash
python appendix_h_fanformer_sin.py \
  --output_dir ./outputs/appendix_h_run \
  --pretrained_name /data/zzy/periodicity/models/Qwen/Qwen2.5-0.5B \
  --model_name Qwen2.5Embedding-FANformer \
  --epochs 100 \
  --eval_every 1 \
  --plot_every 10 \
  --train_size 12000 \
  --id_test_size 2000 \
  --ood_test_size 2000 \
  --batch_size 64 \
  --layers 5 \
  --num_heads 8 \
  --norm_first true \
  --id_left -9.42477796076938 \
  --id_right 9.42477796076938 \
  --ood_left -18.84955592153876 \
  --ood_right 18.84955592153876
```

### 7.2 主要参数

* `--output_dir`
  结果目录。训练脚本会自动在目录名后追加时间戳。
* `--pretrained_name`
  Qwen2.5 tokenizer / embedding 的来源，可以是模型名，也可以是本地目录。
* `--model_name`
  模型类型，可选：
  * `Qwen2.5Embedding-FANformer`
  * `Qwen2.5Embedding-Transformer`
* `--layers`
  主干网络层数
* `--num_heads`
  注意力头数
* `--norm_first`
  是否使用 pre-norm
* `--train_size`
  训练样本数
* `--id_test_size`
  ID 测试样本数
* `--ood_test_size`
  OOD 测试样本数
* `--epochs`
  训练轮数
* `--eval_every`
  每隔多少个 epoch 评测一次
* `--plot_every`
  每隔多少个 epoch 额外保存一张 `prediction_curve_epoch{N}.png`
* `--id_left`, `--id_right`
  ID 区间边界
* `--ood_left`, `--ood_right`
  OOD 区间边界

### 7.3 训练输出目录说明

训练完成后，目录中通常会包含：

* `config.json`
  本次训练的完整配置
* `dataset_preview.json`
  数据样本预览
* `metrics_history.json`
  每次评估时记录的完整指标历史
* `last_model.pt`
  最后一个 epoch 的模型权重
* `last_eval.json`
  最后一个 epoch 的 ID/OOD 评估结果摘要
* `prediction_curve.png`
  最后一个 epoch 的预测曲线图
* `prediction_curve_epoch{N}.png`
  训练中间 epoch 的预测曲线图，仅在命中 `plot_every` 时保存
* `cross_entropy_loss_curve.png`
  交叉熵曲线
* `mse_curve.png`
  数值 MSE 曲线

### 7.4 训练输出图像解释

`prediction_curve.png`

* 横轴：`x`
* 纵轴：真实值 / 预测值
* 显示 ID 和 OOD 背景区间
* 用于观察最终曲线形态

`prediction_curve_epoch{N}.png`

* 第 `N` 个 epoch 的预测曲线
* 用于观察训练过程中的拟合与崩坏过程

`cross_entropy_loss_curve.png`

* 横轴：epoch
* 纵轴：平均交叉熵
* 曲线：
  * `Train`
  * `ID`
  * `OOD`

`mse_curve.png`

* 横轴：epoch
* 纵轴：解码为浮点数之后的 MSE
* 曲线：
  * `ID MSE`
  * `OOD MSE`

## 8. 统一评测入口

统一入口脚本是 [run_serialized_sin_evaluations.py](/home/zzy/periodicity_generalization/run_serialized_sin_evaluations.py:1)。

它现在有两种模式：

* `single`
  对单个训练结果目录做完整评测
* `compare`
  对两个训练结果目录做对比分析

所有输出都会放进你指定的目录名后自动追加时间戳的新目录中。

## 9. 单实验评测模式

### 9.1 使用方法

最常见的用法：

```bash
python run_serialized_sin_evaluations.py \
  --checkpoint_dir /home/zzy/periodicity_generalization/outputs/Tr_expand12_500_04-28_11-36-35 \
  --output_dir ./outputs/Tr_expand12_eval
```

带 logit probe 和更细分析的例子：

```bash
python run_serialized_sin_evaluations.py \
  --checkpoint_dir /home/zzy/periodicity_generalization/outputs/Tr_expand12_500_04-28_11-36-35 \
  --output_dir ./outputs/Tr_expand12_eval \
  --id_test_size 2000 \
  --ood_test_size 2000 \
  --logit_probe_left -1.0 \
  --logit_probe_right 1.0 \
  --logit_probe_step 0.1 \
  --interval_width 3.141592653589793 \
  --points_per_interval 256 \
  --phase_points 256 \
  --max_k 6 \
  --shift_multiples 1,2,3,4 \
  --full_curve_points 4096
```

### 9.2 单实验模式输出结构

统一评测目录会包含：

* `evaluation_suite_config.json`
  本次统一评测的完整参数
* `evaluation_suite_summary.json`
  三个子评测模块的结果摘要
* `basic_eval/`
* `logit_probe/`
  如果传了 probe 参数才会生成
* `periodicity_analysis/`

### 9.3 `basic_eval/` 输出说明

`basic_eval/` 里包含：

* `prediction_curve.png`
* `cross_entropy_loss_curve.png`
* `mse_curve.png`
* `last_eval.json`

含义：

`prediction_curve.png`

* 当前 checkpoint 在指定 ID/OOD 区间上的真实曲线与预测曲线

`cross_entropy_loss_curve.png`

* 当前 checkpoint 的 ID / OOD 交叉熵分布图
* 这是单 checkpoint 的柱状图，不是训练过程折线图

`mse_curve.png`

* 当前 checkpoint 的 ID / OOD 数值 MSE 分布图

`last_eval.json`

* 当前 checkpoint 的：
  * `id_loss`
  * `ood_loss`
  * `id_metrics`
  * `ood_metrics`
  * 样本预览

### 9.4 `logit_probe/` 输出说明

只有传入以下 3 个参数时才会生成：

* `--logit_probe_left`
* `--logit_probe_right`
* `--logit_probe_step`

目录内容：

* `logit_probe.jsonl`
* `logit_probe_plot.png`

`logit_probe.jsonl`

每一行对应一个 probe 点，包含：

* `x_value`
* `y_value`
* `input_text`
* `target_text`
* `pred_text`
* `pred_value`
* `allowed_tokens`
* `raw_logits`
* `masked_logits`

`logit_probe_plot.png`

* 横轴：`x`
* 左轴：
  * 真实值 `y_value`
  * 预测值 `pred_value`
* 右轴：
  * `+` 的 logit
  * `-` 的 logit

用于分析符号位决策机制。

### 9.5 `periodicity_analysis/` 输出说明

该目录对应第二阶段分析，包含：

* `analysis_summary.json`
* `full_prediction_preview.json`
* `full_prediction_curve.png`
* `partition_mae.png`
* `partition_gap.png`
* `period_consistency.png`
* `half_period_metrics.png`
* `shift_matching_plus.png`
* `shift_matching_minus.png`
* `local_sine_fit_params.png`

#### `analysis_summary.json`

第二阶段分析的完整数值结果，包括：

* 分区间 OOD 误差
* 周期一致性
* 半周期关系
* shift matching
* 局部正弦拟合参数

#### `full_prediction_preview.json`

全域预测曲线前 200 个样本点的预览。

#### `full_prediction_curve.png`

* 横轴：全 OOD 范围内的 `x`
* 纵轴：真实值 / 预测值
* 用于观察更大范围上的整体预测形态

#### `partition_mae.png`

* 横轴：OOD 区间编号 `k`
* 纵轴：该区间上的 `MAE`
* 左右 OOD 分别画线

对应第二阶段分析里的 `E_k`。

#### `partition_gap.png`

* 横轴：OOD 区间编号 `k`
* 纵轴：`gap = E_k - E_0`

用来看 OOD 泛化差距。

#### `period_consistency.png`

* 横轴：周期编号 `k`
* 纵轴：`C_k`

用于判断模型是否保持 `2π` 周期一致性。

#### `half_period_metrics.png`

* 横轴：右侧 OOD 区间编号 `k`
* 两条曲线：
  * `C_pi_plus`
  * `C_pi_minus`

用于判断模型是否更像学到了错误的 `π` 周期复制，还是 `π` 取反关系。

#### `shift_matching_plus.png`

* 热力图
* 横轴：shift 倍数 `nπ`
* 纵轴：右侧 OOD 区间编号 `k`
* 值：`D_plus(Δ)`

用于分析 OOD 区间更像复制了训练域中的哪一段。

#### `shift_matching_minus.png`

* 热力图
* 横轴：shift 倍数 `nπ`
* 纵轴：右侧 OOD 区间编号 `k`
* 值：`D_minus(Δ)`

用于分析是否存在带符号的 shift 复制。

#### `local_sine_fit_params.png`

包含 4 个子图，对每个右侧 OOD 区间做：

`f_hat(x) ≈ A_k sin(omega_k x + phi_k) + c_k`

子图含义：

* 左上：`A_k`
  * 横轴：右侧 OOD 区间编号 `k`
  * 纵轴：局部拟合振幅
* 右上：`omega_k`
  * 横轴：右侧 OOD 区间编号 `k`
  * 纵轴：局部拟合频率
* 左下：`phi_k`
  * 横轴：右侧 OOD 区间编号 `k`
  * 纵轴：局部拟合相位
* 右下：`c_k`
  * 横轴：右侧 OOD 区间编号 `k`
  * 纵轴：局部拟合偏置

这张图用于区分模型 OOD 崩坏时主要错在：

* 振幅
* 周期长度
* 相位
* 偏置

## 10. 双实验对比模式

### 10.1 使用方法

最推荐的用法是直接传两个训练输出目录：

```bash
python run_serialized_sin_evaluations.py \
  --mode compare \
  --run_a outputs/exp_a \
  --run_b outputs/exp_b \
  --output_dir ./outputs/compare_a_b
```

也支持直接传两份 `metrics_history.json`：

```bash
python run_serialized_sin_evaluations.py \
  --mode compare \
  --metrics_a /path/to/exp_a/metrics_history.json \
  --metrics_b /path/to/exp_b/metrics_history.json \
  --label_a exp_a \
  --label_b exp_b \
  --output_dir ./outputs/compare_a_b
```

### 10.2 对比模式输出结构

会生成一个带时间戳目录，里面包含：

* `compare_suite_config.json`
* `compare_suite_summary.json`
* `metrics_history_compare/`
* `ood_id_gap_integral_compare/`

### 10.3 `metrics_history_compare/` 输出说明

目录中包含：

* `train_aligned.png`
* `id_aligned.png`

`train_aligned.png`

包含两个子图：

* 左图：`ID Loss vs Train Loss`
* 右图：`OOD Loss vs Train Loss`

用于回答：

* 当两次实验达到相同训练误差水平时
* 哪个实验的 ID 更低
* 哪个实验的 OOD 更低

`id_aligned.png`

包含两个子图：

* 左图：`Train Loss vs ID Loss`
* 右图：`OOD Loss vs ID Loss`

用于回答：

* 当两次实验达到相同 ID 水平时
* 哪个实验训练更容易
* 哪个实验 OOD 更好

### 10.4 `ood_id_gap_integral_compare/` 输出说明

目录中包含：

* `integral_curves.png`
* `difference_curve.png`
* `ratio_curve.png`

这里定义：

`g(epoch) = ood_loss(epoch) - id_loss(epoch)`

累计积分使用离散梯形积分近似：

`I_k = Σ 0.5 * (g_i + g_{i-1}) * (epoch_i - epoch_{i-1})`

#### `integral_curves.png`

* 横轴：epoch
* 纵轴：`I(epoch)`
* 两条曲线：两次实验各自的累计积分曲线

#### `difference_curve.png`

* 横轴：共同 epoch 网格
* 纵轴：`I_A - I_B`

用于比较两次实验在 OOD-ID gap 上的累积差异。

#### `ratio_curve.png`

* 横轴：共同 epoch 网格
* 纵轴：`I_A / I_B`

用于比较两次实验在累积 gap 上的比例关系。

## 11. 旧脚本兼容说明

根目录保留了以下脚本名：

* `evaluate_serialized_sin_model.py`
* `analyze_periodicity_generalization.py`
* `plot_logit_probe.py`
* `compare_metrics_history.py`
* `compare_ood_id_gap_integral.py`
* `run_serialized_sin_evaluations.py`

但这些脚本现在主要是 wrapper，真实实现都位于 [evaluation](/home/zzy/periodicity_generalization/evaluation:1)。

也就是说：

* 旧命令通常仍可运行
* 新开发与维护应以 `evaluation/` 下的文件为准

## 12. 推荐使用方式

如果你只是要做一个训练结果的完整分析，推荐直接：

```bash
python run_serialized_sin_evaluations.py \
  --checkpoint_dir /path/to/train_output_dir \
  --output_dir ./outputs/my_eval
```

如果你要比较两个训练实验，推荐直接：

```bash
python run_serialized_sin_evaluations.py \
  --mode compare \
  --run_a /path/to/exp_a \
  --run_b /path/to/exp_b \
  --output_dir ./outputs/my_compare
```

这样基本不需要再分别调用分散的评测脚本。所有结果都会自动放进统一的输出目录中。
