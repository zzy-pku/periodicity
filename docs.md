
# 项目说明：FANFormer 对 `sin(x)` 的序列化建模实验

## 1. 项目目标

本项目用于复现论文附录 H 中的 hidden periodicity 实验，考察 **FANFormer** 在“把实数写成固定长度数字串后建模 `sin(x)`”这一设定上的表现，重点关注：

* 模型能否在训练区间内正确拟合 `sin(x)`
* 模型能否在训练区间外继续保持正确的周期外推

论文附录 H 已明确的约束如下：

* 训练区间为 `[-3π, 3π]`
* `x` 被写成固定 **10 位**字符串
* 使用 **Qwen2.5 tokenizer**
* 使用 **Qwen2.5 embedding**，并在训练期间冻结
* 输出使用相同的 digit alignment，并最终解码成数值

本仓库的这次实现遵循上述约束，并采用单独脚本完成，不干扰原有复合周期实验代码。

## 2. 任务定义

目标函数为：

`y = sin(x)`

模型学习从输入 `x` 的数字串到输出 `y` 的数字串的映射。

### 数据区间

* 训练集：`x ∈ [-3π, 3π]`
* ID 测试集：同样位于 `[-3π, 3π]`，但采样点与训练集不同
* OOD 测试集：`x ∈ [-6π, -3π) ∪ (3π, 6π]`

这个划分对应论文附录 H 的“训练区间内拟合 + 训练区间外外推”目标。

## 3. 输入与输出设计

### 3.1 固定 10 位格式

输入 `x` 和输出 `y = sin(x)` 统一格式化为固定 10 位字符串，例如：

* `+00.000000`
* `-03.141593`
* `+18.849556`

固定格式由以下位置组成：

* 第 1 位：符号位 `+` 或 `-`
* 第 2 到第 3 位：两位整数位
* 第 4 位：小数点 `.`
* 第 5 到第 10 位：6 位小数

接近 0 的值统一规范为 `+00.000000`，避免 `-00.000000` 带来无意义噪声。

之所以采用 `±DD.dddddd` 而不是 `±D.ddddddd`，是因为 OOD 测试区间扩展到了 `±6π ≈ ±18.85`。若仍使用单整数位格式，区间外样本将无法保持固定 10 位长度。

### 3.2 Tokenization

输入输出都使用 **Qwen2.5 tokenizer** 编码，且不添加额外 special tokens。

在本项目中，固定 10 位字符串会被 Qwen2.5 tokenizer 稳定切分为 10 个单字符 token，因此 digit alignment 可以完整保留。

### 3.3 输出空间

虽然输入使用 Qwen2.5 tokenizer 和 embedding，但输出字符集合仅包含：

* `+`
* `-`
* `.`
* `0` 到 `9`

训练时仅对这 13 个合法字符做分类，推理后再映射回对应字符并拼接成数字串。这样保持论文设定的同时，避免构造一个巨大的全词表输出层。

## 4. 模型方案

第一阶段仅复现 **FANFormer**，不做 Transformer 对照组。

具体实现口径如下：

* 使用仓库中的 `Qwen2.5Embedding-FANformer`
* `pretrained_name` 使用 `Qwen/Qwen2.5-0.5B`
* 冻结 pretrained embedding
* 保留 FANFormer 主体结构
* 对本任务关闭 causal mask，使每个输出位置都能访问完整输入串

关闭 causal mask 的原因是：本实验不是 next-token prediction，而是固定长度输入串到固定长度输出串的映射。若保留因果掩码，模型只能看到输入前缀，不符合该任务的监督形式。

## 5. 数据生成方案

数据生成分两步进行：

### 5.1 数值样本生成

首先在对应区间内均匀采样 `x`：

* 训练集在 `[-3π, 3π]` 上采样
* ID 测试集在 `[-3π, 3π]` 上另采样一组不重合点
* OOD 测试集在 `[-6π, -3π)` 与 `(3π, 6π]` 上采样

然后对每个样本计算：

* `y = sin(x)`

### 5.2 序列化

随后将 `(x, y)` 都格式化为固定 10 位字符串：

* 输入：`x_str`
* 标签：`y_str`

最终每条样本都是：

* `input_text = x_str`
* `target_text = y_str`
* `x_value = x`
* `y_value = sin(x)`

保留原始浮点值用于评估和画图。

## 6. 训练目标

本实验采用 **token-level Cross Entropy**，而不是数值回归 MSE。

原因如下：

* 模型输出是数字字符串序列，不是单个浮点数
* 每个位置都是一个离散字符分类问题
* 论文原文写法也更接近“输出数字串，再解码为数值”

训练时对输出序列每个位置做 13 分类，损失函数为逐位置交叉熵平均：

`L = mean_t CE(logits_t, label_t)`

如果未来改成直接预测浮点数，才考虑 MSE；本次复现不采用这一路线。

## 7. 优化器与超参数

第一阶段默认采用：

* 优化器：`AdamW`
* 学习率：`1e-5`
* 权重衰减：`0.01`

这些设置与本仓库现有主实验保持一致，适合作为附录 H 复现的起点。

训练脚本暴露以下可调项：

* 训练样本数
* ID/OOD 测试样本数
* epoch 数
* batch size
* FANFormer 层数
* hidden attention heads
* eval/save 频率

## 8. 评估指标与结果产物

第一阶段的结果产物聚焦在“训练跑通并生成附录 H 风格图像”，不做大规模 sweep 或多 seed 汇总。

### 8.1 评估指标

训练和验证时记录：

* token accuracy
* exact-match accuracy
* 数值 MAE
* 数值 MSE

其中数值误差在把预测字符串解码为浮点数后计算。

### 8.2 结果可视化

脚本需要生成：

* 交叉熵 loss 曲线
* 数值 MSE 曲线
* ID / OOD 上的数值预测散点或曲线图
* 可选的中间 epoch 预测曲线图，文件名包含 epoch 数
* 一份最后一个 epoch 的输入、目标、预测、解码结果和误差文件

目标是能够直观看到：

* 训练区间内是否拟合 `sin(x)`
* 训练区间外是否沿着真实周期继续外推

## 9. 第一阶段执行范围

本阶段只做以下工作：

* 将附录 H 的执行方案补充到本文档
* 编写独立训练与评估脚本
* 跑通最小可用实验
* 自测数据流、训练、解码和绘图是否正常

暂不做：

* Transformer 对照组
* 多随机种子统计
* 更大规模超参数搜索
* FANFormer 结构变体对比

## 10. 后续任务

在第一阶段完成后，再继续探索：

* FANFormer 结构修改对 hidden periodicity 的影响
* 是否需要加入 Transformer 对照
* 是否需要扩展为多 seed 和更接近论文图形的完整复现


python appendix_h_fanformer_sin.py     --output_dir ./outputs/smoke     --epochs 100     --eval_every 1     --train_size 2000     --id_test_size 400     --ood_test_size 400     --batch_size 32     --layers 2     --num_heads 8  --id_left -3.141592653589793  --id_right 3.141592653589793  --ood_left -6.283185307179586     --ood_right 6.283185307179586  --plot_every 10   ----pretrained_name ///   --model_name Qwen2.5Embedding-FANformer   --norm_first true


 python evaluate_serialized_sin_model.py \
    --checkpoint_dir outputs/smoke_04-23_15-33-24 \
    --output_dir /tmp/eval_probe_test \
    --id_test_size 32 \
    --ood_test_size 32 \
    --id_left -9.42477796076938 \
    --id_right 9.42477796076938 \
    --ood_left -18.84955592153876 \
    --ood_right 18.84955592153876 \
    --logit_probe_left -1.0 \
    --logit_probe_right 1.0 \
    --logit_probe_step 1.0