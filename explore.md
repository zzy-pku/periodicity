
# 周期性文本建模 OOD 泛化指标总结


# 1. 分区间 OOD 误差

## 1.1 区间定义

训练右边界为：

[
x_{\max}=3\pi
]

定义右侧第 (k) 个 OOD 区间：

[
I_k=[3\pi+(k-1)\pi,\ 3\pi+k\pi]
]

其中：

[
k=1,2,3,\dots
]

例如：

[
I_1=[3\pi,4\pi]
]

[
I_2=[4\pi,5\pi]
]

---

## 1.2 区间 MAE

定义第 (k) 个区间上的平均绝对误差：

[
E_k
===

\frac{1}{|I_k|}
\int_{x\in I_k}
|\hat f(x)-\sin x|,dx
]

离散采样版本：

[
E_k
===

\frac{1}{N_k}
\sum_{x_i\in I_k}
|\hat f(x_i)-\sin x_i|
]

其中 (N_k) 是区间 (I_k) 内的测试点数量。

### 作用

这个指标用于观察：

[
E_1,E_2,E_3,\dots
]

是否随着外推距离增加而变大。

如果：

[
E_1<E_2<E_3<\cdots
]

说明模型越往外推越不稳定。

---

## 1.3 误差增长率

定义相邻 OOD 区间之间的误差增量：

[
\Delta E_k = E_{k+1}-E_k
]

也可以对 (E_k) 关于 (k) 做线性拟合：

[
E_k \approx ak+b
]

其中斜率 (a) 表示外推误差增长速度。

### 作用

* (a\approx 0)：误差没有明显随距离增长；
* (a>0)：误差随外推距离增长；
* (a) 越大，说明远距离外推越不稳定。

---

# 2. 周期一致性指标

周期函数应该满足：

[
f(x+2\pi)=f(x)
]

因此，模型如果真正学到周期性，也应该满足：

[
\hat f(x+2\pi)\approx \hat f(x)
]

---

## 2.1 同相位跨周期一致性

令：

[
x_{j,k}=\theta_j+2\pi k
]

其中：

[
\theta_j\in[0,2\pi)
]

表示相位，(k) 表示周期编号。

定义：

[
C_k
===

\frac{1}{N}
\sum_{j=1}^{N}
\left|
\hat f(\theta_j+2\pi k)
-----------------------

\hat f(\theta_j)
\right|
]

### 作用

这个指标直接测试模型是否学到了：

[
x\sim x+2\pi k
]

也就是周期等价关系。

理想情况下：

[
C_k\approx 0
]

并且 (C_k) 不应该随着 (k) 增大而显著增长。

---

## 2.2 周期一致性退化率

定义：

[
\Delta C_k=C_{k+1}-C_k
]

或者拟合：

[
C_k\approx ak+b
]

其中斜率 (a) 表示周期一致性随外推距离退化的速度。

### 作用

如果：

[
C_1<C_2<C_3<\cdots
]

说明模型在远处越来越无法保持：

[
\hat f(x+2\pi)=\hat f(x)
]

---

# 3. OOD Generalization Gap

设域内误差为：

[
E_0
===

\frac{1}{N_0}
\sum_{x_i\in[-3\pi,3\pi]}
|\hat f(x_i)-\sin x_i|
]

定义第 (k) 个 OOD 区间的泛化差距：

[
G_k=E_k-E_0
]

也可以定义相对泛化差距：

[
R_k=
\frac{E_k}{E_0+\epsilon}
]

其中 (\epsilon) 是一个很小的正数，用于避免除零。

### 作用

因为你已经观察到域内拟合很精确，所以 (E_0) 很小。此时 (G_k) 可以直接反映：

> 域外比域内差多少。

如果：

[
G_k\gg 0
]

说明模型虽然记住或拟合了训练域，但没有稳定泛化到域外周期。

---

# 4. 半周期关系指标

对于正弦函数，不仅有：

[
\sin(x+2\pi)=\sin x
]

还有半周期取反关系：

[
\sin(x+\pi)=-\sin x
]

因此需要检查模型是否错误地把半周期取反学成了半周期复制。

---

## 4.1 半周期复制误差

定义：

[
C_{\pi}^{+}
===========

\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat f(x_i+\pi)-\hat f(x_i)
\right|
]

如果 (C_{\pi}^{+}) 很小，说明模型倾向于认为：

[
\hat f(x+\pi)\approx \hat f(x)
]

也就是错误地学成了 (\pi)-周期。

---

## 4.2 半周期取反误差

定义：

[
C_{\pi}^{-}
===========

\frac{1}{N}
\sum_{i=1}^{N}
\left|
\hat f(x_i+\pi)+\hat f(x_i)
\right|
]

真实的 (\sin x) 应该满足：

[
C_{\pi}^{-}\approx 0
]

因为：

[
\sin(x+\pi)+\sin x=0
]

### 判断逻辑

如果：

[
C_{\pi}^{+}<C_{\pi}^{-}
]

说明模型更像学到了错误的：

[
\hat f(x+\pi)\approx \hat f(x)
]

而不是正确的：

[
\hat f(x+\pi)\approx-\hat f(x)
]

这可以解释你观察到的：

> ([3\pi,4\pi]) 区间像是重复了 ([2\pi,3\pi]) 区间。

---

# 5. Shift-Matching 指标

这个指标用于判断模型在 OOD 区间到底在“复制”训练域中的哪一段。

---

## 5.1 正向 shift-matching

对于某个 OOD 区间 (I)，定义：

[
D_{+}(\Delta)
=============

\frac{1}{|I|}
\int_{x\in I}
\left|
\hat f(x)-\hat f(x-\Delta)
\right|dx
]

离散版本：

[
D_{+}(\Delta)
=============

\frac{1}{N_I}
\sum_{x_i\in I}
\left|
\hat f(x_i)-\hat f(x_i-\Delta)
\right|
]

其中 (\Delta) 可以取：

[
\Delta\in{\pi,2\pi,3\pi,4\pi,\dots}
]

然后定义最匹配的 shift：

[
\Delta^{*}_{+}
==============

\arg\min_{\Delta}
D_{+}(\Delta)
]

### 作用

如果在：

[
I=[3\pi,4\pi]
]

上得到：

[
\Delta^{*}_{+}=\pi
]

说明模型在这个区间更像是在复制：

[
[2\pi,3\pi]
]

而不是根据正确周期回到：

[
[\pi,2\pi]
]

---

## 5.2 带符号 shift-matching

由于正弦函数存在半周期取反关系，还需要定义：

[
D_{-}(\Delta)
=============

\frac{1}{|I|}
\int_{x\in I}
\left|
\hat f(x)+\hat f(x-\Delta)
\right|dx
]

离散版本：

[
D_{-}(\Delta)
=============

\frac{1}{N_I}
\sum_{x_i\in I}
\left|
\hat f(x_i)+\hat f(x_i-\Delta)
\right|
]

对于真实 (\sin x)，有：

[
D_{+}(2\pi)\approx 0
]

[
D_{-}(\pi)\approx 0
]

### 判断逻辑

如果实际观察到：

[
D_{+}(\pi)\ll D_{-}(\pi)
]

并且：

[
D_{+}(\pi)\ll D_{+}(2\pi)
]

说明模型错误地把半周期取反关系学成了半周期复制。

---

# 6. 局部正弦拟合指标

为了判断远处曲线为什么“不稳定”，可以在每个 OOD 区间上拟合一个局部正弦函数。

---

## 6.1 局部拟合模型

在第 (k) 个区间 (I_k) 上，拟合：

[
\hat f(x)
\approx
A_k\sin(\omega_k x+\phi_k)+c_k
]

其中：

* (A_k)：局部振幅；
* (\omega_k)：局部频率；
* (\phi_k)：局部相位；
* (c_k)：局部偏置。

理想情况下：

[
A_k\approx 1
]

[
\omega_k\approx 1
]

[
\phi_k\approx 0
]

[
c_k\approx 0
]

---

## 6.2 振幅稳定性

记录每个区间的振幅：

[
A_1,A_2,A_3,\dots
]

定义振幅抖动：

[
S_A=\operatorname{Var}_k(A_k)
]

### 作用

如果 (A_k) 越来越小，说明外推出现振幅衰减。

如果 (A_k) 忽大忽小，说明外推振幅不稳定。

---

## 6.3 频率稳定性

记录每个区间的频率：

[
\omega_1,\omega_2,\omega_3,\dots
]

定义频率误差：

[
F_k=|\omega_k-1|
]

定义频率抖动：

[
S_{\omega}=\operatorname{Var}_k(\omega_k)
]

### 作用

如果：

[
\omega_k\neq 1
]

说明模型学到的局部周期长度不正确。

如果：

[
S_{\omega}
]

较大，说明模型越往外推，局部周期越不稳定。

---

## 6.4 相位漂移

记录每个区间的相位：

[
\phi_1,\phi_2,\phi_3,\dots
]

可以拟合：

[
\phi_k\approx ak+b
]

其中 (a) 表示相位漂移速度。

也可以定义：

[
D_{\phi}=|a|
]

### 作用

如果 (D_{\phi}) 较大，说明相位误差随着外推距离不断累积。

---

## 6.5 偏置误差

定义每个区间的偏置绝对值：

[
B_k=|c_k|
]

### 作用

如果 (c_k\neq 0)，说明模型输出整体上移或下移。

---

## 6.6 局部正弦拟合残差

定义：

[
R^{\text{fit}}_k
================

\frac{1}{N_k}
\sum_{x_i\in I_k}
\left|
\hat f(x_i)
-----------

\left(
A_k\sin(\omega_k x_i+\phi_k)+c_k
\right)
\right|
]

### 作用

如果 (R^{\text{fit}}_k) 很小，说明预测曲线虽然可能周期或相位不对，但仍然像一个正弦函数。

如果 (R^{\text{fit}}_k) 很大，说明预测曲线已经不是简单的正弦形状，而是出现了更复杂的波形崩坏。

---

# 7. 平滑性与局部抖动指标

如果远处曲线出现锯齿、毛刺、局部抖动，需要观察预测序列的差分。

设区间 (I_k) 内采样点为：

[
x_1,x_2,\dots,x_{N_k}
]

对应预测为：

[
\hat y_1,\hat y_2,\dots,\hat y_{N_k}
]

---

## 7.1 一阶差分能量 / Total Variation

定义：

[
TV_k
====

\frac{1}{N_k-1}
\sum_{i=1}^{N_k-1}
|\hat y_{i+1}-\hat y_i|
]

### 作用

这个指标衡量曲线局部变化幅度。

如果 (TV_k) 随 (k) 增大，说明越往外推，预测曲线局部波动越强。

---

## 7.2 二阶差分能量 / Curvature

定义：

[
Curv_k
======

\frac{1}{N_k-2}
\sum_{i=2}^{N_k-1}
|\hat y_{i+1}-2\hat y_i+\hat y_{i-1}|
]

### 作用

这个指标衡量曲线的局部弯曲和抖动程度。

如果 (Curv_k) 很大，说明曲线出现不平滑、毛刺或高频噪声。

---

## 7.3 平滑性退化率

可以拟合：

[
TV_k\approx ak+b
]

或：

[
Curv_k\approx ak+b
]

其中斜率 (a) 表示平滑性随外推距离退化的速度。

---

# 8. 零点稳定性指标

真实正弦函数的零点为：

[
x=n\pi
]

相邻零点间距为：

[
\pi
]

---

## 8.1 零点位置误差

设模型预测曲线的零点为：

[
\hat z_1,\hat z_2,\dots,\hat z_M
]

真实零点为：

[
z_n=n\pi
]

定义零点位置误差：

[
Z_{\text{pos}}
==============

\frac{1}{M}
\sum_{m=1}^{M}
|\hat z_m-z_m|
]

---

## 8.2 零点间距误差

定义相邻预测零点间距：

[
\Delta \hat z_m=\hat z_{m+1}-\hat z_m
]

真实间距为：

[
\pi
]

定义零点间距误差：

[
Z_{\text{gap}}
==============

\frac{1}{M-1}
\sum_{m=1}^{M-1}
|\Delta \hat z_m-\pi|
]

定义零点间距方差：

[
Z_{\text{var}}
==============

\operatorname{Var}_m(\Delta \hat z_m)
]

### 作用

* (Z_{\text{gap}}) 大：模型预测的零点间距不对；
* (Z_{\text{var}}) 大：模型预测周期不稳定。

---

# 9. 峰值稳定性指标

真实 (\sin x) 的极大值位置为：

[
x=\frac{\pi}{2}+2k\pi
]

极小值位置为：

[
x=\frac{3\pi}{2}+2k\pi
]

---

## 9.1 极大值位置误差

设模型预测的极大值位置为：

[
\hat p_k^{\max}
]

真实极大值位置为：

[
p_k^{\max}
==========

\frac{\pi}{2}+2k\pi
]

定义：

[
P_{\max}
========

\frac{1}{K}
\sum_{k=1}^{K}
|\hat p_k^{\max}-p_k^{\max}|
]

---

## 9.2 极小值位置误差

设模型预测的极小值位置为：

[
\hat p_k^{\min}
]

真实极小值位置为：

[
p_k^{\min}
==========

\frac{3\pi}{2}+2k\pi
]

定义：

[
P_{\min}
========

\frac{1}{K}
\sum_{k=1}^{K}
|\hat p_k^{\min}-p_k^{\min}|
]

---

## 9.3 峰值间距稳定性

设模型预测的相邻极大值点间距为：

[
\Delta \hat p_k
===============

## \hat p_{k+1}^{\max}

\hat p_k^{\max}
]

真实间距为：

[
2\pi
]

定义：

[
P_{\text{gap}}
==============

\frac{1}{K-1}
\sum_{k=1}^{K-1}
|\Delta \hat p_k-2\pi|
]

定义峰值间距方差：

[
P_{\text{var}}
==============

\operatorname{Var}_k(\Delta \hat p_k)
]

### 作用

这些指标用于判断模型是否保持稳定周期长度。

如果：

[
P_{\text{gap}}
]

或：

[
P_{\text{var}}
]

很大，说明模型预测的周期位置越来越乱。

---

# 10. 相位分桶误差

为了观察模型在哪些相位区域出错，可以按照：

[
\theta=x\bmod 2\pi
]

对测试点分桶。

令第 (b) 个相位 bin 为：

[
B_b\subset[0,2\pi)
]

定义第 (k) 个周期、第 (b) 个相位 bin 上的误差：

[
E_{b,k}
=======

\frac{1}{|B_b|}
\sum_{\theta_j\in B_b}
\left|
\hat f(\theta_j+2\pi k)-\sin\theta_j
\right|
]

### 作用

可以画 heatmap：

| 横轴            | 纵轴               | 值         |
| ------------- | ---------------- | --------- |
| phase bin (b) | period index (k) | (E_{b,k}) |

这个指标可以发现：

* 是否在过零点附近错误更大；
* 是否在峰值附近振幅压缩；
* 是否误差随周期编号 (k) 系统性增加；
* 是否出现相位漂移。

---

# 11. Same-Phase Variance 指标

如果模型真正学到周期性，那么同一相位、不同周期上的预测应该接近。

---

## 11.1 同相位跨周期方差

定义：

[
V_{\text{same}}
===============

\frac{1}{N}
\sum_{j=1}^{N}
\operatorname{Var}_{k}
\left[
\hat f(\theta_j+2\pi k)
\right]
]

### 作用

如果 (V_{\text{same}}) 很大，说明同一个相位在不同周期上的预测不一致。

---

## 11.2 不同相位方差

定义：

[
V_{\text{phase}}
================

\operatorname{Var}*{j}
\left[
\frac{1}{K}
\sum*{k=1}^{K}
\hat f(\theta_j+2\pi k)
\right]
]

### 作用

不同相位的输出本来就应该不同，因此 (V_{\text{phase}}) 不应该太小。

---

## 11.3 周期等价类分数

定义：

[
S_{\text{phase}}
================

\frac{
V_{\text{same}}
}{
V_{\text{phase}}+\epsilon
}
]

理想情况下：

[
S_{\text{phase}}\ll 1
]

### 注意

这个指标不能单独使用。因为如果模型输出常数，(V_{\text{same}}) 也会很小。因此它需要和 (E_k)、(A_k) 一起看。

---

# 12. 文本格式相关辅助指标

虽然你的核心问题是周期 OOD 泛化，但由于输入输出都是文本格式，仍然需要排除文本格式失败。

---

## 12.1 Valid Format Rate

定义合法输出格式，例如：

```text
[+/-][0-9][0-9].[0-9][0-9][0-9][0-9][0-9]
```

合法格式比例为：

[
VFR
===

\frac{
#{\text{valid outputs}}
}{
#{\text{all outputs}}
}
]

### 作用

如果 (VFR) 很低，说明模型不是周期没学好，而是文本生成格式已经崩坏。

---

## 12.2 Sign Accuracy

定义：

[
Acc_{\text{sign}}
=================

\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1
\left[
\operatorname{sign}(\hat f(x_i))
================================

\operatorname{sign}(\sin x_i)
\right]
]

### 作用

尤其关注过零点附近：

[
x\approx n\pi
]

如果符号错误很多，说明模型没有稳定掌握正负周期结构。

---

## 12.3 Position-wise Token Accuracy

设目标文本为：

[
s_i=(s_{i,1},s_{i,2},\dots,s_{i,L})
]

模型输出为：

[
\hat s_i=(\hat s_{i,1},\hat s_{i,2},\dots,\hat s_{i,L})
]

第 (l) 个位置的准确率为：

[
Acc_l
=====

\frac{1}{N}
\sum_{i=1}^{N}
\mathbf 1[\hat s_{i,l}=s_{i,l}]
]

### 作用

可以检查模型是：

* sign 出错；
* 整数位出错；
* 小数前几位出错；
* 还是只是低位小数出错。

---

# 13. Modulo-Reduction Diagnostic

这是一个重要的对照实验。

对于 OOD 输入：

[
x=\theta+2\pi k
]

比较两种输入：

1. 直接输入：

[
\text{text}(x)
]

2. 先规约回域内相位：

[
\text{text}(x\bmod 2\pi)
]

定义：

[
M_k
===

\frac{1}{N}
\sum_{j=1}^{N}
\left|
\hat f(\theta_j+2\pi k)
-----------------------

\hat f(\theta_j)
\right|
]

这个指标与 (C_k) 形式相同，但实验解释不同。

### 作用

如果输入 (x\bmod 2\pi) 时预测准确，而直接输入 (x) 时预测失败，说明模型不是不会输出 (\sin)，而是没有学会：

[
x\mapsto x\bmod 2\pi
]

也就是没有学会周期规约。

---


# 15. 推荐实验结论判断逻辑

## 15.1 如果 (E_k) 随 (k) 增大

说明：

[
\text{模型外推误差随距离增长}
]

也就是越往外推越不稳定。

---

## 15.2 如果 (C_k) 随 (k) 增大

说明模型没有稳定学到：

[
\hat f(x+2\pi)=\hat f(x)
]

---

## 15.3 如果 (C_\pi^+<C_\pi^-)

说明模型可能错误地学成了：

[
\hat f(x+\pi)\approx \hat f(x)
]

而不是：

[
\hat f(x+\pi)\approx-\hat f(x)
]

---

## 15.4 如果 (D_+(\pi)) 在 ([3\pi,4\pi]) 上最小

说明模型在第一个右侧 OOD 区间更像是在复制：

[
[2\pi,3\pi]
]

而不是根据正确周期对应到：

[
[\pi,2\pi]
]

---

## 15.5 如果 (\omega_k\neq 1)

说明模型学到的局部周期长度不正确。

---

## 15.6 如果 (\phi_k) 随 (k) 漂移

说明模型出现了累积相位误差。

---

## 15.7 如果 (A_k) 随 (k) 衰减或震荡

说明模型出现了振幅衰减或振幅不稳定。

---

## 15.8 如果 (TV_k)、(Curv_k) 增大

说明远处曲线出现局部抖动、毛刺或高频不稳定。

---

# 16. 最终建议

你的实验核心应该围绕三条主线组织：

## 第一条：误差是否随外推距离增长

核心指标：

[
E_k,\quad \Delta E_k,\quad \text{slope}(E_k)
]

---

## 第二条：模型是否学到周期等价关系

核心指标：

[
C_k,\quad C_{\pi}^{+},\quad C_{\pi}^{-},\quad D_{+}(\Delta),\quad D_{-}(\Delta)
]

---

## 第三条：远处波形是否稳定

核心指标：

[
A_k,\quad \omega_k,\quad \phi_k,\quad c_k,\quad R_k^{fit},\quad TV_k,\quad Curv_k
]

这三类指标合起来可以比较完整地回答：

> 模型到底是学到了 (\sin x) 的周期规律，还是只在训练域内拟合准确，而在域外通过局部复制、错误频率、相位漂移或不稳定波形进行失败外推。
