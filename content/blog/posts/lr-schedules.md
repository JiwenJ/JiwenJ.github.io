---
title: "不同优化器的最优 LR Schedule"
date: "2026-10-31T00:00:00+08:00"
author: ["Jiwen Jiang"]
description: "从终点损失出发，依次推导 SGD、Adafactor、Adam 和 Muon 的 LR Schedule。"
summary: "SGD 的线性上界与反比精确解，Adafactor 的相对步长，Adam 的动量修正，以及 Muon 的矩阵推导。"
tags: ["Optimization", "Learning Rate", "SGD", "Adafactor", "Adam", "Muon"]
categories: ["Deep Learning"]
draft: false
math: true
isCJKLanguage: true
ShowToc: true
TocOpen: false
aliases: ["/blog/posts/lr-schedule/"]
citationKey: "jiang2026lrschedules"
citationUrl: "https://jiwenj.github.io/blog/posts/lr-schedules/"
references:
  - id: 1
    title: "Optimal Linear Decay Learning Rate Schedules and Further Refinements"
    authors: "Aaron Defazio, Ashok Cutkosky, Harsh Mehta, Konstantin Mishchenko"
    source: "arXiv:2310.07831v2, 2024"
    url: "https://arxiv.org/abs/2310.07831v2"
  - id: 2
    title: "让炼丹更科学一些（六）：自上而下的精妙构造"
    authors: "苏剑林"
    source: "科学空间, 2026"
    url: "https://spaces.ac.cn/archives/11540"
  - id: 3
    title: "让炼丹更科学一些（十）：单调性假设的拆与补"
    authors: "苏剑林"
    source: "科学空间, 2026"
    url: "https://spaces.ac.cn/archives/11885"
  - id: 4
    title: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
    authors: "John Duchi, Elad Hazan, Yoram Singer"
    source: "JMLR 12, 2011"
    url: "https://www.jmlr.org/papers/v12/duchi11a.html"
  - id: 5
    title: "Adam: A Method for Stochastic Optimization"
    authors: "Diederik P. Kingma, Jimmy Ba"
    source: "ICLR 2015"
    url: "https://arxiv.org/abs/1412.6980"
  - id: 6
    title: "On the Convergence of Adam and Beyond"
    authors: "Sashank J. Reddi, Satyen Kale, Sanjiv Kumar"
    source: "ICLR 2018"
    url: "https://openreview.net/forum?id=ryQu7f-RZ"
  - id: 7
    title: "Decoupled Weight Decay Regularization"
    authors: "Ilya Loshchilov, Frank Hutter"
    source: "ICLR 2019"
    url: "https://arxiv.org/abs/1711.05101"
  - id: 8
    title: "Muon: An Optimizer for Hidden Layers in Neural Networks"
    authors: "Keller Jordan"
    source: "作者博客, 2024"
    url: "https://kellerjordan.github.io/posts/muon/"
  - id: 9
    title: "Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization"
    authors: "Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang"
    source: "arXiv:2606.16899v2, 2026"
    url: "https://arxiv.org/abs/2606.16899v2"
  - id: 10
    title: "Deriving Muon"
    authors: "Jeremy Bernstein"
    source: "作者博客, 2025"
    url: "https://jeremybernste.in/writing/deriving-muon"
  - id: 11
    title: "Gradient-based Hyperparameter Optimization through Reversible Learning"
    authors: "Dougal Maclaurin, David Duvenaud, Ryan Adams"
    source: "ICML 2015"
    url: "https://proceedings.mlr.press/v37/maclaurin15.html"
  - id: 12
    title: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
    authors: "Noam Shazeer, Mitchell Stern"
    source: "ICML 2018"
    url: "https://proceedings.mlr.press/v80/shazeer18a.html"
---


给定训练预算，怎样选择每一步的学习率，才能让最后的损失最小？答案取决于目标函数、梯度噪声和优化器的更新方式。

统一把更新写成 $\theta_{t+1}=\theta_t-\eta_tu_t$。$u_t$ 包含预条件、动量或矩阵正交化，$\eta_t$ 是外部标量学习率。要优化的是固定预算后的 $\mathbb E[f(\theta_T)-f_*]$，其中 $f_*=f(\theta_*)$。

一般凸问题可以先优化终点误差上界；在可解的二次模型中，则能直接最小化真实终点损失。SGD 的 Linear Decay 来自前一种分析{{< cite 1 2 >}}，并不是所有优化器、所有目标下的统一答案。

## 一、SGD

先假设 $f$ 凸，$g_t$ 是当前点的条件无偏随机次梯度，外部权重为确定性序列。初始参数为 $\theta_1$，输出为 $\theta_T$，最后的 $\eta_T=0$ 只是边界约定[^index]。

### 把终点误差写成加权和

引入正权重 $w_1,\ldots,w_T$，定义

$$
S=\sum_{t=1}^{T}w_t,
\qquad
\eta_t=\frac{w_t\sum_{s=t+1}^{T}w_s}{S}.
$$

再引入一个只用于证明的辅助序列

$$
z_1=\theta_1,
\qquad z_{t+1}=z_t-w_tg_t.
$$

两条轨迹满足

$$
\theta_{t+1}-\theta_t
=\frac{\sum_{s=t+1}^{T}w_s}{S}(z_{t+1}-z_t).
$$

Defazio 等人的终点转换定理，把这个关系与凸性结合，得到

$$
\mathbb E[f(\theta_T)-f_*]
\leq\frac1S\mathbb E\sum_{t=1}^{T}
w_t\langle g_t,z_t-\theta_*\rangle.
$$

右端现在是一个可以用平方距离递推控制的量。设 $D=\|\theta_1-\theta_*\|$，展开

$$
\|z_{t+1}-\theta_*\|^2
=\|z_t-\theta_*\|^2
-2w_t\langle g_t,z_t-\theta_*\rangle
+w_t^2\|g_t\|^2.
$$

求和后中间项相消，末端平方距离非负，于是

$$
\boxed{
\mathbb E[f(\theta_T)-f_*]
\leq
\frac{D^2+\sum_{t=1}^{T}w_t^2\mathbb E\|g_t\|^2}{2S}.
}
$$

梯度仍然在实际参数 $\theta_t$ 处计算。辅助序列不需要再做一次前向和反向传播。

### 梯度能量用同一个常数控制

若所有时刻都有 $\mathbb E\|g_t\|^2\leq G^2$，就可以优化

$$
\mathcal B(w)=\frac{D^2+G^2\sum_t w_t^2}{2\sum_t w_t}.
$$

先固定总和 $S$。由 Cauchy–Schwarz，

$$
\sum_{t=1}^{T}w_t^2\geq\frac{S^2}{T},
$$

等号在 $w_t=S/T$ 时成立。因此最好的权重形状是常数，剩下的幅值问题为

$$
\min_{S>0}\left(\frac{D^2}{2S}+\frac{G^2S}{2T}\right).
$$

对 $S$ 求导，得到

$$
S^*=\frac{D\sqrt T}{G},
\qquad w_t^*=\frac{D}{G\sqrt T}.
$$

代回 $\eta_t$ 的构造式：

$$
\boxed{
\eta_t^*=\frac{D}{G\sqrt T}\left(1-\frac tT\right),\qquad \mathbb E[f(\theta_T)-f_*]\leq\frac{DG}{\sqrt T}.
}
$$

线性来自“剩余权重的占比”：每一步的权重相等，剩余权重就与剩余步数成正比。这个速率达到一般凸随机优化的最优最坏情况阶{{< cite 1 >}}。实际训练通常不知道 $D$ 和 $G$，因此会保留线性形状，另外搜索峰值学习率。

### Warmup 后线性降到 0

固定总预算 $T$，在 $T_w$ 处结束 warmup，峰值学习率为 $\eta_{\rm pk}$，最小学习率为 $0$。用 $0\leq t\leq T$ 表示训练进度，这条调度曲线可以写成一行：

$$
\boxed{\eta(t)=\eta_{\rm pk}\min\!\left\{\frac{t}{T_w},\frac{T-t}{T-T_w}\right\},\qquad 0\lt T_w\lt T.}
$$

它在 $t=T_w$ 达到峰值，随后直接线性降到 $\eta(T)=0$，中间没有恒定学习率阶段。$t=T$ 是预算边界；最后一次实际更新是否使用零学习率，要按更新索引约定区分[^index]。

前面的常数梯度界推出了线性衰减形状，没有同时求出 warmup 长度。Defazio 等人的实验为非自适应调度加入了固定 warmup；使用随时间变化的梯度统计所作的 refinement，也能产生 warmup，但曲线通常不再是严格的两段直线{{< cite 1 >}}。

若先固定 warmup 前缀，可以把 warmup 后的参数作为新的初始点，在剩余区间重新应用终点上界。此时仍须满足凸性、原始梯度条件无偏等假设，并对候选后续调度使用共同的距离与能量界。这个分析支持衰减段的线性形状，不等于联合证明了整条 warmup 加衰减曲线、峰值和 warmup 长度都最优。

### 梯度随时间变化时

如果有一组预先给定的正数 $a_t$，满足 $\mathbb E\|g_t\|^2\leq a_t$，同样可以求解。令

$$
Q_t=\sum_{s=1}^{t}a_s^{-1}.
$$

固定 $S$ 时，加权 Cauchy–Schwarz 给出

$$
S^2\leq\left(\sum_t a_tw_t^2\right)Q_T.
$$

所以最优权重与 $a_t^{-1}$ 成正比。再优化 $S$，得到

$$
\boxed{
w_t^*=\frac{D}{a_t\sqrt{Q_T}},\qquad \eta_t^*=\frac{D}{a_t\sqrt{Q_T}} \left(1-\frac{Q_t}{Q_T}\right).
}
$$

常数 $a_t$ 会恢复 Linear Decay。若早期梯度能量大，前面的 $a_t^{-1}$ 会压低步长；能量随后下降时，这个因子可能产生 warmup，但并非任意下降曲线都会产生 warmup。

这里的 $a_t$ 必须作为固定上界参与优化。若改用一次训练中观测到的梯度曲线，换 LR Schedule 后梯度也会变，公式就变成基于旧轨迹的 refinement；原论文采用的正是先记录、平滑，再用于后续训练的方式{{< cite 1 2 >}}。它还需要未来的 $Q_T$，不能直接当作只看当前梯度的在线学习率算法。

### 二次模型的精确最优解

如果目标函数和噪声已知，可以绕过统一上界，直接求终点损失的最小值。考虑带加性噪声的一维二次模型：

$$
f(x)=\frac h2x^2,\qquad
x_{t+1}=(1-h\eta_t)x_t-\eta_t\epsilon_t.
$$

设 $h>0$、$r_0=\mathbb E[x_0^2]>0$，噪声条件均值为零、条件方差为固定的 $\sigma^2>0$，学习率为非负确定性序列。这里从 $x_0$ 开始，执行 $N$ 次更新，最小化 $\mathbb E f(x_N)$。记 $r_t=\mathbb E[x_t^2]$，有精确递推

$$
r_{t+1}=(1-h\eta_t)^2r_t+\sigma^2\eta_t^2.
$$

对步长配方：

$$
r_{t+1} =\frac{\sigma^2r_t}{h^2r_t+\sigma^2} +(h^2r_t+\sigma^2) \left(\eta_t-\frac{hr_t}{h^2r_t+\sigma^2}\right)^2.
$$

因此一步最优为 $\eta_t^*=hr_t/(h^2r_t+\sigma^2)$。而最小可达的下一步误差

$$
\phi(r)=\frac{\sigma^2r}{h^2r+\sigma^2}
$$

关于 $r$ 单调增加。任何 LR Schedule 都有 $r_{t+1}\geq\phi(r_t)$，所以逐步达到等号，会同时最小化每个时刻的误差。这个标量模型中，一步最优恰好也是任意终点的全局最优[^scalar]。

最优递推满足

$$
\frac1{r_{t+1}^*}=\frac1{r_t^*}+\frac{h^2}{\sigma^2},
$$

从而

$$
\boxed{
\eta_t^*=\frac1{h\left(t+1+\dfrac{\sigma^2}{h^2r_0}\right)},\qquad \mathbb E f(x_N)_{\min} =\frac{h\sigma^2r_0}{2(\sigma^2+Nh^2r_0)}.
}
$$

最优步长按反比衰减，不依赖预先指定的训练终点。

### 与 Linear 和 Cosine 比较

取 $h=\sigma^2=r_0=1$，执行 $N=100$ 次更新。精确最优 LR Schedule 是 $\eta_t=1/(t+2)$。另外三组 LR Schedule 分别为

$$
\begin{aligned}
\eta_t^{\rm linear}&=\alpha(1-t/N),\\
\eta_t^{\rm cosine}&=\tfrac\alpha2[1+\cos(\pi t/N)],\\
\eta_t^{\rm constant}&=\alpha,
\qquad t=0,\ldots,N-1.
\end{aligned}
$$

对每一组分别在 $\alpha\in[0,1]$ 内搜索使终点损失最小的幅值。下图直接计算期望误差递推，没有抽样噪声。

<figure>
<picture>
  <source media="(max-width: 640px)" srcset="/images/lr-schedules/quadratic-mobile.svg">
  <img src="/images/lr-schedules/quadratic.svg" alt="带噪一维二次损失下四种 LR Schedule 及其精确期望损失，反比 LR Schedule 达到解析最优，其他 LR Schedule 分别调节幅值。" loading="lazy" width="1200">
</picture>
<figcaption>四条曲线使用同一个损失、噪声方差和更新预算。学习率轴与损失轴均为对数刻度；比较的是这个二次模型的终点损失。</figcaption>
</figure>

| LR Schedule | 最优或搜索得到的幅值 | 第 100 步期望损失 |
| --- | --- | --- |
| 精确最优 $1/(t+2)$ | 由解析式确定 | $0.0049505$ |
| Linear | $\alpha\approx0.068448$ | $0.0062609$ |
| Cosine | $\alpha\approx0.068092$ | $0.0069959$ |
| Constant | $\alpha\approx0.029453$ | $0.0087200$ |

在这个二次模型中，Linear 的终点损失低于 Cosine 和 Constant，但高于精确的反比衰减解。前面推导的 Linear Decay 最小化一般凸问题的统一上界；这里利用二次结构求解真实终点损失，得到不同的最优形状。

## 二、Adafactor

### 更新规则：因子化二阶矩与相对步长

Adafactor 用行、列统计量近似二阶矩，再对预条件更新做裁剪。以下采用原论文中不含一阶动量的矩阵版本{{< cite 12 >}}。设 $W_t,G_t\in\mathbb R^{m\times n}$，$G_t$ 在 $W_t$ 处计算，$t\geq1$、$R_0=C_0=0$，稳定项 $\epsilon_1,\epsilon_2$ 与裁剪阈值 $d$ 均为正；所有平方、开方和除法均逐元素进行。

先给平方梯度加一个正的稳定项：

$$
B_t=G_t^{\odot2}+\epsilon_1\mathbf1_m\mathbf1_n^\top.
$$

行统计 $R_t\in\mathbb R^{m\times1}$ 与列统计 $C_t\in\mathbb R^{1\times n}$ 分别更新为

$$
R_t=\beta_{2,t}R_{t-1}+(1-\beta_{2,t})B_t\mathbf1_n,
$$

$$
C_t=\beta_{2,t}C_{t-1}+(1-\beta_{2,t})\mathbf1_m^\top B_t.
$$

据此重建二阶矩近似，并形成预条件方向：

$$
\widehat V_t=\frac{R_tC_t}{\mathbf1_m^\top R_t},\qquad U_t=G_t\oslash\sqrt{\widehat V_t}.
$$

因子化的是二阶矩估计，不是梯度本身。二阶矩状态只需存 $m+n$ 个数。接着定义 $\operatorname{RMS}(U)=\|U\|_F/\sqrt{mn}$，对更新做 RMS 裁剪：

$$
\kappa_t=\max\!\left(1,\frac{\operatorname{RMS}(U_t)}{d}\right),\qquad D_t=U_t/\kappa_t.
$$

最后乘上参数尺度 $s_t$ 和相对步长 $\rho_t$：

$$
s_t=\max(\epsilon_2,\operatorname{RMS}(W_t)),\qquad W_{t+1}=W_t-\rho_ts_tD_t.
$$

这里应区分两条曲线：$\rho_t$ 是**相对步长**，$\alpha_t=\rho_ts_t$ 是乘在裁剪方向 $D_t$ 上的**绝对步长**。即使 $\rho_t$ 线性衰减，只要参数 RMS 在变，$\alpha_t$ 就不再是直线。

原论文提出的配置为 $\beta_{2,t}=1-t^{-0.8}$、$\rho_t=\min(10^{-2},t^{-1/2})$，也实验了带 warmup 的 $\rho_t=\min(10^{-6}t,t^{-1/2})$。这些是算法配置与实验选择，不能直接视为终点损失的最优解。

### 先求一步最优的相对步长

条件在更新前的历史 $\mathcal F_{t-1}$ 上，$s_t$ 已知，而 $D_t$ 仍取决于当前随机梯度。定义方向的对齐与平方长度：

$$
\mathcal A_t=\mathbb E[\langle\nabla f(W_t),D_t\rangle_F\mid\mathcal F_{t-1}],\qquad \mathcal V_t=\mathbb E[\|D_t\|_F^2\mid\mathcal F_{t-1}].
$$

若 $f$ 的梯度是 $L_s$-Lipschitz，光滑性给出

$$
\mathbb E[f(W_t-\rho s_tD_t)\mid\mathcal F_{t-1}]\leq f(W_t)-\rho s_t\mathcal A_t+\frac{L_s\rho^2s_t^2\mathcal V_t}{2}.
$$

当 $\mathcal V_t>0$，对右端求非负极小值得到

$$
\boxed{\rho_t^{\rm one}=\frac{[\mathcal A_t]_+}{L_ss_t\mathcal V_t},\qquad \alpha_t^{\rm one}=\frac{[\mathcal A_t]_+}{L_s\mathcal V_t}.}
$$

裁剪保证 $\mathcal V_t\leq mn\,d^2$，但没有固定 $\mathcal A_t$。所以“更新长度有界”本身推不出 Linear 或 Inverse Square Root。这一步只最小化当前下降上界，还没有考虑学习率对后续参数、二阶矩和裁剪状态的影响。

### 从变度量递推求终点上界

把矩阵展平成向量，记 $g_t=\operatorname{vec}(G_t)$、$\theta_t=\operatorname{vec}(W_t)$。Adafactor 可以写成

$$
\theta_{t+1}=\theta_t-\rho_tH_t^{-1}g_t,\qquad H_t=\frac{\kappa_t}{s_t}\operatorname{diag}\!\left(\operatorname{vec}\sqrt{\widehat V_t}\right).
$$

滑动平均、更新裁剪和参数尺度都进入了 $H_t$，因此不能假设 $H_t$ 单调增加。沿用 SGD 的权重构造 $\rho_t=w_t\sum_{s>t}w_s/S$，引入 $z_{t+1}=z_t-w_tH_t^{-1}g_t$、$z_1=\theta_1$，写 $e_t=z_t-\theta_*$。加权平方距离满足

$$
2w_t\langle g_t,e_t\rangle=\|e_t\|_{H_t}^2-\|e_{t+1}\|_{H_t}^2+w_t^2g_t^\top H_t^{-1}g_t.
$$

前两项求和，得到

$$
\|e_1\|_{H_1}^2-\|e_{T+1}\|_{H_T}^2+\sum_{t=2}^{T}e_t^\top(H_t-H_{t-1})e_t.
$$

若辅助轨迹满足 $\|e_t\|\leq R$，可以用预条件矩阵的累计正向变化控制它：

$$
\Lambda_T(H)=\lambda_{\max}(H_1)+\sum_{t=2}^{T}[\lambda_{\max}(H_t-H_{t-1})]_+.
$$

因为每个变化项不超过 $R^2[\lambda_{\max}(H_t-H_{t-1})]_+$，上面的距离和至多为 $R^2\Lambda_T(H)$。这个变度量处理也适用于其他非单调预条件器{{< cite 3 >}}。

在凸目标、原始随机梯度条件无偏的假设下，代入终点转换：

$$
\mathbb E[f(\theta_T)-f_*]\leq\frac{R^2\mathbb E\Lambda_T(H)+\sum_t w_t^2\mathbb E[g_t^\top H_t^{-1}g_t]}{2S}.
$$

这里不要求预条件后的方向无偏。距离恒等式对每个样本成立，无偏性只用于原始梯度 $g_t$ 与凸性的关系。Adafactor 对应的能量为

$$
g_t^\top H_t^{-1}g_t=\frac{s_t}{\kappa_t}\sum_{i,j}\frac{G_{t,ij}^2}{\sqrt{\widehat V_{t,ij}}}.
$$

注意它不同于更新的欧氏平方长度 $g_t^\top H_t^{-2}g_t$。若对候选 LR Schedule 存在共同的确定性界

$$
R^2\mathbb E\Lambda_T(H)\leq A,\qquad \mathbb E[g_t^\top H_t^{-1}g_t]\leq a_t,\qquad A>0,\ a_t>0,
$$

要优化的上界就是 $\mathcal B(w)=(A+\sum_ta_tw_t^2)/(2\sum_tw_t)$。令 $Q_t=\sum_{s=1}^ta_s^{-1}$，沿用 SGD 的加权 Cauchy–Schwarz，得到

$$
\boxed{\rho_t^*=\sqrt{\frac A{Q_T}}\frac1{a_t}\left(1-\frac{Q_t}{Q_T}\right),\qquad \mathcal B_{\min}=\sqrt{\frac A{Q_T}}.}
$$

将 $a_t$ 用同一个常数 $a$ 控制时，便得到相对步长的 Linear Decay：

$$
\boxed{\rho_t^*=\sqrt{\frac A{aT}}\left(1-\frac tT\right),\qquad \alpha_t^*=s_t\rho_t^*.}
$$

这个结果优化的是固定界 $A,a_t$ 下的终点误差上界，求出的调度还须属于这些界成立的候选集合。实际的 $s_t,\kappa_t,\widehat V_t$ 会随调度改变；直接把某次训练的统计量代入，并不能得到原训练问题的全局最优解。要让上界趋于零，还需 $A/Q_T\to0$，仅有每一步的 $H_t$ 有界并不足够。


<details class="technical-note">
<summary>补充：AdaGrad 的单调预条件与外部线性衰减</summary>

**预条件以后要换一个能量**

考虑没有一阶动量的预条件更新

$$
\theta_{t+1}=\theta_t-\eta_t H_t^{-1}g_t,
\qquad H_t\succ0.
$$

将辅助更新改为 $z_{t+1}=z_t-w_tH_t^{-1}g_t$，平方距离也改用 $H_t$ 定义的度量。若 $H_t\succeq H_{t-1}$，且辅助轨迹满足 $\|z_t-\theta_*\|\leq R$，相同方法给出

$$
\mathbb E[f(\theta_T)-f_*]
\leq
\frac{R^2\mathbb E\operatorname{tr}(H_T)
+\sum_t w_t^2\mathbb E[g_t^\top H_t^{-1}g_t]}{2S}.
$$

这个推广由预条件距离的望远镜求和得到{{< cite 3 >}}。它需要控制辅助轨迹；仅仅知道实际参数 $\theta_t$ 有界还不够。

注意这里出现的是

$$
g_t^\top H_t^{-1}g_t,
$$

而更新向量的欧氏平方长度是 $g_t^\top H_t^{-2}g_t$。少一个逆矩阵幂次，会改变整条调度曲线。

如果对所比较的 LR Schedule 有共同的界

$$
R^2\mathbb E\operatorname{tr}(H_T)\leq A,
\qquad
\mathbb E[g_t^\top H_t^{-1}g_t]\leq B,
$$

那么问题再次变成

$$
\min_{w_t>0}\frac{A+B\sum_t w_t^2}{2\sum_t w_t},
$$

于是

$$
\boxed{\eta_t^*=\sqrt{\frac{A}{BT}}\left(1-\frac tT\right).}
$$

$A$、$B$ 固定是这个优化步骤的条件。求出的 LR Schedule 还须属于这些共同界成立的候选集合；否则需要在可行集合内重新优化。

**为什么单调预条件矩阵会给出迹项**

写 $e_t=z_t-\theta_*$，直接展开有

$$
2w_t\langle g_t,e_t\rangle
=\|e_t\|_{H_t}^2-\|e_{t+1}\|_{H_t}^2
+w_t^2g_t^\top H_t^{-1}g_t.
$$

前两项求和后成为

$$
\|e_1\|_{H_1}^2-\|e_{T+1}\|_{H_T}^2
+\sum_{t=2}^{T}e_t^\top(H_t-H_{t-1})e_t.
$$

对半正定矩阵 $M$，$e^\top Me\leq\|e\|^2\operatorname{tr}(M)$。用 $\|e_t\|\leq R$ 控制距离，再将迹相加，便得到 $R^2\operatorname{tr}(H_T)$。



**AdaGrad 的外部调度与坐标步长**

AdaGrad 的对角预条件矩阵为

$$
H_{t,ii}=\sqrt{\epsilon+\sum_{s=1}^{t}g_{s,i}^2}.
$$

累计平方和单调增加，满足这里需要的矩阵单调性。不过，预条件梯度能量并不因此恒定：若原始梯度的二阶矩近乎平稳，累计分母仍会增长。常数 $B$ 是用于分析的统一上界。取线性外部调度后，更新写成

$$
\boxed{
\theta_{t+1,i}=\theta_{t,i}
-\underbrace{\eta_0\left(1-\frac tT\right)}_{\text{全局学习率}}
\frac{g_{t,i}}{\sqrt{\epsilon+\sum_{s=1}^{t}g_{s,i}^2}}.
}
$$

逐坐标的完整步长包含自适应分母，因此它本身不必是直线。

把 AdaGrad 的经典遗憾界与终点转换结合，还能得到等权结论，并保留逐坐标的梯度信息{{< cite 4 >}}。

**用 AdaGrad 的遗憾界重新求一次最优权重**

给辅助 AdaGrad 输入 $h_t=w_tg_t$。为简化记号，先忽略数值稳定项，零分母且零分子时把该坐标更新定义为零。若存在统一常数 $D_\infty$，使 $\|z_t-\theta_*\|_\infty\leq D_\infty$，取基础步长 $\alpha=D_\infty/\sqrt2$，有

$$
\sum_t w_t\langle g_t,z_t-\theta_*\rangle
\leq\sqrt2D_\infty\sum_i\sqrt{\sum_t w_t^2g_{t,i}^2}.
$$

若 $\mathbb E[g_{t,i}^2]\leq G_i^2$，由 Jensen 不等式及终点转换，

$$
\mathbb E[f(\theta_T)-f_*]
\leq\sqrt2D_\infty\Bigl(\sum_iG_i\Bigr)
\frac{\sqrt{\sum_t w_t^2}}{\sum_t w_t}.
$$

最后一个比值的最小值是 $T^{-1/2}$，等权重取到。此时 $w_t$ 在辅助 AdaGrad 的分子与累计平方根中抵消；终点转换额外乘上的剩余权重比例就是 $1-t/T$。这给出 AdaGrad 外部线性调度的另一条证明，并保留了它逐坐标适应梯度大小的结构。



</details>

## 三、Adam

### 二阶矩预条件与一阶动量

Adam 用滑动平均估计梯度的一阶、二阶矩{{< cite 5 >}}：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2}.
$$

从 $m_0=v_0=0$ 出发，偏差修正后的状态与更新为

$$
\widehat m_t=\frac{m_t}{1-\beta_1^t},\qquad \widehat v_t=\frac{v_t}{1-\beta_2^t},\qquad u_t=\widehat m_t\oslash(\sqrt{\widehat v_t}+\epsilon).
$$

令 $H_t=\operatorname{diag}(\sqrt{\widehat v_t}+\epsilon)$，则 $\theta_{t+1}=\theta_t-\eta_tH_t^{-1}\widehat m_t$。与 Adafactor 一样，二阶矩滑动平均不保证 $H_t$ 单调，所以仍需控制 $\Lambda_T(H)$。Adam 的收敛反例与 AMSGrad 对二阶矩历史最大值的使用，也说明不能省略对预条件器变化的控制{{< cite 6 >}}。

一阶动量带来另一个差别：

$$
\mathbb E[m_t\mid\mathcal F_{t-1}]=\beta_1m_{t-1}+(1-\beta_1)\nabla f(\theta_t).
$$

它一般不等于当前梯度。偏差修正消除的是从零初始化的缩放偏差，没有消除旧梯度与当前梯度之间的差异。

### 将动量偏差保留在终点上界中

令辅助更新为 $z_{t+1}=z_t-w_tH_t^{-1}\widehat m_t$，仍取 $\eta_t=w_t\sum_{s>t}w_s/S$。展开距离得到的是 $\langle\widehat m_t,z_t-\theta_*\rangle$，而凸性需要原始梯度 $g_t$。把 $g_t=\widehat m_t+(g_t-\widehat m_t)$ 代入，得到

$$
\mathbb E[f(\theta_T)-f_*]\leq\frac{R^2\mathbb E\Lambda_T(H)+\sum_tw_t^2\mathbb E[\widehat m_t^\top H_t^{-1}\widehat m_t]}{2S}+\mathcal C_T,
$$

其中动量修正项为

$$
\mathcal C_T=\frac1S\sum_tw_t\,\mathbb E\langle g_t-\widehat m_t,z_t-\theta_*\rangle.
$$

仍假设目标凸、原始随机梯度条件无偏、$\|z_t-\theta_*\|\leq R$。选取对候选 LR Schedule 共同成立的固定界：

$$
A\geq R^2\mathbb E\Lambda_T(H),\qquad a_t\geq\mathbb E[\widehat m_t^\top H_t^{-1}\widehat m_t],
$$

$$
b_t\geq R\,\mathbb E\|g_t-\widehat m_t\|,\qquad A>0,\ a_t>0.
$$

由 Cauchy–Schwarz，$\mathcal C_T\leq\sum_tw_tb_t/S$。


需要优化的完整表达式变成

$$
\mathcal B_{\rm mom}(w)
=\frac{A+\sum_ta_tw_t^2+2\sum_tb_tw_t}{2\sum_tw_t}.
$$

若标量方程

$$
\sum_t\frac{(\lambda-b_t)^2}{a_t}=A
$$

有满足 $\lambda>\max_t b_t$ 的解，则

$$
\boxed{
w_t^*=\frac{\lambda-b_t}{a_t},\qquad \eta_t^*=w_t^*\frac{\sum_{s>t}w_s^*}{\sum_sw_s^*},\qquad \mathcal B_{\rm mom,min}=\lambda.
}
$$

验证只需配方。对任意正权重，

$$
2S[\mathcal B_{\rm mom}(w)-\lambda]
=\sum_ta_t(w_t-w_t^*)^2\geq0.
$$

若 $b_t=b$ 为常数，它只使最小上界增加 $b$，不改变最佳权重形状。若偏差界随时间变化，最优权重也会变化。正分支条件不满足时，最优解可能落到零权重边界，不能直接使用上面的正权重公式[^bias]。


当 $a_t=a$、$b_t=b$ 都取常数时，$\lambda=b+\sqrt{Aa/T}$，所以

$$
\boxed{\eta_t^*=\sqrt{\frac A{aT}}\left(1-\frac tT\right),\qquad \mathcal B_{\rm mom,min}=b+\sqrt{\frac{Aa}{T}}.}
$$

共同的常数能量界与偏差界给出了这个线性解，求出的权重还须属于这些界成立的候选集合。若 $b$ 不随训练预算减小，上界还会留下一个误差底；这并不是 Adam 自动收敛到最优点的证明。

### 直接优化真实终点：把矩估计也纳入状态

对于实际非凸训练，可以把每一步学习率作为变量，沿完整训练过程求导{{< cite 11 >}}。以 AdamW 为例，更新额外包含解耦权重衰减{{< cite 7 >}}：

$$
\theta_{t+1}=(1-\lambda_{\rm wd}\eta_t)\theta_t-\eta_tu_t.
$$

取 $X_t=(\theta_t,m_{t-1},v_{t-1})$，完整更新记为 $X_{t+1}=F_t(X_t,\eta_t,\xi_t)$。这里执行 $t=1,\ldots,N$ 共 $N$ 次更新，目标为 $J(\eta)=\mathbb E f(\theta_{N+1})$。对固定的随机样本序列反向传播：

$$
p_{N+1}=(\nabla f(\theta_{N+1}),0,0),\qquad p_t=(D_XF_t)^\top p_{t+1}.
$$

在可微、允许交换求导与期望、且基础随机采样分布不依赖调度的条件下，有

$$
\boxed{\frac{\partial J}{\partial\eta_t}=-\mathbb E\langle p_{t+1}^{\theta},u_t+\lambda_{\rm wd}\theta_t\rangle.}
$$

$p_{t+1}^{\theta}$ 包含此后所有梯度与矩估计的变化，不能替换成当前梯度。内部最优点要求这个导数为零；在 $\eta_t=0$ 的边界处，右导数应非负。这给出了数值求解完整 LR Schedule 的条件，但非凸问题的驻点通常不保证全局最优。

对 Adafactor，同样取 $X_t=(W_t,R_{t-1},C_{t-1})$，得到 $\partial J/\partial\rho_t=-\mathbb E\langle p_{t+1}^{W},s_tD_t\rangle_F$。反向递推必须包含参数尺度、二阶矩重建与裁剪；在裁剪和尺度下限的转折点，需要按实际实现处理分段导数。

AdamW 的权重衰减还会带来累计因子 $\prod_t(1-\lambda_{\rm wd}\eta_t)$。固定衰减系数而更换 LR Schedule，同时也改变了这部分训练动态，因此前面无权重衰减的凸上界不能直接原样套用。

## 四、Muon

沿用 SGD 一节的终点上界分析，无动量、精确极分解的理想 Muon 在共同的常数能量界与受控的预条件矩阵变差下，也会得到线性衰减。关键在于这些量是否受控：正交化固定了部分更新尺度，却没有固定梯度的谱、方向以及动量偏差。

以下先分析对当前梯度做精确极分解、没有动量与权重衰减的理想更新，再加入动量修正。实际 Muon 对动量做有限次 Newton–Schulz 变换，因此不能把理想更新的结论直接当作实际训练的最优调度{{< cite 8 10 >}}。

### 与 SGD 使用同一终点目标

#### 把正交化写成矩阵预条件

设参数与梯度为矩阵 $W_t,G_t$。对只保留非零奇异值的紧致 SVD，定义

$$
G_t=P_t\Sigma_tV_t^\top,
\qquad \operatorname{polar}(G_t)=P_tV_t^\top.
$$

零梯度的更新定义为零。令

$$
H_t=(G_tG_t^\top)^{1/2},
$$

则利用 Moore–Penrose 伪逆 $H_t^\dagger$，有

$$
H_t^\dagger G_t=\operatorname{polar}(G_t),
\qquad H_tH_t^\dagger G_t=G_t.
$$

因此理想 Muon 可以写成 $W_{t+1}=W_t-\eta_tH_t^\dagger G_t$。这个表达式也覆盖非方阵和秩亏矩阵；满行秩时，伪逆就是普通逆。

它在 Adafactor 一节的变度量分析中对应的梯度能量是

$$
\operatorname{tr}(G_t^\top H_t^\dagger G_t) =\operatorname{tr}(\Sigma_t) =\sum_j\sigma_j(G_t) =\|G_t\|_*.
$$

这里 $\|\cdot\|_*$ 专指**核范数，即奇异值之和**。SGD 的能量是 $\mathbb E\|g_t\|^2$；理想 Muon 在这个证明里出现的则是 $\mathbb E\|G_t\|_*$。

#### 建立终点上界

继续使用 $S=\sum_t w_t$ 和 $\eta_t=w_t\sum_{s>t}w_s/S$。辅助矩阵按

$$
Z_{t+1}=Z_t-w_tH_t^\dagger G_t,
\qquad Z_1=W_1
$$

更新。令 $E_t=Z_t-W_*$，使用半范数 $\|E\|_H^2=\operatorname{tr}(E^\top HE)$。由 $H_tH_t^\dagger G_t=G_t$，直接展开得到

$$
2w_t\langle G_t,E_t\rangle_F
=\|E_t\|_{H_t}^2-\|E_{t+1}\|_{H_t}^2
+w_t^2\|G_t\|_*.
$$

$H_t$ 依赖当前梯度，并不妨碍这个逐样本恒等式成立。但它不具有累计 AdaGrad 的单调性，因此需要保留矩阵变差

$$
\Lambda_T^{\rm Muon}
=\lambda_{\max}(H_1)
+\sum_{t=2}^{T}[\lambda_{\max}(H_t-H_{t-1})]_+.
$$

假设目标凸、$G_t$ 在当前 $W_t$ 处条件无偏，辅助轨迹满足 $\|E_t\|_F\leq R$。代入终点转换，就得到

$$
\boxed{
\mathbb E[f(W_T)-f_*]
\leq
\frac{R^2\mathbb E\Lambda_T^{\rm Muon}
+\sum_t w_t^2\mathbb E\|G_t\|_*}{2S}.
}
$$

这里没有额外乘上矩阵列数，因为变化项直接满足

$$
\operatorname{tr}(E_t^\top\Delta H_t E_t)
\leq R^2[\lambda_{\max}(\Delta H_t)]_+.
$$

#### 求出依赖梯度谱的 LR Schedule

对候选 LR Schedule 采用共同的确定性界

$$
R^2\mathbb E\Lambda_T^{\rm Muon}\leq A,\qquad \mathbb E\|G_t\|_*\leq a_t, \qquad A>0,\ a_t>0.
$$

要最小化的表达式为

$$
\mathcal B_{\rm Muon}(w)
=\frac{A+\sum_t a_tw_t^2}{2\sum_t w_t}.
$$

令 $Q_t=\sum_{s=1}^{t}a_s^{-1}$。沿用 SGD 中的权重优化，固定总和时 $w_t\propto a_t^{-1}$，再对总和求极小，得到

$$
\boxed{
\eta_t^{\rm Muon} =\sqrt{\frac{A}{Q_T}}\frac1{a_t} \left(1-\frac{Q_t}{Q_T}\right),\qquad \mathcal B_{\min}=\sqrt{\frac{A}{Q_T}}.
}
$$

若核范数能量用同一个常数 $a$ 控制，便有

$$
\boxed{
\eta_t^{\rm Muon}
=\sqrt{\frac{A}{aT}}\left(1-\frac tT\right).
}
$$

这是常数能量界下的一个特例。这里优化的是 $\mathcal B_{\rm Muon}$，没有直接求出某个具体训练任务的真实终点损失最小值。解出的权重还须处于共同距离界、能量界成立的候选集合中。非恒定轮廓下，调度随核范数能量及其剩余累积量变化。若 $a_t$ 来自一次预训练记录，这仍然是依赖旧轨迹和未来统计的 refinement，不能直接当作可在线实现的最优控制。

要让这个上界随训练步数趋于零，还需要 $A/Q_T\to0$。$A$ 包含预条件矩阵的累计变化：即使核范数能量稳定，$H_t$ 的方向振荡也可能使 $\Lambda_T^{\rm Muon}$ 随 $T$ 线性增长。

#### 动量 Muon 中必须保留的修正

本小节将第 $t$ 步送入正交化的矩阵记为 $M_t$，它可以是普通动量，也可以包含 Nesterov 修正。用 $H_t=(M_tM_t^\top)^{1/2}$ 重复上面的代数，控制的是 $\langle M_t,Z_t-W_*\rangle_F$；凸性需要的仍然是当前随机梯度 $G_t$。两者相差

$$
\mathcal C_T
=\frac1S\sum_t w_t\,
\mathbb E\langle G_t-M_t,Z_t-W_*\rangle_F.
$$

因此，在精确极分解、无权重衰减的动量版本中，上界变为

$$
\mathbb E[f(W_T)-f_*]
\leq
\frac{R^2\mathbb E\Lambda_T(M)
+\sum_t w_t^2\mathbb E\|M_t\|_*}{2S}
+\mathcal C_T.
$$

即使动量的核范数近乎恒定，$\mathcal C_T$ 也不自动为零。可以用辅助轨迹的距离界，将它控制为

$$
\mathcal C_T\leq\frac1S\sum_t w_tb_t,
\qquad b_t\geq R\,\mathbb E\|G_t-M_t\|_F.
$$


取 $a_t\geq\mathbb E\|M_t\|_*$、$A\geq R^2\mathbb E\Lambda_T(M)$，并将它们与 $b_t$ 一起固定为候选调度的共同界，就得到 Adam 一节的 $\mathcal B_{\rm mom}(w)$。因此在正分支成立时，仍由

$$
\sum_t\frac{(\lambda-b_t)^2}{a_t}=A,\qquad w_t^*=\frac{\lambda-b_t}{a_t},\qquad \eta_t^*=w_t^*\frac{\sum_{s>t}w_s^*}{\sum_sw_s^*}
$$

确定上界最优调度。这里的能量换成了动量矩阵的核范数；动量的核范数稳定，并不意味着偏差项 $b_t$ 可以省略。

上述结果要求 $A,a_t,b_t$ 对候选 LR Schedule 共同成立。若用有限次 Newton–Schulz 代替精确极分解，还需在距离展开中保留近似残差。

矩阵级倍率也会进入结果。若理想更新是 $s_t\operatorname{polar}(G_t)$，其中 $s_t>0$，可将度量改为 $H_t/s_t$：能量变成 $s_t\|G_t\|_*$，矩阵变差也须同时改变。对恒定 $s>0$，学习率相应除以 $s$，实际位移保持一致。多个矩阵共用一个 LR Schedule 时，可以在分块度量下累加能量和变差；包含 AdamW 参数组的混合优化器，还需纳入那些参数组自身的界。

#### Warmup 与零终点约束

因此，给 Muon 使用前面的 warmup 加 Linear Decay 至 $0$，与这套分析并不矛盾。若固定 warmup，并在后续区间满足上述共同常数界，可以得到线性的上界最优形状。对带动量的版本，还需共同的常数偏差界；若偏差界随时间变化，最优权重一般不再恒定。仅凭优化器叫 Muon，或输出矩阵近乎正交，不能断言哪条曲线在实际任务上全局最优。


### 带动量时，真实终点如何求解

带动量 Muon 要保留参数与动量两个状态。以固定 $\beta$、矩阵倍率 $s$ 和衰减系数 $\lambda_{\rm wd}$ 为例：

$$
M_{t+1}=\beta M_t+(1-\beta)G(W_t,\xi_t),\qquad W_{t+1}=(1-\lambda_{\rm wd}\eta_t)W_t -\eta_t s\Phi(M_{t+1}).
$$

$\Phi$ 可以是精确极分解，也可以是实现中的有限次 Newton–Schulz 变换；使用 Nesterov 时应换成对应的输入矩阵。先把每一步非负学习率都作为变量，要解的是

$$
\min_{\eta_0,\ldots,\eta_{N-1}\geq0}
J(\eta)=\mathbb E_\xi f(W_N(\eta)).
$$

写 $X_t=(W_t,M_t)$，将完整更新记为 $X_{t+1}=F_t(X_t,\eta_t,\xi_t)$。这里 $\xi$ 的分布不依赖学习率；若噪声随参数变化，其依赖须已包含在 $G(W,\xi)$ 中。在可微且可交换求导与期望的条件下，对固定的样本序列反向传播：

$$
p_N=(\nabla f(W_N),0),\qquad p_t=(D_XF_t)^\top p_{t+1}.
$$

于是每一个时刻的学习率梯度为

$$
\boxed{
\frac{\partial J}{\partial\eta_t}
=-\mathbb E\left\langle
p_{t+1}^{W},\,
s\Phi(M_{t+1})+\lambda_{\rm wd}W_t
\right\rangle_F.
}
$$

对内部最优点，这个导数为零；若最优值落在 $\eta_t=0$，相应右导数应非负。$p_{t+1}^{W}$ 衡量当前位移对最终损失的影响，已经包含此后所有梯度、动量和正交化的变化。若只优化下一步真实损失 $f(W_{t+1})$，对应导数使用 $\nabla f(W_{t+1})$；使用当前梯度只给出一阶下降近似。

沿完整训练轨迹反向传播，可以计算 $\nabla_\eta J$，再在非负约束下优化各步学习率{{< cite 11 >}}。这个外层问题一般非凸，驻点条件只是必要条件；具体解依赖目标、数据分布和训练预算，通常需要数值求解。精确极分解在秩变化处还可能不可微。

若问题额外固定 warmup 前缀，就只优化其后的学习率；若要求最后一次实际更新的学习率为零，也应固定该变量。若进一步把衰减段限定为直线，形状已经由约束指定，剩下要选的是峰值等尚未固定的参数。这与允许所有非负学习率序列参与比较，是不同的优化问题。


<details class="technical-note">
<summary>补充：一个特殊矩阵模型的精确解，为什么可以不是 Linear</summary>

这里改为直接最小化一个二次模型的真实终点损失，优化所有非负标量步长，不固定 warmup，也不要求最后一次实际更新的步长为零。模型使用等奇异值参数和与参数列空间正交的特殊噪声，并去掉动量。下面得到的反比渐近形状只属于这个模型，不是实际 Muon 的通用最优 LR Schedule。

**模型与噪声假设**

设 $W_t\in\mathbb R^{m\times k}$，$m\geq2k$，目标为

$$
f(W)=\frac h2\|W\|_F^2,
\qquad h>0.
$$

初始矩阵满足 $W_0^\top W_0=(r_0/k)I_k$，$r_0>0$。也就是说，它的 $k$ 个奇异值相等。令 $r_t=\|W_t\|_F^2$，随机梯度取为

$$
G_t=hW_t+\Xi_t,
$$

其中噪声满足

$$
\mathbb E[\Xi_t\mid\mathcal F_{t-1}]=0,\qquad W_t^\top\Xi_t=0,\qquad \Xi_t^\top\Xi_t=\frac{\sigma^2}{k}I_k, \qquad \sigma>0.
$$

这里的噪声与当前参数列空间正交，且 Frobenius 范数固定为 $\sigma$。这样的无偏随机梯度确实可以构造：在 $W_t$ 的正交补中取 $k$ 个正交单位向量组成 $O_t$，再令 $\Xi_t=\pm\sigma O_t/\sqrt k$，正负号独立且等概率。条件 $m\geq2k$ 保证有足够的正交补空间。

噪声的方向依赖当前参数，因而不同于固定分布的加性高斯噪声，也不同于 SGD 一节的标量加性噪声模型。两节的数值用于各自模型内比较调度，不用于比较优化器的优劣。在这些条件下，损失可以由一个标量状态精确递推。

若当前 $W_t^\top W_t=(r_t/k)I_k$，则

$$
G_t^\top G_t
=\frac{h^2r_t+\sigma^2}{k}I_k.
$$

因此精确极分解有显式表达式：

$$
U_t=\operatorname{polar}(G_t) =\frac{\sqrt k\,G_t}{\sqrt{h^2r_t+\sigma^2}}, \qquad U_t^\top U_t=I_k.
$$

将它代入 $W_{t+1}=W_t-\eta_tU_t$，得到

$$
\boxed{
r_{t+1}
=r_t-\frac{2\eta_t h\sqrt k\,r_t}
{\sqrt{h^2r_t+\sigma^2}}+k\eta_t^2.
}
$$

这个等式对每个噪声样本都成立，尚未取期望。与此同时，

$$
W_{t+1}^\top W_{t+1}=\frac{r_{t+1}}kI_k,
$$

所以初始的等奇异值结构保持不变。因此，损失的演化只需跟踪标量 $r_t$。

**配方，再证明全程最优**

把上面的递推配方：

$$
r_{t+1} =\frac{\sigma^2r_t}{h^2r_t+\sigma^2} +k\left( \eta_t-\frac{hr_t}{\sqrt k\sqrt{h^2r_t+\sigma^2}} \right)^2.
$$

因此，当前状态下的最优步长是

$$
\eta^*(r)=\frac{hr}{\sqrt k\sqrt{h^2r+\sigma^2}},
$$

其最小下一步状态为

$$
\phi(r)=\frac{\sigma^2r}{h^2r+\sigma^2}.
$$

还差一步才能称为全程最优。注意

$$
\phi'(r)=\frac{\sigma^4}{(h^2r+\sigma^2)^2}>0.
$$

任意非负学习率都有 $r_{t+1}\geq\phi(r_t)$。由于 $\phi$ 单调，归纳可得

$$
r_N\geq\phi^{\circ N}(r_0).
$$

每一步选择 $\eta^*(r_t)$ 会达到等号。因此它同时最小化任意预算 $N$ 下的 $f(W_N)=hr_N/2$。这不仅覆盖预先设定的 LR Schedule，也覆盖根据历史、甚至当前噪声选择标量步长的策略：配方得到的下界对每条轨迹都成立。

**解出时间形式**

最优状态满足

$$
\frac1{r_{t+1}^*}=\frac1{r_t^*}+\frac{h^2}{\sigma^2}.
$$

记 $c=\sigma^2/(h^2r_0)$，于是

$$
r_t^*=\frac{\sigma^2}{h^2(t+c)}.
$$

代回步长，得到这个特殊模型中、无预设 warmup 与末次零步长约束的精确最优解：

$$
\boxed{
\eta_t^* =\frac{\sigma}{h\sqrt k} \frac1{\sqrt{(t+c)(t+c+1)}},\qquad \mathbb E f(W_N)_{\min} =\frac{\sigma^2}{2h(N+c)}.
}
$$

它没有在预定终点归零，且当 $t\gg c$ 时，

$$
\eta_t^*\sim\frac{\sigma}{h\sqrt k}\frac1t.
$$

矩阵宽度 $k$ 的因子来自 $\|U_t\|_F^2=k$。若实现使用 $sU_t$、$s>0$ 为固定倍率，则对应学习率为 $\eta_t^*/s$。

由于梯度的奇异值相等，极分解在这里等价于整体归一化。若改用同一个随机梯度做 SGD，相同位移对应

$$
\eta_t^{\rm SGD}
=\frac{\sqrt k}{\sqrt{h^2r_t+\sigma^2}}\eta_t^{\rm Muon}.
$$

在最优轨迹上，它化为 $\eta_t^{\rm SGD}=1/[h(t+c+1)]$。两种优化器能走出同一条参数轨迹，但对应的外部学习率并不相同。一般矩阵的奇异值不相等时，极分解会分别改变各个谱方向，这个标量换算就不再成立。

取 $h=\sigma^2=r_0=1$、$k=4$，执行 $N=100$ 次更新。精确解为 $\eta_t^*=1/[2\sqrt{(t+1)(t+2)}]$。作为比较，Linear、Cosine 和 Constant 分别取 $\alpha(1-t/N)$、$\alpha[1+\cos(\pi t/N)]/2$ 和 $\alpha$，索引均为 $t=0,\ldots,N-1$。对每组分别在峰值 $\alpha\in[0,1/\sqrt k]$ 内优化真实终点损失。

<figure>
<picture>
  <source media="(max-width: 640px)" srcset="/images/lr-schedules/muon-quadratic-mobile.svg">
  <img src="/images/lr-schedules/muon-quadratic-desktop.svg" alt="可解矩阵 Muon 模型的最优学习率及终点损失，与分别调节峰值的 Linear、Cosine 和 Constant 比较。" loading="lazy" width="1200">
</picture>
<figcaption>同一个矩阵 Muon 模型、相同预算下的精确递推。各组比较分别搜索峰值；纵轴为对数刻度。</figcaption>
</figure>

| LR Schedule | 最优或搜索得到的峰值 | 第 100 步期望损失 |
| --- | --- | --- |
| 精确最优 | $\eta_0^*=1/(2\sqrt2)$ | $0.0049505$ |
| Linear | $\alpha\approx0.0367602$ | $0.0064220$ |
| Cosine | $\alpha\approx0.0366828$ | $0.0071489$ |
| Constant | $\alpha\approx0.0158727$ | $0.0092292$ |

这个精确解依赖等奇异值与正交噪声假设。各向异性梯度、动量或近似正交化都会改变递推，$1/t$ 的渐近形状也不再由上述证明保证。

在这个特殊模型中，固定 warmup 前缀只会改变后续推导的初始状态 $r_\tau$，不会改变配方和单调性证明。若 $\eta_N=0$ 只是预算结束后不再执行的边界值，它不影响终点损失；若要求最后一次实际更新 $\eta_{N-1}=0$，则其前仍按上述公式优化，末步保持参数不变。固定 warmup 和终点归零本身，并不能推出中间必须是一条直线。

</details>

<details class="technical-note">
<summary>补充：一步下降上界，以及 Hyperball 的步长换算</summary>

对于一般方向 $u_t$，可以先解一个较局部的问题。假设 $f$ 的梯度是 $L_s$-Lipschitz，条件在更新前的历史上，定义

$$
\mathcal A_t=\mathbb E[\langle\nabla f(\theta_t),u_t\rangle\mid\mathcal F_{t-1}],\qquad \mathcal V_t=\mathbb E[\|u_t\|^2\mid\mathcal F_{t-1}].
$$

光滑性给出

$$
\mathbb E[f(\theta_t-\eta u_t)\mid\mathcal F_{t-1}]
\leq f(\theta_t)-\eta \mathcal A_t+\frac{L_s\eta^2\mathcal V_t}{2}.
$$

当 $\mathcal V_t>0$，这个二次上界在非负步长中的最小值取于

$$
\boxed{\eta_t^{\rm one}=\frac{[\mathcal A_t]_+}{L_s\mathcal V_t}.}
$$

分子度量方向与真实梯度的对齐，分母度量更新的平方长度。它允许不同优化器使用不同的 $u_t$，但解的是当前这一步的上界，不是固定终点的全程最优控制。

对无偏 SGD，若梯度噪声协方差为 $\Sigma_t$，上式化为

$$
\eta_t^{\rm one}
=\frac{\|\nabla f(\theta_t)\|^2}
{L_s\bigl(\|\nabla f(\theta_t)\|^2+\operatorname{tr}\Sigma_t\bigr)}.
$$

当信号相对噪声变小，一步最优上界步长下降。若更新带动量或者根据当前梯度生成预条件子，就必须重新计算 $\mathcal A_t,\mathcal V_t$，不能默认它们仍有这个表达式。

**正交化改变了长度与对齐**

Muon 对动量矩阵做近似正交化，实际实现还涉及 Newton–Schulz 近似与矩阵形状相关的倍率{{< cite 8 >}}。先去掉这些实现因素，考虑无噪声、无动量的理想情形：令真实矩阵梯度的紧致 SVD 为

$$
G=P\Sigma Q^\top,\qquad U=PQ^\top,
\qquad r=\operatorname{rank}(G)>0.
$$

直接计算得到

$$
\langle G,U\rangle_F=\sum_j\sigma_j(G)=\|G\|_*,
\qquad \|U\|_F^2=r.
$$

所以这一理想方向的一步上界最优步长为

$$
\boxed{\eta^{\rm one}=\frac{\|G\|_*}{L_sr}.}
$$

如果实现还乘了一个标量 $s>0$，即使用 $sU$，对应学习率会变成 $\|G\|_*/(L_ssr)$。更新的倍率必须与学习率约定一起讨论。

真实 Muon 对动量矩阵而非当前真实梯度做变换，随机正交化又是非线性的，此时对齐项一般不是 $\|\nabla f\|_*$。即使输出矩阵的 Frobenius 范数近乎固定，也没有固定它与真实梯度的夹角，因此不能只由“更新长度稳定”确定全程最优 LR Schedule。

**Hyperball 调度的是投影前的相对步长**

Hyperball 的受约束参数块使用

$$
W^+=R\frac{W-\eta_H R\,U/\|U\|_F}
{\|W-\eta_H R\,U/\|U\|_F\|_F},
\qquad \|W\|_F=R.
$$

这个归一化包装器可以接在 Adam 或 Muon 后面{{< cite 9 >}}。这里假设 $R>0$、$U\ne0$；零更新可直接跳过。记 $c=\langle W/R,U/\|U\|_F\rangle_F$。当 $0\leq\eta_H<1$，实际转角满足

$$
\theta=\arctan\frac{\eta_H\sqrt{1-c^2}}{1-\eta_Hc}.
$$

推导见[Hyperball 的几何分析](/blog/posts/hyperball/)。只有在小步长且更新近乎切向时，才有 $\theta\approx\eta_H$。对一般 $c$，给 $\eta_H$ 一条直线并不会给出严格线性的转角。

若目标是匹配普通更新 $W^+=W-\eta_{\rm direct}U$ 的投影前位移，则必须换算

$$
\eta_{\rm direct}
=\frac{\eta_H R}{\|U\|_F}.
$$

后续求最优 LR Schedule 时，还需要把球面投影写进损失递推。前面的欧氏终点定理和一步光滑性上界，都不能原样套在投影后的完整更新上。

</details>

## 引用本文

{{< cite_article >}}

[^index]: 终点转换包含 $T$ 个权重和梯度位置，但从 $\theta_1$ 到 $\theta_T$ 是 $T-1$ 次非零更新。末端 $g_T$ 用于理论界，不要求实现为了一个零步长再做反向传播。数值例子单独采用 $t=0,\ldots,N-1$，恰好执行 $N$ 次更新；Linear 在最后一次更新时为 $\alpha/N$，在预算边界 $t=N$ 才为零。

[^scalar]: 证明使用了单个状态 $r_t$ 和单调函数 $\phi$。多维二次损失用一个标量学习率时，不同特征方向的误差会耦合，一步最优一般不再自动意味着终点最优。这里也没有优化依赖当前样本或参数值的自适应反馈策略，只在确定性学习率序列中求最优。

[^bias]: 正分支存在的条件是 $A>\sum_t(\max_s b_s-b_t)^2/a_t$。若只优化代数表达式并允许非负权重，一般解为 $w_t=(\lambda-b_t)_+/a_t$，其中 $\sum_t(\lambda-b_t)_+^2/a_t=A$。将零权重方案放回终点转换时，还要处理末端零权重或用正权重极限；正文采用正分支以保留统一的终点编号。
