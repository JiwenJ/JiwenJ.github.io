---
title: "不同优化器的最优 LR Schedule"
date: "2026-10-31T00:00:00+08:00"
author: ["Jiwen Jiang"]
description: "从终点损失出发，推导 SGD、AdaGrad 和 Muon 的学习率调度。"
summary: "一个矩阵 Muon 模型的精确最优 LR Schedule、带动量时的全程求解条件，以及 SGD、AdaGrad 的终点误差上界。附推导与可复现代码。"
tags: ["Optimization", "Learning Rate", "SGD", "AdaGrad", "Muon"]
categories: ["Deep Learning"]
draft: false
math: true
isCJKLanguage: true
ShowToc: true
TocOpen: false
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
---

给定训练预算，怎样选择每一步的学习率，才能让最后的损失最小？这个问题需要同时给出目标函数、梯度噪声和优化器更新。仅凭“用了 Muon”或者“更新范数稳定”，还不能确定 LR Schedule 的形状。

先从 SGD 的终点误差上界开始：优化其中的权重，会得到 Linear Decay{{< cite 1 2 >}}，同一方法也能推广到 AdaGrad{{< cite 3 >}}。接着为无动量、精确极分解的 Muon 建立一个可解的矩阵随机模型，直接求真实终点损失的最优 LR Schedule。这个解与 $[(t+c)(t+c+1)]^{-1/2}$ 成正比，渐近为 $1/t$。带动量时，再把优化器状态和后续更新一起纳入求解。

## 一、先确定优化什么

统一记参数更新为

$$
\theta_{t+1}=\theta_t-\eta_t u_t.
$$

$u_t$ 是优化器生成的方向，包含动量、逐坐标预条件或矩阵正交化；$\eta_t$ 是外部标量学习率。SGD 的 $u_t=g_t$，AdaGrad 则把 $g_t$ 除以累计平方梯度的平方根。相同的 $\eta_t$，在两种优化器下对应不同的实际位移。

本文主要关注固定预算后的期望损失

$$
\mathbb E[f(\theta_T)-f_*],
\qquad f_*=f(\theta_*).
$$

它与平均参数 $\bar\theta_T$ 的损失、每一步下降多少，分别是不同的目标。下文区分两种求解：SGD、AdaGrad 和变度量分析优化的是**终点误差上界**；两个可解的二次模型直接优化**真实终点损失**。

先假设 $f$ 凸，$g_t$ 是当前点的无偏随机次梯度，外部权重为确定性序列。具体的有界性条件会在用到时给出。沿用终点分析的编号：初始参数为 $\theta_1$，输出为 $\theta_T$，最后的 $\eta_T=0$ 只是边界约定[^index]。

## 二、SGD 的线性衰减从哪里来

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
\begin{aligned}
\eta_t^*&=\frac{D}{G\sqrt T}\left(1-\frac tT\right),\\[6pt]
\mathbb E[f(\theta_T)-f_*]&\leq\frac{DG}{\sqrt T}.
\end{aligned}
}
$$

线性来自“剩余权重的占比”：每一步的权重相等，剩余权重就与剩余步数成正比。这个速率达到一般凸随机优化的最优最坏情况阶{{< cite 1 >}}。实际训练通常不知道 $D$ 和 $G$，因此会保留线性形状，另外搜索峰值学习率。

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
\begin{aligned}
w_t^*&=\frac{D}{a_t\sqrt{Q_T}},\\[6pt]
\eta_t^*&=\frac{D}{a_t\sqrt{Q_T}}
\left(1-\frac{Q_t}{Q_T}\right).
\end{aligned}
}
$$

常数 $a_t$ 会恢复 Linear Decay。若早期梯度能量大，前面的 $a_t^{-1}$ 会压低步长；能量随后下降时，这个因子可能产生 warmup，但并非任意下降曲线都会产生 warmup。

这里的 $a_t$ 必须作为固定上界参与优化。若改用一次训练中观测到的梯度曲线，换 LR Schedule 后梯度也会变，公式就变成基于旧轨迹的 refinement；原论文采用的正是先记录、平滑，再用于后续训练的方式{{< cite 1 2 >}}。它还需要未来的 $Q_T$，不能直接当作只看当前梯度的在线学习率算法。

## 三、AdaGrad 的预条件与外部调度

### 预条件以后要换一个能量

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

$A$、$B$ 固定是这个优化步骤的条件。求出的 LR Schedule 还须属于这些共同界成立的候选集合；否则需要在可行集合内重新优化。不能一边改变 LR Schedule 和轨迹，一边把实际的 $H_T$、梯度能量当作永远不变的已知数。

<details class="technical-note">
<summary>为什么单调预条件矩阵会给出迹项</summary>

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

</details>

### AdaGrad 的外部调度与坐标步长

AdaGrad 的对角预条件矩阵为

$$
H_{t,ii}=\sqrt{\epsilon+\sum_{s=1}^{t}g_{s,i}^2}.
$$

累计平方和单调增加，满足上一节需要的矩阵单调性。不过，预条件梯度能量并不因此恒定：若原始梯度的二阶矩近乎平稳，累计分母仍会增长。常数 $B$ 是用于分析的统一上界。取线性外部调度后，更新写成

$$
\boxed{
\theta_{t+1,i}=\theta_{t,i}
-\underbrace{\eta_0\left(1-\frac tT\right)}_{\text{全局学习率}}
\frac{g_{t,i}}{\sqrt{\epsilon+\sum_{s=1}^{t}g_{s,i}^2}}.
}
$$

逐坐标的完整步长包含自适应分母，因此它本身不必是直线。

还可以利用 AdaGrad 的经典遗憾界，得到更直接的等权结论{{< cite 4 >}}。下面将它与前面的终点转换组合，避免把所有预条件变化都压进一个粗略的迹上界。

<details class="technical-note">
<summary>用 AdaGrad 的遗憾界重新求一次最优权重</summary>

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

## 四、Adam 和动量多了哪些项

RMSProp 用滑动平均替代累计平方和。Adam 进一步引入一阶动量，其核心状态为

$$
\begin{aligned}
m_t&=\beta_1m_{t-1}+(1-\beta_1)g_t,\\
v_t&=\beta_2v_{t-1}+(1-\beta_2)g_t^{\odot2},\\
u_t&=\frac{\widehat m_t}{\sqrt{\widehat v_t}+\epsilon}.
\end{aligned}
$$

帽子表示从零初始化时的一阶、二阶矩偏差修正{{< cite 5 >}}。这里有两个不同的问题。

### 滑动平均失去单调性

当新梯度变小，$v_t$ 可以下降，因而不能直接使用 $H_t\succeq H_{t-1}$。对仍然使用当前梯度的更新 $H_t^{-1}g_t$，可以把迹项换成预条件矩阵的累计变化：

$$
\Lambda_T=\lambda_{\max}(H_1)
+\sum_{t=2}^{T}\bigl[\lambda_{\max}(H_t-H_{t-1})\bigr]_+,
$$

其中 $[a]_+=\max(a,0)$。因为

$$
e_t^\top(H_t-H_{t-1})e_t
\leq R^2\bigl[\lambda_{\max}(H_t-H_{t-1})\bigr]_+,
$$

前面的终点界仍可成立，只需将 $R^2\mathbb E\operatorname{tr}(H_T)$ 换成 $R^2\mathbb E\Lambda_T${{< cite 3 >}}。但即使 $H_t$ 始终有界，反复振荡也可能使 $\Lambda_T$ 持续增长；写出一个误差上界，不保证这个上界随训练步数趋于零。

Adam 的收敛反例以及 AMSGrad 对二阶矩历史最大值的使用，都说明这一步不能略去{{< cite 6 >}}。若已控制 $\Lambda_T$，并将能量项用共同常数界住，仍可得到线性的上界最优形状；这些条件需要先于结论成立。

### 一阶动量不是当前点的无偏梯度

终点转换依赖 $\mathbb E[g_t\mid\theta_1,\ldots,\theta_t]\in\partial f(\theta_t)$。一般来说，

$$
\mathbb E[m_t\mid\mathcal F_{t-1}]
=\beta_1m_{t-1}+(1-\beta_1)\nabla f(\theta_t),
$$

它不等于当前梯度。偏差修正消除的是从零初始化的缩放偏差，并没有消除过去梯度与当前梯度之间的差异。SGD with momentum 也有同一个问题。

因此，不能把所有公式中的 $g_t$ 直接替换成 $m_t$，就宣布完成 Adam 的证明。可以为带动量的基础算法单独建立遗憾界，再使用通用终点转换；也可以分析扩展状态 $(\theta_t,m_t,v_t)$。两条路线都需要额外的工作。

AdamW 还包含解耦权重衰减{{< cite 7 >}}：

$$
\theta_{t+1}=(1-\eta_t\lambda)\theta_t-\eta_tu_t.
$$

即使暂时忽略梯度更新，权重也会乘上 $\prod_t(1-\eta_t\lambda)$。固定 $\lambda$ 而改变 LR Schedule，同时改变了累计衰减；在相同模型上比较 Linear 与 Cosine 时，这也是训练过程发生的变化。

## 五、直接求 Muon 的最优 LR Schedule

先固定一个能够完整求解的目标和噪声模型。这一节使用当前随机梯度的精确极分解，不含动量、权重衰减或矩阵倍率；实际 Muon 对动量做 Newton–Schulz 变换，放在本节最后讨论{{< cite 8 10 >}}。

### 一个闭合的矩阵随机模型

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
\begin{aligned}
\mathbb E[\Xi_t\mid\mathcal F_{t-1}]&=0,\\
W_t^\top\Xi_t&=0,\\
\Xi_t^\top\Xi_t&=\frac{\sigma^2}{k}I_k,
\qquad \sigma>0.
\end{aligned}
$$

这里的噪声与当前参数列空间正交，且 Frobenius 范数固定为 $\sigma$。这样的无偏随机梯度确实可以构造：在 $W_t$ 的正交补中取 $k$ 个正交单位向量组成 $Q_t$，再令 $\Xi_t=\pm\sigma Q_t/\sqrt k$，正负号独立且等概率。条件 $m\geq2k$ 保证有足够的正交补空间。

这是一个依赖当前参数的随机梯度模型，不能等同于固定分布的加性高斯噪声。它的作用是保留矩阵正交化和非零噪声，同时让训练状态能够精确递推。

若当前 $W_t^\top W_t=(r_t/k)I_k$，则

$$
G_t^\top G_t
=\frac{h^2r_t+\sigma^2}{k}I_k.
$$

因此精确极分解有显式表达式：

$$
\begin{aligned}
U_t&=\operatorname{polar}(G_t)\\
&=\frac{\sqrt k\,G_t}{\sqrt{h^2r_t+\sigma^2}},
\qquad U_t^\top U_t=I_k.
\end{aligned}
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

所以初始的等奇异值结构保持不变。原本的矩阵训练过程，关于损失所需的状态闭合为一个标量 $r_t$。

### 配方，再证明全程最优

把上面的递推配方：

$$
\begin{aligned}
r_{t+1}
&=\frac{\sigma^2r_t}{h^2r_t+\sigma^2}\\
&\quad+k\left(
\eta_t-\frac{hr_t}{\sqrt k\sqrt{h^2r_t+\sigma^2}}
\right)^2.
\end{aligned}
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

### 解出时间形式

最优状态满足

$$
\frac1{r_{t+1}^*}=\frac1{r_t^*}+\frac{h^2}{\sigma^2}.
$$

记 $c=\sigma^2/(h^2r_0)$，于是

$$
r_t^*=\frac{\sigma^2}{h^2(t+c)}.
$$

代回步长，得到这个矩阵 Muon 模型的精确最优 LR Schedule：

$$
\boxed{
\begin{aligned}
\eta_t^*
&=\frac{\sigma}{h\sqrt k}
\frac1{\sqrt{(t+c)(t+c+1)}},\\[8pt]
\mathbb E f(W_N)_{\min}
&=\frac{\sigma^2}{2h(N+c)}.
\end{aligned}
}
$$

它没有在预定终点归零，且当 $t\gg c$ 时，

$$
\eta_t^*\sim\frac{\sigma}{h\sqrt k}\frac1t.
$$

这里没有预设 Linear、Cosine 或任何衰减族，LR Schedule 是从 Muon 更新和损失递推中解出来的。矩阵宽度 $k$ 的因子来自 $\|U_t\|_F^2=k$。若实现使用 $sU_t$、$s>0$ 为固定倍率，则对应学习率为 $\eta_t^*/s$。

这个模型还说明了为什么要注明学习率的定义。由于梯度的奇异值相等，极分解在这里等价于整体归一化。若改用同一个随机梯度做 SGD，相同位移对应

$$
\eta_t^{\rm SGD}
=\frac{\sqrt k}{\sqrt{h^2r_t+\sigma^2}}\eta_t^{\rm Muon}.
$$

在最优轨迹上，它化为 $\eta_t^{\rm SGD}=1/[h(t+c+1)]$。两种优化器能走出同一条参数轨迹，但它们报告的外部学习率并不相同。一般矩阵的奇异值不相等时，极分解会分别改变各个谱方向，这个标量换算就不再成立。

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

这里证明的是这个等奇异值、正交噪声模型内的全局最优。各向异性梯度、动量、近似正交化都会改变状态递推，因而不能直接把这个 $1/t$ 形状当作真实神经网络训练的最优答案。[复现代码](/code/muon-quadratic-check.py)、[逐步数据](/images/lr-schedules/muon-quadratic-comparison.csv)和[验证报告](/images/lr-schedules/muon-quadratic-verification.json)同时提供矩阵级恒等式检查与 LR Schedule 比较。

### 带动量时，怎样求全程最优

带动量 Muon 要保留参数与动量两个状态。以固定 $\beta$、矩阵倍率 $s$ 和衰减系数 $\lambda_{\rm wd}$ 为例：

$$
\begin{aligned}
M_{t+1}&=\beta M_t+(1-\beta)G(W_t,\xi_t),\\
W_{t+1}&=(1-\lambda_{\rm wd}\eta_t)W_t
-\eta_t s\Phi(M_{t+1}).
\end{aligned}
$$

$\Phi$ 可以是精确极分解，也可以是实现中的有限次 Newton–Schulz 变换；使用 Nesterov 时应换成对应的输入矩阵。现在要解的是

$$
\min_{\eta_0,\ldots,\eta_{N-1}\geq0}
J(\eta)=\mathbb E_\xi f(W_N(\eta)).
$$

写 $X_t=(W_t,M_t)$，将完整更新记为 $X_{t+1}=F_t(X_t,\eta_t,\xi_t)$。这里 $\xi$ 的分布不依赖学习率；若噪声随参数变化，其依赖须已包含在 $G(W,\xi)$ 中。在可微且可交换求导与期望的条件下，对固定的样本序列反向传播：

$$
\begin{aligned}
p_N&=(\nabla f(W_N),0),\\
p_t&=(D_XF_t)^\top p_{t+1}.
\end{aligned}
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

这给出了对实际更新求 LR Schedule 的方法：展开训练，用完整状态反传计算 $\nabla_\eta J$，在非负约束下优化各步学习率。通过训练轨迹反向优化 LR Schedule，已有超参数优化工作的先例{{< cite 11 >}}。这个外层问题一般非凸，驻点条件只是必要条件；需要指定目标、数据分布和预算后数值求解。精确极分解在秩发生变化时也可能不可微。本文没有运行神经网络上的这项外层优化，也没有由此宣布某个固定衰减族普遍最优。

## 六、Muon 的终点误差上界

### 把正交化写成矩阵预条件

先分析直接正交化当前随机梯度、没有动量与权重衰减、使用精确极分解的理想 Muon。Muon 的作者也区分了这个理论形式与对动量做 Newton–Schulz 近似的实际实现{{< cite 8 10 >}}。

设参数与梯度为矩阵 $W_t,G_t$。对只保留非零奇异值的紧致 SVD，定义

$$
G_t=P_t\Sigma_tQ_t^\top,
\qquad \operatorname{polar}(G_t)=P_tQ_t^\top.
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

它在前面度量分析中对应的梯度能量是

$$
\begin{aligned}
\operatorname{tr}(G_t^\top H_t^\dagger G_t)
&=\operatorname{tr}(\Sigma_t)\\
&=\sum_j\sigma_j(G_t)
=\|G_t\|_*.
\end{aligned}
$$

这里 $\|\cdot\|_*$ 专指**核范数，即奇异值之和**。SGD 的能量是 $\mathbb E\|g_t\|^2$；理想 Muon 在这个证明里出现的则是 $\mathbb E\|G_t\|_*$。

### 建立终点上界

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

这是将前面的终点转换和变度量恒等式应用到理想 Muon 得到的条件性上界。

### 求出依赖梯度谱的 LR Schedule

对候选 LR Schedule 采用共同的确定性界

$$
\begin{aligned}
R^2\mathbb E\Lambda_T^{\rm Muon}&\leq A,\\
\mathbb E\|G_t\|_*&\leq a_t,
\qquad A>0,\ a_t>0.
\end{aligned}
$$

要最小化的表达式为

$$
\mathcal B_{\rm Muon}(w)
=\frac{A+\sum_t a_tw_t^2}{2\sum_t w_t}.
$$

令 $Q_t=\sum_{s=1}^{t}a_s^{-1}$。与第二节相同，固定总和时 $w_t\propto a_t^{-1}$，再对总和求极小，得到

$$
\boxed{
\begin{aligned}
\eta_t^{\rm Muon}
&=\sqrt{\frac{A}{Q_T}}\frac1{a_t}
\left(1-\frac{Q_t}{Q_T}\right),\\[6pt]
\mathcal B_{\min}&=\sqrt{\frac{A}{Q_T}}.
\end{aligned}
}
$$

若核范数能量用同一个常数 $a$ 控制，便有

$$
\boxed{
\eta_t^{\rm Muon}
=\sqrt{\frac{A}{aT}}\left(1-\frac tT\right).
}
$$

这是常数能量界下的一个特例。这里优化的是 $\mathcal B_{\rm Muon}$，与上一节直接优化真实终点损失的目标不同。解出的权重还须处于共同距离界、能量界成立的候选集合中。非恒定轮廓下，调度随核范数能量及其剩余累积量变化。若 $a_t$ 来自一次预训练记录，这仍然是依赖旧轨迹和未来统计的 refinement，不能直接当作可在线实现的最优控制。

还有一个决定收敛的量没有被线性形状消除：$A$ 包含预条件矩阵的累计变化。即使核范数能量稳定，$H_t$ 的方向振荡也可能使 $\Lambda_T^{\rm Muon}$ 随 $T$ 线性增长。要让上述界趋于零，还需要 $A/Q_T\to0$。固定预算下求出上界最优形状，本身并不保证这条 LR Schedule 渐近收敛。

### 动量 Muon 中必须保留的修正

实际送入正交化的矩阵记为 $M_t$，它可能是动量，也可能包含 Nesterov 修正。用 $H_t=(M_tM_t^\top)^{1/2}$ 重复上面的代数，控制的是 $\langle M_t,Z_t-W_*\rangle_F$；凸性需要的仍然是当前随机梯度 $G_t$。两者相差

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

现在设 $a_t\geq\mathbb E\|M_t\|_*$，$A$ 控制度量变差；$A,a_t,b_t$ 都使用对候选 LR Schedule 共同成立的固定界。需要优化的完整表达式变成

$$
\mathcal B_{\rm mom}(w)
=\frac{A+\sum_ta_tw_t^2+2\sum_tb_tw_t}{2\sum_tw_t}.
$$

这也能解。若标量方程

$$
\sum_t\frac{(\lambda-b_t)^2}{a_t}=A
$$

有满足 $\lambda>\max_t b_t$ 的解，则

$$
\boxed{
\begin{aligned}
w_t^*&=\frac{\lambda-b_t}{a_t},\\[6pt]
\eta_t^*&=w_t^*\frac{\sum_{s>t}w_s^*}{\sum_sw_s^*},\\[6pt]
\mathcal B_{\rm mom,min}&=\lambda.
\end{aligned}
}
$$

验证只需配方。对任意正权重，

$$
2S[\mathcal B_{\rm mom}(w)-\lambda]
=\sum_ta_t(w_t-w_t^*)^2\geq0.
$$

这个结果保留了动量偏差的代价。若 $b_t=b$ 为常数，它只使最小上界增加 $b$，不改变最佳权重形状。若偏差界随时间变化，最优权重也会变化。正分支条件不满足时，最优解可能落到零权重边界，不能直接使用上面的正权重公式[^bias]。

因此，对带动量、精确极分解的版本，也能在给定这些界后求出终点上界最优的 LR Schedule。真实训练能否满足有用的共同界，仍取决于梯度与动量的演化。有限次 Newton–Schulz 还会带来相对精确极分解的残差，需在距离展开中保留。

矩阵级倍率也会进入结果。若理想更新是 $s_t\operatorname{polar}(G_t)$，其中 $s_t>0$，可将度量改为 $H_t/s_t$：能量变成 $s_t\|G_t\|_*$，矩阵变差也须同时改变。对恒定 $s>0$，学习率相应除以 $s$，实际位移保持一致。多个矩阵共用一个 LR Schedule 时，可以在分块度量下累加能量和变差；包含 AdamW 参数组的混合优化器，还需纳入那些参数组自身的界。

上述伪逆恒等式、核范数能量、变度量求和与权重极小值，可用[Muon 推导验证脚本](/code/muon-schedule-check.py)及其[数值报告](/images/lr-schedules/muon-verification.json)检查。它验证的是矩阵代数与固定统计量的上界优化。

## 七、给定更新方向的一步最优

对于一般方向 $u_t$，可以先解一个较局部的问题。假设 $f$ 的梯度是 $L_s$-Lipschitz，条件在更新前的历史上，定义

$$
\begin{aligned}
a_t&=\mathbb E[\langle\nabla f(\theta_t),u_t\rangle\mid\mathcal F_{t-1}],\\
b_t&=\mathbb E[\|u_t\|^2\mid\mathcal F_{t-1}].
\end{aligned}
$$

光滑性给出

$$
\mathbb E[f(\theta_t-\eta u_t)\mid\mathcal F_{t-1}]
\leq f(\theta_t)-\eta a_t+\frac{L_s\eta^2b_t}{2}.
$$

当 $b_t>0$，这个二次上界在非负步长中的最小值取于

$$
\boxed{\eta_t^{\rm one}=\frac{[a_t]_+}{L_sb_t}.}
$$

分子度量方向与真实梯度的对齐，分母度量更新的平方长度。它允许不同优化器使用不同的 $u_t$，但解的是当前这一步的上界，不是固定终点的全程最优控制。

对无偏 SGD，若梯度噪声协方差为 $\Sigma_t$，上式化为

$$
\eta_t^{\rm one}
=\frac{\|\nabla f(\theta_t)\|^2}
{L_s\bigl(\|\nabla f(\theta_t)\|^2+\operatorname{tr}\Sigma_t\bigr)}.
$$

当信号相对噪声变小，一步最优上界步长下降。若更新带动量或者根据当前梯度生成预条件子，就必须重新计算 $a_t,b_t$，不能默认它们仍有这个表达式。

### 正交化改变了长度与对齐

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

### Hyperball 调度的是投影前的相对步长

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

## 八、SGD 的精确二次模型

再对 SGD 解一次真实终点损失。这里使用标量加性噪声模型，与第五节 Muon 的矩阵正交噪声模型不同，两组数值不用于比较优化器的优劣：

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
\begin{aligned}
r_{t+1}
={}&\frac{\sigma^2r_t}{h^2r_t+\sigma^2}\\
&+(h^2r_t+\sigma^2)
\left(\eta_t-\frac{hr_t}{h^2r_t+\sigma^2}\right)^2.
\end{aligned}
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
\begin{aligned}
\eta_t^*&=\frac1{h\left(t+1+\dfrac{\sigma^2}{h^2r_0}\right)},\\[8pt]
\mathbb E f(x_N)_{\min}
&=\frac{h\sigma^2r_0}{2(\sigma^2+Nh^2r_0)}.
\end{aligned}
}
$$

这里得到的是反比衰减，而且不依赖提前指定的训练终点。二次损失的曲率结构，让我们可以比一般凸问题的统一上界求得更具体的答案。

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

在这个模型中，Linear 优于另外两组固定形状，但仍高于精确最优值。一般凸问题下的最坏情况结论，与具体二次问题上的结果，可以同时成立。更换曲率、噪声或优化器状态以后，应重新建立对应的递推；这张表不提供神经网络训练中四种 LR Schedule 的通用排名。

[验证脚本](/code/lr-schedule-check.py)、[逐步数据](/images/lr-schedules/comparison.csv)和[数值报告](/images/lr-schedules/verification.json)可以下载。脚本核对解析最优值，并为每条用于比较的 LR Schedule 独立搜索幅值。

<details class="technical-note">
<summary>复现曲线与数值</summary>

在装有 NumPy 和 Matplotlib 的环境中，将脚本保存为 `lr-schedule-check.py`，运行：

```bash
python3 lr-schedule-check.py \
  --csv comparison.csv \
  --report verification.json \
  --figure quadratic.svg \
  --mobile-figure quadratic-mobile.svg
```

所有比较都使用相同的 $N=100$ 和精确二阶矩递推，不需要随机种子。

</details>

## 引用本文

{{< cite_article >}}

[^index]: 终点转换包含 $T$ 个权重和梯度位置，但从 $\theta_1$ 到 $\theta_T$ 是 $T-1$ 次非零更新。末端 $g_T$ 用于理论界，不要求实现为了一个零步长再做反向传播。数值例子单独采用 $t=0,\ldots,N-1$，恰好执行 $N$ 次更新；Linear 在最后一次更新时为 $\alpha/N$，在预算边界 $t=N$ 才为零。

[^scalar]: 证明使用了单个状态 $r_t$ 和单调函数 $\phi$。多维二次损失用一个标量学习率时，不同特征方向的误差会耦合，一步最优一般不再自动意味着终点最优。这里也没有优化依赖当前样本或参数值的自适应反馈策略，只在确定性学习率序列中求最优。

[^bias]: 正分支存在的条件是 $A>\sum_t(\max_s b_s-b_t)^2/a_t$。若只优化代数表达式并允许非负权重，一般解为 $w_t=(\lambda-b_t)_+/a_t$，其中 $\sum_t(\lambda-b_t)_+^2/a_t=A$。将零权重方案放回终点转换时，还要处理末端零权重或用正权重极限；正文采用正分支以保留统一的终点编号。
