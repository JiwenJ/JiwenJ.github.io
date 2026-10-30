---
title: "Hyperball：权重范数与角向更新"
date: "2026-10-30"
author: ["Jiwen Jiang"]
description: "固定范数的更新规则、球面几何与学习率调度。"
summary: "从权重衰减的范数动力学解释 Hyperball，推导球面上的实际转角，再看它与 Muon、μP 和学习率调度的关系。"
mathSummary: '从权重衰减的范数动力学解释 Hyperball，推导球面上的实际转角，再看它与 Muon、$\mu\mathrm{P}$ 和学习率调度的关系。'
tags: ["Hyperball", "Muon", "Optimization"]
categories: ["Deep Learning"]
draft: false
math: true
isCJKLanguage: true
ShowToc: true
TocOpen: false
citationKey: "jiang2026hyperball"
citationUrl: "https://jiwenj.github.io/blog/posts/hyperball/"
references:
  - id: 1
    title: "Fantastic Pretraining Optimizers and Where to Find Them II: Hyperball Optimization"
    authors: "Kaiyue Wen, Xingyu Dang, Kaifeng Lyu, Tengyu Ma, Percy Liang"
    source: "arXiv:2606.16899v2, 2026"
    url: "https://arxiv.org/abs/2606.16899v2"
  - id: 2
    title: "Rotational Equilibrium: How Weight Decay Balances Learning Across Neural Networks"
    authors: "Atli Kosson, Bettina Messmer, Martin Jaggi"
    source: "ICML 2024"
    url: "https://arxiv.org/abs/2305.17212"
  - id: 3
    title: "Muon: An Optimizer for Hidden Layers in Neural Networks"
    authors: "Keller Jordan"
    source: "作者博客, 2024"
    url: "https://kellerjordan.github.io/posts/muon/"
  - id: 4
    title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
    authors: "Greg Yang et al."
    source: "arXiv:2203.03466, 2022"
    url: "https://arxiv.org/abs/2203.03466"
  - id: 5
    title: "Hyperball May Not Be a Free Lunch"
    authors: "Yihao Xiao et al."
    source: "arXiv:2607.22444, 2026"
    url: "https://arxiv.org/abs/2607.22444"
  - id: 6
    title: "Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining"
    authors: "Zihan Liu, Ruiheng Zheng, Shaobo Zhang, Changxin Tian, Kunlong Chen, Zhiqiang Zhang, Lei Wu"
    source: "arXiv:2608.24814, 2026"
    url: "https://arxiv.org/abs/2608.24814"
  - id: 7
    title: "MuonHyperball"
    authors: "NVIDIA"
    source: "Emerging-Optimizers 文档"
    url: "https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html"
---

同样的学习率，不一定意味着同样的学习速度。对一个方向比长度更重要的权重矩阵，范数越大，相同的一步更新通常转得越少。权重衰减会改变这个范数，也就间接改变了训练的步幅。

Hyperball 把这件事显式化：固定每个受约束的权重矩阵的 Frobenius 范数，将优化器给出的更新归一化，走一步以后再投回球面。它可以接在 Adam 或 Muon 后面，分别得到 AdamH 和 MuonH{{< cite 1 >}}。更新方向仍由基础优化器产生，Hyperball 负责它的长度以及权重所处的半径。

## 一、权重范数与有效步长

### 同样的一步，转过不同的角度

Frobenius 范数就是所有矩阵元素平方和的平方根。把矩阵展平成向量，可以直接使用欧氏几何。记当前权重为 $W$，基础优化器给出的更新为 $U$；$U$ 已包含动量、预条件或正交化，但不含外部学习率和权重衰减。普通更新写成

$$
W^+=W-\eta U.
$$

先看 $U$ 与 $W$ 正交的情形。设 $R=\|W\|_F$，$q=\|U\|_F$，那么新旧权重之间的夹角为

$$
\theta=\arctan\frac{\eta q}{R}.
$$

例如 $q=1$、$\eta=0.1$ 时，$R=1$ 对应约 $5.71^\circ$，$R=10$ 只对应约 $0.57^\circ$。两次更新的绝对长度都是 $0.1$，方向变化却相差近十倍。

这使相对更新长度

$$
\rho=\frac{\eta\|U\|_F}{\|W\|_F}
$$

成为比名义学习率更有用的观察量。不过，$\rho$ 还没有包含更新方向的信息；上面的反正切关系用到了正交条件。

### 尺度不变性与权重衰减

在理想的尺度不变参数块中，保持其他参数不变，对所有 $a>0$ 都有 $L(aW)=L(W)$。对 $a$ 求导得到

$$
\langle W,\nabla_W L(W)\rangle_F=0.
$$

原始梯度位于球面的切空间。此时增大权重范数并不改变当前函数，却会影响后续更新转过的角度。这是从旋转速度理解权重衰减的出发点{{< cite 2 >}}。实际 Transformer 中，残差、归一化增益和非齐次激活会破坏简单的尺度对称性，不能因为网络用了 RMSNorm，就认定每个矩阵都严格满足这个假设[^scale]。

带解耦权重衰减的更新为

$$
W_{t+1}=\alpha_t W_t-\eta_t U_t,
\qquad \alpha_t=1-\eta_t\lambda.
$$

直接展开平方范数：

$$
\begin{aligned}
\|W_{t+1}\|_F^2
={}&\alpha_t^2\|W_t\|_F^2+\eta_t^2\|U_t\|_F^2\\
&-2\alpha_t\eta_t\langle W_t,U_t\rangle_F.
\end{aligned}
$$

衰减使第一项缩小，更新通过后两项改变范数。即使原始梯度与权重正交，动量累积、Adam 的逐坐标预条件或 Muon 的矩阵变换，也未必保留这个正交关系。因此不能普遍删去交叉项。

<details class="technical-note">
<summary>一个可直接求解的范数平衡模型</summary>

固定 $\eta,\lambda$，假设更新是平稳的零均值扰动，满足 $\mathbb E\langle W_t,U_t\rangle_F=0$，且 $\mathbb E\|U_t\|_F^2=Q$ 为常数。记 $r_t^2=\mathbb E\|W_t\|_F^2$，有

$$
r_{t+1}^2=(1-\eta\lambda)^2r_t^2+\eta^2Q.
$$

当 $0<\eta\lambda<2$ 时，这个递推的平衡点为

$$
r_*^2=\frac{\eta^2Q}{1-(1-\eta\lambda)^2}
\approx\frac{\eta Q}{2\lambda}.
$$

近似使用了 $\eta\lambda\ll1$。用更新的均方尺度与平衡半径组成相对步长，可得

$$
\frac{\eta\sqrt Q}{r_*}\approx\sqrt{2\eta\lambda}.
$$

这是一个忽略更新时间相关性的随机游走模型，用来说明衰减怎样通过半径影响步幅。实际 Adam 或 Muon 的动量、更新与权重的相关性，以及变化中的学习率日程，都需要另行处理；该式也不是单步转角的精确公式。

</details>

## 二、Hyperball 的球面更新

### 固定半径，归一化更新

对一个受约束的矩阵，在初始化时保存 $R=\|W_0\|_F>0$，以后保持这个半径。Hyperball 的更新分为三步{{< cite 1 >}}：

$$
\begin{aligned}
V_t&=\frac{U_t}{\|U_t\|_F},\\
\widetilde W_{t+1}&=W_t-\eta_t R V_t,\\
W_{t+1}&=R\frac{\widetilde W_{t+1}}
                   {\|\widetilde W_{t+1}\|_F}.
\end{aligned}
$$

第一步只保留基础更新的方向；第二步的提议位移长度为 $\eta_t R$；第三步沿径向回到半径为 $R$ 的球面。这里的约束是 $\|W\|_F=R$，不是允许权重在球体内部任意移动。

每个矩阵有自己的半径。若 $W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$ 的元素采用独立零均值高斯初始化，方差为 $1/d_{\rm in}$，则

$$
\mathbb E\|W_0\|_F^2=d_{\rm out}.
$$

所以典型半径接近 $\sqrt{d_{\rm out}}$，而不是所有矩阵统一设成 1。半径保留了初始化的尺度选择。

### 学习率与实际转角

令 $E=W/R$、$V=U/\|U\|_F$，两者都是单位范数，记 $c=\langle E,V\rangle_F$。归一化后的更新可以直接写成

$$
E^+=\frac{E-\eta V}{\sqrt{1+\eta^2-2\eta c}}.
$$

因此，新旧权重的夹角满足

$$
\cos\theta=\frac{1-\eta c}{\sqrt{1+\eta^2-2\eta c}}.
$$

在 $0\le\eta<1$ 时，分母非零且 $1-\eta c>0$，也可以写成

$$
\theta=\arctan\!\left(
\frac{\eta\sqrt{1-c^2}}{1-\eta c}
\right).
$$

若 $c=0$，就得到 $\theta=\arctan\eta$；只有在小步长下才近似等于 $\eta$。若 $c=\pm1$，更新完全沿径向，在这个步长范围内投影后会回到原来的权重位置。固定 $\eta$ 并不固定转角。

<figure>
<picture>
<source media="(max-width: 600px)" srcset="/images/hyperball/geometry-mobile.svg">
<img src="/images/hyperball/geometry.svg" loading="lazy" decoding="async" alt="Hyperball 的二维球面更新示意，以及固定学习率下转角随权重和更新的夹角余弦变化的曲线。">
</picture>
<figcaption>二维几何计算。图 (a) 展示提议更新与径向投影；图 (b) 固定学习率、改变权重与更新的相对方向。曲线来自上式，不涉及模型训练。</figcaption>
</figure>

投影后的实际位移长度也不是 $\eta R$，而是球面上两点之间的弦长：

$$
\frac{\|W^+-W\|_F}{R}=2\sin\frac{\theta}{2}.
$$

于是有三个不同的量：$\eta$ 是投影前的相对步长，$\theta$ 是实际转角，$2\sin(\theta/2)$ 是归一化的弦长。小角度下后两者接近；它们与 $\eta$ 接近，还需要更新近似切向。后续对 Hyperball 的分析也专门讨论了这个方向相关性{{< cite 5 >}}。

<details class="technical-note">
<summary>它与先投影到切空间的更新有何区别</summary>

分解 $V=cE+V_\perp$，其中 $\langle E,V_\perp\rangle_F=0$。原算法投影前的方向为

$$
E-\eta V=(1-\eta c)E-\eta V_\perp.
$$

若 $1-\eta c>0$，它与

$$
E-\frac{\eta}{1-\eta c}V_\perp
$$

有相同的归一化方向。若先删除径向分量，再按同一个 $\eta$ 更新，得到的却是 $E-\eta V_\perp$，两者一般不同。径向分量虽然不直接沿球面移动，仍会通过归一化改变有限步长下的转角。

</details>

[几何验证脚本](/code/hyperball-geometry.py)和[逐点数据](/data/hyperball/geometry.csv)可下载。脚本直接计算更新前后的夹角，并与解析式比较；它还检查固定半径以及把 $U$ 乘以正数后更新结果不变这两个性质。

<details class="technical-note">
<summary>复现几何图</summary>

在装有 NumPy 和 Matplotlib 的环境中，将脚本下载到一个目录，并在该目录运行：

```bash
python3 hyperball-geometry.py \
  --csv geometry.csv --report verification.json \
  --figure geometry.svg --mobile-figure geometry-mobile.svg
```

脚本以双精度计算 450 个随机矩阵案例及 500 步连续更新，检查范数、角度和更新倍率不变性；同时覆盖零更新和无定义投影等边界。输出的 CSV 与 SVG 都是几何计算结果。

</details>

### Muon、范数约束与参数化

Muon 处理更新矩阵的奇异值。在理想的 SVD 表述中，若动量矩阵为 $M=P\Sigma Q^\top$，正交化更新为 $PQ^\top$；实际实现通常使用有限次 Newton–Schulz 迭代近似这个变换{{< cite 3 >}}。Hyperball 在这之后做标量归一化与权重投影，二者作用的对象不同。

如果 $U=s\,\operatorname{Ortho}(M)$，其中 $s>0$ 只是一个矩阵级标量，那么

$$
\frac{U}{\|U\|_F}
=\frac{\operatorname{Ortho}(M)}{\|\operatorname{Ortho}(M)\|_F}.
$$

这类统一的更新倍率会被消掉，逐行、逐列倍率则一般不会。与此相对，权重半径 $R$、矩阵如何拆分、前向计算中的倍率仍然保留在算法中。将融合的 QKV 当作一个块或拆成三个块，会改变归一化和半径约束的单位。

Frobenius 约束也没有固定谱范数。例如，两个 $d\times d$ 矩阵

$$
A=\frac{R}{\sqrt d}I,
\qquad B=\operatorname{diag}(R,0,\ldots,0)
$$

都有 Frobenius 范数 $R$，谱范数却分别是 $R/\sqrt d$ 与 $R$。所以固定总平方和，仍允许不同的奇异值分布和最大方向放大率。

在[上一篇文章](/blog/posts/mup/)中，$\mu\mathrm{P}$ 通过配套的参数坐标、初始化和更新规则研究宽度迁移{{< cite 4 >}}。Hyperball 则在给定模型和参数块上控制训练中的范数。前者讨论跨尺度的参数化，后者改变范数动力学；不能仅由球面约束推导出任意架构的超参数迁移。

### 实现时的参数块

原论文将约束用于注意力和 MLP 的权重矩阵，嵌入、归一化增益等其他参数使用普通优化器{{< cite 1 >}}。把所有参数拼成一个大向量再统一归一化，会让大矩阵支配总范数，不再是逐矩阵的 Hyperball。

包装器接收的是基础优化器生成的 $U_t$，而不是已经执行过一次参数更新的权重。动量等状态照常更新；在受约束的矩阵上，用归一化和投影替代原来的解耦权重衰减。若额外保留衰减，投影前会出现 $(1-\eta\lambda)W-\eta RV$，在 $1-\eta\lambda>0$ 时相当于又把相对步长改为 $\eta/(1-\eta\lambda)$。

NVIDIA Emerging-Optimizers 已提供 `MuonHyperball`，采用相同的更新式，并要求参数在构造时已具有指定的 Frobenius 半径{{< cite 7 >}}。分布式实现还涉及整个参数块的范数归约；按各个分片独立归一化会改变算法。低精度权重的范数计算与投影通常需要更高精度的累加，而保存和恢复优化器时也要保留各块的 $R$[^zero]。

## 三、实验结果与学习率调度

### 原论文的证据

原论文在 Qwen3 风格的预训练设置中报告：对 1.2B 模型，MuonH 相对拟合的 AdamW 基线获得约 20%–30% 的 token-equivalent speedup，而对应 MuonW 约为 10%{{< cite 1 >}}。这里比较的是达到相同损失所需的 token 数，不能直接读成墙钟时间缩短了 20%–30%。

具体做法是先拟合 AdamW 的损失与训练预算关系，再根据其他方法的最终损失，反推出 AdamW 所需的 token 数。论文使用的比值为 $N_{\rm AdamW}/N_{\rm method}$；它和直接测量每秒处理多少 token 是不同的指标。

迁移实验则固定每次训练 10B tokens，分别扫描宽度和深度：宽度实验固定 $L=4$、改变 $d\in[128,2048]$；深度实验固定 $d=128$、改变 $L\in[4,512]$。在带 QK-Norm 的混合归一化架构中，Hyperball 变体的最佳学习率漂移约为 $1.4$ 倍{{< cite 1 >}}。这些扫描提供了迁移的经验证据，其范围仍由所测架构、参数块和训练预算限定。

### 固定范数以后，日程仍然重要

Hyperball May Not Be a Free Lunch 观察到 MuonH 早期可能落后、后期反超带衰减的 Muon。作者用依赖训练状态的学习率匹配实验，在所测配置中复现了部分这种差异；径向分量对单步角向有效学习率的直接影响较小，单凭它不足以解释整个现象{{< cite 5 >}}。更激进的学习率衰减可以改善 MuonH 的早期损失，却可能损害后期表现。固定半径之后，步长日程仍影响收敛过程。

另一项工作 Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining 在直接更新 $W^+=W-\eta_{\rm direct}U$ 的约定下，用 $\eta_{\rm direct}/\|W\|_F$ 定义其 ELR，并通过匹配日程研究损失曲线的重合{{< cite 6 >}}。在作者测试的设置中，保留不同范数控制方式、只调整学习率，也能近似复现损失轨迹；重合精度受到 QK-Norm、可学习 RMSNorm 增益以及日程变化速度的影响。

套到 Hyperball 时，要先换算学习率。记它的学习率为 $\eta_H$，则投影前的步子是 $-\eta_H R U/\|U\|_F$，所以对应

$$
\eta_{\rm direct}=\frac{\eta_H R}{\|U\|_F},
\qquad
\operatorname{ELR}=\frac{\eta_{\rm direct}}{R}
=\frac{\eta_H}{\|U\|_F}.
$$

它不是把 Hyperball 的学习率直接除以半径得到的 $\eta_H/R$。这个 ELR 又与本文的相对更新长度 $\rho$ 不同：$\rho$ 还包含更新范数，而精确转角还依赖方向余弦。Hyperball 把投影前的 $\rho$ 设成 $\eta_H$，实际角度仍满足第二部分的公式。

使用同一条名义学习率日程比较 MuonW 与 MuonH，并不保证匹配它们的实际角向更新：前者的范数会演化，后者的半径固定，二者的更新方向也可能随训练分离。是否更快收敛，要由完整训练轨迹回答；固定半径本身并没有决定何时应当减小步幅。

## 引用本文

{{< cite_article >}}

[^scale]: 忽略归一化中的 $\epsilon$ 时，$\operatorname{Norm}(aWx)=\operatorname{Norm}(Wx)$ 可以给出局部的尺度不变例子。相反，$x+W\operatorname{Norm}(x)$ 中只缩放 $W$ 会改变残差分支相对主干的大小。允许同步调整其他增益、保持表达能力不变，也不同于固定其他参数后满足 $L(aW)=L(W)$。

[^zero]: $U_t=0$ 时无法归一化，可将该参数块的本次更新设为零。若提议点恰好为零，径向投影同样无定义；$0\le\eta<1$ 且当前范数为 $R>0$ 时可排除这个退化情形。几何脚本使用双精度并显式处理这些边界，不是完整的混合精度或分布式优化器实现。
