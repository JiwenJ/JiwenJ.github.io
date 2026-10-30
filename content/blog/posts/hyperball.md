---
title: "Hyperball：权重范数与角向更新"
date: "2026-10-30"
lastmod: "2026-10-31"
author: ["Jiwen Jiang"]
description: "Hyperball 的球面更新、角向步长与学习率迁移。"
summary: "Hyperball 固定权重范数并归一化更新。本文推导其实际转角，介绍学习率调度、HyperTransfer、HyperP 及相关实验。"
mathSummary: 'Hyperball 固定权重范数并归一化更新。本文推导其实际转角，介绍学习率调度、HyperTransfer、HyperP 及相关实验。'
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
  - id: 8
    title: "Rethinking Language Model Scaling under Transferable Hypersphere Optimization"
    authors: "Liliang Ren, Yang Liu, Yelong Shen, Weizhu Chen"
    source: "arXiv:2603.28743v2, 2026"
    url: "https://arxiv.org/abs/2603.28743v2"
  - id: 9
    title: "HyperTransfer: Understanding the Equivalence between Base Optimizer and Hyperball"
    authors: "Jinghui Yuan, Hongtao Zhang, Jade Zou, Tianyu Li, Wenjie Zhou, Tianyu He, Wei Chen"
    source: "arXiv:2609.07017v1, 2026"
    url: "https://arxiv.org/abs/2609.07017v1"
  - id: 10
    title: "The Best Optimizer Depends on Batch Size"
    authors: "Xingyu Dang, Kaiyue Wen, Sadhika Malladi"
    source: "arXiv:2610.08975v2, 2026"
    url: "https://arxiv.org/abs/2610.08975v2"
  - id: 11
    title: "Muon Reduces the Training Cost of Regulatory DNA Transformers"
    authors: "Viraj Doshi, Malhar Bhide, Akshat Singh, Yash Rathod, Aditya Sivakumar"
    source: "bioRxiv, 2026"
    url: "https://doi.org/10.64898/2026.07.17.739267"
  - id: 12
    title: "Truncate or Not? A Scaling Study of Looped Transformers with Truncated Backpropagation"
    authors: "Xuyuan Xiong, Siyu Chen, Zihan Shao, Yue Su, Weiyang Jin, Ping Luo, Zhuoran Yang, Xihui Liu"
    source: "arXiv:2610.26030v2, 2026"
    url: "https://arxiv.org/abs/2610.26030v2"
  - id: 13
    title: "Demystifying Manifold Constraints in LLM Pre-training"
    authors: "Kang An, Jiaxiang Li, Donald Goldfarb, Shiqian Ma"
    source: "arXiv:2605.04418v2, 2026"
    url: "https://arxiv.org/abs/2605.04418v2"
  - id: 14
    title: "Improving Neural Network Training by Decoupling the Magnitude and Direction of Weight Vectors"
    authors: "Alexander Hägele, Alejandro Hernández-Cano, Atli Kosson, Martin Jaggi"
    source: "arXiv:2606.25971v2, 2026"
    url: "https://arxiv.org/abs/2606.25971v2"
---

对尺度不变的权重矩阵，范数越大，同样长度的切向更新转过的角度越小。权重衰减通过改变范数，间接影响这个角度。

Hyperball 固定每个受约束矩阵的 Frobenius 范数，将优化器给出的更新归一化，走一步后再投回球面。接在 Adam 或 Muon 后面，就分别得到 AdamH 和 MuonH{{< cite 1 >}}。

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

定义相对更新长度

$$
\rho=\frac{\eta\|U\|_F}{\|W\|_F}
$$

则在 $U\perp W$ 时，$\theta=\arctan\rho$。非正交时，转角还取决于更新方向。

### 尺度不变性与权重衰减

在理想的尺度不变参数块中，保持其他参数不变，对所有 $a>0$ 都有 $L(aW)=L(W)$。对 $a$ 求导得到

$$
\langle W,\nabla_W L(W)\rangle_F=0.
$$

原始梯度位于球面的切空间。增大权重范数不改变当前函数，却会影响后续更新的转角{{< cite 2 >}}。实际 Transformer 中，残差、归一化增益和非齐次激活可能破坏尺度对称性；使用 RMSNorm 的矩阵也未必严格尺度不变[^scale]。

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

衰减使第一项缩小，更新通过后两项改变范数。即使原始梯度与权重正交，动量、Adam 的逐坐标预条件或 Muon 的矩阵变换也未必保留正交关系，交叉项可能非零。

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

这个平稳随机游走模型忽略了更新的时间相关性；上式给出的是均方尺度下的相对步长。

</details>

<details class="technical-note">
<summary>保留动量自相关后的平衡半径</summary>

原论文进一步保留了不同时间更新的相关性{{< cite 1 >}}。仍记 $Q=\mathbb E\|U_t\|_F^2$，定义

$$
a_h=\frac{\mathbb E\langle U_t,U_{t-h}\rangle_F}{Q},
\qquad C_\alpha=\sum_{h\ge1}\alpha^{h-1}a_h.
$$

在无限历史的平稳模型中，$\alpha=1-\eta\lambda\in(0,1)$ 时有

$$
r_*^2=\frac{\eta^2Q(1+2\alpha C_\alpha)}{1-\alpha^2}.
$$

没有时间相关性时，$C_\alpha=0$，恢复上面的结果。论文在独立各向同性高斯梯度与理想化基础更新下计算这些相关量：理想 Adam 有 $a_h=\beta_1^h$，SVD Muon 则涉及矩阵符号映射的自相关核。

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

第一步只保留基础更新的方向；第二步的提议位移长度为 $\eta_t R$；第三步沿径向回到球面，保持 $\|W\|_F=R$。

每个矩阵有自己的半径。若 $W\in\mathbb R^{d_{\rm out}\times d_{\rm in}}$ 的元素采用独立零均值高斯初始化，方差为 $1/d_{\rm in}$，则

$$
\mathbb E\|W_0\|_F^2=d_{\rm out}.
$$

典型半径因此接近 $\sqrt{d_{\rm out}}$，随初始化尺度和矩阵形状而定。

<details class="technical-note">
<summary>固定范数会限制表达能力吗</summary>

原论文给出一个局部的重参数化例子{{< cite 1 >}}。若线性层写成

$$
f(h;W,\gamma)=W\bigl(\gamma\odot\operatorname{RMSNorm}(h)\bigr),
$$

若 $\gamma$ 可训练，同时令 $W\mapsto aW$、$\gamma\mapsto\gamma/a$，函数值保持不变。增益能独立吸收矩阵尺度时，固定范数便可保留这部分表达能力；共享增益、固定增益或其他分支可能阻止这种补偿。这里同时改变了 $W$ 和 $\gamma$，与固定其他参数时的 $L(aW)=L(W)$ 条件不同，也不保证训练轨迹相同。

</details>

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

$\eta$ 是投影前的相对步长，$\theta$ 是实际转角，$2\sin(\theta/2)$ 是归一化的弦长。小角度下后两者接近；更新近似切向时，它们才同时接近 $\eta${{< cite 5 >}}。

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

脚本以双精度计算 450 个随机矩阵案例及 500 步连续更新，检查范数、角度和更新倍率不变性，同时覆盖零更新和无定义投影的情况。

</details>

### Muon、范数约束与参数化

Muon 处理更新矩阵的奇异值。在理想的 SVD 表述中，若动量矩阵为 $M=P\Sigma Q^\top$，正交化更新为 $PQ^\top$；实际实现通常使用有限次 Newton–Schulz 迭代近似这个变换{{< cite 3 >}}。Hyperball 在这之后做标量归一化与权重投影，二者作用的对象不同。

如果 $U=s\,\operatorname{Ortho}(M)$，其中 $s>0$ 只是一个矩阵级标量，那么

$$
\frac{U}{\|U\|_F}
=\frac{\operatorname{Ortho}(M)}{\|\operatorname{Ortho}(M)\|_F}.
$$

矩阵级更新倍率会被消掉，逐行、逐列倍率则一般不会。权重半径 $R$、矩阵拆分方式和前向倍率仍然影响算法。例如，将融合的 QKV 当作一个块或拆成三个块，会改变归一化和半径约束的单位。

Frobenius 约束也没有固定谱范数。例如，两个 $d\times d$ 矩阵

$$
A=\frac{R}{\sqrt d}I,
\qquad B=\operatorname{diag}(R,0,\ldots,0)
$$

都有 Frobenius 范数 $R$，谱范数却分别是 $R/\sqrt d$ 与 $R$。所以固定总平方和，仍允许不同的奇异值分布和最大方向放大率。

[上一篇文章](/blog/posts/mup/)介绍了 $\mu\mathrm{P}$ 的宽度迁移规则，包括参数坐标、初始化和更新倍率{{< cite 4 >}}。Hyperball 控制训练中的权重范数；与参数化结合后的尺度迁移将在第四节介绍。

### 实现时的参数块

原论文将约束用于注意力和 MLP 的权重矩阵，嵌入、归一化增益等其他参数使用普通优化器{{< cite 1 >}}。把所有参数拼成一个大向量再统一归一化，会让大矩阵支配总范数，不再是逐矩阵的 Hyperball。

包装器接收的是基础优化器生成的 $U_t$，而不是已经执行过一次参数更新的权重。动量等状态照常更新；在受约束的矩阵上，用归一化和投影替代原来的解耦权重衰减。若额外保留衰减，投影前会出现 $(1-\eta\lambda)W-\eta RV$，在 $1-\eta\lambda>0$ 时相当于又把相对步长改为 $\eta/(1-\eta\lambda)$。

NVIDIA Emerging-Optimizers 已提供 `MuonHyperball`，采用相同的更新式，并要求参数在构造时已具有指定的 Frobenius 半径{{< cite 7 >}}。分布式实现还涉及整个参数块的范数归约；按各个分片独立归一化会改变算法。低精度权重的范数计算与投影通常需要更高精度的累加，而保存和恢复优化器时也要保留各块的 $R$[^zero]。

## 三、实验结果与学习率调度

### 原论文实验

原论文在 Qwen3 风格的预训练设置中报告：对 1.2B 模型，MuonH 相对拟合的 AdamW 基线获得约 20%–30% 的 token-equivalent speedup，而对应 MuonW 约为 10%{{< cite 1 >}}。

论文先拟合 AdamW 的损失与训练预算关系，再根据其他方法的最终损失，估计 AdamW 达到同一损失所需的 token 数。加速比为 $N_{\rm AdamW}/N_{\rm method}$，衡量 token 效率，不包含每步耗时。

迁移实验固定每次训练 10B tokens，分别扫描宽度和深度：宽度实验固定 $L=4$、改变 $d\in[128,2048]$；深度实验固定 $d=128$、改变 $L\in[4,512]$。在带 QK-Norm 的混合归一化架构中，Hyperball 变体的最佳学习率漂移约为 $1.4$ 倍{{< cite 1 >}}。

原论文 v2 还报告了 130M 模型的长预算实验：训练量从 1B 扩展到 128B tokens，在每个预算下分别扫描学习率后，MuonH 的最佳 C4 验证损失均低于 MuonW{{< cite 1 >}}。这些结果使用各预算下单独调优的学习率，并未测试从短预算直接迁移。

### 学习率调度

Hyperball May Not Be a Free Lunch 观察到 MuonH 早期可能落后、后期反超带衰减的 Muon。作者用依赖训练状态的学习率匹配实验，在所测配置中复现了部分这种差异；径向分量对单步角向有效学习率的直接影响较小，单凭它不足以解释整个现象{{< cite 5 >}}。更激进的学习率衰减可以改善 MuonH 的早期损失，却可能损害后期表现。

另一项工作 Effective Learning Rate Governs Loss Dynamics in Language Model Pretraining 在直接更新 $W^+=W-\eta_{\rm direct}U$ 的约定下，用 $\eta_{\rm direct}/\|W\|_F$ 定义其 ELR，并通过匹配日程研究损失曲线的重合{{< cite 6 >}}。在作者测试的设置中，保留不同范数控制方式、只调整学习率，也能近似复现损失轨迹；重合精度受到 QK-Norm、可学习 RMSNorm 增益以及日程变化速度的影响。

套到 Hyperball 时，要先换算学习率。记它的学习率为 $\eta_H$，则投影前的步子是 $-\eta_H R U/\|U\|_F$，所以对应

$$
\eta_{\rm direct}=\frac{\eta_H R}{\|U\|_F},
\qquad
\operatorname{ELR}=\frac{\eta_{\rm direct}}{R}
=\frac{\eta_H}{\|U\|_F}.
$$

该文的 ELR 乘以更新范数，才是本文定义的相对步长 $\rho$。Hyperball 把投影前的 $\rho$ 设为 $\eta_H$；实际转角还取决于方向余弦，见第二节。

MuonW 的权重范数随训练变化，MuonH 的半径固定，二者的更新方向也可能逐渐分离。因此，相同的名义学习率日程未必产生相同的角向更新。

### HyperTransfer 的轨迹映射

HyperTransfer 构造了基础优化器与 Hyperball 之间的逐步映射{{< cite 9 >}}。对严格尺度不变的网络，在相同初始化和 minibatch 序列下，用在线代理范数 $s_t$ 表示对应基础优化器的半径，维持

$$
W_t^{\rm base}=\frac{s_t}{R}W_t^H.
$$

由于尺度不变函数的梯度满足 $\nabla L(aW)=\nabla L(W)/a$，映射先按 $R/s_t$ 重缩放 Hyperball 梯度，再更新动量或自适应状态，并由 $s_t$ 和更新范数确定学习率，使两组参数始终位于同一条射线上。代理范数由递推式在线计算，无须额外运行目标优化器。

这一映射同时校正梯度、优化器状态和学习率。对非尺度不变网络，HyperTransfer+ 在重新缩放的参数 $(s_t/R)W_t^H$ 处求梯度，对齐的是 $L((s_t/R)W_t^H)$，而非固定半径网络的 $L(W_t^H)$。有限精度下，方向仍可能逐渐偏离{{< cite 9 >}}。

### 不同 batch size 与任务上的结果

The Best Optimizer Depends on Batch Size 在相同总 token 预算下，对各个 batch size 分别调参并重复试验，比较权重衰减与 Hyperball 设置{{< cite 10 >}}。在其 Hyperball 实验中，128K tokens/update 时 MuonH 优于 ShampooH，而到 2M 时顺序反转。

调控 DNA Transformer 研究比较了 AdamW、AdamH、MuonW 和 MuonH；在所测设置中，Muon 配合独立权重衰减的表现优于 Hyperball{{< cite 11 >}}。

## 四、HyperP 与尺度迁移

HyperP 将 MuonH/AdamH 与配套参数化结合，研究宽度、深度、训练 token 数和 MoE 粒度的迁移{{< cite 8 >}}。

| 改变的尺度 | HyperP 的规则与适用条件 |
| --- | --- |
| 宽度 | 取 $R=C\sqrt{d_{\rm out}}$，其中 $C$ 与宽度无关；在近各向同性、矩阵长宽比同阶等假设下，隐藏矩阵无需额外乘宽度倍率。readout 和其他参数组仍需配套处理。 |
| 深度 | 联合调整残差分支倍率与层学习率。在小步长、局部 Jacobian 有界等假设下，固定范数不足以维持函数更新尺度。 |
| token 预算 | 隐藏矩阵组采用经验拟合的 $T^{-0.32}$ 倍率。$T$ 是整次训练的 token 预算，区别于一次训练内部的学习率衰减。 |
| MoE 粒度 | 用 SqrtGate 调整专家混合的尺度。改变专家数量或粒度时，还需控制路由、激活与辅助损失。 |

以深度为例，记残差分支倍率为 $a_L$，当权重和更新都归一化时，论文的一阶分析给出

$$
\eta_{\rm layer}=O\!\left(\frac{1}{L a_L}\right).
$$

取 $a_L\propto L^{-1/2}$ 时，层学习率也按 $L^{-1/2}$ 缩放。论文的 Transformer 配方采用 $(2L)^{-1/2}$ 残差倍率，其参数表在全局学习率上，为隐藏矩阵组乘 $T^{-0.32}L^{-1/2}$，为 embedding/vector 和 unembedding 组乘 $L^{-1/2}${{< cite 8 >}}。以 $(T_0,L_0)$ 为基准，隐藏矩阵的相对倍率就是 $(T/T_0)^{-0.32}(L/L_0)^{-1/2}$，其余两组为 $(L/L_0)^{-1/2}$。

后来的 Looped Transformer 研究采用 HyperP 派生的 MuonH 配方，把 token 预算指数从 $0.32$ 重新拟合为 $0.40$，并在其测试设置中发现无需额外深度修正{{< cite 12 >}}。该实验使用共享层的循环网络，与原来的深度分析条件不同。

## 五、实现与延伸阅读

以下实现可用于核对更新顺序、参数分组和半径设置：

| 实现 | 适合查看的内容 |
| --- | --- |
| [Marin / Levanter 的 AdamH 与 MuonH](https://github.com/marin-community/marin/tree/de412954e99488cf004b54104a7eeb4b670d10a2/lib/levanter/src/levanter/optim) | 作者关联的 JAX 训练栈，包含基础方向、范数投影和参数路由；链接固定到源码版本 |
| [Microsoft ArchScale](https://github.com/microsoft/ArchScale) | HyperP 官方代码，将 MuonH、AdamH、SqrtGate 与迁移规则放在同一训练配方中 |
| [NVIDIA MuonHyperball](https://docs.nvidia.com/nemo/emerging-optimizers/latest/apidocs/orthogonalized-optimizers.html#muonhyperball) | PyTorch 库接口；半径由使用者指定，构造时会检查权重范数是否匹配 |

计算更新方向和范数时，参数分组可能不同。例如，逐专家计算 Muon 方向后，仍可能对堆叠的 MoE 专家统一归一化。[Marin 的专家分组调查](https://github.com/marin-community/marin/issues/8621)记录了这种情况；它会改变 Hyperball 的约束对象。

MACRO 比较 Frobenius 与谱球面约束，通过 MuonH 变体实验分析切空间投影、RMSNorm 和低精度更新的作用{{< cite 13 >}}。Magnitude–Direction Decoupling 则为固定范数的方向引入可学习的行或列增益，并采用不同的更新缩放约定{{< cite 14 >}}。

更多论文、讲义与实现见 [Hyperball 资料库](https://github.com/JiwenJ/Awesome-Initialization/blob/main/docs/hyperball.md)。

## 引用本文

{{< cite_article >}}

[^scale]: 忽略归一化中的 $\epsilon$ 时，$\operatorname{Norm}(aWx)=\operatorname{Norm}(Wx)$ 可以给出局部的尺度不变例子。相反，$x+W\operatorname{Norm}(x)$ 中只缩放 $W$ 会改变残差分支相对主干的大小。允许同步调整其他增益、保持表达能力不变，也不同于固定其他参数后满足 $L(aW)=L(W)$。

[^zero]: $U_t=0$ 时无法归一化，可将该参数块的本次更新设为零。若提议点恰好为零，径向投影同样无定义；$0\le\eta<1$ 且当前范数为 $R>0$ 时可排除这个退化情形。几何脚本使用双精度并显式处理这些边界，不是完整的混合精度或分布式优化器实现。
