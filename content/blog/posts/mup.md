---
title: "μP：前身、原理与后续发展"
mathTitle: '$\mu\mathrm{P}$：前身、原理与后续发展'
date: "2026-10-25T18:28:00+08:00"
author: ["Jiwen Jiang"]
description: "从无限宽理论走到最大更新参数化，再看深度、优化器、MoE 与训练预算下的超参数迁移。"
mathDescription: '从无限宽理论走到 $\mu\mathrm{P}$，再看深度、优化器、MoE 与训练预算下的超参数迁移。'
summary: "沿着初始化、NTK 与两种均值场的线索理解 μP，并梳理超参数迁移及后续发展。"
mathSummary: '沿着初始化、NTK 与两种均值场的线索理解 $\mu\mathrm{P}$，并梳理超参数迁移及后续发展。'
tags: ["μP", "Initialization", "NTK", "Mean Field"]
categories: ["Deep Learning"]
draft: false
math: true
isCJKLanguage: true
ShowToc: true
TocOpen: false
citationKey: "jiang2026mup"
citationUrl: "https://jiwenj.github.io/blog/posts/mup/"
references:
  - id: 1
    title: "Understanding the Difficulty of Training Deep Feedforward Neural Networks"
    authors: "Xavier Glorot, Yoshua Bengio"
    source: "AISTATS 2010"
    url: "https://proceedings.mlr.press/v9/glorot10a.html"
  - id: 2
    title: "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
    source: "arXiv:1502.01852, 2015"
    url: "https://arxiv.org/abs/1502.01852"
  - id: 3
    title: "Deep Neural Networks as Gaussian Processes"
    authors: "Jaehoon Lee et al."
    source: "ICLR 2018"
    url: "https://arxiv.org/abs/1711.00165"
  - id: 4
    title: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
    authors: "Arthur Jacot, Franck Gabriel, Clément Hongler"
    source: "NeurIPS 2018"
    url: "https://arxiv.org/abs/1806.07572"
  - id: 5
    title: "A Mean Field View of the Landscape of Two-Layers Neural Networks"
    authors: "Song Mei, Andrea Montanari, Phan-Minh Nguyen"
    source: "arXiv:1804.06561, 2018"
    url: "https://arxiv.org/abs/1804.06561"
  - id: 9
    title: "Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks"
    authors: "Greg Yang, Edward J. Hu"
    source: "ICML 2021"
    url: "https://proceedings.mlr.press/v139/yang21c.html"

  - id: 6
    title: "Exponential Expressivity in Deep Neural Networks through Transient Chaos"
    authors: "Ben Poole et al."
    source: "NIPS 2016"
    url: "https://arxiv.org/abs/1606.05340"
  - id: 7
    title: "Deep Information Propagation"
    authors: "Samuel S. Schoenholz et al."
    source: "ICLR 2017"
    url: "https://arxiv.org/abs/1611.01232"
  - id: 8
    title: "Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes"
    authors: "Greg Yang"
    source: "NeurIPS 2019"
    url: "https://arxiv.org/abs/1910.12478"
  - id: 10
    key: "yang2022tensorprogramsvtuning"
    title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
    authors: "Greg Yang et al."
    source: "arXiv:2203.03466, 2022"
    url: "https://arxiv.org/abs/2203.03466"
  - id: 11
    title: "Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks"
    authors: "Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou"
    source: "arXiv:2310.02244, 2023"
    url: "https://arxiv.org/abs/2310.02244"
  - id: 12
    title: "u-μP: The Unit-Scaled Maximal Update Parametrization"
    authors: "Charlie Blake et al."
    source: "ICLR 2025"
    url: "https://arxiv.org/abs/2407.17465"
  - id: 13
    key: "dey2026dontlazycompletepenables"
    title: "Don't Be Lazy: CompleteP Enables Compute-Efficient Deep Transformers"
    authors: "Nolan Dey et al."
    source: "NeurIPS 2025"
    url: "https://arxiv.org/abs/2505.01618"
  - id: 14
    key: "yang2024spectralconditionfeaturelearning"
    title: "A Spectral Condition for Feature Learning"
    authors: "Greg Yang, James B. Simon, Jeremy Bernstein"
    source: "arXiv:2310.17813, 2023"
    url: "https://arxiv.org/abs/2310.17813"
  - id: 15
    key: "zheng2026spectralconditionmupwidthdepth"
    title: "Spectral Condition for μP under Width-Depth Scaling"
    authors: "Chenyu Zheng et al."
    source: "arXiv:2603.00541, 2026"
    url: "https://arxiv.org/abs/2603.00541"
  - id: 16
    key: "lee2026completep"
    title: "CompleteP for RL: Maintaining Feature Learning When Scaling Deep Reinforcement Learning"
    authors: "Adam Lee et al."
    source: "ICML 2026"
    url: "https://proceedings.mlr.press/v306/lee26ai.html"
  - id: 17
    key: "mlodozeniec2025completedhyperparametertransfermodules"
    title: "Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration"
    authors: "Bruno Mlodozeniec et al."
    source: "arXiv:2512.22382, 2025"
    url: "https://arxiv.org/abs/2512.22382"
  - id: 18
    key: "chickering2026gqamupmaximalparameterizationupdate"
    title: "GQA-μP: The maximal parameterization update for grouped query attention"
    authors: "Kyle R. Chickering et al."
    source: "arXiv:2605.15290, 2026"
    url: "https://arxiv.org/abs/2605.15290"
  - id: 19
    key: "liu2025muonscalablellmtraining"
    title: "Muon is Scalable for LLM Training"
    authors: "Jingyuan Liu et al."
    source: "arXiv:2502.16982, 2025"
    url: "https://arxiv.org/abs/2502.16982"
  - id: 20
    key: "gupta2026extendingmupspectralconditions"
    title: "Extending μP: Spectral Conditions for Feature Learning Across Optimizers"
    authors: "Akshita Gupta et al."
    source: "arXiv:2602.20937, 2026"
    url: "https://arxiv.org/abs/2602.20937"
  - id: 21
    key: "xu2026widthscalingneuraloptimizers"
    title: "On the Width Scaling of Neural Optimizers Under Matrix Operator Norms I: Row/Column Normalization and Hyperparameter Transfer"
    authors: "Ruihan Xu, Jiajin Li, Yiping Lu"
    source: "arXiv:2603.09952, 2026"
    url: "https://arxiv.org/abs/2603.09952"
  - id: 22
    key: "qiu2026hyperparametertransferenablesconsistent"
    title: "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales"
    authors: "Shikai Qiu et al."
    source: "NeurIPS 2025"
    url: "https://arxiv.org/abs/2512.05620"
  - id: 23
    key: "zhao2026principledmuonmumathsfpensuring"
    title: "Towards a Principled Muon under μP: Ensuring Spectral Conditions throughout Training"
    authors: "John Zhao"
    source: "arXiv:2601.01306, 2026"
    url: "https://arxiv.org/abs/2601.01306"
  - id: 24
    key: "xie2026controlledllmtrainingspectral"
    title: "Controlled LLM Training on Spectral Sphere"
    authors: "Tian Xie et al."
    source: "arXiv:2601.08393, 2026"
    url: "https://arxiv.org/abs/2601.08393"
  - id: 25
    key: "velikanov2026learnablemultipliersfreeingscale"
    title: "Learnable Multipliers: Freeing the Scale of Language Model Matrix Layers"
    authors: "Maksim Velikanov et al."
    source: "arXiv:2601.04890, 2026"
    url: "https://arxiv.org/abs/2601.04890"
  - id: 26
    key: "ren2026rethinkinglanguagemodelscaling"
    title: "Rethinking Language Model Scaling under Transferable Hypersphere Optimization"
    authors: "Liliang Ren et al."
    source: "arXiv:2603.28743, 2026"
    url: "https://arxiv.org/abs/2603.28743"
  - id: 27
    key: "malasnicki2025muparametrizationmixtureexperts"
    title: "μ-Parametrization for Mixture of Experts"
    authors: "Jan Małaśnicki et al."
    source: "arXiv:2508.09752, 2025"
    url: "https://arxiv.org/abs/2508.09752"
  - id: 28
    key: "jiang2026hyperparametertransfermixtureofexpertlayers"
    title: "Hyperparameter Transfer with Mixture-of-Expert Layers"
    authors: "Tianze Jiang et al."
    source: "arXiv:2601.20205, 2026"
    url: "https://arxiv.org/abs/2601.20205"
  - id: 29
    key: "vankadara2026scalemixtureofexpertsmupmaximally"
    title: "How to Scale Mixture-of-Experts: From muP to the Maximally Scale-Stable Parameterization"
    authors: "Leena Chennuru Vankadara et al."
    source: "arXiv:2605.14200, 2026"
    url: "https://arxiv.org/abs/2605.14200"
  - id: 30
    key: "peng2026completemueoptimalhyperparametertransfer"
    title: "Complete-muE: Optimal Hyperparameter Transfer and Scaling for MoE Models"
    authors: "Hongwu Peng et al."
    source: "arXiv:2605.23893v2, 2026"
    url: "https://arxiv.org/abs/2605.23893v2"
  - id: 31
    key: "dhir2026why"
    title: "Why Routers Freeze: Infinite Width Learning Dynamics for Mixture of Experts"
    authors: "Anish Dhir, Volkan Cevher, Leena Chennuru Vankadara"
    source: "OpenReview, 2026"
    url: "https://openreview.net/forum?id=AhwB471ARJ"
  - id: 32
    key: "zhou2026setlearningratelargescale"
    title: "How to Set the Learning Rate for Large-Scale Pre-training?"
    authors: "Yunhua Zhou et al."
    source: "arXiv:2601.05049, 2026"
    url: "https://arxiv.org/abs/2601.05049"
  - id: 33
    key: "bergsma2026scalingcollapseefficientpredictable"
    title: "Scaling with Collapse: Efficient and Predictable Training of LLM Families"
    authors: "Shane Bergsma et al."
    source: "ICLR 2026"
    url: "https://arxiv.org/abs/2509.25087"
  - id: 34
    key: "kim2026letsscalestepstep"
    title: "Let's Scale Step by Step: Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts"
    authors: "Nayeon Kim et al."
    source: "arXiv:2608.20061, 2026"
    url: "https://arxiv.org/abs/2608.20061"
---

把网络变宽，最后会得到一个固定核，还是一个仍在学习表示的模型？答案取决于参数化。理解这一点，比先记住一张学习率缩放表更有帮助。

这篇文章沿三条问题展开：$\mu\mathrm{P}$ 从哪些理论问题中走来，它怎样组织训练尺度，以及后来的工作又把问题推向了哪里[^reading]。先把第一部分的线索讲清楚。

## 一、$\mu\mathrm{P}$ 的前身

初始化、NNGP、NTK 与均值场是彼此关联的研究视角。下面的阅读顺序从信号的起点走到训练中的特征变化；它们在历史上也有并行发展。这里固定输入维度与网络深度，让隐藏层宽度 $n$ 增长。$O(1)$ 表示量级不随宽度增长，而非数值恰好等于一[^scales]。

### 标准参数化：同一个函数，可以有不同的训练方式

先从常见的标准参数化（Standard Parametrization，SP）出发：程序直接存储各层权重，以 fan-in 一类规则初始化，再按约定的学习率训练。做宽度比较时，还必须说明这个学习率是否随宽度变化。

参数化要同时交代四件事：**实际存储的参数、前向计算中的倍率、初始化分布、优化器的学习率。** 只知道某个权重的标准差，还不足以确定它会怎样训练。

对于固定的重参数化 $w=s\theta$，普通 SGD 在实际权重坐标下的有效学习率为

$$
\eta_w=\eta_\theta s^2.
$$

这意味着，前向表达式相同，并不自动带来相同的训练轨迹。倍率、初始化与学习率需要一起匹配。

<details class="technical-note">
<summary>展开推导：为什么学习率会多出一个平方？</summary>

设真正参与计算的权重为 $w$，程序存储的参数为 $\theta$，二者满足

$$
w=s\theta,
$$

其中 $s$ 可以依赖模型宽度，但在一次训练中固定。这里的 $\mathcal L$ 仅指数据损失，不含另外施加的权重衰减。使用普通 SGD，根据链式法则，

$$
\nabla_\theta\mathcal L=s\nabla_w\mathcal L,
\qquad
\Delta w=s\Delta\theta
=-\eta_\theta s^2\nabla_w\mathcal L.
$$

因此，以 $\theta$ 为坐标训练时，在 $w$ 坐标下对应的有效学习率为

$$
\eta_w=\eta_\theta s^2.
$$

例如，把 $s$ 改成 $1/\sqrt n$，同时保持 $\eta_\theta$ 不变，会让 $w$ 坐标下的有效学习率多出 $1/n$。即使通过调整初始化使两个实现拥有相同的初始函数分布，它们后续的训练轨迹仍可能不同。


</details>

这个推导限定在普通 SGD。Adam 会利用梯度历史归一化更新，不能直接套用上面的 $s^2$ 规则。讨论宽度缩放之前，必须先固定参数坐标和优化器约定。

### Xavier 与 He：先让信号顺利通过网络

考虑一层线性变换

$$
z_j=\sum_{i=1}^{n_{\rm in}}W_{ji}h_i.
$$

在零均值权重、适当独立性等初始化近似下，它的二阶矩满足

$$
\mathbb E[z_j^2]
\approx n_{\rm in}\operatorname{Var}(W_{ji})\mathbb E[h_i^2].
$$

这说明，输入项变多时，单项权重需要相应缩小，否则信号会被求和放大。反向传播也有类似计算，但求和数量与输出维度有关。

Xavier 初始化综合考虑前向与反向传播，在接近线性的分析下给出常见的方差选择{{< cite 1 >}}：

$$
\operatorname{Var}(W_{ji})
=\frac{2}{n_{\rm in}+n_{\rm out}}.
$$

He 初始化进一步考虑 ReLU：对称输入经过 ReLU 后，二阶矩减半，因此保持前向二阶矩的常见选择为{{< cite 2 >}}

$$
\operatorname{Var}(W_{ji})=\frac{2}{n_{\rm in}}.
$$

这些规则使随机初始化时的激活与梯度更容易保持合适尺度。输入、输出维度不同时，前向和反向的稳定性通常还需要取舍。

接下来要问的是训练问题：梯度乘以学习率后，隐藏表示实际改变了多少？初始化没有独立规定学习率，也没有自动保证宽度变化时的特征更新尺度。这正是后续理论继续追问的地方。

### 高斯过程：描述随机初始化的函数分布

把隐藏层宽度记为 $n$，考虑标量输出

$$
f_n(x)=\frac{1}{\sqrt n}
\sum_{i=1}^{n}a_i\phi(w_i^\top x).
$$

假设各神经元独立同分布初始化，$a_i$ 与 $w_i$ 独立，且 $a_i$ 零均值、相关二阶矩有限。对固定输入，输出是许多独立随机贡献的归一化求和。对一组固定输入同时应用中心极限定理，它们的输出可以收敛到联合高斯分布。

在这个设定下，输出方差为

$$
\operatorname{Var}[f_n(x)]
=\operatorname{Var}[a\phi(w^\top x)].
$$

若右侧是有限非零常数，初始输出的波动便保持 $O(1)$。关键是独立、零均值贡献的求和及其归一化；单个神经元的激活本身并不需要服从高斯分布。

由此得到的高斯过程视角，把随机网络看成一个函数先验。其协方差由初始化和激活函数决定；Lee 等人的工作将这一对应关系系统地扩展到深层网络{{< cite 3 >}}[^gp]。

这里关注的是初始化时 $f_0$ 的随机分布。它可以用于相应的贝叶斯推断，但不能单凭这个对应关系，就断言梯度下降中的隐藏特征是否变化。要描述训练，还需要研究函数对参数的敏感度。

### NTK：输出能学习，隐藏特征却可以近乎不动

神经正切核将参数梯度组合成一个输入之间的相似度：

$$
K_\theta(x,x')
=\left\langle\nabla_\theta f(x),
\nabla_\theta f(x')\right\rangle.
$$

仍使用上一节的 $1/\sqrt n$ 两层网络，并令存储参数 $a_i,w_i$ 的初始化量级为 $O(1)$。若损失导数、输入与相关激活导数保持有界，对单个隐藏权重有

$$
\nabla_{w_i}f_n(x)
=\frac{a_i}{\sqrt n}\phi'(w_i^\top x)x
=O(n^{-1/2}).
$$

因此，在学习率 $\eta=O(1)$ 的普通 SGD 下，固定索引的典型单元有单步 $\Delta w_i=O(n^{-1/2})$。在不随宽度增长的有限训练时间内，并满足相应正则性与稳定性条件时，这类单元的特征变化可以随宽度增大而消失。这里的“固定时间”限定很重要[^time]。

输出仍然可以发生 $O(1)$ 变化。原因是大量微小更新共同作用于输出：单元变化小，并不等于网络函数不学习。NTK 理论刻画的一类极限中，初始化核收敛到确定性极限，并在训练期间保持不变，函数更新由固定核控制{{< cite 4 >}}。

例如，在梯度流下，把训练集上的输出记为向量 $f_t$，便得到

$$
\frac{\mathrm d f_t}{\mathrm dt}
=-K_0\nabla_f\mathcal L(f_t).
$$

这让训练分析大幅简化，也明确了描述边界：核极限中的表示适应能力受到限制。有限宽网络会偏离这个极限；“网络足够宽”本身也不能替代对参数化、学习率与训练时间的说明。

### 均值场：让每个神经元继续改变特征

另一种两层网络尺度是

$$
f_n(x)=\frac{1}{n}
\sum_{i=1}^{n}a_i\phi(w_i^\top x).
$$

现在单个参数对输出的梯度带有 $1/n$。若在上述存储坐标中配合学习率

$$
\eta=n\eta_0,
$$

那么宽度因子会在更新中抵消。例如，对单个样本的损失 $\ell(f_n(x),y)$，

$$
\Delta w_i
=-\eta_0\,\partial_f\ell\,
a_i\phi'(w_i^\top x)x.
$$

当相关量保持有界、损失梯度非退化且 $\eta_0$ 适当时，单个神经元可以保持 $O(1)$ 的更新尺度，特征变化因而不会被宽度强制压到零。这允许非退化的特征学习，但并不保证每个神经元都发生非零更新：数据、激活与对称性仍会影响具体梯度。这里增长的是特定参数坐标下的学习率，不能把它直接当作任意实现的调参建议。

均值场方法把神经元群体写成经验分布

$$
\rho_n=\frac1n\sum_{i=1}^{n}\delta_{(a_i,w_i)},
\qquad
f_n(x)=\int a\phi(w^\top x)\,\mathrm d\rho_n(a,w),
$$

再研究这个分布随训练的演化。Mei 等人的工作说明，在适当缩放极限下，两层网络的 SGD 可以用分布动力学描述{{< cite 5 >}}。

还有一个容易混淆的区别：若初始化时各项独立且零均值，$1/n$ 平均的输出波动通常为 $O(n^{-1/2})$，所以初始输出可以趋近零[^mf-init]。这与每步训练产生非退化的输出和特征变化相容；稳定性并不要求初始输出一定具有非零方差。

因此，比较这个均值场例子与前面的 NTK 例子时，需要同时说明它们各自的初始函数分布与训练尺度。把一边的前向倍率和另一边的学习率拼接起来，通常不会保留原来的理论结论。

### 两种均值场：沿深度传播，还是沿时间学习？

“均值场”这个名字会让人误以为只有一条研究路线。前面介绍的是训练动力学：把一群神经元看成一个分布，再观察它怎样随优化过程移动。

Poole 等人的工作{{< cite 6 >}}和 Schoenholz 等人的 Deep Information Propagation{{< cite 7 >}}研究了另一类问题：随机初始化的信号经过一层又一层网络以后，尺度和输入之间的相关性会怎样变化？这里推进的主要是层数，关注的是信号传播，而非已经训练出的特征。

在基本的宽全连接网络中，把激活前数值的二阶矩记为 $q_\ell$，可以得到这样的递推：

$$
q_{\ell+1}
=\sigma_w^2\,
\mathbb E_{Z\sim\mathcal N(0,1)}
[\phi(\sqrt{q_\ell}Z)^2]
+\sigma_b^2.
$$

这里假设各层独立高斯初始化，权重方差为 $\sigma_w^2/n_\ell$，偏置方差为 $\sigma_b^2$，并在固定深度下取大宽度。要判断不同输入的信息是否被保留，还需要同时追踪两输入之间的协方差。单个 $q_\ell$ 稳定，只回答了信号尺度的一部分问题。

两种视角由此各有分工：信号传播理论沿网络深度追踪随机统计，训练动力学沿训练时间追踪参数分布。区分它们以后，Tensor Programs 的位置就更清楚了：它用更通用的计算语言分析大宽度极限，Tensor Programs I 已将初始化时的高斯过程对应扩展到更广泛的网络计算{{< cite 8 >}}。接下来才是如何在训练中保留特征学习的问题。


## 二、$\mu\mathrm{P}$：把稳定尺度和特征学习放在一起

### 从两层网络走向逐层分析

前面的两种极限已经说明，网络函数可以稳定地学习，而隐藏特征的变化可能消失；也可以通过另一套尺度约定，让单元在训练中继续改变表示。

到了深层网络，问题需要逐层回答：每层表示是否有界？更新对该层表示产生了多大作用？读出层能否把这些变化转成有限的输出更新？仅给整个网络套上一个倍率，通常不足以回答全部问题。

Yang 与 Hu 在 Tensor Programs IV 中研究了宽度相关参数化及其训练极限，提出最大更新参数化 $\mu\mathrm{P}${{< cite 9 >}}。这里的“最大”指在稳定性要求下，特征更新达到不随宽度消失的尺度。它并不要求最大化参数更新范数。

对于一层 $z=Wh$，固定输入时权重变化的直接作用是

$$
\Delta z=(\Delta W)h.
$$

训练产生的 $\Delta W$ 与输入相关。因此，只比较权重元素或整个矩阵的范数，不能代替比较同一输入上的特征变化。参数化要同时规定初始化、前向倍率与各类参数的更新尺度。

### 为什么不同层需要不同规则

输入层把固定维度映射到宽度维度，隐藏矩阵连接两个宽度维度，读出层则回到固定的输出维度。这三种位置在网络变宽时，面临的求和和更新关系不同。

以官方 `mup` 的 Adam 配置为例，隐藏矩阵按宽度倍率调整参数组学习率，读出层还配合前向缩放。迁移的是同一个基准学习率与配套规则，各参数组最终使用的数值学习率可以不同。具体代码与逐层推导留在后续实践稿中，先抓住这套规则需要一起工作的原因。

### 从参数化走向超参数迁移

Tensor Programs V 将这一思路用于 $\mu$Transfer{{< cite 10 >}}：在较小的模型上搜索超参数，再按照 $\mu\mathrm{P}$ 约定迁移到更宽的目标模型。

理论尺度与实验效果仍需分别检查。coordinate check 可以发现激活或更新随宽度异常增长的问题；多宽度的学习率扫描，才回答好的超参数区域是否相近。有限宽度下的曲线不必完全重合，也不能把尺度检查直接当成迁移成功的证明。


## 三、后续发展：宽度以外还有哪些问题

$\mu\mathrm{P}$ 的超参数迁移最初聚焦于宽度。后续研究逐渐把深度、模块结构、优化器、专家配置和训练预算纳入讨论。下面按这些问题梳理相关工作：每条路线改变了哪些缩放变量，又需要补上哪些条件。

### 谱条件：从逐元素尺度到矩阵的作用

Tensor Programs V 用配套的初始化与更新规则实现宽度方向的超参数迁移{{< cite 10 >}}。A Spectral Condition for Feature Learning 进一步用权重矩阵和更新矩阵的谱范数组织这些规则{{< cite 14 >}}，把参数尺度与特征学习联系起来。

谱范数衡量矩阵在输入方向上可能产生的最大放大。它让不同形状的矩阵有了统一的分析语言，也为后面讨论深度和优化器提供了入口。这里仍要区分：对矩阵作用的尺度约束，与有限模型中最优超参数是否一致，是不同层面的结论。

### 深度：参数化还要控制层与层的相互作用

Depth-$\mu\mathrm{P}$ 在 Tensor Programs VI 中研究先取无限宽、再取无限深的残差网络极限{{< cite 11 >}}。除了特征更新的大小，它还关注不同深度位置产生的特征是否足够多样。

这一分析有具体的结构条件。每个残差块只有一层计算时的结论，不能直接变成任意 Transformer 的深度迁移保证；对更复杂的残差块，论文也指出了所研究参数化中的限制。

CompleteP 进一步把问题落到深 Transformer 的计算效率上{{< cite 13 >}}：即使基准学习率能够跨深度迁移，某些参数化的残差块对块内参数的响应仍可能在大深度极限下趋于线性化。CompleteP 将避免这种惰性作为额外设计条件，结合深度方向的超参数迁移与更充分的特征学习。这里的块内线性化不等于整个网络进入固定 NTK 极限；能否迁移超参数与是否有效利用深度，需要分别检查。

Zheng 等人将谱条件扩展到宽度与深度的联合缩放，在残差块内变换数固定时，分别分析单变换和多变换分支{{< cite 15 >}}。CompleteP for RL 则在强化学习中考察这种参数化{{< cite 16 >}}：策略会改变采样分布，因此除了特征更新，还要考察训练中策略与数据分布的变化。文中的控制任务实验提供了相应证据，适用范围需要结合任务和网络结构判断。

### 模块与注意力：迁移规则需要细化到组件

一个 Transformer 包含注意力、前馈层、归一化和残差分支，它们未必适合共用同一组倍率。Completed Hyperparameter Transfer across Modules, Width, Depth, Batch and Duration 将逐模块调参纳入迁移框架{{< cite 17 >}}，研究学习率、AdamW 参数、权重衰减、初始化和残差乘子在宽度、深度、批量与训练时长变化时如何配合。逐模块迁移由文中的 Transformer 实验支持，批量与时长的规则还依赖随机微分方程（SDE）近似及所用调度。

GQA-$\mu\mathrm{P}$ 则针对分组查询注意力分析投影结构与参数缩放{{< cite 18 >}}，引入期望算子范数（expected operator norm），处理 KV 头重复形成的秩亏有效投影，并研究不同重复系数之间的学习率迁移。对新模块，首先需要说清哪些维度增长、哪些参数共享，以及更新如何作用于特征；仅按整个模型的隐藏宽度统一缩放，可能遗漏这些差异。

### 优化器：更新几何与有限宽度修正

更换优化器，也会改变宽度缩放问题。Muon is Scalable for LLM Training 通过权重衰减和更新尺度调整，将 Muon 用于大规模语言模型训练{{< cite 19 >}}。这一实践引出的问题是：经过矩阵预条件或正交化以后，更新应如何随矩阵形状变化？

Extending $\mu\mathrm{P}$ 用谱条件为多种优化器推导宽度缩放规则{{< cite 20 >}}；Xu 等人从矩阵算子范数下的优化几何出发，研究行列归一化与宽度迁移{{< cite 21 >}}。这些分析把优化器的更新方向、矩阵形状和学习率尺度放在一起考察，不能直接沿用普通 SGD 的坐标缩放公式。

即使渐近尺度匹配，有限宽度下的最优学习率仍可能漂移。Qiu 等人研究 Shampoo、SOAP 和 Muon 的学习率及权重衰减迁移，并分析分块与显式谱归一化对有限宽偏差的修正{{< cite 22 >}}。修正效果依赖优化器和块大小，并非叠加这些操作就一定更好。因此，优化器的具体实现也是迁移配方的一部分。

另一组工作关心训练过程中如何持续控制尺度。Muon++ 研究在学习率与谱间隙等条件下维持 $\mu\mathrm{P}$ 的谱条件{{< cite 23 >}}，Spectral Sphere Optimizer（SSO）则对隐藏层二维权重及其更新施加模块级谱范数约束{{< cite 24 >}}。Learnable Multipliers 让矩阵、行或列的倍率参与学习，以调整特征尺度{{< cite 25 >}}；这减少了前向及权重衰减乘子的手工调节需求，但不能替代逐模块学习率乘子的调优。

HyperP 围绕 Frobenius 球面上的 MuonH 优化，研究宽度、深度、token 预算与 MoE 粒度之间的迁移{{< cite 26 >}}。其宽深分析包含近似各向同性、小步长等条件；随 token 预算调整学习率的关系则来自特定模型与预算范围的经验拟合。

### MoE：专家规模与路由学习需要一起分析

MoE 增加了专家数、专家隐藏维度和每个 token 激活的专家数 top-$K$ 等缩放变量。早期的 $\mu$-Parametrization for Mixture of Experts 在固定专家数与 top-$K$、专家隐藏维度随模型宽度同比增长的设定下研究宽度迁移{{< cite 27 >}}。当专家粒度也变化时，这套结果不能直接照搬。

后续工作开始联合分析这些变量。Jiang 等人利用动态均值场理论研究模型宽度、深度、专家数及专家隐藏维度的扩展；其中增加专家数的分析保持激活专家数与总专家数的比例不变{{< cite 28 >}}。How to Scale Mixture-of-Experts 区分三种联合缩放区间，提出最大尺度稳定参数化 MSSP，并指出在这些设定中，最大更新条件本身不足以保证稳定的学习率迁移{{< cite 29 >}}。Complete-muE 则通过 dense FFN、全专家激活的 dense MoE 到 sparse MoE 的映射，研究从密集模型向稀疏模型的近最优超参数迁移{{< cite 30 >}}。

路由器还有独立的学习动力学。Why Routers Freeze 在固定专家数的无限宽极限下分析路由饱和、对称性与专家专门化{{< cite 31 >}}，并讨论趋于零的路由 logits 有限宽波动如何仍然决定 Top-$K$ 选择、打破专家间的对称性。这说明，专家激活保持稳定并不足以回答路由是否在有效学习。

### 训练预算：宽度迁移之后，还要校准训练时长

小模型训练多少 token，会影响超参数搜索得到的答案。How to Set the Learning Rate for Large-Scale Pre-training? 考察模型规模、深度、权重衰减和 token horizon 对最优学习率的影响，并比较拟合与迁移方法{{< cite 32 >}}。它揭示了所测迁移配方在大规模预训练中的局限，也提醒我们把训练预算纳入实验设计。

Scaling with Collapse 研究另一种诊断：在 $\mu\mathrm{P}$ 下保持学习率日程、每参数 token 数（TPP）和归一化 AdamW 时间尺度一致时，不同规模的训练损失曲线能否近似重合{{< cite 33 >}}。它将横轴换成训练完成比例，并用最终损失归一化纵轴，以支持训练诊断与调参早停。这种曲线重合有明确的预算和优化设置条件。

Let's Scale Step by Step 将模型规模迁移与 token-horizon 校准分步应用于采用 Muon 和多头潜在注意力（MLA）的 MoE{{< cite 34 >}}：在激活专家数和每个专家的隐藏维度固定时，联合扩大模型宽度和总专家数，再用经验规律外推更长的训练预算。**跨宽度迁移成功，并不自动证明同一组超参数适用于任意训练时长。** 对长程大模型，仍需验证这两步组合后的效果。

### 数值精度：尺度稳定还要适合硬件表示

u-$\mu\mathrm{P}$ 将最大更新参数化与 Unit Scaling 结合{{< cite 12 >}}，以权重、激活和梯度在初始化时接近单位尺度为目标，协调低精度训练和超参数迁移。这个目标依赖推导缩放因子时的统计假设；论文也观察到，注意力引入的相关性会使部分中间张量在初始化时就偏离单位尺度。因此，它并不保证所有张量从初始化到整个训练过程都严格保持单位方差。

这些工作逐步扩展了超参数迁移的适用对象。阅读或复现时，可以先固定要改变的变量，再检查对应的模块规则、优化器约定与训练预算，最后用尺度检查和超参数扫描验证迁移效果。


## 引用本文

{{< cite_article >}}


[^reading]: 问题组织受到 LawrenceC 的 [Maybe I was too harsh on deep learning theory (three days ago)](https://www.lesswrong.com/posts/6SRq7mZ97Dwuavwb6/maybe-i-was-too-harsh-on-deep-learning-theory-three-days-ago) 启发，尤其是对两种均值场路线的区分。本文的公式与技术结论分别链接到对应原始论文。

[^scales]: 文中的量级分析把数据、输入维度和相应训练时间约定固定。若这些量也随宽度变化，需要重新分析。上文重参数化的学习率推导针对数据损失的普通 SGD，额外正则项与 Adam 更新需要单独处理。

[^gp]: Matthews 等人的 [Gaussian Process Behaviour in Wide Deep Neural Networks](https://arxiv.org/abs/1804.11271) 也研究了深网络的高斯过程极限。不同工作采用的取极限方式与假设需要分别查看。

[^time]: 若训练时间、学习率或其他尺度也随宽度增长，结论可能变化。有限宽网络也可以表现出对 NTK 极限的偏离；这段分析不等于断言实际网络完全不学习特征。

[^mf-init]: 对独立零均值的初始化贡献，均值场的初始输出波动通常随宽度缩小；这与训练中的非退化特征变化相容。比较参数化时，应同时说明初始函数分布与后续学习动力学。
