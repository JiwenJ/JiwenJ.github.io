---
title: "μP：前身、原理与后续发展"
mathTitle: '$\mu\mathrm{P}$：前身、原理与后续发展'
date: "2026-10-25T18:28:00+08:00"
author: ["Jiwen Jiang"]
description: "从无限宽理论走到最大更新参数化，再看深度、优化器、MoE 与训练预算下的超参数迁移。"
mathDescription: '从无限宽理论走到 $\mu\mathrm{P}$，再看深度、优化器、MoE 与训练预算下的超参数迁移。'
summary: "从初始化、NTK 与均值场出发，用一个可复现的 MLP 例子理解 μP，再看深度、优化器与 MoE 的迁移问题。"
mathSummary: '从初始化、NTK 与均值场出发，用一个可复现的 MLP 例子理解 $\mu\mathrm{P}$，再看深度、优化器与 MoE 的迁移问题。'
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
  - id: 35
    title: "mup：最大更新参数化的 PyTorch 实现"
    authors: "Microsoft Research"
    source: "原作者代码与文档"
    url: "https://github.com/microsoft/mup"
---

在小模型上找到一个好学习率之后，把网络加宽，为什么往往还要重新调参？宽度改变了每层参与求和的项数，也改变了梯度作用于特征的方式。初始化能让第一次前向传播保持稳定；要让训练过程也具有可比较的尺度，还需要同时安排各层的更新。

$\mu\mathrm{P}$（Maximal Update Parametrization，最大更新参数化）正是沿着这个问题发展起来的。本文分成三部分：先解释初始化、NTK 与均值场各自回答了什么问题；再用一个 MLP 走通 $\mu\mathrm{P}$ 的尺度规则和检查方法；最后讨论深度、优化器、MoE 与训练预算变化时，还需要补上哪些条件[^reading]。贯穿全文的问题是：**模型变宽以后，每一步还能否以相近的尺度改变表示？**

阅读需要一点反向传播和梯度下降的基础。想先动手的读者，可以从第二部分的 MLP 例子开始；公式推导和实现细节放在可展开的段落里。

## 一、$\mu\mathrm{P}$ 的前身

初始化、神经网络高斯过程（NNGP）、神经正切核（NTK）与均值场是彼此关联的研究视角。下面的阅读顺序从信号的起点走到训练中的特征变化；它们在历史上也有并行发展。这里固定输入维度与网络深度，让隐藏层宽度 $n$ 增长。$O(1)$ 表示量级不随宽度增长，而非数值恰好等于一[^scales]。

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

### 同一个 MLP，逐层回答三个问题

前面的理论留下了三个需要分开回答的问题：初始化的信号是否稳定？输出能否学习？隐藏特征是否也在改变？NTK 例子说明，第二个问题得到肯定答案，并不意味着第三个也是如此。

Yang 与 Hu 在 Tensor Programs IV 中研究了宽度相关参数化及其训练极限，提出最大更新参数化 $\mu\mathrm{P}${{< cite 9 >}}。这里的“最大”指在稳定性约束下，让各层特征保留不随宽度消失的更新尺度；它不要求把数值学习率或参数更新范数设得尽可能大。

下面始终使用同一个例子：两个隐藏层、标量输出的 ReLU MLP。

$$
\begin{aligned}
z_1&=W_1x, & h_1&=\operatorname{ReLU}(z_1),\\
z_2&=W_2h_1, & h_2&=\operatorname{ReLU}(z_2),\\
f(x)&=W_3h_2.
\end{aligned}
$$

其中 $W_1\in\mathbb R^{n\times d}$、$W_2\in\mathbb R^{n\times n}$、$W_3\in\mathbb R^{1\times n}$。固定输入维度 $d$、深度、批量和数据，只增加宽度 $n$。程序直接存储这三个有效权重，不额外添加前向倍率；暂不引入偏置、归一化或权重衰减。这样，表里的学习率就直接作用于公式里的 $W$。

### 关键一步：梯度更新会沿着输入方向累加

初始化时，我们用独立随机项的抵消来计算方差。训练时，同一个输入会同时出现在梯度和前向计算中，不能再照搬这个假设。

对单个样本的损失 $\ell(f(x),y)$，令 $\delta_2=\partial\ell/\partial z_2$。隐藏层梯度与 SGD 更新满足

$$
\nabla_{W_2}\ell=\delta_2h_1^\top,
\qquad
(\Delta W_2)h_1
=-\eta_2\delta_2\|h_1\|^2.
$$

右侧出现的是平方和 $\|h_1\|^2$，通常具有 $n$ 的量级，而不是随机符号求和的 $\sqrt n$ 量级。**梯度与输入的对齐，使许多很小的参数更新仍能对特征产生有限作用。**

这也解释了为什么输入层、隐藏层和读出层要分开处理。输入层只对固定的 $d$ 个输入求和；后两层对 $n$ 个特征求和；读出层还决定反传到隐藏层的梯度尺度。

<details class="technical-note">
<summary>展开推导：三层的 SGD 学习率为什么是 n、1、1/n？</summary>

取输入与隐藏层的初始化标准差分别为 $\sigma_1/\sqrt d$、$\sigma_2/\sqrt n$，读出层取 $\sigma_3/n$，其中 $\sigma_i$ 不随宽度变化。前两个隐藏表示的典型坐标保持常数量级。设损失对输出的导数 $\ell'$ 有界且非退化。

先看读出层。固定 $h_2$，它的更新对输出的作用为

$$
\Delta f_{\rm readout}
=-\eta_{\rm out}\ell'\|h_2\|^2
=O(\eta_{\rm out}n).
$$

因此应取 $\eta_{\rm out}\propto 1/n$，让这部分输出变化保持常数量级。

再看第二层。在上述随机初始化下，

$$
\delta_2=\ell'W_3^\top\odot\operatorname{ReLU}'(z_2)
$$

的典型坐标为 $O(1/n)$。把它代入前面的更新恒等式，$1/n$ 与 $\|h_1\|^2=O(n)$ 抵消，所以 $\eta_{\rm hidden}\propto 1$。

最后看输入层。在这一初始化和固定深度设定下，继续反传得到的 $\delta_1$ 的典型坐标也为 $O(1/n)$。但输入层的直接变化为

$$
(\Delta W_1)x=-\eta_{\rm in}\delta_1\|x\|^2.
$$

$d$ 固定，$\|x\|^2$ 不随 $n$ 增长，因此需要 $\eta_{\rm in}\propto n$。

这些计算解释了初始化附近的尺度选择。完整训练还会改变各层输入，并形成参数与特征之间的相关性。例如，联合更新隐藏层时有

$$
\Delta z_2=(\Delta W_2)h_1+W_2\Delta h_1
 +(\Delta W_2)\Delta h_1.
$$

单独分析第一项还不是对整个训练轨迹的证明；Tensor Programs 的作用，正是系统地处理这些相关性与训练极限{{< cite 9 10 >}}。

</details>

### 一张可以核对实现的尺度表

在上述有效权重坐标中，三类参数采用下面的宽度依赖关系{{< cite 10 >}}。表中的 $\sigma_i$ 与 $\eta_i$ 是不随宽度改变的常数，可以逐层选择；**表规定的是怎样随宽度缩放，常数本身仍要调。**

| 参数 | 初始化标准差 | SGD 学习率 | Adam 学习率 |
| :--- | :--- | :--- | :--- |
| 输入层 $W_1$ | $\sigma_1/\sqrt d$ | $\eta_1 n$ | $\eta_1$ |
| 隐藏层 $W_2$ | $\sigma_2/\sqrt n$ | $\eta_2$ | $\eta_2/n$ |
| 读出层 $W_3$ | $\sigma_3/n$ | $\eta_3/n$ | $\eta_3/n$ |

例如，宽度从 64 增加到 256：初始化标准差分别乘 $1,1/2,1/4$；SGD 学习率分别乘 $4,1,1/4$；Adam 学习率分别乘 $1,1/4,1/4$。这里比较的是同一套参数坐标下的两个模型。

Adam 与 SGD 不同，是因为它会按梯度的二阶矩归一化更新。在 $\epsilon$ 可忽略、梯度非退化的尺度分析下，归一化后的典型坐标更新为常数量级，再由学习率决定实际步幅[^adam]。因此，SGD 中用来补偿小梯度的输入层 $n$ 倍学习率，不能原样搬到 Adam。

读出层 $1/n$ 的初始化也意味着：独立零均值初始化时，初始输出通常会随宽度缩小。$\mu\mathrm{P}$ 允许这种情形；训练中的输出更新仍可保持常数量级。若使用全零读出初始化，第一步的隐藏层梯度会为零，检查曲线时要识别这个特例。

<details class="technical-note">
<summary>实现对照：为什么 MuAdam 的读出层学习率看起来没有除以宽度？</summary>

不同实现可以把同一个缩放因子放在不同位置。原作者公开的 `mup` 实现使用 `MuReadout`、`set_base_shapes` 和 `MuAdam` 配合工作{{< cite 35 >}}。

设相对基准模型的宽度倍率为 $r=n/n_0$。默认输出倍率下，`MuReadout` 存储参数 $V$，前向计算使用的有效权重是

$$
W_3=V/r.
$$

初始化经过 `set_base_shapes` 调整后，$V$ 的标准差相对基准模型保持不变，因此 $W_3$ 的标准差随宽度按 $1/r$ 缩小。`MuAdam` 对 $V$ 使用不随宽度变化的学习率，但 $\Delta W_3=\Delta V/r$，有效更新已经缩小了 $1/r$。隐藏矩阵有两个随宽度增长的维度，`MuAdam` 则直接将其参数组学习率除以 $r$。

因此，上表与这个实现的区别来自存储坐标。使用 `MuReadout` 与 `MuAdam` 时，再手工把读出层学习率除一次 $r$，会重复缩放。若使用普通 PyTorch 初始化，应先初始化模型，再调用 `set_base_shapes` 完成尺度调整，最后创建优化器；若在标记后重新初始化，应使用 `mup.init` 或显式遵守同一套尺度。改变注意力头数、参数共享方式或网络结构时，还要重新确认哪些维度随宽度增长。

</details>

### 跑一次实验：更新有没有随宽度漂移？

下面的实验直接实现前面的 MLP，用 NumPy 手算梯度。固定一批训练数据和一批探测输入，宽度取 32、64、128、256、512、1024，每个宽度运行 16 个随机种子。为了单独观察每层，计算一次 SGD 更新后，**固定该层输入特征，仅替换这一层权重**所产生的预激活变化：

$$
(\Delta W_1)x,\qquad
(\Delta W_2)h_1,\qquad
(\Delta W_3)h_2.
$$

纵轴是这些变化在探测批量上的 RMS，即所有坐标平方均值的平方根。这样不会把“维度更多”误当成“每个坐标变化更大”。本实验的 SP 基线对三层使用同一个、不随宽度变化的学习率；$\mu\mathrm{P}$ 则按上表逐层缩放。

<figure>
<picture>
<source media="(max-width: 600px)" srcset="/images/mup/coordinate-check-mobile.svg">
<img src="/images/mup/coordinate-check.svg" loading="lazy" decoding="async" alt="固定数据与 SGD 基准步长时，SP 的输入层直接更新随宽度缩小，隐藏层与读出层更新增大；μP 的三个面板大致水平。">
</picture>
</figure>

图中实线为 $\mu\mathrm{P}$，虚线为标准参数化 SP；线表示 16 个种子的中位数，阴影表示四分位区间。宽度从 32 增至 1024 时，SP 输入层的直接更新约缩至原来的 0.17 倍，隐藏层增至 4.30 倍，读出层增至 24.59 倍；$\mu\mathrm{P}$ 三者分别约为 1.06、0.99、0.88 倍。应关注各条曲线随宽度的趋势，而非要求不同层具有相同的绝对高度。

这揭示了统一调整学习率的困难：若为了压住 SP 的读出层更新而降低全局学习率，输入层已经很小的更新也会进一步缩小。$\mu\mathrm{P}$ 按层安排尺度，使这些趋势可以同时受控。

<details class="technical-note">
<summary>复现实验：设置、代码与数据</summary>

实验使用平方损失、输入维度 16、训练批量 32、探测批量 64，基准步长为 $10^{-3}$。输入为固定的高斯样本，目标由固定的非线性教师函数生成，并包含常数偏置，使初始批量的输出梯度不会因强制去均值而抵消。SP 与 $\mu\mathrm{P}$ 共享基础随机权重，前两层初始化相同；读出标准差分别为 $1/\sqrt n$ 与 $1/n$。二者的初始输出分布并不相同，所以这张图也不是性能优劣的比较。

这里为展示宽度幂次，将各层初始化常数设为 1，并令 SGD 表里的三个 $\eta_i$ 都等于同一个 `eta`。$\mu\mathrm{P}$ 的更新就是：

```python
# W1: (n, d), W2: (n, n), W3: (1, n)
# g1, g2, g3 are gradients of the same mean squared loss.
W1 -= eta * n * g1
W2 -= eta * g2
W3 -= eta / n * g3
```

SP 则对三层都使用 `eta`。完整脚本还包含中心有限差分检查，用来核对手写反向传播；本次检查的相对误差低于 $3\times10^{-10}$。

[下载完整 Python 脚本](/code/mup-coordinate-check.py) · [下载逐种子实验数据](/data/mup/coordinate-check.csv)。在装有 NumPy 与 Matplotlib 的环境中，进入下载目录运行：

```bash
python3 mup-coordinate-check.py --csv coordinate-check.csv --figure coordinate-check.svg
```

所有随机种子、数据生成方式与统计定义都保存在脚本中。

</details>

这是一个单步、分层的机制检查。实际 coordinate check 还应追踪多个训练步的激活，以及同一探测输入上联合更新前后的特征变化。数据导致的梯度抵消、有限宽度和小学习率都会影响曲线；一条看起来平的曲线，需要结合它测量的量来解释。

### 从参数化走向超参数迁移

Tensor Programs V 将这一思路用于 $\mu$Transfer{{< cite 10 >}}：在较小的模型上搜索超参数，再按照 $\mu\mathrm{P}$ 约定迁移到更宽的目标模型。

从上面的诊断走到实际迁移，还差一个问题：小模型上好的超参数，在目标宽度上是否仍然好？一个可执行的验证顺序是：

1. **先检查实现。** 固定数据、深度、优化器、批量和训练步数，只改变宽度，检查逐层激活、单层直接更新与联合更新的尺度。
2. **再扫描同一组基准学习率。** 用一个小宽度搜索，在一个较大的验证宽度复查；每个候选值都按同一套 $\mu\mathrm{P}$ 规则生成参数组学习率，保持其余训练与评估协议一致。
3. **比较好的区域是否重合。** 画出“基准学习率—验证损失”曲线，结合多个种子检查小模型选出的值在大模型上是否仍接近最优，而不是要求两条损失曲线完全重合。

coordinate check 回答尺度有没有明显出错，学习率扫描回答超参数有没有迁移。前者通过以后，后者仍需验证；上面的 MLP 图只完成了前者的一部分。

到这里，我们一直固定了深度、模块结构、优化器与预算。实际扩模往往会同时改变其中几项，第三部分讨论的正是这些额外变化。


## 三、后续发展：宽度以外还有哪些问题

$\mu\mathrm{P}$ 的超参数迁移最初聚焦于宽度。后续研究逐渐把深度、模块结构、优化器、专家配置和训练预算纳入讨论。下面按这些问题梳理相关工作：每条路线改变了哪些缩放变量，又需要补上哪些条件。

### 谱条件：从逐元素尺度到矩阵的作用

Tensor Programs V 用配套的初始化与更新规则实现宽度方向的超参数迁移{{< cite 10 >}}。A Spectral Condition for Feature Learning 进一步用权重矩阵和更新矩阵的谱范数组织这些规则{{< cite 14 >}}，把参数尺度与特征学习联系起来。

为什么换成谱范数？前面的 MLP 已经告诉我们，真正关心的是 $(\Delta W)h$。对于 $n_{\rm out}\times n_{\rm in}$ 的矩阵，欧氏范数不等式给出

$$
\operatorname{RMS}((\Delta W)h)
\leq \|\Delta W\|_{\rm op}
\sqrt{\frac{n_{\rm in}}{n_{\rm out}}}
\operatorname{RMS}(h).
$$

谱范数 $\|\cdot\|_{\rm op}$ 衡量矩阵对输入方向可能产生的最大放大，形状因子则把向量总范数换成坐标 RMS。这使不同形状的矩阵有了共同的分析语言。不过，上式只是上界；要让实际特征更新不消失，还要考察输入与更新方向的关系。有限模型中最优学习率是否一致，则仍是需要实验回答的问题。

### 深度：参数化还要控制层与层的相互作用

如果目标模型不仅更宽，也更深，只套用上一部分的宽度表就遗漏了一个累积效应：更多残差分支会共同改变表示。因此，深度迁移还需要控制各分支的尺度及其相互作用。

Depth-$\mu\mathrm{P}$ 在 Tensor Programs VI 中研究先取无限宽、再取无限深的残差网络极限{{< cite 11 >}}。除了特征更新的大小，它还关注不同深度位置产生的特征是否足够多样。

这一分析有具体的结构条件。每个残差块只有一层计算时的结论，不能直接变成任意 Transformer 的深度迁移保证；对更复杂的残差块，论文也指出了所研究参数化中的限制。

CompleteP 进一步把问题落到深 Transformer 的计算效率上{{< cite 13 >}}：即使基准学习率能够跨深度迁移，某些参数化的残差块对块内参数的响应仍可能在大深度极限下趋于线性化。CompleteP 将避免这种惰性作为额外设计条件，结合深度方向的超参数迁移与更充分的特征学习。这里的块内线性化不等于整个网络进入固定 NTK 极限；能否迁移超参数与是否有效利用深度，需要分别检查。

Zheng 等人将谱条件扩展到宽度与深度的联合缩放，在残差块内变换数固定时，分别分析单变换和多变换分支{{< cite 15 >}}。CompleteP for RL 则在强化学习中考察这种参数化{{< cite 16 >}}：策略会改变采样分布，因此除了特征更新，还要考察训练中策略与数据分布的变化。文中的控制任务实验提供了相应证据，适用范围需要结合任务和网络结构判断。

### 模块与注意力：迁移规则需要细化到组件

即使总宽度一样，改变注意力头数、KV 共享方式或模块倍率，也会改变参数对特征的作用。一个 Transformer 包含注意力、前馈层、归一化和残差分支，它们未必适合共用同一组倍率。

Mlodozeniec 等人将逐模块调参纳入迁移框架{{< cite 17 >}}，研究学习率、AdamW 参数、权重衰减、初始化和残差乘子在宽度、深度、批量与训练时长变化时如何配合。逐模块迁移由文中的 Transformer 实验支持，批量与时长的规则还依赖随机微分方程（SDE）近似及所用调度。

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

小模型训练多少 token，会影响超参数搜索得到的答案。Zhou 等人考察模型规模、深度、权重衰减和训练 token 总量（token horizon）对最优学习率的影响，并比较拟合与迁移方法{{< cite 32 >}}。它揭示了所测迁移配方在大规模预训练中的局限，也提醒我们把训练预算纳入实验设计。

Scaling with Collapse 研究另一种诊断：在 $\mu\mathrm{P}$ 下保持学习率日程、每参数 token 数（TPP）和归一化 AdamW 时间尺度一致时，不同规模的训练损失曲线能否近似重合{{< cite 33 >}}。它将横轴换成训练完成比例，并用最终损失归一化纵轴，以支持训练诊断与调参早停。这种曲线重合有明确的预算和优化设置条件。

Let's Scale Step by Step 将模型规模迁移与 token-horizon 校准分步应用于采用 Muon 和多头潜在注意力（MLA）的 MoE{{< cite 34 >}}：在激活专家数和每个专家的隐藏维度固定时，联合扩大模型宽度和总专家数，再用经验规律外推更长的训练预算。**跨宽度迁移成功，并不自动证明同一组超参数适用于任意训练时长。** 对长程大模型，仍需验证这两步组合后的效果。

### 数值精度：尺度稳定还要适合硬件表示

u-$\mu\mathrm{P}$ 将最大更新参数化与 Unit Scaling 结合{{< cite 12 >}}，以权重、激活和梯度在初始化时接近单位尺度为目标，协调低精度训练和超参数迁移。这个目标依赖推导缩放因子时的统计假设；论文也观察到，注意力引入的相关性会使部分中间张量在初始化时就偏离单位尺度。因此，它并不保证所有张量从初始化到整个训练过程都严格保持单位方差。

这些方向不必在第一次使用 $\mu\mathrm{P}$ 时全部叠加。先明确自己准备改变什么，才能找到相应的工作：

| 扩模时改变的对象 | 应重新检查的关系 |
| :--- | :--- |
| 只增加隐藏宽度 | 初始化、各参数组学习率与特征更新 |
| 同时增加深度 | 残差分支尺度、块内学习与层间累积 |
| 更换注意力或模块结构 | 增长的维度、参数共享与模块倍率 |
| 更换优化器 | 预条件后的更新几何、矩阵形状与有限宽修正 |
| 改变 MoE 配置 | 总专家数、top-$K$、专家维度与路由学习 |
| 延长训练预算 | 学习率日程、优化器时间尺度与目标 token horizon |

回到开头的小模型调参问题：$\mu\mathrm{P}$ 提供了一套让不同宽度模型更可比较的训练坐标。它的价值在于把“扩大模型以后重新猜学习率”变成可以分析、检查和验证的迁移过程。真正使用时，从固定结构和预算的宽度实验开始，先检查特征尺度，再比较最优学习率区域；每增加一个缩放变量，就补上对应的分析与验证。


## 引用本文

{{< cite_article >}}


[^reading]: 问题组织受到 LawrenceC 的 [Maybe I was too harsh on deep learning theory (three days ago)](https://www.lesswrong.com/posts/6SRq7mZ97Dwuavwb6/maybe-i-was-too-harsh-on-deep-learning-theory-three-days-ago) 启发，尤其是对两种均值场路线的区分。本文的公式与技术结论分别链接到对应原始论文。

[^scales]: 文中的量级分析把数据、输入维度和相应训练时间约定固定。若这些量也随宽度变化，需要重新分析。上文重参数化的学习率推导针对数据损失的普通 SGD，额外正则项与 Adam 更新需要单独处理。

[^gp]: Matthews 等人的 [Gaussian Process Behaviour in Wide Deep Neural Networks](https://arxiv.org/abs/1804.11271) 也研究了深网络的高斯过程极限。不同工作采用的取极限方式与假设需要分别查看。

[^time]: 若训练时间、学习率或其他尺度也随宽度增长，结论可能变化。有限宽网络也可以表现出对 NTK 极限的偏离；这段分析不等于断言实际网络完全不学习特征。

[^mf-init]: 对独立零均值的初始化贡献，均值场的初始输出波动通常随宽度缩小；这与训练中的非退化特征变化相容。比较参数化时，应同时说明初始函数分布与后续学习动力学。

[^adam]: 这里指不含权重衰减的 Adam 尺度分析，假设 $\epsilon$ 相对二阶矩平方根足够小，或随参数坐标一致调整。若 $w=s\theta$ 且 $s>0$，梯度归一化理想情形下有 $\eta_w=s\eta_\theta$，而 SGD 是 $\eta_w=s^2\eta_\theta$。保留 $\epsilon$ 时，还需匹配 $\epsilon_\theta=s\epsilon_w$。实际 `MuAdam` 不会自动替用户缩放 $\epsilon$；AdamW 的解耦权重衰减也需要另行匹配。
