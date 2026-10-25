---
title: "μP 的前身：从标准参数化到特征学习"
mathTitle: '$\mu\mathrm{P}$ 的前身：从标准参数化到特征学习'
date: "2026-10-25T18:28:00+08:00"
author: ["Jiwen Jiang"]
description: "先理解初始化、神经正切核与均值场，再进入 μP。"
mathDescription: '先理解初始化、神经正切核与均值场，再进入 $\mu\mathrm{P}$。'
summary: "从参数化的含义出发，梳理 Xavier/He、NNGP、NTK 与均值场如何为 μP 的特征学习视角铺路。"
mathSummary: '从参数化的含义出发，梳理 Xavier/He、NNGP、NTK 与均值场如何为 $\mu\mathrm{P}$ 的特征学习视角铺路。'
tags: ["μP", "Initialization", "NTK", "Mean Field"]
categories: ["Deep Learning"]
draft: true
math: true
isCJKLanguage: true
ShowToc: true
TocOpen: false
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
  - id: 6
    title: "Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks"
    authors: "Greg Yang, Edward J. Hu"
    source: "ICML 2021"
    url: "https://proceedings.mlr.press/v139/yang21c.html"
---

小模型上找到的学习率，为什么到了大模型上就可能失效？宽度增加以后，前向信号、梯度和参数更新会一起改变。要理解 $\mu\mathrm{P}$ 为什么有机会让不同宽度的模型共享调参经验，可以按照这样的阅读顺序展开：**参数化、稳定初始化、高斯过程、NTK，再到均值场特征学习。**

这些研究路线在历史上彼此交织。这里的顺序服务于理解：先看计算与更新怎样随宽度变化，再看不同理论各自保留了哪些现象，而不是把它们排成一条依次取代前人的发展链。

这里的“宽”指隐藏层的神经元数量。下文用固定输入维度、固定深度的两层网络说明尺度关系；$O(1)$ 表示量级不随宽度增长，并不表示数值恰好等于一。

## 标准参数化：同一个函数，可以有不同的训练方式

先从常见的标准参数化（Standard Parametrization，SP）出发：程序直接存储各层权重，以 fan-in 一类规则初始化，再按约定的学习率训练。做宽度比较时，还必须说明这个学习率是否随宽度变化。

参数化要同时交代四件事：**实际存储的参数、前向计算中的倍率、初始化分布、优化器的学习率。** 只知道某个权重的标准差，还不足以确定它会怎样训练。

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

这个推导限定在普通 SGD。Adam 会利用梯度历史归一化更新，不能直接套用上面的 $s^2$ 规则。讨论宽度缩放之前，必须先固定参数坐标和优化器约定。

## Xavier 与 He：先让信号顺利通过网络

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

## 高斯过程：描述随机初始化的函数分布

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

由此得到的高斯过程视角，把随机网络看成一个函数先验。其协方差由初始化和激活函数决定；Lee 等人的工作将这一对应关系系统地扩展到深层网络{{< cite 3 >}}。

这里关注的是初始化时 $f_0$ 的随机分布。它可以用于相应的贝叶斯推断，但不能单凭这个对应关系，就断言梯度下降中的隐藏特征是否变化。要描述训练，还需要研究函数对参数的敏感度。

## NTK：输出能学习，隐藏特征却可以近乎不动

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

因此，在学习率 $\eta=O(1)$ 的普通 SGD 下，固定索引的典型单元有单步 $\Delta w_i=O(n^{-1/2})$。在不随宽度增长的有限训练时间内，并满足相应正则性与稳定性条件时，这类单元的特征变化可以随宽度增大而消失。这里并未声称所有神经元在任意长训练过程中，都满足同一个严格界。

输出仍然可以发生 $O(1)$ 变化。原因是大量微小更新共同作用于输出：单元变化小，并不等于网络函数不学习。NTK 理论刻画的一类极限中，初始化核收敛到确定性极限，并在训练期间保持不变，函数更新由固定核控制{{< cite 4 >}}。

例如，在梯度流下，把训练集上的输出记为向量 $f_t$，便得到

$$
\frac{\mathrm d f_t}{\mathrm dt}
=-K_0\nabla_f\mathcal L(f_t).
$$

这让训练分析大幅简化，也明确了描述边界：核极限中的表示适应能力受到限制。有限宽网络会偏离这个极限；“网络足够宽”本身也不能替代对参数化、学习率与训练时间的说明。

## 均值场：让每个神经元继续改变特征

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

还有一个容易混淆的区别：若初始化时各项独立且零均值，$1/n$ 平均的输出波动通常为 $O(n^{-1/2})$，所以初始输出可以趋近零。这与每步训练产生非退化的输出和特征变化相容；稳定性并不要求初始输出一定具有非零方差。

因此，比较这个均值场例子与前面的 NTK 例子时，需要同时说明它们各自的初始函数分布与训练尺度。把一边的前向倍率和另一边的学习率拼接起来，通常不会保留原来的理论结论。

## $\mu\mathrm{P}$ 接着回答什么

到这里，可以分开看三件事：初始化控制信号的起点；NTK 展示固定核如何解释一类训练极限；均值场展示另一种尺度选择如何保留特征学习。它们帮助我们看见，**同样让宽度趋于无穷，得到什么训练行为，取决于整套参数化。**

对深层网络，问题还要逐层回答：隐藏表示是否有界？一次训练带来的表示变化是否消失？读出层怎样把这些变化转成有限的输出更新？仅调整两层网络求和前的一个倍率，还不足以自动处理所有层。

Yang 与 Hu 在 Tensor Programs IV 中系统研究了宽度相关参数化及其训练极限，提出能够保留特征学习的 $\mu\mathrm{P}${{< cite 6 >}}。它追求在尺度稳定的约束下，让特征更新达到不会随宽度消失的量级；“最大”需要结合这个约束理解。

这为宽度迁移提供了理论出发点：先使不同宽度的模型具有可比较的训练尺度，再研究小模型上的超参数经验能否迁移。具体迁移规则仍依赖架构、优化器和参数坐标，也需要有限宽度实验检验。均值场仍是理解特征学习的重要视角，而宽度方向的结论也不能无条件延伸到深度、序列长度或训练预算。

## 参考资料

{{< references >}}
