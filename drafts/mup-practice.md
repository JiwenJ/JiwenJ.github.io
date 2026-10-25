---
title: "μP：从小模型调参到大模型迁移"
mathTitle: '$\mu\mathrm{P}$：从小模型调参到大模型迁移'
date: "2026-10-25T18:28:00+08:00"
author: ["Jiwen Jiang"]
description: "从特征更新的尺度出发，理解 μP 的初始化、学习率与超参数迁移。"
mathDescription: '从特征更新的尺度出发，理解 $\mu\mathrm{P}$ 的初始化、学习率与超参数迁移。'
summary: "模型变宽以后，为什么同一个学习率可能失效？从一个 MLP 例子理解 μP，并用 coordinate check 检查实现。"
mathSummary: '模型变宽以后，为什么同一个学习率可能失效？从一个 MLP 例子理解 $\mu\mathrm{P}$，并用 coordinate check 检查实现。'
tags: ["μP", "Optimization", "Scaling"]
categories: ["Deep Learning"]
draft: true
math: true
ShowToc: true
TocOpen: false
references:
  - id: 1
    title: "Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks"
    authors: "Greg Yang, Edward J. Hu"
    source: "ICML 2021"
    url: "https://proceedings.mlr.press/v139/yang21c.html"
  - id: 2
    title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer"
    authors: "Greg Yang et al."
    source: "arXiv:2203.03466, 2022"
    url: "https://arxiv.org/abs/2203.03466"
  - id: 3
    title: "mup：用法、base shapes 与 coordinate check"
    authors: "Microsoft"
    source: "官方代码文档"
    url: "https://github.com/microsoft/mup/blob/main/README.md"
  - id: 4
    title: "MuReadout 实现"
    authors: "Microsoft"
    source: "mup/layer.py"
    url: "https://github.com/microsoft/mup/blob/main/mup/layer.py"
  - id: 5
    title: "MuAdam 与 MuSGD 实现"
    authors: "Microsoft"
    source: "mup/optim.py"
    url: "https://github.com/microsoft/mup/blob/main/mup/optim.py"
  - id: 6
    title: "Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks"
    authors: "Greg Yang, Dingli Yu, Chen Zhu, Soufiane Hayou"
    source: "arXiv:2310.02244, 2023"
    url: "https://arxiv.org/abs/2310.02244"
---

训练大模型时，一个常见的问题是：小模型上调好的学习率，放到更宽的模型上，还能继续用吗？

如果只把隐藏维度放大，其余设置照搬，答案往往不确定。模型宽度会同时改变前向计算、反向梯度和一次优化更新的尺度。**$\mu\mathrm{P}$（Maximal Update Parametrization）**{{< cite 1 >}} 将初始化、学习率和前向倍率视为一套随宽度变化的规则，使小模型调参更有参考价值。

本文先讨论固定深度、只改变隐藏宽度的一族网络。更深的网络、不同训练预算和新的优化器，需要分别检查。

## 1. 初始化稳定，只解决了第一步

考虑一个隐藏层的线性变换，暂时省略偏置：

$$
z_i = \sum_{j=1}^{n} W_{ij}h_j.
$$

在初始化时，假设权重独立、均值为零，与输入独立，且 $\mathbb{E}[h_j^2]=q$。取 $\operatorname{Var}(W_{ij})=\sigma_w^2/n$，就有

$$
\operatorname{Var}(z_i)
=\sum_{j=1}^{n}\frac{\sigma_w^2}{n}\mathbb{E}[h_j^2]
=\sigma_w^2q.
$$

这说明，合适的 fan-in 初始化可以让前向信号的尺度不随宽度发散。但训练还会改变权重。固定输入 $h$，仅考察这一层权重变化的直接作用：

$$
\Delta z_i = \sum_{j=1}^{n}\Delta W_{ij}h_j.
$$

这里的 $\Delta W$ 是由梯度产生的，与激活相关。不能直接沿用初始化时的独立性假设，把这 $n$ 项当成随机噪声求和。**前向方差稳定，并不自动意味着训练中的特征更新稳定。**

## 2. $\mu\mathrm{P}$ 关心的是特征如何变化

$\mu\mathrm{P}$ 不只规定权重如何初始化，还规定不同参数的学习率以及前向计算中的倍率。它希望网络变宽以后，在适当条件下，各层仍有非退化的特征学习，同时保持训练信号的尺度稳定{{< cite 1 >}}。

这里的 “maximal” 说的是满足稳定性约束时的更新尺度。它不表示学习率越大越好，也不表示直接最大化参数更新的范数。

一个容易混淆的地方是比较对象。对于 $n\times n$ 的矩阵，元素数量随 $n^2$ 增长。即使每个元素的更新变小，整个矩阵的 Frobenius 范数仍可能变大。因此，检查参数的总范数，不能代替检查固定输入上的激活和特征变化。

更有用的比较量包括每层激活的均方根：

$$
\operatorname{RMS}(h)=\sqrt{\frac{1}{N}\sum_{k=1}^{N}h_k^2},
$$

以及同一批输入上，更新前后的 $\operatorname{RMS}(h_{t+1}-h_t)$。这里 $N$ 是参与统计的元素总数；比较不同宽度时，使用相同的数据、训练步骤和统计定义。

## 3. 一个具体例子：Adam 下的 MLP

先固定输入维度、输出维度和深度，只改变两个隐藏层的宽度。设基准宽度为 $n_0$，目标宽度为 $n$，宽度倍率为

$$
m=\frac{n}{n_0}.
$$

下面的规则对应 `microsoft/mup`{{< cite 3 >}} 中的 `MuAdam`{{< cite 5 >}} 和无偏置 `MuReadout`{{< cite 4 >}} 实现。它们是一套配合使用的参数表示，不能只取其中一条套到任意模型上。

| 部件 | 张量形状 | 随宽度变化的轴数 | `MuAdam` 中的学习率 |
| --- | --- | --- | --- |
| 输入层 | $n\times d_{\mathrm{in}}$ | 1 | $\eta_0$ |
| 隐藏层 | $n\times n$ | 2 | $\eta_0/m$ |
| 输出层的存储参数 | $d_{\mathrm{out}}\times n$ | 1 | $\eta_0$ |

这里 $d_{\mathrm{in}}$ 与 $d_{\mathrm{out}}$ 固定。输入层和输出层虽然也是二维矩阵，却只有一个随宽度增长的轴；不能看到二维张量就统一除以 $m$。

输出层还包含前向倍率。对于本文使用的无偏置读出层，代码等价于

$$
f(h)=\frac{\alpha}{m}W_{\mathrm{out}}h,
$$

其中 $\alpha$ 是 `output_mult`。`set_base_shapes` 也会调整读出层的初始化。输出层的存储参数保持基准学习率，不等于它对最终输出的作用与宽度无关；前向倍率同样参与了尺度控制。

因此，迁移的是基准超参数 $\eta_0$ 及整套参数化规则。各参数组最终使用的数值学习率可以不同。

## 4. 用最小代码看清实现顺序

下面的例子使用 PyTorch 和 `mup`，在合成数据上执行一步训练。它用于说明接口和参数组，不是模型性能实验。

```python
import torch
from torch import nn
from mup import MuAdam, MuReadout, set_base_shapes


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.input = nn.Linear(32, width, bias=False)
        self.hidden = nn.Linear(width, width, bias=False)
        self.output = MuReadout(width, 10, bias=False)

    def forward(self, x):
        h = torch.relu(self.input(x))
        h = torch.relu(self.hidden(h))
        return self.output(h)


torch.manual_seed(0)
base = MLP(64)
delta = MLP(128)
model = MLP(256)

# base 和 delta 只提供形状信息，不需要训练。
set_base_shapes(model, base, delta=delta)
optimizer = MuAdam(model.parameters(), lr=1e-3)

# 输出隐藏层与其他参数实际使用的学习率。
names = {id(p): name for name, p in model.named_parameters()}
for group in optimizer.param_groups:
    print(group["lr"], [names[id(p)] for p in group["params"]])

x = torch.randn(16, 32)
y = torch.randint(0, 10, (16,))
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.cross_entropy(model(x), y)
loss.backward()
optimizer.step()
```

这里 $m=4$。按上述实现，隐藏层参数组的学习率为 `2.5e-4`，输入层和读出层为 `1e-3`。这些数值来自缩放规则，不是对最优学习率的推荐。

顺序很重要：先构造模型，再设置 base shapes，然后创建优化器。若要自定义初始化，应遵循 `mup` 的初始化接口和约定。恢复已经训练的 checkpoint 时，也不能再次按新模型的方式重缩放参数；官方接口提供了 `rescale_params=False`{{< cite 3 >}}。

## 5. 先检查尺度，再讨论超参数迁移

$\mu$Transfer{{< cite 2 >}} 的实践思路是在较小的模型上搜索超参数，再将基准超参数按 $\mu\mathrm{P}$ 规则迁移到更宽的目标模型。在投入大模型训练前，我会把验证分成两步。

**第一步是 coordinate check。** 选取若干宽度，在相同数据和训练步骤上，记录输入层、隐藏层与读出层的激活统计。既看初始化，也看训练数步后的变化。若这些量随宽度系统性发散，先排查参数化实现，而不是继续扩大搜索预算。

**第二步才是迁移实验。** 在多个宽度上使用相同的基准学习率网格，比较验证损失随学习率变化的曲线，并用不同随机种子检查最优区域是否稳定。

| 检查 | 能回答的问题 | 不能直接推出的结论 |
| --- | --- | --- |
| 激活与更新的尺度检查 | 实现是否出现明显的宽度依赖异常 | 已找到最优学习率 |
| 多宽度学习率扫描 | 好的学习率区域是否相近 | 任意更大模型都必然迁移成功 |
| 目标模型训练验证 | 当前目标配置下的迁移效果 | 新架构、新预算也有同样效果 |

coordinate check 是诊断工具。它通过了，仍需训练实验来判断迁移是否有效；有限宽度下，也不应要求不同模型的每条曲线完全重合。读出层在初始化时随宽度缩小，也可能符合设计预期，需要按各层对应的理论尺度解读。

## 6. 用到 Transformer 时，先说明怎么变宽

Transformer 的“宽度”不只有一个数：`d_model`、每头维度、头数和 FFN 中间维度，都可能变化。不同的变化方式会影响哪些参数轴应当被标记为宽度轴。

例如，固定每头维度、增加头数，与固定头数、增加每头维度，并不是同一个模型族。涉及 attention logits 的缩放时，要先说清楚变化的是哪个维度，再检查对应的参数化{{< cite 3 >}}。

实践中还需要单独核对 embedding 与读出层是否共享权重、归一化参数如何处理，以及优化器、梯度裁剪和学习率调度是否保留了参数组之间的比例。MLP 中的一张缩放表，不能直接替代完整 Transformer 的实现检查。

## 7. 本文的边界

本文围绕固定深度下的宽度变化展开。它不把宽度迁移直接等同于深度、batch size、序列长度或训练 token 预算的迁移。原始 $\mu$Transfer 工作也讨论了部分跨设置的经验结果{{< cite 2 >}}，但这些结果需要结合具体实验条件理解；深度方向另有专门的理论研究{{< cite 6 >}}。

理解 $\mu\mathrm{P}$ 时，值得始终跟踪三个量：**前向信号的尺度、一次更新引起的特征变化，以及不同宽度下好的超参数区域。** 三者各自经过检查，小模型调参才能成为有依据的决策工具。

## 参考资料

{{< references >}}
