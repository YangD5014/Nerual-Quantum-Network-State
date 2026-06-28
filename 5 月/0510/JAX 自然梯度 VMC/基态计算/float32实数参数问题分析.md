# Float32实数参数导致VMC训练失效的原因分析

## 问题描述

在你的H₂基态计算案例中，当将神经网络Ansatz（`SingleStateAnsatz`）的参数从复数（`complex`）改为float32实数后，优化无法收敛到接近FCI基准能量的结果。

## 根本原因

### 1. 量子态必须用复数表示

量子力学中，波函数 $\psi(\sigma)$ 是一个**复数函数**：
$$\psi(\sigma) = |\psi(\sigma)|e^{i\theta(\sigma)}$$

复数表示包含两个自由度：
- **振幅** $|\psi(\sigma)|$：决定测量到状态 $\sigma$ 的概率 $P(\sigma) = |\psi(\sigma)|^2$
- **相位** $\theta(\sigma)$：决定量子态之间的干涉效应，是量子计算的核心

当强制使用实数参数时，相位信息 $\theta(\sigma)$ **完全丢失**，波函数退化为 $\psi(\sigma) = |\psi(\sigma)|$，只能描述实数振幅。

### 2. 局部能量计算的物理矛盾

在你的代码中，局部能量定义为：

```python
O_loc = compute_local_energies(machine, params, sigma)
# 其中 ha.get_conn_padded(sigma) 返回的是复数矩阵元 H_σ→η
```

物理公式：
$$E_{\text{loc}}(\sigma) = \sum_{\eta} H_{\sigma\to\eta} \frac{\psi(\eta)}{\psi(\sigma)}$$

其中：
- $H_{\sigma\to\eta}$ 是复数（来自电子哈密顿量的复数矩阵元）
- $\psi(\sigma)$ 当参数为实数时是实数

**问题**：分子 $\psi(\eta)/\psi(\sigma)$ 是实数比值，但分母 $H_{\sigma\to\eta}$ 是复数。这导致局部能量变成复数，而物理上局部能量应该是实数。计算出的复数局部能量取均值后，虚部无法消除，造成能量期望值的系统性偏差。

### 3. 自然梯度（SR）计算的数学不兼容

在你的 `compute_qgt` 函数中，量子几何张量定义为：

$$S_{ij} = \langle \partial_i \log \psi^* \partial_j \log \psi \rangle - \langle \partial_i \log \psi^* \rangle \langle \partial_j \log \psi \rangle$$

其中使用了**共轭**操作 `jnp.conj()`。这要求：
- $\psi$ 是复数可微（holomorphic）的函数
- 梯度 $\nabla \log \psi$ 是复数梯度

当你使用实数参数时：
- `jax.grad(..., holomorphic=True)` 的行为与复数情况完全不同
- `jnp.conj()` 对实数的共轭等于自身，但语义上不兼容holomorphic梯度的定义
- QGT矩阵的结构发生变化，可能不再是正定的，导致自然梯度解不稳定

### 4. force-based梯度计算的问题

在你的 `forces_expect_hermitian` 函数中：

```python
grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)

def weight_and_mean(grad_component):
    weights = O_centered.reshape(...)
    return jnp.mean(weights * jnp.conj(grad_component), axis=0)
```

当使用实数参数时：
- `grad_component` 是实数梯度
- `jnp.conj(grad_component) = grad_component`（共轭等于自身）
- 但物理上要求的 $\nabla \log \psi^*$（共轭梯度）与 $\nabla \log \psi$ 的关系在实数域不成立

### 5. 复数流形 vs 实数流形

优化器实际上是在参数流形上进行梯度下降：

| 方面 | 复数参数 | 实数参数 |
|------|---------|---------|
| 参数空间维度 | $2N$（实部+虚部） | $N$ |
| 梯度流形 | 复数流形 $\mathbb{C}^N$ | 实数流形 $\mathbb{R}^N$ |
| 能描述的波函数 | 振幅+相位 | 仅振幅 |
| 量子态表达能力 | 完整 | 严重受限 |

复数参数空间允许波函数取任意相位，这是描述量子叠加和纠缠的必要条件。

## 为什么最终能量偏差大

1. **能量期望值不准确**：复数局部能量取实部后仍有系统偏差
2. **梯度方向错误**：自然梯度计算基于复数共轭结构，实数参数下方向不正确
3. **优化目标本质不同**：复数参数优化的是复数波函数，实数参数优化的是实数振幅（忽略相位）
4. **无法收敛到正确基态**：H₂基态是一个有非零相位的量子态，实数神经网络无法精确表达

## 结论

**量子变分蒙特卡洛（VMC）要求复数波函数表示**，这是由量子力学的本质决定的。强行使用实数参数会导致：
- 物理方程的不兼容（复数矩阵元 vs 实数波函数）
- 数学结构的不兼容（holomorphic梯度 vs 普通实数梯度）
- 表达能力的不兼容（振幅+相位 vs 仅振幅）

这是原理层面的限制，而非实现细节问题。解决方案是保持参数为 `complex` 类型（如你代码中 `param_dtype=complex` 所做的那样）。
