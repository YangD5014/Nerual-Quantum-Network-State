# NES-VMC 自然梯度下降实验分析报告

## 1. 实验对比

### 1.1 纯梯度下降 vs 自然梯度下降

| 指标 | 纯梯度下降 | 自然梯度下降 |
|------|------------|--------------|
| Loss 初始值 | -1.307 | -1.307 |
| Loss 最终值 | **-1.397** | -1.341 |
| 梯度收敛性 | 良好 | 较差 |
| 训练时间 | 21.96 秒 | 285.27 秒 |
| 时间比 | 1x | **~13x** |

### 1.2 关键观察

**纯梯度下降（正常收敛）：**
```
Step   0 | Loss: -1.307 | E0: -1.056 | E1: -0.251
Step   5 | Loss: -1.340 | E0: -1.029 | E1: -0.311
Step  10 | Loss: -1.355 | E0: -1.027 | E1: -0.329
Step  15 | Loss: -1.367 | E0: -1.056 | E1: -0.311
Step  19 | Loss: -1.397 | E0: -1.105 | E1: -0.292  ← Loss 持续下降
```

**自然梯度下降（收敛受阻）：**
```
Step   0 | Loss: -1.307 | E0: -1.056 | E1: -0.251
Step   5 | Loss: -1.317 | E0: -1.055 | E1: -0.262
Step  10 | Loss: -1.324 | E0: -1.047 | E1: -0.278
Step  15 | Loss: -1.328 | E0: -1.045 | E1: -0.283
Step  19 | Loss: -1.341 | E0: -1.047 | E1: -0.294  ← Loss 下降缓慢
```

## 2. 问题分析

### 2.1 QGT 计算的时间开销

`compute_nes_qgt` 函数的实现：
```python
def compute_nes_qgt(total_machine, params, samples, diag_shift=0.01):
    # 1. 单样本梯度 - 使用 Python 循环，无法向量化
    def _single_grad(x):
        return jax.grad(lambda p: total_machine(p, x), holomorphic=True)(params)

    # 2. 逐样本计算梯度
    grads = [_single_grad(x) for x in samples]  # ← O(N_samples) 次前向传播
```

**问题**：
- 每个样本需要一次 `jax.grad` 调用
- 对于 32 链 × 200 样本 = 6400 样本，每次迭代需要 6400 次梯度计算
- 每次梯度计算都是完整的前向+反向传播

### 2.2 QGT 梯度计算的物理问题

**标准 VMC 的 QGT**：
对于基态 VMC，QGT 衡量的是波函数参数空间的几何结构：
$$S_{ij} = \langle\partial_i\psi|\partial_j\psi\rangle - \langle\partial_i\psi|\psi\rangle\langle\partial_j\psi|\psi\rangle$$

**NES-VMC 的特殊性**：
NES-VMC 的损失函数是：
$$\mathcal{L} = \mathrm{Tr}(E_L) = \mathrm{Tr}(\Psi^{-1}H\Psi)$$

对应的梯度是：
$$\nabla_\theta \mathcal{L} = 2\langle\mathrm{Tr}(E_L - \bar{E}_L)\nabla_\theta \log\Psi\rangle$$

**问题所在**：
当前 QGT 计算的是 `total_machine` 的梯度，即 $\nabla \log\Psi$ 的几何张量，但这与损失函数 $\mathcal{L}$ 的梯度不匹配。

自然梯度应该是：
$$\text{natural\_grad} = S^{-1} \nabla_\theta \mathcal{L}$$

但这里的 $S$ 应该是损失函数在参数空间的几何张量，而不是单纯的波函数几何张量。

### 2.3 梯度幅度异常

观察梯度值：

| Step | 纯梯度 grad[30] | 自然梯度 grad[30] |
|------|------------------|-------------------|
| 0 | 0.005+0.088j | 0.005+0.088j |
| 5 | -0.018+0.021j | 0.001+0.104j |
| 10 | -0.011+0.004j | 0.007+0.113j |
| 15 | -0.015-0.007j | 0.010+0.129j |
| 19 | **-0.037-0.012j** | **0.015+0.149j** |

**观察**：
1. 纯梯度方向一致地向极小值移动
2. 自然梯度的实部在第 5 步后变为正值，方向异常
3. 自然梯度幅值约为纯梯度的 1/2.5，明显被衰减

## 3. 根本原因

### 3.1 计算效率问题

`compute_nes_qgt` 使用 Python `for` 循环遍历样本：
```python
grads = [_single_grad(x) for x in samples]  # 非向量化
```

应该使用 `jax.vmap` 进行向量化：
```python
grads = jax.vmap(lambda x: jax.grad(lambda p: total_machine(p, x), holomorphic=True)(params))(samples)
```

### 3.2 物理公式问题

NES-VMC 的自然梯度应该考虑损失函数的完整形式。当前的 QGT 只考虑了波函数的对数梯度几何，但没有考虑：

1. **局域能量矩阵 $E_L$ 的结构**：$E_L = \Psi^{-1}H\Psi$ 是 $K \times K$ 矩阵
2. **Trace 操作的梯度权重**：损失函数是 $\mathrm{Tr}(E_L)$，其梯度涉及矩阵求导
3. **Batch 统计量**：需要正确处理多样本的均值和协方差

### 3.3 QGT 正则化问题

QGT 矩阵的条件数影响自然梯度的稳定性：
- 正则化参数 `diag_shift` 过大：梯度被过度阻尼
- 正则化参数过小：矩阵可能接近奇异

## 4. 结论

| 问题 | 原因 | 影响 |
|------|------|------|
| 训练时间过长 | Python 循环 + 逐样本梯度计算 | ~13x 计算开销 |
| 收敛速度变慢 | QGT 没有正确反映损失函数的几何结构 | 无法加速收敛 |
| 梯度方向异常 | 自然梯度计算公式不完整 | 可能发散 |

## 5. 改进建议

### 5.1 向量化 QGT 计算
```python
def compute_nes_qgt_vectorized(machine, params, samples, diag_shift=0.01):
    # 使用 vmap 向量化
    vmap_grad = jax.vmap(
        lambda x: jax.grad(lambda p: machine(p, x), holomorphic=True)(params)
    )
    grads = vmap_grad(samples)
    # ... 其余计算相同
```

### 5.2 正确计算 NES 自然梯度

NES-VMC 的自然梯度需要考虑损失函数的完整形式：
1. 计算 $\nabla_\theta \log\Psi$ 的 QGT
2. 但要用损失函数的梯度 $\nabla_\theta \mathcal{L}$ 而不是波函数的梯度

### 5.3 使用更小的 batch 逐步调试

当前配置：
- N_CHAINS = 32
- N_SAMPLES_PER_CHAIN = 200
- 总样本数 = 6400

建议先用小样本调试 QGT 计算的正确性，再逐步增大。

## 6. 参考文献

- NES-VMC 原始论文
- NetKet 文档: QGT and natural gradients
- Flax NNX 文档: https://flax.readthedocs.io

---

**结论**：自然梯度下降在 NES-VMC 中没有体现出优势，主要原因是 QGT 计算效率低（Python 循环）和物理公式不完整（没有考虑损失函数中 Trace 和矩阵结构的影响）。建议先使用纯梯度下降，待超参数调优完成后再尝试自然梯度。