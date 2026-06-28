# Netket_Sampler_631G.ipynb 失败分析

## 1. 运行结果对比

| Notebook | 最终能量 | FCI 基准 | 误差 |
|----------|---------|---------|------|
| Netket_631G.ipynb | -1.0192 Ha | -1.0261 Ha | 0.7% |
| Netket_Sampler_631G.ipynb | **-0.9461 Ha** | -1.0261 Ha | **7.8%** |

`Netket_Sampler_631G.ipynb` 收敛到的能量 **恰好等于 HF 基准能量** (`-0.94605220 Ha`)，说明神经网络完全无法超越 Hartree-Fock 近似。

---

## 2. 训练过程分析

```
Step   0 | E: 0.54212596 ± 0.018589 | Error: 1.568262
Step  50 | E: -0.94605227 ± 0.000000 | Error: 0.080083
Step 100 | E: -0.94605227 ± 0.000000 | Error: 0.080083
...（此后完全不变）
```

**关键观察：**
1. **标准差从 0.018 → 0.000**：所有样本计算出完全相同的局部能量
2. **能量停滞在 HF 能级**：神经网络拒绝探索 HF 态以外的相空间
3. **早期快速收敛**：50 步就"收敛"了，此后再无变化

---

## 3. 根本原因

### 3.1 采样器重置问题

```python
for step in range(N_ITER):
    sampler_state = sampler.reset(machine, params, sampler_state)  # ❌ 每步都重置！

    samples, sampler_state = sampler.sample(
        machine, params, state=sampler_state,
        chain_length=20  # ❌ 太少
    )
```

**问题：**
- 每步迭代都调用 `sampler.reset()`，这破坏了马尔可夫链的连续性
- `chain_length=20` 太小，样本自关联性极高
- 导致采样严重受阻，模型只能"看到"HF 态附近的局部能量景观

### 3.2 QGT 求解数值不稳定

```python
qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.1)
grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)

natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)  # ❌ 直接 solve
natural_grad = grad_unravel_fn(natural_grad)
grad = natural_grad
```

**问题：**
- 直接使用 `jnp.linalg.solve` 而非 SVD 分解，当 QGT 病态时数值不稳定
- NetKet 使用的是更稳健的 SVD-based 求解器
- 当采样只来自 HF 态附近时，QGT 接近奇异，自然梯度计算完全失效

### 3.3 复数梯度的共轭问题

```python
def weight_and_mean(grad_component):
    weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
    return jnp.mean(weights * jnp.conj(grad_component), axis=0)  # ❌ 可能符号错误
```

**问题：**
- 对于 **holomorphic** 网络，不应该使用 `jnp.conj()`
- NetKet 的 `forces_expect_hermitian` 对 holomorphic 函数直接使用 `weights * grad_component`
- 错误使用共轭会导致梯度方向完全错误

---

## 4. 为什么只能收敛到 HF 态？

### 4.1 物理图像

VMC 的目标是找到最小化能量的波函数：
```
E[ψ] = ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩
```

如果神经网络的参数初始化不好 + 采样受限 + 梯度错误，就会导致：

1. **初始化**: 网络可能从接近 HF 态的参数开始
2. **采样**: 采样器只能访问 HF 态附近的构型
3. **梯度**: 即使能量梯度指向 FCI 基态，错误QGT 和共轭会使自然梯度指向 HF 态
4. **收敛**: 网络"认为"HF 态就是能量最低点，停止优化

### 4.2 能量对比

| 态 | 能量 (Ha) |
|----|----------|
| HF | -0.9461 |
| FCI | -1.0261 |
| 差异 | 0.08 Ha (约 2.2 eV) |

这个差异正是相关能 (correlation energy)，是 VMC 必须捕获的目标。

---

## 5. Netket_631G.ipynb 为什么成功？

```python
gs = nk.driver.VMC(
    ha,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1, holomorphic=True),
)
gs.run(400)
```

**关键差异：**

1. **成熟的采样器**: NetKet 的 `MetropolisSampler` 正确实现了 proposal-accept 机制
2. **正确的 chain_length**: `n_samples=1008` 足够大
3. **正确的 QGT**: 使用 SR preconditioner，内部使用 SVD 分解
4. **正确的梯度**: NetKet 内部实现的 `forces_expect_hermitian` 对 holomorphic 函数不错误使用共轭
5. **不做 reset**: VMC driver 内部管理采样状态，不每步重置

---

## 6. 修复建议

### 方案 A：修复纯 JAX 实现（不推荐）

1. 移除每步的 `sampler.reset()`
2. 使用 `chain_length=100` 或更大
3. QGT 求解改用 SVD: `jnp.linalg.svd(qgt_reg, full_matrices=False)`
4. 移除 `jnp.conj()`，因为网络是 holomorphic 的

### 方案 B：使用 NetKet 驱动（推荐）

直接使用 `Netket_631G.ipynb` 的架构，它已经证明可以工作。

### 方案 C：调试 `compute_qgt` 函数

需要检查 `NES_VMC_H2_631G.py` 中 `compute_qgt` 的实现，确认：
- 是否使用了正确的 QGT 定义
- diag_shift 是否足够大
- 是否对复数参数有正确处理

---

## 7. 总结

| 问题 | 严重程度 | 影响 |
|------|---------|------|
| 每步 reset 采样器 | 🔴 致命 | 采样效率极低 |
| chain_length 太小 | 🔴 致命 | 样本自关联性高 |
| QGT 直接 solve | 🟠 严重 | 数值不稳定 |
| 错误使用共轭 | 🟠 严重 | 梯度方向错误 |

这些问题共同导致网络被困在 HF 态的局部最小值中，无法达到 FCI 基态。
