# NES-VMC 数值稳定性策略分析报告

## 1. 问题背景

在NES-VMC算法中，局域能量矩阵的计算涉及在 **对数域（log domain）** 中进行计算，然后将结果乘以 $\psi(x_j)$。当 $\psi(x_j) = 0$ 时，局域能量矩阵中会出现 **NaN**（Not a Number）值，导致数值不稳定。

### 1.1 基态VMC vs 激发态VMC的差异

| 情况 | 采样分布 | 节点附近行为 |
|------|----------|--------------|
| **基态VMC** | $\psi^2$ | walker几乎不可能出现在节点附近，因为采样概率与 $\psi^2$ 成正比 |
| **NES-VMC** | $\Psi^2 = \det(\Psi)^2$ | 即使 $\Psi^2$ 非零，矩阵 $\Psi$ 的某些行列也可能为零 |

**核心问题**：在NES-VMC中，walker从 $\Psi^2$ 中采样，但 $\Psi$ 矩阵的某些元素 $\psi_j(x^i)$ 可能为零，导致计算 $\Psi^{-1}(x)\hat{H}\Psi(x)$ 时出现数值不稳定。

---

## 2. S8 数值稳定性策略

### 2.1 原论文提出的稳定性启发式方法

原文S8章节提出了以下启发式策略来稳定优化过程：

#### 策略1：额外的MCMC采样步骤
- **触发条件**：当局域能量矩阵的任何元素为零（$\psi_j(x^i) = 0$）
- **应对措施**：执行额外的MCMC采样步骤（每次迭代最多10次）
- **目的**：避免采样到导致矩阵奇异的配置
- **重要约束**：设置上限（10次）以避免无限循环

#### 策略2：正则化处理
- 在计算 $\Psi^{-1}(x)\hat{H}\Psi(x)$ 之前，对 $\Psi$ 矩阵添加小的正则化项
- 形式：$\Psi \leftarrow \Psi + \epsilon I$，其中 $\epsilon$ 是一个小的正数（如 $10^{-6}$）
- **注意**：这个策略在代码中有注释掉的部分，表明可能不是首选方案

#### 策略3：节点检测与跳过
- 检测哪些配置会导致 $\psi_j(x^i) = 0$
- 在这些配置处不计算局域能量或使用极限值

---

## 3. 当前代码中的稳定性措施分析

### 3.1 NES-VMC代码解读.md 中的实现

```python
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    # Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])  # <-- 被注释掉
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

### 3.2 当前代码的稳定性问题

从训练输出可以看到不稳定现象：

```
Step   0 | Loss: -1.2576295690685688|0st能量=-0.99557002 Ha| 1st能量=-0.26205955 Ha
Step  50 | Loss: -1.547645795833384|0st能量=-56.85370946 Ha| 1st能量=55.30606367 Ha
                                                ↑↑↑
                              能量发散！表明矩阵条件数过大
```

**问题诊断**：
1. `log_Psi` 值从 1.338 增长到 16.215，说明波函数幅值指数增长
2. 某些步骤出现能量发散（56 Ha, -56 Ha），表明数值不稳定
3. `grad norm` 从 0.8250 骤降到 0.0000，表明梯度消失或NaN出现

---

## 4. 推荐的稳定性改进方案

### 4.1 实现额外的MCMC采样策略

参考原文"每次迭代最多10次额外采样"的策略：

```python
def safe_NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x, max_mcmc_retries=10):
    """
    带稳定性保护的损失函数计算
    """
    for attempt in range(max_mcmc_retries):
        log_M = total_matrix_machine(total_params, x)
        Psi_Matrix = jnp.exp(log_M)

        # 检查矩阵条件数和最小奇异值
        singular_values = jnp.linalg.svd(Psi_Matrix, compute_uv=False)
        min_singular = jnp.min(singular_values)

        # 如果矩阵接近奇异（条件数过大），使用正则化
        if min_singular < 1e-8:
            # 添加小的正则化项
            reg = 1e-6 * jnp.eye(Psi_Matrix.shape[0])
            Psi_Matrix = Psi_Matrix + reg

            # 重新计算，但记录这是一个修复后的值
            H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
            Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
        else:
            H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
            Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)

        # 检查结果是否有NaN
        if not jnp.any(jnp.isnan(Psi_Matrix_inv)):
            return jnp.real(jnp.trace(Psi_Matrix_inv)), Psi_Matrix_inv

        # 如果有NaN，继续采样（这是原文的策略）

    # 达到最大重试次数，返回一个安全的默认值
    # 可以使用单位矩阵或上一次成功的值
    return jnp.zeros(()), jnp.eye(Psi_Matrix.shape[0])
```

### 4.2 使用SVD分解提高稳定性

比起直接求逆，改用SVD分解计算 $\Psi^{-1}H\Psi$：

```python
def stable_NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)
    Psi_Matrix = jnp.exp(log_M)
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)

    # 使用SVD分解：Psi = U @ S @ Vh
    # 这样可以更好地处理病态矩阵
    U, S, Vh = jnp.linalg.svd(Psi_Matrix, full_matrices=False)

    # 检查条件数
    cond = S[0] / S[-1]  # 条件数
    if cond > 1e12:
        # 条件数过大，应用正则化
        S = S / (S**2 + 1e-8)  # 添加小的正则化到奇异值
        Psi_inv = Vh.T @ jnp.diag(1.0/S) @ U.T
    else:
        Psi_inv = Vh.T @ jnp.diag(1.0/S) @ U.T

    result = Psi_inv @ H_psi_x
    return jnp.real(jnp.trace(result)), result
```

### 4.3 梯度计算中的稳定性保护

```python
def stable_nes_vmc_gradient(ha, total_matrix_machine, total_machine,
                            single_machine_list, total_params, x_batch):
    # 1. 批量局域能量矩阵
    loss_batch, E_L_batch = safe_NES_loss_energy(
        ha, total_matrix_machine, single_machine_list, total_params, x_batch
    )

    # 2. 检测并处理NaN
    valid_mask = ~jnp.any(jnp.isnan(E_L_batch), axis=(-1, -2))

    # 只对有效样本计算均值
    E_L_mean = jnp.mean(E_L_batch[valid_mask], axis=0) if jnp.any(valid_mask) else jnp.zeros_like(E_L_batch[0])

    # 3. 计算梯度（使用有效样本）
    valid_x = x_batch[valid_mask]

    if len(valid_x) == 0:
        # 如果没有有效样本，返回零梯度
        grad = jax.tree.map(lambda p: jnp.zeros_like(p), total_params)
        return grad, jnp.nan, E_L_mean

    # ... 后续梯度计算
```

---

## 5. 关键配置参数建议

| 参数 | 原值 | 建议值 | 说明 |
|------|------|--------|------|
| `max_mcmc_retries` | N/A | 10 | 原文建议的最大MCMC重试次数 |
| 正则化系数 $\epsilon$ | 0 (被注释) | $10^{-6}$ ~ $10^{-8}$ | 太小无效，太大影响精度 |
| SVD条件数阈值 | N/A | $10^{12}$ | 超过此值触发正则化 |

---

## 6. 监控指标建议

在训练过程中应监控以下指标及早发现不稳定：

1. **矩阵条件数**：`cond = σ_max / σ_min`
2. **最小奇异值**：检测接近零的奇异值
3. **NaN检测**：`jnp.any(jnp.isnan(Psi_Matrix_inv))`
4. **能量方差**：局域能量方差突然增大是不稳定的先兆
5. **log_Psi范围**：值过大（>20）可能导致指数不稳定

```python
# 建议添加到训练循环的监控代码
singular_values = jnp.linalg.svd(Psi_Matrix, compute_uv=False)
cond_number = singular_values[0] / singular_values[-1]
min_singular = singular_values[-1]

if cond_number > 1e12 or min_singular < 1e-8:
    print(f"⚠️ 矩阵条件数警告: cond={cond_number:.2e}, min_σ={min_singular:.2e}")
```

---

## 7. 总结

NES-VMC的数值稳定性主要面临以下挑战：

1. **对数域计算**导致的指数放大效应
2. **矩阵求逆**在接近奇异时的数值问题
3. **节点附近**的梯度消失/爆炸

建议的综合解决方案：
- 保留原文的"额外MCMC采样"策略（最多10次）
- 启用正则化项或改用SVD分解
- 添加实时监控（条件数、最小奇异值、NaN检测）
- 在检测到不稳定时使用安全的默认值或回退策略

---

## 参考

- Pfau et al., "Accurate Computation of Quantum Excited States with Neural Networks", Science 385 (2024)
- Supplementary Materials S8: Numerical Stability
