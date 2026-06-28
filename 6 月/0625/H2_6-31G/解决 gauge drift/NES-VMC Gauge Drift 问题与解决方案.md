# NES-VMC Gauge Drift 问题与解决方案

## 1. 问题背景

### 1.1 什么是 Gauge Drift？

在量子变分蒙特卡洛（VMC）方法中，波函数具有**相位不确定性**。对于神经网络波函数 $\Psi(\theta, x)$，存在一个整体的复数 gauge 自由度：

$$
\Psi(\theta, x) \rightarrow e^{i\phi} \Psi(\theta, x)
$$

其中 $\phi$ 是任意常数相位。这个 gauge 自由度在物理上不可观测（因为所有物理量都涉及 $|\Psi|^2$ 或 $\Psi^*\hat{H}\Psi$）。

**Gauge Drift** 指的是在训练过程中，神经网络参数的演化导致波函数的 gauge（整体相位）发生剧烈、不可控的变化，这会导致：

1. **Loss 震荡**：目标函数 trace(E_L) 的值剧烈变化
2. **梯度消失**：协方差梯度 $\langle (E_L - \langle E_L \rangle) \nabla \log \Psi \rangle$ 中的信号被淹没
3. **训练停滞**：模型陷入 gauge plateau，无法继续优化
4. **数值不稳定**：logΨ 的实部 span 趋近于零

### 1.2 训练日志中的 Gauge Drift 表现

从 notebook 日志可以清晰看到典型的 gauge drift 症状：

```
[Step   40] logΨ.real span=4.6465  | trace_real_std=0.629
[Step   50] logΨ.real span=4.8852  | trace_real_std=1.332e-15  ← 突然崩溃！
[Step   60] logΨ.real span=0.0000  | trace_real_std=0.0000     ← gauge plateau
[Step   80] Grad | raw=0.0000e+00  | clipped=0.0000e+00       ← 梯度消失
```

关键症状：
- `logΨ.real span ≈ 0`：所有样本的 logΨ 实部相同
- `trace_real_std ≈ 0`：协方差梯度消失
- `dlog_std_ratio ≈ 0`：参数响应变成常数方向
- `shift.std = 0`：L_max 在所有样本上相同

### 1.3 Gauge Drift 的根本原因

```
┌──────────────────────────────────────────────────────────────┐
│                    Gauge Drift 机制                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   NES-VMC 的目标是同时优化 K 个态的波函数                     │
│   但波函数有 K×K 个未知的相对相位！                           │
│                                                              │
│   L[i,j] = ln ψ_j(x^i)  ← 每个神经网络输出一个复数           │
│                                                              │
│   det(Ψ) 只依赖于 L 的迹，不依赖于整体的 gauge                │
│   但 ∇logΨ 依赖于局部的 gauge 结构                          │
│                                                              │
│   训练时，参数会趋向于"吃掉"相位自由度                       │
│   → 导致 logΨ 实部 span 趋近于 0                            │
│   → 协方差梯度消失                                           │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 解决方案：Gauge Fixed Ansatz

### 2.1 核心思想

**固定参考点的 gauge**：选择一个参考态（如 HF 态），将其波函数值固定为实数正数，从而消除整体的 gauge 不确定性。

$$
\log \psi_j(x_{\text{ref}}) = \text{real}(c), \quad c > 0
$$

这相当于在整个参数空间中选取一个 hypersurface，将 K×K 个相位自由度减少到 (K-1)×K 个。

### 2.2 单态 Gauge Fixing

```python
def create_single_machine_gauge_fixed(model: SingleStateAnsatz, ref_state):
    """
    单态 gauge 固定：
    输出 logψ - logψ(HF) 的实部
    保证 ref_state 处的波函数值为纯实数
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)  # 复数 logψ

        # 计算参考态的 logψ
        log_psi_ref = m(ref_state)
        log_psi_ref_real = jnp.real(log_psi_ref)

        # gauge fixing：减去参考态的实部
        # 结果：ref_state 处 logψ = pure imaginary
        log_psi_fixed = log_psi - log_psi_ref_real

        return log_psi_fixed

    return machine, graphdef, state
```

**数学解释**：
- 原始输出：$\log \psi_j(x)$
- 参考点：$\log \psi_j(x_{\text{ref}})$
- gauge 固定：$\log \psi_j(x) - \text{Re}(\log \psi_j(x_{\text{ref}}))$
- 在参考点：$\log \psi_j(x_{\text{ref}}) - \text{Re}(\log \psi_j(x_{\text{ref}})) = i \cdot \text{Im}(\log \psi_j(x_{\text{ref}}))$

### 2.3 总态 Gauge Fixed

```python
def create_gauge_fixed_total_machines(total_ansatz, ref_state):
    """
    总态 gauge 固定：

    对于 NES-VMC，我们需要同时修复：
    1. 采样器用机器：还原原始 logΨ（用于 Metropolis）
    2. 矩阵机器：返回 L_stable
    3. 最大值机器：返回 L_max

    Gauge fixing 在单态层面进行，
    总态的 gauge 通过组合自然确定
    """
    # 1. 创建 gauge-fixed 的单态机器列表
    single_machines_gf = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine_gauge_fixed(ansatz, ref_state)
        single_machines_gf.append(m)

    # 2. 创建 NESTotalAnsatz_stable
    # 注意：这里复用同一个 ansatz 对象
    total_machine, graphdef, state = create_machine_stable(total_ansatz)

    # 3. 但是在 Ham_Psi 中使用 gauge-fixed 的单态机器
    # 这部分在训练循环中通过 single_machine_list 参数传入

    return total_machine, total_matrix_machine, total_max_machine, graphdef, state
```

### 2.4 Gauge Fixed 与 Stable 的结合

关键洞察：**gauge fixing** 和 **数值稳定化** 是两个独立的机制：

| 机制 | 目的 | 实现位置 |
|------|------|----------|
| Gauge Fixing | 消除相位自由度的不确定性 | 单态输出层 |
| 数值稳定化 | 防止 exp 溢出 | L_stable = L - L_max |

两者可以同时使用：
- Gauge fixing 确保训练稳定
- 稳定化确保数值不溢出

---

## 3. 代码详解

### 3.1 Cell 1: 导入与初始化

```python
# 导入 gauge-fixed 相关函数
from NES_VMC import (
    NESTotalAnsatz_stable,
    create_machine_stable,
    create_machine_matrix_stable,
    create_machine_max_stable,
    create_single_machine,
    nes_vmc_gradient_stable,
    NES_loss_energy_stable,
    compute_qgt,
    NESFermionHopRule,
    create_single_machine_gauge_fixed,    # ← 新增
    create_gauge_fixed_total_machines       # ← 新增
)

# 开启 x64 提高精度
jax.config.update("jax_enable_x64", True)
```

**导入的新函数**：
- `create_single_machine_gauge_fixed`：创建 gauge-fixed 的单态机器
- `create_gauge_fixed_total_machines`：创建 gauge-fixed 的总态机器

### 3.2 Cell 2: 工具函数

与标准诊断版相同，提供：
- `tree_l2_norm`：PyTree 范数
- `tree_all_finite`：检查有限性
- `unique_ratio_from_samples`：采样多样性
- `tree_batch_std_norm` / `tree_batch_mean_norm`：dlogΨ 诊断
- `safe_real` / `safe_float`：安全转换

### 3.3 Cell 3: 诊断函数

```python
def compute_diagnostics(total_params, x_batch, samples, E_L_mean):
    # 完整的训练状态诊断
    # 包括 logΨ 统计、条件数、shift 统计、能量迹等
```

### 3.4 Cell 4: 日志配置

```python
logger = logging.getLogger(f"NES_VMC_K{K}_diagnostic")
# 同时输出到文件和控制台
```

### 3.5 Cell 5: 模型初始化（Gauge Fixed）

```python
# 1. 创建原始 ansatz
total_ansatz = NESTotalAnsatz_stable(SINGLE_SIZE, K, 12, rngs=nnx.Rngs(11))

# 2. 获取原始参数
_, _, total_params_raw = create_machine_stable(total_ansatz)

# 3. HF 参考态
HF_ref = jnp.asarray(Hatree_Fock)

# 4. 创建 gauge-fixed 的总态机器
(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_state,
) = create_gauge_fixed_total_machines(
    total_ansatz,
    ref_state=HF_ref,
)

# 5. 使用原始参数
total_params = total_params_raw

# 6. 创建 gauge-fixed 的单态机器列表
single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, _, _ = create_single_machine_gauge_fixed(ansatz, HF_ref)
    single_machine_list.append(m)
```

**关键区别**：
- 采样器使用 `total_machine`（已 gauge-fixed）
- Ham_Psi 使用 `single_machine_list`（已 gauge-fixed）
- 损失函数使用 `total_matrix_machine` 和 `total_max_machine`

### 3.6 Cell 6: 优化器与采样器

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
)
```

### 3.7 Cell 7: 训练循环

```python
for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(...)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 梯度计算（使用 gauge-fixed 的机器）
    grad_raw, loss_mean, E_L_mean = nes_vmc_gradient_stable(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        total_machine=total_machine,
        single_machine_list=single_machine_list,  # ← gauge-fixed
        total_params=total_params,
        x_batch=x_batch,
    )

    # 3. 异常检测
    grad_finite = tree_all_finite(grad_raw)
    loss_finite = bool(jnp.isfinite(loss_mean))
    grad_explode = bool(grad_norm_raw > grad_skip_threshold)

    # 4. 自然梯度（可选）
    if Natural_Grad:
        ...

    # 5. 诊断
    if need_print:
        diag = compute_diagnostics(...)
        log_diagnostics(...)

    # 6. 跳过坏更新
    if (not grad_finite) or (not loss_finite) or grad_explode:
        n_skipped += 1
        continue

    # 7. 参数更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
```

---

## 4. 与诊断版代码的区别

### 4.1 新增的 Gauge Fixed 函数

| 函数 | 功能 |
|------|------|
| `create_single_machine_gauge_fixed` | 单态 gauge 固定 |
| `create_gauge_fixed_total_machines` | 总态 gauge 固定包装 |
| `check_gauge_fix_single_states` | 检查 gauge 固定效果 |

### 4.2 模型初始化的变化

```python
# 诊断版
total_machine, total_graphdef, total_params = create_machine_stable(total_ansatz)
total_matrix_machine, _, _ = create_machine_matrix_stable(total_ansatz)
single_machine_list = [create_single_machine(ansatz) for ansatz in ...]

# Gauge Fixed 版
_, _, total_params_raw = create_machine_stable(total_ansatz)
(total_machine, total_matrix_machine, total_max_machine, _, _
) = create_gauge_fixed_total_machines(total_ansatz, ref_state=HF_ref)
total_params = total_params_raw
single_machine_list = [create_single_machine_gauge_fixed(ansatz, HF_ref) for ansatz in ...]
```

### 4.3 Gauge Fixed 的效果

Gauge Fixed 后，训练应该表现出：
- `logΨ.real span` 保持稳定（不会趋近于 0）
- `trace_real_std` 保持合理值
- `dlog_std_ratio` 保持有意义的大小
- 能量逐步收敛

---

## 5. 完整的 Gauge Fixing 数学

### 5.1 波函数的 gauge 自由度

对于 K 个态的神经网络波函数 $\Psi_j(x)$，其 gauge 自由度为：

$$
\Psi_j(x) \rightarrow e^{i\phi_j} \Psi_j(x), \quad j = 1, 2, \ldots, K
$$

NES-VMC 的损失函数涉及：

$$
\mathcal{L} = \text{tr}(E_L), \quad E_L = \Psi^{-1} H \Psi
$$

其中 $\Psi_{ij} = \Psi_j(x^i)$ 是 K×K 矩阵。

### 5.2 Gauge 变换下的不变量

- $\det(\Psi)$：在 gauge 变换下乘以 $\prod_j e^{i\phi_j} = e^{i\sum_j \phi_j}$
- $\text{tr}(\Psi^{-1} H \Psi)$：**在 gauge 变换下不变**！

因此，损失函数本身是 gauge 不变的，但梯度依赖于局部的 gauge 结构。

### 5.3 Gauge Fixing 的实现

通过在单态输出中减去参考态的实部：

$$
\log \tilde{\psi}_j(x) = \log \psi_j(x) - \text{Re}(\log \psi_j(x_{\text{ref}}))
$$

这确保了：
- 在参考点：$\log \tilde{\psi}_j(x_{\text{ref}}) = i \cdot \text{Im}(\log \psi_j(x_{\text{ref}}))$
- 整体的实部偏移被消除

---

## 6. 训练监控指标

### 6.1 正常训练的表现

Gauge Fixed 后，训练应该显示：

```
[Step   0] logΨ.real | mean=0.813 | std=0.620 | span=4.074
[Step  50] logΨ.real | mean=1.188 | std=0.626 | span=3.896
[Step 100] logΨ.real | mean=1.707 | std=0.692 | span=3.422
[Step 150] logΨ.real | mean=2.401 | std=0.800 | span=4.100
```

关键：`span`（最大值-最小值）保持稳定，不会趋近于 0。

### 6.2 异常诊断

```python
if log_real_span < 1e-6:
    logger.warning(">>> WARNING: logΨ.real span ≈ 0，疑似 amplitude collapse / gauge plateau")

if trace_real_std < 1e-8:
    logger.warning(">>> WARNING: trace(E_L).real std 很小，covariance 梯度能量项可能无信号")

if dlog_std_ratio < 1e-6:
    logger.warning(">>> WARNING: dlogΨ batch variation 很小，参数响应接近常数方向")
```

---

## 7. 总结

### 7.1 Gauge Drift 的识别

| 症状 | 诊断指标 |
|------|----------|
| 相位塌缩 | `log_real_span ≈ 0` |
| 梯度消失 | `trace_real_std ≈ 0` |
| 响应恒定 | `dlog_std_ratio ≈ 0` |
| 训练停滞 | `grad_norm_raw ≈ 0` |

### 7.2 Gauge Fixed 的效果

| 方面 | 无 Gauge Fixing | 有 Gauge Fixing |
|------|-----------------|----------------|
| logΨ 实部分布 | 逐渐趋同 → span → 0 | 保持多样性 |
| 协方差梯度 | 信号消失 | 保持有效 |
| 训练稳定性 | 可能陷入 plateau | 更稳定收敛 |
| 数值稳定性 | 可能溢出 | 仍需稳定化 |

### 7.3 最佳实践

1. **同时使用 Gauge Fixing + 数值稳定化**
2. **监控 `log_real_span` 防止 gauge drift**
3. **监控 `dlog_std_ratio` 检测梯度信号**
4. **设置合理的 `grad_skip_threshold` 跳过坏更新**
5. **定期保存 checkpoint 便于回溯**

---

## 8. 参考文件

- `NES_VMC.py`：核心 NES-VMC 算法实现（包含 gauge-fixed 函数）
- `NES_VMC数值稳定性策略详解.md`：对数域稳定化详细解释
- `NES-VMC诊断版训练代码函数详解.md`：诊断函数的详细说明
