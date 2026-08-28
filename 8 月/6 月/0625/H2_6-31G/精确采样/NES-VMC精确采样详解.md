# NES-VMC 精确采样（Exact Enumeration）详解

## 1. 背景与动机

### 1.1 为什么需要精确采样？

在标准的 NES-VMC 训练中，我们使用 **Metropolis-Hastings 采样器** 从 $|\Psi_\theta|^2$ 分布中生成样本。然而，当采样过程出现问题时（如采样塌缩、提案不足、bad-state），很难判断问题根源是：

- **采样器本身**的缺陷
- **NES 核心计算**（E_L 矩阵、梯度）的缺陷
- **Ansatz 结构**的问题

### 1.2 精确采样的解决方案

精确采样（Exact Enumeration）方案的核心思想：

> **完全抛弃 Metropolis 采样器，枚举所有合法的 NES 构型**

这样可以：
1. **消除采样器**作为变量，专注于诊断 NES 核心计算
2. **精确计算** $|\Psi_\theta|^2$ 权重下的期望值
3. **提供 ground truth** 用于验证采样版的结果

---

## 2. 核心公式

### 2.1 枚举合法 NES 构型

NES 要求 K 个副本的子组态互不相同：

$$
X = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_K), \quad \mathbf{x}_i \in \{0,1\}^{n_{\text{spin}}}
$$

$$
\forall i \neq j: \mathbf{x}_i \neq \mathbf{x}_j
$$

对于 H₂ / 6-31G 系统：
- 单系统 Hilbert 维度：16 个 determinant
- K=3 时，合法 ordered NES samples 数量：

$$
N_{\text{valid}} = 16 \times 15 \times 14 = 3360
$$

### 2.2 稳定化 E_L 矩阵计算

枚举版使用与稳定版相同的数值稳定化策略：

$$
L^{\text{stable}} = L - c
$$

$$
\Psi^{\text{stable}} = \exp(L^{\text{stable}})
$$

$$
(H\Psi)^{\text{stable}}_{ij} = e^{-c} H\psi_j(\mathbf{x}_i)
$$

$$
E_L = (\Psi^{\text{stable}})^{-1} (H\Psi)^{\text{stable}}
$$

### 2.3 精确加权 Loss

定义精确权重：

$$
p_\theta(X) = \frac{|\Psi_\theta(X)|^2}{\sum_X |\Psi_\theta(X)|^2}
$$

精确 loss：

$$
L_{\text{exact}} = \sum_X p_\theta(X) \cdot \text{Re}[\text{Tr}(E_L(X))]
$$

精确加权的 E_L 均值：

$$
\bar{E}_L = \sum_X p_\theta(X) E_L(X)
$$

### 2.4 Exact Covariance 梯度

与 VMC 的 sample mean 不同，枚举版使用 **exact weighted mean**：

$$
g_{\text{exact}} = \sum_X p_\theta(X) \left( \text{Tr}E_L(X) - \langle \text{Tr}E_L \rangle \right) \overline{\nabla \log \Psi(X)}
$$

---

## 3. 代码实现

### 3.1 前提条件

假设以下对象已定义：

```python
K = 3
SINGLE_SIZE = hi.size  # H2 / 6-31G = 8

ha          # NetKet Hamiltonian
hi          # Hilbert space
E_fcis      # FCI benchmark energies

total_machine       # gauge-fixed total machine
total_matrix_machine    # returns L_stable
total_max_machine       # returns shift
single_machine_list     # list of K single machines
total_params            # network parameters
Ham_Psi_scaled         # scaled Ham_Psi function
```

### 3.2 枚举合法 NES 构型

```python
def build_valid_nes_configurations(hi, K, single_size, dtype=jnp.float32):
    """
    枚举所有合法 NES samples:
        X = [x_1, ..., x_K]
    要求:
        x_i != x_j  for i != j

    返回:
        valid_X: shape (N_valid, K, single_size)
        perm_indices: shape (N_valid, K)
        single_states: shape (N_single, single_size)
    """
    single_states = jnp.asarray(hi.all_states(), dtype=dtype)
    n_single = single_states.shape[0]

    perm_indices_np = np.asarray(
        list(itertools.permutations(range(n_single), K)),
        dtype=np.int32,
    )

    perm_indices = jnp.asarray(perm_indices_np)
    valid_X = single_states[perm_indices]

    return valid_X, perm_indices, single_states
```

**期望输出**：

```
single Hilbert size = 16
K = 3
valid_X.shape = (3360, 3, 8)
expected valid count = 3360
```

### 3.3 稳定计算 E_L 矩阵

```python
def nes_local_energy_batch_stable_enum(
    ha,
    total_matrix_machine,
    total_max_machine,
    single_machine_list,
    total_params,
    x_batch,
    cond_threshold=1e12,
):
    """
    对一批 NES samples 精确计算 local energy matrix.
    """
    # 1. 稳定化矩阵
    L_stable = total_matrix_machine(total_params, x_batch)
    shift = total_max_machine(total_params, x_batch)
    Psi_stable = jnp.exp(L_stable)

    # 2. 稳定化 Ham_Psi
    HPsi_stable = Ham_Psi_scaled(
        ha=ha,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch,
        shift=shift,
    )

    # 3. 求解 E_L
    def solve_one(Psi, HPsi):
        return jnp.linalg.solve(Psi, HPsi)

    E_L_batch = jax.vmap(solve_one)(Psi_stable, HPsi_stable)

    # 4. 计算 loss 和 validity
    trace_batch = jnp.trace(E_L_batch, axis1=-2, axis2=-1)
    loss_batch = jnp.real(trace_batch)
    cond_Psi = jax.vmap(jnp.linalg.cond)(Psi_stable)

    valid = (
        jnp.all(jnp.isfinite(L_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(Psi_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(HPsi_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(E_L_batch), axis=(-2, -1))
        & jnp.isfinite(loss_batch)
        & jnp.isfinite(cond_Psi)
        & (cond_Psi < cond_threshold)
    )

    return loss_batch, E_L_batch, aux
```

### 3.4 Exact NES 权重

```python
def exact_nes_weights(total_machine, total_params, x_all):
    """
    对所有合法 NES samples 计算 normalized |Psi|^2 权重.
    """
    logPsi = total_machine(total_params, x_all)
    logw = 2.0 * jnp.real(logPsi)
    finite = jnp.isfinite(logw)

    # 防止 -inf / nan 进入 logsumexp
    logw_safe = jnp.where(finite, logw, -jnp.inf)

    logZ = jax.nn.logsumexp(logw_safe)
    weights = jnp.exp(logw_safe - logZ)

    # 保险
    weights = jnp.where(jnp.isfinite(weights), weights, 0.0)
    weights = weights / (jnp.sum(weights) + 1e-300)

    return weights, logPsi, logw
```

### 3.5 Exact Weighted Loss

```python
def exact_nes_loss_and_aux(
    total_params, x_all, ha,
    total_machine, total_matrix_machine,
    total_max_machine, single_machine_list,
):
    """
    exact enumeration 版本的 NES loss.
    """
    # 1. 计算权重
    weights, logPsi, logw = exact_nes_weights(
        total_machine=total_machine,
        total_params=total_params,
        x_all=x_all,
    )

    # 2. 计算 E_L
    loss_batch, E_L_batch, aux = nes_local_energy_batch_stable_enum(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_all,
    )

    # 3. Valid mask
    valid = aux["valid"] & jnp.isfinite(weights)
    weights_valid = jnp.where(valid, weights, 0.0)
    weights_valid = weights_valid / (jnp.sum(weights_valid) + 1e-300)

    # 4. 加权 loss
    loss_batch_safe = jnp.where(valid, loss_batch, 0.0)
    loss_exact = jnp.sum(weights_valid * loss_batch_safe)

    # 5. 加权 E_L mean
    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0 + 0.0j)
    E_L_mean = jnp.sum(
        weights_valid[:, None, None] * E_L_safe,
        axis=0,
    )

    # 6. 熵和有效样本数
    prob_entropy = -jnp.sum(
        jnp.where(weights_valid > 0, weights_valid * jnp.log(weights_valid + 1e-300), 0.0)
    )
    eff_sample_size = 1.0 / (jnp.sum(weights_valid ** 2) + 1e-300)

    return loss_exact, out
```

### 3.6 Exact Covariance 梯度

```python
def exact_nes_gradient_covariance(
    total_params, x_all, ha,
    total_machine, total_matrix_machine,
    total_max_machine, single_machine_list,
):
    """
    exact enumeration 版本的 NES-VMC covariance gradient.
    """
    loss_exact, aux = exact_nes_loss_and_aux(...)

    # 停止梯度传播
    weights = jax.lax.stop_gradient(aux["weights"])
    valid = jax.lax.stop_gradient(aux["valid"])

    E_L_batch = aux["E_L_batch"]
    E_L_mean = aux["E_L_mean"]

    # 中心化
    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0 + 0.0j)
    E_L_centered = E_L_safe - E_L_mean[None, :, :]
    trace_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)
    trace_centered = jnp.where(valid, trace_centered, 0.0 + 0.0j)
    trace_centered = jax.lax.stop_gradient(trace_centered)

    # 梯度
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    dlog_tree = jax.vmap(grad_logPsi, in_axes=(None, 0))(total_params, x_all)

    def combine_one_param(g_batch):
        expand_shape = (weights.shape[0],) + (1,) * (g_batch.ndim - 1)
        w = weights.reshape(expand_shape)
        tc = trace_centered.reshape(expand_shape)
        valid_f = valid.astype(g_batch.real.dtype).reshape(expand_shape)
        g_safe = jnp.where(jnp.isfinite(g_batch), g_batch, 0.0 + 0.0j)
        weighted = w * tc * jnp.conj(g_safe) * valid_f
        return jnp.sum(weighted, axis=0)

    grad_tree = jax.tree_util.tree_map(combine_one_param, dlog_tree)
    grad_flat, _ = ravel_pytree(grad_tree)
    grad_norm = jnp.linalg.norm(grad_flat)

    return grad_tree, loss_exact, aux["E_L_mean"], aux
```

### 3.7 诊断函数

```python
def exact_enum_diagnostics(aux, E_fcis=None, K=None):
    E_L_mean = aux["E_L_mean"]

    # Hermitian 化
    E_L_herm = 0.5 * (E_L_mean + E_L_mean.conj().T)
    eig_herm = jnp.linalg.eigvalsh(E_L_herm)

    # Raw 本征值
    eig_raw = jnp.linalg.eigvals(E_L_mean)
    eig_raw = eig_raw[jnp.argsort(jnp.real(eig_raw))]

    # Hermiticity 误差
    herm_error = (
        jnp.linalg.norm(E_L_mean - E_L_mean.conj().T)
        / (jnp.linalg.norm(E_L_mean) + 1e-12)
    )
    imag_norm = jnp.linalg.norm(jnp.imag(E_L_mean))

    # 输出
    print("-" * 90)
    print(f"Loss exact        = {float(loss_exact): .10f}")
    if E_fcis is not None and K is not None:
        target = float(np.sum(np.asarray(E_fcis[:K])))
        print(f"Target sum FCI    = {target: .10f}")
        print(f"Gap               = {float(loss_exact) - target:+.10f}")

    print(f"Grad norm         = {float(grad_norm):.6e}")
    print(f"valid_ratio       = {float(valid_ratio):.6f}")
    print(f"cond(Psi) mean    = {float(cond_mean):.6e}")
    print(f"cond(Psi) max     = {float(cond_max):.6e}")
    print(f"prob entropy      = {float(entropy):.6f}")
    print(f"effective N       = {float(eff_n):.2f}")
    print(f"herm_error        = {float(herm_error):.6e}")
    print(f"imag_norm         = {float(imag_norm):.6e}")

    print("E_herm:")
    for i, e in enumerate(np.asarray(eig_herm)):
        print(f"  E{i}_herm = {e: .10f}")

    print("E_raw:")
    for i, e in enumerate(np.asarray(eig_raw)):
        print(f"  E{i}_raw  = {e.real: .10f} {e.imag:+.3e}j")
```

### 3.8 Exact Span 诊断

检查 K 个 single ansatz 张成的线性子空间：

```python
def exact_span_diagnostics(hi, ha, single_machine_list, total_params, eps=1e-10):
    """
    检查 K 个 single ansatz 张成的线性子空间。
    """
    states = jnp.asarray(hi.all_states())
    H = jnp.asarray(ha.to_dense())

    cols = []
    for j, machine_j in enumerate(single_machine_list):
        params_j = total_params["single_ansatz_list"][j]
        logpsi_j = machine_j(params_j, states)
        logpsi_j = logpsi_j - jnp.max(jnp.real(logpsi_j))
        psi_j = jnp.exp(logpsi_j)
        cols.append(psi_j)

    Phi = jnp.stack(cols, axis=1)  # (N_single, K)

    # 重叠矩阵和有效哈密顿
    S = Phi.conj().T @ Phi
    H_eff = Phi.conj().T @ H @ Phi

    K_local = len(single_machine_list)
    S_reg = S + eps * jnp.eye(K_local, dtype=S.dtype)

    # Ritz 值
    ritz_mat = jnp.linalg.solve(S_reg, H_eff)
    ritz_vals = jnp.linalg.eigvals(ritz_mat)
    ritz_vals = ritz_vals[jnp.argsort(jnp.real(ritz_vals))]

    # Overlap
    norms = jnp.sqrt(jnp.real(jnp.diag(S)))
    overlap = S / (norms[:, None] * norms[None, :] + eps)
    cond_S = jnp.linalg.cond(S)

    return ritz_vals, S, overlap, cond_S
```

---

## 4. 训练流程

### 4.1 超参数选择

```python
Natural_Grad = False
lr_enum = 5e-4        # 保守学习率
clip_norm_enum = 0.5  # 梯度裁剪
N_ITER_ENUM = 500
PRINT_EVERY = 10
```

### 4.2 训练循环

```python
optimizer_enum = optax.chain(
    optax.clip_by_global_norm(clip_norm_enum),
    optax.sgd(learning_rate=lr_enum),
)
opt_state_enum = optimizer_enum.init(total_params)

for step in range(N_ITER_ENUM):
    # 计算梯度和 loss
    grad_raw, loss_exact, E_L_mean, aux = exact_nes_gradient_covariance(...)

    # 检查有效性
    grad_finite = jnp.all(jnp.isfinite(grad_flat))
    loss_finite = jnp.isfinite(loss_exact)

    if (not bool(grad_finite)) or (not bool(loss_finite)):
        print(f"[Step {step}] STOP: non-finite loss/grad")
        break

    # 参数更新
    updates, opt_state_enum = optimizer_enum.update(grad_raw, opt_state_enum, total_params)
    total_params = optax.apply_updates(total_params, updates)

    # 定期诊断
    if step % PRINT_EVERY == 0 or step == N_ITER_ENUM - 1:
        diag_out = exact_enum_diagnostics(aux, E_fcis=E_fcis, K=K)
```

---

## 5. 结果判断指南

### 情况 A：枚举版 loss 能稳定下降

```
✅ NES loss / E_L matrix / gradient 主链路大概率没问题
❌ 问题可能在：MCMC sampler collapse / proposal 不够强 / learning rate 太大
```

### 情况 B：枚举版也很快 NaN / cond 爆炸

```
❌ sampler 不是主锅
✅ 问题可能在：
   - NESTotalAnsatz 的 K 列线性相关
   - E_L matrix solve 病态
   - Jastrow amplitude drift
   - optimizer 过猛
   - gauge fixing 不够
```

### 情况 C：loss 下降，但 exact Ritz 不靠近 FCI 三态

```
⚠️ NES loss 在优化，但 K 个 single ansatz 没有正确张成低能三维子空间
→ 需要：FCI / CISD pretraining / orthogonalization penalty / state overlap diagnostics
```

### 情况 D：exact span Ritz 接近 FCI，但 NES loss 不对

```
⚠️ single ansatz 张成空间其实对了，但 NES determinant / local E_L estimator 或权重公式有问题
→ 需要检查：total_machine / total_matrix_machine / Ham_Psi_scaled / E_L solve / prob weights
```

---

## 6. 关键坑点

### 6.1 Ψ 条件数计算错误

**错误代码**：

```python
psi_mat = total_matrix_machine(total_params, x_single)[0]
psi_cond = jnp.linalg.cond(psi_mat)  # ❌ 错！这是 L_stable
```

**正确代码**：

```python
L_stable_single = total_matrix_machine(total_params, x_single)[0]
Psi_stable_single = jnp.exp(L_stable_single)
psi_cond = jnp.linalg.cond(Psi_stable_single)  # ✅ 正确
```

### 6.2 Gauge Drift

枚举版可以与 **Gauge Fixed Wrapper** 配合使用：

```python
total_machine, total_matrix_machine, total_max_machine, _, _ = \
    create_gauge_fixed_total_machines(total_model, ref_state)
```

Gauge fixing 可以消除 logψ 的全局相位不确定性。

---

## 7. 与采样版对比

| 特性 | 枚举版 (Exact) | 采样版 (MCMC) |
|------|---------------|---------------|
| 样本数 | 固定 3360 (K=3) | 可变 |
| 权重 | 精确 $|\Psi|^2$ 权重 | MC 估计 |
| 计算成本 | O(N_valid) | O(N_samples) |
| 诊断价值 | **高**（消除采样不确定性） | 中 |
| 实用性 | 仅适合小 Hilbert 空间 | 可扩展到大系统 |

---

## 8. 文件依赖

| 文件 | 说明 |
|------|------|
| `NES_VMC.py` | 核心 NES-VMC 实现（Ansatz、采样器、稳定化函数） |
| `NES_VMC_H2_631G.py` | H₂ 6-31G 专用工具（edges、基态计算） |
| `行动指南.md` | 本文档的原始指导 |
| `精确采样/*.ipynb` | 枚举版训练测试 notebook |

---

## 9. 总结

精确采样方案是诊断 NES-VMC 训练问题的**利器**：

1. **完全可控**：消除采样器的不确定性
2. **精确权重**：精确计算 $|\Psi|^2$ 分布下的期望值
3. **完整诊断**：提供 Loss、梯度、valid_ratio、cond(Psi)、entropy 等
4. **子空间验证**：exact_span_diagnostics 检查 ansatz 表达能力

通过枚举版和采样版的**对比**，可以精确定位训练问题的根源。

---

*文档生成时间：2026-06-28*
