# `stop_gradient` 决策指南：何时切断、何时保留

> **核心原则**：`stop_gradient` 切断的是**梯度链路**，不影响前向数值。
> 决策的关键在于：**这个量的梯度是否对当前用途有意义？**
>
> 本文以 NES-VMC LiH 行列规范修复为案例，系统性地讲解如何判断。

---

## 1. 三句话原则

| 情况 | 应该 stop_gradient？ | 原因 |
|:-----|:-------------------:|:-----|
| 该量**只用于前向数值计算**（loss、稳定性），不需要它对参数的梯度 | ✅ 是 | 切断梯度链路，防止数值溢出/不需要的梯度贡献 |
| 该量**用于计算梯度**（如 `jax.grad(f)` 中的 `f`），且梯度必须完整 | ❌ 否 | 切断梯度会导致 QGT 畸形、自然梯度爆炸 |
| 该量是**规范方向的冗余自由度**，但你想从梯度中消除它 | ❌ 否（对 logΨ 输出）<br>✅ 是（对内部中间变量） | 见 §3 详解 |

---

## 2. 决策流程（四步法）

当你不确定是否应该对某个中间量使用 `stop_gradient` 时，依次回答以下四个问题：

### 问题 1：这个量会不会被 `jax.grad` 微分？

**检查方法**：向上追溯调用链，看它最终是否出现在某个被 `jax.grad`、`jax.value_and_grad` 调用的函数输出中。

```python
# 情况 A：会被微分 → 不要 stop_gradient
log_Psi_gauge = ...          # ← 这个会被 grad(total_machine) 微分
return log_Psi_gauge         # total_machine 的最终输出

# 情况 B：不会被微分 → 可以考虑 stop_gradient
L_stable = ...               # ← 这个只用于 exp(L_stable)，不直接参与 grad
return log_Psi_gauge, L_stable
```

**NES-VMC 案例**：
- `log_Psi_gauge` → **会被** `jax.grad(total_machine)` 微分 → ❌ 不能 stop_gradient
- `L_stable` → **不会被** grad 直接微分（只用于 `exp(L_stable)` 构造 Ψ 矩阵）→ ✅ 可以 stop_gradient

### 问题 2：这个量的梯度是否有物理意义？

**检查方法**：问自己"如果我不切断这个梯度，参数会朝哪个方向更新？这个方向是物理上有意义的吗？"

```
有物理意义 → ❌ 不要 stop_gradient
   例如：logψ_j(x) 的梯度 → 更新 ansatz 参数，改变波函数形状

无物理意义（规范冗余）→ ✅ 可以考虑 stop_gradient
   例如：row_mean = mean_j(L[i,j]) → 改变的是所有 walker 的全局偏移，
        不改变 |Ψ|²，不改变任何可观测量
```

**NES-VMC 案例**：
- `row_mean`（行均值）→ 仅改变 logΨ 的全局偏移，不改变 `|Ψ|²`，**无物理意义** → ✅ 可以对内部计算 stop_gradient
- 但 `log_Psi_gauge = slogdet(...) + K*shift + sum(row_mean)` → 这三项合起来构成了**完整的波函数相位和模长**，**有物理意义** → ❌ 不能 stop_gradient

**关键陷阱**：`row_mean` 单独看是无物理意义的，但它**参与构成**了有物理意义的 `log_Psi_gauge`。你不能在它参与计算之前就切断梯度。

### 问题 3：这个量的数值稳定性是否需要保护？

**检查方法**：问自己"这个量会不会溢出/下溢？如果它爆炸了，会不会毁掉整个计算？"

```python
# 数值不稳定 → ✅ 对它做 stop_gradient 是合理的
shift = jnp.max(jnp.real(L))       # 可能非常大（如 67.0）
shift = jax.lax.stop_gradient(shift)  # 切断梯度，只在前向中使用
L_stable = L - shift               # 确保 exp(L_stable) 不会溢出
```

**原理**：`shift` 的作用是数值稳定化，它不应该影响梯度方向。如果 `shift` 有梯度，它会在优化过程中引入额外的"数值梯度"，干扰物理梯度。

**NES-VMC 案例**：
- `shift = max(Re(L))` → 可能从 9.26 漂移到 67.29 → ✅ 应该 stop_gradient（对 loss 计算路径）
- 但 `shift_full`（用于 `log_Psi_gauge` 的路径）→ ❌ 不应该 stop_gradient（因为 `log_Psi_gauge` 需要完整梯度）

### 问题 4：切断梯度后，下游计算是否正确？

**检查方法**：逐项验证每个下游消费者：
- 梯度消费者：`jax.grad(total_machine)` → 需要完整梯度
- 数值消费者：`NES_loss_energy_stable` → 只需要数值正确

```python
# 正确的做法：两条独立路径
# 路径 A：完整梯度（供 grad 使用）
log_Psi_gauge = slogdet(...) + K * shift_full + det_correction_full
# 路径 B：数值稳定化（供 loss 使用）
L_stable = L_centered - stop_gradient(row_mean) - stop_gradient(shift_full)
```

**NES-VMC 案例（修复前）**：
```python
# ❌ 错误：所有路径共用同一个 stop_gradient 版本
shift = stop_gradient(max(...))
row_mean = stop_gradient(mean(...))
log_Psi_gauge = slogdet(...) + K*shift + det_correction  # ← shift 和 det_correction 都无梯度！
L_stable = L_centered - row_mean - shift                  # ← 数值正确，但梯度链已断
```

**NES-VMC 案例（修复后）**：
```python
# ✅ 正确：两条路径分离
shift_full = max(...)                        # 含梯度
row_mean = mean(...)                         # 含梯度
log_Psi_gauge = slogdet(...) + K*shift_full + sum(row_mean)  # 完整梯度

shift = stop_gradient(shift_full)            # 切断梯度
row_mean_stop = stop_gradient(row_mean)
L_stable = L_centered - row_mean_stop - shift  # 数值稳定化，不影响梯度
```

---

## 3. 典型案例：NES-VMC 中的三种 stop_gradient

以下是 NES-VMC 中 `stop_gradient` 出现的三种典型场景，它们的决策逻辑完全不同：

### 场景 A：`ref_logs`（列规范参考值）— ❌ 不应该 stop_gradient

```python
ref_val = ansatz_j(self.ref_state)    # ← 不要 stop_gradient
log_col_centered = log_x - ref_val
```

**为什么？** `ref_val` 依赖于参数 `θ`（因为 ansatz_j 的参数在训练）。如果对它 stop_gradient：
- `logΨ_col = logΨ_raw - stop_grad(logψ(ref))`
- `∇logΨ_col = ∇logΨ_raw - 0 = ∇logΨ_raw`
- **列规范方向没有从梯度中消除**，优化器仍然可以在列规范方向上漂移

不 stop_gradient 的效果：
- `∇logΨ_col = ∇logΨ_raw - ∇logψ(ref)`
- 在梯度公式中，`tr_centered` 已经中心化（均值 0），常数偏移被消去
- **列规范方向的自然梯度严格为零** ✓

> **经验法则**：规范参考值（reference）不应该 stop_gradient，否则规范修复形同虚设。

### 场景 B：`valid` mask（采样有效性标记）— ✅ 必须 stop_gradient

```python
valid = jnp.isfinite(log_Psi_batch)   # 布尔数组
valid = jax.lax.stop_gradient(valid)  # ← 必须 stop_gradient
```

**为什么？** `valid` 是 0/1 离散值，它的梯度要么为零要么未定义。如果不 stop_gradient：
- JAX 在尝试对布尔值求梯度时会报错或产生 NaN
- 即使不报错，`∇valid` 也是零，只会白白增加计算图复杂度

> **经验法则**：离散值（布尔 mask、整数索引、argmax 结果）必须 stop_gradient。

### 场景 C：`shift`（数值稳定化偏移）— ✅ 对 loss 路径 stop，❌ 对 grad 路径不停

```python
# 对 loss 路径：stop_gradient
shift = jax.lax.stop_gradient(jnp.max(jnp.real(L)))
L_stable = L - shift

# 对 grad 路径：不 stop_gradient
shift_full = jnp.max(jnp.real(L))           # 含梯度
log_Psi_gauge = slogdet(exp(L - shift_full)) + K * shift_full
```

**为什么？** 这是最容易出错的地方。`shift` 的双重角色：
1. **数值稳定化**：`exp(L - shift)` 防止溢出 → 只需要数值，不需要梯度 → ✅ stop_gradient
2. **logΨ 的一部分**：`logΨ = slogdet(...) + K*shift` → 需要完整梯度 → ❌ 不能 stop_gradient

**历史错误**：之前的实现在两条路径上都用了同一个 `shift = stop_gradient(...)`，导致 `log_Psi_gauge` 的梯度缺失了 `K*∇shift` 项。

> **经验法则**：当一个量同时用于"数值稳定化"和"梯度计算"时，**必须分成两条路径**，一条 stop_gradient，一条保留梯度。

---

## 4. 快速检查清单

在你写代码时，对照以下清单逐一检查：

### 4.1 前向数值正确性

- [ ] `exp(L_stable)` 不会溢出（`L_stable` 的实部最大值 ≤ 0）
- [ ] `log_Psi_gauge` 的数值与前向计算一致（没有因为 stop_gradient 导致数值偏移）
- [ ] `E_L = Ψ⁻¹ H Ψ` 的结果与不使用规范修复时一致

### 4.2 梯度正确性

- [ ] `log_Psi_gauge` 的所有组成部分都有梯度（没有被意外 stop_gradient）
- [ ] `ref_logs`（列规范参考值）没有 stop_gradient
- [ ] `row_mean` 用于 `log_Psi_gauge` 的部分没有 stop_gradient
- [ ] 梯度在列规范方向上接近零（验证列修复有效）
- [ ] QGT 矩阵的最小特征值 ≈ 0（数量 = 规范自由度的数量）

### 4.3 数值稳定性

- [ ] `shift` 被 stop_gradient 后，`L_stable` 不会导致 exp 溢出
- [ ] 偶数步/奇数步的梯度幅度没有出现周期性震荡（如原始日志中 Step 2,3,5,6 爆发）
- [ ] `cond(Ψ)` 保持在合理范围（< 50）

### 4.4 物理正确性

- [ ] 能量 E0 收敛到已知精确值（LiH: -7.8636 Ha）
- [ ] logΨ mean 不再单调漂移
- [ ] Metropolis 接受率保持合理（0.2 ~ 0.8）

---

## 5. 调试工具

### 5.1 检查梯度是否完整

```python
import jax

# 打印 log_Psi_gauge 的梯度幅值
grad_fn = jax.grad(total_machine, argnums=0, holomorphic=True)
grad = grad_fn(params, x_batch)
grad_norm = jax.tree_util.tree_map(jnp.linalg.norm, grad)
print(f"Gradient norm: {jnp.mean(grad_norm):.4f}")

# 如果梯度 norm 异常大或异常小，说明梯度链路可能断裂
```

### 5.2 检查 QGT 特征值

```python
qgt = make_qgt_fn(total_machine)
s_matrix = qgt(params, sigma, diag_shift=0.1)[0]
eigvals = jnp.linalg.eigvalsh(s_matrix)
print(f"QGT min eigval: {jnp.min(eigvals):.6e}")
print(f"QGT max eigval: {jnp.max(eigvals):.6e}")
print(f"QGT condition number: {jnp.max(jnp.abs(eigvals)) / jnp.min(jnp.abs(eigvals)):.2e}")

# 最小特征值应该 ≈ 0（规范方向），但如果它不是精确的 0 而是 1e-10 量级，
# 说明 stop_gradient 可能切断了部分梯度
```

### 5.3 对比前向与梯度的一致性

```python
# 用数值梯度验证解析梯度
from jax import jax

def check_gradient_consistency(params, x):
    # 解析梯度
    grad_analytic = jax.grad(total_machine, argnums=0)(params, x)

    # 数值梯度（扰动法）
    eps = 1e-5
    params_flat, unravel = jax.tree_util.tree_flatten(params)
    grad_numerical = []
    for i, p in enumerate(params_flat):
        p_plus = p.at[0].add(eps)
        p_minus = p.at[0].sub(eps)
        params_plus = unravel(jax.tree_util.tree_replace({0: p_plus}, params_flat))
        params_minus = unravel(jax.tree_util.tree_replace({0: p_minus}, params_flat))
        diff = (total_machine(params_plus, x) - total_machine(params_minus, x)) / (2 * eps)
        grad_numerical.append(diff / jnp.abs(diff) * jnp.abs(grad_analytic[0][0]))

    return jnp.mean(jnp.abs(grad_analytic - jnp.array(grad_numerical)))

# 如果这个值 > 1e-3，说明梯度链路有问题（可能是 stop_gradient 误用）
```

---

## 6. 总结：stop_gradient 的三个使用场景

```
┌─────────────────────────────────────────────────────────────────┐
│  stop_gradient 的正确使用场景                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 数值稳定化（最常见）                                           │
│     shift = max(Re(L))                                           │
│     shift = stop_gradient(shift)  ← 只用于 exp(L-shift)           │
│     L_stable = L - shift                                          │
│     ⚠️ 但 logΨ = slogdet(exp(L_stable)) + K*shift                │
│            这里的 shift 必须用 stop_gradient 之前的版本！           │
│                                                                 │
│  2. 离散值 / 布尔 mask                                             │
│     valid = jnp.isfinite(log_Psi)                                │
│     valid = stop_gradient(valid)  ← 离散值无梯度                    │
│                                                                 │
│  3. 规范方向的冗余自由度（内部中间变量）                              │
│     row_mean = mean_j(L)                    ← 内部用              │
│     row_mean = stop_gradient(row_mean)    ← 切断对 loss 的影响     │
│     L_rowfixed = L - row_mean                                       │
│     ⚠️ 但 logΨ = slogdet(...) + sum(row_mean)                    │
│            这里的 row_mean 必须用 stop_gradient 之前的版本！         │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  stop_gradient 的错误使用场景（会导致梯度爆炸）                       │
│                                                                 │
│  ❌ 对最终输出 stop_gradient                                        │
│     log_Psi_gauge = ... + K*shift + det_correction               │
│     return log_Psi_gauge  ← 这个值的梯度必须完整！                  │
│                                                                 │
│  ❌ 对规范参考值 stop_gradient                                      │
│     ref_val = stop_gradient(ansatz(ref_state))                   │
│     ← 这会使列规范修复失效，梯度中仍包含规范方向分量                  │
│                                                                 │
│  ❌ 对参与 logΨ 计算的中间变量 stop_gradient                         │
│     row_mean = stop_gradient(mean_j(L))                          │
│     log_Psi = slogdet(...) + sum(row_mean)  ← row_mean 被切断了    │
│     ← 导致 ∇logΨ 缺失 K*∇shift + ∇sum(row_mean)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7. 附：本案例的修复前后对比

### 修复前（错误）

```python
def _forward_single(self, x_single):
    # ... 列修复 ...
    L_centered = ...

    row_mean = jnp.mean(L_centered, axis=1, keepdims=True)
    row_mean = jax.lax.stop_gradient(row_mean)    # ❌ 过早切断
    L_rowfixed = L_centered - row_mean

    shift = jnp.max(jnp.real(L_rowfixed))
    shift = jax.lax.stop_gradient(shift)           # ❌ 过早切断
    L_stable = L_rowfixed - shift

    det_correction = jnp.sum(row_mean[:, 0])
    det_correction = jax.lax.stop_gradient(det_correction)  # ❌ 过早切断

    log_Psi_gauge = slogdet(...) + K*shift + det_correction
    # ↑ 所有三项都被 stop_gradient，∇logΨ 缺失
    return log_Psi_gauge, L_stable, shift, L_rowfixed
```

**后果**：`∇logΨ` 缺少 `∇(K·shift + sum(row_mean))`，QGT 矩阵在行规范方向上特征值塌缩，求逆时爆炸。

### 修复后（正确）

```python
def _forward_single(self, x_single):
    # ... 列修复 ...
    L_centered = ...

    # ========== 路径 A：完整梯度（用于 log_Psi_gauge）==========
    row_mean = jnp.mean(L_centered, axis=1, keepdims=True)  # 含梯度
    L_rowfixed_full = L_centered - row_mean
    shift_full = jnp.max(jnp.real(L_rowfixed_full))          # 含梯度
    log_Psi_gauge = slogdet(exp(L_rowfixed_full - shift_full)) + K*shift_full + sum(row_mean)

    # ========== 路径 B：数值稳定化（用于 loss）==========
    shift = jax.lax.stop_gradient(shift_full)               # 切断梯度
    row_mean_stop = jax.lax.stop_gradient(row_mean)         # 切断梯度
    L_stable = L_centered - row_mean_stop - shift
    # L_stable 的数值与路径 A 完全一致，但梯度链被切断
    return log_Psi_gauge, L_stable, shift, L_rowfixed
```

**效果**：`∇logΨ` 完整，QGT 矩阵在规范方向上有精确的零特征值（物理要求），在非规范方向上有正常特征值（数值稳定）。
