# NES-VMC LiH (K=4) 行规范修复分析报告

> **分析日期**: 2026-08-23  
> **对比文件**:
> - 原版：`experiments/LiH分子/LiH_molecule_K4_STO-3G-compare.ipynb` + `GaugeFixing00_K4_LiH_STO-3G.log`
> - 修复版：`experiments/LiH分子/LiH_molecule_K4_STO-3G-gaugefixing.ipynb` + `GaugeFixing01_K4_LiH_STO-3G.log`
> - 核心代码：`experiments/LiH分子/NES_VMC_V1.py`

---

## 0. 核心结论

**修复版代码的行规范修复并未生效**。两个日志文件的 `logΨ mean` 完全一致（Step 0: 9.264 → Step 29: 31.532），说明修改没有起作用。

**根本原因**：`NESTotalAnsatz_gauge_stable_full` 类的 `_forward_single` 方法虽然实现了行规范修复，但 `create_gauge_fixed_total_machines` 函数创建的 wrapper **没有调用这个方法**，而是使用了自己内部定义的 `_compute_L_centered_single` 和 `_stable_from_L` 函数，这两个函数**只做了列规范修复，没有做行规范修复**。

---

## 1. 代码对比分析

### 1.1 原版代码（compare.ipynb）

```python
# 使用 NESTotalAnsatz_stable
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)

(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_params,
) = create_gauge_fixed_total_machines(
    total_ansatz,
    Hatree_Fock,
)
```

**问题**：`NESTotalAnsatz_stable` 和 `create_gauge_fixed_total_machines` 都**只做了列规范修复**（减去参考态的 `logψ_j(ref)`），没有处理行规范。

### 1.2 修复版代码（gaugefixing.ipynb）

```python
# 使用 NESTotalAnsatz_gauge_stable_full
total_ansatz = NESTotalAnsatz_gauge_stable_full(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    ref_state=Hatree_Fock,
    rngs=nnx.Rngs(11),
)

(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_params,
) = create_gauge_fixed_total_machines(
    total_ansatz,
    Hatree_Fock,
)
```

**意图**：`NESTotalAnsatz_gauge_stable_full` 类的 `_forward_single` 方法（NES_VMC_V1.py#L281-336）确实实现了行规范修复：

```python
# NES_VMC_V1.py#L301-310
# ==========【新增：行规范修复 Row gauge fixing】==========
row_mean = jnp.mean(L, axis=1, keepdims=True)   # shape [K,1]
row_mean = jax.lax.stop_gradient(row_mean)      # 切断梯度
L_rowfixed = L - row_mean                       # 消去行规范自由度
det_correction = jnp.sum(row_mean[:, 0])        # 行列式修正
```

**但是**，`create_gauge_fixed_total_machines` 函数（NES_VMC_V1.py#L1300-1374）内部定义了自己的 `_compute_L_centered_single` 和 `_stable_from_L`：

```python
# NES_VMC_V1.py#L1324-1350
def _compute_L_centered_single(m, x_single):
    """只做了列规范修复"""
    cols = []
    for j in range(K):
        ansatz_j = m.single_ansatz_list[j]
        log_col = ansatz_j(x_single)
        log_ref = ansatz_j(ref_state)
        log_col_centered = log_col - log_ref  # 列规范修复
        cols.append(log_col_centered)
    L_centered = jnp.stack(cols, axis=1)
    return L_centered  # ❌ 没有行规范修复！

# NES_VMC_V1.py#L1352-1367
def _stable_from_L(L):
    """只做数值稳定化"""
    shift = jnp.max(jnp.real(L))
    shift = jax.lax.stop_gradient(shift)
    L_stable = L - shift
    Psi_stable = jnp.exp(L_stable)
    sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
    log_det_stable = log_abs_det + 1j * jnp.angle(sign)
    log_det_centered = log_det_stable + K * shift
    return log_det_centered, L_stable, shift  # ❌ 没有行规范修复！
```

**关键问题**：这两个函数**直接操作模型的 `single_ansatz_list`**，完全绕过了模型的 `_forward_single` 方法。所以即使 `NESTotalAnsatz_gauge_stable_full._forward_single` 实现了行规范修复，也**不会被调用**。

---

## 2. 日志对比验证

### 2.1 前 30 步 `logΨ mean` 对比

| Step | 原版 (GaugeFixing00) | 修复版 (GaugeFixing01) | 差异 |
|:-----|:---------------------|:-----------------------|:-----|
| 0    | 9.264                | 9.264                  | 0.000 |
| 5    | 14.765               | 14.765                 | 0.000 |
| 10   | 18.971               | 18.971                 | 0.000 |
| 15   | 22.374               | 22.374                 | 0.000 |
| 20   | 26.246               | 26.246                 | 0.000 |
| 25   | 29.426               | 29.426                 | 0.000 |
| 29   | 31.532               | 31.532                 | 0.000 |

**结论**：两个日志的 `logΨ mean` **完全一致**（精度到小数点后 3 位），证明行规范修复**没有生效**。

### 2.2 其他指标对比

- `min(logΨ)`、`max(logΨ)`：完全一致
- `cond(Ψ)`：完全一致（如 Step 26: 7.52e+01）
- `‖∇raw‖`：完全一致（如 Step 29: 92.73）
- `E0`：完全一致（如 Step 29: -7.7483）

**所有指标都完全一致**，进一步证实修复代码没有起作用。

---

## 3. 根因总结

### 3.1 代码架构问题

`create_gauge_fixed_total_machines` 函数的设计存在**职责混乱**：

1. **模型类**（`NESTotalAnsatz_gauge_stable_full`）负责定义"如何计算 logΨ"，包括规范修复
2. **wrapper 函数**（`create_gauge_fixed_total_machines`）负责创建供 NetKet 使用的 callable
3. **但是** wrapper 函数**没有调用模型的前向方法**，而是自己重新实现了一套逻辑

这导致：
- 模型的 `_forward_single` 方法中的行规范修复被忽略
- wrapper 函数中的 `_compute_L_centered_single` 只做了列规范修复
- 最终效果：**行规范修复完全没有生效**

### 3.2 为什么会有这种设计？

可能的原因：
1. **历史遗留**：最初只有 `NESTotalAnsatz_stable`，后来为了修复行规范创建了 `NESTotalAnsatz_gauge_stable_full`，但忘记更新 wrapper 函数
2. **性能考虑**：wrapper 函数直接操作 `single_ansatz_list` 可能比调用 `_forward_single` 更高效（避免了额外的函数调用开销）
3. **灵活性**：wrapper 函数可以更灵活地控制返回哪些中间结果（如 `L_stable`、`shift`）

但无论原因是什么，**当前的实现是错误的**。

---

## 4. 修复建议

### 4.1 [推荐] 方案 A：让 wrapper 调用模型的 `_forward_single`

修改 `create_gauge_fixed_total_machines` 函数，让它调用模型的 `_forward_single` 方法：

```python
def create_gauge_fixed_total_machines(total_model, ref_state):
    graphdef, state = nnx.split(total_model)
    K = total_model.K
    n_spin = total_model.n_spin
    flat_size = K * n_spin
    ref_state = jnp.asarray(ref_state)

    def _one_from_matrix(m, x_single):
        """
        x_single: shape (K, n_spin)
        调用模型的 _forward_single 方法，自动应用行规范修复
        """
        # 调用模型的 _forward_single
        log_Psi_gauge, L_stable, shift = m._forward_single(x_single)
        return log_Psi_gauge, L_stable, shift

    # ... 其余代码保持不变
```

**优点**：
- 代码简洁，职责清晰
- 模型的规范修复逻辑不会被忽略
- 未来修改模型时，wrapper 自动继承

**缺点**：
- 需要确保 `_forward_single` 的返回值格式与 wrapper 期望的一致
- 可能需要调整 `_forward_single` 的返回值（当前返回四元组，wrapper 需要三元组）

### 4.2 [次推荐] 方案 B：在 wrapper 函数中实现行规范修复

如果不想改动 wrapper 的调用逻辑，可以在 `_compute_L_centered_single` 中直接添加行规范修复：

```python
def _compute_L_centered_single(m, x_single):
    """
    x_single: shape (K, n_spin)
    返回:
        L_centered: shape (K, K)
    """
    cols = []
    for j in range(K):
        ansatz_j = m.single_ansatz_list[j]
        log_col = ansatz_j(x_single)
        log_ref = ansatz_j(ref_state)
        log_col_centered = log_col - log_ref
        cols.append(log_col_centered)
    
    L_centered = jnp.stack(cols, axis=1)  # (K, K)
    
    # ==========【新增：行规范修复】==========
    row_mean = jnp.mean(L_centered, axis=1, keepdims=True)  # shape [K,1]
    row_mean = jax.lax.stop_gradient(row_mean)
    L_centered = L_centered - row_mean  # 消去行规范自由度
    
    return L_centered
```

同时需要在 `_stable_from_L` 中添加行列式修正：

```python
def _stable_from_L(L):
    """
    L: shape (K, K) - 已经过行规范修复
    """
    # 计算行投影的行列式修正
    # 注意：这里的 L 已经是 L - row_mean，所以需要重新计算原始的 row_mean
    # 但 row_mean 已经被 stop_gradient，无法直接获取
    # 解决方案：在 _compute_L_centered_single 中返回 row_mean
    
    shift = jnp.max(jnp.real(L))
    shift = jax.lax.stop_gradient(shift)
    L_stable = L - shift
    Psi_stable = jnp.exp(L_stable)
    
    sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
    log_det_stable = log_abs_det + 1j * jnp.angle(sign)
    log_det_centered = log_det_stable + K * shift
    
    return log_det_centered, L_stable, shift
```

**问题**：需要在 `_compute_L_centered_single` 中返回 `row_mean`，以便在 `_stable_from_L` 中计算行列式修正。这需要修改函数签名。

### 4.3 [可选] 方案 C：周期性 re-centering（补丁方案）

如果不想改动核心代码，可以在训练循环中每 N 步强行把 `logΨ mean` 拉回目标值：

```python
# 在训练循环中
if step % 50 == 0 and step > 0:
    target_logPsi = 10.0  # 或其他目标值
    current_logPsi = log_Psi_batch.mean()
    delta = target_logPsi - current_logPsi
    
    # 通过调整输出 bias 实现
    for ans in total_ansatz.single_ansatz_list:
        # 注意：这需要访问 ansatz 的输出层 bias
        # 具体实现取决于 SingleStateAnsatz 的结构
        pass
```

**优点**：实现简单，不需要重写核心代码  
**缺点**：是个"补丁"，理论上仍可能与优化器互相干扰

---

## 5. 验证修复是否生效

修复后，应该观察到以下现象：

1. **`logΨ mean` 不再单调上升**：应该在某个值附近波动（如 10±5），而不是从 9 漂移到 67
2. **`cond(Ψ)` 保持在合理范围**：应该 < 50，不会出现 236 这种异常值
3. **`‖∇raw‖` 不会出现爆发**：不会出现从 ~90 跳到 1393 的情况
4. **能量收敛不受影响**：E0 应该仍然能收敛到 -7.8629 附近

---

## 6. 一句话总结

**修复版代码的行规范修复没有生效**，因为 `create_gauge_fixed_total_machines` 函数创建的 wrapper **没有调用模型的 `_forward_single` 方法**，而是使用了自己内部定义的只做了列规范修复的函数。**建议让 wrapper 调用模型的 `_forward_single`（方案 A）**，或者在 wrapper 函数中直接实现行规范修复（方案 B）。

---

## 附录：代码调用链对比

### 原版（compare.ipynb）

```
LiH_molecule_K4_STO-3G-compare.ipynb
  ↓ 使用
NESTotalAnsatz_stable
  ↓ 传入
create_gauge_fixed_total_machines()
  ↓ 内部定义
_compute_L_centered_single()  # 只做了列规范修复
_stable_from_L()              # 只做数值稳定化
  ↓ 返回
total_machine, total_matrix_machine, total_max_machine
  ↓ 用于训练
grad_fn, qgt_fn
```

### 修复版（gaugefixing.ipynb）

```
LiH_molecule_K4_STO-3G-gaugefixing.ipynb
  ↓ 使用
NESTotalAnsatz_gauge_stable_full  # ✅ _forward_single 实现了行规范修复
  ↓ 传入
create_gauge_fixed_total_machines()
  ↓ 内部定义（❌ 没有调用模型的 _forward_single）
_compute_L_centered_single()  # ❌ 只做了列规范修复
_stable_from_L()              # ❌ 只做数值稳定化
  ↓ 返回
total_machine, total_matrix_machine, total_max_machine  # ❌ 行规范修复未生效
  ↓ 用于训练
grad_fn, qgt_fn
```

**关键断点**：`create_gauge_fixed_total_machines` 函数没有调用模型的 `_forward_single` 方法。
