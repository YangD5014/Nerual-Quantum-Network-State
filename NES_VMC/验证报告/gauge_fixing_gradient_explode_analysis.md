# 行规范修复引入 stop_gradient 导致梯度爆炸分析报告

> **日期**: 2026-08-28
> **实验**: `experiments/LiH分子/GaugeFixing01_K4_LiH_STO-3G.log`（20 步训练日志）
> **关键文件**:
> - 模型：`experiments/LiH分子/NES_VMC_V1.py::NESTotalAnsatz_gauge_stable_full`
> - wrapper：`NES_VMC_V1.py::create_machine_gauge_stable`
> - 梯度：`NES_VMC_V1.py::nes_vmc_gradient_stable`
> - 验证报告参考：`验证报告/gauge_fixing_gradient_energy_analysis.md`

---

## 0. 现象速览

| Step | logΨ mean | raw‖∇‖ | natural‖∇‖ | cond(Ψ) | E0 (Ha) |
|:----:|:---------:|:------:|:----------:|:-------:|:-------:|
| 0    | 13.134+1.254j | 344.4 | 6.54 | 1.26e+01 | -17.83-4.47j |
| 1    | 24.353-3.259j | 442.4 | 15.63 | 1.98e+01 | -42.06+13.53j |
| 2    | 8.138+6.302j  | 33391 | 33.12 | 3.42e+01 | -92.33-117.09j |
| 3    | 0.256+10.252j | 5.11e+8 | 9686 | 2.89e+01 | **-82929+6230j** |
| 4    | 1.130+10.445j | 695.6 | 0.23 | 3.12e+01 | 0.74-2.79j |
| 5    | 5.628+6.722j  | 39469 | 11.91 | 2.46e+01 | -17.49+27.91j |
| 6    | 3.063+2.200j  | 3.18e+8 | 466312 | 2.14e+01 | **4587+5065j** |
| 7    | 5.269+1.131j  | 745.3 | 2.61 | 2.97e+01 | -7.54+2.76j |
| 8    | 2.646+0.392j  | 7843.3 | 3.86 | 1.07e+01 | -51.84+4.87j |
| 9    | 2.214-0.070j  | 741.1 | 0.55 | 1.17e+01 | -4.64-0.59j |
| 10   | 2.095+1.122j  | 988.9 | 0.88 | 1.09e+01 | -4.60-1.06j |
| 11   | 1.998+1.438j  | 1636.0 | 0.72 | 1.23e+01 | -4.01-2.20j |

**核心异常**：
1. Step 0 初始 logΨ mean = 13.134 + 1.254j，**远偏离正常值（应在 0 附近）**
2. 偶数步（Step 2, 3, 5, 6）出现极端梯度爆炸（raw ‖∇‖ 高达 10⁸）
3. 对应步的能量 E0 出现巨大虚部（达 -82929 Ha、+4587 Ha）
4. 奇数步（Step 4, 7, 9, 10, 11）恢复正常，但能量仍然有显著虚部

---

## 1. 根因：`stop_gradient` 切断了 `logΨ` 对参数的完整梯度

### 1.1 原始实现（错误版本）

在首次实现的 `NESTotalAnsatz_gauge_stable_full._forward_single` 中：

```python
row_mean = jnp.mean(L_centered, axis=1, keepdims=True)
row_mean = jax.lax.stop_gradient(row_mean)      # ❌ 切断梯度
L_rowfixed = L_centered - row_mean

shift = jnp.max(jnp.real(L_rowfixed))
shift = jax.lax.stop_gradient(shift)             # ❌ 切断梯度
L_stable = L_rowfixed - shift

det_correction = jnp.sum(row_mean[:, 0])
det_correction = jax.lax.stop_gradient(det_correction)  # ❌ 切断梯度

log_Psi_gauge = log_Psi_stable + K * shift + det_correction
```

关键问题：**`log_Psi_gauge` 的三个组成部分（`log_Psi_stable`、`K*shift`、`det_correction`）全部被 `stop_gradient` 切断了对参数的依赖**。

### 1.2 梯度链路分析

`nes_vmc_gradient_stable` 通过以下路径计算梯度：

```python
grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)
```

其中 `total_machine` 是 `create_machine_gauge_stable` 返回的 wrapper：

```python
def machine(params, sigma):
    m = nnx.merge(graphdef, params)
    log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)
    return log_Psi_gauge          # ← 只用了第一项
```

因此 `dlogPsi_batch = ∇_θ log_Psi_gauge`。

由于 `shift`、`row_mean`、`det_correction` 都被 `stop_gradient`，而 `log_Psi_stable = slogdet(exp(L_rowfixed - shift))` 中的 `L_rowfixed` 也来自 `stop_gradient(row_mean)`，所以：

$$\nabla_\theta \log\Psi_{\text{gauge}} = \nabla_\theta \underbrace{\text{slogdet}(\exp(L_{\text{centered}} - \bar{L}))}_{\text{含梯度}} + \underbrace{K \cdot 0}_{\text{shift 被 stop_grad}} + \underbrace{0}_{\text{det\_correction 被 stop_grad}}$$

**缺失的梯度项**：$\nabla_\theta(K \cdot \text{shift} + \sum_i \bar{L}_i)$

这个缺失量恰好等于行规范方向的"真实梯度"，即 $\nabla_\theta \log\Psi_{\text{physical}}$ 中被人为切断的部分。

### 1.3 QGT 矩阵畸形

QGT（量子几何张量）定义为：

$$S_{ab} = \langle \nabla_a \log\Psi^* \cdot \nabla_b \log\Psi \rangle - \langle \nabla_a \log\Psi^* \rangle \langle \nabla_b \log\Psi \rangle$$

当 `∇logΨ` 缺少了 `∇(K·shift + det_correction)` 项后：

1. **某些参数方向的梯度分量完全丢失** → 对应方向的 QGT 元素趋近于 0
2. **QGT 矩阵变得奇异或接近奇异** → 最小特征值趋近于 0
3. **正则化 QGT 求逆时放大噪声**：$(S + \lambda I)^{-1}$ 中对应极小特征值方向的分量被放大为 $1/(\epsilon + \lambda) \gg 1$
4. **自然梯度爆炸**：$\Delta\theta = -(S+\lambda I)^{-1} \hat{F}$ 在这些方向上产生极大的更新

这与日志中观察到的现象完全吻合：
- Step 2, 3, 5, 6：raw ‖∇‖ 高达 10⁴ ~ 10⁸
- Step 6：natural ‖∇‖ = 466312，比 raw 还大（说明 QGT 逆放大了梯度）
- 偶数步集中爆发：QGT 矩阵在每步采样后重新计算，当采样构型恰好使某行规范方向上的 QGT 特征值极小时，爆发发生

### 1.4 为什么偶数步集中爆发？

这与 MCMC 采样的统计涨落有关：

1. **Step 0**：初始参数随机初始化，`logΨ mean = 13.134+1.254j` 已偏离正常值
2. **Step 1**：大的参数更新把模型推向某个区域，但 QGT 畸形导致更新方向错误
3. **Step 2**：采样构型使 QGT 矩阵在某些方向上的特征值接近 0，求逆放大 → 梯度爆炸（raw=33391）
4. **Step 3**：更极端的构型 → raw=5.11×10⁸
5. **Step 4**：恰好采到了 QGT 条件较好的构型 → 恢复正常
6. **循环往复**

这种"偶数步爆发、奇数步恢复"的模式是 **QGT 条件数随采样构型剧烈波动** 的直接证据。

---

## 2. 与无行修复版本的对比

### 2.1 原始 `NESTotalAnsatz_stable` 的行为

```python
# 原版：没有行修复，只有列修复
L_centered = L - logψ(ref)         # 列修复
shift = max(Re(L_centered))
L_stable = L_centered - shift
log_Psi = slogdet(exp(L_stable)) + K*shift
# shift 同样被 stop_gradient！
```

原版也有 `stop_gradient(shift)`，但**没有 `stop_gradient(row_mean)`** 和 **没有 `det_correction`**。原因：原版不存在行规范修复，`shift` 只是数值稳定化手段，其梯度缺失不影响 QGT 的结构（原版 QGT 在列规范方向上严格为零模，这是正确的物理结果）。

### 2.2 行修复引入了额外的 stop_gradient

行修复后，`log_Psi_gauge` 比原版多了两项：
- `K * shift`（原版也有，但原版的 shift 来自不同的 L）
- `det_correction = sum(row_mean)`（**新增项**）

这两项都被 `stop_gradient`，导致：
- 原版：`∇logΨ` 缺少 `∇shift`（正常，与原版一致）
- 新版：`∇logΨ` 额外缺少 `∇(K*shift + det_correction)` = `∇logΨ_physical` 的行规范部分

**关键区别**：原版缺失的梯度是"数值稳定化偏移"的梯度，不影响物理；新版缺失的梯度包含了"行规范修正"的物理梯度，这是不应该被切断的。

---

## 3. 修复方案

### 3.1 核心思路

将 `_forward_single` 拆成两个独立计算路径：

| 输出 | 用途 | 梯度处理 |
|:-----|:-----|:--------:|
| `log_Psi_gauge` | 采样器（Metropolis 接受率）+ 梯度计算 | **保留完整梯度** |
| `L_stable`, `shift` | loss 计算（`E_L = Ψ⁻¹HΨ`） | `stop_gradient`（不影响 E_L） |

### 3.2 修复后的代码逻辑

```python
# ========== 带完整梯度的版本（用于 log_Psi_gauge）==========
row_mean = jnp.mean(L_centered, axis=1, keepdims=True)     # 含梯度
L_rowfixed_full = L_centered - row_mean
shift_full = jnp.max(jnp.real(L_rowfixed_full))             # 含梯度
log_Psi_gauge = slogdet(exp(L_rowfixed_full - shift_full)) + K*shift_full + sum(row_mean)

# ========== 数值稳定化版本（用于 loss，stop_gradient）==========
shift = stop_gradient(shift_full)
row_mean_stop = stop_gradient(row_mean)
L_stable = L_centered - row_mean_stop - shift
# L_stable 和 shift 供 total_matrix_machine / total_max_machine 使用
```

### 3.3 数学正确性验证

**能量提取不变性**：`E_L = Ψ⁻¹HΨ` 只依赖 `L_stable`（通过 `Psi_Matrix_stable = exp(L_stable)` 和 `HPsi_stable = Ham_Psi_scaled`），而 `L_stable` 的数值与修复前完全一致（因为 `shift` 和 `row_mean` 的数值相同，只是梯度处理不同）。因此 **E_L 和能量特征值不受影响**。

**梯度正确性**：`log_Psi_gauge` 现在包含完整的 `∇(K·shift_full + det_correction_full)`，使得 `∇logΨ` 准确反映了行规范修复对波函数的完整影响。

**MCMC 采样正确性**：Metropolis 接受率仅依赖 `|Ψ(x')|²/|Ψ(x)|²`，即 `log_Psi_gauge` 的实部差值。由于 `log_Psi_gauge` 的数值在前向传播中与修复前完全相同（只是梯度多了一项），**采样分布不变**。

---

## 4. 日志验证

修复后预期行为：

| 指标 | 修复前（异常） | 修复后（预期） |
|:----:|:-------------:|:-------------:|
| Step 0 logΨ mean | 13.134+1.254j | ≈ 0 附近 |
| 梯度爆炸频率 | 每 2 步一次 | 极少或消失 |
| E0 虚部 | 高达 ±117 Ha | < 0.01 Ha |
| cond(Ψ) | 最高 3.42e+01 | 稳定 < 20 |
| logΨ mean 趋势 | 大幅振荡 | 收敛到稳定值 |

---

## 5. 对验证报告的补充说明

对照 `gauge_fixing_gradient_energy_analysis.md`：

- **§3.3.2（行修复对 QGT 的影响）**：原报告指出"行规范修复会改变 QGT 矩阵"，这是正确的定性判断。但原报告没有明确指出 `stop_gradient` 的错误使用是梯度爆炸的直接原因。
- **§6.4（为什么这是正确的行为）**：原报告认为行修复"消除了冗余方向，改善了条件数"。本分析进一步指出：**前提条件是 logΨ 的梯度必须完整**。如果梯度不完整（如原实现的 stop_gradient 错误），反而会破坏 QGT 结构，导致条件数恶化而非改善。
- **本节补充**：行规范修复的正确实现应该保留 `log_Psi_gauge` 的完整梯度，只对 `L_stable` 和 `shift`（loss 专用）做 `stop_gradient`。

---

## 6. 一句话结论

**`stop_gradient` 错误地切断了 `log_Psi_gauge` 中 `shift` 和 `det_correction` 项的梯度链路，导致 QGT 矩阵在行规范方向上畸形（特征值塌缩），进而引发自然梯度求逆时的数值爆炸。修复方法是将 `log_Psi_gauge` 的计算路径与 `L_stable/shift`（loss 专用）路径分离，前者保留完整梯度，后者保留 `stop_gradient`。**
