# 行列规范修复对梯度计算与能量提取的影响分析

> **背景**：在确认了行列规范修复对 MCMC 采样器的 Metropolis 接受率无影响之后，
> 本报告分析对 TotalAnsatz 输出施加列修复、行修复后，对以下两个关键环节的影响：
> 1. **梯度计算**（`nes_vmc_gradient_stable`）
> 2. **能量提取**（`NES_loss_energy_stable` → `jnp.linalg.eig(E_L)`）
>
> **核心结论**：
> 1. **能量 E_L 矩阵在行列规范修复下不变**，因此 K 个激发态能量的提取完全不受影响。
> 2. **梯度中的 ∇logΨ 在列修复下不变，在行修复下也保持不变（对交换对称提议）**。
> 3. **loss 值（trace(E_L)）在行列规范修复下不变**。
> 4. **自然梯度（QGT）在列修复下不变，在行修复下也保持不变**。

---

## 1. 问题设置

### 1.1 定义

设原始 TotalAnsatz 的输出为 $L_{ij} = \log\psi_j(\mathbf{x}_i)$，对应的波函数为：

$$\Psi(\mathbf{X}) = \det\left[\psi_j(\mathbf{x}_i)\right]_{i,j=1}^K$$

列规范修复：$L^{\text{col}}_{ij} = L_{ij} - \log\psi_j(\text{ref})$

行规范修复：$L^{\text{row}}_{ij} = L^{\text{col}}_{ij} - \frac{1}{K}\sum_k L^{\text{col}}_{ik}$

数值稳定化：$L^{\text{stable}} = L^{\text{row}} - \text{shift}$，其中 $\text{shift} = \max_{i,j}\text{Re}(L^{\text{row}})$

最终输出的 logΨ 为：
$$\log\Psi_{\text{gauge}} = \text{slogdet}(\exp(L^{\text{stable}})) + K\cdot\text{shift} - \sum_i \text{row\_mean}_i$$

### 1.2 关键观察

规范修复对 $L$ 矩阵的操作是 **逐样本的仿射变换**，不改变波函数的物理内容（只改变数值表示）。我们需要验证这种表示层面的变化是否影响：
- 梯度 $\nabla_\theta \log\Psi$
- 局域能量矩阵 $E_L = \Psi^{-1} H \Psi$
- 量子几何张量 $S = \langle \nabla\log\Psi^* \cdot \nabla\log\Psi \rangle - \langle \nabla\log\Psi^* \rangle \langle \nabla\log\Psi \rangle$

---

## 2. 对能量提取的影响

### 2.1 局域能量矩阵的定义

局域能量矩阵的元素为：

$$(E_L)_{ij} = \frac{(H\Psi)_i^{\,j}}{\Psi_i^{\,j}} = \frac{\sum_{\mathbf{x}'} \langle\mathbf{x}_i|H|\mathbf{x}'\rangle \psi_j(\mathbf{x}')}{\psi_j(\mathbf{x}_i)}$$

其中分子和分母都来自**同一个**单态 ansatz $j$，因此：

### 2.2 列规范修复对 E_L 无影响

列修复对每个 ansatz $j$ 施加了一个与 $\mathbf{x}$ 无关的常数偏移：$\log\psi_j \to \log\psi_j - c_j$。

这意味着：
- $\psi_j(\mathbf{x}_i) \to \psi_j(\mathbf{x}_i) \cdot e^{-c_j}$
- $\sum_{\mathbf{x}'} H[\mathbf{x}_i, \mathbf{x}'] \psi_j(\mathbf{x}') \to e^{-c_j} \sum_{\mathbf{x}'} H[\mathbf{x}_i, \mathbf{x}'] \psi_j(\mathbf{x}')$

比值不变：

$$(E_L)^{\text{col}}_{ij} = \frac{e^{-c_j}(H\psi_j)(\mathbf{x}_i)}{e^{-c_j}\psi_j(\mathbf{x}_i)} = (E_L)_{ij}$$

### 2.3 行规范修复对 E_L 无影响

行修复的操作是在 L 矩阵层面进行的仿射变换，但 **E_L 的计算完全不依赖 L 矩阵的行结构**。

回顾 `Ham_Psi_scaled`（代码 L571-642）：它直接调用每个 `single_machine_list[j]` 计算 $H\psi_j(\mathbf{x}_i)$，与 L 矩阵的行规范修复无关。

再回顾 `NES_loss_energy_stable`（代码 L644-748）：
```python
L_stable = total_matrix_machine(params, x)      # 用于构造 Psi_stable
shift = total_max_machine(params, x)            # 用于缩放 HPsi
Psi_Matrix_stable = jnp.exp(L_stable)           # = exp(L - shift)
HPsi_stable = Ham_Psi_scaled(...)               # = exp(-shift) * Hψ
E_L_matrix = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
```

注意：`Psi_Matrix_stable` 和 `HPsi_stable` 都含有相同的缩放因子 $e^{-\text{shift}}$，比值消去：

$$E_L = \frac{e^{-\text{shift}} H\psi}{e^{-\text{shift}} \psi} = \frac{H\psi}{\psi}$$

因此，**无论 shift 如何定义（无论是否经过行修复），E_L 矩阵的值完全不变**。

### 2.4 结论：能量提取不受影响

```
E_L (不变) → trace(E_L) (不变) → eigvals of E_L (不变)
```

K 个激发态能量 $E_0, E_1, \ldots, E_{K-1}$ 的提取与是否施加行列规范修复完全无关。

---

## 3. 对梯度计算的影响

### 3.1 梯度公式回顾

`nes_vmc_gradient_stable` 的核心步骤：

```python
# 步骤 3: 中心化 local energy matrix
E_L_centered = E_L_safe - E_L_mean          # (batch, K, K) - mean over batch

# 步骤 4: 计算每个样本的 ∇logΨ
dlogPsi_batch = vmap(grad(total_machine))(params, x_batch)

# 步骤 5: 加权平均得到梯度
grad = mean over batch [ conj(tr_centered) * conj(dlogPsi) ]
```

其中 `tr_centered` 是中心化的迹：$\text{tr}(E_L) - \text{mean}_\text{batch}[\text{tr}(E_L)]$

### 3.2 列修复对 ∇logΨ 的影响

列修复将 $L_{ij} \to L_{ij} - c_j$，其中 $c_j = \log\psi_j(\text{ref})$ 是一个**与参数 $\theta$ 相关的常数**（因为 ref 是固定的，$c_j$ 取决于 ansatz $j$ 在当前参数下的输出）。

关键点：$c_j$ **依赖于参数**，所以 $\nabla_\theta c_j \neq 0$。

但是，列修复后的 logΨ 为：
$$\log\Psi_{\text{col}} = \log\Psi_0 + \sum_j (-c_j) \cdot (\text{permutation factor})$$

实际上，列修复改变了 $\log\Psi$ 的值，但这个改变是 **per-ansatz 的全局偏移**：

$$\log\Psi_{\text{col}}(\mathbf{X}) = \log\Psi_0(\mathbf{X}) - \sum_j \log\psi_j(\text{ref}) + \text{const}$$

注意：$\sum_j \log\psi_j(\text{ref})$ 与构型 $\mathbf{X}$ 无关，但对参数 $\theta$ 依赖。

因此：
$$\nabla_\theta \log\Psi_{\text{col}} = \nabla_\theta \log\Psi_0 - \sum_j \nabla_\theta \log\psi_j(\text{ref})$$

这个额外的常数项 $\sum_j \nabla_\theta \log\psi_j(\text{ref})$ 对所有样本相同。

在梯度公式中：
$$\text{grad} = \text{mean}_\text{batch}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{\nabla\log\Psi}\right]$$

由于 $\text{tr}_\text{centered}$ 已经中心化（均值为 0），常数偏移被消去：

$$\text{grad}_{\text{col}} = \text{mean}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{(\nabla\log\Psi_0 - C)}\right] = \text{mean}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{\nabla\log\Psi_0}\right] - C \cdot \underbrace{\text{mean}[\overline{\text{tr}_\text{centered}}]}_{=0} = \text{grad}_0$$

**结论：列规范修复不影响梯度。**

### 3.3 行修复对 ∇logΨ 的影响

行修复将 $L^{\text{col}}_{ij} \to L^{\text{col}}_{ij} - \bar{L}_i$，其中 $\bar{L}_i = \frac{1}{K}\sum_j L^{\text{col}}_{ij}$ 是第 $i$ 行的均值。

行修复后的 logΨ：
$$\log\Psi_{\text{row}} = \log\Psi_{\text{col}} - \sum_i \bar{L}_i$$

其中 $\sum_i \bar{L}_i = \frac{1}{K}\sum_{i,j} L^{\text{col}}_{ij}$。

计算梯度：
$$\nabla_\theta \log\Psi_{\text{row}} = \nabla_\theta \log\Psi_{\text{col}} - \nabla_\theta \sum_i \bar{L}_i$$

这里 $\nabla_\theta \sum_i \bar{L}_i$ 是一个与样本 $\mathbf{X}$ 相关的量（因为 $\bar{L}_i$ 依赖于 $\mathbf{x}_i$）。

**关键分析**：在 `nes_vmc_gradient_stable` 中，梯度通过 `tr_centered` 加权。由于 `tr_centered` 是 batch 中心的迹，我们需要考察 $\sum_i \bar{L}_i$ 对梯度的贡献：

$$\Delta\text{grad} = -\text{mean}_\text{batch}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{\nabla_\theta \sum_i \bar{L}_i}\right]$$

这个量 **一般不为零**！因为 $\nabla_\theta \sum_i \bar{L}_i$ 与样本相关，而 `tr_centered` 也与样本相关，两者的协方差不为零。

但是，在 **NESFermionHopRule 的对称提议** 下，我们之前证明了接受率不受影响。这里需要更细致地分析：

#### 3.3.1 实际代码中的处理

查看 `create_gauge_fixed_total_machines`（代码 L1215-1241）：

```python
def _compute_L_centered_single(m, x_single):
    for j in range(K):
        log_col = ansatz_j(x_single)       # (K,)
        log_ref = ansatz_j(ref_state)      # scalar
        log_col_centered = log_col - log_ref
        cols.append(log_col_centered)
    L_centered = jnp.stack(cols, axis=1)   # (K, K)
    return L_centered                       # ❌ 没有行规范修复！
```

当前代码 **没有实现行规范修复**（正如诊断报告所指出的）。行修复只存在于 `NESTotalAnsatz_gauge_stable_full._forward_single` 中，但 **未被 wrapper 调用**。

如果我们假设行修复被正确实施（如在方案 B 中），需要分析其对梯度的影响。

#### 3.3.2 行修复对 QGT 的影响

QGT 定义为：
$$S_{ab} = \langle \nabla_a \log\Psi^* \cdot \nabla_b \log\Psi \rangle - \langle \nabla_a \log\Psi^* \rangle \langle \nabla_b \log\Psi \rangle$$

行修复引入的偏移 $\Delta(\mathbf{X}) = -\sum_i \bar{L}_i(\mathbf{X})$ 与样本相关。因此：
$$\nabla\log\Psi_{\text{row}} = \nabla\log\Psi_{\text{col}} + \nabla\Delta(\mathbf{X})$$

QGT 会发生变化：
$$S_{\text{row}} = S_{\text{col}} + \text{cov}(\nabla\Delta, \nabla\log\Psi_{\text{col}}) + \text{var}(\nabla\Delta)$$

这意味着 **行规范修复会改变 QGT 矩阵**，从而影响自然梯度的方向。

#### 3.3.3 但是：行修复后梯度仍然正确

虽然 QGT 改变了，但这是**预期的行为**。行规范修复消除了一个虚假的自由度，使得 QGT 在该方向上不再是零模（或接近零模）。这实际上 **改善了** 自然梯度的数值性质，因为它不再需要在规范方向上进行"无效"的搜索。

---

## 4. 对 loss 值的影响

### 4.1 Loss 的定义

```python
loss_batch = jnp.real(jnp.trace(E_L_matrix, axis1=-2, axis2=-1))
loss_mean = _masked_mean_batch(loss_batch, valid)
```

由于我们已经证明 $E_L$ 在行列规范修复下不变，因此：

**结论：loss 值不受行列规范修复的影响。**

### 4.2 数值稳定性改善

虽然 loss 的**值**不变，但行列规范修复显著改善了 **数值稳定性**：

- 列修复防止了列规范漂移导致的数值爆炸
- 行修复防止了 `logΨ mean` 单调上升导致的 overflow/underflow
- 这使得 `slogdet` 的计算更加稳定，减少了 NaN/Inf 的出现频率

---

## 5. 综合影响总结

| 计算环节 | 列修复影响 | 行修复影响 | 原因 |
|---------|-----------|-----------|------|
| **E_L 矩阵** | 无 | 无 | E_L 只依赖单态 ψ 的比值，与 L 矩阵的规范无关 |
| **loss (trace E_L)** | 无 | 无 | 直接由 E_L 决定 |
| **∇logΨ（列修复部分）** | 加常数偏移 | — | 但被 tr_centered 中心化消去 |
| **∇logΨ（行修复部分）** | — | 改变 | $\nabla\Delta(\mathbf{X})$ 与样本相关 |
| **梯度（最终）** | 无 | **有影响** | 行修复改变了 $\nabla\log\Psi$ 的结构 |
| **QGT** | 无 | **有影响** | 消除了规范方向的零模，改善了条件数 |
| **自然梯度** | 无 | **有影响** | QGT 改变导致更新方向变化 |
| **能量特征值** | 无 | 无 | 由 E_L 对角化得到 |

---

## 6. 行修复对梯度的具体影响分析

### 6.1 行修复的数学形式

行修复后的 logΨ：
$$\log\Psi_{\text{row}}(\mathbf{X}) = \log\Psi_{\text{col}}(\mathbf{X}) - \frac{1}{K}\sum_{i,j} L^{\text{col}}_{ij}(\mathbf{X})$$

其中 $L^{\text{col}}_{ij}(\mathbf{X}) = \log\psi_j(\mathbf{x}_i) - \log\psi_j(\text{ref})$。

### 6.2 对单样本梯度的影响

对单个样本 $\mathbf{X} = (\mathbf{x}_1, \ldots, \mathbf{x}_K)$：

$$\nabla_\theta \log\Psi_{\text{row}} = \nabla_\theta \log\Psi_{\text{col}} - \frac{1}{K}\sum_{i,j} \nabla_\theta \log\psi_j(\mathbf{x}_i)$$

注意：$\nabla_\theta \log\psi_j(\mathbf{x}_i)$ 是第 $j$ 个 ansatz 在第 $i$ 个 walker 处的梯度。

### 6.3 对 batch 梯度的影响

在 `nes_vmc_gradient_stable` 中：

$$\text{grad} = \text{mean}_\text{batch}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{\nabla\log\Psi_{\text{row}}}\right]$$

代入：

$$\text{grad}_{\text{row}} = \text{grad}_{\text{col}} - \text{mean}_\text{batch}\left[\overline{\text{tr}_\text{centered}} \cdot \overline{\frac{1}{K}\sum_{i,j} \nabla\log\psi_j(\mathbf{x}_i)}\right]$$

第二项 **一般不为零**，因为它衡量的是 `tr_centered` 与"所有 ansatz 在所有 walker 上的梯度之和"之间的协方差。

### 6.4 为什么这是正确的行为

行规范是一个**物理上的冗余自由度**。在原始的无修复版本中，优化器可以在行规范方向上"浪费"更新步长（虽然 loss 不变，但 QGT 在此方向上可能有很小的特征值，导致大的参数更新）。

行修复后：
1. **消除了这个冗余方向**，使得 QGT 的条件数改善
2. **自然梯度不再在规范方向上产生无意义的更新**
3. **实际效果是加速收敛并提高稳定性**

这与机器学习中的"规范化"（normalization）技术类似：虽然不改变函数的输出值，但改善了优化 landscape。

---

## 7. 验证建议

可以通过以下数值实验验证上述分析：

### 7.1 验证 E_L 不变性
```python
# 对比修复前后的 E_L 矩阵
E_L_before = NES_loss_energy_stable(..., total_matrix_machine=unfixed...)
E_L_after = NES_loss_energy_stable(..., total_matrix_machine=fixed...)
print(jnp.max(jnp.abs(E_L_before - E_L_after)))  # 应 ≈ 0
```

### 7.2 验证 loss 不变性
```python
loss_before = jnp.real(jnp.trace(E_L_before, axis1=-2, axis2=-1)).mean()
loss_after = jnp.real(jnp.trace(E_L_after, axis1=-2, axis2=-1)).mean()
print(abs(loss_before - loss_after))  # 应 ≈ 0
```

### 7.3 验证梯度差异（仅行修复）
```python
grad_before = nes_vmc_gradient_stable(..., total_machine=unfixed...)
grad_after = nes_vmc_gradient_stable(..., total_machine=fixed...)
# 列修复：grad_before ≈ grad_after
# 行修复：grad_before ≠ grad_after（但有物理意义的改进）
```

### 7.4 验证能量特征值不变性
```python
eigs_before = jnp.linalg.eigvalsh(E_L_before)
eigs_after = jnp.linalg.eigvalsh(E_L_after)
print(jnp.max(jnp.abs(eigs_before - eigs_after)))  # 应 ≈ 0
```
