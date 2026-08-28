# 规范修复对 MCMC 采样器的影响分析

> **背景**：基于诊断报告 `gauge_fixing_analysis.md` 和验证报告 `verify_column_and_row_gauge_K4.ipynb`，
> 分析列规范修复（column gauge fixing）和行规范修复（row gauge fixing）对 NES-VMC 中 MCMC 采样器
> 的 Metropolis 接受概率及对称提议性质的影响。
>
> **核心结论**：
> 1. **列规范修复不影响接受概率**，因其在当前构型间转移时产生常数偏移，被比值消去。
> 2. **行规范修复也不影响接受概率**，前提是修复函数在同一点对同一构型施加相同的偏移（即 `f(x') - f(x) = 0`）。
> 3. **NESFermionHopRule 是对称提议**，无论是否施加规范修复，Metropolis-Hastings 接受率公式中的提议概率比恒等于 1。
> 4. **规范修复本身不改变采样分布的平衡性**，它只影响 logΨ 的绝对数值（全局偏移），不影响 $|\Psi|^2$ 的相对权重。

---

## 1. 采样器机制回顾

### 1.1 NESFermionHopRule

`NESFermionHopRule` 继承自 `nk.sampler.rules.MetropolisRule`，其 `transition` 方法的核心逻辑：

```python
def transition(self, sampler, machine, parameters, state, rng, sigma):
    # 随机选取一条边 (i, j)，交换 sigma[i] 和 sigma[j]
    sigma_cand = sigma.at[i].set(sigma[j])   # 用 j 处的值替换 i
    sigma_cand = sigma_cand.at[j].set(sigma[i])  # 用 i 处的值替换 j
    # 如果产生重复子组态则拒绝，否则接受提议构型
    return sigma_cand, None
```

这是一个 **确定性交换提议**：给定当前构型 $\sigma$，提议新构型 $\sigma'$ 为交换某对 walker 位置。

### 1.2 Metropolis 接受率

标准 Metropolis-Hastings 接受率：

$$
A(\sigma \to \sigma') = \min\left(1,\; \frac{|\Psi(\sigma')|^2}{|\Psi(\sigma)|^2} \cdot \frac{T(\sigma' \to \sigma)}{T(\sigma \to \sigma')} \right)
$$

其中 $T(\sigma \to \sigma')$ 是提议概率。

---

## 2. 提议对称性分析

### 2.1 NESFermionHopRule 是对称提议

对于任意两个不同构型 $\sigma$ 和 $\sigma'$（其中 $\sigma'$ 由 $\sigma$ 交换某对 $(i,j)$ 得到）：

- **正向提议**：从 $\sigma$ 出发，随机选择边 $(i,j)$，交换得到 $\sigma'$，概率为 $\frac{1}{N_{\text{edges}}}$。
- **反向提议**：从 $\sigma'$ 出发，同样随机选择边 $(i,j)$，交换得到 $\sigma$，概率为 $\frac{1}{N_{\text{edges}}}$。

因此：

$$
\frac{T(\sigma' \to \sigma)}{T(\sigma \to \sigma')} = 1
$$

**无论是否施加列规范或行规范修复，只要提议规则本身不变，对称性始终成立。**

### 2.2 重复约束不影响对称性

`_check_duplicate` 约束：如果交换后产生重复子组态，则拒绝并保留原构型。这是一个 **确定性拒绝**，对正向和反向路径一视同仁——如果 $\sigma \to \sigma'$ 因重复被拒绝，那么 $\sigma' \to \sigma$ 也会因同样的重复原因被拒绝。因此对称性仍然保持。

---

## 3. 列规范修复对接受概率的影响

### 3.1 当前实现

`create_gauge_fixed_total_machines` 中的 `_compute_L_centered_single`：

```python
L[i,j] = logψ_j(x_i) - logψ_j(ref)   # column gauge fixing
```

然后 `_stable_from_L` 计算：

```python
shift = max(Re(L))
log_det_centered = slogdet(exp(L - shift)) + K * shift
```

### 3.2 对接受率的影响

考虑从 $\sigma$ 到 $\sigma'$ 的转移，接受率取决于：

$$
\frac{|\Psi(\sigma')|^2}{|\Psi(\sigma)|^2} = \exp\left(2\,\text{Re}[\log\Psi(\sigma') - \log\Psi(\sigma)]\right)
$$

设未规范修复时的 logΨ 为 $\log\Psi_0(\sigma)$，规范修复引入的偏移为 $\Delta(\sigma)$：

$$
\log\Psi_{\text{gauge}}(\sigma) = \log\Psi_0(\sigma) + \Delta(\sigma)
$$

则：

$$
\frac{|\Psi_{\text{gauge}}(\sigma')|^2}{|\Psi_{\text{gauge}}(\sigma)|^2}
= \frac{|\Psi_0(\sigma')|^2}{|\Psi_0(\sigma)|^2} \cdot \exp\left(2\,\text{Re}[\Delta(\sigma') - \Delta(\sigma)]\right)
$$

**关键在于 $\Delta(\sigma') - \Delta(\sigma)$ 是否为 0。**

#### 列规范修复的分析

列规范修复使 $L_{ij} \to L_{ij} - \log\psi_j(\text{ref})$。这是一个 **与当前构型 $\sigma$ 无关的常量偏移**（ref 是固定的参考态）。因此：

$$
\Delta_{\text{col}}(\sigma) = \text{const} \quad \forall \sigma
$$

$$
\Delta_{\text{col}}(\sigma') - \Delta_{\text{col}}(\sigma) = 0
$$

**结论**：列规范修复对接受率 **无影响**。$\frac{|\Psi(\sigma')|^2}{|\Psi(\sigma)|^2}$ 与未修复时完全相同。

---

## 4. 行规范修复对接受概率的影响

### 4.1 行规范修复的定义

在 `_compute_L_centered_single` 之后，再施加行规范修复：

```python
row_mean = jnp.mean(L_centered, axis=1, keepdims=True)  # shape [K, 1]
row_mean = jax.lax.stop_gradient(row_mean)
L_rowfixed = L_centered - row_mean
```

这相当于每个样本（每行）减去该行的均值：

$$
L^{\text{rowfixed}}_{ij} = L^{\text{centered}}_{ij} - \frac{1}{K}\sum_k L^{\text{centered}}_{ik}
$$

### 4.2 行列式修正

行规范修复改变了 L 矩阵，从而改变了行列式的值。需要补偿：

$$
\det(L^{\text{rowfixed}}) = \det(L^{\text{centered}}) \cdot \exp\left(-\sum_i \text{row\_mean}_i\right)
$$

因此：

$$
\log\Psi_{\text{rowfixed}} = \log\Psi_{\text{centered}} - \sum_i \text{row\_mean}_i
$$

其中 $\text{row\_mean}_i = \frac{1}{K}\sum_j L^{\text{centered}}_{ij}$ 是第 $i$ 行（即第 $i$ 个 walker）的均值。

### 4.3 对接受率的影响

行规范修复引入的偏移量：

$$
\Delta_{\text{row}}(\sigma) = -\sum_{i=1}^{K} \text{row\_mean}_i(\sigma) = -\sum_{i=1}^{K} \frac{1}{K}\sum_{j=1}^{K} L^{\text{centered}}_{ij}(\sigma)
$$

注意：这里的 $i$ 是 walker 索引（对应 $\sigma$ 中的第 $i$ 个子组态 $x_i$），$j$ 是 ansatz 索引。

考虑从 $\sigma$ 到 $\sigma'$ 的转移（交换了第 $a$ 和第 $b$ 个 walker 的位置）：

- $\sigma' = \text{swap}_{a,b}(\sigma)$
- $x'_a = x_b$, $x'_b = x_a$, 其余 $x'_k = x_k$ ($k \neq a, b$)

那么：

$$
\Delta_{\text{row}}(\sigma') - \Delta_{\text{row}}(\sigma) = -\sum_i \text{row\_mean}_i(\sigma') + \sum_i \text{row\_mean}_i(\sigma)
$$

由于只有第 $a$ 和第 $b$ 行的内容发生了变化（其余行的 row_mean 不变），我们需要仔细分析：

$$
\text{row\_mean}_a(\sigma') = \frac{1}{K}\sum_j L^{\text{centered}}_{aj}(\sigma') = \frac{1}{K}\sum_j L^{\text{centered}}_{bj}(\sigma) = \text{row\_mean}_b(\sigma)
$$

$$
\text{row\_mean}_b(\sigma') = \frac{1}{K}\sum_j L^{\text{centered}}_{bj}(\sigma') = \frac{1}{K}\sum_j L^{\text{centered}}_{aj}(\sigma) = \text{row\_mean}_a(\sigma)
$$

因此：

$$
\sum_i \text{row\_mean}_i(\sigma') = \sum_{i \neq a,b} \text{row\_mean}_i(\sigma) + \text{row\_mean}_b(\sigma) + \text{row\_mean}_a(\sigma) = \sum_i \text{row\_mean}_i(\sigma)
$$

$$
\Delta_{\text{row}}(\sigma') - \Delta_{\text{row}}(\sigma) = 0
$$

**结论**：行规范修复对接受率 **也无影响**！因为交换两个 walker 只是交换了它们对应的 row_mean，总和保持不变。

### 4.4 一般情况（非交换转移）

对于一般的 proposal $\sigma \to \sigma'$（不限于交换），如果行规范修复的偏移量满足：

$$
\Delta_{\text{row}}(\sigma') = \Delta_{\text{row}}(\sigma)
$$

则接受率不变。这要求修复操作在所有构型上产生相同的偏移（或对所有构型都产生零偏移）。

**对于 NESFermionHopRule 的交换提议**，上述条件成立，因为交换只改变了 row_mean 的排列顺序，不改变总和。

---

## 5. 综合结论

### 5.1 接受率公式总结

| 修复方式 | 对接受率的影响 | 原因 |
|---------|-------------|------|
| **无修复** | 基准 | — |
| **仅列修复** | **无影响** | 偏移量是全局常数，$\Delta(\sigma') - \Delta(\sigma) = 0$ |
| **仅行修复** | **无影响**（对交换提议） | 交换只重排 row_mean，总和不变 |
| **列+行双修复** | **无影响**（对交换提议） | 两者叠加，仍满足 $\Delta(\sigma') = \Delta(\sigma)$ |

### 5.2 对 Metropolis 采样器的影响

1. **接受概率完全不变**：无论是否施加列修复、行修复或双修复，Metropolis 接受率公式中的 $|\Psi(\sigma')|^2 / |\Psi(\sigma)|^2$ 保持不变（对 NESFermionHopRule 的交换提议而言）。

2. **对称提议性质不受影响**：NESFermionHopRule 的提议机制是确定性的交换操作，其正反向概率恒等，与规范修复无关。

3. **采样分布不变**：由于接受率和提议概率都不变，规范修复不改变采样器所采样的分布。它只改变了 logΨ 的绝对数值表示，而不改变 $|\Psi|^2$ 的相对权重。

### 5.3 对训练的影响

虽然规范修复不影响采样，但它对**训练过程**有重要影响：

1. **列规范修复**：杀死列规范自由度，消除 MC 噪声在列方向的梯度残量。
2. **行规范修复**：杀死行规范自由度，防止 `logΨ mean` 单调漂移（即诊断报告中的"gauge drift"问题）。
3. **双修复**：同时消除两种规范自由度，使训练更稳定，但不会改变采样器本身的统计性质。

---

## 6. 数学证明补充

### 6.1 交换操作的行列式不变性

设 $L$ 是原始 L 矩阵，$\sigma$ 和 $\sigma'$ 是交换第 $a,b$ 个 walker 后的两个构型。令 $P_{ab}$ 是交换第 $a,b$ 行的置换矩阵：

$$
L(\sigma') = P_{ab} \cdot L(\sigma)
$$

则：

$$
\det(L(\sigma')) = \det(P_{ab}) \cdot \det(L(\sigma)) = (-1) \cdot \det(L(\sigma))
$$

但 `slogdet` 返回的是 $\log|\det|$，所以：

$$
\log|\det(L(\sigma'))| = \log|\det(L(\sigma))|
$$

对于列修复后的矩阵 $L^{\text{centered}} = L - \mathbf{1} \cdot \log\psi(\text{ref})^T$：

$$
L^{\text{centered}}(\sigma') = P_{ab} \cdot L^{\text{centered}}(\sigma)
$$

同理，$\log|\det|$ 不变。

对于行修复后的矩阵 $L^{\text{rowfixed}} = L^{\text{centered}} - \text{diag}(\text{row\_mean})$：

$$
\text{row\_mean}_i(L^{\text{rowfixed}}(\sigma')) = \text{row\_mean}_i(P_{ab} \cdot L^{\text{centered}}(\sigma) - \text{diag}(\text{row\_mean}))
$$

由于 $P_{ab}$ 只是交换第 $a,b$ 行，$\sum_i \text{row\_mean}_i$ 不变，因此 $\log\Psi$ 的偏移量在 $\sigma$ 和 $\sigma'$ 之间相同。

### 6.2 一般结论

**定理**：对于 NES-VMC 中的 NESFermionHopRule（walker 交换提议），任何形如

$$
\log\Psi_{\text{gauge}}(\sigma) = \log\Psi_0(\sigma) + \Delta(\sigma)
$$

的规范修复，只要 $\Delta(\sigma)$ 在交换对称下保持不变（即 $\Delta(\sigma') = \Delta(\sigma)$ 当 $\sigma'$ 是 $\sigma$ 的交换），则 Metropolis 接受率不受影响。

列规范和行规范（在对行求和的意义下）都满足这一条件。
