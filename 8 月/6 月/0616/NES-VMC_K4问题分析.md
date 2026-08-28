# NES-VMC K=4 问题分析报告

**生成日期**: 2026-06-19
**分析对象**: 采样器系列-1-存档版K4测试.ipynb

---

## 1. 问题描述

### 问题一：联合采样 X = [x₁,x₂,x₃,x₄] 计算 log(Psi) 时出现行列式相同

**现象**: 在 K=4 时，不同的配置（configuration）产生完全相同的 log(Psi) 值，导致 Loss 几乎不变。

**观察到的证据** (notebook cell 59-60):
```
配置 (0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0) → log(Psi) = 0.5240766-2.3737399j
配置 (0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0) → log(Psi) = 0.5240766-2.3737399j
```

两个不同的配置产生了完全相同的 log(Psi)。

### 问题二：NESTotalAnsatz 输出值基本相同，仅虚数部分不同

**现象**: 不同配置产生的 log(Psi) 只有虚数部分（相位）不同，实部完全相同。

```
log(Psi) = 0.5240766-2.3737399j  (虚部: -2.374)
log(Psi) = 0.5240766+0.76785276j (虚部: +0.768)
```

---

## 2. 数学根源分析

### 2.1 NES Ansatz 的结构

NES-VMC 的核心 ansatz (`NESTotalAnsatz.__call__`) 构建如下矩阵 L:

```python
L[i,j] = single_ansatz_list[j](x_single[i])
```

其中:
- i = 0,...,K-1 表示 K 个子系统
- j = 0,...,K-1 表示 K 个单态 ansatz
- x_single[i] 是第 i 个子系统的量子态

最终波函数为:
```
Ψ(x) = det(exp(L))
log(Ψ) = log(det(exp(L))) = slogdet(exp(L))
```

### 2.2 问题一的原因：行置换对称性

查看 notebook cell 59-60 的矩阵输出:

**配置 0 的矩阵 L0:**
```
Row 0 = single_ansatz_output(x0) = [0.739-0.330j, 0.215-0.453j, -0.055+0.083j, 0.064+0.190j]
Row 1 = single_ansatz_output(x1) = [0.626+0.153j, 0.458+0.150j, -0.326-0.368j, 0.299-0.089j]
Row 2 = single_ansatz_output(x2) = [0.695+2.222j, 0.647+0.609j, -1.476-0.478j, -0.109-0.132j]
Row 3 = single_ansatz_output(x3) = [0.649+1.133j, 0.819+0.227j, -0.332+0.451j, -0.383+0.443j]
```

**配置 2 的矩阵 L2 (行顺序不同):**
```
Row 0 = single_ansatz_output(x1) = [0.626+0.153j, 0.458+0.150j, -0.326-0.368j, 0.299-0.089j]
Row 1 = single_ansatz_output(x0) = [0.739-0.330j, 0.215-0.453j, -0.055+0.083j, 0.064+0.190j]
Row 2 = single_ansatz_output(x2) = [0.695+2.222j, 0.647+0.609j, -1.476-0.478j, -0.109-0.132j]
Row 3 = single_ansatz_output(x3) = [0.649+1.133j, 0.819+0.227j, -0.332+0.451j, -0.383+0.443j]
```

**关键发现**: L2 是 L0 经过**行交换**得到的矩阵。

对于行列式:
```
det(L2) = det(L0) × sign(permutation)
```

因为 L0 和 L2 只相差一个置换矩阵 P (使得 L2 = P × L0)，而 det(P) = ±1。

### 2.3 问题二的原因：log(Psi) 的结构

```
log(Ψ) = log|det(exp(L))| + i × arg(det(exp(L)))
       = log|det|         + i × (arg(det) + 2πn)
       = log_abs_det       + i × arg_sign
```

- **实部** = log|det(exp(L))| = trace(L) = Σᵢ L[i,i]

  由于 trace 不随行置换改变（trace(P×L) = trace(L×P) = trace(L)），实部相同。

- **虚部** = arg(det(exp(L))) = 取决于 det 的相位

  由于 det(L2) = det(P×L0) = det(P)×det(L0) = ±det(L0)，相位可能改变符号。

---

## 3. 是否为 NES-VMC 的天然缺陷？

### 3.1 缺陷的本质

**是的，这是一个结构性的限制，而非实现 bug。**

NES-VMC 的核心思想是用 K 个副本的联合来表示激发态。当 K=4 且单个希尔伯特空间维度 D=4 时：

- 可能的行排列数: K! = 4! = 24
- 这意味着**最多只有 24 种不同的行置换模式**
- 每种模式产生的 log|det| 相同（因为 trace 不变）
- 只有相位可能改变

### 3.2 为什么 K=3 没有这个问题？

在 notebook 的 K=3 版本中，由于 K! = 6 < 系统可能的状态数，行置换的简并度相对较低。

当 K=4 时：
- 单系统状态数 = 4 (两个自旋轨道，各 2 个占据态)
- 组合状态数 = C(4,1)×C(4,1) = 16 (每个自旋各选1个粒子)
- 但 4 个副本取自同一个系统，实际上只有有限种"有效排列"

### 3.3 简并度分析

| K | K! (行置换数) | 单系统维度 | 简并风险 |
|---|---------------|-----------|---------|
| 2 | 2 | 4 | 低 |
| 3 | 6 | 4 | 中 |
| 4 | 24 | 4 | **高** |
| 5 | 120 | 4 | 中 (但计算成本高) |

---

## 4. 解决方案

### 方案一：增加希尔伯特空间维度（推荐）

使用更多的轨道或更复杂的系统，使单系统维度 >> K。

```python
# 分子轨道 NES
n_orbitals = 8  # 而不是 2
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(2,2),  # 两个粒子
)
```

### 方案二：引入非对称权重

修改 ansatz 结构，打破 L[i,j] = f_j(x_i) 的对称性：

```python
def __call__(self, x: jax.Array):
    x_single = x.reshape(self.K, self.n_spin)

    # 添加可学习的非对称权重
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            # 加入权重 w[i,j] 打破对称性
            w_ij = self.weight_matrix[i, j]  # 可学习参数
            L = L.at[i, j].set(
                w_ij * self.single_ansatz_list[j](x_single[i])
            )

    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
    log_Psi = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi, L
```

### 方案三：使用 Pfaffian 代替行列式

对于反对称矩阵，可以使用 Pfaffian 代替行列式：

```
Ψ(x) = Pf(A)  而不是 det(M)
```

Pfaffian 不具有行置换对称性（Pf(PAP^T) = det(P) × Pf(A)）。

### 方案四：引入子系统身份标识

给每个子系统添加一个身份 embedding，使 L[i,j] 不仅依赖于 x_i，还依赖于子系统索引 i：

```python
def __call__(self, x: jax.Array):
    x_single = x.reshape(self.K, self.n_spin)

    # 子系统身份 embedding
    subsystem_id = jnp.arange(self.K)  # [0, 1, 2, 3]

    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            # 将子系统身份作为额外输入
            x_with_id = jnp.concatenate([x_single[i], jnp.onehot(i, self.K)])
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_with_id)
            )
```

### 方案五：使用复数域的 multi-determinat 扩展

当前 ansatz 本质上是一个 single-determinant ansatz。可以扩展为:

```python
# 使用多个 determinant 的线性组合
log(Ψ) = log(Σ_c c_k × det_k(L))
```

每个 det_k 可以使用不同的 ansatz 网络或不同的子系统排列。

---

## 5. 梯度消失问题

Notebook 输出显示:
```
grad norm before = 0.0000
grad norm after = 0.0000
```

这是因为:
1. **Loss 几乎恒定**: 由于 det 相同，local energy 相同
2. **梯度 = 0**: dE_L/dparams ≈ 0，因为 E_L 本身就是常数

这不是梯度消失的技术问题（如 vanishing gradient in RNN），而是**问题结构导致 Loss landscape 是平坦的**。

---

## 6. 推荐行动方案

1. **短期**: 增加单系统维度（如使用更多轨道）
2. **中期**: 实现方案二或方案四，打破置换对称性
3. **长期**: 考虑使用 Pfaffian 或 multi-determinant 扩展

---

## 7. 总结

| 问题 | 原因 | 是否天然缺陷 | 严重程度 |
|-----|------|-------------|---------|
| 行列式相同 | 行置换不改变 det | **是** | 高 |
| log(Psi) 实部相同 | trace 置换不变 | **是** | 中 |
| 虚部仅相位不同 | det 相位可能变号 | **是** | 中 |
| 梯度趋于零 | Loss landscape 平坦 | **是** | 高 |

NES-VMC 在 K=4, D=4（系统维度等于副本数）时确实存在结构性限制。建议优先通过增加系统复杂度或引入非对称权重来缓解此问题。
