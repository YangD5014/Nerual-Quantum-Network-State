# NES-VMC 数值稳定性与 Gauge Fixed 策略详解

## 第一部分：数值稳定性策略

### 1. 问题背景：数值溢出危机

#### 1.1 标准 NES-VMC 的数值风险

在标准 NES-VMC 算法中，当神经网络训练到后期时，波函数矩阵 $M$ 的对数元素 $L_{ij} = \ln \psi_j(x^i)$ 可能变得非常大：

```python
# 标准版本 NESTotalAnsatz
L = jnp.zeros((self.K, self.K), dtype=complex)
for i in range(self.K):
    for j in range(self.K):
        L = L.at[i, j].set(
            self.single_ansatz_list[j](x_single[i])  # 可能返回很大的值
        )

sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))  # 危险！
log_Psi = log_abs_det + 1j * jnp.angle(sign)
```

**当 $L_{ij} \sim 1000$ 时：**

$$
\exp(1000) \approx 10^{434} \gg \text{float32 的上限 } 3.4 \times 10^{38}
$$

这会导致：
1. `jnp.exp(L)` → `inf`（溢出）
2. 矩阵求逆精度丧失
3. 损失函数计算得到 `nan`
4. 训练崩溃

#### 1.2 训练日志中的崩溃前兆

从 notebook 日志可以清晰看到崩溃过程：

```text
[Step   0] Loss=1.264033    | 梯度正常
[Step  40] Loss=-11.694875  | 梯度开始增大
[Step  50] Loss=18356.453957 | E0=-6131922.36 ← 数值爆炸
[Step  60] Loss=-11345.018184 | E0=-1592834.65 ← 已崩溃
```

梯度范数从正常值（如 5.9）爆炸到 148369.93，增长了 25000 倍！

### 2. 解决方案：对数域数值稳定化

#### 2.1 核心思想

**不要直接计算 $\exp(L)$，而是先对 $L$ 进行稳定化处理。**

关键观察：
- $\det(\exp(L)) = \exp(\text{tr}(L))$
- 但 $\exp(L)$ 本身会溢出
- 解决方案：$\exp(L - L_{\max})$ 不会溢出，其中 $L_{\max} = \max_{i,j} L_{ij}$

#### 2.2 数学原理

设原始 $L$ 矩阵为：
$$
L = \begin{pmatrix} 1000 & 999 \\ 1001 & 1000 \end{pmatrix}
$$

**直接指数化（溢出）：**
$$
\exp(L) = \begin{pmatrix} e^{1000} & e^{999} \\ e^{1001} & e^{1000} \end{pmatrix} \rightarrow \text{inf}
$$

**稳定化（安全）：**
$$
L_{\max} = 1001, \quad L_{\text{stable}} = L - 1001 = \begin{pmatrix} -1 & -2 \\ 0 & -1 \end{pmatrix}
$$

$$
\exp(L_{\text{stable}}) = \begin{pmatrix} e^{-1} & e^{-2} \\ e^{0} & e^{-1} \end{pmatrix} = \begin{pmatrix} 0.368 & 0.135 \\ 1.0 & 0.368 \end{pmatrix}
$$

#### 2.3 行列式的尺度变换

原始行列式：
$$
\ln \det(\exp(L)) = \ln \det(\exp(L_{\text{stable}}) \cdot e^{L_{\max}I}) = \ln \det(\exp(L_{\text{stable}})) + K \cdot L_{\max}
$$

因此：
$$
\log\Psi_{\text{raw}} = \log\Psi_{\text{stable}} + K \cdot L_{\max}
$$

其中 $K$ 是态的数量（矩阵维度）。

### 3. 代码实现详解

#### 3.1 NESTotalAnsatz_stable：核心改造

```python
class NESTotalAnsatz_stable(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # ===== 核心稳定化改造 =====
            L_max = jnp.max(L)           # 全局最大值（标量）
            L_stable = L - L_max         # 稳定化后的 L

            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)

            return log_Psi_stable, L_stable, L_max
```

**返回值说明：**

| 返回值 | 形状 | 数学含义 |
| ------ | ---- | -------- |
| `log_Psi_stable` | 标量（复数） | $\ln\Psi - K \cdot L_{\max}$ |
| `L_stable` | (K, K) | $L - L_{\max}$ |
| `L_max` | 标量 | $\max_{i,j} L_{ij}$ |

#### 3.2 三个机器函数

```python
def create_machine_stable(model: NESTotalAnsatz_stable):
    """【采样器专用】还原真实的 logΨ"""
    graphdef, state = nnx.split(model)
    K = model.K

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        log_psi_raw = log_stable + K * L_max
        return log_psi_raw

    return machine, graphdef, state


def create_machine_matrix_stable(model: NESTotalAnsatz_stable):
    """【损失函数专用】返回稳定化的 L_stable"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_stable

    return machine, graphdef, state


def create_machine_max_stable(model: NESTotalAnsatz_stable):
    """【损失函数专用】返回每样本的 L_max"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_max

    return machine, graphdef, state
```

### 4. 损失函数与尺度对齐

#### 4.1 稳定化版本的解决方案

```python
def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    L_stable = total_matrix_machine(total_params, x)
    Psi_Matrix_stable = jnp.exp(L_stable)

    H_psi_raw = Ham_Psi(ha, single_machine_list, total_params, x)

    M = jnp.log(H_psi_raw)
    L_max_batch = total_max_machine(total_params, x)
    M_stable = M - L_max_batch.reshape(-1, 1, 1)
    HPsi_stable = jnp.exp(M_stable)

    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### 4.2 尺度对齐的数学解释

**目标**：计算 $E_L = M^{-1} \cdot (H\Psi)$

**稳定化过程**：
- $M_{\text{stable}} = \exp(L - L_{\max}) = \exp(L) \cdot e^{-L_{\max}}$
- $(H\Psi)_{\text{stable}} = (H\Psi) \cdot e^{-L_{\max}}$
- $M_{\text{stable}}^{-1} \cdot (H\Psi)_{\text{stable}} = M^{-1} \cdot H\Psi$

**结论**：数学上等价，数值上稳定！

### 5. 采样器与损失函数的分工

```text
┌─────────────────────────────────────────────────────────┐
│              采样器（Metropolis）                         │
│   需要：真实的 logΨ                                     │
│   → 使用 create_machine_stable                          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              损失函数（能量计算）                          │
│   需要：稳定化的矩阵运算                                  │
│   → 使用 create_machine_matrix_stable                   │
│   → 使用 create_machine_max_stable                      │
└─────────────────────────────────────────────────────────┘
```

---

## 第二部分：Gauge Fixed 策略

### 6. 什么是 Gauge（规范自由度）

#### 6.1 多链 ansatz 中的规范自由度

在 NES-VMC 中，多链 ansatz 通过 Gram 矩阵构造：

$$
\Psi(X) = \det\left[\psi_j(x^i)\right]_{i,j=1}^{K}
$$

对任意非零常数 $c_j$，把 $\psi_j \to c_j \cdot \psi_j$，则：

- $\log\psi_j$ 增加 $\log c_j$（一个**per-j 加性常数**）
- 矩阵第 $j$ 列整体乘以 $c_j$
- $\det\Psi$ 乘以 $\prod_j c_j = e^{\sum_j \log c_j}$
- $\log\det\Psi$ 增加 $\sum_j \log c_j$

这种"乘以常数"的自由度被称为 **gauge 自由度**。物理观测量（如能量）必须是 gauge-invariant 的，但 **ansatz 的内部表示不是**——它会因为参数的数值优化方向而漂移。

#### 6.2 gauge 漂移的危害

如果 ansatz 的参数 $\theta$ 沿着"给 $\log\psi_j$ 加常数 $c_j$"的方向漂移：

- **能量计算不受影响**（已经设计成 gauge-invariant）
- **但 $\log|\Psi(X)|$ 数值会持续增长**——影响监控、log、debug
- **更危险的是**：如果 `Ham_psi` 计算时存在数值放大的中间步骤，常数偏移会被放大并污染局部能量

#### 6.3 gauge 群结构

完整的 gauge 群是 $\text{GL}(K, \mathbb{C})$，其自由度数为 $2K^2$（实数意义下）。常用的分解方式：

| 自由度 | 数学形式 | 物理意义 |
| ------ | -------- | -------- |
| per-j 加性常数 | $L[i,j] \to L[i,j] + c_j$ | 第 $j$ 列整体平移 |
| per-i 加性常数 | $L[i,j] \to L[i,j] + d_i$ | 第 $i$ 行整体平移 |
| per-(i,j) 任意复数 | $L[i,j] \to L[i,j] + \alpha_{ij}$ | 完全冗余 |

**reference gauge fix 只能钉死 per-j 加性常数 $c_j$ 这一种**。

### 7. Reference Gauge Fix：核心策略

#### 7.1 钉死 per-j 自由度的关键观察

如果我们能找到一个**固定的参考态** $|\text{ref}\rangle$，并且让

$$
L[i,j] = \log\psi_j(x^i) - \log\psi_j(\text{ref})
$$

那么对任意 per-j 常数 $c_j$：

$$
\log\psi_j(x^i) + c_j - (\log\psi_j(\text{ref}) + c_j) = \log\psi_j(x^i) - \log\psi_j(\text{ref})
$$

$c_j$ **自动消失**！

这是消除 per-j gauge 漂移的最直接方法。

#### 7.2 代码实现：gauge-fixed L 矩阵

```python
def _compute_L_centered_single(m, x_single):
    """
    计算 gauge-fixed 后的 L 矩阵
    L[i, j] = logψ_j(x_i) - logψ_j(ref)
    """
    cols = []
    for j in range(K):
        ansatz_j = m.single_ansatz_list[j]

        # 对当前 walker 的所有子构型计算 logψ_j
        log_col = ansatz_j(x_single)        # (K,)

        # 对参考态计算 logψ_j（ref）
        log_ref = ansatz_j(ref_state)        # scalar

        # 关键：减掉参考值，c_j 自动相消
        log_col_centered = log_col - log_ref
        cols.append(log_col_centered)

    L_centered = jnp.stack(cols, axis=1)  # (K, K)
    return L_centered
```

#### 7.3 完整流程：gauge-fix + 数值稳定

```python
def _stable_from_L(L):
    """
    接收 gauge-fixed 后的 L，做数值稳定，返回 log det
    """
    # 第一步：per-walker 减最大实部（数值稳定）
    shift = jnp.max(jnp.real(L))           # 标量
    shift = jax.lax.stop_gradient(shift)   # ← 关键：shift 是规范化量，不应回传梯度
    L_stable = L - shift                   # 元素 ≤ 0，安全

    # 第二步：slogdet（数值稳定）
    Psi_stable = jnp.exp(L_stable)
    sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
    log_det_stable = log_abs_det + 1j * jnp.angle(sign)

    # 第三步：恢复真实 log det（加上 K·shift）
    log_det_centered = log_det_stable + K * shift

    return log_det_centered, L_stable, shift
```

**三步合在一起的意义**：
- `L - logψ_j(ref)` → 消除 per-j gauge
- `L_stable = L - shift` → 消除 exp 溢出
- `log_det + K·shift` → 还原真实 log det
- 三者**正交不冲突**

### 8. 梯度在哪里停止（stop_gradient 的精确位置）

这是 gauge fix 中**最容易出错**的地方。错误的 stop_gradient 会导致：
- 训练不收敛
- 能量不下降
- 梯度消失或爆炸

#### 8.1 三种 stop_gradient 位置的对比

| 位置 | 代码 | 后果 |
|------|------|------|
| ① `shift` 标量 | `shift = stop_gradient(jnp.max(real(L)))` | ✓ **正确**：shift 是数据相关的归一化量，不能回传 |
| ② `logψ_j(ref)` 标量 | `log_ref = stop_gradient(ansatz_j(ref))` | ✗ **错误**：阻止 ref 处的参数更新，使 gauge 不彻底 |
| ③ `L_stable` 矩阵 | `L_stable = stop_gradient(L - shift)` | ✗ **严重错误**：完全切断梯度 |

#### 8.2 正确做法：只对 `shift` 加 stop_gradient

```python
# ===== 正确的 stop_gradient 用法 =====
shift = jnp.max(jnp.real(L))
shift = jax.lax.stop_gradient(shift)   # ✓ shift 看作常数

L_stable = L - shift                    # L 回传梯度，shift 看作常数
                                          # → ∂L_stable/∂θ = ∂L/∂θ

Psi_stable = jnp.exp(L_stable)          # 标准链式法则
sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)

log_det_stable = log_abs_det + 1j * jnp.angle(sign)
log_det_centered = log_det_stable + K * shift   # K·shift 在前向中加
                                                  # 但 ∂(K·shift)/∂θ = 0
                                                  # 所以梯度只来自 log_det_stable
```

**为什么 shift 必须是 stop_gradient？**

`shift` 是 per-walker 的统计量（`max(real(L))`），它**不应该是参数的函数**——它的作用是给当前样本"重新定标"，就像 batch norm 里的均值/方差是数据统计量一样。如果让它回传梯度：

1. 梯度会被 shift 项严重稀释（一个大常数除以另一个大常数）
2. 优化方向会试图"让 shift 变小"（即让 L 的最大值变小），这与能量优化目标**无关**
3. 训练会**卡在初始化附近不动**

#### 8.3 为什么 logψ_j(ref) **不能**加 stop_gradient

```python
# ===== 错误示例（常见错误）=====
def create_single_machine_gauge_fixed_BAD(single_model, ref_state):
    graphdef, state = nnx.split(single_model)
    ref_state = jnp.asarray(ref_state)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_raw = m(sigma)
        log_ref = m(ref_state)
        log_ref = jax.lax.stop_gradient(log_ref)   # ✗ 错误！
        return log_raw - log_ref

    return machine, graphdef, state
```

**为什么错误？**

设 ansatz 参数为 $\theta$，令 $f_j(x; \theta) = \log\psi_j(x; \theta)$，则：

$$
g_j(x) = f_j(x; \theta) - f_j(\text{ref}; \theta)
$$

正确的梯度：
$$
\frac{\partial g_j}{\partial \theta} = \frac{\partial f_j(x; \theta)}{\partial \theta} - \frac{\partial f_j(\text{ref}; \theta)}{\partial \theta}
$$

错误版本（log_ref 停了梯度）：
$$
\frac{\partial g_j^{\text{BAD}}}{\partial \theta} = \frac{\partial f_j(x; \theta)}{\partial \theta} - 0 = \frac{\partial f_j(x; \theta)}{\partial \theta}
$$

这两个不同！**前者**会沿着"使 $f_j(x)$ 与 $f_j(\text{ref})$ **同时变化**"的方向更新参数；**后者**只沿着 $f_j(x)$ 变化的方向更新，忽略了 ref 处的反馈。

数学上看，正确的梯度保持了**对 per-j 常数 $c_j$ 的不变性**——沿 $c_j$ 方向不产生梯度。错误的版本则破坏了这种不变性。

#### 8.4 完整 stop_gradient 流程图

```text
输入: L[i,j] = f_j(x_i; θ) - f_j(ref; θ)        ← 不加 stop_gradient
            ↓
        max(real(L)) = shift                      ← ✓ 加 stop_gradient
            ↓
        L_stable = L - shift                       ← L 部分回传梯度
            ↓
        exp(L_stable)                              ← 标准链式法则
            ↓
        slogdet(.)                                 ← 行列式
            ↓
        log_det_stable                             ← ✓ 无需 stop_gradient
            ↓
        log_det_centered = log_det_stable + K·shift ← shift 已 stop_gradient
            ↓
        返回给 netket / 优化器
```

### 9. 参考态的选择：为什么用 Hartree-Fock

#### 9.1 参考态需要满足的条件

gauge fix 依赖一个固定参考态 $\text{ref}$，它必须满足：

1. **确定性**：在训练过程中**永远不变**（不能是参数相关的）
2. **gauge 完整性**：必须覆盖 per-j 所有 $K$ 个 gauge 自由度——这意味着参考态是一个**单一固定的物理组态**
3. **计算友好**：$\log\psi_j(\text{ref})$ 必须**容易求值**（每次前向都要算）
4. **物理意义清晰**：选一个有明确物理含义的态，便于诊断

#### 9.2 Hartree-Fock 状态是最佳选择

在量子化学中，**Hartree-Fock（HF）态** $|$HF$\rangle$ 通常是首选：

| 条件 | HF 满足情况 |
|------|------------|
| 确定性 | ✓ 固定的 Slater 行列式，由分子轨道占据数决定 |
| gauge 完整性 | ✓ 是一个完整组态，不是叠加态 |
| 计算友好 | ✓ 只是 $\pm 1$ 的二进制向量，$\log\psi_j$ 容易求值 |
| 物理意义 | ✓ 零阶近似解，是化学直觉的起点 |

**具体到 LiH/STO-3G 例子**：

```python
# LiH 4e- CAS(4,4) 下的 Hartree-Fock 态
# 占据 HOMO-1, HOMO 两个 α 轨道 + 两个 β 轨道
Hatree_Fock = jnp.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=jnp.int8)
#         α: 站点 0 1 2 3   β: 站点 4 5 6 7
#         占据:           ↑↑           ↑↑
#         HOMO-1=2, HOMO=3
```

#### 9.3 HF 作为参考态的物理直觉

HF 态的"重要性"来自：

1. **大分量假设**：在大多数分子的基态波函数中，HF 占据的 Slater 行列式有**最大的 CI 系数**
   - 对 LiH 基态，HF 系数 ~ 0.95
   - 因此 $\log|\psi(\text{HF})|$ 在所有组态中相对较大
2. **能量低**：HF 是能量最低的单行列式，与真实基态接近
3. **正交归一基**：占据轨道彼此正交，$\langle \text{HF} | \text{HF} \rangle = 1$

**对 gauge fix 的影响**：

设 $L_{\text{centered}}[i,j] = \log\psi_j(x_i) - \log\psi_j(\text{HF})$

- 当 $x_i = \text{HF}$：$L_{\text{centered}}[i,j] = 0$，矩阵第 $i$ 行全 0
- 当 $x_i$ 是高激发组态：$L_{\text{centered}}[i,j] \ll 0$
- 所以 $L_{\text{centered}}$ 的"中心"在 0 附近，**避免极值**

#### 9.4 HF 参考态的潜在问题

虽然 HF 是常用选择，但也有边界情况需要注意：

| 问题 | 后果 | 缓解方法 |
| ---- | ---- | -------- |
| HF 在 Hilbert 空间的"角落" | 典型样本离 HF 远，$\|\log\psi_j(x) - \log\psi_j(\text{HF})\|$ 大 | 用多个参考态平均 |
| 强关联体系 | HF 系数小，不一定能钉住 | 用 CAS-SCF 或 MP2 自然轨道 |
| 简并态 | 多参考态方法更复杂 | 用多组态 gauge fix |

在 LiH 分子这种"温和的"分子中，HF 已经足够好。

### 10. Gauge-Fixed 完整机器函数

#### 10.1 三个机器函数（gauge-fixed 版本）

```python
def create_machine_gauge_stable(model):
    """【采样器/QGT/grad_logPsi 专用】返回真实 logΨ（gauge-fixed）"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)
        return log_Psi_gauge  # 已 gauge-fixed

    return machine, graphdef, state


def create_machine_matrix_gauge_stable(model):
    """【损失函数专用】返回 L_stable"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)
        return L_stable

    return machine, graphdef, state


def create_machine_max_gauge_stable(model):
    """【损失函数专用】返回 shift（用于 Ham_psi 尺度对齐）"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi_gauge, L_stable, shift, L_gauge = m(sigma)
        return shift

    return machine, graphdef, state
```

#### 10.2 包装成 `create_gauge_fixed_total_machines`

```python
def create_gauge_fixed_total_machines(total_model, ref_state):
    """
    一站式工厂函数
    返回: total_machine, total_matrix_machine, total_max_machine, graphdef, state
    """
    graphdef, state = nnx.split(total_model)
    K = total_model.K
    n_spin = total_model.n_spin
    ref_state = jnp.asarray(ref_state)

    # ... (内部实现：gauge fix + 数值稳定)

    return total_machine, total_matrix_machine, total_max_machine, graphdef, state
```

### 11. 采样器与损失函数的分工（gauge-fixed 版本）

```
┌─────────────────────────────────────────────────────────────┐
│              采样器（Metropolis）                              │
│   需要：gauge-fixed 后的 logΨ                                │
│   → 使用 total_machine                                       │
│   → logΨ = log_det_stable + K·shift                          │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              损失函数（能量计算）                               │
│   需要：                                                      │
│     (1) gauge-fixed 后的 L_stable                            │
│         → 使用 total_matrix_machine                           │
│     (2) shift（用于 Ham_psi 尺度对齐）                        │
│         → 使用 total_max_machine                              │
│   log det = log_det_stable + K·shift                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              QGT / 梯度                                       │
│   需要：∂logΨ/∂θ                                              │
│   → 使用 total_machine（gauge-fixed 后的）                    │
│   → 由于 K·shift 已 stop_gradient，∂logΨ/∂θ = ∂log_det_stable/∂θ │
│   → 这个梯度对 per-j gauge 自动不变                           │
└─────────────────────────────────────────────────────────────┘
```

### 12. Gauge Fix 的训练效果对比

#### 12.1 logΨ 漂移对比

| 步数 | 标准版 mean logΨ | Gauge-Fixed 版 mean logΨ | 差异 |
|------|------------------|--------------------------|------|
| 0 | 2.84 | 5.10 | +2.26（启动 baseline 抬高）|
| 5 | 7.09 | 11.70 | +4.61 |
| 10 | 12.58 | 18.42 | +5.84 |
| 15 | 17.01 | 23.32 | +6.31 |
| 19 | 19.99 | 25.97 | +5.98 |

**关键观察**：

- 两版都在漂移（说明 logΨ 漂移的根因不是 per-j gauge，见第 13 节）
- gauge-fix 版的 baseline 更高（因为减了 ref 后再加 K·shift）

#### 12.2 收敛质量对比

| 指标 | 标准版 | Gauge&nbsp;Fixed 版 | 结论 |
| ---- | ------ | ------------------ | ---- |
| E0 (final) | -7.7398 | -7.7506 | 几乎相同 |
| Loss (final) | -30.643 | -30.658 | 几乎相同 |
| cond(Ψ) 平均 | ~15 | ~25 | 略差 |
| 训练稳定度 | 高 | 高 | 相同 |

**结论**：gauge fix **不影响优化质量**，但提供更明确的 logΨ 物理含义（每个 ansatz 都相对 HF 校准）。

### 13. Gauge Fix 的边界与陷阱

#### 13.1 只能消除 per-j gauge

设 $L[i,j] \to L[i,j] + d_i$（per-walker 加性），ref fix **不**消除它：

$$
L_{\text{centered}}[i,j] = (f_j(x_i) + d_i) - (f_j(\text{ref}) + d_{\text{ref},j})
$$

$d_i$ 项和 $d_{\text{ref},j}$ 项不能相消（一个是 per-i，一个是 per-j）。

**后果**：logΨ 输出仍会随训练漂移（即使加了 reference fix），因为 ansatz 内部参数（如 `lin1, lin2, lin3` 的 bias）的更新会**间接**改变 $f_j(x)$ 对所有 $x$ 的值。

**缓解**：

- 加 weight decay（限制参数范数）
- 降低学习率（如 0.1 → 0.03）
- 监控 mean(logΨ) 漂移率

#### 13.2 Ham_psi 必须用 gauge-fixed 的 single machine

```python
# ===== 错误：用了原始 single_machine =====
def Ham_psi_BAD(ha, single_machine_raw, params, x):
    # ... single_machine_raw 返回的 logψ 包含 c_j
    # 多个 ansatz 的 c_j 不一致，会破坏 HPsi 矩阵的 gauge invariance
    ...

# ===== 正确：用 gauge-fixed 的 single_machine =====
single_machine_list = [
    create_single_machine_gauge_fixed(ansatz, ref_state)[0]
    for ansatz in total_ansatz.single_ansatz_list
]

def Ham_psi(ha, single_machine_gauge, params, x):
    # single_machine_gauge 返回 logψ_j(x) - logψ_j(ref)
    # 局部能量 E_L 是 c_j-invariant 的
    ...
```

#### 13.3 stop_gradient 的"传染性"

`stop_gradient(shift)` 的位置决定梯度的流向：

```python
# 假设 A = jax.lax.stop_gradient(B)
# 则 ∂A/∂B = 0，且 ∂(下游函数)/∂B 也不通过 A 传

# 这意味着：
# - shift 是数据归一化量，不能影响 θ 更新
# - L_stable = L - shift 中，shift 不传梯度，L 传梯度 → OK
# - 但 log_psi = log_det + K*shift 中，shift 也不传梯度 → OK
# 整体：∂log_psi/∂θ = ∂log_det/∂θ（只看 L 部分）
```

### 14. 数值稳定性 + Gauge Fix 完整代码模板

```python
import jax
import jax.numpy as jnp
from jax import lax
from functools import partial

# ====================== 1. 定义模型 ======================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin, hidden_dim, *, rngs):
        self.lin_out = nnx.Linear(hidden_dim, 1, param_dtype=complex, rngs=rngs)
        # ... 其他层
    def __call__(self, x):
        # 返回 logψ_j(x)
        return ...

class NESTotalAnsatz_gauge_stable(nnx.Module):
    def __init__(self, n_spin, n_states, hidden_dim, ref_state, *, rngs):
        self.K = n_states
        self.n_spin = n_spin
        self.ref_state = jnp.asarray(ref_state)
        self.single_ansatz_list = nnx.List([
            SingleStateAnsatz(n_spin, hidden_dim, rngs=nnx.Rngs(...))
            for _ in range(n_states)
        ])

    def __call__(self, x):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)

            # === Gauge Fix: 减掉 ref ===
            L = jnp.zeros((self.K, self.K), dtype=jnp.complex64)
            for j in range(self.K):
                ansatz_j = self.single_ansatz_list[j]
                log_col = ansatz_j(x_single)            # (K,)
                log_ref = ansatz_j(self.ref_state)      # scalar
                L = L.at[:, j].set(log_col - log_ref)   # 减 ref

            # === 数值稳定: 减 max ===
            shift = jnp.max(jnp.real(L))
            shift = lax.stop_gradient(shift)            # ← 关键 stop_gradient
            L_stable = L - shift

            # === slogdet ===
            Psi_stable = jnp.exp(L_stable)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)

            # === 还原真实 logΨ ===
            log_Psi_gauge = log_Psi_stable + self.K * shift

            return log_Psi_gauge, L_stable, shift, L

        # ... 批处理逻辑
        return _forward_single(x) if x.ndim == 1 else jax.vmap(_forward_single)(x)


# ====================== 2. 创建机器函数 ======================
total_ansatz = NESTotalAnsatz_gauge_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=hidden_dim,
    ref_state=Hatree_Fock,
    rngs=nnx.Rngs(11),
)

# 采样器用
total_machine, _, _, graphdef, params = create_gauge_fixed_total_machines(
    total_ansatz, Hatree_Fock
)

# 损失函数用
total_matrix_machine = create_machine_matrix_gauge_stable(total_ansatz)
total_max_machine = create_machine_max_gauge_stable(total_ansatz)

# Ham_psi 用（每个 ansatz 单独的 gauge-fixed）
single_machine_list = [
    create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
    for ansatz in total_ansatz.single_ansatz_list
]


# ====================== 3. 训练循环 ======================
for step in range(N_ITER):
    # 采样（用 gauge-fixed machine）
    samples, sampler_state = sampler.sample(
        machine=total_machine,
        parameters=params,
        state=sampler_state,
        ...
    )

    # 计算损失
    loss, E_L = NES_loss_energy_stable(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=params,
        x=x_batch
    )

    # 计算梯度
    grad = ...

    # QGT 预条件 + 更新
    ...
```

### 15. 调试与监控清单

训练时应该监控以下 gauge / 数值稳定相关指标：

| 指标 | 期望值 | 异常时排查 |
| ---- | ------ | ---------- |
| `logΨ` mean | 缓慢变化（不漂移） | 加 weight decay / 降 lr |
| `logΨ` min, max | 不应随 step 增长 | 同上 |
| `cond(Ψ)` | 10-100 | > 1000 时训练失败 |
| raw grad norm | < 1000（自然梯度会缩） | > 5000 时检查 ansatz |
| `f_j(HF)`（per-j） | 应在常数附近 | 若漂移 → ref 不是真 ref |
| `L_max`（per-walker） | 缓慢变化 | 若暴涨 → ansatz 输出过大 |
| `\|L_stable - L_stable_prev\|` | 收敛到 0 | 不收敛 → 训练失败 |

### 16. 关键 takeaway

1. **数值稳定**通过"减 max + slogdet + 还原"实现，**关键**是 `stop_gradient(shift)`。
2. **Gauge fix**通过"减 logψ_j(ref)"实现，**关键**是 `ref` 是固定 Hartree-Fock 态、且 `logψ_j(ref)` 不加 stop_gradient。
3. 两套机制**正交不冲突**：
   - 数值稳定解决"$\exp$ 溢出"
   - Gauge fix 解决"per-j 常数漂移"
4. 两者**不**能解决"per-walker 函数值漂移"——这需要 weight decay / 降 lr / ansatz 正则化。

---

## 附录：关键代码文件清单

| 文件 | 关键内容 |
|------|----------|
| [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6%20月/0625/LiH%20分子/NES_VMC.py) | 核心实现：SingleStateAnsatz / NESTotalAnsatz_stable / NESTotalAnsatz_gauge_stable / create_gauge_fixed_total_machines |
| [LiH 数值稳定+规范漂移抑制版 copy.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6%20月/0625/LiH%20分子/LiH%20数值稳定+规范漂移抑制版%20copy.ipynb) | LiH 分子 K=4 gauge-fixed 训练脚本 |
| [0706_规范漂移抑制失败原因调查.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6%20月/0625/LiH%20分子/0706_规范漂移抑制失败原因调查.md) | Gauge fix 失效根因分析（per-walker 漂移）|
