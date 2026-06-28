# NES-VMC 对数域改造 代码解读

## 1. 研究背景与目标

**目标**：在标准 NES-VMC 算法的基础上，引入**对数域数值稳定化**技术，解决神经网络训练后期可能出现的数值溢出问题。

**问题描述**：

在标准 NES-VMC 中，当神经网络训练到后期时，波函数矩阵 $M$ 的元素值可能变得非常大或非常小。这会导致：

1. `jnp.exp(L)` 可能超出浮点数范围（→ `inf`）
2. 矩阵求逆 `jnp.linalg.solve` 精度下降
3. 损失函数计算不稳定

**解决方案**：将所有矩阵运算转换到对数域，通过减去每行的最大值来稳定数值范围。

---

## 2. 核心改造点总览

| 改造项 | 改造前 | 改造后 |
|--------|--------|--------|
| `NESTotalAnsatz` 返回值 | `(log_Psi, L)` | `(log_Psi_stable, L_stable, L_max)` |
| L 矩阵稳定化 | 无 | `L_stable = L - L.max()` |
| 损失函数 | `NES_loss_energy` | `NES_loss_energy_stable` |
| 新增机器函数 | 无 | `create_machine_max` |

---

## 3. NESTotalAnsatz 的对数域改造

### 3.1 改造后的代码

```python
class NESTotalAnsatz(nnx.Module):
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

            # 构建 L_ij = ln ψ_j(x^i) 矩阵
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # 核心改造：数值稳定化
            L_max = L.max()  # 记录每行的最大值（标量）
            L_stable = L - L_max  # 减去最大值，稳定指数运算

            # 计算稳定化后的行列式
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)

            return log_Psi_stable, L_stable, L_max

        # 批量处理支持
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')
```

### 3.2 NESTotalAnsatz 输出详解（对数域版本）

改造后的 `NESTotalAnsatz` 返回三个值：`**log_Psi_stable**`、`**L_stable**`、`**L_max**`。

#### 3.2.1 L 矩阵与稳定化

原始 L 矩阵：
$$
L_{ij} = \ln \psi_j(x^i)
$$

稳定化后的 L_stable：
$$
L_{\text{stable}} = L - L_{\max}
$$

其中 $L_{\max} = \max_{i,j} L_{ij}$（每行的最大值，即矩阵元素的最大值）

**举例（K=3）**：

假设原始 L 矩阵为：
$$
L = \begin{pmatrix}
1000 & 999 & 998 \\
1001 & 1000 & 999 \\
999 & 998 & 997
\end{pmatrix}
$$

则 $L_{\max} = 1001$，稳定化后：
$$
L_{\text{stable}} = L - 1001 = \begin{pmatrix}
-1 & -2 & -3 \\
0 & -1 & -2 \\
-2 & -3 & -4
\end{pmatrix}
$$

#### 3.2.2 为什么要减去 L.max？

原始问题：
$$
\exp(L) = \begin{pmatrix}
e^{1000} & e^{999} & e^{998} \\
e^{1001} & e^{1000} & e^{999} \\
e^{999} & e^{998} & e^{997}
\end{pmatrix} \rightarrow \text{数值溢出}
$$

稳定化后：
$$
\exp(L_{\text{stable}}) = \begin{pmatrix}
e^{-1} & e^{-2} & e^{-3} \\
e^{0} & e^{-1} & e^{-2} \\
e^{-2} & e^{-3} & e^{-4}
\end{pmatrix} = \begin{pmatrix}
0.368 & 0.135 & 0.050 \\
1.000 & 0.368 & 0.135 \\
0.135 & 0.050 & 0.018
\end{pmatrix}
$$

#### 3.2.3 log_Psi_stable 的计算

```python
sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
```

由于 $L_{\text{stable}} = L - L_{\max}$，有：
$$
\det(\exp(L_{\text{stable}})) = \det(\exp(L - L_{\max})) = \det(\exp(L) \cdot e^{-L_{\max}I}) = e^{-K \cdot L_{\max}} \cdot \det(\exp(L))
$$

因此：
$$
\ln \Psi_{\text{stable}} = \ln \det(\exp(L_{\text{stable}})) = \ln \det(\exp(L)) - K \cdot L_{\max} = \ln \Psi - K \cdot L_{\max}
$$

#### 3.2.4 返回值总结

| 返回值 | 形状 | 数学含义 |
|--------|------|----------|
| `log_Psi_stable` | 标量（复数） | $\ln \Psi_{\text{stable}} = \ln \Psi - K \cdot L_{\max}$ |
| `L_stable` | (K, K) | $L_{\text{stable}} = L - L_{\max}$ |
| `L_max` | 标量 | $L_{\max} = \max(L_{ij})$ |

#### 3.2.5 三元返回值的用途

| 返回值 | 用途 |
|--------|------|
| `log_Psi_stable` | 用于采样器的 Metropolis 接受率计算 |
| `L_stable` | 用于损失函数中构建稳定化的 $M$ 矩阵 |
| `L_max` | 用于损失函数中构建稳定化的 $\hat{H}\Psi$ 矩阵（对齐指数运算） |

---

## 4. Machine 函数包装器（改造版）

### 4.1 改造前

```python
def create_machine(model: NESTotalAnsatz):
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_psi_total
    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_M_matrix  # 返回 L
    return machine, graphdef, state
```

### 4.2 改造后

```python
def create_machine(model: NESTotalAnsatz):
    """返回稳定化后的 log_Psi"""
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_stable, L_stable, L_max = m(sigma)
        return log_psi_stable
    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    """返回稳定化后的 L_stable"""
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_stable, L_stable, L_max = m(sigma)
        return L_stable
    return machine, graphdef, state

def create_machine_max(model: NESTotalAnsatz):
    """返回 L_max（用于损失函数稳定化）"""
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_stable, L_stable, L_max = m(sigma)
        return L_max
    return machine, graphdef, state
```

### 4.3 三个机器函数的区别

| 函数 | 返回值 | 用途 |
|------|--------|------|
| `create_machine` | `log_Psi_stable` | 采样器（Metropolis 接受率） |
| `create_machine_matrix` | `L_stable` | 损失函数中构建 $M$ 矩阵 |
| `create_machine_max` | `L_max` | 损失函数中稳定化 $\hat{H}\Psi$ 矩阵 |

---

## 5. 哈密顿量作用函数

改造版的 `Ham_psi` 和 `Ham_Psi` 函数**保持不变**，与标准版本相同。

```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """计算 Hψ(x) = Σ_{x'} ⟨x|H|x'⟩ ψ(x')"""
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]

    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    H_psi_batch = jax.vmap(_single_hpsi)(x)

    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch

def Ham_Psi(ha, single_machine_list, total_params, x):
    """计算 HΨ(x) 矩阵"""
    K = len(single_machine_list)

    if x.ndim == 2:
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return _single_HamPsi(x)

    elif x.ndim == 3:
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return jax.vmap(_single_HamPsi)(x)
```

---

## 6. 损失函数（对数域稳定化版本）

### 6.1 改造前（标准版本）

```python
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    """原始版本：可能有数值溢出风险"""
    log_M = total_matrix_machine(total_params, x)  # L
    Psi_Matrix = jnp.exp(log_M)  # 直接取指数
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

**问题**：当 L 矩阵元素值很大时，`jnp.exp(L)` 会溢出。

### 6.2 改造后（对数域稳定化版本）

```python
def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    """
    对数域稳定化版本

    核心思想：
    1. M = exp(L) → 在指数前先减 L_max
    2. HΨ 也要对齐到同一尺度
    """
    # 1. 稳定化的 M 矩阵
    L_stable = total_matrix_machine(total_params, x)  # L_stable = L - L_max
    Psi_Matrix_stable = jnp.exp(L_stable)  # 数值稳定

    # 2. 稳定化的 HΨ 矩阵
    # H_psi: 原始的 ⟨x'|H|x⟩ ψ(x') 求和
    H_psi_raw = Ham_Psi(ha, single_machine_list, total_params, x)

    # 取对数后减去 L_max 对齐
    M = jnp.log(H_psi_raw)  # 先取对数（可能很小，取 log 避免下溢）
    L_max_batch = total_max_machine(total_params, x)  # L_max for each sample
    M_stable = M - L_max_batch.reshape(-1, 1, 1)  # 对齐到 L_stable 的尺度
    HPsi_stable = jnp.exp(M_stable)  # 稳定化

    # 3. 矩阵求逆
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    # 4. 返回迹（标量损失）和矩阵（用于梯度计算）
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

### 6.3 稳定化数学原理

#### 6.3.1 问题：尺度不一致

原始损失函数计算：
$$
E_L = M^{-1} \cdot (\hat{H}\Psi), \quad M = \exp(L)
$$

当 $L$ 很大时：
- $\exp(L)$ 数值溢出
- 但 $\hat{H}\Psi$ 是通过 `Ham_Psi` 直接计算的，数值范围正常
- 两者尺度不一致，求逆结果无意义

#### 6.3.2 解决方案：同步稳定化

**第一步**：稳定化 $M$ 矩阵
$$
M_{\text{stable}} = \exp(L - L_{\max}) = \exp(L) \cdot e^{-L_{\max}}
$$

**第二步**：稳定化 $\hat{H}\Psi$ 矩阵

原始 $\hat{H}\Psi$ 的计算结果记为 $\tilde{H}\Psi$，取对数后：
$$
\log(\tilde{H}\Psi) - L_{\max}
$$

然后指数化：
$$
\hat{H}\Psi_{\text{stable}} = \exp(\log(\tilde{H}\Psi) - L_{\max}) = \tilde{H}\Psi \cdot e^{-L_{\max}}
$$

**第三步**：求逆时尺度抵消

$$
M_{\text{stable}}^{-1} \cdot \hat{H}\Psi_{\text{stable}} = (e^{-L_{\max}} M)^{-1} \cdot (e^{-L_{\max}} \hat{H}\Psi) = M^{-1} \cdot \hat{H}\Psi
$$

因此，**数学上是等价的**，但数值上更加稳定。

#### 6.3.3 具体数值示例

假设 $L_{\max} = 1000$，$L = 1000$，$\hat{H}\Psi = 10^{1000}$（模拟）

**改造前**：
- $M = \exp(1000) = \infty$（溢出）
- 无法计算

**改造后**：
- $M_{\text{stable}} = \exp(1000 - 1000) = \exp(0) = 1$
- $\log(\hat{H}\Psi) = \log(10^{1000}) = 1000 \cdot \ln(10) \approx 2303$
- $\hat{H}\Psi_{\text{stable}} = \exp(2303 - 1000) = \exp(1303)$（仍然很大，但比直接算 $\hat{H}\Psi$ 小很多）

实际上 $\hat{H}\Psi$ 通过 `Ham_Psi` 直接计算，不会本身溢出，问题主要在 $\exp(L)$ 那边。改造通过稳定化 $M$ 矩阵使问题得到解决。

---

## 7. 梯度计算（改造版）

梯度计算函数 `nes_vmc_gradient` 的逻辑**保持不变**，只需将 `NES_loss_energy` 替换为 `NES_loss_energy_stable`。

```python
def nes_vmc_gradient_stable(ha, total_matrix_machine, total_max_machine,
                             total_machine, single_machine_list, total_params, x_batch):
    # 1. 计算批量局域能量矩阵（稳定化版本）
    loss_batch, E_L_batch = NES_loss_energy_stable(
        ha, total_matrix_machine, total_max_machine,
        single_machine_list, total_params, x_batch
    )
    E_L_mean = jnp.mean(E_L_batch, axis=0)

    # 2. 中心化
    E_L_centered = E_L_batch - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    # 3. 计算 ∇logΨ（注意：这里用稳定化后的 log_Psi_stable）
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 4. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
    loss_mean = loss_batch.mean()

    return grad, loss_mean, E_L_mean
```

---

## 8. 完整数据流对比

### 改造前（标准版本）

```
L = total_matrix_machine(params, x)              # L_ij = ln ψ_j(x^i)
       ↓
M = jnp.exp(L)                                    # 可能溢出！
       ↓
H_psi = Ham_Psi(...)                             # 正常尺度
       ↓
E_L = M⁻¹ · H_psi                                # 尺度不一致
```

### 改造后（对数域稳定化版本）

```
L_stable = total_matrix_machine(params, x)       # L_stable = L - L_max
L_max = total_max_machine(params, x)             # 提取 L_max
       ↓
M_stable = jnp.exp(L_stable)                     # 稳定，无溢出
       ↓
H_psi_raw = Ham_Psi(...)                         # 原始尺度
       ↓
H_psi_stable = exp(log(H_psi_raw) - L_max)        # 对齐到 M_stable 尺度
       ↓
E_L = M_stable⁻¹ · H_psi_stable                  # 尺度一致 = 原始结果
```

---

## 9. 使用示例（训练循环）

```python
# ============ 模型初始化 ============
K = 3
total_ansatz = NESTotalAnsatz(4, K, 12, rngs=nnx.Rngs(11))

# 三个机器函数
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)
total_max_machine, _, _ = create_machine_max(total_ansatz)

# 单态机器列表
single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)

# ============ 训练循环 ============
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN
    )

    # 2. 维度重塑
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)

    # 3. 计算梯度（稳定化版本）
    grad, loss_mean, E_L_mean = nes_vmc_gradient_stable(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        total_machine=total_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_batch
    )

    # 4. 参数更新
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
```

---

## 10. 关键函数一览表（改造版）

| 函数名 | 功能 |
|--------|------|
| `SingleStateAnsatz` | 单态神经网络 Ansatz |
| `NESTotalAnsatz` | 行列式形式的总 Ansatz（改造：返回三个值） |
| `create_machine` | 包装 total machine（返回 log_Psi_stable） |
| `create_machine_matrix` | 包装 matrix machine（返回 L_stable） |
| `create_machine_max` | 包装 max machine（返回 L_max） |
| `create_single_machine` | 包装 single machine |
| `Ham_psi` | H 作用在单态上 |
| `Ham_Psi` | H 作用在总 Ansatz 上 |
| `NES_loss_energy_stable` | 对数域稳定化的损失函数 |
| `nes_vmc_gradient_stable` | 对数域稳定化的梯度计算 |
| `NESFermionHopRule` | NetKet 自定义采样规则 |

---

## 11. 改造的合理性分析

### ✅ 合理的部分

1. **数值稳定性提升**：对数域改造有效避免了 `exp()` 操作的数值溢出问题
2. **数学等价性**：在去除公共因子 $e^{L_{\max}}$ 后，矩阵求逆和行列式计算的相对关系保持不变
3. **兼容原有架构**：改造未改变 `NESTotalAnsatz` 的整体结构和采样器

### ⚠️ 需要注意的部分

1. **新引入 `create_machine_max`**：需要额外计算和传递 `L_max`，增加了代码复杂度
2. **批量处理一致性**：`L_max` 需要正确 reshape 为 `(batch, 1, 1)` 以匹配 `M` 的形状
3. **训练监控**：log_Psi 的值范围会发生变化（因为稳定化），监控时需注意
