# NES-VMC 代码解读（采样器系列-1-存档版K3）

## 1. 研究背景与目标

**目标**：基于 NetKet 框架和 Flax.nnx API，复现 **NES-VMC（Natural Excited State Variational Monte Carlo）算法**，用于计算量子多体系统（H₂ 分子）的前 $K$ 个激发态能量。

**核心思想**：将原系统前 $K$ 个激发态的求解问题，等价转化为扩展希尔伯特空间中"扩展系统"的基态求解问题。

---

## 2. H₂ 分子定义与 FCI 基准

Notebook 第一个 cell 定义了 H₂ 分子和基准能量：

```python
# H₂ 分子定义（键长 1.4 Å）
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
```

**输出**：
```
E0 = -1.01546825 Ha  |  激发能：0.0000 eV
E1 = -0.87542794 Ha  |  激发能：3.8107 eV
E2 = -0.42938376 Ha  |  激发能：15.9482 eV
E3 = -0.26922131 Ha  |  激发能：20.3064 eV
```

---

## 3. 希尔伯特空间定义

```python
# 单系统希尔伯特空间：2 个轨道，自旋 1/2，每个自旋通道 1 个电子
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

K = 3  # NES 扩展副本数（本文 K=3，可计算前3个激发态）
hi_ext = hi ** K  # 扩展希尔伯特空间 = K 个原系统副本的张量积
SINGLE_SIZE = hi.size  # 单子系统维度 = 4
```

对于 H₂ 分子（STO-3G 基组，自旋守恒），单系统有 4 种合法组态：
- `[1, 0, 1, 0]`（α₁占据, β₁占据）
- `[0, 1, 0, 1]`（α₂占据, β₂占据）
- `[1, 0, 0, 1]`（α₁占据, β₂占据）
- `[0, 1, 1, 0]`（α₂占据, β₁占据）

扩展希尔伯特空间 `hi_ext = hi ** K` 的维度为 $4^K$，但由于 NES-VMC 的行列式约束，实际合法构型数为 $4^K - 4^{K-1}$（禁止重复组态）。

---

## 4. 哈密顿量定义

```python
# 使用 NetKet 实验性 API 从 PySCF 分子构建哈密顿算符
ha = nkx.operator.from_pyscf_molecule(mol)
```

---

## 5. Ansatz 架构

### 5.1 SingleStateAnsatz（单态 Ansatz）

```python
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)  # 输出 ln(ψ(x))
```

- **输入**：单粒子组态 `x`（形状 `(n_spin,)` = `(4,)`）
- **输出**：复数值 `ln(ψ(x))`
- **作用**：近似单个电子波函数 $\psi_i(x)$

### 5.2 NESTotalAnsatz（总 Ansatz / 行列式 Ansatz）

```python
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states           # 扩展副本数
        self.n_spin = n_spin_orbitals

        # 创建 K 个独立的 SingleStateAnsatz
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # x_single: (K, n_spin) 的组态矩阵
            x_single = x_single.reshape(self.K, self.n_spin)

            # 构建 L_ij = ln ψ_j(x^i) 矩阵
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # 计算行列式：Ψ(x) = det(exp(L)) = det(M)
            Psi_matrix = jnp.exp(L)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_matrix)
            log_Psi = log_abs_det + 1j * jnp.angle(sign)

            return log_Psi, L  # 返回 ln Ψ 和 中间矩阵 L

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

### 5.3 NESTotalAnsatz 的输出详解

`NESTotalAnsatz` 的前向传播返回两个值：**`log_Psi`** 和 **`L`矩阵**。

#### 5.3.1 中间矩阵 L 的构造

对于一个扩展组态 $\mathbf{x} = (x^1, x^2, \ldots, x^K)$，中间矩阵 $L$ 的定义为：

$$
L_{ij} = \ln \psi_j(x^i)
$$

即矩阵的第 $i$ 行第 $j$ 列元素，是第 $j$ 个单态 Ansatz 在第 $i$ 个组态上的对数振幅。

**举例（K=3）**：

假设我们有一个扩展组态，包含 3 个子组态：
- $x^1 = [1, 0, 1, 0]$（组态1）
- $x^2 = [0, 1, 0, 1]$（组态2）
- $x^3 = [1, 0, 0, 1]$（组态3）

我们有 3 个单态 Ansatz：$\psi_1, \psi_2, \psi_3$

计算 $L$ 矩阵：

$$
L = \begin{pmatrix}
\ln \psi_1(x^1) & \ln \psi_2(x^1) & \ln \psi_3(x^1) \\
\ln \psi_1(x^2) & \ln \psi_2(x^2) & \ln \psi_3(x^2) \\
\ln \psi_1(x^3) & \ln \psi_2(x^3) & \ln \psi_3(x^3)
\end{pmatrix}
$$

代码实现：

```python
L = jnp.zeros((self.K, self.K), dtype=complex)
for i in range(self.K):
    for j in range(self.K):
        L = L.at[i, j].set(
            self.single_ansatz_list[j](x_single[i])  # ψ_j(x^i)
        )
```

#### 5.3.2 M 矩阵（未取对数）

$$
M_{ij} = \exp(L_{ij}) = \psi_j(x^i)
$$

即：

$$
M = \begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \psi_3(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \psi_3(x^2) \\
\psi_1(x^3) & \psi_2(x^3) & \psi_3(x^3)
\end{pmatrix}
$$

注意这里和通常的 Slater 行列式不同：矩阵的**行**索引组态，**列**索引波函数。

#### 5.3.3 总 Ansatz：行列式

$$
\Psi(\mathbf{x}) = \det(M) = \det\begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \psi_3(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \psi_3(x^2) \\
\psi_1(x^3) & \psi_2(x^3) & \psi_3(x^3)
\end{pmatrix}
$$

#### 5.3.4 log_Psi 的计算

```python
Psi_matrix = jnp.exp(L)
sign, log_abs_det = jnp.linalg.slogdet(Psi_matrix)
log_Psi = log_abs_det + 1j * jnp.angle(sign)
```

这里使用了 `slogdet`（log-determinant）来保证数值稳定性：
- `log_abs_det = ln|det(Psi_matrix)|`
- `angle(sign)` 是复数行列式的相位

因此：
$$
\ln \Psi(\mathbf{x}) = \ln |\det(M)| + i \cdot \arg(\det(M))
$$

#### 5.3.5 返回值总结

| 返回值 | 形状 | 数学含义 |
|--------|------|----------|
| `log_Psi` | 标量（复数） | $\ln \Psi(\mathbf{x}) = \ln \det(M)$ |
| `L` | (K, K) | $L_{ij} = \ln \psi_j(x^i)$ |

**为什么需要返回两个值？**

1. `log_Psi`：用于采样器的接受率计算（Metropolis 准则需要 $\ln \Psi$）
2. `L`：用于损失函数 `NES_loss_energy` 中构建 $M$ 矩阵并求逆

#### 5.3.6 行列式为零的物理意义

如果扩展组态中存在两个相同的子组态，即 $x^i = x^j$（$i \neq j$），则矩阵 $M$ 的两行相同，行列式为零。这对应物理上的"交换对称性"问题。因此 NES-VMC 采样必须**禁止重复组态**。

### 5.4 Machine 函数包装器

```python
def create_machine(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_psi_total  # 只返回 log_Psi

    return machine, graphdef, state

def create_single_machine(model: SingleStateAnsatz):
    """单态 machine 包装器"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    """返回中间矩阵 L（用于损失函数计算）"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_M_matrix  # 返回 L = ln M 矩阵

    return machine, graphdef, state
```

三个包装器的区别：

| 函数 | 返回值 | 用途 |
|------|--------|------|
| `create_machine` | `log_Psi` | 采样器（Metropolis 接受率） |
| `create_machine_matrix` | `L` | 损失函数计算 |
| `create_single_machine` | `ln ψ(x)` | 单态能量计算 |

---

## 6. 哈密顿量作用函数

### 6.1 Ham_psi：作用在单态上

```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    计算 Hψ(x) = Σ_{x'} ⟨x|H|x'⟩ ψ(x')

    支持单个态 (n_spin,) 和批量态 (batch_size, n_spin)
    """
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)  # 获取连接态和矩阵元
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    H_psi_batch = jax.vmap(_single_hpsi)(x)

    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
```

**功能**：对单态波函数 $\psi_j(x^i)$ 作用哈密顿算符，得到 $\hat{H}\psi_j(x^i)$

### 6.2 Ham_Psi：作用在总 Ansatz 上

```python
def Ham_Psi(ha, single_machine_list, total_params, x):
    """
    计算 HΨ(x) 矩阵：
    [Hψ₁(x¹)  Hψ₂(x¹)  ...  Hψ_K(x¹)]
    [Hψ₁(x²)  Hψ₂(x²)  ...  Hψ_K(x²)]
    [  ...      ...    ...    ...  ]
    [Hψ₁(x^K)  Hψ₂(x^K) ...  Hψ_K(x^K)]
    """
    K = len(single_machine_list)

    if x.ndim == 2:
        # 输入: (K, n_spin) → 单个扩展态 → 返回 (K, K)
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
        # 输入: (batch, K, n_spin) → 批量 → 返回 (batch, K, K)
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

**功能**：构建 $\hat{H}\Psi(\mathbf{x})$ 矩阵，其元素为：

$$
[\hat{H}\Psi(\mathbf{x})]_{ij} = \hat{H}\psi_j(x^i)
$$

---

## 7. 损失函数

### 7.1 NES_loss_energy

```python
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    """
    计算局域能量矩阵 E_L(x) = Ψ⁻¹(x) H Ψ(x)
    损失函数 = Tr(E_L(x)) = Tr(Ψ⁻¹ H Ψ)
    """
    log_M = total_matrix_machine(total_params, x)  # L = ln M 矩阵
    Psi_Matrix = jnp.exp(log_M)                      # M = exp(L)

    # 计算 HΨ
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)

    # E_L = M⁻¹ HΨ = Ψ⁻¹ H Ψ
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)

    # 返回迹（标量损失）和矩阵（用于梯度计算）
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

**数学含义**：

1. **矩阵 M**（来自 `total_matrix_machine`）：
$$
M_{ij} = \psi_j(x^i)
$$

2. **矩阵 $\hat{H}\Psi$**（来自 `Ham_Psi`）：
$$
[\hat{H}\Psi]_{ij} = \hat{H}\psi_j(x^i)
$$

3. **局域能量矩阵**：
$$
E_L(\mathbf{x}) = M^{-1} \cdot (\hat{H}\Psi) = \Psi^{-1} \hat{H} \Psi
$$

4. **损失函数**（局域能量的迹）：
$$
\mathcal{L}(\mathbf{x}) = \mathrm{Tr}[E_L(\mathbf{x})] = \sum_{i=1}^{K} [E_L]_{ii}
$$

展开来写：
$$
\mathcal{L}(\mathbf{x}) = \sum_{i=1}^{K} \sum_{j=1}^{K} [M^{-1}]_{ij} [\hat{H}\Psi]_{ji}
$$

---

## 8. 梯度计算

```python
def nes_vmc_gradient(ha, total_matrix_machine, total_machine, single_machine_list, total_params, x_batch):
    # 1. 计算批量局域能量矩阵
    loss_batch, E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)

    # 2. 中心化
    E_L_centered = E_L_batch - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    # 3. 计算 ∇logΨ
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

**梯度公式推导**：

损失函数是局域能量矩阵的迹：
$$
\mathcal{L} = \mathrm{Tr}(E_L) = \mathrm{Tr}(M^{-1} \hat{H}\Psi)
$$

对参数 $\theta$ 求导（使用复数全纯梯度）：

$$
\frac{\partial \mathcal{L}}{\partial \theta} = 2 \cdot \mathbb{E}_{\mathbf{x} \sim |\Psi|^2} \left[ \mathrm{Tr}\left( (E_L - \bar{E}_L) \cdot \frac{\partial \ln M}{\partial \theta} \right) \right]
$$

其中：
- $E_L - \bar{E}_L$ 是中心化的局域能量矩阵
- $\frac{\partial \ln M}{\partial \theta}$ 是波函数矩阵对参数的对数梯度
- 迹运算 $\mathrm{Tr}$ 对应代码中的 `jnp.trace(...)`

---

## 9. NetKet 自定义采样器（重点）

### 9.1 采样器设计背景

NES-VMC 的采样要求：
1. 在扩展希尔伯特空间 $\mathbf{x} = (x^1, x^2, \ldots, x^K)$ 上采样
2. **关键约束**：$x^i \neq x^j$（当 $i \neq j$），否则行列式为零

### 9.2 NESFermionHopRule 自定义跃迁规则

Notebook 中 Cell 2 定义了自定义采样规则：

```python
@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray
    K: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        """
        NES 约束：检查 K 个子组态是否有重复
        返回标量布尔值（适配 JAX while_loop）
        """
        sub = sigma_ext.reshape((-1, self.K, self.single_size))
        # 检查是否存在任意两个子组态完全相同
        return jnp.any(
            jnp.all(sub[..., 1:, :] == sub[..., 0:1, :], axis=-1), axis=-1
        ).squeeze()

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """跃迁规则：费米子跳跃 + NES 约束检查"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        # 随机选择跃迁边
        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:, 0], sel_e[:, 1]

        # 执行交换
        sigma_cand = sigma.at[jnp.arange(batch_size), i].set(sigma[jnp.arange(batch_size), j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size), j].set(sigma[jnp.arange(batch_size), i])

        # NES 约束：拒绝导致重复组态的跃迁
        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """生成随机初始态（确保不重复）"""
        sigma_shape = state.σ.shape
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100

            def cond(c):
                return (c[0] < max_tries) & c[2]

            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)
                s = hilbert.random_state(k_new)
                is_dup = self._check_duplicate(s)
                return (tries + 1, k, is_dup, s)

            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)

        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)
```

**三个核心方法**：

1. **`_check_duplicate`**：检查扩展组态中是否有重复子组态
   - 将 `(K * single_size,)` 的一维数组 reshape 为 `(K, single_size)`
   - 比较第 0 个子组态与第 1, 2, ..., K-1 个子组态是否相同
   - 若存在相同，返回 `True`（该组态非法）

2. **`transition`**：Metropolis 跃迁规则
   - 随机选择一条跃迁边
   - 交换边两端格点的占据数（费米子跳跃）
   - 若跃迁后产生重复组态，则拒绝

3. **`random_state`**：生成随机初始态
   - 使用 `jax.lax.while_loop` 循环生成
   - 若生成的组态有重复，重新生成
   - 最多尝试 100 次

### 9.3 扩展希尔伯特空间的跃迁边构造

Notebook 中 Cell 3 构造了扩展空间的跃迁边：

```python
SINGLE_SIZE = hi.size  # = 4

# 单系统的费米子跃迁边
single_edges = ((0, 1), (2, 3))  # α轨道跳跃、β轨道跳跃

# 构造 K 个副本的跃迁边
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE  # 每个副本的偏移量
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)

print(ext_edges)
# 对于 K=3：
# [[ 0  1]
#  [ 2  3]
#  [ 4  5]
#  [ 6  7]
#  [ 8  9]
#  [10 11]]
```

**物理含义**：

单系统（K=1）的跃迁边：
- 边 `(0, 1)`：α轨道电子在轨道0和轨道1之间跳跃
- 边 `(2, 3)`：β轨道电子在轨道0和轨道1之间跳跃

扩展系统（K=3）的跃迁边：
- 副本0的边：`(0,1)`, `(2,3)`
- 副本1的边：`(4,5)`, `(6,7)`
- 副本2的边：`(8,9)`, `(10,11)`

每个副本独立进行费米子跳跃，但**禁止跨副本跳跃**。

### 9.4 采样器初始化与采样

```python
# 创建自定义采样器
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20,
)

# 初始化采样器状态
sampler_state = nes_sampler.init_state(total_machine, total_params, seed=1)

# 采样
samples_raw, sampler_state = nes_sampler.sample(
    total_machine, total_params, state=sampler_state, chain_length=40
)
samples_raw.shape  # (16, 40, 12) = (n_chains, chain_length, K * SINGLE_SIZE)
```

---

## 10. 训练循环

```python
N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER = 400
Natural_Grad = False  # 或 True（使用自然梯度）

# 优化器
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# 初始化采样器
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN
    )

    # 2. 维度重塑
    samples = samples_raw.reshape(-1, hi_ext.size)  # (-1, K*4)
    x_batch = samples.reshape(-1, K, 4)             # (-1, K, 4)

    # 3. 计算梯度
    grad, loss_mean, E_L_mean = nes_vmc_gradient(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_machine=total_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_batch
    )

    # 4. （可选）自然梯度
    if Natural_Grad:
        grad_flat, grad_unravel_fn = ravel_pytree(grad)
        qgt_reg, _ = compute_qgt(total_machine, total_params, x_batch, diag_shift=0.1)
        natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
        grad = grad_unravel_fn(natural_grad_flat)

    # 5. 参数更新
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    # 6. 记录与监控
    log_Psi_batch = total_machine(total_params, x_batch)
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)

    if step % 50 == 0:
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        print(f"grad norm = {jnp.linalg.norm(ravel_pytree(grad)[0]):.4f}")
        print(f"Step {step:3d} | Loss: {loss_mean} | 0st能量={eig_vals[0]:.8f} Ha | 1st能量={eig_vals[1]:.8f} Ha | 2st能量={eig_vals[2]:.8f} Ha")
```

---

## 11. 激发态能量提取

训练完成后，通过对局域能量矩阵进行对角化提取激发态能量：

```python
# 在训练循环中
eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
# eig_vals 包含按能量排序的本征值：E₀ ≤ E₁ ≤ E₂ ≤ ...
```

**数学原理**：

平均局域能量矩阵：
$$
\bar{E}_L = \mathbb{E}_{\mathbf{x} \sim \Psi^2}[E_L(\mathbf{x})]
$$

对角化：
$$
\bar{E}_L = U \Lambda U^{-1}, \quad \Lambda = \mathrm{diag}(E_0, E_1, \ldots, E_{K-1})
$$

对角线元素即为各激发态能量。

---

## 12. 完整数据流图

```
┌─────────────────────────────────────────────────────────────────┐
│                        H₂ 分子定义                              │
│  geometry, basis='STO-3G', bond_length=1.4                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FCI 基准能量计算                              │
│  E_fcis = [-1.015, -0.875, -0.429, -0.269] Ha                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                 希尔伯特空间定义 (NetKet)                        │
│  hi = SpinOrbitalFermions(n_orbitals=2, s=1/2)                 │
│  K = 3, hi_ext = hi ** K                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              Ansatz 初始化 (Flax.nnx)                           │
│  NESTotalAnsatz: K 个 SingleStateAnsatz                        │
│  create_machine() → total_machine, total_graphdef, params     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           NetKet 自定义采样器 (NESFermionHopRule)                │
│  - edges: 扩展空间的费米子跃迁边                                 │
│  - _check_duplicate(): NES 约束（禁止重复组态）                 │
│  - Metropolis 采样 + 约束拒绝                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       MCMC 采样                                  │
│  nes_sampler.sample() → samples_raw: (n_chains, chain_len, 12) │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     梯度计算                                     │
│  nes_vmc_gradient():                                            │
│  1. NES_loss_energy() → E_L_batch                               │
│  2. 中心化 E_L                                                    │
│  3. 计算 ∇logΨ                                                    │
│  4. 加权平均 → grad                                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    参数更新 (SGD / Natural Grad)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   激发态能量提取                                  │
│  对 E_L_mean 对角化 → eig_vals = [E₀, E₁, E₂]                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 13. 关键函数一览表

| 函数名 | 文件来源 | 功能 |
|--------|----------|------|
| `SingleStateAnsatz` | NES_VMC.py | 单态神经网络 Ansatz |
| `NESTotalAnsatz` | NES_VMC.py | 行列式形式的总 Ansatz |
| `create_machine` | NES_VMC.py | 包装 total machine |
| `create_single_machine` | NES_VMC.py | 包装 single machine |
| `create_machine_matrix` | NES_VMC.py | 返回中间矩阵 L |
| `Ham_psi` | NES_VMC.py | H 作用在单态上 |
| `Ham_Psi` | NES_VMC.py | H 作用在总 Ansatz 上 |
| `NES_loss_energy` | NES_VMC.py | 局域能量矩阵和损失函数 |
| `nes_vmc_gradient` | NES_VMC.py | NES-VMC 梯度计算 |
| `NESFermionHopRule` | NES_VMC.py | NetKet 自定义采样规则 |
| `compute_qgt` | NES_VMC.py | 量子几何张量计算 |
| `sampler_info` | NES_VMC.py | 采样统计信息 |
