# NES-VMC 数值稳定性策略详解

## 1. 问题背景：数值溢出危机

### 1.1 标准 NES-VMC 的数值风险

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

### 1.2 训练日志中的崩溃前兆

从 notebook 日志可以清晰看到崩溃过程：

```
[Step   0] Loss=1.264033    | 梯度正常
[Step  40] Loss=-11.694875  | 梯度开始增大
[Step  50] Loss=18356.453957 | E0=-6131922.36 ← 数值爆炸
[Step  60] Loss=-11345.018184 | E0=-1592834.65 ← 已崩溃
```

梯度范数从正常值（如 5.9）爆炸到 148369.93，增长了 25000 倍！

---

## 2. 解决方案：对数域数值稳定化

### 2.1 核心思想

**不要直接计算 $\exp(L)$，而是先对 $L$ 进行稳定化处理。**

关键观察：
- $\det(\exp(L)) = \exp(\text{tr}(L))$
- 但 $\exp(L)$ 本身会溢出
- 解决方案：$\exp(L - L_{\max})$ 不会溢出，其中 $L_{\max} = \max_{i,j} L_{ij}$

### 2.2 数学原理

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

### 2.3 行列式的尺度变换

原始行列式：
$$
\ln \det(\exp(L)) = \ln \det(\exp(L_{\text{stable}}) \cdot e^{L_{\max}I}) = \ln \det(\exp(L_{\text{stable}})) + K \cdot L_{\max}
$$

因此：
$$
\log\Psi_{\text{raw}} = \log\Psi_{\text{stable}} + K \cdot L_{\max}
$$

其中 $K$ 是态的数量（矩阵维度）。

---

## 3. 代码实现详解

### 3.1 NESTotalAnsatz_stable：核心改造

```python
class NESTotalAnsatz_stable(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states           # 扩展态数量
        self.n_spin = n_spin_orbitals  # 单态轨道数

        # 创建 K 个独立的单态 Ansatz
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # x_single: (K, n_spin) - K 个样本，每个 n_spin 维
            x_single = x_single.reshape(self.K, self.n_spin)

            # 构建 L 矩阵：L[i,j] = ln ψ_j(x^i)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # ===== 核心稳定化改造 =====
            L_max = jnp.max(L)           # 全局最大值（标量）
            L_stable = L - L_max         # 稳定化后的 L

            # 使用 slogdet 计算行列式（数值稳定）
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)

            # 返回三个值而非两个
            return log_Psi_stable, L_stable, L_max
```

**返回值说明：**

| 返回值 | 形状 | 数学含义 |
|--------|------|----------|
| `log_Psi_stable` | 标量（复数） | $\ln\Psi - K \cdot L_{\max}$ |
| `L_stable` | (K, K) | $L - L_{\max}$ |
| `L_max` | 标量 | $\max_{i,j} L_{ij}$ |

### 3.2 批量处理逻辑

```python
# 安全的批量处理
if x.ndim == 2 and x.shape[-1] == self.n_spin:
    # 单个样本：(n_spin,) → 直接处理
    return _forward_single(x)

elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
    # 批量展平：(batch, K*n_spin) → reshape 后 vmap
    x = x.reshape(-1, self.K, self.n_spin)
    return jax.vmap(_forward_single)(x)

elif x.ndim == 3:
    # 已经是批量：(batch, K, n_spin) → 直接 vmap
    x = x.reshape(-1, self.K, self.n_spin)
    return jax.vmap(_forward_single)(x)

elif x.ndim == 1:
    # 单样本展平：(K*n_spin,) → reshape 后处理
    x = x[None, :]
    x = x.reshape(self.K, self.n_spin)
    return _forward_single(x)
```

### 3.3 三个机器函数

```python
def create_machine_stable(model: NESTotalAnsatz_stable):
    """
    【采样器专用】
    还原真实的 logΨ 用于 Metropolis 接受率计算
    """
    graphdef, state = nnx.split(model)
    K = model.K

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        # 关键：还原原始 logΨ
        # logΨ_raw = logΨ_stable + K * L_max
        log_psi_raw = log_stable + K * L_max
        return log_psi_raw

    return machine, graphdef, state

def create_machine_matrix_stable(model: NESTotalAnsatz_stable):
    """
    【损失函数专用】
    返回稳定化的 L_stable 用于构建 M 矩阵
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_stable

    return machine, graphdef, state

def create_machine_max_stable(model: NESTotalAnsatz_stable):
    """
    【损失函数专用】
    返回每样本的 L_max，用于 Ham*Psi 尺度对齐
    """
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_stable, L_stable, L_max = m(sigma)
        return L_max

    return machine, graphdef, state
```

**三个函数的用途区分：**

| 函数 | 返回值 | 使用场景 |
|------|--------|----------|
| `create_machine_stable` | `log_psi_raw` | Metropolis 采样器的接受率计算 |
| `create_machine_matrix_stable` | `L_stable` | 损失函数中构建 $M = \exp(L_{\text{stable}})$ |
| `create_machine_max_stable` | `L_max` | 损失函数中稳定化 $H\Psi$ 矩阵 |

---

## 4. 损失函数与尺度对齐

### 4.1 标准版本的问题

```python
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)  # L
    Psi_Matrix = jnp.exp(log_M)  # 危险！可能溢出！
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

**问题**：当 $L$ 很大时 $\exp(L) \to \text{inf}$，而 $H\Psi$ 是正常计算的，尺度不一致。

### 4.2 稳定化版本的解决方案

```python
def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    """
    对数域稳定化版本

    核心思想：
    1. M = exp(L) → 先减 L_max 再指数
    2. HΨ 也要对齐到同一尺度
    """
    # 1. 稳定化的 M 矩阵
    L_stable = total_matrix_machine(total_params, x)  # L_stable = L - L_max
    Psi_Matrix_stable = jnp.exp(L_stable)  # 数值稳定

    # 2. 稳定化的 HΨ 矩阵
    # H_psi_raw: 原始尺度下的 Hψ
    H_psi_raw = Ham_Psi(ha, single_machine_list, total_params, x)

    # 对齐到 L_stable 的尺度
    M = jnp.log(H_psi_raw)  # 先取对数
    L_max_batch = total_max_machine(total_params, x)  # 每样本的 L_max
    M_stable = M - L_max_batch.reshape(-1, 1, 1)  # 对齐
    HPsi_stable = jnp.exp(M_stable)  # 稳定化

    # 3. 矩阵求逆（尺度一致，结果等价于原始）
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    # 4. 返回迹和矩阵
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

### 4.3 尺度对齐的数学解释

**目标**：计算 $E_L = M^{-1} \cdot (H\Psi)$，其中 $M = \exp(L)$

**稳定化过程**：

第一步：稳定化 $M$ 矩阵
$$
M_{\text{stable}} = \exp(L - L_{\max}) = \exp(L) \cdot e^{-L_{\max}}
$$

第二步：稳定化 $H\Psi$ 矩阵
$$
(H\Psi)_{\text{stable}} = \exp(\ln(H\Psi) - L_{\max}) = (H\Psi) \cdot e^{-L_{\max}}
$$

第三步：求逆（尺度抵消）
$$
M_{\text{stable}}^{-1} \cdot (H\Psi)_{\text{stable}} = (e^{-L_{\max}} M)^{-1} \cdot (e^{-L_{\max}} H\Psi) = M^{-1} \cdot H\Psi
$$

**结论**：数学上等价，数值上稳定！

---

## 5. 采样器与损失函数的分工

### 5.1 为什么需要不同的机器函数？

```
┌─────────────────────────────────────────────────────────┐
│                     采样器（Metropolis）                  │
│                                                         │
│   需要：真实的 logΨ 用于接受率                            │
│   a = min(1, exp(logΨ_new - logΨ_old))                   │
│                                                         │
│   → 使用 create_machine_stable                          │
│   → 返回 log_psi_raw = log_stable + K * L_max           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                     损失函数（能量计算）                   │
│                                                         │
│   需要：稳定化的矩阵运算                                  │
│   E_L = trace(M_stable⁻¹ · HPsi_stable)                │
│                                                         │
│   → 使用 create_machine_matrix_stable                   │
│   → 返回 L_stable                                       │
└─────────────────────────────────────────────────────────┘
```

### 5.2 为什么不直接用 log_Psi_stable 做采样？

因为 Metropolis 接受率公式：
$$
a = \min\left(1, \frac{|\Psi_{\text{new}}|^2}{|\Psi_{\text{old}}|^2}\right) = \min\left(1, \exp(2 \cdot \text{Re}(\ln \Psi_{\text{new}} - \ln \Psi_{\text{old}}))\right)
$$

如果用 $\ln \Psi_{\text{stable}}$，会丢失 $K \cdot L_{\max}$ 项，导致：
- 不同样本的接受率计算出现系统偏差
- 采样分布偏离正确分布
- 训练收敛到错误的结果

---

## 6. Ham_psi 与 Ham_Psi：哈密顿量作用

### 6.1 单态哈密顿量作用

```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    计算 Hψ(x) = Σ_{x'} ⟨x|H|x'⟩ ψ(x')

    支持：单个态 (n_spin,) 或 批量态 (batch, n_spin)
    """
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)  # 邻居构型 + 矩阵元
        log_psi_vals = single_machine(params, x_primes)  # ln ψ(x')
        psi_vals = jnp.exp(log_psi_vals)  # ψ(x')
        return jnp.sum(mels * psi_vals)  # Σ ⟨x|H|x'⟩ ψ(x')

    H_psi_batch = jax.vmap(_single_hpsi)(x)

    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
```

**物理含义**：
- 对于输入构型 $x$，找出所有通过哈密顿量连接的近邻 $x'$
- 计算跃迁矩阵元 $\langle x | H | x' \rangle$
- 乘以对应波函数值 $\psi(x')$
- 求和得到 $H\psi(x)$

### 6.2 总Ansatz的哈密顿量作用

```python
def Ham_Psi(ha, single_machine_list, total_params, x):
    """
    计算 HΨ(x) 矩阵

    输入：x (K, n_spin) 或 (batch, K, n_spin)
    输出：HPsi (K, K) 或 (batch, K, K)

    其中 [HΨ]_{ij} = ⟨x^i| H | ψ_j⟩
    """
    K = len(single_machine_list)

    if x.ndim == 2:
        # 单个扩展态：(K, n_spin) → (K, K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # 第 i 个子构型
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 批量扩展态：(batch, K, n_spin) → (batch, K, K)
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

## 7. 梯度计算与自然梯度

### 7.1 原始变分梯度

```python
def nes_vmc_gradient(ha, total_matrix_machine, total_machine,
                    single_machine_list, total_params, x_batch):
    # 1. 批量局域能量矩阵
    loss_batch, E_L_batch = NES_loss_energy(
        ha, total_matrix_machine, single_machine_list, total_params, x_batch
    )
    E_L_mean = jnp.mean(E_L_batch, axis=0)  # (K, K)

    # 2. 中心化
    E_L_centered = E_L_batch - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)  # (batch,)

    # 3. 计算 ∇logΨ
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)  # PyTree of (batch, ...)

    # 4. 加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
    loss_mean = loss_batch.mean()

    return grad, loss_mean, E_L_mean
```

### 7.2 自然梯度预条件

```python
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）

    QGT 定义：S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩

    这对应费舍尔信息矩阵，是自然梯度的核心
    """
    n_samples = sigma.shape[0]

    # 步骤 1: 计算每个样本的 ∇log ψ
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: machine(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    # 步骤 2: 展平为矩阵 (n_samples, n_params)
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    # 步骤 3: 中心化
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean

    # 步骤 4: 计算 QGT
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered

    # 步骤 5: 正则化
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn
```

### 7.3 完整训练循环

```python
# 优化器：梯度裁剪 + SGD
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),  # 先裁剪
    optax.sgd(learning_rate=lr)            # 再更新
)
opt_state = optimizer.init(total_params)

for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=machine_wrapper,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 原始变分梯度
    grad_raw, loss_mean, E_L_mean = nes_vmc_gradient(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_machine=total_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_batch
    )

    # 3. 自然梯度预条件
    if Natural_Grad:
        qgt_reg_mat, _ = compute_qgt(
            total_machine, total_params, x_batch, diag_shift=qgt_diag_shift
        )
        grad_raw_flat = ravel_pytree(grad_raw)[0]
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
    else:
        grad_update = grad_raw

    # 4. 梯度裁剪 + 参数更新（optimizer 内部自动裁剪）
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
```

---

## 8. 完整数据流对比

### 8.1 标准版本（不稳定）

```
┌──────────────────────────────────────────────────────────────┐
│                        标准 NES-VMC                           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   L = total_matrix_machine(params, x)                        │
│       ↓                                                      │
│   M = jnp.exp(L)                              ← 可能溢出！    │
│       ↓                                                      │
│   H_psi = Ham_Psi(...)                      ← 正常尺度        │
│       ↓                                                      │
│   E_L = M⁻¹ · H_psi                          ← 尺度不一致！   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 8.2 稳定版本

```
┌──────────────────────────────────────────────────────────────┐
│                     对数域稳定化 NES-VMC                       │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│   L_stable = total_matrix_machine(params, x)  # L - L_max   │
│   L_max = total_max_machine(params, x)                       │
│       ↓                                                      │
│   M_stable = jnp.exp(L_stable)              ← 数值稳定       │
│       ↓                                                      │
│   H_psi_raw = Ham_Psi(...)                   ← 原始尺度      │
│       ↓                                                      │
│   HPsi_stable = exp(log(H_psi_raw) - L_max)  ← 对齐尺度      │
│       ↓                                                      │
│   E_L = M_stable⁻¹ · HPsi_stable             ← 尺度一致      │
│         = 原始的正确结果                                    │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## 9. 训练监控与异常检测

### 9.1 梯度爆炸的早期预警

```python
# 异常检测
has_nan_grad = jnp.any(jnp.isnan(grad_raw_flat))
grad_explode = grad_norm_raw > 5000.0  # 阈值

if has_nan_grad or grad_explode:
    logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan_grad}, raw_grad_norm={grad_norm_raw:.2f}")
```

### 9.2 波函数矩阵条件数监控

```python
# 计算波函数矩阵条件数
x_single = x_batch[0:1, ...]
psi_mat = total_matrix_machine(total_params, x_single)[0]
psi_cond = jnp.linalg.cond(psi_mat)

logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
```

**条件数意义**：
- $\cond(M) = \sigma_{\max} / \sigma_{\min}$
- 条件数越大，矩阵求逆越不稳定
- 训练后期条件数爆炸是崩溃的前兆

### 9.3 日志输出解读

```
[Step  40] logΨ: mean=4.791+1.406j | min=-0.636+1.230j | max=6.341+2.373j
梯度监控 | raw=64.3611 | natural=379.5282 | clipped=5.0000(上限5.0)
Ψ矩阵条件数 cond(Ψ) = 1.31e+01
Loss=-11.694875 | E0=-12.65665564 | E1=-7.26041931 | E2=8.22219949 | E3=8.22219949
```

**关键指标**：
- `logΨ` 范围：反映波函数的幅值
- `raw gradient`：QGT 前的原始梯度范数
- `natural gradient`：QGT 预条件后的梯度范数
- `clipped gradient`：梯度裁剪后的范数（$\le$ clip_norm）
- `cond(Ψ)`：矩阵条件数，越大越不稳定

---

## 10. 关键代码文件清单

| 文件 | 功能 |
|------|------|
| `NES_VMC.py` | 核心 NES-VMC 算法实现 |
| `h2-6-31G-K3-Stable版本.ipynb` | H₂ 分子的稳定版训练 notebook |

### 10.1 NES_VMC.py 核心类和函数

| 类/函数 | 说明 |
|---------|------|
| `SingleStateAnsatz` | 单态神经网络 Ansatz（两层隐层的 FFNN） |
| `NESTotalAnsatz` | 标准版总 Ansatz（返回 `log_Psi, L`） |
| `NESTotalAnsatz_stable` | 稳定版总 Ansatz（返回 `log_Psi_stable, L_stable, L_max`） |
| `create_machine` | 标准版机器函数包装器 |
| `create_machine_stable` | 稳定版采样器用机器函数（还原原始 logΨ） |
| `create_machine_matrix` | 标准版矩阵机器函数 |
| `create_machine_matrix_stable` | 稳定版矩阵机器函数（返回 L_stable） |
| `create_machine_max_stable` | 稳定版最大值机器函数（返回 L_max） |
| `create_single_machine` | 单态机器函数包装器 |
| `Ham_psi` | 单态哈密顿量作用 |
| `Ham_Psi` | 总 Ansatz 哈密顿量作用矩阵 |
| `NES_loss_energy` | 标准损失函数 |
| `NES_loss_energy_stable` | 稳定化损失函数 |
| `nes_vmc_gradient` | 变分梯度计算 |
| `compute_qgt` | 量子几何张量计算 |
| `NESFermionHopRule` | NES 约束的 Metropolis 跃迁规则 |

---

## 11. 数值稳定性策略总结

### 11.1 稳定化技术一览

| 技术 | 作用 | 实现位置 |
|------|------|----------|
| 对数域减最大值 | 防止 exp 溢出 | `NESTotalAnsatz_stable.__call__` |
| slogdet 行列式 | 稳定行列式计算 | `NESTotalAnsatz_stable.__call__` |
| HΨ 尺度对齐 | 保证矩阵求逆正确性 | `NES_loss_energy_stable` |
| 梯度裁剪 | 防止梯度爆炸 | 训练循环 |
| QGT 正则化 | 抑制自然梯度爆炸 | `compute_qgt` |
| 条件数监控 | 早期预警 | 训练循环 |

### 11.2 改造的合理性

✅ **数值稳定性提升**：有效避免 `exp()` 溢出
✅ **数学等价性**：稳定化操作在矩阵求逆时相互抵消
✅ **兼容原有架构**：采样器使用还原的原始 logΨ，不影响采样分布
⚠️ **额外复杂度**：引入了三个机器函数和 `L_max` 的传递

### 11.3 使用建议

1. **优先使用稳定版本**：所有新项目都应使用 `NESTotalAnsatz_stable`
2. **监控梯度范数**：设置合理的告警阈值（如 5000）
3. **监控条件数**：条件数 > 100 时应警惕
4. **合理设置超参**：
   - `clip_norm = 5.0`（梯度裁剪上限）
   - `qgt_diag_shift = 0.1`（QGT 正则化）
   - `lr = 0.01`（学习率）
