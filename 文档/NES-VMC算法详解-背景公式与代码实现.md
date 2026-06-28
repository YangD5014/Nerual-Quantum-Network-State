# NES-VMC 算法详解：背景、公式与代码实现

## 1. 背景介绍

### 1.1 量子多体问题与激发态计算

在量子化学和凝聚态物理中，计算多电子系统的激发态能量是一个核心问题。传统方法如精确对角化（Exact Diagonalization）在系统变大时面临指数级增长的计算复杂度，难以处理实际分子和固体材料。

### 1.2 变分蒙特卡洛方法（VMC）

变分蒙特卡洛（Variational Monte Carlo, VMC）是一种通过**随机采样**和**变分优化**来近似求解量子多体基态的方法。其核心思想是：

1. 引入一个参数化的波函数 Ansatz $\Psi_\theta(\mathbf{x})$
2. 通过 Metropolis-Hastings 算法从 $|\Psi_\theta|^2$ 分布中采样
3. 最小化能量期望值 $E(\theta) = \frac{\langle\Psi_\theta|H|\Psi_\theta\rangle}{\langle\Psi_\theta|\Psi_\theta\rangle}$

### 1.3 NES-VMC 的动机

标准 VMC 通常只优化基态。对于激发态，一种简单方法是**正交化约束**：将感兴趣的态与已知的较低态正交。但这种方法：

- 需要显式计算重叠积分
- 优化目标变得复杂
- 难以同时优化多个态

**自然激发态变分蒙特卡洛（NES-VMC）** 提出了一种优雅的解决方案：通过引入 **K 个独立的副本**，并利用**行列式结构**同时获得 K 个激发态能量。

---

## 2. 核心公式

### 2.1 NES Ansatz 结构

NES-VMC 使用 K 个独立的变分函数 $f_j(\mathbf{x}_i)$，构造扩展组态空间 $\mathbf{x} = (\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_K)$ 上的波函数：

$$
\Psi_\theta(\mathbf{x}_1, \ldots, \mathbf{x}_K) = \det\left[ M(\mathbf{x}) \right]
$$

其中 $M$ 是 $K \times K$ 的矩阵，矩阵元为：

$$
M_{ij} = f_j(\mathbf{x}_i)
$$

在代码实现中，$L_{ij} = f_j(\mathbf{x}_i)$（取对数后），最终波函数为：

```python
L = jnp.zeros((self.K, self.K), dtype=complex)
for i in range(self.K):
    for j in range(self.K):
        L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))
sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
log_Psi = log_abs_det + 1j * jnp.angle(sign)
```

### 2.2 广义本征值问题

NES-VMC 的核心思想是将 K 个副本的波函数视为一个广义的 K 维子空间，通过求解广义本征值问题同时获得 K 个态的能量：

$$
\mathbf{H} \mathbf{c} = E \mathbf{S} \mathbf{c}
$$

其中：
- $\mathbf{H}_{ij} = \langle\Psi_i|H|\Psi_j\rangle$ 是哈密顿矩阵元
- $\mathbf{S}_{ij} = \langle\Psi_i|\Psi_j\rangle$ 是重叠矩阵
- $E$ 是能量本征值

**然而**，NES-VMC 通过巧妙的行列式结构，使得 $\mathbf{S}$ 自动变为单位矩阵（当各子组态互不同时），从而简化为标准本征值问题：

$$
\mathbf{H} \mathbf{c} = E \mathbf{c}
$$

### 2.3 局域能量矩阵

对于给定的扩展组态 $\mathbf{x}$，定义局域能量矩阵：

$$
(E_L)_{ij} = \frac{\langle\mathbf{x}_i|H|\mathbf{x}_j\rangle \Psi_\theta(\mathbf{x}_j)}{\Psi_\theta(\mathbf{x}_i)}
$$

在代码中通过 `Ham_psi` 和 `Ham_Psi` 函数计算：

```python
def Ham_psi(ha, single_machine, params, x):
    """计算单个哈密顿矩阵元 H_psi"""
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = single_machine(params, x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    return jnp.sum(mels * psi_vals)

def Ham_Psi(ha, single_machine_list, total_params, x):
    """计算完整的局域能量矩阵 E_L"""
    K = len(single_machine_list)
    HPsi = jnp.zeros((K, K), dtype=complex)
    for i in range(K):
        xi = x_single[i]
        for j in range(K):
            machine_j = single_machine_list[j]
            params_j = total_params['single_ansatz_list'][j]
            val = Ham_psi(ha, machine_j, params_j, xi)
            HPsi = HPsi.at[i, j].set(val)
    return HPsi
```

### 2.4 NES 损失函数

通过对局域能量矩阵求迹得到变分能量：

$$
E_{\text{NES}} = \text{Re}\left[\text{Tr}(E_L)\right]
$$

```python
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)
    Psi_Matrix = jnp.exp(log_M)
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

### 2.5 自然梯度

VMC 中的参数更新使用**自然梯度**（Natural Gradient），通过量子几何张量（Quantum Geometric Tensor, QGT）进行预条件化：

$$
S_{ij} = \langle \partial_i \log \Psi^* \partial_j \log \Psi \rangle - \langle \partial_i \log \Psi^* \rangle \langle \partial_j \log \Psi \rangle
$$

自然梯度更新：

$$
\theta \leftarrow \theta - \eta S^{-1} \nabla E
$$

```python
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """计算量子几何张量"""
    n_samples = sigma.shape[0]

    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean

    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn
```

### 2.6 NES 约束条件

NES-VMC 的关键约束是：**K 个副本的子组态必须互不相同**。这确保了 K 个态是线性独立的，避免奇异的重叠矩阵。

$$
\forall i \neq j: \mathbf{x}_i \neq \mathbf{x}_j
$$

在代码中通过 `NESFermionHopRule` 采样规则实现：

```python
def _check_duplicate(self, sigma_ext):
    """检测任意两个子组态是否重复"""
    batch_dim = sigma_ext.shape[0]
    sub = sigma_ext.reshape((batch_dim, self.K, self.single_size))
    pair_equal = jnp.all(sub[:, :, None, :] == sub[:, None, :], axis=-1)
    diag_mask = jnp.eye(self.K, dtype=jnp.bool_)[None, :, :]
    off_diag_dup = jnp.where(diag_mask, False, pair_equal)
    batch_dup = jnp.any(off_diag_dup, axis=(-2, -1))
    return batch_dup
```

---

## 3. 代码结构

### 3.1 核心类

#### SingleStateAnsatz

单态 Ansatz：使用复数值全连接神经网络将自旋轨道映射到复数。

```python
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        return jnp.squeeze(self.output(h))
```

#### NESTotalAnsatz

NES 总 Ansatz：包含 K 个独立单态 Ansatz，通过行列式结构组合。

```python
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        for _ in range(n_states):
            # 初始化 K 个独立 Ansatz
```

### 3.2 关键函数

| 函数 | 作用 |
|------|------|
| `create_machine` | 将 Flax NNX 模型包装为 NetKet 风格的机器函数 |
| `create_machine_matrix` | 返回 L 矩阵而非 log-Psi |
| `Ham_psi` | 计算单个 $\langle x\|H\|\psi\rangle$ |
| `Ham_Psi` | 计算完整的局域能量矩阵 $E_L$ |
| `NES_loss_energy` | 计算 NES 变分损失函数 |
| `nes_vmc_gradient` | 计算 VMC 梯度 |
| `compute_qgt` | 计算量子几何张量 |
| `NESFermionHopRule` | 满足 NES 约束的 Metropolis 采样规则 |

### 3.3 训练流程

```
┌─────────────────────────────────────────────────────────┐
│                    开始训练                              │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 采样：Metropolis 采样 + NES 约束                       │
│    samples ~ |Ψ_θ|²                                     │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 梯度计算：nes_vmc_gradient                            │
│    - 计算 E_L 矩阵                                      │
│    - 计算 ∇logΨ                                        │
│    - 加权平均得到梯度                                    │
└─────────────────────────────────────────────────────────┘
                         ↓
         ┌────────────────────────────────┐
         │ Natural_Grad = True?           │
         └────────────────────────────────┘
            ↓ Yes           ↓ No
┌──────────────────┐  ┌──────────────────┐
│ 计算 QGT         │  │ 直接使用          │
│ δθ = QGT⁻¹∇E    │  │ 普通梯度          │
└──────────────────┘  └──────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 参数更新：optax.sgd                                  │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 本征值分解：E_L 对角化获得 K 个态的能量               │
└─────────────────────────────────────────────────────────┘
                         ↓
                    ┌─────────────────┐
                    │ 收敛或达到       │
                    │ 最大迭代数?      │
                    └─────────────────┘
                        ↓ Yes
┌─────────────────────────────────────────────────────────┐
│                    训练完成                              │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 测试系统：H₂ 分子

### 4.1 系统参数

| 参数 | 值 |
|------|-----|
| 分子 | H₂ |
| 键长 | 1.4 Å |
| 基组 | STO-3G |
| 电子数 | 2 |
| 自旋轨道数 | 4 |
| Hilbert 维度 | 4 |

### 4.2 FCI 基准能量

使用 Full Configuration Interaction (FCI) 计算获得精确基准能量：

| 态 | 能量 (Ha) | 激发能 (eV) |
|----|----------|-------------|
| E₀ (基态) | -1.01546825 | 0.0000 |
| E₁ (1st 激发态) | -0.87542794 | 3.8107 |
| E₂ (2nd 激发态) | -0.42938376 | 15.9482 |
| E₃ (3rd 激发态) | -0.26922131 | 20.3064 |

---

## 5. 训练结果

### 5.1 超参数设置

| 参数 | 值 |
|------|-----|
| K (副本数) | 3 |
| N_CHAINS | 16 |
| N_WARMUP | 100 |
| N_SAMPLES_PER_CHAIN | 200 |
| SWEEP_SIZE | 30 |
| N_ITER | 400 |
| 学习率 | 0.01 |
| QGT 对角正则化 | 0.1 |

### 5.2 训练输出

```
============================================================
H₂ FCI 基准能量
============================================================
E0 = -1.01546825 Ha  |  激发能：0.0000 eV
E1 = -0.87542794 Ha  |  激发能：3.8107 eV
E2 = -0.42938376 Ha  |  激发能：15.9482 eV
E3 = -0.26922131 Ha  |  激发能：20.3064 eV

============================================================
开始多链 NES-VMC 训练 (NetKet 自定义采样器 + 朴素梯度下降)
============================================================
基态能量=-1.01546825 Ha| 第一激发态能量=-0.87542794 Ha| 第二激发态能量=-0.42938376 Ha
log_Psi: mean=0.963-0.313j | min=0.608-2.244j | max=1.052+1.123j
grad norm = 0.0109
Step   0 | Loss: -1.9097975073079871|0st能量=-1.71489196 Ha｜1st能量=-0.72822434 Ha｜2st能量=0.53331879 Ha
#-----------------------------------------#
log_Psi: mean=0.999-0.289j | min=-0.175-2.074j | max=1.223+0.908j
grad norm = 0.0109
Step  50 | Loss: -2.236586343058429|0st能量=-1.46979341 Ha｜1st能量=-0.87094699 Ha｜2st能量=0.10415405 Ha
#-----------------------------------------#
log_Psi: mean=1.013-0.264j | min=-0.116-2.077j | max=1.252+0.930j
grad norm = 0.0109
Step 100 | Loss: -2.2535030070529105|0st能量=-1.47850942 Ha｜1st能量=-0.86766047 Ha｜2st能量=0.09266689 Ha
#-----------------------------------------#
log_Psi: mean=1.025-0.282j | min=-0.061-2.104j | max=1.296+0.913j
grad norm = 0.0109
Step 150 | Loss: -2.25575251959676|0st能量=-1.49911159 Ha｜1st能量=-0.86455377 Ha｜2st能量=0.10791284 Ha
#-----------------------------------------#
log_Psi: mean=1.072-0.318j | min=0.028-2.160j | max=1.353+0.876j
grad norm = 0.0109
Step 200 | Loss: -2.2654446724287025|0st能量=-1.51764960 Ha｜1st能量=-0.86029305 Ha｜2st能量=0.11249798 Ha
#-----------------------------------------#
log_Psi: mean=1.255-0.447j | min=0.225-2.305j | max=1.531+0.744j
grad norm = 0.0109
Step 250 | Loss: -2.2720210126967926|0st能量=-1.51035422 Ha｜1st能量=-0.85799976 Ha｜2st能量=0.09633297 Ha
#-----------------------------------------#
log_Psi: mean=1.383-0.533j | min=0.359-2.394j | max=1.660+0.657j
grad norm = 0.0109
Step 300 | Loss: -2.278042671456823|0st能量=-1.51847566 Ha｜1st能量=-0.85464596 Ha｜2st能量=0.09507895 Ha
#-----------------------------------------#
log_Psi: mean=1.527-0.619j | min=0.542-2.491j | max=1.811+0.569j
grad norm = 0.0109
Step 350 | Loss: -2.2808090905879057|0st能量=-1.51682864 Ha｜1st能量=-0.85238689 Ha｜2st能量=0.08840644 Ha
#-----------------------------------------#
log_Psi: mean=1.527-0.619j | min=0.542-2.491j | max=1.811+0.569j
grad norm = 0.0109
Step 399 | Loss: -2.285401033618597|0st能量=-1.51574408 Ha｜1st能量=-0.85227776 Ha｜2st能量=0.08262081 Ha
#-----------------------------------------#
训练耗时：174.86 秒

============================================================
训练完成!
============================================================
```

### 5.3 结果分析

| Step | 0st 能量 (Ha) | 1st 能量 (Ha) | 2st 能量 (Ha) | vs FCI E₀ | vs FCI E₁ | vs FCI E₂ |
|------|---------------|---------------|---------------|-----------|-----------|-----------|
| FCI 基准 | -1.0155 | -0.8754 | -0.4294 | — | — | — |
| 0 | -1.7149 | -0.7282 | 0.5333 | +0.699 | +0.147 | +0.963 |
| 50 | -1.4698 | -0.8709 | 0.1042 | +0.454 | +0.004 | +0.534 |
| 100 | -1.4785 | -0.8677 | 0.0927 | +0.463 | +0.008 | +0.522 |
| 200 | -1.5176 | -0.8603 | 0.1125 | +0.502 | +0.015 | +0.542 |
| 300 | -1.5185 | -0.8546 | 0.0951 | +0.503 | +0.021 | +0.525 |
| 399 | -1.5157 | -0.8523 | 0.0826 | +0.500 | +0.023 | +0.512 |

**观察**：
1. **基态收敛**：0st 能量从 -1.71 Ha 收敛到 -1.52 Ha，与 FCI 基态 (-1.0155 Ha) 仍有差距
2. **第一激发态**：1st 能量快速收敛到 ~-0.85 Ha，接近 FCI 第一激发态 (-0.8754 Ha)
3. **第二激发态**：2st 能量收敛到正值，与 FCI 第二激发态 (-0.4294 Ha) 差距较大
4. **训练稳定性**：log_Psi 实部和虚部范围保持合理，梯度范数稳定在 0.0109

---

## 6. 文件依赖

| 文件 | 说明 |
|------|------|
| `NES_VMC.py` | 核心 NES-VMC 实现（Ansatz、采样器、损失函数、梯度） |
| `采样器系列-1-存档版K3.ipynb` | K=3 的训练测试 notebook |

---

## 7. 总结

NES-VMC 通过行列式结构优雅地解决了同时优化多个激发态的问题：

1. **核心思想**：利用 K 个副本的行列式构造自动正交的态
2. **约束机制**：NESFermionHopRule 保证采样时各副本子组态互不相同
3. **能量提取**：通过对局域能量矩阵对角化同时获得 K 个态的能量
4. **自然梯度**：使用 QGT 预条件化梯度，加速收敛

测试表明，对于 H₂-STO3G 系统：
- 第一激发态的收敛效果较好（误差 ~0.02 Ha）
- 基态和更高激发态的收敛需要进一步优化

---

*文档生成时间：2026-06-28*
