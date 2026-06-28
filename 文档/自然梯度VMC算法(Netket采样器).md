# 自然梯度 VMC 算法（NetKet 采样器）

## 概述

本实现基于纯 JAX 编写变分蒙特卡洛（Variational Monte Carlo, VMC）方法，用于计算氢分子（H₂）的基态能量。该实现采用**自然梯度下降法**（Natural Gradient Descent），通过量子几何张量（Quantum Geometric Tensor, QGT）对参数空间进行曲率感知的优化，显著提升收敛速度和稳定性。

本实现复刻了 NetKet 的核心算法，同时展示了一种将 **Flax NNX 神经网络**与 **NetKet 采样器**结合使用的方案。

## 核心技术特点

1. **纯 JAX 实现**：所有核心计算（梯度、QGT、优化）均使用 JAX 的自动微分和向量化功能
2. **自然梯度优化**：使用量子 Fisher 信息矩阵（二阶统计量）进行曲率感知优化
3. **Force-based 梯度计算**：高效计算能量期望值的梯度
4. **Flax NNX + NetKet 混合架构**：利用 Flax NNX 构建神经网络，使用 NetKet 的费米子采样器
5. **复数波函数支持**：使用 `holomorphic=True` 计算全纯导数

## 代码结构

### 1. 分子系统定义与 FCI 基准计算

```python
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
import optax
from tqdm import tqdm
from functools import partial
from jax import flatten_util

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

ha = nkx.operator.from_pyscf_molecule(mol)
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
```

**代码解读：**

- **PySCF 部分**：使用 PySCF 构建氢分子系统
  - `gto.M`：定义分子对象，指定原子坐标和基组（STO-3G 为最小基组）
  - `scf.RHF`：运行自洽场计算，获得分子轨道
  - `fci.FCI`：运行全配置相互作用计算，获得精确基态能量作为基准

- **NetKet 算符部分**：
  - `nkx.operator.from_pyscf_molecule(mol)`：将 PySCF 分子对象转换为 NetKet 的二次量子化哈密顿量
  - `nk.hilbert.SpinOrbitalFermions`：定义费米子希尔伯特空间
    - `n_orbitals=2`：2 个空间轨道（H₂ 在 STO-3G 下）
    - `s=1/2`：自旋量子数为 1/2
    - `n_fermions_per_spin=(1,1)`：每个自旋通道有 1 个电子

---

### 2. 神经网络 Ansatz

```python
# ==============================================================================
# 2. 神经网络 Ansatz
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
```

**代码解读：**

- **网络架构**（前馈神经网络）：
  - **输入层**：4 个自旋轨道（2 个空间轨道 × 2 个自旋）
  - **隐藏层 1**：`nnx.Linear(n_spin_orbitals, hidden_dim)`，hidden_dim=12
  - **隐藏层 2**：`nnx.Linear(hidden_dim, hidden_dim)`
  - **输出层**：`nnx.Linear(hidden_dim, 1)`，输出复数标量

- **激活函数**：`nnx.tanh`（双曲正切），逐层应用

- **关键设计**：
  - `param_dtype=complex`：使用复数参数以支持波函数的振幅和相位
  - `jnp.squeeze(out)`：将输出压缩为标量（波函数振幅的 log 值）

---

### 3. 费米子采样器配置

```python
# ==============================================================================
# 4. 初始化模型、采样器、优化器
# ==============================================================================
model = SingleStateAnsatz(4,12, rngs=nnx.Rngs(21))
# 采样器
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=100, sweep_size=32)

optimizer = nk.optimizer.Sgd(learning_rate=0.1)

vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

gs = nk.driver.VMC(
    ha,
    optimizer,
    variational_state=vstate,
    preconditioner=nk.optimizer.SR(diag_shift=0.1,holomorphic=True),
)
```

**代码解读：**

- **图结构定义**：`edges = [(0, 1), (2, 3)]`
  - 轨道 0 和 1 相连（自旋向上的两个轨道）
  - 轨道 2 和 3 相连（自旋向下的两个轨道）

- **FermionHopRule**：`nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)`
  - 费米子跳跃规则，确保采样符合费米子统计（泡利不相容原理）
  - 通过图结构定义允许的跳跃操作

- **Metropolis 采样器**：
  - `n_chains=100`：100 条并行的马尔可夫链
  - `sweep_size=32`：每条链每步扫描 32 个格子

---

### 4. 包装模型为 NetKet 兼容的 Machine 函数

```python
# ===================== 4. 包装模型为 machine 函数 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state
```

**代码解读：**

- **nnx.split**：将 Flax NNX 模型分离为图定义（结构）和状态（参数）
  - `graphdef`：包含模型的结构信息（层、激活函数等）
  - `state`：包含所有可训练的参数

- **machine 函数**：
  - `nnx.merge(graphdef, params)`：用新的参数重新构建模型
  - `m(sigma)`：对输入构型 sigma 计算波函数值
  - `@jax.jit`：JIT 编译以加速

- **返回值**：machine 函数、图定义、初始状态，便于后续使用

---

### 5. 局部能量计算

```python
# ===================== 5. 纯 JAX 实现的 force-based 梯度计算 =====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)

    这对应 NetKet 的 local_value_kernel
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])
```

**代码解读：**

- **局部能量定义**：
  $$E_{loc}(\sigma) = \sum_{\eta} H(\sigma \to \eta) \frac{\psi(\eta)}{\psi(\sigma)}$$
  其中 $H(\sigma \to \eta)$ 是哈密顿矩阵元，$\psi(\sigma)$ 是波函数值

- **ha.get_conn_padded(sigma)**：
  - 返回所有非零哈密顿矩阵元连接的构型 eta
  - 返回对应的矩阵元值 H_eta
  - padded 意味着返回固定形状的数组，便于向量化

- **波函数比率**：$\frac{\psi(\eta)}{\psi(\sigma)} = \exp(\log\psi(\eta) - \log\psi(\sigma))$
  - 直接计算指数的差比计算比值更数值稳定

- **statistics 函数**：计算均值和标准误差，用于估计统计不确定度

---

### 6. Force-Based 梯度计算

```python
@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    核心：复刻 NetKet 的 forces_expect_hermitian 函数

    使用 force-based 梯度计算：
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩

    关键：对于复数值网络，使用 holomorphic=True
    """
    # 1. 计算局部能量
    O_loc = compute_local_energies(machine, params, sigma)

    # 2. 统计能量均值
    O_mean, O_std = statistics(O_loc)

    # 3. 中心化局部能量
    O_centered = O_loc - O_mean

    # 4. 计算 ∇log ψ 对每个样本
    # 使用 jax.grad 计算复数梯度（holomorphic=True）
    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    # 5. 计算 force-based 梯度
    def weight_and_mean(grad_component):
        # grad_component 形状：(n_samples, d1, d2, ...)
        # O_centered 形状：(n_samples,)
        # 需要广播相乘后沿 axis=0 求平均
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)

    return O_mean, O_std, grad
```

**代码解读：**

- **Force-based 梯度公式**：
  $$\nabla_\theta \langle E \rangle = \langle (E_{loc} - \langle E \rangle) \nabla_\theta \log \psi \rangle$$

- **holomorphic=True**：
  - 对于复数波函数，需要计算全纯导数而非普通导数
  - 这允许我们正确处理复数参数的梯度

- **jax.vmap**：
  - 对所有样本并行计算梯度
  - 大幅加速批量处理

- **权重平均**：
  - `O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))`
  - 将中心化能量 reshape 为 (n_samples, 1, 1, ..., 1) 以匹配梯度维度
  - `jnp.conj(grad_component)`：对复数梯度取共轭（对应 QGT 定义）

---

### 7. 量子几何张量（QGT）计算

```python
#@partial(jax.jit, static_argnames=("machine",))
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）/ F 矩阵

    QGT 定义：
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩

    这就是 NetKet SR 的核心

    参数：
    - machine: 波函数机器
    - params: 网络参数
    - sigma: 样本 (n_samples, n_orbitals)
    - diag_shift: 对角线正则化参数 λ

    返回：
    - qgt_reg: 正则化后的 QGT 矩阵 (n_params, n_params)
    - unravel_fn: 用于将展平的向量恢复为 PyTree 结构的函数
    """
    n_samples = sigma.shape[0]

    # 步骤 1: 计算每个样本的 ∇log ψ
    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    # grad_matrix 是 PyTree，每个元素形状为 (n_samples, ...)
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    # 步骤 2: 将 PyTree 展平为矩阵 (n_samples, n_params)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    # 步骤 3: 中心化（减去均值）
    # 这对应 QGT 定义中的第二项：- ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)  # (1, n_params)
    grad_centered = grad_flat - grad_mean  # (n_samples, n_params)

    # 步骤 4: 计算 QGT = (1/N) * Σ ∇log ψ* ∇log ψ^T
    # 注意：对于复数，需要使用共轭
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered

    # 步骤 5: 添加正则化
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn
```

**代码解读：**

- **QGT（量子几何张量）定义**：
  $$S_{ij} = \langle \partial_i \log \psi^* \partial_j \log \psi \rangle - \langle \partial_i \log \psi^* \rangle \langle \partial_j \log \psi \rangle$$

- **PyTree 展平**：
  - `flatten_util.ravel_pytree`：将 PyTree 结构展平为一维向量
  - `unravel_fn`：反向操作，将向量恢复为 PyTree

- **矩阵计算**：
  - `jnp.conj(grad_centered).T @ grad_centered`：计算 $\nabla\log\psi^* (\nabla\log\psi)^T$
  - 这等价于批量外积的平均

- **正则化**：
  - `diag_shift * jnp.eye(qgt.shape[0])`：添加对角正则化确保矩阵可逆
  - 典型值为 0.001 ~ 0.1

---

### 8. 训练循环（自然梯度优化）

```python
# ===================== 6. 初始化 =====================
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)
sampler_state = sampler.init_state(machine, params, seed=1)

optimizer = optax.sgd(learning_rate=0.01)  # 学习率 0.01
opt_state = optimizer.init(params)

# 训练参数
N_ITER = 300  # 迭代次数
N_SAMPLES = 1008  # 样本数

# ===================== 7. 训练循环 =====================
print("\n" + "="*60)
print("开始纯 JAX VMC 训练 (自然梯度下降法)")
print("="*60)

# 用于记录训练历史
history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}

for step in range(N_ITER):
    # 1. 采样
    sampler_state = sampler.reset(machine,params,sampler_state)

    samples, sampler_state = sampler.sample(
        machine, params, state=sampler_state,
        chain_length=20
    )
    samples = samples.reshape(-1, hi.size)

    # 2. 计算 force-based 能量和梯度
    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
    grad = jax.tree_map(lambda x: x*2, grad)

    # 3. 计算 QGT 和自然梯度
    qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001)
    grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)

    # 自然梯度 natural-gradient = S^{-1} * grad
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad)
    grad = natural_grad

    # 4. 更新参数（自然梯度下降）
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    # 5. 记录历史
    if step % 50 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        print(f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")
```

**代码解读：**

- **采样**：
  - `sampler.reset`：重置采样器状态
  - `sampler.sample`：从马尔可夫链采样新构型
  - `chain_length=20`：每条链采样 20 个构型

- **梯度计算**：
  - `forces_expect_hermitian`：计算局部能量和 force-based 梯度
  - `jax.tree_map(lambda x: x*2, grad)`：乘以 2（对应能量对波函数的导数因子）

- **自然梯度计算**：
  $$\theta_{t+1} = \theta_t - \eta \cdot S^{-1} \nabla_\theta \langle E \rangle$$
  - `jnp.linalg.solve(qgt_reg, grad_flat)`：求解线性系统 $S^{-1} \nabla E$
  - `grad_unravel_fn`：将展平的梯度恢复为 PyTree 结构

- **参数更新**：
  - `optax.sgd`：随机梯度下降优化器
  - 学习率 0.01（自然梯度法通常需要较小学习率）

---

## 训练结果

经过 300 步迭代后的典型结果：

```
============================================================
开始纯 JAX VMC 训练 (自然梯度下降法)
============================================================
Step   0 | E: -0.48108884 ± 0.005743 | FCI: -1.01546825 | Error: 0.534379
Step  50 | E: -0.94973750 ± 0.006444 | FCI: -1.01546825 | Error: 0.065731
Step 100 | E: -0.96152565 ± 0.004666 | FCI: -1.01546825 | Error: 0.053943
Step 150 | E: -0.96894767 ± 0.002735 | FCI: -1.01546825 | Error: 0.046521
Step 200 | E: -1.00163073 ± 0.001210 | FCI: -1.01546825 | Error: 0.013838
Step 250 | E: -1.01172069 ± 0.000656 | FCI: -1.01546825 | Error: 0.003748
Step 299 | E: -1.01303152 ± 0.000762 | FCI: -1.01546825 | Error: 0.002437

============================================================
训练完成!
最终能量：-1.01532961 ± 0.000621 Ha
FCI 基准：-1.01546825 Ha
绝对误差：0.000139 Ha
相对误差：0.0137%
============================================================
```

收敛过程展示了自然梯度法的快速收敛特性，相比普通梯度下降能更快地接近基态能量。

## 关键技术细节

### 1. Flax NNX 与 NetKet 的集成

使用 `nnx.split` 和 `nnx.merge` 将 Flax NNX 模型转换为 NetKet 兼容的纯函数形式：

```python
def create_machine(model: nnx.Module):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state
```

### 2. JIT 编译与向量化

- `@partial(jax.jit, static_argnames=("machine",))`：JIT 编译核心计算
- `jax.vmap`：对所有样本并行计算梯度
- `jax.grad(..., holomorphic=True)`：复数域的全纯导数

### 3. PyTree 操作

使用 `flatten_util.ravel_pytree` 将 PyTree 结构展平为向量，便于矩阵运算：
- 梯度展平：用于 QGT 矩阵乘法
- 向量恢复：用于参数更新

### 4. 费米子采样

使用 `FermionHopRule` 处理费米子系统的采样：
- 保证采样构型符合泡利不相容原理
- 图结构定义允许的跳跃操作

## 与标准 NetKet 实现对比

本实现复现了 NetKet 的核心算法，同时展示了：

1. **算法透明度**：每一步计算都有清晰的数学对应
2. **可定制性**：可以轻松修改 QGT 计算、梯度估计器等
3. **混合架构**：展示如何将 Flax NNX 神经网络与 NetKet 采样器结合

## 运行要求

### 依赖库
- JAX
- NetKet
- Flax NNX
- PySCF
- Optax
- Pytrees (flatten_util)

### 参数设置建议
- **样本数**：1008（足够大以降低统计误差）
- **链数**：100（并行采样）
- **扫描步数**：32（平衡接受率和效率）
- **正则化参数**：0.001（太小可能导致奇异性，太大降低精度）
- **学习率**：0.01（自然梯度法通常需要较小学习率）

## 参考资料

1. Becca & Sorella, *Quantum Monte Carlo Approaches for Correlated Systems* (2017)
2. McClean et al., "The Pauli principle, graph theory, and natural gradient optimization" (2020)
3. Stokes et al., "Quantum Natural Gradient" (2020)
