# NES-VMC 算法在 NetKet 官方 API 中的复现：目的与进展

## 1. 研究目的

**目标**：完全基于 NetKet 框架的高层 API（`MCState`、`VMC` 驱动）复现 **NES-VMC（Natural Excited State Variational Monte Carlo）算法**，用于计算量子多体系统（如 H₂ 分子）的前 $K$ 个激发态能量。

**要求**：

- 使用 NetKet 内置的扩展希尔伯特空间 `hi ** K`
- 使用 `MCState` 管理变分态与采样器
- 使用 `VMC` 驱动自动执行训练循环
- 最终通过训练得到的模型，对角化平均局域能量矩阵，获得基态与激发态能量

## 2. NES-VMC 算法核心思想

### 2.1 问题背景

在量子力学中，我们通常需要求解哈密顿算符 $\hat{H}$ 的本征值问题，即找到最低的 $K$ 个本征函数。对于量子多体系统，直接对角化哈密顿矩阵通常是不可行的，因为希尔伯特空间的维度随粒子数指数增长。

NES-VMC 将原系统前 $K$ 个激发态的求解问题**等价转化为一个"扩展系统"的基态求解问题**。

### 2.2 扩展希尔伯特空间

设 $x = (x\_1, \dots, x\_N)$ 表示一组包含 $N$ 个粒子的粒子集（particle set），其中 $x\_i$ 表示第 $i$ 个粒子的状态。扩展希尔伯特空间由 $K$ 个原系统副本张量积构成，每个配置对应 $K$ 个组态 $\mathbf{x} = (x^1, \dots, x^K)$。

### 2.3 TotalAnsatz 的构成

设 $\psi_i$ 表示第 $i$ 个 $N$ 粒子波函数（可能未归一化），则 **TotalAnsatz** 定义为矩阵 $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$ 的行列式：

$$
\Psi(\mathbf{x}) \equiv \det\begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \cdots & \psi_K(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \cdots & \psi_K(x^2) \\
\vdots & \vdots & \ddots & \vdots \\
\psi_1(x^K) & \psi_2(x^K) & \cdots & \psi_K(x^K)
\end{pmatrix}

$$

其中：

- $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$：将所有电子集合与所有波函数结合的矩阵
- $\psi_i(x^j)$：第 $i$ 个单态 Ansatz 在第 $j$ 个粒子集上的值
- $\Psi(\mathbf{x}) = \det(\Psi(\mathbf{x}))$：总 Ansatz，可以看作是由 $N$ 粒子波函数组成的未归一化 Slater 行列式

**关键性质**：通过将总 Ansatz 表示为单态 Ansatz 的行列式，可以防止不同 Ansatz 坍缩到同一状态，而不需要显式要求它们正交。

### 2.4 扩展哈密顿量

定义扩展哈密顿量 $\tilde{H} = \hat{H}\_1 \oplus \hat{H}\_2 \oplus \cdots \oplus \hat{H}\_K$，其中 $\hat{H}\_i$ 是仅作用于第 $i$ 个粒子集的哈密顿量。$\tilde{H}$ 的基态能量等于原系统 $\hat{H}$ 最低 $K$ 个能量之和，其基态波函数正是上述行列式形式的 $\Psi^\star$。

## 3. 损失函数

### 3.1 目标函数（Rayleigh 商）

NES-VMC 的目标函数为扩展哈密顿量关于总 Ansatz 的 Rayleigh 商：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\langle\Psi|\Psi\rangle}

$$

利用矩阵行列式引理，可以将其重写为迹形式：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\det(S)} = \mathrm{Tr}\left(S^{-1}\hat{H}\right)

$$

其中 $S$ 为重叠矩阵：

$$
S = \begin{pmatrix}
\langle\psi\_1|\psi\_1\rangle & \cdots & \langle\psi\_1|\psi\_K\rangle \\
\vdots & \ddots & \vdots \\
\langle\psi\_K|\psi\_1\rangle & \cdots & \langle\psi\_K|\psi\_K\rangle
\end{pmatrix}

$$

### 3.2 局域能量矩阵

通过 Monte Carlo 采样，损失函数可以写成期望值形式：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\Psi^{-1}(\mathbf{x})\tilde{H}\Psi(\mathbf{x})\right)\right]

$$

定义**局域能量矩阵**为：

$$
E\_L(\mathbf{x}) \equiv \Psi^{-1}(\mathbf{x})\tilde{H}\Psi(\mathbf{x})

$$

这是一个 $K \times K$ 矩阵，其迹即为标量局域能量。当 $K = 1$ 时，这退化为标准 VMC 中的局域能量。

## 4. 梯度公式

### 4.1 标准 VMC 梯度回顾

对于基态 VMC，能量关于变分参数 $\theta$ 的梯度为：

$$
\nabla\_\theta \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} = 2\mathbb{E}_{x \sim \psi^2}\left[\left(E_L(x) - \mathbb{E}_{x' \sim \psi^2}[E_L(x')]\right)\nabla\_\theta \log|\psi(x)|\right]

$$

### 4.2 NES-VMC 梯度

对于总 Ansatz，梯度计算类似。损失函数是迹形式 $\mathcal{L} = \mathrm{Tr}(E_L(\mathbf{x}))$，定义对数幅度：

$$
\log|\Psi(\mathbf{x})| = \log\det(\Psi(\mathbf{x})) = \mathrm{Tr}\left(\log(\Psi(\mathbf{x}))\right)

$$

梯度公式为：

$$
\nabla_\theta \mathcal{L} = 2\mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\left(E_L(\mathbf{x}) - \bar{E}_L\right)\nabla_\theta \log\Psi(\mathbf{x})\right)\right]

$$

其中：
- $E_L(\mathbf{x}) = \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})$ 是局域能量矩阵
- $\bar{E}_L = \mathbb{E}_{\mathbf{x}' \sim \Psi^2}[E_L(\mathbf{x}')]$ 是局域能量矩阵的期望值
- $\nabla_\theta \log\Psi(\mathbf{x})$ 是波函数矩阵对参数 $\theta$ 的对数梯度

当 $K = 1$ 时，上式退化为标准 VMC 的梯度公式。

### 4.3 批量 walker 的梯度估计

与标准 VMC 类似，可以使用同一批次中独立的 walker 来获得无偏梯度估计：

$$
\nabla_\theta \mathcal{L} = \frac{N-1}{2N}\mathbb{E}_{x\_1,\dots,x\_N}\left[\frac{1}{N}\sum_{i=1}^N\left(E_L(x_i) - \frac{1}{N}\sum\_{j=1}^N E_L(x_j)\right)\nabla_\theta \log|\Psi(x_i)|\right]

$$

## 5. 激发态能量提取

### 5.1 能量矩阵的对角化

训练完成后，通过大量采样累积局域能量矩阵：

$$
\bar{E}_L = \mathbb{E}_{\mathbf{x} \sim \Psi^2}[E_L(\mathbf{x})]

$$

然后对 $\bar{E}\_L$ 进行对角化：

$$
\bar{E}_L = U\Lambda U^{-1}

$$

其中 $\Lambda = \mathrm{diag}(E\_1, E\_2, \dots, E\_K)$ 包含按能量排序的本征值。

### 5.2 物理解释

当单态 Ansatz 是本征函数的线性组合 $\psi\_i = \sum\_j a\_{ij}\psi\_j^\star$ 时，有：

$$
\Psi^{-1}\hat{H}\Psi = A^{-1}\Lambda A

$$

其中 $A$ 是系数矩阵。因此，通过对角化可以直接获得各激发态的能量 $E\_1, E\_2, \dots, E\_K$。
以下代码不要更改 需要强调的是 edges =[α1,α2,β1,β2] 这样的顺序


## 6. 代码实现
### 6.1 NES-VMC 算法实现
```python
"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
"""
"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
"""
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
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import time
from functools import partial

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
# ===================== H₂ 分子定义 & FCI 基准 =====================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")
# ===================== NetKet 哈密顿量和采样器 =====================
ha = nkx.operator.from_pyscf_molecule(mol)

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K=2
hi_ext = hi**K
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)

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
        return jnp.squeeze(out)

class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        
        # 每个输出都是 logψ(x) ✅
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # 形状：[K, n_spin]
            #print(f'x_single.shape: {x_single.shape}')
            x_single = x_single.reshape(self.K, self.n_spin)
            # ==============================
            # 正确构建 L_ij = log ψ_j(x^i)
            # 无vmap错误 ！！！
            # ==============================
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            
            # ==============================
            # 你的核心公式 100% 正确
            # logΨ = log det( exp(L) )
            # ==============================
            Psi_matrix = jnp.exp(L)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_matrix)
            log_Psi = log_abs_det + 1j * jnp.angle(sign)
            
            return log_Psi, L
        
        # 安全的批量处理
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            # 直接处理单个样本
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin*self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            # 直接处理批量样本
            return jax.vmap(_forward_single)(x)
        
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim ==1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')
            
total_ansatz = NESTotalAnsatz(4,2,8,rngs=nnx.Rngs(12))


    
def create_machine(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix = m(sigma)
        return log_psi_total

    return machine, graphdef, state


total_ansatz = NESTotalAnsatz(4,K,rngs=nnx.Rngs(12))
machine, graphdef, params = create_machine(total_ansatz)

def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])

def Ham_psi(ha: nk.operator.DiscreteOperator, model:SingleStateAnsatz, x):
    """计算 Hψ(x)，model 输出 log_psi 时完全正确"""
    x_primes, mels = ha.get_conn_padded(x)
    # 1. 计算所有 σ' 的 log_psi
    log_psi_vals = jax.vmap(model)(x_primes)
    # 2. 指数还原成 ψ(σ')
    psi_vals = jnp.exp(log_psi_vals)
    # 3. 求和得到 Hψ(x)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

def Ham_Psi(ha: nk.operator.DiscreteOperator, total_ansatz:NESTotalAnsatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    hilber_size = total_ansatz.n_spin
    k = total_ansatz.K
    x = x.reshape(k, hilber_size)
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi


grad_logPsi = jax.grad(machine, argnums=0, holomorphic=True)

# 向量化（批量 walker）
vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

def NES_loss_energy(ha, graphdef, params, x):
    # 先获取 Psi 矩阵
    total_model = nnx.merge(graphdef, params)
    log_psi_det,log_M = total_model(x)
    Psi_Matrix = jnp.exp(log_M)

    H_psi_x = Ham_Psi(ha, total_model, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    #return jnp.trace(Psi_Matrix_inv), Psi_Matrix_inv
    return jnp.real(jnp.trace(Psi_Matrix_inv)), Psi_Matrix_inv

@partial(jax.vmap, in_axes=(None, None, None, 0))
def compute_local_energy_matrix_batch(ha: nk.operator.DiscreteOperator,graphdef, params, x_batch):
    loss_val, E_L = NES_loss_energy(ha, graphdef, params, x_batch)
    return E_L

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator, graphdef, params, x_batch):
    """
    ✅ 最终正确版：完全对齐 NetKet + NES-VMC 论文
    公式：∇⟨E⟩ = ⟨ (tr(E_loc) - tr(E_mean)) * ∇logΨ* ⟩
    """
    # 1. 批量局域能量矩阵
    E_L_batch = compute_local_energy_matrix_batch(ha, graphdef, params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    
    # 2. 计算 tr(E_loc) 和 tr(E_mean) → ✅ 加了 real
    tr_E_loc_batch = jnp.real(jnp.trace(E_L_batch, axis1=1, axis2=2))
    tr_E_mean = jnp.real(jnp.trace(E_L_mean))
    
    # 3. 中心化能量
    tr_centered = tr_E_loc_batch - tr_E_mean

    # 4. 计算 ∇logΨ
    dlogPsi_batch = vmap_grad_logPsi(params, x_batch)

    # 5. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = jnp.mean(tr_E_loc_batch)
    return grad, loss_mean, E_L_mean

def extract_excitation_energies(params, model_graphdef, K=2, n_samples=10000):
    """
    从训练好的模型中提取激发态能量
    """
    # 生成大量样本
    total_ansatz = nnx.merge(model_graphdef, params)
    machine, _, _ = create_machine(total_ansatz)
    sampler_state = sampler.init_state(machine, params)
    
    samples, _ = sampler.sample(
        machine, params, state=sampler_state, chain_length=n_samples//sampler.n_chains
    )
    samples = samples.reshape(-1, K, 4)
    
    # 计算平均局域能量矩阵
    E_L, _ = compute_local_energy_matrix(model_graphdef, params, samples, ha, K)
    E_L_avg = jnp.mean(E_L, axis=0)
    
    # 对角化
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_avg)
    
    # 排序并输出结果
    print("\n" + "="*60)
    print("NES-VMC 激发态能量结果")
    print("="*60)
    for i, e in enumerate(eig_vals):
        exc = (e - eig_vals[0]) * 27.2114
        fci_e = E_fcis[i] if i < len(E_fcis) else None
        fci_exc = (fci_e - E_fcis[0]) * 27.2114 if fci_e is not None else None
        
        print(f"E{i}: {e:.8f} Ha (FCI: {fci_e:.8f} Ha) | 激发能: {exc:.4f} eV (FCI: {fci_exc:.4f} eV)")
    
    return eig_vals, E_L_avg




    # 向量化：批处理 → 行处理
    batch_apply = jax.vmap(lambda m, s: jax.vmap(apply_hamiltonian_to_M_row)(m, s))
    # print(f'M.shape={M.shape}')
    # print(f'sigma.shape={sigma.shape}')
    H_M = batch_apply(M, sigma)
    return H_M

def generate_random_initial_states(hi_ext, n_chains, seed=42):
    """
    🔥 兼容 TensorDiscreteHilbert！永远生成 x1 ≠ x2 的合法扩展态
    自动把扩展态切分成两个单态，保证绝不相同
    """
    import jax.numpy as jnp
    import jax.random as jr

    key = jr.PRNGKey(seed)
    n_spin = hi_ext.size // 2  # 自动获取单个系统的自旋数
    init_states = []

    for _ in range(n_chains):
        # 随机生成两个独立的单态
        key, k1, k2 = jr.split(key, 3)
        
        # 生成两个不同的随机态
        # 方法：先生成，若相同就重新生成，直到不同
        while True:
            s1 = hi_ext.random_state(k1)
            s2 = hi_ext.random_state(k2)
            
            # 切分扩展态 → 拿到内部两个真实子态
            x1 = s1[:n_spin]
            x2 = s2[n_spin:]
            
            # 保证子态不相等
            if not jnp.all(x1 == x2):
                break
            
            # 相等就换新随机数
            key, k2 = jr.split(key)

        # 拼接成合法扩展态 [x1, x2]
        ext_state = jnp.concatenate([x1, x2])
        init_states.append(ext_state)

    return jnp.stack(init_states)


def init_sampler_state(hi, n_chains, seed=42):
    init_states = generate_random_initial_states(hi, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)  # 每条链独立随机数
    return (init_states, chain_keys)

# ==============================================================================

def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total = m(sigma)
        return log_psi_total

    return machine, graphdef, state

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
    sigma = sigma.reshape(-1,2,4)
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

# ===================== 核心修改：K 作为显式参数传入 =====================
def make_get_all_next_states(K: int, SINGLE_HILBERT_SIZE:int,edges):
    """
    K: 显式传入的扩展副本数（NES-VMC 的 K）
    edges: 跃迁边，保持你的顺序不变
    """
    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        next_states = []
        valid_masks = []

        for (i, j) in edges:
            occ_i = S[..., i]
            occ_j = S[..., j]
            # 原始费米子跃迁有效条件
            valid_hop = (occ_i != occ_j)

            # 执行跃迁
            new_state = S.at[..., i].set(occ_j).at[..., j].set(occ_i)

            # ----------------------------------------------------------------
            # 核心：按 K 切分 x1, x2, ..., xK
            # ----------------------------------------------------------------
            x_list = jnp.split(new_state, K, axis=-1)

            # 检查：任意两个子组态不能相等
            has_duplicate = False
            for a in range(K):
                for b in range(a + 1, K):
                    equal = jnp.all(x_list[a] == x_list[b], axis=-1)
                    has_duplicate = jnp.logical_or(has_duplicate, equal)

            # 最终有效：能跃迁 + 无重复组态
            valid = valid_hop & (~has_duplicate)
            next_states.append(new_state)
            valid_masks.append(valid)

        return jnp.stack(next_states), jnp.stack(valid_masks)

    return get_all_next_states_jit


# ==============================================
# 3. Metropolis 单步跃迁
# ==============================================
def make_metropolis_hastings_step(K: int, SINGLE_HILBERT_SIZE:int,edges, machine):
    get_all_next = make_get_all_next_states(K,SINGLE_HILBERT_SIZE,edges)
    
    @jax.jit
    def mh_step(params, state: jnp.ndarray, key: jax.Array):
        candidates, valid_mask = get_all_next(state[None, :])
        candidates = candidates[:, 0]
        valid_mask = valid_mask[:, 0]
        
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, len(edges))
        cand = candidates[idx]
        is_valid = valid_mask[idx]
        
        log_curr = machine(params, state)
        log_cand = machine(params, cand)
        log_acc = 2 * jnp.real(log_cand - log_curr)
        
        key, subk = jax.random.split(key)
        accept = is_valid & (log_acc > jnp.log(jax.random.uniform(subk)))
        new_state = jnp.where(accept, cand, state)
        return new_state, key
    
    return mh_step

@partial(jax.jit, static_argnums=(0,1,3,4,6,7,8))
def mcmc_sampler_multichain(
    n_samples_per_chain: int,
    n_warmup: int,             # 单位：sweep
    sampler_state: tuple,      # ✅ NetKet 风格状态：(current_states, chain_keys)
    edges: tuple,
    machine: callable,
    params: dict,
    sweep_size: int = 32,       # ✅ 保留 sweep_size
    K: int = 2,
    SINGLE_HILBERT_SIZE: int = 4,
):
    # 解开 sampler_state（和 NetKet 完全一致）
    current_states, current_keys = sampler_state
    #n_chains = current_states.shape[0]
    mh_step = make_metropolis_hastings_step(K, SINGLE_HILBERT_SIZE, edges, machine)

    # -------------------------
    # 一次 sweep = 连续跳 sweep_size 次
    # -------------------------
    def single_sweep(carry, _):
        states, keys = carry
        # 多链并行 VMAP
        (new_s, new_k), _ = jax.lax.scan(
            lambda c, _: (jax.vmap(mh_step, in_axes=(None, 0, 0))(params, c[0], c[1]), None),
            (states, keys),
            length=sweep_size
        )
        return (new_s, new_k), new_s

    # -------------------------
    # 1) Warmup（仅更新状态，不保存样本）
    # -------------------------
    if n_warmup > 0:
        (current_states, current_keys), _ = jax.lax.scan(
            single_sweep, (current_states, current_keys), length=n_warmup
        )

    # -------------------------
    # 2) 正式采样（保存样本 + 更新最终状态）
    # -------------------------
    (final_states, final_keys), samples = jax.lax.scan(
        single_sweep, (current_states, current_keys), length=n_samples_per_chain
    )

    # 打包新的 sampler_state（返回给下一次迭代）
    new_sampler_state = (final_states, final_keys)
    
    # 展平样本：[n_samples, n_chains, n_sites] → [n_samples*n_chains, n_sites]
    #samples_flat = samples.reshape(-1, current_states.shape[-1])
    return samples, new_sampler_state

import time
# ======================
# 超参数
# ======================

if __name__ == '__main__':
    N_CHAINS = 16 
    N_WARMUP = 32
    N_SAMPLES_PER_CHAIN = 100
    SWEEP_SIZE = 32
    N_ITER =100

    # ======================
    # 初始化 ONCE
    # ======================
    rngs = nnx.Rngs(21)
    model = NESTotalAnsatz(4,2,12,rngs=rngs)
    machine, graphdef, params = create_machine(model)

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(params)

    # ===================== 7. 训练循环（多链版本） =====================
    print("\n" + "="*60)
    print("开始多链 NES-VMC 训练 (自然梯度下降法)")
    print("="*60)

    history = {
        'step': [],
        'energy': [],
        'energy_std': [],
        'error': []
    }
    sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)  # 每次迭代换种子避免初始状态固定
    start_time = time.time()
    for step in range(N_ITER):
        # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
        # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
        samples,sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=N_SAMPLES_PER_CHAIN,
            n_warmup=N_WARMUP,
            sampler_state=sampler_state,
            edges=((0, 1), (2, 3),(4,5),(6,7)),
            machine=machine,
            params=params,
        )
        #samples = samples.reshape(-1,2,4)

        # 3. 计算能量和自然梯度（逻辑和原代码一致）
        grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                    graphdef=graphdef,
                                                    params=params,
                                                    x_batch=samples.reshape(-1,2,4))
        #grad = jax.tree_map(lambda x: x*2, grad)
        qgt_reg,qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001) 
        grad_flat , grad_unravel_fn = flatten_util.ravel_pytree(grad)
    
        # 自然梯度求解
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad)
        grad = natural_grad
            
        # 4. 更新参数
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
            
        # 5. 记录历史
        if step % 5 == 0 or step == N_ITER - 1:
            eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
            history['step'].append(step)
            print(f"Step {step:3d} | Loss: {loss_mean}｜eig_vals: {eig_vals}")

    end_time = time.time()
    print(f"训练耗时：{end_time - start_time:.2f} 秒")
    # 最终结果
    print("\n" + "="*60)
    print(f"训练完成!")
    # print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
    # print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
    # print(f"绝对误差：{final_error:.6f} Ha")
    # print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
    print("="*60)


```
### 6.2 NES-VMC TotalWaveFunction 设定

我在讨论 TotalWaveFunction 的设定: 用于默认任何 model 的输出都是$\ln(\psi(x))$
以下是论文中的描述:
$$ \Psi(\mathbf{x}) \triangleq
\begin{pmatrix}
\psi_1(\mathbf{x}^1) & \dots & \psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\psi_1(\mathbf{x}^K) & \dots & \psi_K(\mathbf{x}^K)
\end{pmatrix}
$$
但是在真实代码中往往为了避免数值溢出 都会使用对数域
在代码上往往要使用对数域来避免数值溢出 因此:
$$
\ln{\Psi(\mathbf{x})} = \ln\left[\, \det
\begin{pmatrix}
\psi_1(\mathbf{x}_1) & \dots & \psi_K(\mathbf{x}_1) \\
\vdots & \ddots & \vdots \\
\psi_1(\mathbf{x}_K) & \dots & \psi_K(\mathbf{x}_K)
\end{pmatrix}
\,\right]
$$ 

### 6.3 Ham_psi 和 Ham_Psi函数
由于 NES-VMC 的损失函数应该是: 
你要区分符号: TotalAnsatz的值是: $\Psi(\mathbf{X})$或者$\ln{\Psi(\mathbf{x})}$ 
这里的 X 与 x的区别你也是知道的
SingleStateAnsatz的输出是$\ln(\psi(\mathbf{x}))$

$$
\hat{H}\Psi(\mathbf{x}) \triangleq 
\begin{pmatrix}
\hat{H}\psi_1(\mathbf{x}^1) & \dots & \hat{H}\psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\hat{H}\psi_1(\mathbf{x}^K) & \dots & \hat{H}\psi_K(\mathbf{x}^K)
\end{pmatrix}
$$

$$
\mathcal{L} = \mathrm{Tr}\left(\Psi(\mathbf{x})^{-1} \hat{H}\Psi(\mathbf{x})\right)
$$

### 6.4 VMC 代码案例参考
你需要知道的是，以下代码是 VMC 的版本代码 ，以下的代码我已经多次测试过没有问题，你可以直接相信。
你在判断之前代码的时候 要参考以下代码里的实现。
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
import time


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
# ===================== 4. 包装模型为 machine 函数 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state

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
    # grad = ⟨(E_loc - E_mean) ∇log ψ⟩ = (1/N) Σ (E_loc[i] - E_mean) ∇log ψ(σ[i])
    # grad_matrix 已经是 PyTree 结构，每个元素的形状是 (n_samples, ...)
    # 关键修复：O_centered 形状为 (n_samples,)，需要正确广播到梯度张量的每个维度
    # 使用 reshape 将 O_centered 变为 (n_samples, 1, 1, ..., 1) 以匹配梯度张量
    def weight_and_mean(grad_component):
        # grad_component 形状：(n_samples, d1, d2, ...)
        # O_centered 形状：(n_samples,)
        # 需要广播相乘后沿 axis=0 求平均
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad


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
start_time = time.time()
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
    #qgt_reg, unravel_fn = compute_qgt(machine,params,vstate.samples.reshape(-1,4),0.001)
    qgt_reg,qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001) 
    grad_flat , grad_unravel_fn = flatten_util.ravel_pytree(grad)
  
    #自然梯度 natural-gradient = S^{-1} * grad
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
end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")
# 最终结果
final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
final_error = jnp.abs(final_energy.real - E_fcis[0])
print("\n" + "="*60)
print(f"训练完成!")
print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
print(f"绝对误差：{final_error:.6f} Ha")
print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)

```