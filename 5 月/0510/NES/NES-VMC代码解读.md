# NES-VMC 算法在 NetKet 官方 API 中的复现：目的与进展

## 1. 研究目的

**目标**：基于 NetKet 框架、Flax.nnx 的相关 API复现 **NES-VMC（Natural Excited State Variational Monte Carlo）算法**，用于计算量子多体系统（如 H₂ 分子）的前 $K$ 个激发态能量。

**要求**：

- 使用 NetKet 内置的扩展希尔伯特空间 `hi ** K`
- 最终通过训练得到的模型，对角化平均局域能量矩阵，获得基态与激发态能量

## 2. NES-VMC 算法核心思想

### 2.1 问题背景

在量子力学中，我们通常需要求解哈密顿算符 $\hat{H}$ 的本征值问题，即找到最低的 $K$ 个本征函数。对于量子多体系统，直接对角化哈密顿矩阵通常是不可行的，因为希尔伯特空间的维度随粒子数指数增长。

NES-VMC 将原系统前 $K$ 个激发态的求解问题**等价转化为一个"扩展系统"的基态求解问题**。
以下是问题的描述， 非必要不要修改
```python
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
edges = [(0, 1), (2, 3),(4, 5),(6,7)]

```

### 2.2 扩展希尔伯特空间

设 $\mathbf{X} = (x_1, \dots, x_N)$ 表示一组包含 $N$ 个粒子的粒子集（particle set），其中 $x_i$ 表示第 $i$ 个粒子的状态。扩展希尔伯特空间由 $K$ 个原系统副本张量积构成，每个配置对应 $K$ 个组态 $\mathbf{x} = (x^1, \dots, x^K)$。


### 2.4 SingleStateAnsatz 的构成  
$\psi(\mathbf{x})$ 对应着普通 VMC 算法的 Ansatz， 需要注意的是在本案例中 $\mathbf{x}$ 对应着粒子数守恒、自旋守恒、STO-3G 下的 $H_2$ 分子的4种合法组态，并且自旋顺序是 $[ \alpha_1, \alpha_2,\beta_1,\beta_2]$ 4种合法组态是 $[1,0,1,0],[0,1,0,1],[1,0,1,1],[1,0,0,1]$
对于 SingleStateAnsatz 的代码是：
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
        return jnp.squeeze(out)
def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state
```
需要注意的是本案例中，默认model的参数是复数值，输出是 $\ln{\psi(x)}$
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
            
            ansatz = SingleStateAnsatz(
                n_spin_orbitals, 
                hidden_dim, 
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)
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

def create_machine_matrix(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        #print(f'x.shape: {sigma.shape}  ')
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix = m(sigma)
        return log_M_matrix

    return machine, graphdef, state


```
NESTotalAnsatz 的输出为 
$$
\ln{\Psi(\mathbf{X})} = \ln{\det{\mathbf{M} \triangleq \ln{\det{     \begin{pmatrix}
\psi_1(\mathbf{x}^1) & \dots & \psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\psi_1(\mathbf{x}^K) & \dots & \psi_K(\mathbf{x}^K)
\end{pmatrix}}}}}$$

### 2.4 扩展哈密顿量

定义扩展哈密顿量 $\tilde{H} = \hat{H}_1 \oplus \hat{H}_2 \oplus \cdots \oplus \hat{H}_K$，其中 $\hat{H}_i$ 是仅作用于第 $i$ 个粒子集的哈密顿量。$\tilde{H}$ 的基态能量等于原系统 $\hat{H}$ 最低 $K$ 个能量之和，其基态波函数正是上述行列式形式的 $\Psi^\star$。
这里由于 Netket似乎不支持这样的哈密顿量直和形式，我们使用间接的方案：
```python
hi.all_states() 
>>Array([[0, 1, 0, 1],
       [0, 1, 1, 0],
       [1, 0, 0, 1],
       [1, 0, 1, 0]], dtype=int8)

K=2
hi_ext = hi**K
hi_ext.all_states()
>>Array([[0, 1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 1, 1, 0],
       [0, 1, 0, 1, 1, 0, 0, 1],
       [0, 1, 0, 1, 1, 0, 1, 0],
       [0, 1, 1, 0, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 1, 1, 0],
       [0, 1, 1, 0, 1, 0, 0, 1],
       [0, 1, 1, 0, 1, 0, 1, 0],
       [1, 0, 0, 1, 0, 1, 0, 1],
       [1, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 0, 1, 1, 0, 0, 1],
       [1, 0, 0, 1, 1, 0, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1],
       [1, 0, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 1, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 1, 0]], dtype=int8)
```

### 2.5 Ham_psi 和 Ham_Psi 函数  
由于 NES-VMC 的损失函数(对应着原文 Eq.29 )
TotalAnsatz的值是: $\Psi(\mathbf{X})$或者$\ln{\Psi(\mathbf{x})}$ 
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
```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    🔥 同时支持：
    - 单个态 x: (n_spin,)
    - 批量态 x: (batch_size, n_spin)
    """
    # ======================
    # 核心：自动给单个样本增加 batch 维度
    # ======================
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    # ======================
    # 向量化计算（批处理）
    # ======================
    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    # 批量处理
    H_psi_batch = jax.vmap(_single_hpsi)(x)

    # ======================
    # 如果是单个输入，就压回单个输出
    # ======================
    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
    
def Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)
    # ======================
    # 核心：单样本 与 批处理 自动兼容
    # ======================
    if x.ndim == 2:
        # 输入形状：(K, n_spin) → 单个扩展态 → 返回 (K,K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # 单态：(4,)
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 输入形状：(batch, K, n_spin) → 批量 → 返回 (batch, K, K)
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
        
        # 自动批处理！
        return jax.vmap(_single_HamPsi)(x)

    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")
```
其中`Ham_psi` 是用来计算 $\hat{H}\psi_1(\mathbf{x}^1)$  
`Ham_Psi` 是用来计算 $\hat{H}\Psi(\mathbf{x})$  
### 2.6 采样器的设置  

在 NES-VMC 算法中，采样器需要在**扩展希尔伯特空间** $\mathbf{x} = (x^1, x^2, \ldots, x^K)$ 上进行，其中每个 $x^k$ 属于原系统的希尔伯特空间 $\mathcal{H}$。

#### 核心约束：禁止重复组态

**关键约束**：扩展态必须满足 $x^i \neq x^j$（当 $i \neq j$ 时）。这是因为总 Ansatz 的行列式结构要求各副本的组态必须互不相同，否则矩阵 $\Psi(\mathbf{x})$ 将出现相同的行/列，导致行列式为零。

对于 $K=2$ 的情况，扩展态的合法构型数为 $N_s^2 - N_s = 4^2 - 4 = 12$（其中 $N_s=4$ 是 $H_2$ 分子的单系统希尔伯特空间维度），而非简单的 $4^2 = 16$。

#### 方案 1：NetKet 内置采样器

代码使用 NetKet 的 `TensorRule` 来构建扩展希尔伯特空间的采样器：

```python
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)
```

其中：
- `FermionHopRule`：在单系统希尔伯特空间上执行费米子跃迁（满足粒子数守恒）
- `TensorRule`：将单系统采样规则复制 $K$ 份，应用到扩展希尔伯特空间

#### 方案二: 自定义 MCMC 采样器 (采用✅)

由于 NetKet 内置采样器无法完全处理"禁止相同组态"的约束，代码还实现了完全自定义的 MCMC 采样器：

**1. 跃迁生成**（`make_get_all_next_states`）：
- 遍历所有跃迁边 $(i, j)$，执行费米子跃迁
- 跃迁后按 $K$ 份切分，检查是否存在重复组态 $x^a = x^b$
- 只有既满足费米子跃迁条件、又无重复组态的跃迁才是有效的

**2. Metropolis-Hastings 接受拒绝**（`make_metropolis_hastings_step`）：
```python
log_acc = 2 * jnp.real(log_cand - log_curr)  # 对数接受率
accept = is_valid & (log_acc > jnp.log(jax.random.uniform(subk)))
```

**3. 多链采样**（`mcmc_sampler_multichain`）：
- 并行执行 $N_{chains}$ 条马尔可夫链
- 每次 sweep 执行 `sweep_size` 次跃迁
- 先 warmup（热化），再正式采样

#### 初始状态生成

`generate_random_initial_states` 函数确保生成的初始状态满足约束：
```python
while True:
    s1 = hi_ext.random_state(k1)
    s2 = hi_ext.random_state(k2)
    x1 = s1[:n_spin]
    x2 = s2[n_spin:]
    if not jnp.all(x1 == x2):  # 保证 x1 ≠ x2
        break
ext_state = jnp.concatenate([x1, x2])
```
完整代码:
```python
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



```
## 3. 损失函数

### 3.1 目标函数（Rayleigh 商）

NES-VMC 的目标函数为扩展哈密顿量关于总 Ansatz 的 Rayleigh 商：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\langle\Psi|\Psi\rangle}

$$

利用矩阵行列式引理，可以将其重写为迹形式：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\det(S)} = \mathrm{Tr}\left(S^{-1}\hat{H}\right) = \mathrm{Tr}\left(\Psi^{-1}\tilde{H}\Psi\right)

$$
其中 $\Psi^{-1}H\Psi$ 使用以下函数计算：
```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

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
\nabla_\theta \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} = 2\mathbb{E}_{x \sim \psi^2}\left[\left(E_L(x) - \mathbb{E}_{x' \sim \psi^2}[E_L(x')]\right)\nabla_\theta \log|\psi(x)|\right]

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

```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. 批量局域能量矩阵
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    #print(f'E_L_batch.shape={E_L_batch.shape}')
    
    tr_batch = loss_batch
    tr_mean = tr_batch.mean()
    tr_centered = tr_batch - tr_mean  # ✅ 正确的权重

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. 计算 ∇logΨs
    dlogPsi_batch = vmap_grad_logPsia(total_params, x_batch)

    # 5. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean
```

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
import time
from NES_VMC import E_fcis,mcmc_sampler_multichain
# ======================
# 超参数
# ======================
N_CHAINS = 16
N_WARMUP = 50
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 20
N_ITER =10

rngs = nnx.Rngs(42)
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
single_ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef, total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (自然梯度下降法)")

print("超参数")
print("="*60)

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix':[],
    'natural_grad':[],
    'grad_flat':[],
    'samples':[],
    'log_Psi':[],
    'log_M':[]
}
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")
sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)  # 每次迭代换种子避免初始状态固定
start_time = time.time()
for step in range(N_ITER):
    # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
    # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
    samples,sampler_state = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        sampler_state=sampler_state,
        edges=((0,1),(2,3),(4,5),(6,7)),
        machine=total_machine,
        params=total_params,
    )
    #samples = samples.reshape(-1,2,4)

    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    #model_output = log(\Psi(X)) 
    # qgt_reg,qgt_unravel_fun = compute_nes_qgt(total_machine, total_params, samples.reshape(-1,K,4), diag_shift=0.01) 
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    
    # # # 自然梯度求解
    # natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    # natural_grad = grad_unravel_fn(natural_grad)
    # grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    # 5. 记录历史
    if step % 1 == 0 or step == N_ITER - 1:
        # total_model =  nnx.merge(graphdef,total_params)
        # log_Psi,log_M  = total_model(samples.reshape(-1,2,4))
        eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        history['step'].append(step)
        history['E_Lmatrix'].append(E_L_mean)
        history['samples'].append(samples)
        history['loss'].append(loss_mean)
        # #history['natural_grad'].append(natural_grad)
        # history['grad_flat'].append(grad_flat)
        # history['log_Psi'].append(log_Psi)
        # history['log_M'].append(log_M)
        history['params'].append(total_params)
        print(f"Step {step:3d} | Loss: {loss_mean}|基态能量={eig_vals[0]:.8f} Ha| 第一激发态能量={eig_vals[1]:.8f} Ha Ha")
        print(f'grad={grad_flat[30:32]}')


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

>>============================================================
开始多链 NES-VMC 训练 (自然梯度下降法)
超参数
============================================================
基态能量=-1.01546825 Ha| 第一激发态能量=-0.87542794 Ha| 第二激发态能量=-0.42938376 Ha
Step   0 | Loss: -1.3085838975566997|基态能量=-1.05045325 Ha| 第一激发态能量=-0.25813065 Ha Ha
grad=[0.00301162+0.0929836j 0.02226221-0.0151168j]
Step   1 | Loss: -1.3259665446678726|基态能量=-1.05050409 Ha| 第一激发态能量=-0.27546246 Ha Ha
grad=[-0.00760139+0.05941801j  0.01502238-0.01215661j]
Step   2 | Loss: -1.3264937500558491|基态能量=-1.04017703 Ha| 第一激发态能量=-0.28631672 Ha Ha
grad=[-0.01531673+0.04694783j  0.01295463-0.01031044j]
Step   3 | Loss: -1.333463731665521|基态能量=-1.03364422 Ha| 第一激发态能量=-0.29981951 Ha Ha
grad=[-0.01488133+0.0326869j   0.00963188-0.00921168j]
Step   4 | Loss: -1.3379080072010145|基态能量=-1.03390325 Ha| 第一激发态能量=-0.30400475 Ha Ha
grad=[-0.01720937+0.02579935j  0.00915731-0.00704474j]
Step   5 | Loss: -1.3383651686642133|基态能量=-1.02921225 Ha| 第一激发态能量=-0.30915292 Ha Ha
grad=[-0.01998693+0.02244732j  0.01000103-0.00572663j]
Step   6 | Loss: -1.3442234589225837|基态能量=-1.02048401 Ha| 第一激发态能量=-0.32373945 Ha Ha
grad=[-0.0157033 +0.01600828j  0.00830118-0.00537434j]
Step   7 | Loss: -1.3472340497418507|基态能量=-1.02255476 Ha| 第一激发态能量=-0.32467928 Ha Ha
grad=[-0.01563603+0.01150525j  0.00907717-0.00364779j]
Step   8 | Loss: -1.345280102095183|基态能量=-1.02554312 Ha| 第一激发态能量=-0.31973699 Ha Ha
grad=[-0.01589478+0.01024677j  0.0082061 -0.00259513j]
Step   9 | Loss: -1.347768045219396|基态能量=-1.03290681 Ha| 第一激发态能量=-0.31486124 Ha Ha
grad=[-0.01027277+0.00898995j  0.00476339-0.00223605j]
训练耗时：8.21 秒

============================================================
训练完成!
============================================================
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

你需要知道的是:`SingleStateAnsatz`的输出是 $\ln(\psi(\mathbf{x}))$  
而`NESTotalAnsatz`的输出是: $\ln{\Psi(\mathbf{X})}$ 

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
import time 
from functools import partial
from jax import flatten_util
from VMC_tool import hi, edges,ha,SingleStateAnsatz,create_machine,compute_local_energies,\
    compute_qgt,forces_expect_hermitian,E_fcis
    
import jax
import jax.numpy as jnp
from functools import partial

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

   
# ===================== 6. 初始化（适配多链） =====================
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params)

# 训练参数（调整为多链）
N_ITER = 300
N_CHAINS = 16  # 并行链数（可调，建议8-32）
N_SAMPLES_PER_CHAIN = 100  # 每条链采样数 → 总样本数=16*63=1008（和原单链总样本数一致）
N_WARMUP = 32

# ===================== 7. 训练循环（多链版本） =====================
print("\n" + "="*60)
print("开始多链 VMC 训练 (自然梯度下降法)")
print("="*60)

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}
start_time = time.time()
for step in range(N_ITER):
    # 1. 生成多链随机初始状态（模仿NetKet，无需手动指定单个initial_state）
    initial_states = generate_random_initial_states(hi, N_CHAINS, seed=21+step)  # 每次迭代换种子避免初始状态固定
    
    # 2. 多链采样（总样本数=16*63=1008，和原单链一致）
    samples = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        initial_states=initial_states,
        edges=((0, 1), (2, 3)),
        machine=machine,
        params=params,
        seed=21+step
    )
    
    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
    grad = jax.tree_map(lambda x: x*2, grad)
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