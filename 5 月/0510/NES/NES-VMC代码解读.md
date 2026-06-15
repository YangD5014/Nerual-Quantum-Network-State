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
后面会提到虽然损失函数的定义是：$\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})$ , 这里的 $\mathcal{H}$ 是指拓展哈密顿量 $\tilde{H}$。  
但是可以被等价地看作下面的式子:
$$ \begin{align*}
\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})
&= \mathrm{Tr}\left[ \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x}) \right]
\end{align*} $$
其中的 $\hat{H}\Psi(\mathbf{x})$ 由下面的  `Ham_psi` 和 `Ham_Psi` 函数给出

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

在 NES-VMC 算法中，采样器需要在**扩展希尔伯特空间** $\mathbf{x} = (x^1, x^2, \ldots, x^K)$ 上进行，一次性采样 K 个组态, 其中每个 $x^k$ 属于原系统的希尔伯特空间 $\hat{H}$。  
我基于 Netket 的 Sampler + 自定义Rule 来构成了 NES-VMC 的采样器。

```python
# 单系统希尔伯特空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2  # NES 扩展副本数
hi_ext = hi ** K  # 扩展希尔伯特空间
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4
single_edges = ((0, 1), (2, 3))  # 费米子跃迁边
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =50
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # 转为jax数组（关键修复）
print(ext_edges)

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)

sampler_state = nes_sampler.init_state(total_machine, total_params, seed=1)
samples_raw, sampler_state = nes_sampler.sample(
    total_machine, total_params, state=sampler_state, chain_length=40
)
samples_raw.shape

```

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

完整代码:
```python
@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    # 【仅保留edges：JAX只允许存jax数组，彻底删除hi_ext！】
    edges: jnp.ndarray

    def _check_duplicate(self, sigma_ext):
        """NES约束：子组态不重复（直接用全局K和SINGLE_SIZE，完全安全）"""
        sub = sigma_ext.reshape((*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        return jnp.any(jnp.all(sub[...,1:,:] == sub[...,0:1,:], axis=-1), axis=-1)

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """跃迁规则（无修改）"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """【核心修复】用 sampler.hilbert 替代自定义hi_ext（NetKet标准写法，永不报错）"""
        sigma_shape = state.σ.shape
        # 直接从采样器获取希尔伯特空间（官方标准用法，100%兼容JAX）
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100  # 防死循环
            def cond(c): 
                return (c[0] < max_tries) & c[2]
            
            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)  # 每次更新RNG，防死循环
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

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. 批量局域能量矩阵
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    #print(f'E_L_batch.shape={E_L_batch.shape}')
    
    E_L_centered = E_L_batch - E_L_mean
    
    tr_centered =  jnp.trace(E_L_centered, axis1=-2, axis2=-1) 

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. 计算 ∇logΨs
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 5. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean

```

### 3.2 局域能量矩阵

通过 Monte Carlo 采样，损失函数可以写成期望值形式：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})\right)\right]

$$

定义**局域能量矩阵**为：

$$
E\_L(\mathbf{x}) \equiv \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})

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


## 6.测试案例
```python

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine,init_sampler_state,\
    generate_random_initial_states,ha,SingleStateAnsatz,create_single_machine,\
        create_machine_matrix,Ham_psi,Ham_Psi,NES_loss_energy,nes_vmc_gradient,hi,E_fcis,mcmc_sampler_multichain,\
            compute_qgt
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
from collections import Counter
import numpy as np
K=2
hi_ext = hi**K

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =200
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4


total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    
    

optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # 转为jax数组（关键修复）

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)


# 采样器状态初始化（替代原 init_sampler_state）
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ==================== 训练循环（仅替换采样部分） ====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (NetKet 自定义采样器 + 朴素梯度下降)")
print("="*60)
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")

history = {
    'step': [],
    'energy_0st': [],
    'energy_1st': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix':[],
    'natural_grad':[],
    'grad_flat':[],
    'samples':[],
    'log_Psi':[],
    'log_M':[],
    'log_Psi_mean':[],
    'log_Psi_min':[],
    'log_Psi_max':[],
    'grad_norm':[],
}

start_time = time.time()
for step in range(N_ITER):
    # 2. 正式采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine, parameters=total_params, 
        state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
        # 3. 维度重塑，适配梯度函数输入
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)
    # 3. 计算能量和自然梯度（逻辑和原代码一致）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))
    #grad = jax.tree_util.tree_map(lambda x: x * 2, grad)
    
    grad_flat , grad_unravel_fn = ravel_pytree(grad)
    # qgt_reg, unravel_fn = compute_qgt(total_machine, total_params, samples.reshape(-1,2,4), diag_shift=0.1)
    
    # # # 自然梯度求解
    # natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    # natural_grad = grad_unravel_fn(natural_grad_flat)
    # grad = natural_grad
        
    # 4. 更新参数
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)
    
    
    log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
    grad_norm = jnp.linalg.norm(grad_flat)
    
    
    history['step'].append(step)
    history['E_Lmatrix'].append(E_L_mean)
    history['samples'].append(samples)
    history['loss'].append(loss_mean)
    history['log_Psi_mean'].append(log_Psi_batch.mean())
    history['log_Psi_min'].append(log_Psi_batch.min())
    history['log_Psi_max'].append(log_Psi_batch.max())
    history['grad_norm'].append(grad_norm)
    history['energy_0st'].append(eig_vals[0])
    history['energy_1st'].append(eig_vals[1])
    history['params'].append(total_params)
    # 5. 记录历史
    if step % 50 == 0 or step == N_ITER - 1:
        # --------------------- 【NES-VMC 监控模板】直接用 ---------------------
        # 1. 监控 log_Psi
        #log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")

        # 2. 监控梯度范数
        
        print(f"grad norm = {grad_norm:.4f}")

        # 5. 局域能量矩阵
        #print(f"E_L mean =\n{E_L_mean}")
    
        #eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        # #history['natural_grad'].append(natural_grad)
        # history['grad_flat'].append(grad_flat)
        # history['log_Psi'].append(log_Psi)
        # history['log_M'].append(log_M)
        
        print(f"Step {step:3d} | Loss: {loss_mean}|0st能量={eig_vals[0]:.8f} Ha| 1st能量={eig_vals[1]:.8f} Ha")
        # print(f'grad={grad_flat[30:31]}')
        print('#-----------------------------------------#')


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
输出是:
```python

============================================================
开始多链 NES-VMC 训练 (NetKet 自定义采样器 + 朴素梯度下降)
============================================================
基态能量=-1.01546825 Ha| 第一激发态能量=-0.87542794 Ha| 第二激发态能量=-0.42938376 Ha
log_Psi: mean=1.338+0.070j | min=0.024-3.083j | max=1.513+0.075j
grad norm = 0.8250
Step   0 | Loss: -1.2576295690685688|0st能量=-0.99557002 Ha| 1st能量=-0.26205955 Ha
#-----------------------------------------#
log_Psi: mean=11.451-0.414j | min=9.999-2.433j | max=11.630+1.492j
grad norm = 0.5640
Step  50 | Loss: -1.547645795833384|0st能量=-56.85370946 Ha| 1st能量=55.30606367 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 100 | Loss: -1.5937140959638543|0st能量=-0.94145467 Ha| 1st能量=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 150 | Loss: -1.5937140959638543|0st能量=-0.94145467 Ha| 1st能量=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=10.994+0.302j | min=9.058-2.065j | max=11.009+2.080j
grad norm = 0.1187
Step 199 | Loss: -1.5891390172196438|0st能量=-1.60155252 Ha| 1st能量=0.01241350 Ha
#-----------------------------------------#
...

============================================================
训练完成!
============================================================
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

```