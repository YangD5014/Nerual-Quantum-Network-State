我想基于自制 Sampler 来计算自然梯度 VMC 的梯度。
以下是不可以更改的代码，主要是一些问题背景的描述 已经必备的工具函数  
我需要额外强调一下 我对 edges 的定义应该是正确的：这是一个 STO3G 基组下的 H₂ 分子，二次量子化表征的情况下, 且满足费米子数目守恒、自旋守恒的前提下 跃迁路径也就只有[(0,1),(2,3)] 注意默认的排序是[α1 α2 β1 β2]
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
```
接下来的是, 我自制的 MCMC Sampler 相关代码
你要重点关注 给出合理的修改意见
```python

import jax
import jax.numpy as jnp
from functools import partial

def make_get_all_next_states(edges):
    edges = tuple(tuple(e) for e in edges)

    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        """
        适配多链：S形状从 (n_orbitals,) → (n_chains, n_orbitals)
        """
        next_states = []
        masks = []

        for (i, j) in edges:
            occ_i = S[..., i]  # 多链维度：(n_chains,)
            occ_j = S[..., j]
            valid = (occ_i == 1) & (occ_j == 0) | (occ_i == 0) & (occ_j == 1)
            # 对每条链单独翻转i/j位
            new_state = S.at[..., i].set(occ_j).at[..., j].set(occ_i)
            next_states.append(new_state)
            masks.append(valid)

        # 输出形状：(n_edges, n_chains, n_orbitals) 和 (n_edges, n_chains)
        return jnp.stack(next_states), jnp.stack(masks)
    
    return get_all_next_states_jit

# ==============================
# 改造1：MH步骤适配多链
# ==============================
def make_metropolis_hastings_step(edges, machine, params):
    get_all_next = make_get_all_next_states(edges)
    
    @jax.jit
    def metropolis_hastings_step_jit(S: jnp.ndarray, key: jax.Array):
        """
        S: (n_chains, n_orbitals) → 多链状态
        key: 随机数种子（每条链独立拆分）
        """
        n_chains = S.shape[0]
        # 1. 生成候选状态：(n_edges, n_chains, n_orbitals)
        candidates, valid_mask = get_all_next(S)  # valid_mask: (n_edges, n_chains)
        
        # 2. 每条链独立选候选（避免所有链选同一个edge）
        key, subk = jax.random.split(key)
        subkeys = jax.random.split(subk, n_chains)  # (n_chains, 2)
        # 对每条链采样候选索引
        idx = jax.vmap(lambda k: jax.random.choice(k, candidates.shape[0]))(subkeys)  # (n_chains,)
        
        # 3. 按索引取候选状态（多链）
        # 先构造索引：(n_chains,) → (n_chains, 2) (edge_idx, chain_idx)
        chain_idx = jnp.arange(n_chains)
        S_cand = candidates[idx, chain_idx]  # (n_chains, n_orbitals)
        is_valid = valid_mask[idx, chain_idx]  # (n_chains,)

        # 4. 计算接受率（多链并行）
        log_psi_curr = machine(params, S)  # (n_chains,)
        log_psi_cand = machine(params, S_cand)  # (n_chains,)
        log_accept_ratio = 2 * jnp.real(log_psi_cand - log_psi_curr)  # (n_chains,)

        # 5. 每条链独立判断是否接受
        key, subk = jax.random.split(key)
        subkeys = jax.random.split(subk, n_chains)
        u = jax.vmap(lambda k: jax.random.uniform(k))(subkeys)  # (n_chains,)
        accept = is_valid & (log_accept_ratio > jnp.log(u))  # (n_chains,)

        # 6. 更新每条链的状态
        S_new = jnp.where(accept[:, None], S_cand, S)  # 广播accept到(n_chains, n_orbitals)
        return S_new, accept, key

    return metropolis_hastings_step_jit

# ==============================
# 改造2：多链采样器核心
@partial(jax.jit, static_argnums=(0,1,3,4,6))
def mcmc_sampler_multichain(
    n_samples_per_chain: int,  
    n_warmup: int,
    initial_states: jnp.ndarray,  # (n_chains, n_orbitals)
    edges: tuple[tuple[int, int]],
    machine: callable,
    params: dict,  # 注意：params是PyTree，不是jnp.ndarray
    seed: int = 42
):
    # 关键修改1：为每条链生成独立的随机数种子
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, initial_states.shape[0])  # (n_chains, 2) → 每条链独立key
    
    # 绑定params到MH步骤（移除params绑定，改为每次传入）
    mh_step = make_metropolis_hastings_step(edges, machine)  # 改造mh_step，不提前绑定params

    # 预烧：每条链独立warmup
    def warmup_loop(carry, _):
        states, rngs = carry  # rngs: (n_chains, 2)
        # 每条链独立执行MH步骤
        def single_chain_step(state, rng):
            new_state, _, new_rng = mh_step(params, state, rng)  # 传入params
            return new_state, new_rng
        
        new_states, new_rngs = jax.vmap(single_chain_step)(states, rngs)
        return (new_states, new_rngs), None

    (current_states, chain_keys), _ = jax.lax.scan(
        warmup_loop,
        (initial_states, chain_keys),
        xs=None,
        length=n_warmup
    )

    # 采样：每条链独立采样 + 链内去相关（每隔10步取一个样本）
    sample_interval = 10  # 去相关步长，复刻NetKet的n_discard_per_chain
    effective_samples = n_samples_per_chain // sample_interval
    
    def sample_loop(carry, _):
        states, rngs = carry
        # 先执行sample_interval步MH，再取样本（去相关）
        def multi_step_chain(state, rng):
            def step(carry, _):
                s, r = carry
                new_s, _, new_r = mh_step(params, s, r)
                return (new_s, new_r), None
            (final_s, final_r), _ = jax.lax.scan(step, (state, rng), None, length=sample_interval)
            return final_s, final_r
        
        new_states, new_rngs = jax.vmap(multi_step_chain)(states, rngs)
        return (new_states, new_rngs), new_states

    (_, _), samples = jax.lax.scan(
        sample_loop,
        (current_states, chain_keys),
        xs=None,
        length=effective_samples  # 仅保留去相关后的样本
    )
    # samples形状：(effective_samples, n_chains, n_orbitals)
    samples = samples.reshape(-1, initial_states.shape[-1])
    return samples

# 同步修改make_metropolis_hastings_step：移除params提前绑定
def make_metropolis_hastings_step(edges, machine):
    get_all_next = make_get_all_next_states(edges)
    
    @jax.jit
    def metropolis_hastings_step_jit(params, S: jnp.ndarray, key: jax.Array):
        """
        修改：params从外部传入，而非提前绑定
        S: (n_orbitals,) → 单链状态（vmap后支持多链）
        key: 单链独立key
        """
        # 1. 生成候选状态：(n_edges, n_orbitals)
        candidates, valid_mask = get_all_next(S[None, ...])  # 扩展为(1, n_orbitals)适配get_all_next
        candidates = candidates[:, 0, :]  # (n_edges, n_orbitals)
        valid_mask = valid_mask[:, 0]    # (n_edges,)

        # 2. 单链选候选
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, len(edges))
        S_cand = candidates[idx]
        is_valid = valid_mask[idx]

        # 3. 计算接受率（单链）
        log_psi_curr = machine(params, S)
        log_psi_cand = machine(params, S_cand)
        log_accept_ratio = 2 * jnp.real(log_psi_cand - log_psi_curr)

        # 4. 单链判断接受
        key, subk = jax.random.split(key)
        u = jax.random.uniform(subk)
        accept = is_valid & (log_accept_ratio > jnp.log(u))

        # 5. 更新状态
        S_new = jnp.where(accept, S_cand, S)
        return S_new, accept, key

    return metropolis_hastings_step_jit

# ==============================
# 改造3：生成多链随机初始状态（模拟NetKet的默认行为）
# ==============================
def generate_random_initial_states(hilbert, n_chains: int, seed: int = 42):
    """
    模仿NetKet：从希尔伯特空间随机生成多链初始状态
    hilbert: NetKet的SpinOrbitalFermions希尔伯特空间
    n_chains: 链数
    """
    key = jax.random.PRNGKey(seed)
    # 希尔伯特空间的随机采样（NetKet内部逻辑）
    return hilbert.random_state(key, n_chains)


rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

samples = mcmc_sampler_multichain(
    n_samples_per_chain=100,
    n_warmup=100,
    initial_states=generate_random_initial_states(hi,16,2),
    edges=((0,1),(2,3)),
    machine=machine,
    params=params,
    seed=42
)
samples.shape

```

以下是主要的训练代码：
```python
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
N_WARMUP = 100

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
以上基于自制的 Sampler 的 VMC 训练结果：

============================================================
开始多链 VMC 训练 (自然梯度下降法)
============================================================
Step   0 | E: -0.50305463 ± 0.007761 | FCI: -1.01546825 | Error: 0.512414
Step  50 | E: -0.98545635 ± 0.002506 | FCI: -1.01546825 | Error: 0.030012
Step 100 | E: -1.00149581 ± 0.003946 | FCI: -1.01546825 | Error: 0.013972
Step 150 | E: -0.65240585 ± 0.000000 | FCI: -1.01546825 | Error: 0.363062
Step 200 | E: -0.65240585 ± 0.000000 | FCI: -1.01546825 | Error: 0.363062

最为核心的问题是: 1.相比于 Netket的 Sampler, 为什么我的基态能量求解结果，能量下降更慢？
如果使用 Netket的 Sampler，那么结果为:
============================================================
开始纯 JAX VMC 训练 (自然梯度下降法)
============================================================
/var/folders/8x/k_m4pmb11437ktb_r6tjzt2c0000gn/T/ipykernel_15019/1511232318.py:39: DeprecationWarning: jax.tree_map is deprecated: use jax.tree.map (jax v0.4.25 or newer) or jax.tree_util.tree_map (any JAX version).
  grad = jax.tree_map(lambda x: x*2, grad)
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

明显可以看出区别 我认为之所以会出现差异 是因为我自制的 Sampler 本质上没有并行多链采样 而是单链采样 
而 Netket 的 Sampler 是并行多链采样，所以人家的效率高。
原因是这样吗？我该怎么修改采样器才可以做到和 Netket 的 Sampler 一样高效呢？