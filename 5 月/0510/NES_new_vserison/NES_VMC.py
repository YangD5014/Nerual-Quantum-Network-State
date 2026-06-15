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
import flax.nnx as nnx
import optax
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import time
from functools import partial
from jax.flatten_util import ravel_pytree
from collections import Counter

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
#K=3
# hi_ext = hi**K
# edges = [(0, 1), (2, 3),(4, 5),(6,7)]

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
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
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

def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

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



# total_ansatz = NESTotalAnsatz(4,K,8,rngs=nnx.Rngs(12))
# machine, graphdef, params = create_machine(total_ansatz)

def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])

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

def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 添加正则化项，防止矩阵奇异
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv


# @partial(jax.vmap, in_axes=(None, None, None, 0))
# def compute_local_energy_matrix_batch(ha: nk.operator.DiscreteOperator,graphdef, params, x_batch):
#     loss_val, E_L = NES_loss_energy(ha, graphdef, params, x_batch)
#     return E_L

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


def compute_nes_qgt(machine, graphdef, params, samples, diag_shift=0.01):
    """
    ###########################################################
    ✅ 最终无报错版 | 完全适配 model(x) = (ln detM, lnM[2,2])
    ✅ 严格遵循你的 QGT 公式：<g*g> - <g*><g>
    ✅ 解决 JAX 只能对标量求导的核心限制
    ###########################################################
    """

    # --------------------------
    # 1. 前向：返回 展平后的 lnM (向量，不是矩阵！)
    # --------------------------
    def forward_logM_flat(params, x):
        model = nnx.merge(graphdef, params)
        ln_det_M, ln_M = model(x)        # ln_M: [2,2] 矩阵
        return jnp.ravel(ln_M)            # ✅ 转成向量 [4]，JAX 允许求导

    # --------------------------
    # 2. 单样本梯度（对标量求和后求导，完全合法）
    # --------------------------
    def grad_single(x):
        # 定义：对 展平向量的“实部+虚部”求和 → 变成标量
        def scalar_forward(params, x):
            f = forward_logM_flat(params, x)
            return jnp.real(f).sum() + 1j * jnp.imag(f).sum()

        # 对标量求导 → 不报错！
        grad = jax.grad(scalar_forward, holomorphic=True)(params, x)
        grad_flat, _ = ravel_pytree(grad)
        return grad_flat

    # --------------------------
    # 3. 批量所有样本
    # --------------------------
    grads = jax.vmap(grad_single)(samples)    # [N_samples, N_params]

    # --------------------------
    # 4. 严格按你的公式计算 QGT
    # --------------------------
    term1 = jnp.mean(grads[..., None] * grads[:, None, :].conj(), axis=0)
    g_mean = jnp.mean(grads, axis=0)
    term2 = g_mean[..., None] * g_mean[None, :].conj()
    S = term1 - term2

    # 正则化
    S_reg = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)

    return S_reg, ravel_pytree(params)[1]


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
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
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

def sampler_info(samples:jnp.array,K:int):
    test_samples = np.array(samples.reshape(-1, 4*K))
    count = Counter(tuple(each_row.tolist()) for each_row in test_samples)
    for tpl, count_ in count.items():
        print(f"元组 {tpl} 出现了 {count_} 次")
    return count

import jax
import jax.numpy as jnp
import netket as nk

SINGLE_SIZE = hi.size

@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray
    K: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        """NES约束：子组态不重复
        🔥 核心修复：返回【标量布尔值】，匹配while_loop初始值形状
        """
        sub = sigma_ext.reshape((-1, self.K, self.single_size))
        # 原代码返回数组 → 改为 .squeeze() 压缩成标量！
        return jnp.any(jnp.all(sub[...,1:,:] == sub[...,0:1,:], axis=-1), axis=-1).squeeze()

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """跃迁规则（完全不变）"""
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
        """随机态生成（仅修复标量形状）"""
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
                is_dup = self._check_duplicate(s)  # 现在是标量！
                return (tries + 1, k, is_dup, s)
            
            # 初始值 c[2] = True（标量布尔值），和body返回值形状完全匹配
            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)
        
        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)


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
