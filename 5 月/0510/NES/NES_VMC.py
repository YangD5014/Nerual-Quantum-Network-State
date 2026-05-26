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
    """
    NES-VMC 总波函数 Ansatz
    支持自动处理：
      - 单样本：输入 (K, n_spin_orbitals)
      - 批量样本：输入 (batch_size, K, n_spin_orbitals)
    输出：
      log_psi_total : 标量 或 (batch_size,)
      log_M_matrix  : (K,K) 或 (batch_size,K,K)
    """
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        # 初始化 K 个独立单态波函数
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        # ----------------------
        # 单样本处理函数 (核心)
        # x_single: (K, n_spin)
        # ----------------------
        #print(f'收到了x.shape={x.shape}')
        def _forward_single(x_single):
            # 构造 K×K 矩阵 M: M[i,j] = ψ_j(x_i)
            #print(f'x_single.shape={x_single.shape}')
            M = []
            for i in range(self.K):
                row = []
                for j in range(self.K):
                    val = self.single_ansatz_list[j](x_single[i])
                    row.append(val)
                M.append(jnp.stack(row))
            log_M = jnp.stack(M)  # (K, K)
            log_det = jnp.linalg.det(log_M)
            return log_det, log_M

        # ----------------------
        # 自动判断：单条 / 批量
        # ----------------------
        if x.shape[-1] == self.n_spin:
            #print('A')
            if x.ndim ==2:
                log_psi, log_M = _forward_single(x)
            elif x.ndim ==3:
                log_psi, log_M = jax.vmap(_forward_single)(x)
            else:
                raise ValueError(f"Input array must have shape ({self.K},) or got shape {x.shape}")
        elif x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, K, self.n_spin)
            #print(f'转换后x.shape={x.shape}')
            log_psi, log_M = jax.vmap(_forward_single)(x)
        else:
            raise ValueError(f"Input array must have shape ({self.K},) or ({self.K},) but got shape {x.shape}")
        return log_psi, log_M
    
    
    
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





def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


def Ham_psi(ha:nk.operator.DiscreteOperator, model, x):
    """计算 Hψ(x)"""
    x_primes, mels = ha.get_conn_padded(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def Ham_Psi(ha, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    k = total_ansatz.n_states
    if x.shape[0] != k:
        raise ValueError(f"Input array must have shape ({k},) but got shape {x.shape}")

    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi


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


def apply_hamiltonian_on_M(model, hamiltonian, M, sigma, K):
    """
    计算 扩展哈密顿量作用在矩阵 M 上： H_M = H_ext @ M
    对应公式： (H M)_{i,j} = ⟨x^i|H|ψ_j⟩ = sum_η H_{x^i η} ψ_j(η)
    """
    sigma = sigma.reshape(-1, K, 4)
    def apply_hamiltonian_to_M_row(M_row, x_i):
        # 获取 H 连接态 η 与矩阵元 ⟨η|H|x_i⟩
        eta_i, H_mat_i = hamiltonian.get_conn_padded(x_i[None, :])
        eta_i = eta_i[0]
        H_mat_i = H_mat_i[0]

        H_M_row = []
        for j in range(K):
            psi_j = model.single_ansatz_list[j]
            psi_j_eta = jax.vmap(psi_j)(eta_i)
            H_psi_j = jnp.sum(H_mat_i * psi_j_eta)
            H_M_row.append(H_psi_j)
        return jnp.stack(H_M_row)

    # 向量化：批处理 → 行处理
    batch_apply = jax.vmap(lambda m, s: jax.vmap(apply_hamiltonian_to_M_row)(m, s))
    # print(f'M.shape={M.shape}')
    # print(f'sigma.shape={sigma.shape}')
    H_M = batch_apply(M, sigma)
    return H_M



# ==============================================================================
# 🔥 函数 2：计算 NES-VMC 局域能量矩阵 E_L = M⁻¹ · H · M
# ==============================================================================
def compute_local_energy_matrix(model_graphdef, params, sigma, hamiltonian, K):
    """
    对外接口：计算局域能量矩阵
    公式：E_L(x) = M⁻¹(x) · H_ext · M(x)
    拆分为两步：
      1. 计算 H_M = H_ext · M
      2. 计算 E_L = M⁻¹ @ H_M
    """
    # 重建模型
    model = nnx.merge(model_graphdef, params)

    # 1. 前向传播得到 M 矩阵 和 logΨ
    log_psi, M = model(sigma)
    M += 0.01 * jnp.eye(K)

    # 2. 调用子函数：计算 H_ext · M
    H_M = apply_hamiltonian_on_M(model, hamiltonian, M, sigma, K)

    # 3. 矩阵求逆 + 乘法：E_L = M⁻¹ @ H_M
    def mat_solve(mat1, mat2):
        return jnp.linalg.solve(mat1, mat2)  # 🔥 这里是关键替换

    E_L = jax.vmap(mat_solve)(M, H_M)       # 🔥 vmap 不变

    return E_L, log_psi ,M


def compute_loss_and_grad(model_graphdef, params, sigma, hamiltonian, K):

    E_L, log_psi, log_M = compute_local_energy_matrix(model_graphdef, params, sigma, hamiltonian, K)
    tr_el = jnp.trace(E_L, axis1=-2, axis2=-1)  # (B,)
    loss = jnp.mean(tr_el)

    # ---------------------------
    # 3. 均值 & 中心化（正常）
    # ---------------------------
    E_L_mean = jnp.mean(E_L, axis=0)
    #print(E_L_mean)
    E_L_centered = E_L - E_L_mean
    weights = jnp.trace(E_L_centered, axis1=-2, axis2=-1)  # (B,)

    def total_loss(p):
        model = nnx.merge(model_graphdef, p)
        logp, _ = model(sigma)
        el, _, _ = compute_local_energy_matrix(model_graphdef, p, sigma, hamiltonian, K)
        return jnp.real(jnp.mean(jnp.trace(el, axis1=-2, axis2=-1)))

    # 直接对总损失求导 → 完全稳定
    grads = grad(total_loss)(params)

    # ---------------------------
    # 5. 返回
    # ---------------------------
    return loss, grads, E_L_mean


# ==============================================================================
# ✅ 1. 【严格协方差版】QGT 计算（VMC / NES-VMC 标准定义）
# ==============================================================================
@partial(jax.jit, static_argnames=("model_graphdef",))
def compute_QGT(model_graphdef, params, sigma):
    """
    标准 QGT = Cov( ∇logΨ, ∇logΨ† )
    完全符合你说的：QGT = 梯度协方差矩阵
    """
    def log_psi(p, x):
        model = nnx.merge(model_graphdef, p)
        log_p, _ = model(x)
        return jnp.real(log_p)  # log|Ψ|

    # 计算批量梯度 ∇logΨ
    grad_log = grad(log_psi, argnums=0)
    batch_g = vmap(grad_log, (None, 0))(params, sigma)  # (B, ...)

    # 均值 E[g]
    mean_g = jax.tree.map(lambda g: jnp.nanmean(g, axis=0), batch_g)

    # 协方差 E[gg†] - E[g]E[g]†
    def qgt_cov(g, mg):
        eg = jnp.nanmean(g * jnp.conj(g), axis=0)
        return eg - mg * jnp.conj(mg)

    S = jax.tree.map(qgt_cov, batch_g, mean_g)
    return S

@jax.jit
def apply_natural_gradient(grads, S, eps=1e-4):
    return jax.tree.map(lambda g, s: g / (s + eps), grads, S)



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



def generate_random_initial_states(hi, n_chains: int, seed: int = 42):
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, n_chains)
    return jax.vmap(lambda k: hi.random_state(k))(keys)


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
    n_chains = current_states.shape[0]
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

# 初始化链状态 + 随机数状态（构成 sampler_state）
def init_sampler_state(hi, n_chains, seed=42):
    init_states = generate_random_initial_states(hi, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)  # 每条链独立随机数
    return (init_states, chain_keys)


