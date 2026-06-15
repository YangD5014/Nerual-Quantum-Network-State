"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现
适配 H2O 分子 / STO-3G 基组 / 前 3 个激发态 (K=4)

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，
在 STO-3G 基组下计算 H2O 分子的前 3 个激发态能量。
H2O 配置：O-H 键长 0.9572 Å，H-O-H 键角 104.52° (标准实验几何)
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
# 1. H2O 分子定义 & FCI 基准
# ==============================================================================
# H2O 标准实验几何
# O-H 键长 = 0.9572 Å，H-O-H 键角 = 104.52°
# 1 Å = 1.8897259886 bohr
# 转换：r_OH = 0.9572 * 1.8897 = 1.8089 bohr
# 键角 = 104.52°，半角 = 52.26°
# H 坐标: (±r*sin(52.26°), r*cos(52.26°), 0) = (±1.4309, 1.1072, 0)
bond_length = 0.9572  # Å
angle = 104.52        # degree
half_angle = np.deg2rad(angle / 2.0)
r_bohr = bond_length * 1.8897259886
hx = r_bohr * np.sin(half_angle)
hy = r_bohr * np.cos(half_angle)

geometry = [
    ('O', (0., 0., 0.)),
    ('H', (hx, hy, 0.)),
    ('H', (-hx, hy, 0.)),
]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准：需要 4 个能量 (基态 + 3 个激发态)
n_excited = 3
n_states = n_excited + 1  # K = 4
cisolver = fci.FCI(mf)
cisolver.nroots = n_states
E_fcis, fcivec = cisolver.kernel()
print("=" * 60)
print("H2O / STO-3G FCI 基准能量")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")
# ===================== NetKet 哈密顿量 =====================
ha = nkx.operator.from_pyscf_molecule(mol)
# 转换为 JAX 兼容格式 (否则 Numba operator 不能在 jax.jit / jax.grad 中使用)
ha = ha.to_jax_operator()

# ===================== H2O 希尔伯特空间 =====================
# H2O: O(1s,2s,2p) + H(1s) + H(1s) = 7 空间轨道
# 自旋轨道数 = 2 * 7 = 14
# 电子数: O(8e) + H(1e) + H(1e) = 10e -> alpha=5, beta=5
n_orbitals = mol.nao_nr()  # 应该是 7
n_electrons = mol.nelectron  # 应该是 10
n_alpha = n_electrons // 2
n_beta = n_electrons - n_alpha
print(f"H2O 空间轨道数 = {n_orbitals} | alpha 电子 = {n_alpha} | beta 电子 = {n_beta}")

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(n_alpha, n_beta),
)
SINGLE_SIZE = hi.size  # 14 (自旋轨道数)
print(f"SINGLE_SIZE (单子系统维度) = {SINGLE_SIZE}")

# ===================== K = 4 (基态 + 3 激发态) =====================
K = 4
hi_ext = hi ** K

# ===================== 费米子跃迁边 =====================
# 自旋块内 (alpha 块 [0..n_orbitals-1] 和 beta 块 [n_orbitals..2*n_orbitals-1])
# 同一自旋块内的两轨道交换，alpha+alpha / beta+beta
def build_single_edges(n_orb):
    edges = []
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            edges.append((i, j))  # alpha 块
    for i in range(n_orb):
        for j in range(i + 1, n_orb):
            edges.append((n_orb + i, n_orb + j))  # beta 块
    return tuple(edges)

single_edges = build_single_edges(n_orbitals)
print(f"单系统跃迁边数 = {len(single_edges)}")

# ==============================================================================
# 2. 神经网络 Ansatz 定义
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 32, *, rngs: nnx.Rngs):
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
    NES 总 Ansatz：行列式结构
    Ψ(x¹, ..., xᴷ) = det[ψᵢ(xⱼ)]
    """

    def __init__(self, n_spin_orbitals: int, n_states: int = 4, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        # 使用普通 Python list（兼容 Flax NNX 0.8.5 + 新版本）
        self.single_ansatz_list = []
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


def create_machine(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数 (返回 log|Ψ|)"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_psi_total

    return machine, graphdef, state


def create_single_machine(model: SingleStateAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数 (返回单态 log ψ)"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state


def create_machine_matrix(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数 (返回 log M 矩阵)"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix = m(sigma)
        return log_M_matrix

    return machine, graphdef, state


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    计算 H|ψ>(x)，支持单样本和批处理
    """
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]

    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    H_psi_batch = jax.vmap(_single_hpsi)(x)

    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch


def Ham_Psi(ha, single_machine_list, total_params, x):
    """计算 H Ψ 矩阵 (K x K)，支持单/批输入"""
    K_local = len(single_machine_list)
    if x.ndim == 2:
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K_local, K_local), dtype=complex)
            for i in range(K_local):
                xi = x_single[i]
                for j in range(K_local):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return _single_HamPsi(x)
    elif x.ndim == 3:
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K_local, K_local), dtype=complex)
            for i in range(K_local):
                xi = x_single[i]
                for j in range(K_local):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return jax.vmap(_single_HamPsi)(x)
    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")


def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)
    Psi_Matrix = jnp.exp(log_M)
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv


def nes_vmc_gradient(ha, total_matrix_machine, total_machine, single_machine_list, total_params, x_batch):
    loss_batch, E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    E_L_centered = E_L_batch - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean


def generate_random_initial_states(hi_ext, n_chains, seed=42):
    """生成不重复的扩展态初始样本"""
    key = jax.random.PRNGKey(seed)
    n_spin = hi_ext.size // K
    init_states = []
    for _ in range(n_chains):
        while True:
            key, k1, k2 = jax.random.split(key, 3)
            ext = hi_ext.random_state(k1)
            sub = ext.reshape(K, n_spin)
            unique = True
            for a in range(K):
                for b in range(a + 1, K):
                    if jnp.all(sub[a] == sub[b]):
                        unique = False
                        break
                if not unique:
                    break
            if unique:
                init_states.append(ext)
                break
    return jnp.stack(init_states)


def init_sampler_state(hi, n_chains, seed=42):
    init_states = generate_random_initial_states(hi, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)
    return (init_states, chain_keys)


def make_get_all_next_states(K_local, single_size, edges):
    """生成所有候选跃迁态 + 有效掩码 (约束子态不重复)"""
    edges_arr = jnp.array(edges)

    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        # edges 列表里的所有 (i, j) 对同时计算候选态
        # 这里采用简单实现：单条边调用一次
        def single_edge(S, edge):
            i, j = edge[0], edge[1]
            occ_i = S[i]
            occ_j = S[j]
            valid_hop = (occ_i != occ_j)
            new_state = S.at[i].set(occ_j).at[j].set(occ_i)
            x_list = jnp.split(new_state, K_local, axis=-1)
            has_dup = jnp.array(False)
            for a in range(K_local):
                for b in range(a + 1, K_local):
                    eq = jnp.all(x_list[a] == x_list[b])
                    has_dup = jnp.logical_or(has_dup, eq)
            valid = valid_hop & (~has_dup)
            return new_state, valid

        # vmap over edges
        candidates, valids = jax.vmap(lambda e: single_edge(S, e))(edges_arr)
        return candidates, valids

    return get_all_next_states_jit


def make_metropolis_hastings_step(K_local, single_size, edges, machine):
    get_all_next = make_get_all_next_states(K_local, single_size, edges)

    @jax.jit
    def mh_step(params, state: jnp.ndarray, key: jax.Array):
        candidates, valid_mask = get_all_next(state)
        # candidates: (n_edges, single_size*K)
        key, subk = jax.random.split(key)
        idx = jax.random.randint(subk, (), 0, candidates.shape[0])
        cand = candidates[idx]
        is_valid = valid_mask[idx]

        log_curr = machine(params, state)
        log_cand = machine(params, cand)
        # |Ψ(cand)/Ψ(state)|^2 = exp(2 Re(log_cand - log_curr))
        log_acc = 2 * jnp.real(log_cand - log_curr)

        key, subk = jax.random.split(key)
        accept = is_valid & (log_acc > jnp.log(jax.random.uniform(subk)))
        new_state = jnp.where(accept, cand, state)
        return new_state, key

    return mh_step


@partial(jax.jit, static_argnums=(0, 1, 3, 4, 6, 7, 8))
def mcmc_sampler_multichain(
    n_samples_per_chain,
    n_warmup,
    sampler_state,
    edges,
    machine,
    params,
    sweep_size=32,
    K_local=4,
    SINGLE_HILBERT_SIZE=14,
):
    edges_arr = jnp.array(edges)
    current_states, current_keys = sampler_state
    mh_step = make_metropolis_hastings_step(K_local, SINGLE_HILBERT_SIZE, edges_arr, machine)

    def single_sweep(carry, _):
        states, keys = carry
        (new_s, new_k), _ = jax.lax.scan(
            lambda c, _: (jax.vmap(mh_step, in_axes=(None, 0, 0))(params, c[0], c[1]), None),
            (states, keys),
            xs=None,
            length=sweep_size
        )
        return (new_s, new_k), new_s

    if n_warmup > 0:
        (current_states, current_keys), _ = jax.lax.scan(
            single_sweep, (current_states, current_keys), xs=None, length=n_warmup
        )

    (final_states, final_keys), samples = jax.lax.scan(
        single_sweep, (current_states, current_keys), xs=None, length=n_samples_per_chain
    )

    new_sampler_state = (final_states, final_keys)
    return samples, new_sampler_state


def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    量子几何张量 QGT = <g*g> - <g*><g>
    """
    n_samples = sigma.shape[0]

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: machine(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    return qgt_reg, unravel_fn


@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    """
    NetKet 自定义 Metropolis 跃迁规则：
    - 同一自旋块内 (alpha 或 beta) 随机选一对轨道做费米子跃迁
    - 拒绝会导致任意两子组态重复的候选态
    """
    edges: jnp.ndarray
    K_local: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        sub = sigma_ext.reshape((-1, self.K_local, self.single_size))
        return jnp.any(jnp.all(sub[..., 1:, :] == sub[..., 0:1, :], axis=-1), axis=-1).squeeze()

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:, 0], sel_e[:, 1]

        sigma_cand = sigma.at[jnp.arange(batch_size), i].set(sigma[jnp.arange(batch_size), j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size), j].set(sigma[jnp.arange(batch_size), i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)
        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        sigma_shape = state.σ.shape
        hilbert = sampler.hilbert
        # NetKet sampler 要求 float64
        target_dtype = jnp.float64

        def gen_single(key):
            max_tries = 100

            def cond(c):
                return (c[0] < max_tries) & c[2]

            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)
                s = hilbert.random_state(k_new).astype(target_dtype)
                is_dup = self._check_duplicate(s)
                return (tries + 1, k, is_dup, s)

            init_c = (0, key, True, hilbert.random_state(key).astype(target_dtype))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            fallback = hilbert.random_state(key).astype(target_dtype)
            return jax.lax.cond(is_dup, lambda: fallback, lambda: s)

        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)


def sampler_info(samples, K_local):
    test_samples = np.array(samples.reshape(-1, SINGLE_SIZE * K_local))
    count = Counter(tuple(each_row.tolist()) for each_row in test_samples)
    for tpl, count_ in count.items():
        print(f"元组 {tpl} 出现了 {count_} 次")
    return count
