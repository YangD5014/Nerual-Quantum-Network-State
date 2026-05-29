"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现 - 修复版

主要修复：
1. 使用正确的NetKet采样器API（ExchangeRule代替FermionHopRule）
2. 确保ha在正确的希尔伯特空间（4维）上工作
3. 正确构造扩展哈密顿量 H̃ = H ⊕ H
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
import time

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
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

# ==============================================================================
# 2. 希尔伯特空间和哈密顿量
# ==============================================================================
hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
print(f"\n单链希尔伯特空间: {hi.size} 维")

# 哈密顿量在单链（4维）上
ha = nkx.operator.from_pyscf_molecule(mol)
print(f"哈密顿量希尔伯特空间: {ha.hilbert.size} 维")

# 转换为JAX兼容版本（解决Numba算符不能在JAX变换内部使用的问题）
ha = ha.to_jax_operator()
print(f"哈密顿量已转换为JAX兼容版本")

# K=2扩展系统
K=2
SINGLE_HILBERT_SIZE = hi.size  # 4
hi_ext = hi ** K
print(f"扩展希尔伯特空间: {hi_ext.size} 维")

# 采样器设置 - 使用ExchangeRule（替代FermionHopRule）
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.ExchangeRule(graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)

# ==============================================================================
# 3. Ansatz 定义
# ==============================================================================
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
    """NES-VMC 总 Ansatz：构造矩阵 Ψ[x^i,j] = ψ_j(x^i)"""
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=rngs)
            for _ in range(self.K)
        ]

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
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

# ==============================================================================
# 4. Machine 创建
# ==============================================================================
def create_machine(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M = m(sigma)
        return log_psi_total

    return machine, graphdef, state

def create_machine_for_grad(model):
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)[0]
        return jnp.real(log_psi)
    return machine, graphdef, state

# ==============================================================================
# 5. 关键函数：Ham_psi 和 Ham_Psi
# ==============================================================================
def Ham_psi(ha, model, x):
    """
    计算 Hψ(x)
    ha: 哈密顿量（作用在 SINGLE_HILBERT_SIZE 维空间）
    model: 单态 ansatz
    x: 单个组态（ SINGLE_HILBERT_SIZE 维）
    """
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = jax.vmap(model)(x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

def Ham_Psi(ha, total_ansatz, x):
    """
    计算扩展哈密顿量作用在总 Ansatz 上的矩阵

    扩展哈密顿量 H̃ = H ⊕ H ⊕ ... ⊕ H
    对于 K=2: H̃ = H ⊕ H

    H_Psi[i,j] = ⟨x^i|H|ψ_j⟩ = Hψ_j(x^i)
    其中 x^i 是第 i 个副本的组态
    """
    hilber_size = total_ansatz.n_spin
    k = total_ansatz.K

    x_split = x.reshape(k, hilber_size)

    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x_split[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    H_Psi = jnp.array(H_psi_x_i).reshape(k, k)
    return H_Psi

# ==============================================================================
# 6. 损失函数和梯度
# ==============================================================================
def NES_loss_energy(ha, graphdef, params, x):
    """
    计算 NES-VMC 损失函数：L = Tr(Ψ^{-1} H̃ Ψ)

    这里 Ψ 是 K×K 矩阵，H̃ 是扩展哈密顿量
    """
    total_model = nnx.merge(graphdef, params)
    log_psi_det, log_M = total_model(x)
    Psi_Matrix = jnp.exp(log_M)

    H_Psi = Ham_Psi(ha, total_model, x)

    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_Psi)

    trace = jnp.trace(Psi_Matrix_inv)
    return jnp.real(trace), Psi_Matrix_inv

@partial(jax.vmap, in_axes=(None, None, None, 0))
def compute_local_energy_matrix_batch(ha, graphdef, params, x_batch):
    loss_val, E_L = NES_loss_energy(ha, graphdef, params, x_batch)
    return E_L

def nes_vmc_gradient(ha, graphdef, params, x_batch, machine_for_grad, vmap_grad_logPsi):
    """
    计算 NES-VMC 梯度
    ∇⟨E⟩ = ⟨ (Tr(E_loc) - Tr(E_mean)) * ∇logΨ* ⟩
    """
    E_L_batch = compute_local_energy_matrix_batch(ha, graphdef, params, x_batch)

    valid_mask = ~jnp.any(
        jnp.isnan(E_L_batch) | jnp.isinf(E_L_batch),
        axis=(1, 2)
    )
    E_L_batch = E_L_batch[valid_mask]
    x_batch = x_batch[valid_mask]

    if len(x_batch) == 0:
        print("警告：没有有效样本！")
        return {}, 0.0, jnp.zeros((K, K))

    E_L_mean = jnp.mean(E_L_batch, axis=0)

    tr_E_loc_batch = jnp.real(jnp.trace(E_L_batch, axis1=1, axis2=2))
    tr_E_mean = jnp.real(jnp.trace(E_L_mean))

    tr_centered = tr_E_loc_batch - tr_E_mean

    dlogPsi_batch = vmap_grad_logPsi(params, x_batch)

    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,)*(grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = jnp.mean(tr_E_loc_batch)
    return grad, loss_mean, E_L_mean

# ==============================================================================
# 7. QGT 计算（自然梯度）
# ==============================================================================
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    sigma = sigma.reshape(-1, K, SINGLE_HILBERT_SIZE)
    n_samples = sigma.shape[0]

    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean

    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn

# ==============================================================================
# 8. 采样器相关
# ==============================================================================
def generate_random_initial_states(hi_ext, n_chains, seed=42):
    """生成随机的初始状态，确保两个副本的组态不同"""
    import jax.random as jr

    key = jr.PRNGKey(seed)
    n_spin = hi_ext.size // K
    init_states = []

    for _ in range(n_chains):
        while True:
            s1 = hi_ext.random_state(key)
            key = jr.split(key)[1]
            s2 = hi_ext.random_state(key)
            key = jr.split(key)[1]

            x1 = s1[:n_spin]
            x2 = s2[n_spin:]

            if not jnp.all(x1 == x2):
                break

        ext_state = jnp.concatenate([x1, x2])
        init_states.append(ext_state)

    return jnp.stack(init_states)

def init_sampler_state(hi_ext, n_chains, seed=42):
    init_states = generate_random_initial_states(hi_ext, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)
    return (init_states, chain_keys)

def make_get_all_next_states(K, SINGLE_HILBERT_SIZE, edges):
    @jax.jit
    def get_all_next_states_jit(S):
        next_states = []
        valid_masks = []

        for (i, j) in edges:
            occ_i = S[..., i]
            occ_j = S[..., j]
            valid_hop = (occ_i != occ_j)

            new_state = S.at[..., i].set(occ_j).at[..., j].set(occ_i)

            x_list = jnp.split(new_state, K, axis=-1)

            has_duplicate = False
            for a in range(K):
                for b in range(a + 1, K):
                    equal = jnp.all(x_list[a] == x_list[b], axis=-1)
                    has_duplicate = jnp.logical_or(has_duplicate, equal)

            valid = valid_hop & (~has_duplicate)
            next_states.append(new_state)
            valid_masks.append(valid)

        return jnp.stack(next_states), jnp.stack(valid_masks)

    return get_all_next_states_jit

def make_metropolis_hastings_step(K, SINGLE_HILBERT_SIZE, edges, machine):
    get_all_next = make_get_all_next_states(K, SINGLE_HILBERT_SIZE, edges)

    @jax.jit
    def mh_step(params, state, key):
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

@partial(jax.jit, static_argnums=(0, 1, 3, 4, 6, 7, 8))
def mcmc_sampler_multichain(
    n_samples_per_chain, n_warmup, sampler_state, edges,
    machine, params, sweep_size=32, K=2, SINGLE_HILBERT_SIZE=4
):
    current_states, current_keys = sampler_state
    mh_step = make_metropolis_hastings_step(K, SINGLE_HILBERT_SIZE, edges, machine)

    def single_sweep(carry, _):
        states, keys = carry
        (new_s, new_k), _ = jax.lax.scan(
            lambda c, _: (jax.vmap(mh_step, in_axes=(None, 0, 0))(params, c[0], c[1]), None),
            (states, keys),
            length=sweep_size
        )
        return (new_s, new_k), new_s

    if n_warmup > 0:
        (current_states, current_keys), _ = jax.lax.scan(
            single_sweep, (current_states, current_keys), length=n_warmup
        )

    (final_states, final_keys), samples = jax.lax.scan(
        single_sweep, (current_states, current_keys), length=n_samples_per_chain
    )

    new_sampler_state = (final_states, final_keys)
    return samples, new_sampler_state

# ==============================================================================
# 9. 训练循环
# ==============================================================================
if __name__ == '__main__':
    N_CHAINS = 16
    N_WARMUP = 32
    N_SAMPLES_PER_CHAIN = 100
    SWEEP_SIZE = 32
    N_ITER = 100

    rngs = nnx.Rngs(21)
    model = NESTotalAnsatz(4, K, 12, rngs=rngs)
    machine, graphdef, params = create_machine(model)
    machine_for_grad, _, params_for_grad = create_machine_for_grad(model)

    grad_logPsi = jax.grad(lambda p, s: machine_for_grad(p, s), argnums=0)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(params)

    print("\n" + "="*60)
    print("开始多链 NES-VMC 训练 (自然梯度下降法)")
    print("="*60)

    history = {
        'step': [],
        'energy': [],
        'eig_vals': []
    }

    sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=21)
    start_time = time.time()

    for step in range(N_ITER):
        samples, sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=N_SAMPLES_PER_CHAIN,
            n_warmup=N_WARMUP,
            sampler_state=sampler_state,
            edges=((0, 1), (2, 3), (4, 5), (6, 7)),
            machine=machine,
            params=params,
        )

        grad, loss_mean, E_L_mean = nes_vmc_gradient(
            ha=ha,
            graphdef=graphdef,
            params=params,
            x_batch=samples.reshape(-1, K, SINGLE_HILBERT_SIZE),
            machine_for_grad=machine_for_grad,
            vmap_grad_logPsi=vmap_grad_logPsi
        )

        try:
            qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001)
            grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
            natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
            grad = grad_unravel_fn(natural_grad)
        except Exception as e:
            print(f"QGT计算出错，使用普通梯度: {e}")

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 5 == 0 or step == N_ITER - 1:
            eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
            history['step'].append(step)
            history['energy'].append(loss_mean)
            history['eig_vals'].append(eig_vals)

            print(f"Step {step:3d} | Loss: {loss_mean:.8f} | eig_vals: {eig_vals}")

    end_time = time.time()
    print(f"训练耗时：{end_time - start_time:.2f} 秒")

    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

    print("\n最终特征值（激发态能量）：")
    final_eig_vals = history['eig_vals'][-1]
    for i, e in enumerate(final_eig_vals):
        fci_e = E_fcis[i] if i < len(E_fcis) else None
        error = abs(e - fci_e) if fci_e else None
        print(f"  E{i} = {e:.8f} Ha (FCI: {fci_e:.8f} Ha) | 误差: {error:.6f} Ha" if error else f"  E{i} = {e:.8f} Ha")
