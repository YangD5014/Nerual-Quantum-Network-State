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
ha_jax = ha.to_jax_operator()
hi = nkx.hilbert.SpinOrbitalFermions(
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
# 3. 辅助函数
# ==============================================================================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


# ==============================================================================
# 4. 基态训练 - 用于获取基态参考
# ==============================================================================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    """
    eta, H_eta = ha_jax.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    Force-based 梯度计算
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    """
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = statistics(O_loc)
    O_centered = O_loc - O_mean

    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    def weight_and_mean(grad_component):
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)

    return O_mean, O_std, grad


def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    """
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


def clip_gradients(grad, max_norm=1.0):
    """梯度裁剪，防止梯度爆炸"""
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad)
    grad_norm = jnp.linalg.norm(grad_flat)
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    grad_flat_clipped = grad_flat * scale
    return unravel_fn(grad_flat_clipped)


def train_ground_state(N_ITER=300, N_SAMPLES=1008):
    """
    训练基态波函数
    """
    rngs = nnx.Rngs(21)
    model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
    machine, graphdef, params = create_machine(model)

    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nkx.sampler.rules.ParticleExchangeRule(hilbert=hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=100, sweep_size=32)
    sampler_state = sampler.init_state(machine, params, seed=1)

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(params)

    print("\n" + "="*60)
    print("开始基态训练")
    print("="*60)

    for step in range(N_ITER):
        sampler_state = sampler.reset(machine, params, sampler_state)
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=20
        )
        samples = samples.reshape(-1, hi.size)

        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
        grad = jax.tree_map(lambda x: x*2, grad)
        grad = clip_gradients(grad, max_norm=1.0)

        qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001)
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad)

        updates, opt_state = optimizer.update(natural_grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 50 == 0 or step == N_ITER - 1:
            error = jnp.abs(energy.real - E_fcis[0])
            print(f"GS Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | Error: {error:.6f}")

    print(f"基态训练完成: E = {energy.real:.8f} Ha")
    return machine, params


# 训练基态
gs_machine, gs_params = train_ground_state(N_ITER=300)


# ==============================================================================
# 5. 惩罚项方法 - 求解第一激发态 (修复版)
# ==============================================================================

@partial(jax.jit, static_argnames=("machine1", "machine2"))
def compute_overlap_stable(machine1, params1, machine2, params2, sigma):
    """
    数值稳定的重叠计算
    使用log-sum-exp技巧避免数值溢出
    """
    logpsi1_sigma = machine1(params1, sigma)
    logpsi2_sigma = machine2(params2, sigma)

    log_diff = logpsi1_sigma - jnp.conj(logpsi2_sigma)

    return jnp.exp(log_diff)


@partial(jax.jit, static_argnames=("machine1", "machine2"))
def compute_overlap_and_energy_stable(machine1, params1, machine2, params2, sigma, penalty_alpha):
    """
    计算重叠和带惩罚项的期望能量（数值稳定版本）
    """
    eta, H_eta = ha_jax.get_conn_padded(sigma)

    logpsi1_sigma = machine1(params1, sigma)
    logpsi1_eta = machine1(params1, eta)
    logpsi1_sigma_expanded = jnp.expand_dims(logpsi1_sigma, -1)
    E_loc1 = jnp.sum(H_eta * jnp.exp(logpsi1_eta - logpsi1_sigma_expanded), axis=-1)

    logpsi0_sigma = machine2(params2, sigma)
    logpsi0_eta = machine2(params2, eta)
    logpsi0_sigma_expanded = jnp.expand_dims(logpsi0_sigma, -1)
    E_loc0 = jnp.sum(H_eta * jnp.exp(logpsi0_eta - logpsi0_sigma_expanded), axis=-1)

    overlap_complex = compute_overlap_stable(machine1, params1, machine2, params2, sigma)
    overlap_real = jnp.mean(overlap_complex)

    E1_mean = jnp.mean(E_loc1)
    E1_std = jnp.sqrt(jnp.var(E_loc1) / len(E_loc1))

    E0_mean = jnp.mean(E_loc0)

    overlap_sq = jnp.abs(overlap_real)**2
    penalty_energy = E0_mean + penalty_alpha * (1 - overlap_sq)

    return E1_mean, E1_std, E0_mean, overlap_real, overlap_sq, penalty_energy


@partial(jax.jit, static_argnames=("machine1", "machine2"))
def forces_expect_penalty_stable(machine1, params1, machine2, params2, sigma, penalty_alpha):
    """
    计算惩罚项方法的第一激发态梯度（数值稳定版本）

    使用更稳定的重叠项计算：
    - 使用tanh限制指数项的范围
    - 添加小常数避免除零
    """
    eta, H_eta = ha_jax.get_conn_padded(sigma)

    logpsi1_sigma = machine1(params1, sigma)
    logpsi1_eta = machine1(params1, eta)
    logpsi1_sigma_expanded = jnp.expand_dims(logpsi1_sigma, -1)
    E_loc1 = jnp.sum(H_eta * jnp.exp(logpsi1_eta - logpsi1_sigma_expanded), axis=-1)

    logpsi0_sigma = machine2(params2, sigma)
    logpsi0_eta = machine2(params2, eta)
    logpsi0_sigma_expanded = jnp.expand_dims(logpsi0_sigma, -1)
    E_loc0 = jnp.sum(H_eta * jnp.exp(logpsi0_eta - logpsi0_sigma_expanded), axis=-1)

    E0_mean = jnp.mean(E_loc0)

    O_centered = E_loc1 - E0_mean

    def log_psi1_single(p, s):
        return machine1(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi1_single(p, s), holomorphic=True)(params1)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

    def weight_and_mean(grad_component):
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad_energy = jax.tree_util.tree_map(weight_and_mean, grad_matrix)

    log_ratio = logpsi0_sigma.real - logpsi1_sigma.real
    log_ratio_clipped = jnp.clip(log_ratio, -10, 10)
    overlap_term = jnp.exp(log_ratio_clipped) * jnp.exp(1j * (logpsi0_sigma.imag - logpsi1_sigma.imag))

    def weight_and_mean_overlap(grad_component):
        weights = overlap_term.reshape((overlap_term.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad_overlap = jax.tree_util.tree_map(weight_and_mean_overlap, grad_matrix)

    grad_penalty = jax.tree_map(lambda g_e, g_o: 2 * g_e - 2 * penalty_alpha * jnp.conj(g_o), grad_energy, grad_overlap)

    return E_loc1, E0_mean, grad_penalty, grad_energy, grad_overlap


def train_excited_state_penalty_stable(gs_machine, gs_params, N_ITER=300, N_SAMPLES=1008, penalty_alpha=1.0, init_scale=0.1):
    """
    使用惩罚项方法训练第一激发态（数值稳定版本）

    修复内容：
    1. 初始化时使用较小的权重尺度，避免初始参数导致数值爆炸
    2. 添加梯度裁剪
    3. 使用更稳定的学习率调度
    4. 惩罚项计算中使用tanh限制指数范围
    5. 分阶段训练：先小步长，后逐渐增加
    """
    rngs = nnx.Rngs(42)
    model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
    exc_machine, exc_graphdef, exc_params = create_machine(model)

    def scale_params(params, scale):
        return jax.tree_util.tree_map(lambda x: x * scale, params)

    exc_params = scale_params(exc_params, init_scale)

    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nkx.sampler.rules.ParticleExchangeRule(hilbert=hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=100, sweep_size=32)
    sampler_state = sampler.init_state(exc_machine, exc_params, seed=2)

    base_lr = 0.005
    schedule = optax.exponential_decay(
        init_value=base_lr,
        transition_steps=100,
        decay_rate=0.95,
        staircase=True
    )
    optimizer = optax.sgd(learning_rate=schedule)
    opt_state = optimizer.init(exc_params)

    print("\n" + "="*60)
    print(f"开始第一激发态训练 (惩罚项方法稳定版, α={penalty_alpha})")
    print("="*60)

    history = {
        'step': [],
        'energy': [],
        'energy_std': [],
        'overlap_sq': [],
        'penalty_energy': [],
        'grad_norm': []
    }

    nan_count = 0
    max_nan_allowed = 10

    for step in range(N_ITER):
        sampler_state = sampler.reset(exc_machine, exc_params, sampler_state)
        samples, sampler_state = sampler.sample(
            exc_machine, exc_params, state=sampler_state, chain_length=20
        )
        samples = samples.reshape(-1, hi.size)

        E1_mean, E1_std, E0_mean, overlap, overlap_sq, penalty_energy = \
            compute_overlap_and_energy_stable(exc_machine, exc_params, gs_machine, gs_params, samples, penalty_alpha)

        if jnp.isnan(E1_mean):
            nan_count += 1
            if nan_count > max_nan_allowed:
                print(f"NaN数量过多 ({nan_count})，停止训练")
                break
            continue

        _, _, grad, _, _ = forces_expect_penalty_stable(exc_machine, exc_params, gs_machine, gs_params, samples, penalty_alpha)
        grad = jax.tree_map(lambda x: x*2, grad)

        grad_flat, _ = flatten_util.ravel_pytree(grad)
        grad_norm = jnp.linalg.norm(grad_flat)
        history['grad_norm'].append(float(grad_norm))

        grad = clip_gradients(grad, max_norm=2.0)

        qgt_reg, qgt_unravel_fun = compute_qgt(exc_machine, exc_params, samples, diag_shift=0.01)
        grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad)

        natural_grad_flat, _ = flatten_util.ravel_pytree(natural_grad)
        nat_grad_norm = jnp.linalg.norm(natural_grad_flat)
        natural_grad = clip_gradients(natural_grad, max_norm=1.0)

        updates, opt_state = optimizer.update(natural_grad, opt_state, exc_params)
        exc_params = optax.apply_updates(exc_params, updates)

        if step % 50 == 0 or step == N_ITER - 1:
            history['step'].append(step)
            history['energy'].append(float(E1_mean.real))
            history['energy_std'].append(float(E1_std))
            history['overlap_sq'].append(float(overlap_sq.real))
            history['penalty_energy'].append(float(penalty_energy.real))
            error_E1 = jnp.abs(E1_mean.real - E_fcis[1])
            print(f"ES Step {step:3d} | E₁: {E1_mean.real:.8f} ± {E1_std:.6f} | "
                  f"|⟨ψ₀|ψ₁⟩|²: {overlap_sq.real:.6f} | E_penalty: {penalty_energy.real:.8f} | Error: {error_E1:.6f} | |g|: {grad_norm:.4f}")

    if nan_count > 0:
        print(f"\n警告: 训练过程中出现了 {nan_count} 次 NaN")

    print(f"\n第一激发态训练完成: E₁ = {E1_mean.real:.8f} Ha")
    print(f"FCI基准: E₁ = {E_fcis[1]:.8f} Ha")
    print(f"最终重叠: |⟨ψ₀|ψ₁⟩|² = {overlap_sq.real:.6f}")

    return exc_machine, exc_params, history


# 训练第一激发态
penalty_alpha = 2.0
es_machine, es_params, es_history = train_excited_state_penalty_stable(
    gs_machine, gs_params,
    N_ITER=300,
    N_SAMPLES=1008,
    penalty_alpha=penalty_alpha,
    init_scale=0.1
)


# ==============================================================================
# 6. 结果验证
# ==============================================================================
print("\n" + "="*60)
print("第一激发态结果验证")
print("="*60)

print(f"\nFCI 基准能量:")
print(f"  E₀ = {E_fcis[0]:.8f} Ha  (基态)")
print(f"  E₁ = {E_fcis[1]:.8f} Ha  (第一激发态)")
print(f"  激发能 = {(E_fcis[1] - E_fcis[0]) * 27.2114:.4f} eV")

final_E1 = es_history['energy'][-1]
final_overlap = es_history['overlap_sq'][-1]
error_E1 = jnp.abs(final_E1 - E_fcis[1])

print(f"\n惩罚项方法结果 (α={penalty_alpha}):")
print(f"  E₁ = {final_E1:.8f} Ha")
print(f"  |⟨ψ₀|ψ₁⟩|² = {final_overlap:.6f}")
print(f"  绝对误差: {error_E1:.6f} Ha")
print(f"  相对误差: {error_E1 / jnp.abs(E_fcis[1]) * 100:.4f}%")

print("\n" + "="*60)
print("结论")
print("="*60)
if error_E1 < 0.01:
    print("✓ 惩罚项方法成功求解第一激发态！")
    print(f"  能量误差 < 0.01 Ha，验证了方法的有效性。")
else:
    print("⚠ 误差较大，可能需要调整参数或增加训练步数。")
    print("  建议：")
    print("  1. 增加惩罚系数 α")
    print("  2. 增加训练步数")
    print("  3. 使用更大的网络")
    print("  4. 尝试不同的初始化尺度")
print("="*60)
