import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import optax
from functools import partial
from jax import flatten_util


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma, hamiltonian):
    """计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)"""
    eta, H_eta = hamiltonian.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma_expanded = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma_expanded), axis=-1)


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma, hamiltonian):
    """计算能量期望和梯度（复数值网络，holomorphic=True）"""
    O_loc = compute_local_energies(machine, params, sigma, hamiltonian)
    O_mean, O_std = statistics(O_loc)
    O_loc_centered = O_loc - O_mean
    
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    def weight_and_mean(grad_component):
        weights = O_loc_centered.reshape((O_loc_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_with_penalty(machine, params_es, params_gs, sigma_es, sigma_gs, 
                               lambda_penalty, hamiltonian):
    """
    计算带惩罚项的激发态能量和梯度
    """
    O_loc_es = compute_local_energies(machine, params_es, sigma_es, hamiltonian)
    E_mean, E_std = statistics(O_loc_es)
    
    O_loc_centered = O_loc_es - E_mean
    
    # psi_loc_1 = ⟨Ψ₀|σ⟩ / ⟨Ψ₁|σ⟩，在 sigma_es 上采样
    psi_loc_1 = jnp.exp(machine(params_gs, sigma_es) - machine(params_es, sigma_es))
    psi_1_mean = jnp.mean(psi_loc_1)
    
    # psi_loc_2 = ⟨Ψ₁|σ⟩ / ⟨Ψ₀|σ⟩，在 sigma_gs 上采样
    psi_loc_2 = jnp.exp(machine(params_es, sigma_gs) - machine(params_gs, sigma_gs))
    psi_2_mean = jnp.mean(psi_loc_2)
    
    penalty_term = lambda_penalty * psi_1_mean * psi_2_mean
    
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample_es(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params_es)
    
    grad_matrix_es = jax.vmap(compute_grad_for_sample_es)(sigma_es)
    
    def weight_energy_grad(grad_component):
        weights = O_loc_centered.reshape((O_loc_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    energy_grad = jax.tree_util.tree_map(weight_energy_grad, grad_matrix_es)
    
    psi_loc_2_centered = psi_loc_2 - psi_2_mean
    psi_loc_2_weighted = psi_loc_2_centered * lambda_penalty * psi_1_mean
    
    O_loc_with_penalty = O_loc_es + psi_loc_2_weighted
    O_loc_penalty_centered = O_loc_with_penalty - jnp.mean(O_loc_with_penalty)
    
    def weight_total_grad(grad_component):
        weights = O_loc_penalty_centered.reshape((O_loc_penalty_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    penalty_grad = jax.tree_util.tree_map(weight_total_grad, grad_matrix_es)
    
    total_grad = jax.tree_map(lambda e, p: 2 * e + p, energy_grad, penalty_grad)
    
    total_loss = E_mean + penalty_term
    
    return total_loss, E_mean, E_std, total_grad, jnp.abs(psi_1_mean)**2


def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """计算量子几何张量（QGT）/ 自然梯度预 conditioner"""
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
# 主程序：H₂ 分子的基态和第一激发态计算
# ==============================================================================
if __name__ == "__main__":
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
    hi = nkx.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1,1),
    )

    class SingleStateAnsatz(nn.Module):
        def __init__(self, n_spin_orbitals: int, hidden_dim=16):
            super().__init__()
            self.linear1 = nn.Dense(hidden_dim, complex_dtype=jnp.complex64)
            self.linear2 = nn.Dense(hidden_dim, complex_dtype=jnp.complex64)
            self.output = nn.Dense(1, complex_dtype=jnp.complex64)

        def __call__(self, x):
            h = jnp.tanh(self.linear1(x.astype(jnp.complex64)))
            h = jnp.tanh(self.linear2(h))
            out = self.output(h)
            return jnp.squeeze(out)

    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nk.sampler.rules.ExchangeRule(graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, single_rule, n_chains=100, sweep_size=32)

    print("\n" + "="*60)
    print("第一阶段：基态 VMC 训练")
    print("="*60)

    model_gs = SingleStateAnsatz(4, hidden_dim=12)
    params_gs = model_gs.init(jax.PRNGKey(21), jnp.ones((4,)))
    
    def machine_gs(params, sigma):
        return model_gs.apply(params, sigma)
    
    sampler_state_gs = sampler.init_state(machine_gs, params_gs, seed=1)

    optimizer_gs = optax.sgd(learning_rate=0.01)
    opt_state_gs = optimizer_gs.init(params_gs)

    N_ITER_GS = 300

    for step in range(N_ITER_GS):
        sampler_state_gs = sampler.reset(machine_gs, params_gs, sampler_state_gs)
        samples_gs, sampler_state_gs = sampler.sample(
            machine_gs, params_gs, state=sampler_state_gs, chain_length=20
        )
        samples_gs = samples_gs.reshape(-1, hi.size)
        
        energy_gs, energy_std_gs, grad_gs = forces_expect_hermitian(machine_gs, params_gs, samples_gs, ha)
        grad_gs = jax.tree_map(lambda x: x*2, grad_gs)
        qgt_reg_gs, _ = compute_qgt(machine_gs, params_gs, samples_gs, diag_shift=0.001)
        grad_flat_gs, grad_unravel_gs = flatten_util.ravel_pytree(grad_gs)
        natural_grad_gs = jnp.linalg.solve(qgt_reg_gs, grad_flat_gs)
        natural_grad_gs = grad_unravel_gs(natural_grad_gs)
        
        updates_gs, opt_state_gs = optimizer_gs.update(natural_grad_gs, opt_state_gs, params_gs)
        params_gs = optax.apply_updates(params_gs, updates_gs)
        
        if step % 50 == 0 or step == N_ITER_GS - 1:
            error = jnp.abs(energy_gs.real - E_fcis[0])
            print(f"GS Step {step:3d} | E: {energy_gs.real:.8f} ± {energy_std_gs:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")

    final_energy_gs, final_std_gs, _ = forces_expect_hermitian(machine_gs, params_gs, samples_gs, ha)
    print(f"\n基态训练完成: {final_energy_gs.real:.8f} ± {final_std_gs:.6f} Ha (FCI: {E_fcis[0]:.8f})")

    print("\n" + "="*60)
    print("第二阶段：第一激发态 VMC 训练（能量 + 正交惩罚）")
    print("="*60)

    model_es = SingleStateAnsatz(4, hidden_dim=12)
    params_es = model_es.init(jax.PRNGKey(42), jnp.ones((4,)))
    
    def machine_es(params, sigma):
        return model_es.apply(params, sigma)

    sampler_state_es = sampler.init_state(machine_es, params_es, seed=2)

    optimizer_es = optax.sgd(learning_rate=0.005)
    opt_state_es = optimizer_es.init(params_es)

    N_ITER_ES = 400
    LAMBDA_PENALTY = 10.0

    for step in range(N_ITER_ES):
        sampler_state_es = sampler.reset(machine_es, params_es, sampler_state_es)
        samples_es, sampler_state_es = sampler.sample(
            machine_es, params_es, state=sampler_state_es, chain_length=20
        )
        samples_es = samples_es.reshape(-1, hi.size)
        
        sampler_state_gs = sampler.reset(machine_gs, params_gs, sampler_state_gs)
        samples_gs_loop, sampler_state_gs = sampler.sample(
            machine_gs, params_gs, state=sampler_state_gs, chain_length=20
        )
        samples_gs_loop = samples_gs_loop.reshape(-1, hi.size)
        
        total_loss, energy_es, energy_std_es, grad_es, overlap_sq = forces_expect_with_penalty(
            machine_es, params_es, params_gs,
            samples_es, samples_gs_loop,
            LAMBDA_PENALTY, ha
        )
        
        qgt_reg_es, _ = compute_qgt(machine_es, params_es, samples_es, diag_shift=0.001)
        grad_flat_es, grad_unravel_es = flatten_util.ravel_pytree(grad_es)
        natural_grad_es = jnp.linalg.solve(qgt_reg_es, grad_flat_es)
        natural_grad_es = grad_unravel_es(natural_grad_es)
        
        updates_es, opt_state_es = optimizer_es.update(natural_grad_es, opt_state_es, params_es)
        params_es = optax.apply_updates(params_es, updates_es)
        
        if step % 50 == 0 or step == N_ITER_ES - 1:
            error = jnp.abs(energy_es.real - E_fcis[1])
            print(f"ES Step {step:3d} | E: {energy_es.real:.8f} ± {energy_std_es:.6f} | "
                  f"FCI: {E_fcis[1]:.8f} | Err: {error:.6f} | Overlap²: {overlap_sq:.6f}")

    final_energy_es, final_std_es, _ = forces_expect_hermitian(machine_es, params_es, samples_es, ha)
    print(f"\n激发态训练完成: {final_energy_es.real:.8f} ± {final_std_es:.6f} Ha (FCI: {E_fcis[1]:.8f})")
    
    excitation_energy = (final_energy_es.real - final_energy_gs.real) * 27.2114
    fci_excitation = (E_fcis[1] - E_fcis[0]) * 27.2114
    print(f"\n激发能: {excitation_energy:.4f} eV (FCI: {fci_excitation:.4f} eV)")
    print(f"激发能误差: {abs(excitation_energy - fci_excitation):.4f} eV")
