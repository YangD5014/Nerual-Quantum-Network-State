# ===================== 测试不同 diag_shift 的效果 =====================
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
from pyscf import gto, scf, fci
from flax import nnx
from functools import partial
from jax import flatten_util
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.experimental.sampler.rules.fermion_2nd import ParticleExchangeRule

from functools import partial
from jax import flatten_util

bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

ha = nkx.operator.from_pyscf_molecule(mol)
ha_jax = ha.to_jax_operator()

hi = SpinOrbitalFermions(n_orbitals=2, s=1/2, n_fermions_per_spin=(1,1))
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = ParticleExchangeRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)

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

def create_machine(model: nnx.Module):
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    return machine, graphdef, state

@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha_jax.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

def compute_qgt_only(machine, params, sigma, diag_shift=0.1):
    n_samples = sigma.shape[0]
    def log_psi_single(p, s):
        return machine(p, s)
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    grad_flat, _ = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    return qgt_reg

rngs = nnx.Rngs(21)
model = SingleStateAnsatz(n_spin_orbitals=hi.size, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

sampler_state = sampler.init_state(machine, params, seed=1)
samples, sampler_state = sampler.sample(machine, params, state=sampler_state, chain_length=1008 // 16)
samples = samples.reshape(-1, hi.size)

print("="*70)
print("测试不同 diag_shift 值对 QGT 条件数的影响")
print("="*70)

for diag_shift in [0.0001, 0.001, 0.01, 0.1, 1.0]:
    qgt_reg = compute_qgt_only(machine, params, samples, diag_shift=diag_shift)
    eigenvalues = jnp.linalg.eigvalsh(qgt_reg)
    condition_number = eigenvalues.max() / eigenvalues.min()

    print(f"\ndiag_shift = {diag_shift}")
    print(f"  最小特征值: {eigenvalues.min():.8f}")
    print(f"  最大特征值: {eigenvalues.max():.8f}")
    print(f"  条件数: {condition_number:.2f}")

print("\n" + "="*70)
print("注意：如果不加正则化（diag_shift=0），QGT 可能接近奇异")
print("="*70)

qgt_no_reg = compute_qgt_only(machine, params, samples, diag_shift=0.0)
eigenvalues_no_reg = jnp.linalg.eigvalsh(qgt_no_reg)
condition_number_no_reg = eigenvalues_no_reg.max() / eigenvalues_no_reg.min()
print(f"\ndiag_shift = 0.0 (无正则化)")
print(f"  最小特征值: {eigenvalues_no_reg.min():.10f}")
print(f"  最大特征值: {eigenvalues_no_reg.max():.8f}")
print(f"  条件数: {condition_number_no_reg:.2f}")
