#我要尝试一下 自己使用 FFN 来求解 H2分子的系统基态
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from tqdm import tqdm
from functools import partial

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
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# NetKet 哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)

# 原始 Hilbert 空间（2个轨道，每个自旋1个电子）
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

def forward(params, x):
    log_psi,state = nnx.call(params)(x)
    return log_psi

def compute_local_energies_nnx(model: nnx.Module, hamiltonian:nk.operator.DiscreteOperator, sigma:jnp.array):
    eta, H_sigmaeta = hamiltonian.get_conn_padded(sigma)
    # NNX: 直接调用模型，参数在 model 内部
    logpsi_sigma = model(sigma)
    logpsi_eta   = model(eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
    return res


@partial(jax.jit, static_argnames='model')
def estimate_energy(model, hamiltonian, sigma):
    E_loc = compute_local_energies_nnx(model, hamiltonian, sigma)
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)

    return nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)

@partial(jax.jit, static_argnames=("model_forward",))
def energy_and_grad(model_state, model_forward, hamiltonian, samples):
    def loss_fn(state):
        log_psi = model_forward(state, samples)
        eta, H_sigmaeta = hamiltonian.get_conn_padded(samples)
        logpsi_eta = model_forward(state, eta)
        
        log_psi = jnp.expand_dims(log_psi, -1)
        Eloc = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - log_psi), axis=-1)
        energy = jnp.mean(Eloc)
        return energy.real, energy

    (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
    return energy, grads

rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4,12,rngs=rngs)
compute_local_energies_nnx(model,ha,hi.all_states()[1])  #Array(-0.61917148+0.01953178j, dtype=complex128)
compute_local_energies_nnx(model,ha,hi.all_states()) #Array([-0.19832496+0.04889976j, -0.61917148+0.01953178j, 0.45999249-0.65375447j, -0.63328659-0.10401855j],      dtype=complex128)
estimate_energy(model,ha,hi.all_states()) #-0.25-0.17j ± 0.26 [σ²=0.28]

rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, 12, rngs=rngs)
model_state = nnx.split(model)  # 官方标准
# 你的采样器（不变）
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)
sampler_state = sampler.init_state(forward, model_state, seed=1)
samples, sampler_state = sampler.sample(
        forward, model_state, state=sampler_state, chain_length=200
    )

optimizer = optax.adam(learning_rate=1e-2)
opt_state = optimizer.init(model_state)
n_iters = 300
for i in tqdm(range(n_iters)):
    # sample
    sampler_state = sampler.reset(forward, model_state, state=sampler_state)
    samples, sampler_state = sampler.sample(forward, model_state, state=sampler_state, chain_length=200)

    # compute energy and gradient
    E, E_grad = energy_and_grad(model_state, forward, ha, samples)

    # update parameters
    updates, opt_state = optimizer.update(E_grad, opt_state, model_state)
    model_state = optax.apply_updates(model_state, updates)
    if i % 10 == 0:
        print(E)
    
    # # log energy
    # logger(step=i, item={'Energy':E})