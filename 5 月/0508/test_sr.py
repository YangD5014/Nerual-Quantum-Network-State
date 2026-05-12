# ===================== 环境配置 =====================
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial
from jax import flatten_util
from netket.experimental.hilbert import SpinOrbitalFermions
from netket.experimental.sampler.rules.fermion_2nd import ParticleExchangeRule

print(f"JAX version: {jax.__version__}")
print(f"NetKet version: {nk.__version__}")

# ===================== 1. H₂ 分子定义 & FCI 基准 =====================
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
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

# ===================== 2. NetKet 哈密顿量和采样器 =====================
ha = nkx.operator.from_pyscf_molecule(mol)
ha_jax = ha.to_jax_operator()

hi = SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = ParticleExchangeRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)

print(f"Hilbert space size: {hi.size}")

# ===================== 3. 神经网络 Ansatz =====================
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

# ===================== 4. 包装模型为 machine 函数 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state

# ===================== 5. 核心函数 =====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha_jax.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

def statistics(x):
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])

@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
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

def compute_qgt_only(machine, params, sigma, diag_shift=0.1):
    """只计算 QGT，不应用 SR"""
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

def apply_sr_preconditioner(qgt_reg, grad):
    """应用 SR 预处理：求解 S⁻¹ ∇E"""
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad)
    nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    nat_grad = unravel_fn(nat_grad_flat)
    return nat_grad

# ===================== 6. 初始化 =====================
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(n_spin_orbitals=hi.size, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

sampler_state = sampler.init_state(machine, params, seed=1)
optimizer = optax.sgd(learning_rate=0.1)
opt_state = optimizer.init(params)

N_ITER = 300
N_SAMPLES = 1008
USE_SR = True
DIAG_SHIFT = 0.1

print("初始化完成!")
n_params, _ = flatten_util.ravel_pytree(params)
print(f"模型参数数量：{n_params.shape[0]}")
print(f"学习率：0.1")
print(f"使用 SR: {USE_SR}")
print(f"Diag shift: {DIAG_SHIFT}")

# ===================== 7. 训练循环 =====================
print("\n" + "="*60)
print("开始 VMC 训练 (SR 自然梯度优化)")
print("="*60)

history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}

for step in range(N_ITER):
    samples, sampler_state = sampler.sample(
        machine, params, state=sampler_state,
        chain_length=N_SAMPLES // sampler.n_chains
    )
    samples = samples.reshape(-1, hi.size)

    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)

    if USE_SR:
        qgt_reg = compute_qgt_only(machine, params, samples, diag_shift=DIAG_SHIFT)
        grad = apply_sr_preconditioner(qgt_reg, grad)

    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    if step % 10 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        print(f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")

final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
final_error = jnp.abs(final_energy.real - E_fcis[0])
print("\n" + "="*60)
print(f"训练完成!")
print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
print(f"绝对误差：{final_error:.6f} Ha")
print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)

# ===================== 8. QGT 分析 =====================
print("\n" + "="*60)
print("QGT 性质分析")
print("="*60)

qgt = compute_qgt_only(machine, params, samples, diag_shift=DIAG_SHIFT)
eigenvalues = jnp.linalg.eigvalsh(qgt)

print(f"\nQGT 特征值统计:")
print(f"最小特征值：{eigenvalues.min():.6f}")
print(f"最大特征值：{eigenvalues.max():.6f}")
print(f"条件数：{eigenvalues.max() / eigenvalues.min():.2f}")

if eigenvalues.max() / eigenvalues.min() > 1000:
    print("\n⚠️  条件数较大，QGT 接近奇异")
else:
    print("\n✅ QGT 性态良好")

print("\n代码运行成功!")
