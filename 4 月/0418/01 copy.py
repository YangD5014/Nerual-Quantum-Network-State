# ==============================================================================
# 【不可更改片段开始】
# ==============================================================================
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
from jax.scipy.sparse.linalg import cg

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
hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
# ==============================================================================
# 【不可更改片段结束】
# ==============================================================================

# ==============================================================================
# 2. 神经网络 Ansatz
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=jnp.complex128)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=jnp.complex128)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=jnp.complex128)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(jnp.complex128)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

# ==============================================================================
# 3. 核心工具函数（修复维度广播错误）
# ==============================================================================
@partial(jax.jit, static_argnums=0)
def logpsi(graphdef: nnx.GraphDef, state: nnx.State, x: jax.Array) -> jax.Array:
    model = nnx.merge(graphdef, state)
    return model(x)

@partial(jax.jit, static_argnums=0)
def compute_local_energies(graphdef: nnx.GraphDef, state: nnx.State, sigma: jax.Array) -> jax.Array:
    eta, H_sigmaeta = ha.get_conn_padded(sigma)
    logpsi_sigma = logpsi(graphdef, state, sigma)
    logpsi_eta = logpsi(graphdef, state, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

# VMC 能量+梯度（修复维度广播）
@partial(jax.jit, static_argnums=0)
def vmc_energy_and_grad(graphdef: nnx.GraphDef, state: nnx.State, samples: jax.Array):
    E_loc = compute_local_energies(graphdef, state, samples)
    E_mean = jnp.mean(E_loc).real
    E_loc_centered = E_loc - E_mean  # 保持形状 (n_samples,)

    # 1. 定义单样本 logpsi
    def logpsi_single(s, single_x):
        return logpsi(graphdef, s, single_x)

    # 2. 复函数求导
    grad_logpsi_single = jax.grad(logpsi_single, argnums=0, holomorphic=True)
    grad_logpsi_vmap = jax.vmap(grad_logpsi_single, in_axes=(None, 0))(state, samples)

    # 3. 关键修复：利用 JAX 自动广播，无需手动扩展维度
    # E_loc_centered (n_samples,) 会自动广播到 grad_logpsi_vmap (n_samples, ...)
    grad = jax.tree_map(
        lambda g: 2 * jnp.real(jnp.mean(E_loc_centered.conj()[:, jnp.newaxis, jnp.newaxis] * g, axis=0)),
        grad_logpsi_vmap
    )
    return E_mean, grad

# SR 预条件（同步修复维度广播）
@partial(jax.jit, static_argnums=0)
def apply_sr(
    graphdef: nnx.GraphDef, state: nnx.State, samples: jax.Array, grad: nnx.State,
    diag_shift: float = 0.1, maxiter: int = 50
):
    def logpsi_single(s, x):
        return logpsi(graphdef, s, x)
    grad_logpsi_single = jax.grad(logpsi_single, argnums=0, holomorphic=True)
    grad_logpsi_vmap = jax.vmap(grad_logpsi_single, in_axes=(None, 0))(state, samples)

    def qgt_matvec(v):
        # 同样利用自动广播
        dot = jax.tree_map(lambda g, vv: jnp.sum(g.conj() * vv, axis=tuple(range(1, g.ndim))), grad_logpsi_vmap, v)
        dot = jnp.stack(jax.tree_util.tree_leaves(dot), axis=-1).sum(axis=-1)
        res = jax.tree_map(lambda g: jnp.mean(g * dot[(slice(None),) + (jnp.newaxis,) * (g.ndim - 1)], axis=0), grad_logpsi_vmap)
        res = jax.tree_map(lambda r, vv: r + diag_shift * vv, res, v)
        return res

    grad_sr, _ = cg(qgt_matvec, grad, maxiter=maxiter)
    return grad_sr

# ==============================================================================
# 4. 初始化 + 训练循环
# ==============================================================================
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, 12, rngs=rngs)
graphdef, model_state = nnx.split(model)

# 采样器前向函数
def forward(state, x):
    return logpsi(graphdef, state, x)

# 采样器初始化
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)
sampler_state = sampler.init_state(forward, model_state, seed=1)

# 优化器
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(model_state)

# 主训练循环
n_iters = 300
print("="*60)
print("NNX 手写 VMC 训练（修复维度广播错误）")
print("="*60)
for i in tqdm(range(n_iters)):
    # 采样
    sampler_state = sampler.reset(forward, model_state, state=sampler_state)
    samples, sampler_state = sampler.sample(forward, model_state, state=sampler_state, chain_length=200)
    samples = samples.reshape(-1, hi.size)

    # 计算能量+梯度
    E, E_grad = vmc_energy_and_grad(graphdef, model_state, samples)

    # SR 优化
    E_grad_sr = apply_sr(graphdef, model_state, samples, E_grad)

    # 参数更新
    updates, opt_state = optimizer.update(E_grad_sr, opt_state, model_state)
    model_state = optax.apply_updates(model_state, updates)

    # 日志
    if i % 30 == 0:
        err = abs(E - E_fcis[0])
        print(f"Iter {i:3d} | E = {E:.8f} Ha | 误差 = {err:.8f} Ha")

# 最终结果
final_E = jnp.mean(compute_local_energies(graphdef, model_state, hi.all_states())).real
print("="*60)
print(f"最终能量: {final_E:.8f} Ha")
print(f"基准能量: {E_fcis[0]:.8f} Ha")
print(f"绝对误差: {abs(final_E - E_fcis[0]):.8f} Ha")
print("="*60)