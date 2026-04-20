# ===================== 【不可修改部分 - 严格保留】 =====================
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
from jax import flatten_util

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

edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)
# ===================== 【不可修改部分 结束】 =====================

# ==============================================================================
# 核心实现：适配 Netket 接口 + 高精度能量/梯度/SR计算
# ==============================================================================

# ---------------------- 1. 包装 NNX 模型为 Netket 采样器兼容的 machine 函数 ----------------------
# Netket 强制要求 machine 签名：machine(params, σ) → jax.Array (logψ)
def create_machine(model: nnx.Module):
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        # 合并参数并前向传播
        m = nnx.merge(graphdef, params)
        return m(sigma)
    return machine, graphdef, state

# ---------------------- 2. 核心物理计算函数（复刻原生 Netket 逻辑） ----------------------
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """计算局部能量 Eloc，与原生 Netket 完全一致"""
    eta, H_sigmaeta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

@partial(jax.jit, static_argnames=("machine",))
def energy_and_grad(machine, params, samples):
    """计算能量和解析梯度（复数值，holomorphic=True 强制开启）"""
    def loss_fn(p):
        Eloc = compute_local_energies(machine, p, samples)
        E = jnp.mean(Eloc)
        return E.real, E  # loss=实部能量, aux=全能量
    # 复数值神经网络必须用 holomorphic=True
    (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True, holomorphic=True)(params)
    return energy, grads

# ---------------------- 3. 高精度 SR 自然梯度（匹配原生 Netket diag_shift=0.1） ----------------------
@partial(jax.jit, static_argnums=0)
def compute_qgt(machine, params, samples):
    """计算量子几何张量 QGT"""
    n_samples = samples.shape[0]
    # 计算 logψ 对参数的梯度
    grad_logpsi = jax.vmap(jax.grad(lambda p, x: machine(p, x), holomorphic=True), in_axes=(None, 0))(params, samples)
    # 展平梯度为向量
    g_flat, unflatten = flatten_util.ravel_pytree(grad_logpsi)
    g_flat = g_flat.reshape(n_samples, -1)
    # 中心化梯度
    g_mean = jnp.mean(g_flat, axis=0, keepdims=True)
    g_centered = g_flat - g_mean
    # 计算 QGT
    qgt = (1.0 / n_samples) * jnp.conj(g_centered).T @ g_centered
    return qgt.real, unflatten

@partial(jax.jit, static_argnums=0)
def apply_sr(machine, params, samples, grad, diag_shift=0.1):
    """SR 预处理：自然梯度更新（原生 Netket 核心参数）"""
    qgt, unflatten = compute_qgt(machine, params, samples)
    # 正则化（匹配原生 Netket diag_shift=0.1）
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    # 展平梯度并求解
    g_flat, _ = flatten_util.ravel_pytree(grad)
    nat_grad_flat = jnp.linalg.solve(qgt_reg, g_flat)
    # 恢复梯度树形结构
    nat_grad = unflatten(nat_grad_flat)
    return nat_grad

# ==============================================================================
# 训练初始化（超参数严格匹配原生 Netket 高精度版本）
# ==============================================================================
# 1. 初始化 NNX 模型（hidden_dim=12，与原生一致）
rngs = nnx.Rngs(seed=21)
model = SingleStateAnsatz(n_spin_orbitals=hi.size, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

# 2. 初始化采样器状态（严格适配 Netket 接口）
sampler_state = sampler.init_state(machine, params, seed=1)

# 3. 优化器（原生用 SGD+lr=0.1，效果最优；也可用 Adam）
optimizer = optax.sgd(learning_rate=0.1)
opt_state = optimizer.init(params)

# 4. 训练超参数
N_ITER = 300
N_SAMPLES = 1008  # 原生 Netket 样本数，保证精度
USE_SR = True     # 开启 SR 是精度核心

# ==============================================================================
# 训练循环（高精度收敛）
# ==============================================================================
print("\n" + "="*60)
print("开始 Flax NNX 手动训练（SR 开启）")
print("="*60)

for step in tqdm(range(N_ITER)):
    # 1. 采样：生成样本 + 重塑为 (n_samples, hilbert_size)
    samples, sampler_state = sampler.sample(
        machine, params, state=sampler_state, chain_length=N_SAMPLES // sampler.n_chains
    )
    samples = samples.reshape(-1, hi.size)  # 适配能量计算

    # 2. 计算能量和梯度
    energy, grad = energy_and_grad(machine, params, samples)

    # 3. SR 自然梯度更新
    if USE_SR:
        grad = apply_sr(machine, params, samples, grad)

    # 4. 优化器更新参数
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)

    # 5. 日志输出
    if step % 20 == 0:
        error = jnp.abs(energy.real - E_fcis[0])
        print(f"\nStep {step:3d} | Energy: {energy.real:.8f} Ha | FCI: {E_fcis[0]:.8f} Ha | Error: {error:.6f} Ha")

# ==============================================================================
# 最终结果
# ==============================================================================
final_energy, _ = energy_and_grad(machine, params, samples)
final_error = jnp.abs(final_energy.real - E_fcis[0])
print("\n" + "="*60)
print(f"训练完成！")
print(f"最终基态能量: {final_energy.real:.8f} Ha")
print(f"FCI 基准能量: {E_fcis[0]:.8f} Ha")
print(f"绝对误差: {final_error:.6f} Ha")
print("="*60)