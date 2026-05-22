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
# 4. 初始化模型、采样器、优化器
# ==============================================================================
model = SingleStateAnsatz(4,12, rngs=nnx.Rngs(21))
# 采样器
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)

# ===================== 4. 包装模型为 machine 函数 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state

# ===================== 5. 纯 JAX 实现的 force-based 梯度计算 =====================
# 注意：不能对包含 ha.get_conn_padded 的函数使用 @jax.jit
# 因为 NetKet 的 Numba 算子不兼容 JAX 追踪
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    
    这对应 NetKet 的 local_value_kernel
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


# 注意：不能对包含 ha.get_conn_padded 的函数使用 @jax.jit
# 因为 NetKet 的 Numba 算子不兼容 JAX 追踪
def forces_expect_hermitian(machine, params, sigma):
    """
    核心：复刻 NetKet 的 forces_expect_hermitian 函数
    
    使用 force-based 梯度计算：
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    
    关键：对于复数值网络，使用 holomorphic=True
    """
    # 1. 计算局部能量
    O_loc = compute_local_energies(machine, params, sigma)
    
    # 2. 统计能量均值
    O_mean, O_std = statistics(O_loc)
    
    # 3. 中心化局部能量
    O_centered = O_loc - O_mean
    
    # 4. 计算 ∇log ψ 对每个样本
    # 使用 jax.grad 计算复数梯度（holomorphic=True）
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    # 5. 计算 force-based 梯度
    # grad = ⟨(E_loc - E_mean) ∇log ψ⟩ = (1/N) Σ (E_loc[i] - E_mean) ∇log ψ(σ[i])
    # grad_matrix 已经是 PyTree 结构，每个元素的形状是 (n_samples, ...)
    # 关键修复：O_centered 形状为 (n_samples,)，需要正确广播到梯度张量的每个维度
    # 使用 reshape 将 O_centered 变为 (n_samples, 1, 1, ..., 1) 以匹配梯度张量
    def weight_and_mean(grad_component):
        # grad_component 形状：(n_samples, d1, d2, ...)
        # O_centered 形状：(n_samples,)
        # 需要广播相乘后沿 axis=0 求平均
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad


#@partial(jax.jit, static_argnames=("machine",))
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


def make_get_all_next_states(edges):
    # 把 edges 变成 静态 Python 元组！核心！
    edges = tuple(tuple(e) for e in edges)

    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        s_arr = S
        next_states = []
        masks = []

        # 现在 edges 是 Python 静态元组 → 完全安全！
        for (i, j) in edges:
            occ_i = s_arr[i]
            occ_j = s_arr[j]
            valid = (occ_i == 1) & (occ_j == 0) | (occ_i == 0) & (occ_j == 1)
            new_state = s_arr.at[i].set(occ_j).at[j].set(occ_i)
            next_states.append(new_state)
            masks.append(valid)

        return jnp.stack(next_states), jnp.stack(masks)
    
    return get_all_next_states_jit

def make_metropolis_hastings_step(edges, log_pdf):
    get_all_next = make_get_all_next_states(edges)
    
    @jax.jit
    def metropolis_hastings_step_jit(S: jnp.ndarray, key: jax.Array):
        # 1. 候选
        candidates, valid_mask = get_all_next(S)
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, candidates.shape[0])
        S_cand = candidates[idx]
        is_valid = valid_mask[idx]

        # ==============================
        # ✅ 严格正确：log_pdf = ln(ψ)
        # ==============================
        log_accept_ratio = 2 * (
            jnp.real(log_pdf(S_cand)) - jnp.real(log_pdf(S))
        )

        # 接受条件（无exp，超快）
        key, subk = jax.random.split(key)
        u = jax.random.uniform(subk)
        accept = is_valid & (log_accept_ratio > jnp.log(u))

        S_new = jnp.where(accept, S_cand, S)
        return S_new, accept, key

    return metropolis_hastings_step_jit


# 🔥 正确静态参数
@partial(jax.jit, static_argnums=(0, 1, 3, 4,5))
def mcmc_sampler(
    n_samples: int,
    n_warmup: int,
    initial_state: jnp.ndarray,
    edges: tuple[tuple[int, int]],
    log_pdf: callable,
    seed: int = 42
):
    key = jax.random.PRNGKey(seed)
    mh_step = make_metropolis_hastings_step(edges, log_pdf)

    # 预烧
    def warmup_loop(carry, _):
        state, rng = carry
        state, _, rng = mh_step(state, rng)
        return (state, rng), None

    (current_state, key), _ = jax.lax.scan(
        warmup_loop,
        (initial_state, key),
        xs=None,
        length=n_warmup
    )

    # 采样
    def sample_loop(carry, _):
        state, rng = carry
        state, accepted, rng = mh_step(state, rng)
        return (state, rng), state

    (_, _), samples = jax.lax.scan(
        sample_loop,
        (current_state, key),
        xs=None,
        length=n_samples
    )

    return samples

# ===================== 6. 初始化 =====================
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

optimizer = optax.sgd(learning_rate=0.01)  # 学习率 0.01
opt_state = optimizer.init(params)

# 训练参数
N_ITER = 300  # 迭代次数
N_SAMPLES = 1008  # 样本数

# ===================== 7. 训练循环 =====================
print("\n" + "="*60)
print("开始纯 JAX VMC 训练 (自然梯度下降法)")
print("="*60)

# 用于记录训练历史
history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}

# 初始化MCMC链状态（用于保持链的连续性）
current_state = hi.all_states()[0]
is_first_iteration = True

for step in range(N_ITER):
    # 1. 采样 - 第一次充分预烧，之后只热化少量步以保持链连续性
    if is_first_iteration:
        n_warmup = 200
        is_first_iteration = False
    else:
        n_warmup = 10

    # 创建curried版本的log_pdf，只接受state参数
    def log_pdf_curried(sigma):
        return machine(params, sigma)

    samples = mcmc_sampler(
        n_samples=N_SAMPLES,
        n_warmup=n_warmup,
        initial_state=current_state,
        edges=((0, 1), (2, 3)),
        log_pdf=log_pdf_curried,
        seed=step
    )

    current_state = samples[-1]  # 保存最后状态用于下次迭代

    # 2. 计算 force-based 能量和梯度
    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
    grad = jax.tree_map(lambda x: x*2, grad)
    qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001) 
    grad_flat , grad_unravel_fn = flatten_util.ravel_pytree(grad)
  
    #自然梯度 natural-gradient = S^{-1} * grad
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad)
    grad = natural_grad
        
    # 4. 更新参数（自然梯度下降）
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 5. 记录历史
    if step % 50 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        print(f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")

# 最终结果
final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
final_error = jnp.abs(final_energy.real - E_fcis[0])
print("\n" + "="*60)
print(f"训练完成!")
print(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
print(f"FCI 基准：{E_fcis[0]:.8f} Ha")
print(f"绝对误差：{final_error:.6f} Ha")
print(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
print("="*60)

