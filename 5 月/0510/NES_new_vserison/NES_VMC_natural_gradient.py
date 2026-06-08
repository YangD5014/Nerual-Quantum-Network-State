"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现 - 自然梯度版本

基于 JAX/Flax NNX 实现的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
本版本实现了完整的自然梯度下降（Natural Gradient Descent）优化。

算法核心：
- 使用扩展希尔伯特空间将多激发态问题转化为单基态问题
- 通过行列式 Ansatz 自动保证状态间的正交性
- 使用量子几何张量（QGT）进行自然梯度下降加速收敛
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
from jax.flatten_util import ravel_pytree
import time

# ==============================================================================
# 1. H₂ 分子定义与 FCI 基准
# ==============================================================================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("=" * 60)
print("H₂ 分子 FCI 基准能量")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

# ==============================================================================
# 2. 希尔伯特空间与哈密顿量设置
# ==============================================================================
ha = nkx.operator.from_pyscf_molecule(mol)
# 转换为 JAX 兼容格式
ha = ha.to_jax_operator()

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)

K = 2  # 扩展副本数
hi_ext = hi ** K

# 跃迁边定义 (alpha1, alpha2, beta1, beta2)
edges = [(0, 1), (2, 3), (4, 5), (6, 7)]
SINGLE_HILBERT_SIZE = 4

# ==============================================================================
# 3. 神经网络 Ansatz 定义
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    """
    单态 Ansatz：复数值前馈神经网络

    输入: x ∈ {0,1}^n_spin (单粒子组态)
    输出: log ψ(x) (对数波函数幅值)
    """

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        # 复数值线性层
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

    通过行列式结构自动保证不同 Ansatz 副本的正交性，
    避免训练过程中状态坍缩到同一激发态。
    """

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        # 创建 K 个独立的单态 Ansatz
        self.single_ansatz_list = []
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        """
        前向传播

        参数:
            x: 扩展态组态，形状可以是：
               - (K * n_spin,)  单个扩展态
               - (batch, K * n_spin)  批量扩展态
               - (batch, K, n_spin)  批量扩展态

        返回:
            log_Psi: log det(Ψ)
            L: log ψᵢ(xⱼ) 矩阵
        """
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)

            # 构建 L_ij = log ψ_j(x^i)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))

            # Ψ = exp(L), 计算行列式
            Psi_matrix = jnp.exp(L)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_matrix)
            log_Psi = log_abs_det + 1j * jnp.angle(sign)

            return log_Psi, L

        # 处理不同输入形状
        if x.ndim == 1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            return jax.vmap(_forward_single)(x)
        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')


# ==============================================================================
# 4. 机器学习工具函数
# ==============================================================================
def create_machine(model):
    """将 NESTotalAnsatz 模型包装为返回 log|Ψ| 的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, _ = m(sigma)  # 只返回 log|Ψ|，不要矩阵
        return log_psi_total

    return machine, graphdef, state


def create_single_machine(model):
    """将 SingleStateAnsatz 模型包装为返回 log|ψ| 的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


def create_machine_matrix(model: NESTotalAnsatz):
    """创建返回 log ψ 矩阵的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        _, log_M = m(sigma)
        return log_M

    return machine, graphdef, state


# ==============================================================================
# 5. 哈密顿量作用函数
# ==============================================================================
def Ham_psi(ha, single_machine, params, x):
    """
    计算 H ψ(x)

    支持单个样本和批量样本自动处理。
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
    return H_psi_batch


def Ham_Psi(ha, single_machine_list, total_params, x):
    """
    计算 H Ψ(x) = [H ψ_i(x^j)]

    返回 K×K 能量矩阵
    """
    K = len(single_machine_list)

    if x.ndim == 2:
        # 单个扩展态
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, x_single[i])
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 批量扩展态
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, x_single[i])
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        return jax.vmap(_single_HamPsi)(x)

    raise ValueError(f"不支持的输入形状: {x.shape}")


# ==============================================================================
# 6. NES-VMC 损失函数与梯度
# ==============================================================================
def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    """
    计算 NES-VMC 损失函数

    L = Tr(Ψ⁻¹ H Ψ) = Tr(E_L)

    其中 E_L 是局域能量矩阵
    """
    log_M = total_matrix_machine(total_params, x)
    Psi_Matrix = jnp.exp(log_M)
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv


def nes_vmc_gradient(ha, total_matrix_machine, total_machine, single_machine_list, total_params, x_batch):
    """
    计算 NES-VMC 梯度

    ∇L = 2⟨Tr(E_L - ⟨E_L⟩) ∇log Ψ⟩
    """
    loss_batch, E_L_batch = NES_loss_energy(
        ha, total_matrix_machine, single_machine_list, total_params, x_batch
    )
    E_L_mean = jnp.mean(E_L_batch, axis=0)

    tr_batch = loss_batch
    tr_mean = tr_batch.mean()
    tr_centered = tr_batch - tr_mean

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
    loss_mean = loss_batch.mean()

    return grad, loss_mean, E_L_mean


# ==============================================================================
# 7. 量子几何张量 (QGT) 与自然梯度
# ==============================================================================
def compute_nes_qgt(machine, params, samples, diag_shift=0.01):
    """
    计算量子几何张量 (Quantum Geometric Tensor)

    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩

    这是自然梯度下降的核心：
    natural_grad = S⁻¹ × gradient
    """
    def _single_grad(x):
        return jax.grad(lambda p: machine(p, x), holomorphic=True)(params)

    grads = [_single_grad(x) for x in samples]
    flat_grads = [ravel_pytree(g)[0] for g in grads]
    grads_flat = jnp.stack(flat_grads)

    # QGT 计算
    term1 = jnp.mean(grads_flat[..., None] * grads_flat[:, None, :].conj(), axis=0)
    g_mean = jnp.mean(grads_flat, axis=0)
    term2 = g_mean[..., None] * g_mean[None, :].conj()
    S = term1 - term2

    # 正则化
    S_reg = S + diag_shift * jnp.eye(S.shape[0], dtype=S.dtype)

    return S_reg, ravel_pytree(params)[1]


def compute_natural_gradient(grad, qgt_reg, unravel_fn):
    """
    计算自然梯度

    natural_grad = S⁻¹ × grad
    """
    grad_flat, grad_unravel_fn = ravel_pytree(grad)
    natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad_flat)
    return natural_grad


# ==============================================================================
# 8. MCMC 采样器
# ==============================================================================
def generate_random_initial_states(hi_ext, n_chains, seed=42):
    """生成满足 x¹ ≠ x² 约束的随机初始态"""
    key = jax.random.PRNGKey(seed)
    n_spin = hi_ext.size // 2
    init_states = []

    for _ in range(n_chains):
        key, k1, k2 = jax.random.split(key, 3)
        while True:
            s1 = hi_ext.random_state(k1)
            s2 = hi_ext.random_state(k2)
            x1 = s1[:n_spin]
            x2 = s2[n_spin:]
            if not jnp.all(x1 == x2):
                break
            key, k2 = jax.random.split(key)
        ext_state = jnp.concatenate([x1, x2])
        init_states.append(ext_state)

    return jnp.stack(init_states)


def init_sampler_state(hi, n_chains, seed=42):
    """初始化采样器状态"""
    init_states = generate_random_initial_states(hi, n_chains, seed)
    key = jax.random.PRNGKey(seed)
    chain_keys = jax.random.split(key, n_chains)
    return (init_states, chain_keys)


def make_get_all_next_states(K, SINGLE_HILBERT_SIZE, edges):
    """生成所有可能的下一状态"""
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
    """Metropolis-Hastings 单步"""
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
    n_samples_per_chain,
    n_warmup,
    sampler_state,
    edges,
    machine,
    params,
    sweep_size=32,
    K=2,
    SINGLE_HILBERT_SIZE=4,
):
    """多链 MCMC 采样器"""
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
# 9. 主训练函数
# ==============================================================================
def train_nes_vmc_natural_gradient(
    n_iter=300,
    n_chains=16,
    n_warmup=50,
    n_samples_per_chain=100,
    sweep_size=32,
    learning_rate=0.01,
    qgt_diag_shift=0.01,
    seed=42,
    hidden_dim=12,
    print_every=20,
):
    """
    使用自然梯度下降训练 NES-VMC

    返回:
        history: 训练历史记录
    """
    print("\n" + "=" * 60)
    print("NES-VMC 自然梯度下降训练")
    print("=" * 60)
    print(f"超参数:")
    print(f"  - 迭代次数: {n_iter}")
    print(f"  - 链数: {n_chains}")
    print(f"  - 每链样本数: {n_samples_per_chain}")
    print(f"  - 热化步数: {n_warmup}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - QGT 正则化: {qgt_diag_shift}")
    print(f"  - 隐藏层维度: {hidden_dim}")
    print("=" * 60)

    # 初始化模型
    rngs = nnx.Rngs(seed)
    total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=hidden_dim, rngs=rngs)
    total_machine, total_graphdef, total_params = create_machine(total_ansatz)
    total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)

    # 创建单态 machine 列表
    single_machine_list = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine(ansatz)
        single_machine_list.append(m)

    # 初始化优化器
    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(total_params)

    # 初始化采样器
    sampler_state = init_sampler_state(hi_ext, n_chains, seed=seed)

    # 训练历史
    history = {
        'step': [],
        'loss': [],
        'eigvals': [],
        'qgt_condition': [],
    }

    print(f"\nFCI 基准: E0 = {E_fcis[0]:.8f} Ha, E1 = {E_fcis[1]:.8f} Ha")
    print("-" * 60)

    start_time = time.time()

    for step in range(n_iter):
        # MCMC 采样
        samples, sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=n_samples_per_chain,
            n_warmup=n_warmup,
            sampler_state=sampler_state,
            edges=tuple(edges),
            machine=total_machine,
            params=total_params,
            sweep_size=sweep_size,
            K=K,
            SINGLE_HILBERT_SIZE=SINGLE_HILBERT_SIZE,
        )

        x_batch = samples.reshape(-1, K, SINGLE_HILBERT_SIZE)

        # 计算梯度
        grad, loss_mean, E_L_mean = nes_vmc_gradient(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
        )

        # 乘以 2（复数梯度因子）
        grad = jax.tree_util.tree_map(lambda x: x * 2, grad)

        # 计算 QGT 和自然梯度
        qgt_reg, _ = compute_nes_qgt(
            total_machine, total_params, x_batch, diag_shift=qgt_diag_shift
        )

        # 检查 QGT 条件数
        cond_number = jnp.linalg.cond(qgt_reg)
        if step % print_every == 0:
            history['qgt_condition'].append(float(cond_number))

        # 计算自然梯度
        natural_grad = compute_natural_gradient(grad, qgt_reg, None)

        # 更新参数
        updates, opt_state = optimizer.update(natural_grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        # 记录历史
        if step % print_every == 0 or step == n_iter - 1:
            eig_vals = jnp.linalg.eigvalsh(E_L_mean.real)
            history['step'].append(step)
            history['loss'].append(float(loss_mean))
            history['eigvals'].append(eig_vals)

            print(f"Step {step:4d} | Loss: {loss_mean:12.8f} | "
                  f"E0: {eig_vals[0]:.8f} (FCI: {E_fcis[0]:.8f}) | "
                  f"E1: {eig_vals[1]:.8f} (FCI: {E_fcis[1]:.8f}) | "
                  f"Cond(QGT): {cond_number:.2e}")

    end_time = time.time()

    print("-" * 60)
    print(f"训练完成! 耗时: {end_time - start_time:.2f} 秒")
    print("=" * 60)

    return history, total_params, total_graphdef


# ==============================================================================
# 10. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    # 训练参数
    N_ITER = 300
    N_CHAINS = 16
    N_WARMUP = 50
    N_SAMPLES_PER_CHAIN = 100
    SWEEP_SIZE = 32
    LEARNING_RATE = 0.02
    QGT_DIAG_SHIFT = 0.01

    # 训练
    history, final_params, graphdef = train_nes_vmc_natural_gradient(
        n_iter=N_ITER,
        n_chains=N_CHAINS,
        n_warmup=N_WARMUP,
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        sweep_size=SWEEP_SIZE,
        learning_rate=LEARNING_RATE,
        qgt_diag_shift=QGT_DIAG_SHIFT,
        seed=42,
        hidden_dim=12,
        print_every=30,
    )

    # 最终结果分析
    print("\n" + "=" * 60)
    print("最终结果")
    print("=" * 60)

    final_eigvals = history['eigvals'][-1]
    print(f"\nNES-VMC 计算的激发态能量:")
    for i, e in enumerate(final_eigvals):
        fci_e = E_fcis[i] if i < len(E_fcis) else None
        if fci_e is not None:
            error = (e - fci_e) * 27.2114
            print(f"  E{i} = {e:.8f} Ha (FCI: {fci_e:.8f} Ha) | 误差: {error:.4f} eV")
        else:
            print(f"  E{i} = {e:.8f} Ha")

    print("\n基态能量对比:")
    print(f"  NES-VMC: {final_eigvals[0]:.8f} Ha")
    print(f"  FCI:     {E_fcis[0]:.8f} Ha")
    print(f"  误差:    {(final_eigvals[0] - E_fcis[0]) * 27.2114:.4f} eV")