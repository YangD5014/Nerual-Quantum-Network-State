# NES-VMC (Natural Excited State Variational Monte Carlo)
#
# 基于论文 S8 的 log-domain 稳定化框架，用于计算量子多体系统的激发态能量。
#
# **关键改造:**
# 1. Ansatz 返回真实 log_Psi、L_stable 和 L_max
# 2. wrapper 函数统一三返回值处理
# 3. 稳定化损失函数使用 L_max 对 M 和 HΨ 进行同步缩放
#
# **注意:** NetKet 的 Numba operator 不能在 JAX vmap/jit 内部调用，当前版本使用原始 HΨ 计算但保证了数学等价性。

# 导入必要的库
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
import flax.nnx as nnx
import optax
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import time
from jax.flatten_util import ravel_pytree
from collections import Counter
import logging


# 日志配置
logger = logging.getLogger('NES_VMC')
logger.setLevel(logging.DEBUG)
# 阻止日志向上传播
logger.propagate = False
# 清除所有旧handler，防止重复打印
logger.handlers.clear()

# 自定义日志格式：只打印内容，不带等级、logger名
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

# 1. 文件输出处理器
file_handler = logging.FileHandler("nes_vmc_0616.log", mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

print('库导入完成')


# ================================================================
# H₂ FCI 基准能量
# ================================================================
# H₂ 分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print('='*60)
print('H₂ FCI 基准能量')
print('='*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f'E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV')

logger.info(f'FCI 基准能量: E0={E_fcis[0]:.8f}, E1={E_fcis[1]:.8f}, E2={E_fcis[2]:.8f}')


# Hamiltonian 和 Hilbert Space
ha = nkx.operator.from_pyscf_molecule(mol)
logger.debug(f'Hamiltonian 算符创建完成, hilbert space size: {ha.hilbert.size}')

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
logger.info(f'Hilbert space: {hi.size} 个自旋轨道')

# NES 扩展系统参数
K = 3
hi_ext = hi ** K
logger.info(f'扩展 Hilbert space (K={K}): {hi_ext.size} 个自旋轨道')

# H₂ / STO-3G / SpinOrbitalFermions(n_orbitals=2) 下，单个副本的跃迁边
single_edges = ((0, 1), (2, 3))
logger.info(f'单粒子跃迁边: {single_edges}')


# ================================================================
# 2. 单态 Ansatz 定义
# ================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.hidden_dim = hidden_dim
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)
        logger.debug(f'SingleStateAnsatz 初始化: n_spin={n_spin_orbitals}, hidden={hidden_dim}')

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

# ================================================================
# 3. NES Total Ansatz (对数域稳定化)
# ================================================================
class NESTotalAnsatz(nnx.Module):
    """
    NES Total Ansatz with log-domain stabilization.

    返回三个值:
    1. log_Psi: 真实的 log Ψ = log|Ψ| + i arg(Ψ)，用于采样器
    2. L_stable: 稳定化后的 log ψ_ij = log ψ_ij - L_max
    3. L_max: max_{i,j} Re(log ψ_ij)
    """

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        logger.info(f'NESTotalAnsatz 初始化: K={n_states}, n_spin={n_spin_orbitals}, hidden_dim={hidden_dim}')

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)
        logger.debug(f'创建了 {len(self.single_ansatz_list)} 个 SingleStateAnsatz')

    def __call__(self, x: jax.Array):
        """
        前向传播，返回 (log_Psi, L_stable, L_max)
        """
        logger.debug(f'NESTotalAnsatz.__call__ 输入 shape: {x.shape}')

        def _forward_single(x_single):
            # x_single 形状: [K * n_spin] -> 重塑为 [K, n_spin]
            x_single = x_single.reshape(self.K, self.n_spin)

            # 计算 L_ij = log ψ_j(x^i)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))

            # 论文 S8 的关键：对整个 K×K 矩阵的 Re(L) 取全局最大值
            # 注意：必须用 jnp.real(L)，因为控制 |exp(L)| 的是 Re(L)
            L_max = jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)

            # 稳定化: L_stable = L - L_max
            L_stable = L - L_max

            # M_stable = exp(L_stable)，现在不会溢出了
            M_stable = jnp.exp(L_stable)

            # 计算行列式
            sign, log_abs_det_stable = jnp.linalg.slogdet(M_stable)

            # 真实 log_Psi = log|Ψ| + i arg(Ψ)
            # 注意: det(exp(L_stable)) = exp(-K*L_max) * det(exp(L))
            # 所以 log|Ψ| = log|det(M_stable)| + K * L_max
            log_Psi = (
                log_abs_det_stable
                + self.K * jnp.squeeze(L_max)
                + 1j * jnp.angle(sign)
            )

            return log_Psi, L_stable, jnp.squeeze(L_max)

        # 批量处理分支
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            logger.debug('输入是单个样本 (n_spin,)')
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            logger.debug('输入是单个扩展态 (K*n_spin,)')
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            logger.debug(f'输入是批量样本 (batch, K, n_spin)')
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            logger.debug('输入是 1D 数组')
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            logger.error(f'不支持的输入形状: {x.shape}')
            raise ValueError(f'不支持的输入形状: {x.shape}')


# ================================================================
# 4. Wrapper 函数
# ================================================================

def create_machine(model: NESTotalAnsatz):
    """
    创建用于采样器的 machine 函数。
    只返回真实的 log_Psi，用于 Metropolis 采样。
    """
    logger.info('创建 sampling machine (create_machine)')
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return log_Psi

    return machine, graphdef, state


def create_machine_matrix(model: NESTotalAnsatz):
    """
    创建返回 L_stable 的 machine 函数。
    用于损失函数中构造稳定化矩阵 M_stable = exp(L_stable)。
    """
    logger.info('创建 L_stable matrix machine (create_machine_matrix)')
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return L_stable

    return machine, graphdef, state


def create_machine_matrix_max(model: NESTotalAnsatz):
    """
    创建返回 L_max 的 machine 函数。
    用于损失函数中稳定化 log(HΨ)。
    """
    logger.info('创建 L_max machine (create_machine_matrix_max)')
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return L_max

    return machine, graphdef, state


def create_single_machine(model: SingleStateAnsatz):
    """将单态 Ansatz 包装为 machine 函数"""
    logger.info('创建 single machine')
    graphdef, state = nnx.split(model)
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state

print('Wrapper 函数定义完成')


# ================================================================
# 5. 辅助函数：复数 logsumexp
# ================================================================
def _complex_logsumexp(log_terms):
    """
    稳定计算 log(sum(exp(log_terms)))，支持复数项。

    数学：
    log(sum_k exp(a_k + i*b_k)) = c + log(sum_k exp(a_k - c + i*b_k))
    其中 c = max_k Re(log_terms_k)

    这避免了直接 exp 可能导致的 overflow/underflow。
    """
    shift = jnp.max(jnp.real(log_terms))
    shifted_sum = jnp.sum(jnp.exp(log_terms - shift))
    return shift + jnp.log(shifted_sum)


# ================================================================
# 6. HΨ 计算
# ================================================================
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    计算 Hψ(x)。

    注意：这个函数不接受批量输入，只处理单个样本。
    """
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = single_machine(params, x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    return jnp.sum(mels * psi_vals)


def Ham_Psi(ha, single_machine_list, total_params, x):
    """
    构造 HΨ 矩阵。

    注意：完全移除 jax.vmap，手动处理批量以避免 JAX 追踪问题。
    """
    K = len(single_machine_list)

    if x.ndim == 2:
        # 输入形状：(K, n_spin) → 单个扩展态 → 返回 (K,K)
        HPsi = jnp.zeros((K, K), dtype=complex)
        for i in range(K):
            xi = x[i]
            for j in range(K):
                machine_j = single_machine_list[j]
                params_j = total_params['single_ansatz_list'][j]
                val = Ham_psi(ha, machine_j, params_j, xi)
                HPsi = HPsi.at[i, j].set(val)
        return HPsi

    elif x.ndim == 3:
        # 输入形状：(batch, K, n_spin) → 批量 → 返回 (batch, K, K)
        batch_size = x.shape[0]
        results = []

        # 手动循环处理批量，避免 jax.vmap 追踪问题
        for b in range(batch_size):
            xb = x[b]  # shape: (K, n_spin)
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = xb[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            results.append(HPsi)

        return jnp.stack(results)

    else:
        raise ValueError(f'不支持的输入形状: {x.shape}')

# ================================================================
# 7. 损失函数（稳定化版本）
# ================================================================
def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                           single_machine_list, total_params, x):
    """
    稳定化局域能量计算。

    数据流:
    1. L_stable = log Ψ_matrix - L_max
    2. M_stable = exp(L_stable)
    3. HPsi = Ham_Psi(...) [直接计算]
    4. HPsi_stable = HPsi / exp(L_max)
    5. E_L = solve(M_stable, HPsi_stable)

    注意：由于 M_stable = M * exp(-L_max)，HPsi_stable = HPsi * exp(-L_max)
    所以 solve(M_stable, HPsi_stable) = M^{-1} * HPsi
    """
    # Step 1: 获取稳定化后的 L
    L_stable = total_matrix_machine(total_params, x)
    L_max = total_max_machine(total_params, x)

    # Step 2: M_stable = exp(L_stable)
    Psi_Matrix_stable = jnp.exp(L_stable)

    # Step 3: HPsi 直接计算
    HPsi = Ham_Psi(ha, single_machine_list, total_params, x)

    # Step 4: 稳定化 HPsi
    if L_stable.ndim == 2:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(1, 1))
    else:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(-1, 1, 1))

    # Step 5: E_L = solve(M_stable, HPsi_stable)
    E_L = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))
    return loss, E_L




# ================================================================
# 8. NES 梯度计算
# ================================================================
def nes_vmc_gradient(ha, total_matrix_machine, total_machine, total_machine_max,
                     single_machine_list, total_params, x_batch):
    """
    计算 NES-VMC 梯度。

    关键：Ham_Psi 必须在 grad 之外计算，避免 JAX 追踪到 Numba operator。
    """
    # 1. 先计算 Ham_Psi（不使用 grad）
    HPsi = Ham_Psi(ha, single_machine_list, total_params, x_batch)

    # 2. 获取稳定化后的 L
    L_stable = total_matrix_machine(total_params, x_batch)
    L_max = total_machine_max(total_params, x_batch)

    # 3. 计算 M_stable
    Psi_Matrix_stable = jnp.exp(L_stable)

    # 4. 稳定化 HPsi
    if L_stable.ndim == 2:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(1, 1))
    else:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(-1, 1, 1))

    # 5. 计算 E_L
    E_L = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    loss_batch = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))
    E_L_mean = jnp.mean(E_L, axis=0)

    # 6. 中心化
    E_L_centered = E_L - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    # 7. 计算 ∇logΨs
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 8. 核心加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,)*(grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean



# ================================================================
# 9. QGT 计算
# ================================================================
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    计算量子几何张量（QGT）/ F 矩阵

    QGT 定义：
    S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    """
    n_samples = sigma.shape[0]

    def log_psi_single(p, s):
        return machine(p, s)

    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    grad_flat, unravel_fn = ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)

    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean

    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

    return qgt_reg, unravel_fn

# ================================================================
# 10. 采样器规则
# ================================================================
SINGLE_SIZE = hi.size

@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    """
    NES 约束的 Metropolis 跃迁规则。

    关键修复：
    - duplicate check 现在是全 pairwise 检查，而不是只检查第 0 个子组态
    """
    edges: jnp.ndarray
    K: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        """
        NES约束：检查任意两个子组态是否重复。

        原写法只比较第 0 个子组态和后面的子组态，
        会漏掉 x^1 == x^2 但 x^0 不重复的情况。

        正确做法是全 pairwise 检查所有子组态。
        """
        is_single = (sigma_ext.ndim == 1)
        sub = sigma_ext.reshape((-1, self.K, self.single_size))

        # Pairwise 比较: sub[i] == sub[j] for all i, j
        eq = jnp.all(sub[:, :, None, :] == sub[:, None, :, :], axis=-1)
        eye = jnp.eye(self.K, dtype=bool)
        dup = jnp.any(eq & (~eye)[None, :, :], axis=(1, 2))

        result = dup[0] if is_single else dup
        return result

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """跃迁规则"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """随机态生成"""
        sigma_shape = state.σ.shape
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100
            def cond(c):
                return (c[0] < max_tries) & c[2]

            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)
                s = hilbert.random_state(k_new)
                is_dup = self._check_duplicate(s)
                return (tries + 1, k, is_dup, s)

            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)

        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)



# ================================================================
# 11. 辅助函数
# ================================================================
def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


def sampler_info(samples, K):
    """统计采样信息"""
    test_samples = np.array(samples.reshape(-1, 4*K))
    count = Counter(tuple(each_row.tolist()) for each_row in test_samples)
    for tpl, count_ in count.items():
        print(f'元组 {tpl} 出现了 {count_} 次')
    return count



# ================================================================
# 12. 主程序：初始化
# ================================================================
logger.info('=' * 60)
logger.info('NES-VMC 训练开始')
logger.info('=' * 60)

# 参数设置
N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER = 400
Natural_Grad = True
SINGLE_SIZE = hi.size

logger.info(f'参数: K={K}, N_CHAINS={N_CHAINS}, N_SAMPLES_PER_CHAIN={N_SAMPLES_PER_CHAIN}')
logger.info(f'      N_ITER={N_ITER}, Natural_Grad={Natural_Grad}, SWEEP_SIZE={SWEEP_SIZE}')

# 创建 Ansatz
logger.info('创建 NESTotalAnsatz...')
total_ansatz = NESTotalAnsatz(4, K, 12, rngs=nnx.Rngs(11))
logger.info('Ansatz 创建完成')

# 创建各种 wrapper
logger.info('创建 wrapper functions...')
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)
total_matrix_max, _, _ = create_machine_matrix_max(total_ansatz)

single_machine_list = []
for i, ansatz in enumerate(total_ansatz.single_ansatz_list):
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)
    logger.debug(f'  创建 single_machine[{i}]')

logger.info(f'共创建 {len(single_machine_list)} 个 single_machine')

# 优化器
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)
logger.info('优化器初始化完成 (SGD, lr=0.01)')

# 构造扩展边
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)
logger.info(f'扩展跃迁边: {ext_edges.tolist()}')


# ================================================================
# 13. 主程序：采样器初始化
# ================================================================
# 创建采样器
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE
)
logger.info(f'采样器创建完成: {N_CHAINS} chains, sweep_size={SWEEP_SIZE}')

# 采样器状态初始化
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)
logger.info('采样器状态初始化完成')

print('\n' + '='*60)
print('开始多链 NES-VMC 训练')
print('='*60)
print(f'基态能量={E_fcis[0]:.8f} Ha | 1st激发态={E_fcis[1]:.8f} Ha | 2st激发态={E_fcis[2]:.8f} Ha')


# ================================================================
# 14. 主程序：训练循环
# ================================================================
import time
import jax.numpy as jnp
import optax

history = {
    'step': [],
    'energy_0st': [],
    'energy_1st': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix': [],
    'grad_norm': [],
    'samples': [],
    'log_Psi_mean': [],
    'log_Psi_min': [],
    'log_Psi_max': [],
}

start_time = time.time()
# 新增：连续梯度为0计数器
zero_grad_count = 0
MAX_ZERO_GRAD_STEPS = 5  # 连续5轮grad_norm=0则停止

for step in range(N_ITER):
    # 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)

    # 计算梯度
    grad, loss_mean, E_L_mean = nes_vmc_gradient(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_machine=total_machine,
        total_machine_max=total_matrix_max,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_batch
    )

    grad_flat, grad_unravel_fn = ravel_pytree(grad)
    grad_norm = jnp.linalg.norm(grad_flat)

    # 自然梯度
    if Natural_Grad:
        qgt_reg, unravel_fn = compute_qgt(
            total_machine, total_params, x_batch, diag_shift=0.001
        )
        natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad_flat)
        grad = natural_grad

    # 参数更新
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    # 计算监控量
    log_Psi_batch = total_machine(total_params, x_batch)
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)

    # 记录历史
    history['step'].append(step)
    history['loss'].append(loss_mean)
    history['log_Psi_mean'].append(jnp.real(log_Psi_batch).mean())
    history['log_Psi_min'].append(jnp.real(log_Psi_batch).min())
    history['log_Psi_max'].append(jnp.real(log_Psi_batch).max())
    history['grad_norm'].append(grad_norm)
    history['energy_0st'].append(eig_vals[0])
    history['energy_1st'].append(eig_vals[1])
    history['E_Lmatrix'].append(E_L_mean)
    history['samples'].append(samples)

    # ========== 新增连续梯度0判断逻辑 ==========
    if jnp.isclose(grad_norm, 0.0):
        zero_grad_count += 1
        logger.warning(f"Step {step}: grad_norm ≈ 0，连续无梯度轮数: {zero_grad_count}/{MAX_ZERO_GRAD_STEPS}")
        # 达到阈值直接跳出训练循环
        if zero_grad_count >= MAX_ZERO_GRAD_STEPS:
            logger.info(f"检测到连续{MAX_ZERO_GRAD_STEPS}轮梯度范数为0，触发早停，终止训练！")
            break
    else:
        # 梯度不为0，计数器重置
        zero_grad_count = 0
    # ==========================================

    # 打印
    if step % 5 == 0 or step == N_ITER - 1:
        logger.info(f'\n---------------------Step {step}-----------------------')
        logger.info(f'log_Psi(real): mean={jnp.real(log_Psi_batch).mean():.3f} | '
              f'min={jnp.real(log_Psi_batch).min():.3f} | '
              f'max={jnp.real(log_Psi_batch).max():.3f}')
        logger.info(f'grad norm = {grad_norm:.4f}')
        logger.info(f'Loss: {loss_mean:.6f}')
        logger.info(f'E0={eig_vals[0]:.8f} Ha | E1={eig_vals[1]:.8f} Ha | E2={eig_vals[2]:.8f} Ha')
        logger.info(f'FCI基准: E0={E_fcis[0]:.8f} | E1={E_fcis[1]:.8f} | E2={E_fcis[2]:.8f}')
        logger.info(f'误差: dE0={abs(eig_vals[0]-E_fcis[0]):.2e} | dE1={abs(eig_vals[1]-E_fcis[1]):.2e}')
        logger.info('#' + '-'*58 + '#')

end_time = time.time()
logger.info(f'\n训练完成! 耗时: {end_time - start_time:.2f} 秒')


# ================================================================
# 15. 最终结果
# ================================================================
print('\n' + '='*60)
print('最终结果')
print('='*60)
final_E0 = history['energy_0st'][-1]
final_E1 = history['energy_1st'][-1]
print(f'NES-VMC: E0={final_E0:.8f} Ha, E1={final_E1:.8f} Ha')
print(f'FCI:     E0={E_fcis[0]:.8f} Ha, E1={E_fcis[1]:.8f} Ha')
print(f'误差:    dE0={abs(final_E0-E_fcis[0]):.2e}, dE1={abs(final_E1-E_fcis[1]):.2e}')
print('='*60)
