"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

基于论文 S8 的 log-domain 稳定化框架，用于计算量子多体系统的激发态能量。

关键改造:
1. Ansatz 返回真实 log_Psi、L_stable 和 L_max
2. wrapper 函数统一三返回值处理
3. 稳定化损失函数使用 L_max 对 M 和 HΨ 进行同步缩放

注意: NetKet 的 Numba operator 不能在 JAX vmap/jit 内部调用，
这限制了 log-domain HΨ 计算的实现。当前版本使用原始 HΨ 计算。
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
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import time
from jax.flatten_util import ravel_pytree
from collections import Counter
import logging

# ==============================================================================
# 日志配置
# ==============================================================================
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ==============================================================================
# 1. 全局参数 & H₂ 分子定义
# ==============================================================================
logger.info("初始化 H₂ 分子和 FCI 基准能量")

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
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

logger.info(f"FCI 基准能量: E0={E_fcis[0]:.8f}, E1={E_fcis[1]:.8f}, E2={E_fcis[2]:.8f}")

ha = nkx.operator.from_pyscf_molecule(mol)
logger.debug(f"Hamiltonian 算符创建完成, hilbert space size: {ha.hilbert.size}")

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
logger.info(f"Hilbert space: {hi.size} 个自旋轨道")

# NES 扩展系统参数
K = 3
hi_ext = hi ** K
logger.info(f"扩展 Hilbert space (K={K}): {hi_ext.size} 个自旋轨道")

# H₂ / STO-3G / SpinOrbitalFermions(n_orbitals=2) 下，单个副本的跃迁边
single_edges = ((0, 1), (2, 3))
logger.debug(f"单粒子跃迁边: {single_edges}")

# ==============================================================================
# 2. 单态 Ansatz 定义
# ==============================================================================
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.hidden_dim = hidden_dim
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)
        logger.debug(f"SingleStateAnsatz 初始化: n_spin={n_spin_orbitals}, hidden={hidden_dim}")

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

# ==============================================================================
# 3. NES Total Ansatz
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)

            ansatz = SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)

            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )

            # 这里只返回原始 L，不做 exp，不做 slogdet
            return L

        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)

        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)

        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)

        elif x.ndim == 1:
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)

        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')
# ==============================================================================
# ==============================================================================
# 4. Wrapper 函数
# ==============================================================================
def create_machine(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        L = m(sigma)

        L_max = jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)
        L_max = jax.lax.stop_gradient(L_max)

        L_stable = L - L_max
        M_stable = jnp.exp(L_stable)

        sign, log_abs_det_stable = jnp.linalg.slogdet(M_stable)

        K = L.shape[-1]
        log_Psi = (
            log_abs_det_stable
            + K * jnp.squeeze(L_max, axis=(-2, -1))
            + 1j * jnp.angle(sign)
        )

        return log_Psi

    return machine, graphdef, state

def create_machine_aux(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine_aux(params, sigma):
        m = nnx.merge(graphdef, params)
        L = m(sigma)

        L_max = jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)
        L_max = jax.lax.stop_gradient(L_max)

        L_stable = L - L_max

        return L_stable, L_max

    return machine_aux, graphdef, state


def create_single_machine(model: SingleStateAnsatz):
    """将单态 Ansatz 包装为 machine 函数"""
    logger.info("创建 single machine")
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state

# ==============================================================================
# 5. HΨ 计算（对数域版本）
# ==============================================================================
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

# ==============================================================================
# 5. HΨ 计算
# ==============================================================================
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    🔥 同时支持：
    - 单个态 x: (n_spin,)
    - 批量态 x: (batch_size, n_spin)
    """
    # ======================
    # 核心：自动给单个样本增加 batch 维度
    # ======================
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    # ======================
    # 向量化计算（批处理）
    # ======================
    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    # 批量处理
    H_psi_batch = jax.vmap(_single_hpsi)(x)

    # ======================
    # 如果是单个输入，就压回单个输出
    # ======================
    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch
    
def Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)
    # ======================
    # 核心：单样本 与 批处理 自动兼容
    # ======================
    if x.ndim == 2:
        # 输入形状：(K, n_spin) → 单个扩展态 → 返回 (K,K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # 单态：(4,)
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 输入形状：(batch, K, n_spin) → 批量 → 返回 (batch, K, K)
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi
        
        # 自动批处理！
        return jax.vmap(_single_HamPsi)(x)

    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")

# ==============================================================================
# 6. 损失函数（稳定化版本）
# ==============================================================================
def NES_loss_energy_stable(
    ha,
    total_aux_machine,
    single_machine_list,
    total_params,
    x
):
    """
    快速稳定版 NES loss。

    total_aux_machine 返回:
    - L_stable
    - L_max
    """

    # 1. 一次性取出 L_stable 和 L_max
    L_stable, L_max = total_aux_machine(total_params, x)

    # 2. M_stable = exp(L - L_max)
    Psi_Matrix_stable = jnp.exp(L_stable)

    # 3. 用原版快速 vmap 计算 HPsi
    HPsi = Ham_Psi(
        ha,
        single_machine_list,
        total_params,
        x
    )

    # 4. 同步缩放 HPsi
    HPsi = jax.lax.stop_gradient(HPsi)

    if L_stable.ndim == 2:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(1, 1))
    else:
        HPsi_stable = HPsi * jnp.exp(-L_max.reshape(-1, 1, 1))

    # ========== 防护：检查矩阵条件数 ==========
    # 如果 Psi_Matrix_stable 条件数过大，使用正则化
    cond_num = jnp.linalg.cond(Psi_Matrix_stable)

    # 5. E_L = M_stable^{-1} HPsi_stable
    # 使用 lstsq 而非 solve，更稳定
    E_L = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))

    return loss, E_L
# ==============================================================================
# 8. NES 梯度计算
# ==============================================================================
def nes_vmc_gradient(
    ha,
    total_aux_machine,
    total_machine,
    single_machine_list,
    total_params,
    x_batch
):
    # 1. 计算局域能量矩阵
    loss_batch, E_L_batch = NES_loss_energy_stable(
        ha=ha,
        total_aux_machine=total_aux_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch
    )

    E_L_mean = jnp.mean(E_L_batch, axis=0)

    # 2. 中心化
    E_L_centered = E_L_batch - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    # 3. 计算 ∇logΨ
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 4. 加权平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()

    return grad, loss_mean, E_L_mean
# ==============================================================================
# 9. QGT 计算
# ==============================================================================
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

# ==============================================================================
# 10. 采样器规则
# ==============================================================================
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

# ==============================================================================
# 11. 辅助函数
# ==============================================================================
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
        print(f"元组 {tpl} 出现了 {count_} 次")
    return count

# ==============================================================================
# 12. 主程序
# ==============================================================================
if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("NES-VMC 训练开始")
    logger.info("=" * 60)

    # 参数设置
    N_CHAINS = 16
    N_WARMUP = 100
    N_SAMPLES_PER_CHAIN = 200
    SWEEP_SIZE = 30
    N_ITER = 400
    Natural_Grad = True
    SINGLE_SIZE = hi.size

    logger.info(f"参数: K={K}, N_CHAINS={N_CHAINS}, N_SAMPLES_PER_CHAIN={N_SAMPLES_PER_CHAIN}")
    logger.info(f"      N_ITER={N_ITER}, Natural_Grad={Natural_Grad}, SWEEP_SIZE={SWEEP_SIZE}")

    # 创建 Ansatz
    logger.info("创建 NESTotalAnsatz...")
    total_ansatz = NESTotalAnsatz(4, K, 12, rngs=nnx.Rngs(11))
    logger.info("Ansatz 创建完成")

    # 创建各种 wrapper
    logger.info("创建 wrapper functions...")
    total_machine, total_graphdef, total_params = create_machine(total_ansatz)
    total_aux_machine, total_aux_graphdef, total_aux_params = create_machine_aux(total_ansatz)


    single_machine_list = []
    for i, ansatz in enumerate(total_ansatz.single_ansatz_list):
        m, g, p = create_single_machine(ansatz)
        single_machine_list.append(m)
        logger.debug(f"  创建 single_machine[{i}]")

    logger.info(f"共创建 {len(single_machine_list)} 个 single_machine")

    # 优化器
    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(total_params)
    logger.info("优化器初始化完成 (SGD, lr=0.01)")

    # 构造扩展边
    ext_edges = []
    for k in range(K):
        offset = k * SINGLE_SIZE
        for (i, j) in single_edges:
            ext_edges.append((i + offset, j + offset))
    ext_edges = jnp.array(ext_edges)
    logger.info(f"扩展跃迁边: {ext_edges.tolist()}")

    # 创建采样器
    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext,
        rule=nes_rule,
        n_chains=N_CHAINS,
        sweep_size=SWEEP_SIZE
    )
    logger.info(f"采样器创建完成: {N_CHAINS} chains, sweep_size={SWEEP_SIZE}")

    # 采样器状态初始化
    sampler_rng = jax.random.PRNGKey(21)
    sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)
    logger.info("采样器状态初始化完成")

    # 训练循环
    print("\n" + "="*60)
    print("开始多链 NES-VMC 训练")
    print("="*60)
    print(f"基态能量={E_fcis[0]:.8f} Ha | 1st激发态={E_fcis[1]:.8f} Ha | 2st激发态={E_fcis[2]:.8f} Ha")

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

    # ========== 新增：NaN 检测和早停 ==========
    nan_detected = False
    nan_step = -1
    MAX_CONSECUTIVE_NAN = 3  # 连续3次出现NaN则停止

    for step in range(N_ITER):
        logger.debug(f"=== Step {step} ===")

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
            total_aux_machine=total_aux_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch)

        grad_flat, grad_unravel_fn = ravel_pytree(grad)
        grad_norm = jnp.linalg.norm(grad_flat)
        logger.debug(f"  grad norm: {grad_norm:.4f}")

        # ========== 新增：NaN 检测 ==========
        has_nan = jnp.isnan(grad_norm) or jnp.isnan(loss_mean)

        if has_nan:
            logger.warning(f"Step {step}: 检测到 NaN! grad_norm={grad_norm}, loss={loss_mean}")
            nan_detected = True
            nan_step = step
            # 保存现场供调试
            L_stable_debug, L_max_debug = total_aux_machine(total_params, x_batch)
            logger.warning(f"  L_max 值: {L_max_debug.reshape(())}")
            break

        # 自然梯度
        if Natural_Grad:
            qgt_reg, unravel_fn = compute_qgt(
                total_machine, total_params, x_batch, diag_shift=0.001
            )
            qgt_cond = jnp.linalg.cond(qgt_reg)
            logger.debug(f"  QGT 条件数: {qgt_cond:.2e}")

            # QGT 条件数过大时，增加正则化
            if qgt_cond > 1e10:
                logger.warning(f"  QGT 条件数过大 ({qgt_cond:.2e})，增加正则化")
                qgt_reg = qgt_reg + 0.1 * jnp.eye(qgt_reg.shape[0])

            natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
            natural_grad = grad_unravel_fn(natural_grad_flat)
            grad = natural_grad
            logger.debug(f"  自然梯度计算完成")

        # 参数更新
        updates, opt_state = optimizer.update(grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        # 计算监控量
        log_Psi_batch = total_machine(total_params, x_batch)
        eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)

        # ========== 新增：L_max 监控 ==========
        L_stable, L_max = total_aux_machine(total_params, x_batch)
        L_max_mean = jnp.mean(L_max)
        M_cond = jnp.linalg.cond(jnp.exp(L_stable))

        # 记录历史
        history['step'].append(step)
        history['loss'].append(float(loss_mean))
        history['log_Psi_mean'].append(float(jnp.real(log_Psi_batch).mean()))
        history['log_Psi_min'].append(float(jnp.real(log_Psi_batch).min()))
        history['log_Psi_max'].append(float(jnp.real(log_Psi_batch).max()))
        history['grad_norm'].append(float(grad_norm))
        history['energy_0st'].append(float(eig_vals[0]))
        history['energy_1st'].append(float(eig_vals[1]))
        history['E_Lmatrix'].append(np.array(E_L_mean))
        history['samples'].append(np.array(samples))

        # 打印
        if step % 5 == 0 or step == N_ITER - 1:
            print(f"\n--- Step {step} ---")
            print(f"log_Psi(real): mean={jnp.real(log_Psi_batch).mean():.3f} | "
                  f"min={jnp.real(log_Psi_batch).min():.3f} | "
                  f"max={jnp.real(log_Psi_batch).max():.3f}")
            print(f"grad norm = {grad_norm:.4f}")
            print(f"L_max = {L_max_mean:.3f} | M_cond = {M_cond:.2e}")
            print(f"Loss: {loss_mean:.6f}")
            print(f"E0={eig_vals[0]:.8f} Ha | E1={eig_vals[1]:.8f} Ha | E2={eig_vals[2]:.8f} Ha")
            print(f"FCI基准: E0={E_fcis[0]:.8f} | E1={E_fcis[1]:.8f} | E2={E_fcis[2]:.8f}")
            print(f"误差: dE0={abs(eig_vals[0]-E_fcis[0]):.2e} | dE1={abs(eig_vals[1]-E_fcis[1]):.2e}")
            print("#" + "-"*58 + "#")

    end_time = time.time()

    if nan_detected:
        print(f"\n!!! 训练因 NaN 在 Step {nan_step} 停止 !!!")
        print(f"建议检查: L_max 是否过大，矩阵条件数是否正常")
    print(f"\n训练完成! 耗时: {end_time - start_time:.2f} 秒")

    # 最终结果
    print("\n" + "="*60)
    print("最终结果")
    print("="*60)
    final_E0 = history['energy_0st'][-1]
    final_E1 = history['energy_1st'][-1]
    print(f"NES-VMC: E0={final_E0:.8f} Ha, E1={final_E1:.8f} Ha")
    print(f"FCI:     E0={E_fcis[0]:.8f} Ha, E1={E_fcis[1]:.8f} Ha")
    print(f"误差:    dE0={abs(final_E0-E_fcis[0]):.2e}, dE1={abs(final_E1-E_fcis[1]):.2e}")
    print("="*60)
