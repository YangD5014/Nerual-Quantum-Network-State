"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本（30 轮）——P8：双侧一致列规范 + 周期性 gauge reset

要点（与基线 create_gauge_fixed_total_machines 的唯一差别）：
    1) _compute_L_centered_single 末尾：L[walkers, j] -= stop_gradient(g[j])，
       其中 g 是 (K,) 全局列规范（非训练参数，只在 reset 边界手动更新）；
    2) loss 里 Ham_Psi_scaled 算出的 HPsi_stable 再按列乘以 exp(-g)（右侧乘 D_g）。

为什么这是精确的（数学证明）：
    - 基线中 Psi_Matrix = Ψ_raw · D_ref（每列除以 ψ_j(ref)），
      HPsi = H·Ψ_raw·D_ref（single_machine 是 gauge-fixed 的），
      故 E_L = D_ref^{-1} · (Ψ_raw^{-1}·H·Ψ_raw) · D_ref —— 相似变换，
      loss = trace(E_L)、E0-E3（特征值）均严格规范不变。
    - P8 再右乘 D_g = diag(e^{-g})：
        E_L' = (Ψ D_g)^{-1}·(HΨ D_g) = D_g^{-1}·E_L·D_g —— 仍相似变换。
      对任意常数 D_g：trace(D_g^{-1} M D_g) = trace(M) ⇒ loss 逐位不变；
      且 d/dθ trace(D_g^{-1} M D_g) = trace(dM/dθ)（D_g 常数，迹循环性）⇒ 梯度逐位不变。
      dlogΨ'/dθ：W'_{ij} = (A·D_g)^{-H} ∝ (A^{-1})_{ji}·A_{ij} = W ⇒ 逐位不变。
      ⇒ 参数轨迹与基线完全相同（loss 下降曲线、raw 梯度、E0-E3 均与基线一致）。
    - 但 logΨ'_b = slogdet(exp(L_b - g)) = logΨ_b - Σ_j g_j ⇒ 在 reset 边界把
      g 吸收掉当前列均值后，logΨ mean 被重新居中，不再随训练单调升高；
      L 幅值有界 ⇒ exp(L-shift) 永不溢出 ⇒ 基线在 step~113 的 float32 精度崩塌
      （logΨ max ~ 30-50）被彻底消除。

reset 规则：每 RESET_PERIOD 步，用 RAW L（未减 gauge）的列均值更新：
    g[j] += mean_{walker, config} Re/Im L[walker, config, j]   （复数）
这样每步的"漂移"部分被 g 吸收，报告中的 logΨ（已减 g）保持平稳。

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min30_gauge_reset.py
"""
import logging
import os
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from scipy.sparse.linalg import eigsh

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    Ham_Psi_scaled,
    flatten_batched_pytree,
    NESFermionHopRule,
    ravel_pytree,
)


# ====================== P8 核心：双侧一致列规范 + gauge 可重置的 total machine ======================
def create_gauge_reset_total_machines(total_model, ref_state):
    """
    与 NES_VMC_V1.create_gauge_fixed_total_machines 完全一致，唯一区别：
        1) _compute_L_centered_single 末尾追加 L_centered -= sg(g)（按状态=列）；
        2) 新增 total_matrix_machine_raw：不施加 gauge 的原始 L，供 reset 边界测量漂移。

    g 作为**显式动态参数**传入所有 jit 函数（避免 JAX jit 闭包捕获 list 的陷阱：
    闭包中 gauge_holder[0] 只在首次编译时被固化，Python 侧更新永远进不了计算图）。

    返回:
        total_machine, total_matrix_machine, total_max_machine, total_matrix_machine_raw,
        graphdef, state
    """
    graphdef, state = nnx.split(total_model)

    K_ = total_model.K
    n_spin = total_model.n_spin
    flat_size = K_ * n_spin

    ref_state = jnp.asarray(ref_state)

    def _compute_L_centered_single(m, x_single, g, with_gauge):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            log_col = ansatz_j(x_single)          # (K,)
            log_ref = ansatz_j(ref_state)         # scalar
            log_col_centered = log_col - log_ref  # 列规范（与基线一致，不停梯度）
            cols.append(log_col_centered)
        L_centered = jnp.stack(cols, axis=1)      # (K, K)

        if with_gauge:
            # ---- P8：减去全局列规范 g（stop_gradient，动态参数，非训练参数）----
            L_centered = L_centered - jax.lax.stop_gradient(g)[None, :]
            # -----------------------------------------------------------
        return L_centered

    def _stable_from_L(L):
        shift = jnp.max(jnp.real(L))
        shift = jax.lax.stop_gradient(shift)

        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_det_stable = log_abs_det + 1j * jnp.angle(sign)

        log_det_centered = log_det_stable + K_ * shift

        return log_det_centered, L_stable, shift

    def _one_from_matrix(m, x_single, g, with_gauge=True):
        L = _compute_L_centered_single(m, x_single, g, with_gauge)
        return _stable_from_L(L)

    def _as_single_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            assert sigma.shape[0] == flat_size, f"single flat sigma.shape={sigma.shape}"
            return sigma.reshape(K_, n_spin)
        elif sigma.ndim == 2:
            assert sigma.shape == (K_, n_spin), f"single matrix sigma.shape={sigma.shape}"
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to single walker")

    def _as_batch_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 2:
            assert sigma.shape[-1] == flat_size, f"batch flat sigma.shape={sigma.shape}"
            return sigma.reshape(-1, K_, n_spin)
        elif sigma.ndim == 3:
            assert sigma.shape[1:] == (K_, n_spin), f"batch matrix sigma.shape={sigma.shape}"
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to batch walkers")

    @jax.jit
    def total_machine(params, sigma, g):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single, g)[0]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single, g)[0]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x, g)[0])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x, g)[0])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_matrix_machine(params, sigma, g):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single, g)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single, g)[1]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x, g)[1])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x, g)[1])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_max_machine(params, sigma, g):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single, g)[2]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single, g)[2]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x, g)[2])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x, g)[2])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_matrix_machine_raw(params, sigma):
        """原始 L（未减 gauge），供 reset 边界测量漂移。"""
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single, None, with_gauge=False)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single, None, with_gauge=False)[1]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x, None, with_gauge=False)[1])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x, None, with_gauge=False)[1])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    return (
        total_machine,
        total_matrix_machine,
        total_max_machine,
        total_matrix_machine_raw,
        graphdef,
        state,
    )


# ====================== P8 loss：与 Ψ 侧一致地施加列规范 D_g（右侧乘 e^{-g}） ======================
def NES_loss_energy_stable_gauge(
    ha,
    total_matrix_machine,
    total_max_machine,
    single_machine_list,
    total_params,
    x,
    g,
    return_aux: bool = False,
):
    L_stable = total_matrix_machine(total_params, x, g)      # 已含 -gauge
    shift = total_max_machine(total_params, x, g)

    Psi_Matrix_stable = jnp.exp(L_stable)

    HPsi_stable = Ham_Psi_scaled(
        ha=ha,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x,
        shift=shift,
    )

    # ---- P8：HΨ 侧施加与 Ψ 侧相同的列规范 D_g（每列乘 e^{-g_j}）----
    g_sg = jax.lax.stop_gradient(g)
    HPsi_stable = HPsi_stable * jnp.exp(-g_sg)[None, :]
    # --------------------------------------------------------------------------

    E_L_matrix = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    loss_batch = jnp.real(jnp.trace(E_L_matrix, axis1=-2, axis2=-1))

    if not return_aux:
        return loss_batch, E_L_matrix

    valid = (
        jnp.all(jnp.isfinite(Psi_Matrix_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(HPsi_stable), axis=(-2, -1))
        & jnp.all(jnp.isfinite(E_L_matrix), axis=(-2, -1))
        & jnp.isfinite(loss_batch)
    )

    aux = {
        "L_stable": L_stable,
        "shift": shift,
        "Psi_Matrix_stable": Psi_Matrix_stable,
        "HPsi_stable": HPsi_stable,
        "valid": valid,
        "cond_Psi": jax.vmap(jnp.linalg.cond)(Psi_Matrix_stable)
        if Psi_Matrix_stable.ndim == 3
        else jnp.linalg.cond(Psi_Matrix_stable),
    }
    return loss_batch, E_L_matrix, aux


def _masked_mean_batch(x, valid, eps=1e-12):
    valid = jax.lax.stop_gradient(valid)
    valid_f = valid.astype(jnp.float32)
    x_safe = jnp.where(jnp.isfinite(x), x, 0.0)

    if x.ndim == 1:
        numerator = jnp.sum(x_safe * valid_f)
    else:
        shape = (valid_f.shape[0],) + (1,) * (x.ndim - 1)
        numerator = jnp.sum(x_safe * valid_f.reshape(shape), axis=0)

    denominator = jnp.maximum(jnp.sum(valid_f), eps)
    return numerator / denominator


def nes_vmc_gradient_stable_gauge(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
    total_params,
    x_batch,
    g,
    min_valid_ratio: float = 0.25,
    return_aux: bool = False,
):
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable_gauge(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch,
        g=g,
        return_aux=True,
    )

    valid = jax.lax.stop_gradient(loss_aux["valid"])

    batch_size = x_batch.shape[0]
    n_valid = jnp.sum(valid.astype(jnp.float32))
    valid_ratio = n_valid / batch_size

    E_L_mean = _masked_mean_batch(E_L_batch, valid)

    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0)
    E_L_centered = E_L_safe - E_L_mean

    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)
    tr_centered = jnp.where(valid, tr_centered, 0.0)

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0, None))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch, g)

    def weight_and_masked_mean(grad_component):
        grad_safe = jnp.where(jnp.isfinite(grad_component), grad_component, 0.0)
        weights = tr_centered.reshape(
            (-1,) + (1,) * (grad_component.ndim - 1)
        )
        valid_f = valid.astype(jnp.float32).reshape(
            (-1,) + (1,) * (grad_component.ndim - 1)
        )
        weighted = weights * jnp.conj(grad_safe) * valid_f
        denom = jnp.maximum(n_valid, 1.0)
        return jnp.sum(weighted, axis=0) / denom

    grad = jax.tree.map(weight_and_masked_mean, dlogPsi_batch)

    loss_mean = _masked_mean_batch(loss_batch, valid)

    grad_flat, _ = ravel_pytree(grad)
    grad_finite = jnp.all(jnp.isfinite(grad_flat))
    loss_finite = jnp.isfinite(loss_mean)
    enough_valid = valid_ratio >= min_valid_ratio

    aux = {
        "valid": valid,
        "n_valid": n_valid,
        "valid_ratio": valid_ratio,
        "grad_finite": grad_finite,
        "loss_finite": loss_finite,
        "enough_valid": enough_valid,
        "should_skip": ~(grad_finite & loss_finite & enough_valid),
        "loss_aux": loss_aux,
    }

    if return_aux:
        return grad, loss_mean, E_L_mean, aux

    return grad, loss_mean, E_L_mean


def make_grad_fn_gauge(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
):
    @jax.jit
    def grad_fn(total_params, x_batch, g):
        return nes_vmc_gradient_stable_gauge(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_max_machine=total_max_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
            g=g,
            return_aux=True,
        )

    return grad_fn


def make_qgt_fn_gauge(machine):
    """QGT：与基线 make_qgt_fn 一致，仅多一个显式 gauge 参数 g（动态传入）。"""
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0, None))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g):
        n_samples = sigma.shape[0]

        grad_tree_batch = vmap_grad_logpsi(params, sigma, g)

        O = flatten_batched_pytree(grad_tree_batch, n_samples)

        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean

        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))

        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I

    return qgt_fn


# ====================== 行规范坐标函数（监控 RAW L 的漂移，与 gauge 吸收量对照） ======================
def make_gauge_fn(total_model, ref_state):
    graphdef, state = nnx.split(total_model)
    K_ = total_model.K
    n_spin = total_model.n_spin
    ref_state = jnp.asarray(ref_state)

    def _one(m, x_single):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
        L = jnp.stack(cols, axis=1)
        row_mean = jnp.mean(L, axis=1)
        return jnp.sum(row_mean)

    @jax.jit
    def gauge_fn(params, x_batch):
        m = nnx.merge(graphdef, params)
        gs = jax.vmap(lambda x: _one(m, x))(x_batch)
        return jnp.mean(gs)

    @jax.jit
    def col_mean_fn(params, x_batch):
        """raw L 的列均值向量 (K,)，复数：供 reset 边界做 gauge 吸收。"""
        m = nnx.merge(graphdef, params)

        def _offset(x_single):
            cols = []
            for j in range(K_):
                ansatz_j = m.single_ansatz_list[j]
                cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
            return jnp.mean(jnp.stack(cols, axis=1), axis=0)  # (K,)

        return jnp.mean(jax.vmap(_offset)(x_batch), axis=0)

    return gauge_fn, col_mean_fn


# ====================== 日志配置 ======================
logger = logging.getLogger("NES_VMC_K4_gauge_reset")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

os.makedirs("./日志", exist_ok=True)
log_path = "./日志/nes_vmc_0829_K4_LiH_STO-3G_min30_gauge_reset.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 300
N_ITER = 400
Natural_Grad = True
clip_norm = 20.0
lr = 0.1
qgt_diag_shift = 0.1
RESET_PERIOD = 10          # 每隔多少步做一次 gauge reset（30 轮内 3 次）

# ====================== 模型与采样器 ======================
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)

g_current = jnp.zeros(K, dtype=jnp.complex64)   # 全局列规范 g，非训练参数，动态传入 jit

(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_matrix_machine_raw,
    total_graphdef,
    total_params,
) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

single_machine_list = [
    create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
    for ansatz in total_ansatz.single_ansatz_list
]

grad_fn = make_grad_fn_gauge(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
)
qgt_fn = make_qgt_fn_gauge(total_machine)

gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
)

# FCI 精确参考能量
eigvals, _ = eigsh(ha.to_sparse(), k=K, which="SA", tol=1e-10)

# ====================== 优化器 ======================
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

sampler_rng = jax.random.PRNGKey(21)


def sample_machine(params, sigma):
    """双参数封装：采样只需要 |Ψ|² 的转移率，全局列规范 g 是常数平移，不影响 ratio。
    闭包每次调用读取 g_current 的最新值（g 作为动态参数传入 jit）。"""
    return total_machine(params, sigma, g_current)


sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

# ====================== 训练循环 ======================
logger.info("\n" + "=" * 60)
logger.info("开始多链 NES-VMC 训练 | P8: 双侧一致列规范 + 周期性 gauge reset")
logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
    f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}, RESET_PERIOD={RESET_PERIOD}")

start_time = time.time()
for step in range(N_ITER):
    # 1. 采样
    samples_raw, sampler_state = nes_sampler.sample(
        machine=sample_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 梯度与损失
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current)

    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
    grad_update = grad_raw

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

    # 3. QGT 自然梯度
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    # 4. 优化器内部完成梯度裁剪 + SGD 更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    # 5. 能量监控：E_L 矩阵本征值
    eig_vals, _ = jnp.linalg.eig(E_L_mean)
    eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

    # 6. 波函数与Ψ矩阵条件数监控（含 gauge 的值）
    log_Psi_batch = total_machine(total_params, x_batch, g_current)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single, g_current)[0]
    psi_cond = jnp.linalg.cond(psi_mat)

    gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))          # RAW L 的行规范坐标
    g_abs = float(jnp.linalg.norm(g_current))

    # 7. 日志（格式与 GaugeFixing00_K4_LiH_STO-3G.log 对齐）
    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"原始规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | 累计|g| = {g_abs:.4f}")
    logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
    logger.info(
        f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
        f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
    )
    logger.info("#-----------------------------------------#")

    # 8. 周期性 gauge reset：把 raw L 的列均值吸收进全局 g（双侧规范，物理不变）
    if (step + 1) % RESET_PERIOD == 0:
        new_col_mean = col_mean_fn(total_params, x_batch)   # (K,) 复数：当前列均值
        g_current = g_current + new_col_mean                # 模块级变量直接更新
        absorbed = float(jnp.sum(jnp.real(new_col_mean)))
        logger.info(
            f"[GaugeReset@{step + 1}] 吸收 Δg = Σ Re(col_mean) = {absorbed:+.4f} "
            f"| 新|g| = {float(jnp.linalg.norm(g_current)):.4f}"
        )

end_time = time.time()
logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
logger.info("\n" + "=" * 60)
logger.info("训练完成!")
logger.info("=" * 60)