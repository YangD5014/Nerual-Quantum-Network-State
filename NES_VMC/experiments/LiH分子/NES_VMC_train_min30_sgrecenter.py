"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本（30 轮）——P5：per-walker 按列 sg 均值归一化

动机（来自 gaugepin μ=20 脊、diag_shift=1.0 的负结果）：
    1) QGT 上加规范方向脊（μ=20）完全没改变漂移速率 -> 漂移不是单条零模的噪声放大；
    2) diag_shift=1.0 只把漂移速率降 2-3 倍且提前在第 17 步爆炸（基数第 113 步）;
    3) 爆炸前 logΨ max 达 ~48 -> exp(48)~7e20，float32 精度崩塌是爆炸直接机制;
    4) rowfix（改 logdet 路径）梯度方向改变 90.6%，loss 卡 -24.6。

P5 思路（只改 logΨ 的"值"，不动梯度路径、不动 loss 公式）：
    L[i,j] = logψ_j(x_i) - logψ_j(ref)            （列规范，与基线一致）
    L'[i,j] = L[i,j] - sg( mean_i L[i,j] )        （每个 walker 内按列减均值, stop_gradient）
      - 每个 walker 始终有 K 个 config，列均值是良定义的 (K,) 向量；
      - 值层面：Ψ' = Ψ·diag(e^{-mean_j})（列缩放，E_L' = diag(e^{-mean})·E_L，trace 权重变化为
        O(1) 有界量，且不会随训练发散——这正是要抑制的"列规范漂移"）；
      - 梯度层面：sg 使 mean 平坦，dL'/dθ 与基线的差仅是每列乘 e^{-mean_j}（有界），
        不做任何 90° 旋转（对比 rowfix 的 90.6% 失配）；
      - 数值层面：L' 的每列均值为 0 -> |L'| 有界 -> exp(L') 不溢出 -> 消除精度崩塌爆炸源；
      - logΨ' mean = logΨ mean - Σ_j mean_j -> 规范漂移在值层面被压住。

若 P5 的 loss 曲线与基线(->-30.48)相当且 logΨ 不再漂移，则三重要求同时满足。

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min30_sgrecenter.py
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
    make_grad_fn,
    make_qgt_fn,
    NESFermionHopRule,
)


# ====================== P5 核心：per-walker 按列 sg 均值归一化的 total machine ======================
def create_sgrecenter_total_machines(total_model, ref_state):
    """
    与 NES_VMC_V1.create_gauge_fixed_total_machines 完全一致，
    唯一区别：_compute_L_centered_single 末尾追加
        L_centered = L_centered - stop_gradient(mean over rows(configs), keepdims=True)
    （按列 = 按状态归一化，压下列规范/整体标度漂移；梯度经 sg 保持不变向路径。）

    返回:
        total_machine(params, sigma)        -> logΨ'
        total_matrix_machine(params, sigma) -> L'_stable
        total_max_machine(params, sigma)    -> shift
        graphdef, state
    """
    graphdef, state = nnx.split(total_model)

    K_ = total_model.K
    n_spin = total_model.n_spin
    flat_size = K_ * n_spin

    ref_state = jnp.asarray(ref_state)

    def _compute_L_centered_single(m, x_single):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            log_col = ansatz_j(x_single)                 # (K,)
            log_ref = ansatz_j(ref_state)                # scalar
            log_col_centered = log_col - log_ref
            cols.append(log_col_centered)
        L_centered = jnp.stack(cols, axis=1)             # (K, K)

        # ---- P5 唯一新增：按列（状态方向）减均值，stop_gradient ----
        col_mean = jax.lax.stop_gradient(
            jnp.mean(L_centered, axis=0, keepdims=True)  # (1, K)
        )
        L_centered = L_centered - col_mean
        # ---------------------------------------------------------
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

    def _one_from_matrix(m, x_single):
        L = _compute_L_centered_single(m, x_single)
        return _stable_from_L(L)

    def _as_single_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            assert sigma.shape[0] == flat_size
            return sigma.reshape(K_, n_spin)
        elif sigma.ndim == 2:
            assert sigma.shape == (K_, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to single walker")

    def _as_batch_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 2:
            assert sigma.shape[-1] == flat_size
            return sigma.reshape(-1, K_, n_spin)
        elif sigma.ndim == 3:
            assert sigma.shape[1:] == (K_, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to batch walkers")

    @jax.jit
    def total_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            log_det_centered, _, _ = _one_from_matrix(m, x_single)
            return log_det_centered
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                log_det_centered, _, _ = _one_from_matrix(m, x_single)
                return log_det_centered
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[0])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[0])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_matrix_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single)[1]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[1])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[1])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_max_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single)[2]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single)[2]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[2])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[2])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    return total_machine, total_matrix_machine, total_max_machine, graphdef, state


# ====================== 行规范坐标函数（监控原始 L 的漂移，与列缩放的补偿量对照） ======================
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
    def renorm_offset_fn(params, x_batch):
        """P5 每步实际减去的列均值总和：offset = mean_batch Σ_j col_mean_j(walker), 复数"""
        m = nnx.merge(graphdef, params)

        def _offset(x_single):
            cols = []
            for j in range(K_):
                ansatz_j = m.single_ansatz_list[j]
                cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
            L = jnp.stack(cols, axis=1)
            return jnp.sum(jnp.mean(L, axis=0))

        return jnp.mean(jax.vmap(_offset)(x_batch))

    return gauge_fn, renorm_offset_fn


# ====================== 日志配置 ======================
logger = logging.getLogger("NES_VMC_K4_sgrecenter")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

os.makedirs("./日志", exist_ok=True)
log_path = "./日志/nes_vmc_0829_K4_LiH_STO-3G_min30_sgrecenter.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)

# ====================== 超参配置 ======================
N_CHAINS = 16
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER = 30
Natural_Grad = True
clip_norm = 20.0
lr = 0.1
qgt_diag_shift = 0.1

# ====================== 模型与采样器 ======================
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE,
    n_states=K,
    hidden_dim=SINGLE_SIZE + K,
    rngs=nnx.Rngs(11),
)

(
    total_machine,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_params,
) = create_sgrecenter_total_machines(total_ansatz, Hatree_Fock)

single_machine_list = [
    create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
    for ansatz in total_ansatz.single_ansatz_list
]

grad_fn = make_grad_fn(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
)
qgt_fn = make_qgt_fn(total_machine)

gauge_fn, renorm_offset_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
)

eigvals, _ = eigsh(ha.to_sparse(), k=K, which="SA", tol=1e-10)

# ====================== 优化器 ======================
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)

sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ====================== 训练循环 ======================
logger.info("\n" + "=" * 60)
logger.info("开始多链 NES-VMC 训练 | P5: per-walker 按列 sg 均值归一化")
logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
    f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}")

start_time = time.time()
for step in range(N_ITER):
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)

    grad_raw_flat, unravel_fn = jax.flatten_util.ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
    grad_update = grad_raw

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    eig_vals, _ = jnp.linalg.eig(E_L_mean)
    eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

    log_Psi_batch = total_machine(total_params, x_batch)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single)[0]
    psi_cond = jnp.linalg.cond(psi_mat)

    gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))
    renorm_offset = float(jnp.real(renorm_offset_fn(total_params, x_batch)))

    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"行规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | P5 列均值补偿 offset = {renorm_offset:+.4f}")
    logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
    logger.info(
        f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
        f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
    )
    logger.info("#-----------------------------------------#")

end_time = time.time()
logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
logger.info("\n" + "=" * 60)
logger.info("训练完成!")
logger.info("=" * 60)