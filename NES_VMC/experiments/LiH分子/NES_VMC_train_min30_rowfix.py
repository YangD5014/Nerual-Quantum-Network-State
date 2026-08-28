"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本（30 轮）——行规范投影版（row-gauge-projected）

核心修改（与 NES_VMC_train_min30.py 的唯一区别）：
    只替换 采样/梯度 路径上的 total_machine 为行规范投影版本：

        L = 列规范矩阵 (K,K)                     # 与原版相同
        row_mean[i] = (1/K) Σ_j L[i,j]           # 带梯度
        L_rowfixed  = L - row_mean               # 梯度自动投影掉每行加常数的规范方向
        shift       = stop_gradient(max Re(L_rowfixed))
        log_Psi_gauge = slogdet(exp(L_rowfixed - shift)) + K*shift
                        + stop_gradient(Σ row_mean)      # 值补偿，不给梯度

    - 值（实部 + 虚部模 2πi）与原 log_det_centered 完全一致
        => 采样分布（|Ψ|²）、局域能量、Loss 均不变（A/B 干净对照）
    - 梯度对行规范方向严格为零
        => MC 噪声无法沿行规范累积，logΨ mean 不再飘移

    loss 路径（total_matrix_machine / total_max_machine / single_machine_list）
    完全不改，E_L = Ψ⁻¹HΨ 保持精确（避免 GaugeFixing01 的 bug b）。

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min30_rowfix.py
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
from jax.flatten_util import ravel_pytree

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_gauge_fixed_total_machines,
    create_single_machine_gauge_fixed,
    make_grad_fn,
    make_qgt_fn,
    NESFermionHopRule,
)


def create_row_gauge_projected_total_machine(total_model, ref_state):
    """
    行规范投影的 total_machine。

    返回:
        total_machine_rowfix(params, sigma) -> log_Psi_gauge (batch,)

    与原版 create_gauge_fixed_total_machines 的 total_machine 相比：
        值一致（模 2πi），梯度去掉"每行加常数"的规范方向。
    """
    graphdef, state = nnx.split(total_model)

    K = total_model.K
    n_spin = total_model.n_spin
    flat_size = K * n_spin

    ref_state = jnp.asarray(ref_state)

    def _compute_L_centered_single(m, x_single):
        """
        x_single: shape (K, n_spin)
        返回 L[i,j] = logψ_j(x_i) - logψ_j(ref)  （列规范，带梯度）
        """
        cols = []
        for j in range(K):
            ansatz_j = m.single_ansatz_list[j]
            log_col = ansatz_j(x_single)
            log_ref = ansatz_j(ref_state)
            cols.append(log_col - log_ref)
        return jnp.stack(cols, axis=1)  # (K, K)

    def _one_rowfixed_logpsi(m, x_single):
        """
        单个 matrix walker 的行规范投影 logΨ。
        数学：det(exp(L_rowfixed - shift)) 的梯度无行方向；
             值补偿 stop_gradient(Σ row_mean) 使实部与 tr(L) 完全一致。
        """
        L = _compute_L_centered_single(m, x_single)   # (K, K) 列规范
        row_mean = jnp.mean(L, axis=1, keepdims=True)  # (K, 1) 带梯度
        L_rowfixed = L - row_mean                      # 行规范方向梯度严格为零
        shift = jnp.max(jnp.real(L_rowfixed))
        shift = jax.lax.stop_gradient(shift)
        Psi_stable = jnp.exp(L_rowfixed - shift)
        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_det_stable = log_abs_det + 1j * jnp.angle(sign)
        # 值补偿（stop_gradient）：log|det(exp(L_rowfixed))| + Re(Σ row_mean) = log|det(exp(L))|
        correction = jax.lax.stop_gradient(jnp.sum(row_mean[:, 0]))
        return log_det_stable + K * shift + correction

    def _as_single_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            assert sigma.shape[0] == flat_size
            return sigma.reshape(K, n_spin)
        elif sigma.ndim == 2:
            assert sigma.shape == (K, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to single walker")

    def _as_batch_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 2:
            assert sigma.shape[-1] == flat_size
            return sigma.reshape(-1, K, n_spin)
        elif sigma.ndim == 3:
            assert sigma.shape[1:] == (K, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to batch walkers")

    @jax.jit
    def total_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)

        if sigma.ndim == 1:
            return _one_rowfixed_logpsi(m, _as_single_matrix(sigma))
        elif sigma.ndim == 2:
            if sigma.shape == (K, n_spin):
                return _one_rowfixed_logpsi(m, _as_single_matrix(sigma))
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_rowfixed_logpsi(m, x))(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_rowfixed_logpsi(m, x))(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    return total_machine, graphdef, state


# ====================== 日志配置 ======================
logger = logging.getLogger("NES_VMC_K4_rowfix")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(message)s", datefmt="%H:%M:%S")

os.makedirs("./日志", exist_ok=True)
log_path = "./日志/nes_vmc_0829_K4_LiH_STO-3G_min30_rowfix.log"
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

# 原版 wrapper：loss 路径（matrix/max machine）保持不变
(
    total_machine_orig,
    total_matrix_machine,
    total_max_machine,
    total_graphdef,
    total_params,
) = create_gauge_fixed_total_machines(total_ansatz, Hatree_Fock)

# 行规范投影版：只替换采样/梯度用的 total_machine
total_machine, _, _ = create_row_gauge_projected_total_machine(
    total_ansatz, Hatree_Fock
)

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
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ====================== 训练循环 ======================
logger.info("\n" + "=" * 60)
logger.info("开始多链 NES-VMC 训练 | 行规范投影版（row-gauge-projected）")
logger.info("=" * 60)
logger.info(
    f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
    f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
)
logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}")

start_time = time.time()
for step in range(N_ITER):
    # 1. 采样（用行规范投影 machine，值与原版一致 => 采样分布不变）
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # 2. 梯度与损失
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)

    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
    grad_update = grad_raw

    has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
    if has_nan or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

    # 3. QGT 自然梯度
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)
        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_norm_natural = grad_norm_raw

    # 4. 优化器内部完成梯度裁剪 + SGD 更新
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

    # 5. 能量监控
    eig_vals, _ = jnp.linalg.eig(E_L_mean)
    eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

    # 6. 波函数与Ψ矩阵条件数监控
    log_Psi_batch = total_machine(total_params, x_batch)
    x_single = x_batch[0:1, ...]
    psi_mat = total_matrix_machine(total_params, x_single)[0]
    psi_cond = jnp.linalg.cond(psi_mat)

    # 7. 日志
    logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
    logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
    logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
    logger.info(
        f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
        f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
    )
    logger.info("#-----------------------------------------#")

    # 8. 第 0 步：行规范投影 machine 与原版 machine 的值一致性校验
    if step == 0:
        logPsi_orig = total_machine_orig(total_params, x_batch)
        re_diff = jnp.max(jnp.abs(logPsi_orig.real - log_Psi_batch.real))
        im_diff_raw = jnp.max(jnp.abs(logPsi_orig.imag - log_Psi_batch.imag))
        im_diff_wrapped = jnp.minimum(im_diff_raw, 2 * jnp.pi - im_diff_raw)
        logger.info(
            f"[值一致性校验@Step0] max|ΔRe|={re_diff:.3e} | "
            f"max|ΔIm|={im_diff_raw:.3e}（模2π={im_diff_wrapped:.3e}）"
        )

end_time = time.time()
logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
logger.info("\n" + "=" * 60)
logger.info("训练完成!")
logger.info("=" * 60)