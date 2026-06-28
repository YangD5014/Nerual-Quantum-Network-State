
import os
import time
import logging
import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import optax

from jax.flatten_util import ravel_pytree
from pyscf import gto, scf, fci


# ============================================================
# 0. 工具函数
# ============================================================

def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return jnp.array(0.0)
    return jnp.sqrt(
        sum([jnp.sum(jnp.abs(x) ** 2) for x in leaves])
    )


def tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return True
    flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return bool(jnp.all(jnp.asarray(flags)))


def unique_ratio_from_samples(samples):
    """
    samples: shape (n_samples, hi_ext.size)
    """
    arr = np.asarray(samples)
    arr = arr.reshape(arr.shape[0], -1)
    unique = len({tuple(row.tolist()) for row in arr})
    return unique / max(arr.shape[0], 1)


def tree_batch_std_norm(tree, batch_size):
    """
    计算 dlogΨ(params, x) 在 batch 维度上的变化强度。
    如果这个接近 0，说明不同样本上的参数响应几乎一样，
    covariance 梯度会自然消失。
    """
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0

    for leaf in leaves:
        if leaf.ndim >= 1 and leaf.shape[0] == batch_size:
            centered = leaf - jnp.mean(leaf, axis=0, keepdims=True)
            total = total + jnp.sum(jnp.abs(centered) ** 2)

    return jnp.sqrt(total)


def tree_batch_mean_norm(tree, batch_size):
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0

    for leaf in leaves:
        if leaf.ndim >= 1 and leaf.shape[0] == batch_size:
            mean_leaf = jnp.mean(leaf, axis=0)
            total = total + jnp.sum(jnp.abs(mean_leaf) ** 2)

    return jnp.sqrt(total)


def safe_real(x):
    return float(jnp.real(x))


def safe_float(x):
    return float(jnp.asarray(x))


# ============================================================
# 8. 诊断函数
# ============================================================

def compute_diagnostics(
    total_params,
    x_batch,
    samples,
    E_L_mean,
):
    """
    返回当前参数下的完整诊断量。
    注意：这里会额外调用 NES_loss_energy_stable 来得到 E_L_batch。
    """

    n_total = x_batch.shape[0]
    n_diag = min(DIAG_BATCH_SIZE, n_total)

    x_diag = x_batch[:n_diag]
    samples_diag = samples[:n_diag]

    # ---------- logΨ 统计 ----------
    log_Psi_batch = jax.vmap(
        lambda xx: total_machine(total_params, xx)
    )(x_batch)

    log_real = jnp.real(log_Psi_batch)
    log_imag = jnp.imag(log_Psi_batch)

    log_real_mean = jnp.mean(log_real)
    log_real_std = jnp.std(log_real)
    log_real_span = jnp.max(log_real) - jnp.min(log_real)

    log_imag_mean = jnp.mean(log_imag)
    log_imag_std = jnp.std(log_imag)
    log_imag_span = jnp.max(log_imag) - jnp.min(log_imag)

    # ---------- L_stable / Ψ_stable 条件数 ----------
    L_stable_diag = total_matrix_machine(total_params, x_diag)
    Psi_stable_diag = jnp.exp(L_stable_diag)

    conds = jax.vmap(jnp.linalg.cond)(Psi_stable_diag)
    psi_cond_first = conds[0]
    psi_cond_mean = jnp.mean(conds)
    psi_cond_max = jnp.max(conds)

    # ---------- shift 统计 ----------
    shifts = total_max_machine(total_params, x_diag)
    shift_mean = jnp.mean(shifts)
    shift_std = jnp.std(shifts)
    shift_span = jnp.max(shifts) - jnp.min(shifts)

    # ---------- E_L_batch / trace 统计 ----------
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_diag,
        return_aux=True,
    )

    assert E_L_batch.shape[-2:] == (K, K), (
        f"E_L_batch.shape={E_L_batch.shape}, expected (..., {K}, {K})"
    )

    trace_batch = jnp.trace(E_L_batch, axis1=-2, axis2=-1)
    trace_real = jnp.real(trace_batch)
    trace_imag = jnp.imag(trace_batch)

    trace_real_mean = jnp.mean(trace_real)
    trace_real_std = jnp.std(trace_real)
    trace_real_span = jnp.max(trace_real) - jnp.min(trace_real)

    trace_imag_mean = jnp.mean(trace_imag)
    trace_imag_std = jnp.std(trace_imag)
    trace_imag_span = jnp.max(trace_imag) - jnp.min(trace_imag)

    # ---------- valid ratio ----------
    if isinstance(loss_aux, dict) and "valid" in loss_aux:
        valid = loss_aux["valid"]
        valid_ratio = jnp.mean(valid.astype(jnp.float64))
    else:
        valid_ratio = jnp.array(1.0)

    # ---------- E_L_mean Hermiticity ----------
    herm_error = (
        jnp.linalg.norm(E_L_mean - E_L_mean.conj().T)
        / (jnp.linalg.norm(E_L_mean) + 1e-12)
    )

    imag_norm = jnp.linalg.norm(jnp.imag(E_L_mean))

    E_L_herm = 0.5 * (E_L_mean + E_L_mean.conj().T)
    eig_vals_herm = jnp.linalg.eigvalsh(E_L_herm)

    eig_vals_raw = jnp.linalg.eigvals(E_L_mean)
    eig_vals_raw = eig_vals_raw[jnp.argsort(jnp.real(eig_vals_raw))]

    # ---------- sampler unique ratio ----------
    unique_ratio = unique_ratio_from_samples(samples_diag)

    # ---------- dlogΨ batch 方差 ----------
    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    dlogPsi_batch = jax.vmap(
        grad_logPsi,
        in_axes=(None, 0),
    )(total_params, x_diag)

    dlog_std_norm = tree_batch_std_norm(dlogPsi_batch, n_diag)
    dlog_mean_norm = tree_batch_mean_norm(dlogPsi_batch, n_diag)
    dlog_std_ratio = dlog_std_norm / (dlog_mean_norm + 1e-12)

    return {
        "log_Psi_batch": log_Psi_batch,

        "log_real_mean": log_real_mean,
        "log_real_std": log_real_std,
        "log_real_span": log_real_span,

        "log_imag_mean": log_imag_mean,
        "log_imag_std": log_imag_std,
        "log_imag_span": log_imag_span,

        "psi_cond_first": psi_cond_first,
        "psi_cond_mean": psi_cond_mean,
        "psi_cond_max": psi_cond_max,

        "shift_mean": shift_mean,
        "shift_std": shift_std,
        "shift_span": shift_span,

        "trace_real_mean": trace_real_mean,
        "trace_real_std": trace_real_std,
        "trace_real_span": trace_real_span,

        "trace_imag_mean": trace_imag_mean,
        "trace_imag_std": trace_imag_std,
        "trace_imag_span": trace_imag_span,

        "valid_ratio": valid_ratio,

        "herm_error": herm_error,
        "imag_norm": imag_norm,

        "eig_vals_herm": eig_vals_herm,
        "eig_vals_raw": eig_vals_raw,

        "unique_ratio": unique_ratio,

        "dlog_std_norm": dlog_std_norm,
        "dlog_mean_norm": dlog_mean_norm,
        "dlog_std_ratio": dlog_std_ratio,
    }


def log_diagnostics(step, loss_mean, grad_norm_raw, grad_norm_update, grad_norm_clipped, diag):
    eig_vals_herm = diag["eig_vals_herm"]
    eig_vals_raw = diag["eig_vals_raw"]

    energy_herm_str = " | ".join(
        [f"E{i}_herm={eig_vals_herm[i]:.8f}" for i in range(K)]
    )

    energy_raw_str = " | ".join(
        [f"E{i}_raw={eig_vals_raw[i]:.8f}" for i in range(K)]
    )

    logger.info(f"[Step {step:4d}]")
    logger.info(
        f"Loss={loss_mean:.8f} | "
        f"target={target_loss:.8f} | "
        f"gap={float(loss_mean - target_loss):+.8f}"
    )

    logger.info(
        f"Grad | raw={grad_norm_raw:.4e} | "
        f"update={grad_norm_update:.4e} | "
        f"clipped={grad_norm_clipped:.4e} | "
        f"clip_norm={clip_norm:.2e}"
    )

    logger.info(
        "logΨ.real | "
        f"mean={diag['log_real_mean']:.6f} | "
        f"std={diag['log_real_std']:.4e} | "
        f"span={diag['log_real_span']:.4e}"
    )

    logger.info(
        "logΨ.imag | "
        f"mean={diag['log_imag_mean']:.6f} | "
        f"std={diag['log_imag_std']:.4e} | "
        f"span={diag['log_imag_span']:.4e}"
    )

    logger.info(
        "shift | "
        f"mean={diag['shift_mean']:.6f} | "
        f"std={diag['shift_std']:.4e} | "
        f"span={diag['shift_span']:.4e}"
    )

    logger.info(
        "cond(Ψ_stable) | "
        f"first={diag['psi_cond_first']:.4e} | "
        f"mean={diag['psi_cond_mean']:.4e} | "
        f"max={diag['psi_cond_max']:.4e}"
    )

    logger.info(
        "trace(E_L).real | "
        f"mean={diag['trace_real_mean']:.8f} | "
        f"std={diag['trace_real_std']:.4e} | "
        f"span={diag['trace_real_span']:.4e}"
    )

    logger.info(
        "trace(E_L).imag | "
        f"mean={diag['trace_imag_mean']:.8f} | "
        f"std={diag['trace_imag_std']:.4e} | "
        f"span={diag['trace_imag_span']:.4e}"
    )

    logger.info(
        "E_L_mean diagnostics | "
        f"herm_error={diag['herm_error']:.4e} | "
        f"imag_norm={diag['imag_norm']:.4e} | "
        f"valid_ratio={diag['valid_ratio']:.4f}"
    )

    logger.info(
        "sampler / dlogΨ | "
        f"unique_ratio={diag['unique_ratio']:.4f} | "
        f"dlog_std_norm={diag['dlog_std_norm']:.4e} | "
        f"dlog_mean_norm={diag['dlog_mean_norm']:.4e} | "
        f"dlog_std_ratio={diag['dlog_std_ratio']:.4e}"
    )

    logger.info(energy_herm_str)
    logger.info(energy_raw_str)

    # 明确报警，别让模型安静地死
    if float(diag["log_real_span"]) < 1e-6:
        logger.warning(">>> WARNING: logΨ.real span ≈ 0，疑似 amplitude collapse / gauge plateau")

    if float(diag["trace_real_std"]) < 1e-8:
        logger.warning(">>> WARNING: trace(E_L).real std 很小，covariance 梯度能量项可能无信号")

    if float(diag["dlog_std_ratio"]) < 1e-6:
        logger.warning(">>> WARNING: dlogΨ batch variation 很小，参数响应接近常数方向")

    if float(diag["herm_error"]) > 1.0:
        logger.warning(">>> WARNING: E_L_mean 非 Hermitian 程度很大，能量诊断不可信")

    logger.info("#" + "-" * 79)



