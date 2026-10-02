# -*- coding: utf-8 -*-
"""
Frozen-State NES-VMC（Route B：冻结流形）— H2 / 6-31G / K=4 / 实参数 FFNN
==============================================================================
方案依据：NES_VMC/文档/NES-VMC冻结策略/1001-冻结方案.md

与 0923 矩阵自由 deflation（Q0HQ0 显式投影）不同，Route B 的做法是：
  1. Stage 0：标准 NES-VMC（K=4 列）训练，广义本征提取收敛物理态 φ0 = Σ_j v0[j]·ψ_j；
  2. 冻结：φ0 作为 stop-gradient 常量列进入新流形 B = [φ0, ψ1', ψ2', ψ3']；
  3. Stage 1：采样分布 |det[φ0, ψ1', ψ2', ψ3']|²（行列式自动消除 active 列中与 φ0
     线性相关的冗余方向，即 implicit projection），只训练 3 个 fresh active 列；
  4. 对照组（control）：不冻结，4 列全部继续训练相同步数；
  5. 诊断：利用 H2/6-31G 物理空间仅 16 维，用全空间稠密 H 做「精确极限」对照
     （φ0 保真度漂移、精确能级、与 FCI 本征态重叠）。

Ansatz / 梯度 / QGT / 采样均复用 H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.ipynb 的设计
（实参数 FFNN、振幅-相位分离、Wirtinger 拆分梯度、Re(S) 度量、双侧一致列规范）。
"""

import argparse
import logging
import os
import time

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import numpy as np
import optax
from jax.flatten_util import ravel_pytree
from scipy.linalg import eigh as scipy_eigh

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    Ham_psi_scaled,
    NESFermionHopRule,
    flatten_batched_pytree,
)
from NES_VMC_tool import (
    create_gauge_reset_total_machines,
    NES_loss_energy_stable_gauge,
    _masked_mean_batch,
    make_gauge_fn,
    compute_lam_v_from_samples,
)
from H2_6_31G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, E_fcis, Hatree_Fock

jax.config.update("jax_enable_x64", True)

N_REPLICA = K          # NES walker 副本数（冻结前后都保持 K=4，行列式 4×4）
N_ACTIVE = K - 1       # Route B：1 列冻结 + 3 列 active

# ======================================================================
# 1. Ansatz（与 RealFFNN notebook 完全一致）
# ======================================================================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_logA = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_phi = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        logA = jnp.squeeze(self.head_logA(h))
        phi = jnp.squeeze(self.head_phi(h))
        return logA + 1j * phi


class NESTotalAnsatz_stable(nnx.Module):
    """NES 总拟设：K 个「实参数 + 复输出」单态拟设组成的行列式波函数。"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            )


# ======================================================================
# 2. 实参数梯度 / QGT 兼容层（与 RealFFNN notebook 一致，供 Stage 0 / control 用）
# ======================================================================
def _grad_logpsi_real_single(total_machine, params, x_single, g):
    re_grad = jax.grad(lambda p: jnp.real(total_machine(p, x_single, g)))(params)
    im_grad = jax.grad(lambda p: jnp.imag(total_machine(p, x_single, g)))(params)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def nes_vmc_gradient_stable_gauge_real(
    ha, total_matrix_machine, total_max_machine, total_machine,
    single_machine_list, total_params, x_batch, g,
    min_valid_ratio: float = 0.25, return_aux: bool = False,
):
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable_gauge(
        ha=ha, total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine, single_machine_list=single_machine_list,
        total_params=total_params, x=x_batch, g=g, return_aux=True,
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

    dlogPsi_batch = jax.vmap(lambda x: _grad_logpsi_real_single(total_machine, total_params, x, g))(x_batch)

    def weight_and_masked_mean(grad_component):
        grad_safe = jnp.where(jnp.isfinite(grad_component), grad_component, 0.0)
        weights = tr_centered.reshape((-1,) + (1,) * (grad_component.ndim - 1))
        valid_f = valid.astype(jnp.float32).reshape((-1,) + (1,) * (grad_component.ndim - 1))
        weighted = weights * jnp.conj(grad_safe) * valid_f
        denom = jnp.maximum(n_valid, 1.0)
        return 2.0 * jnp.real(jnp.sum(weighted, axis=0) / denom)

    grad = jax.tree.map(weight_and_masked_mean, dlogPsi_batch)
    loss_mean = _masked_mean_batch(loss_batch, valid)

    grad_flat, _ = ravel_pytree(grad)
    aux = {
        "valid": valid, "n_valid": n_valid, "valid_ratio": valid_ratio,
        "grad_finite": jnp.all(jnp.isfinite(grad_flat)),
        "loss_finite": jnp.isfinite(loss_mean),
        "enough_valid": valid_ratio >= min_valid_ratio,
        "should_skip": ~(jnp.all(jnp.isfinite(grad_flat)) & jnp.isfinite(loss_mean) & (valid_ratio >= min_valid_ratio)),
        "loss_aux": loss_aux,
    }
    if return_aux:
        return grad, loss_mean, E_L_mean, aux
    return grad, loss_mean, E_L_mean


def make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list):
    @jax.jit
    def grad_fn(total_params, x_batch, g):
        return nes_vmc_gradient_stable_gauge_real(
            ha=ha, total_matrix_machine=total_matrix_machine,
            total_max_machine=total_max_machine, total_machine=total_machine,
            single_machine_list=single_machine_list, total_params=total_params,
            x_batch=x_batch, g=g, return_aux=True,
        )
    return grad_fn


def make_qgt_fn_gauge_real(machine):
    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g):
        n_samples = sigma.shape[0]
        grad_tree_batch = jax.vmap(lambda s: _grad_logpsi_real_single(machine, params, s, g))(sigma)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        I = jnp.eye(S_real.shape[0], dtype=S_real.dtype)
        return S_real + diag_shift * I
    return qgt_fn


# ======================================================================
# 3. Route B 核心：冻结流形 forward / 梯度 / QGT / MS 估计
# ======================================================================
def make_frozen_bundle(frozen_sms, frozen_params, v0):
    """把 Stage-0 收敛列 + 基态旋转系数打包成冻结列常量。

    返回 frozen_bundle：
        v0        (K,) complex128 —— φ0 = Σ_j v0[j]·ψ_j（gauge-fixed 列）
        sms       K 个 gauge-fixed 单列 machine（常量参数）
        params    对应 K 份冻结参数（Python 侧常量，不被任何 grad 追踪）
    """
    v0 = jnp.asarray(v0, dtype=jnp.complex128)
    return {"v0": v0, "sms": frozen_sms, "params": frozen_params}


def make_frozen_forward(ha, bundle, active_sms):
    """构造冻结流形 B=[φ0, ψ1..ψ_{K-1}] 的前向。

    _one(p, xw, g) 对单个 NES walker xw (K, n_spin)：
        Lf[n,j] = logψ_j^frozen(x_n)          （gauge-fixed）
        La[n,m] = logψ_m^active(x_n) - g[m]   （gauge-fixed + 列规范）
        s       = max Re(Lf ∪ La)             （per-walker 稳定化 shift）
        φ0 列   = Σ_j v0[j]·exp(Lf[:,j]-s)     （H 线性 ⇒ Hφ0 = Σ v0[j]·Hψ_j）
        A = [φ0_scaled, exp(La-s)]            （K×K）
        Z = [Hφ0_scaled, HPa·e^{-g}]          （K×K）
        logΨ = K·s + slogdet(A)；E_L = solve(A, Z)
    φ0 / v0 / 冻结参数全部是闭包常量 ⇒ 梯度天然只流向 active 参数。
    """
    v0 = bundle["v0"]
    f_sms = bundle["sms"]
    f_params = bundle["params"]

    def _one(p, xw, g):
        Lf = jnp.stack([f_sms[j](f_params[j], xw) for j in range(N_REPLICA)], axis=1)   # (K,K)
        La = jnp.stack([active_sms[j](p[j], xw) for j in range(N_ACTIVE)], axis=1)       # (K,K-1)
        La = La - jax.lax.stop_gradient(g)[None, :]

        s = jax.lax.stop_gradient(jnp.max(jnp.real(jnp.concatenate([Lf, La], axis=1))))
        phi_scaled = jnp.exp(Lf - s) @ v0            # (K,) = e^{-s}·φ0(x_n)
        A = jnp.concatenate([phi_scaled[:, None], jnp.exp(La - s)], axis=1)  # (K,K)

        HPf = jnp.stack(
            [Ham_psi_scaled(ha=ha, single_machine=f_sms[j], params=f_params[j], x=xw, shift=s)
             for j in range(N_REPLICA)], axis=1)     # (K,K)
        Hphi0 = HPf @ v0                             # (K,) = e^{-s}·(Hφ0)(x_n)
        HPa = jnp.stack(
            [Ham_psi_scaled(ha=ha, single_machine=active_sms[j], params=p[j], x=xw, shift=s)
             for j in range(N_ACTIVE)], axis=1)     # (K,K-1)
        HPa = HPa * jnp.exp(-jax.lax.stop_gradient(g))[None, :]
        Z = jnp.concatenate([Hphi0[:, None], HPa], axis=1)  # (K,K)

        sign, log_abs = jnp.linalg.slogdet(A)
        logPsi = N_REPLICA * s + log_abs + 1j * jnp.angle(sign)
        E_L = jnp.linalg.solve(A, Z)
        loss = jnp.real(jnp.trace(E_L))
        valid = (jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(Z))
                 & jnp.all(jnp.isfinite(E_L)) & jnp.isfinite(loss))
        return logPsi, E_L, loss, valid

    def core(p, x_batch, g):
        logPsi, E_L, loss, valid = jax.vmap(lambda xw: _one(p, xw, g))(x_batch)
        return logPsi, E_L, loss, valid

    return core, _one


def make_frozen_grad_fn(ha, core, _one):
    """冻结流形梯度：loss=Re tr(E_L)，对 active 参数 Wirtinger 拆分 + 取实部×2。"""

    def dlogpsi_one(p, xw, g):
        re_grad = jax.grad(lambda q: jnp.real(_one(q, xw, g)[0]))(p)
        im_grad = jax.grad(lambda q: jnp.imag(_one(q, xw, g)[0]))(p)
        return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)

    @jax.jit
    def grad_fn(active_params, x_batch, g, min_valid_ratio=0.25):
        logPsi, E_L, loss, valid = core(active_params, x_batch, g)
        valid = jax.lax.stop_gradient(valid)
        n_valid = jnp.sum(valid.astype(jnp.float32))
        valid_ratio = n_valid / x_batch.shape[0]

        E_L_mean = _masked_mean_batch(E_L, valid)
        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)

        dlog_batch = jax.vmap(lambda xw: dlogpsi_one(active_params, xw, g))(x_batch)

        def weighted_mean(comp):
            comp_safe = jnp.where(jnp.isfinite(comp), comp, 0.0)
            w = tr_centered.reshape((-1,) + (1,) * (comp.ndim - 1))
            vf = valid.astype(jnp.float32).reshape((-1,) + (1,) * (comp.ndim - 1))
            return 2.0 * jnp.real(jnp.sum(w * jnp.conj(comp_safe) * vf, axis=0) / jnp.maximum(n_valid, 1.0))

        grad = jax.tree.map(weighted_mean, dlog_batch)
        loss_mean = _masked_mean_batch(loss, valid)
        aux = {"valid": valid, "valid_ratio": valid_ratio,
               "should_skip": ~(jnp.all(jnp.isfinite(ravel_pytree(grad)[0]))
                                & jnp.isfinite(loss_mean) & (valid_ratio >= min_valid_ratio))}
        return grad, loss_mean, E_L_mean, aux

    return grad_fn


def make_frozen_qgt_fn(core, _one):
    def dlogpsi_one(p, xw, g):
        re_grad = jax.grad(lambda q: jnp.real(_one(q, xw, g)[0]))(p)
        im_grad = jax.grad(lambda q: jnp.imag(_one(q, xw, g)[0]))(p)
        return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)

    @jax.jit
    def qgt_fn(active_params, sigma, diag_shift, g):
        n_samples = sigma.shape[0]
        grad_tree_batch = jax.vmap(lambda xw: dlogpsi_one(active_params, xw, g))(sigma)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)
        O_centered = O - jnp.mean(O, axis=0, keepdims=True)
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        return S_real + diag_shift * jnp.eye(S_real.shape[0], dtype=S_real.dtype)

    return qgt_fn


def make_frozen_MS_fn(ha, bundle, active_sms):
    """在冻结流形样本上估计 4×4 的 M_B、S_B（列 0 = φ0）。"""
    v0, f_sms, f_params = bundle["v0"], bundle["sms"], bundle["params"]

    def ms_fn(active_params, x_batch):
        Lf = jnp.stack([f_sms[j](f_params[j], x_batch.reshape(-1, SINGLE_SIZE)).reshape(x_batch.shape[0], N_REPLICA)
                        for j in range(N_REPLICA)], axis=-1)
        La = jnp.stack([active_sms[j](active_params[j], x_batch.reshape(-1, SINGLE_SIZE)).reshape(x_batch.shape[0], N_REPLICA)
                        for j in range(N_ACTIVE)], axis=-1)
        s = jnp.max(jnp.real(jnp.concatenate([Lf, La], axis=-1)), axis=(1, 2))
        phi_scaled = jnp.exp(Lf - s[:, None, None]) @ v0
        A = jnp.concatenate([phi_scaled[:, :, None], jnp.exp(La - s[:, None, None])], axis=-1)

        sb = jnp.broadcast_to(s[:, None], (x_batch.shape[0], N_REPLICA)).reshape(-1)
        xf = x_batch.reshape(-1, SINGLE_SIZE)
        HPf = jnp.stack([Ham_psi_scaled(ha=ha, single_machine=f_sms[j], params=f_params[j], x=xf, shift=sb)
                         .reshape(x_batch.shape[0], N_REPLICA) for j in range(N_REPLICA)], axis=-1)
        Hphi0 = HPf @ v0
        HPa = jnp.stack([Ham_psi_scaled(ha=ha, single_machine=active_sms[j], params=active_params[j], x=xf, shift=sb)
                         .reshape(x_batch.shape[0], N_REPLICA) for j in range(N_ACTIVE)], axis=-1)
        Z = jnp.concatenate([Hphi0[:, :, None], HPa], axis=-1)

        finite = jnp.all(jnp.isfinite(A), axis=(-2, -1)) & jnp.all(jnp.isfinite(Z), axis=(-2, -1))
        An = jnp.where(finite[:, None, None], A, 0.0)
        Zn = jnp.where(finite[:, None, None], Z, 0.0)
        norm = jnp.sqrt(jnp.sum(jnp.abs(An) ** 2, axis=(1, 2)) + 1e-30)[:, None, None]
        An, Zn = An / norm, Zn / norm
        N_eff = jnp.maximum(jnp.sum(finite.astype(jnp.float32)), 1.0)
        S = jnp.einsum("nai,naj->ij", jnp.conj(An), An).astype(jnp.complex128) / N_eff
        M = jnp.einsum("nai,naj->ij", jnp.conj(An), Zn).astype(jnp.complex128) / N_eff
        return 0.5 * (M + jnp.conj(M.T)), 0.5 * (S + jnp.conj(S.T)), finite

    return jax.jit(ms_fn)


# ======================================================================
# 4. 全空间（16 维）精确诊断 —— 小系统「精确极限」
# ======================================================================
def build_exact_reference():
    states = np.asarray(hi.all_states())
    H = np.asarray(ha.to_dense(), dtype=np.complex128)
    evals, evecs = np.linalg.eigh(H)
    return states, H, evals, evecs


def gauge_fixed_basis_vectors(machines, params_list, states):
    """多列 gauge-fixed 基（ψ_j(ref)=1）在全部 16 组态上的稠密矩阵 (16,K)。

    关键：所有列共享同一个 shift（跨列跨组态取 max Re），保证列间相对尺度
    与训练时的 ψ_j 一致（φ0=Σ v0[j]ψ_j 才有正确含义）。
    """
    logs = np.stack(
        [np.asarray(machines[j](params_list[j], jnp.asarray(states, dtype=jnp.float64)), dtype=np.complex128)
         for j in range(len(machines))], axis=1)          # (16, K)
    shift = logs.real.max()
    return np.exp(logs - shift)


def gauge_fixed_column_vectors(machine, params, states):
    """单列 gauge-fixed 稠密向量（仅用于单列场景；多列请用共享 shift 版本）。"""
    logs = np.asarray(machine(params, jnp.asarray(states, dtype=jnp.float64)), dtype=np.complex128)
    shift = logs.real.max()
    return np.exp(logs - shift)


def exact_subspace_analysis(Bvecs, H, evecs, labels=None):
    """对基 Bvecs (16,K) 解精确广义本征问题，返回能级 / 提取态 / 与 FCI 态重叠。"""
    M = Bvecs.conj().T @ H @ Bvecs
    S = Bvecs.conj().T @ Bvecs
    S = 0.5 * (S + S.conj().T)
    # 近奇异基（如随机初始化列近似平行）时加微小 ridge 防 scipy 崩溃
    ridge = 1e-10 * max(np.trace(S).real / S.shape[0], 1.0)
    try:
        lam, V = scipy_eigh(M, S + ridge * np.eye(S.shape[0]))
    except np.linalg.LinAlgError:
        lam, V = scipy_eigh(M, S + 1e-6 * np.eye(S.shape[0]))
    order = np.argsort(lam.real)
    lam, V = lam[order].real, V[:, order]
    out = {"lam": lam, "V": V, "states": []}
    for k in range(Bvecs.shape[1]):
        phi = Bvecs @ V[:, k]
        nrm = np.vdot(phi, phi).real
        ov = [abs(np.vdot(evecs[:, j], phi)) ** 2 / nrm for j in range(min(6, evecs.shape[1]))]
        out["states"].append({"phi": phi, "overlap_exact": ov, "norm": nrm})
    return out


def fidelity(a, b):
    return float(abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real))


def exact_frozen_block_analysis(phi0_vec, Ba, H, evecs):
    """Route B block 分析（不做完整 4×4 对角化，避免文档 §5 的 φ0 旋转）。

    - 冻结态：E_φ0 = ⟨φ0|H|φ0⟩/⟨φ0|φ0⟩（Rayleigh 商），对 FCI 基态保真度。
    - active 态：投影基 P0ψ_j = ψ_j − c_j·φ0（c_j=⟨φ0|ψ_j⟩/⟨φ0|φ0⟩），
      在其上解 3×3 广义本征 → E1..E3 及对 FCI 各态重叠。
    """
    n0 = np.vdot(phi0_vec, phi0_vec).real
    e_phi0 = float(np.vdot(phi0_vec, H @ phi0_vec).real / n0)
    c = (phi0_vec.conj() @ Ba) / n0                      # (K-1,)
    Bp = Ba - np.outer(phi0_vec, c)                      # 投影基 (16,K-1)
    M = Bp.conj().T @ H @ Bp
    S = Bp.conj().T @ Bp
    S = 0.5 * (S + S.conj().T)
    ridge = 1e-10 * max(np.trace(S).real / S.shape[0], 1.0)
    lam, V = scipy_eigh(M, S + ridge * np.eye(S.shape[0]))
    order = np.argsort(lam.real)
    lam, V = lam[order].real, V[:, order]
    states = []
    for k in range(Bp.shape[1]):
        phi = Bp @ V[:, k]
        nn = np.vdot(phi, phi).real
        ov = [abs(np.vdot(evecs[:, j], phi)) ** 2 / nn for j in range(min(6, evecs.shape[1]))]
        states.append({"phi": phi, "overlap_exact": ov})
    return {"e_phi0": e_phi0, "phi0_fid_fci": fidelity(phi0_vec, evecs[:, 0]),
            "lam_active": lam, "states": states}


# ======================================================================
# 5. 训练循环
# ======================================================================
def make_sampler(machine_fn, params, seed):
    nes_rule = NESFermionHopRule(edges=ext_edges, K=N_REPLICA, single_size=SINGLE_SIZE)
    sampler = nk.sampler.MetropolisSampler(hilbert=hi_ext, rule=nes_rule,
                                           n_chains=N_CHAINS_G, sweep_size=SWEEP_G)
    rng = jax.random.PRNGKey(seed)
    return sampler, sampler.init_state(machine_fn, params, rng)


def run_stage0(logger, cfg, states, H, evecs):
    """标准 NES-VMC（K=4 列，双侧一致列规范 + 周期 gauge reset）。"""
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE, n_states=N_REPLICA,
        hidden_dim=SINGLE_SIZE + N_REPLICA, rngs=nnx.Rngs(11))

    g_current = [jnp.zeros(N_REPLICA, dtype=jnp.complex128)]
    (total_machine, total_matrix_machine, total_max_machine,
     total_matrix_machine_raw, total_graphdef, total_params) = \
        create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)
    single_machine_list = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
                           for a in total_ansatz.single_ansatz_list]

    grad_fn = make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine,
                                      total_machine, single_machine_list)
    qgt_fn = make_qgt_fn_gauge_real(total_machine)
    gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g_current[0])

    sampler, sstate = make_sampler(sample_machine, total_params, 21)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.clip_norm), optax.sgd(cfg.lr))
    opt_state = optimizer.init(total_params)

    hist = {"step": [], "loss": [], "E_levels": [], "eig_exact": []}
    t0 = time.time()
    for step in range(cfg.n_iter0):
        samples_raw, sstate = sampler.sample(machine=sample_machine, parameters=total_params,
                                             state=sstate, chain_length=cfg.n_samples)
        x_batch = samples_raw.reshape(-1, N_REPLICA, SINGLE_SIZE)

        grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current[0])
        grad_flat, unravel = ravel_pytree(grad_raw)
        gnorm = float(jnp.linalg.norm(grad_flat))
        if gnorm / np.sqrt(grad_flat.size) > cfg.per_param_clip:
            grad_flat = grad_flat * (cfg.per_param_clip * np.sqrt(grad_flat.size) / gnorm)
            grad_raw = unravel(grad_flat)
        qgt = qgt_fn(total_params, x_batch, cfg.diag_shift, g_current[0])
        ng = jnp.linalg.solve(qgt, ravel_pytree(grad_raw)[0])
        updates, opt_state = optimizer.update(unravel(ng), opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        eig = np.sort(np.asarray(jnp.linalg.eig(E_L_mean)[0]).real)
        hist["step"].append(step)
        hist["loss"].append(float(jnp.real(loss_mean)))
        hist["E_levels"].append(eig[:N_REPLICA])
        if (step + 1) % cfg.diag_every == 0 or step == 0:
            B = gauge_fixed_basis_vectors(single_machine_list,
                                          [total_params["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
            ana = exact_subspace_analysis(B, H, evecs)
            hist["eig_exact"].append((step, ana["lam"]))
            logger.info(f"[Stage0 {step}] loss={hist['loss'][-1]:.6f} | E_L eig={eig[:2]} | exact={ana['lam'][:2]}")

        if (step + 1) % cfg.reset_period == 0:
            g_current[0] = col_mean_fn(total_params, x_batch)

    logger.info(f"Stage0 耗时 {time.time()-t0:.1f}s")
    # 提取 φ0
    lam, v, n_valid, order = compute_lam_v_from_samples(ha, single_machine_list, total_params, x_batch)
    v0 = np.asarray(v[:, 0], dtype=np.complex128)
    B_end = gauge_fixed_basis_vectors(single_machine_list,
                                      [total_params["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
    logger.info(f"Stage0 广义本征 λ={lam} | φ0 保真(对精确基态)={fidelity(B_end @ v0, evecs[:, 0]):.6f}")
    return dict(total_ansatz=total_ansatz, total_params=total_params,
                single_machine_list=single_machine_list, hist=hist, v0=v0, lam=lam)


def run_control(logger, cfg, s0, states, H, evecs):
    """对照组：不冻结，4 列全部继续训练 n_iter1 步。"""
    total_ansatz, total_params = s0["total_ansatz"], s0["total_params"]
    (total_machine, total_matrix_machine, total_max_machine, _, _, _) = \
        create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)
    single_machine_list = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
                           for a in total_ansatz.single_ansatz_list]
    grad_fn = make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine,
                                      total_machine, single_machine_list)
    qgt_fn = make_qgt_fn_gauge_real(total_machine)
    _, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    g = [jnp.zeros(N_REPLICA, dtype=jnp.complex128)]

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g[0])

    sampler, sstate = make_sampler(sample_machine, total_params, 33)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.clip_norm), optax.sgd(cfg.lr))
    opt_state = optimizer.init(total_params)

    hist = {"step": [], "loss": [], "E_levels": [], "eig_exact": [], "phi0_fid": []}
    B0 = gauge_fixed_basis_vectors(single_machine_list,
                                   [total_params["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
    phi0_ref = B0 @ s0["v0"]
    t0 = time.time()
    for step in range(cfg.n_iter1):
        samples_raw, sstate = sampler.sample(machine=sample_machine, parameters=total_params,
                                             state=sstate, chain_length=cfg.n_samples)
        x_batch = samples_raw.reshape(-1, N_REPLICA, SINGLE_SIZE)
        grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g[0])
        grad_flat, unravel = ravel_pytree(grad_raw)
        gnorm = float(jnp.linalg.norm(grad_flat))
        if gnorm / np.sqrt(grad_flat.size) > cfg.per_param_clip:
            grad_flat = grad_flat * (cfg.per_param_clip * np.sqrt(grad_flat.size) / gnorm)
            grad_raw = unravel(grad_flat)
        qgt = qgt_fn(total_params, x_batch, cfg.diag_shift, g[0])
        ng = jnp.linalg.solve(qgt, ravel_pytree(grad_raw)[0])
        updates, opt_state = optimizer.update(unravel(ng), opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        eig = np.sort(np.asarray(jnp.linalg.eig(E_L_mean)[0]).real)
        hist["step"].append(step)
        hist["loss"].append(float(jnp.real(loss_mean)))
        hist["E_levels"].append(eig[:N_REPLICA])
        if (step + 1) % cfg.diag_every == 0 or step == 0:
            B = gauge_fixed_basis_vectors(single_machine_list,
                                          [total_params["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
            ana = exact_subspace_analysis(B, H, evecs)
            # 重新提取当前基态列组合
            Mm, Ss = B.conj().T @ H @ B, B.conj().T @ B
            lv = scipy_eigh(0.5 * (Mm + Mm.conj().T), 0.5 * (Ss + Ss.conj().T))
            v0new = lv[1][:, np.argsort(lv[0].real)[0]]
            phi0_new = B @ v0new
            hist["eig_exact"].append((step, ana["lam"]))
            hist["phi0_fid"].append((step, fidelity(phi0_new, phi0_ref)))
            logger.info(f"[Control {step}] loss={hist['loss'][-1]:.6f} | exact={ana['lam'][:2]} | φ0 漂移保真={hist['phi0_fid'][-1][1]:.6f}")
        if (step + 1) % cfg.reset_period == 0:
            g[0] = col_mean_fn(total_params, x_batch)

    logger.info(f"Control 耗时 {time.time()-t0:.1f}s")
    return dict(total_params=total_params, hist=hist, phi0_ref=phi0_ref)


def run_frozen(logger, cfg, s0, states, H, evecs):
    """Route B：冻结 φ0，流形 B=[φ0, ψ1', ψ2', ψ3']，只训练 3 个 fresh active 列。"""
    bundle = make_frozen_bundle(s0["single_machine_list"], s0["total_params"]["single_ansatz_list"], s0["v0"])

    # fresh active 列（项目工程惯例：Stage1 必须新随机初始化，避免行列式奇异）
    active_ansatz = []
    key = jax.random.PRNGKey(77)
    for j in range(N_ACTIVE):
        key, sub = jax.random.split(key)
        active_ansatz.append(SingleStateAnsatz_SharedBackbone(SINGLE_SIZE, SINGLE_SIZE + N_REPLICA,
                                                              rngs=nnx.Rngs(params=sub)))
    active_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in active_ansatz]
    active_params = [nnx.split(a)[1] for a in active_ansatz]

    core, _one = make_frozen_forward(ha, bundle, active_sms)
    grad_fn = make_frozen_grad_fn(ha, core, _one)
    qgt_fn = make_frozen_qgt_fn(core, _one)
    ms_fn = make_frozen_MS_fn(ha, bundle, active_sms)

    g = [jnp.zeros(N_ACTIVE, dtype=jnp.complex128)]

    def sample_machine(params, sigma):
        x = sigma.reshape(-1, N_REPLICA, SINGLE_SIZE)
        return core(params, x, g[0])[0]

    sampler, sstate = make_sampler(sample_machine, active_params, 45)
    optimizer = optax.chain(optax.clip_by_global_norm(cfg.clip_norm), optax.sgd(cfg.lr))
    opt_state = optimizer.init(active_params)

    hist = {"step": [], "loss": [], "E_levels": [], "eig_exact": [], "phi0_fid": []}
    # 冻结态基向量（常量，共享 shift）
    Bf = gauge_fixed_basis_vectors(s0["single_machine_list"],
                                   [s0["total_params"]["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
    phi0_vec = Bf @ s0["v0"]
    t0 = time.time()
    for step in range(cfg.n_iter1):
        samples_raw, sstate = sampler.sample(machine=sample_machine, parameters=active_params,
                                             state=sstate, chain_length=cfg.n_samples)
        x_batch = samples_raw.reshape(-1, N_REPLICA, SINGLE_SIZE)
        grad_raw, loss_mean, E_L_mean, aux = grad_fn(active_params, x_batch, g[0])
        grad_flat, unravel = ravel_pytree(grad_raw)
        gnorm = float(jnp.linalg.norm(grad_flat))
        if gnorm / np.sqrt(grad_flat.size) > cfg.per_param_clip:
            grad_flat = grad_flat * (cfg.per_param_clip * np.sqrt(grad_flat.size) / gnorm)
            grad_raw = unravel(grad_flat)
        qgt = qgt_fn(active_params, x_batch, cfg.diag_shift, g[0])
        ng = jnp.linalg.solve(qgt, ravel_pytree(grad_raw)[0])
        updates, opt_state = optimizer.update(unravel(ng), opt_state, active_params)
        active_params = optax.apply_updates(active_params, updates)

        eig = np.sort(np.asarray(jnp.linalg.eig(E_L_mean)[0]).real)
        hist["step"].append(step)
        hist["loss"].append(float(jnp.real(loss_mean)))
        hist["E_levels"].append(eig[:N_REPLICA])
        if (step + 1) % cfg.diag_every == 0 or step == 0:
            Ba = gauge_fixed_basis_vectors(active_sms, list(active_params), states)
            blk = exact_frozen_block_analysis(phi0_vec, Ba, H, evecs)
            hist["eig_exact"].append((step, np.array([blk["e_phi0"], *blk["lam_active"]])))
            hist["phi0_fid"].append((step, blk["phi0_fid_fci"]))
            logger.info(f"[Frozen {step}] loss={hist['loss'][-1]:.6f} | E_L eig={eig[:2]} | block λ=[{blk['e_phi0']:.5f}, {blk['lam_active'][0]:.5f}] | φ0 保真(对FCI基态)={blk['phi0_fid_fci']:.6f}")
        if (step + 1) % cfg.reset_period == 0:
            La_means = jnp.stack([jnp.mean(active_sms[j](active_params[j], x_batch.reshape(-1, SINGLE_SIZE)))
                                  for j in range(N_ACTIVE)])
            g[0] = jnp.asarray(La_means, dtype=jnp.complex128)

    logger.info(f"Frozen 耗时 {time.time()-t0:.1f}s")
    # 最终：冻结流形 4×4 广义本征（MCMC 估计）
    Mhat, Shat, fin = ms_fn(active_params, x_batch)
    lamB, VB = scipy_eigh(np.asarray(Mhat), np.asarray(Shat))
    return dict(active_params=active_params, hist=hist, phi0_vec=phi0_vec,
                lamB=lamB.real, VB=VB, active_sms=active_sms)


# ======================================================================
# 6. 全局超参（main 中按 cfg 覆盖）
# ======================================================================
N_CHAINS_G, SWEEP_G = 16, 20


def main():
    global N_CHAINS_G
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-iter0", type=int, default=300)
    ap.add_argument("--n-iter1", type=int, default=200)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--n-chains", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--diag-shift", type=float, default=0.1)
    ap.add_argument("--clip-norm", type=float, default=20.0)
    ap.add_argument("--per-param-clip", type=float, default=0.5)
    ap.add_argument("--reset-period", type=int, default=10)
    ap.add_argument("--diag-every", type=int, default=25)
    ap.add_argument("--tag", type=str, default="full")
    args = ap.parse_args()
    N_CHAINS_G = args.n_chains

    time_str = time.strftime("%y-%m-%d-%H-%M")
    os.makedirs("./日志", exist_ok=True)
    os.makedirs("./data", exist_ok=True)
    logger = logging.getLogger("FrozenRouteB")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")
    fh = logging.FileHandler(f"./日志/{time_str}_frozen_routeB_{args.tag}.log", mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    logger.info(f"FCI: {E_fcis[:4]}")
    states, H, evals, evecs = build_exact_reference()
    logger.info(f"稠密 H 低段能级: {evals[:4]}")

    s0 = run_stage0(logger, args, states, H, evecs)
    ctl = run_control(logger, args, s0, states, H, evecs)
    frz = run_frozen(logger, args, s0, states, H, evecs)

    # ---------------- 最终精确对比 ----------------
    logger.info("=" * 70)
    logger.info("最终对比（全空间精确）")
    ctl_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
               for a in s0["total_ansatz"].single_ansatz_list]
    Bc = gauge_fixed_basis_vectors(ctl_sms,
                                   [ctl["total_params"]["single_ansatz_list"][j] for j in range(N_REPLICA)], states)
    ana_c = exact_subspace_analysis(Bc, H, evecs)
    fid_c = fidelity(Bc @ ana_c["V"][:, 0], ctl["phi0_ref"])
    Ba = gauge_fixed_basis_vectors(frz["active_sms"], list(frz["active_params"]), states)
    blk_f = exact_frozen_block_analysis(frz["phi0_vec"], Ba, H, evecs)
    fid_f = blk_f["phi0_fid_fci"]
    frz["final_block"] = {"e_phi0": blk_f["e_phi0"], "lam_active": blk_f["lam_active"],
                          "overlaps": [st["overlap_exact"] for st in blk_f["states"]]}

    logger.info(f"[Control] exact λ = {ana_c['lam']}")
    logger.info(f"[Control] φ0 漂移保真度 = {fid_c:.6f} | 对 FCI 基态重叠 = {ana_c['states'][0]['overlap_exact'][0]:.6f}")
    for k in range(N_REPLICA):
        logger.info(f"[Control] 态{k} 与 FCI 各态重叠 = {np.round(ana_c['states'][k]['overlap_exact'], 4)}")
    logger.info(f"[Frozen ] block λ = [{blk_f['e_phi0']:.6f}, {np.round(blk_f['lam_active'], 6)}]")
    logger.info(f"[Frozen ] φ0 保真度(对FCI基态) = {fid_f:.6f}（构造上严格不变）")
    for k in range(N_ACTIVE):
        logger.info(f"[Frozen ] active 态{k} 与 FCI 各态重叠 = {np.round(blk_f['states'][k]['overlap_exact'], 4)}")
    logger.info(f"[Frozen ] 冻结流形 MCMC 广义本征 λ_B = {frz['lamB']}")
    e0_component = abs(frz["VB"][0, 0])  # 第一列中 φ0 分量
    logger.info(f"[Frozen ] λ_B 第一本征矢的 φ0 分量 |V_B[0,0]| = {e0_component:.4f}（≈1 ⇒ 块结构成立）")

    out = {
        "cfg": vars(args), "fci": E_fcis[:N_REPLICA],
        "stage0": {"hist": s0["hist"], "v0": s0["v0"], "lam": s0["lam"]},
        "control": {"hist": ctl["hist"], "final_exact": ana_c["lam"],
                    "final_overlaps": [st["overlap_exact"] for st in ana_c["states"]],
                    "phi0_drift_fid": fid_c},
        "frozen": {"hist": frz["hist"], "final_block": frz["final_block"],
                   "phi0_fid_fci": fid_f, "lamB": frz["lamB"]},
    }
    with open(f"./data/{time_str}_frozen_routeB_{args.tag}.pkl", "wb") as f:
        import pickle
        pickle.dump(out, f)
    logger.info(f"结果已保存 → ./data/{time_str}_frozen_routeB_{args.tag}.pkl")


if __name__ == "__main__":
    main()
