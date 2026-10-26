import logging
import os
import time
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh as scipy_eigh
from jax.flatten_util import ravel_pytree

from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    NESTotalAnsatz,
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


# ====================== 基于 MCMC 样本的能级→列映射（M、S、v 全 K×K）======================
def make_MS_estimator_fn(ha, single_machine_list):
    """构造（jit 编译的）样本统计器：输入 (total_params, x_batch)，输出 K×K 的 M̂、Ŝ 与有效掩码。

    每 walker 构造（全程 K×K，不触碰完整 n_states×n_states 的 H，不做精确对角化）：
        P_{nαβ}    = ψ_β(x_{nα})       —— β 列拟设算到该 walker 的 K 个副本构型上
        (HP)_{nαβ} = (Hψ_β)(x_{nα})    —— 局部能量算子 Ham_Psi_scaled 作用在被采样构型上
    逐 walker 按 Frobenius 范数归一化（P、HP 同乘公共标度，不改变广义本征值），
    再做样本平均（并强制 Hermitian）：
        Ŝ = mean_n P_n† P_n ,   M̂ = mean_n P_n† (HP)_n

    返回的 estimator: jitted (total_params, x_batch) -> (Mhat (K,K), Shat (K,K), finite (N,))
    广义本征问题 M v = λ S v 由 compute_lam_v_from_samples 完成本函数不负责。
    """
    K_ = len(single_machine_list)

    def _estimate(total_params, x_batch):
        # ---- 1) 单 walker 的 K×K 阵 P：逐列拟设算到 (N, K, n_spin) 的副本构型 ----
        logP_cols = [
            single_machine_list[j](total_params["single_ansatz_list"][j], x_batch)
            for j in range(K_)
        ]
        logP = jnp.stack(logP_cols, axis=-1)                    # (N, α, β)
        shift_n = jnp.max(logP.real, axis=(1, 2)).reshape(-1)   # 每 walker 公共安全标度
        P = jnp.exp(logP - shift_n.reshape(-1, 1, 1))           # (N, K, K)

        # ---- 2) H·ψ：局部能量算子作用到被采样构型（不构造完整 H 矩阵）----
        HP = Ham_Psi_scaled(
            ha=ha,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x=x_batch,
            shift=shift_n,
        )                                                       # (N, K, K)

        # ---- 3) 有效性掩码 + 逐 walker Frobenius 归一化 ----
        finite = (
            jnp.all(jnp.isfinite(P), axis=(-2, -1))
            & jnp.all(jnp.isfinite(HP), axis=(-2, -1))
        )
        Pn = jnp.where(finite.reshape(-1, 1, 1), P, 0.0)
        HPn = jnp.where(finite.reshape(-1, 1, 1), HP, 0.0)
        norm = jnp.sqrt(jnp.sum(jnp.abs(Pn) ** 2, axis=(1, 2)) + 1e-30).reshape(-1, 1, 1)
        Pn = Pn / norm
        HPn = HPn / norm

        # ---- 4) 样本平均 → Ŝ、M̂（K×K，强制 Hermitian 消浮点误差）----
        N_eff = jnp.maximum(jnp.sum(finite.astype(jnp.float32)), 1.0)
        Shat = jnp.einsum("nai,naj->ij", jnp.conj(Pn), Pn).astype(jnp.complex128) / N_eff
        Mhat = jnp.einsum("nai,naj->ij", jnp.conj(Pn), HPn).astype(jnp.complex128) / N_eff
        Shat = 0.5 * (Shat + jnp.conj(Shat.T))
        Mhat = 0.5 * (Mhat + jnp.conj(Mhat.T))
        return Mhat, Shat, finite

    return jax.jit(_estimate)


def compute_lam_v_from_samples(
    ha,
    single_machine_list,
    total_params,
    x_batch,
    return_MS=False,
    estimator=None,
):
    """基于 MCMC 样本计算广义本征值 λ 与旋转矩阵 v（M、S、v 全部 K×K）。

    在 K 列张成的子空间上解广义本征问题  M v = λ S v：
        M_{ij} = ⟨ψ_i | H | ψ_j⟩ ,  S_{ij} = ⟨ψ_i | ψ_j⟩   （均由样本估计，K×K）
    旋转后第 k 列（能量升序）即第 k 个本征态：
        φ_k = Σ_j ψ_j v[k, j]
    能级 i 对应的「主动列」= argmax_j |v[i, j]|（见 level_to_active_col，分层冻结用）。

    参数
    ----
    ha                  : NetKet 哈密顿量（供局部能量算子 Ham_Psi_scaled 使用）
    single_machine_list : 长度 K 的 gauge-fixed 单列 machine 列表（与训练一致，
                          create_single_machine_gauge_fixed(ansatz, ref)[0]）
    total_params        : 总参数 pytree（含 'single_ansatz_list'）
    x_batch             : (N, K, n_spin) 整数 walker 构型
                          （nes_sampler 样本 reshape(-1, K, n_spin) 而来）
    return_MS           : True 时同时返回 (Mhat, Shat)
    estimator           : 可选，预先构造好的 make_MS_estimator_fn 返回值。
                          训练循环中反复调用时传入它可避免每次重编译。

    返回
    ----
    lam     : (K,) float64，广义本征值（能量升序）
    v       : (K, K) complex，旋转系数（列按能量升序排列）
    n_valid : 参与统计的有效 walker 数
    Mhat, Shat : (K, K) complex（仅 return_MS=True 时返回）
    """
    if estimator is None:
        estimator = make_MS_estimator_fn(ha, single_machine_list)
    Mhat, Shat, finite = estimator(total_params, x_batch)

    M_np = np.asarray(Mhat)
    S_np = np.asarray(Shat)
    lam, v = scipy_eigh(M_np, S_np)             # K×K 复 Hermitian 广义本征问题
    order = np.argsort(lam.real)                # 能量升序
    lam, v = lam[order], v[:, order]

    n_valid = int(np.asarray(finite).sum())
    if return_MS:
        return lam.real, v, n_valid, M_np, S_np
    return lam.real, v, n_valid,order


def level_to_active_col(v, level):
    """能级 level 对应的主动列：旋转矩阵第 level 行绝对值最大的列。

    v: (K, K)，行 = 能级（能量升序），列 = 原始列。
    返回 int 列下标 j，使 |v[level, j]| 最大 —— 该列即与能级 level 最"对齐"的原始列，
    作为分层冻结时 stop_gradient 的目标列。
    """
    return int(np.argmax(np.abs(np.asarray(v)[level])))



# ====================== 【修正版】无偏M、S估计器，|Ψ|^2测度 ======================
def make_MS_unbiased_estimator_fn(ha, single_machine_list):
    """
    【无偏版本】基于 MCMC 样本估计 Gram 矩阵
        S_ij = ⟨ψ_i | ψ_j⟩ , M_ij = ⟨ψ_i | H | ψ_j⟩
    数学：
        Ŝ = Σ_n w_n * P_n† P_n / Σ w_n
        M̂ = Σ_n w_n * P_n† (HP)_n / Σ w_n
    不再做逐walker Frobenius归一化；使用采样重要性权重 w_n 恢复 |Ψ|² 测度。

    参数：
        ha: NetKet哈密顿
        single_machine_list: K个单列ansatz列表
    返回 jitted estimator:
        estimator(total_params, x_batch, w_batch) -> (Mhat, Shat, finite_mask)
            x_batch: (N, K, n_spin) walker构型组
            w_batch: (N,) 重要性权重，来自NES采样器
    """
    K_ = len(single_machine_list)

    def _estimate(total_params, x_batch, w_batch):
        # ---- 1) 计算 P[n, α, β] = ψ_β(x_{nα}) ----
        logP_cols = [
            single_machine_list[j](total_params["single_ansatz_list"][j], x_batch)
            for j in range(K_)
        ]
        logP = jnp.stack(logP_cols, axis=-1)  # (N, K, K)
        shift_n = jnp.max(logP.real, axis=(1, 2)).reshape(-1)   # per‑walker log‑shift 防exp溢出
        P = jnp.exp(logP - shift_n.reshape(-1, 1, 1))           # (N, K, K)

        # ---- 2) H作用于各个列拟设 HP[n,α,β] = (Hψ_β)(x_{nα}) ----
        HP = Ham_Psi_scaled(
            ha=ha,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x=x_batch,
            shift=shift_n,
        )  # (N, K, K)

        # ---- 3) 有限有效性掩码，过滤NaN/Inf样本 ----
        finite = (
            jnp.all(jnp.isfinite(P), axis=(-2, -1))
            & jnp.all(jnp.isfinite(HP), axis=(-2, -1))
            & jnp.isfinite(w_batch)
        )
        # 无效样本置0，权重置0，不参与统计
        Pn = jnp.where(finite.reshape(-1, 1, 1), P, 0.0)
        HPn = jnp.where(finite.reshape(-1, 1, 1), HP, 0.0)
        wn = jnp.where(finite, w_batch, 0.0)  # (N,)

        # ---- 4) 重要性加权平均，得到 M̂、Ŝ；不再做逐walker归一化！ ----
        sum_w = jnp.maximum(jnp.sum(wn), 1e-30)  # 防止除0
        Shat = jnp.einsum("n,nai,naj->ij", wn, jnp.conj(Pn), Pn).astype(jnp.complex128) / sum_w
        Mhat = jnp.einsum("n,nai,naj->ij", wn, jnp.conj(Pn), HPn).astype(jnp.complex128) / sum_w

        # 强制Hermitian，消除浮点引入的微小非厄米噪声
        Shat = 0.5 * (Shat + jnp.conj(Shat.T))
        Mhat = 0.5 * (Mhat + jnp.conj(Mhat.T))

        return Mhat, Shat, finite

    return jax.jit(_estimate)


def compute_lam_v_from_samples_unbiased(
    ha,
    single_machine_list,
    total_params,
    x_batch,
    w_batch,
    return_MS=False,
    estimator=None,
):
    """
    【无偏版本】广义本征求解 M v = λ S v
    参数新增 w_batch: (N,) 采样重要性权重
    """
    if estimator is None:
        estimator = make_MS_unbiased_estimator_fn(ha, single_machine_list)

    Mhat, Shat, finite = estimator(total_params, x_batch, w_batch)
    M_np = np.asarray(Mhat)
    S_np = np.asarray(Shat)

    # 广义厄米本征求解
    lam, v = scipy_eigh(M_np, S_np)
    order = np.argsort(lam.real)
    lam, v = lam[order], v[:, order]
    n_valid = int(np.asarray(finite).sum())

    if return_MS:
        return lam.real, v, n_valid, M_np, S_np
    return lam.real, v, n_valid, order





# ====================== 【正确能级监控】单副本均匀测度下的 Ritz 值 ======================
def make_uniform_level_estimator_fn(ha, single_machine_list, total_graphdef):
    """构造「单副本均匀测度」能级监控估计器（样本估计，不枚举全空间）。

    为什么需要它
    ------------
    物理能级是单副本希尔伯特空间（固定粒子数/自旋扇区）里计数测度下的 Ritz 值：
        S_ij = Σ_σ ψ_i(σ)* ψ_j(σ)
        M_ij = Σ_σ ψ_i(σ)* (Hψ_j)(σ)
    即 ⟨ψ_i|ψ_j⟩、⟨ψ_i|H|ψ_j⟩ 都是对全部基态的**均匀（计数）求和**。

    而 NES 训练/旧监控用的扩展空间测度是 |Ψ|²（p∝|Ψ| 采样 + w∝|Ψ| 重新加权）。
    它一般**不等于**单副本均匀内积（行列式态的单副本边缘分布不均匀、不因子化），
    直接用会得到一个被扭曲的度量：tr(S⁻¹M)=Σλ 的迹仍被保住，但单个 λ 会被
    重新分配（实测 E1/E4 各偏 ∓3.8 mHa，和精确值正好抵消），因此不能当能级读数。

    本估计器的做法（可推广，不依赖全空间枚举）
    ----------------------------------------
    1) 在一个**单副本**参考分布 q(σ) ∝ Σ_j |ψ_j(σ)|² 上采样（该分布支撑全空间）；
    2) 用自归一化重要性权重 w_n ∝ 1/q(σ_n) 把 q 样本还原成均匀测度；
    3) 逐样本计算 P[n,j]=ψ_j(σ_n) 与局部能量 E_L[n,j]=(Hψ_j)(σ_n)/ψ_j(σ_n)，
       其中 Hψ_j 用 ha.get_conn_padded + 逐列 logψ 差分稳定计算；
    4) S、M 在均匀测度下加权平均，再解广义本征问题 M v = λ S v。
    第 2 步中 q 的归一化常数会被自归一化权重消掉，无需知道配分函数。

    参数
    ----
    ha                  : NetKet 哈密顿量（供 get_conn_padded）
    single_machine_list : K 个单列 machine（gauge 常数在广义本征问题中自动抵消）
    total_graphdef      : nnx.split(total_model) 的 graphdef（用于算 q）

    返回
    ----
    estimator(total_params, x_batch) -> (Mhat (K,K), Shat (K,K), finite (N,))
    log_q_fn(total_params, sigma)   -> log Σ_j |ψ_j(σ)|²（供采样器作为 machine 使用）
    """
    K_ = len(single_machine_list)

    def log_q_fn(total_params, sigma):
        """log q(σ) = log Σ_j |ψ_j(σ)|²；NetKet machine_pow=1 时采样分布即 q。"""
        m = nnx.merge(total_graphdef, total_params)
        logs = jnp.stack(
            [m.single_ansatz_list[j](sigma) for j in range(K_)], axis=0
        )
        return jax.nn.logsumexp(2.0 * jnp.real(logs), axis=0)

    @jax.jit
    def _estimate(total_params, x_batch):
        # ---- 1) 各列 logψ_j(σ_n) 与参考 q ----
        logP = jnp.stack(
            [
                single_machine_list[j](
                    total_params["single_ansatz_list"][j], x_batch
                )
                for j in range(K_)
            ],
            axis=-1,
        )                                            # (N, K)
        log_q = log_q_fn(total_params, x_batch)      # (N,)
        finite = jnp.all(jnp.isfinite(logP), axis=1) & jnp.isfinite(log_q)

        # 逐列全局常数平移（广义本征问题对列缩放不变，仅为数值稳定）
        s_col = jax.lax.stop_gradient(jnp.max(jnp.real(logP), axis=0, keepdims=True))
        P = jnp.exp(logP - s_col)                    # (N, K)

        # ---- 2) 局部能量 E_L[n,j] = (Hψ_j)(σ_n)/ψ_j(σ_n) ----
        xp, mels = ha.get_conn_padded(x_batch)       # (N, C, n_spin), (N, C)
        els = []
        for j in range(K_):
            lp = single_machine_list[j](
                total_params["single_ansatz_list"][j], xp
            )                                        # (N, C)
            els.append(jnp.sum(mels * jnp.exp(lp - logP[:, j:j + 1]), axis=1))
        EL = jnp.stack(els, axis=-1)                 # (N, K)

        # ---- 3) 自归一化重要性权重 w ∝ 1/q，还原单副本均匀测度 ----
        logw = jnp.where(finite, -log_q, -jnp.inf)
        logw_max = jax.lax.stop_gradient(jnp.max(logw))
        logw = jnp.maximum(logw, logw_max - 700.0)   # 防 exp 下溢，等价于整体缩放
        w = jnp.where(finite, jnp.exp(logw), 0.0)
        sw = jnp.maximum(jnp.sum(w), 1e-30)

        # ---- 4) 均匀测度下的 S、M ----
        Pw = P * jnp.sqrt(w)[:, None]
        S = (jnp.conj(Pw).T @ Pw) / sw
        M = jnp.einsum(
            "n,ni,nj->ij", w, jnp.conj(P), P * EL
        ) / sw
        S = 0.5 * (S + jnp.conj(S.T))
        M = 0.5 * (M + jnp.conj(M.T))
        return M.astype(jnp.complex128), S.astype(jnp.complex128), finite

    return _estimate, log_q_fn


def compute_lam_v_from_samples_uniform(
    estimator,
    total_params,
    x_batch,
    return_MS=False,
):
    """用单副本均匀测度估计器解 Ritz 广义本征问题 M v = λ S v（能量升序）。"""
    Mhat, Shat, finite = estimator(total_params, x_batch)
    M_np = np.asarray(Mhat)
    S_np = np.asarray(Shat)

    lam, v = scipy_eigh(M_np, S_np)
    order = np.argsort(lam.real)
    lam, v = lam[order], v[:, order]
    n_valid = int(np.asarray(finite).sum())

    if return_MS:
        return lam.real, v, n_valid, M_np, S_np
    return lam.real, v, n_valid, order


def clip_by_per_param_norm(max_per_param_norm: float):
    """
    自定义Optax变换：按【每参数平均L2幅值】做全局缩放裁剪
    per_param_norm = ||g||_2 / sqrt(num_params)
    当 per_param_norm > max_per_param_norm，整体梯度等比缩放，保持梯度方向不变
    兼容复数梯度、任意pytree结构。
    """

    def init_fn(params):
        # 本变换无状态，返回空state
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        # updates: 梯度pytree
        grad_flat, _ = ravel_pytree(updates)
        n_params = grad_flat.size
        global_l2 = jnp.linalg.norm(grad_flat)
        per_param_norm = global_l2 / jnp.sqrt(n_params)

        # 计算缩放系数
        scale = jnp.where(
            per_param_norm > max_per_param_norm,
            max_per_param_norm / per_param_norm,
            1.0
        )
        # pytree全部叶子乘以scale
        updates_clipped = jax.tree.map(lambda arr: arr * scale, updates)
        return updates_clipped, state

    return optax.GradientTransformation(init_fn, update_fn)

