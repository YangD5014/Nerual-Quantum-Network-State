"""
NES-VMC「矩阵自由（Matrix-Free）Deflation 冻结方案」— v2（真正矩阵自由 + MCMC）
================================================================================

对应文档：`NES_VMC/文档/0923-冻结方案策略.md`

与 v1（`NES_VMC_deflation.py`）的本质区别：

    v1 是"最小验证方案"——直接调用 `ha.to_dense()` 拿到 16×16 哈密顿矩阵、枚举全部
    16 个态求 overlap、用 scipy_eigh 做全空间广义本征分解。它只在小系统上成立。

    v2 是"完整可用方案"——**绝不**显式构造哈密顿矩阵、**绝不**枚举全空间：

        1. 哈密顿量 H 只通过 NetKet 的局部连接作用  `ha.get_conn_padded(x)` 出现
           （即 Hψ(x) = Σ_{x'} H[x,x'] ψ(x')，只对 x 的连通组态 x' 求和）；
        2. 冻结态 φ_0 表示成 Stage-0 收敛的 K 列 NN 的 stop-gradient 线性组合：
               φ_0(x) = Σ_k v_{k0} ψ_k(x)     （v_{k0} 是广义本征问题的基态旋-系数）
           ——不是稠密向量，而是一个"冻结的波函数快照"；
        3. 所有 overlap 标量 ⟨φ_0|ψ_j⟩ / ⟨φ_0|φ_0⟩ 用**独立的 |φ_0|² 采样器**做
           重要性采样估计（ratio estimator），这正是文档 §43 指出的大系统下最难的部分；
        4. deflated 局域能量完全 matrix-free：
               A^F[i,j] = ψ_j(x_i) − c_j φ_0(x_i)
               Z[i,j]   = (Hψ_j)(x_i) − c_j (Hφ_0)(x_i) − φ_0(x_i) (h_j − E_0 c_j)
           其中 c_j = ⟨φ_0|ψ_j⟩/⟨φ_0|φ_0⟩, h_j = ⟨φ_0|H|ψ_j⟩/⟨φ_0|φ_0⟩,
               E_0  = ⟨φ_0|H|φ_0⟩/⟨φ_0|φ_0⟩
           全部是全局标量（由 |φ_0|² 采样器估计、stop_gradient）。H 作用于 ψ_j 与 φ_0
           都只用 get_conn，不建任何大矩阵。

两个采样器：
    (a) 冻结态采样器：在 |φ_0|² 上 Metropolis 采样（单希尔伯特空间 hi），用于估计 overlap。
    (b) NES 联合行列式采样器：在 |det[χ_j(x_i)]|² 上采样（hi^K），用于梯度。

关键洞察（详见解读文档）：
    - stop_gradient(F 或 v0/params0) ⇒ ∂(Qψ)=Q∂ψ 自动成立（投影切空间）；
    - deflated local energy 用 Q H Q（而非裸 H Q），等价于文档的投影哈密顿量；
    - Stage-1 必须 fresh re-init，否则冻结后 active 列落在降维子空间、det 必然奇异。

本文件只提供工具箱，训练循环在 notebook（h2-6-31G-K4-deflation-v2.ipynb）。
"""

import numpy as np
import jax
import jax.numpy as jnp

from NES_VMC_V1 import (
    Ham_Psi_scaled,
    Ham_psi_scaled,
    flatten_batched_pytree,
    ravel_pytree,
)


# ============================================================================
# 冻结态 φ_0 的表示：Stage-0 收敛列 NN 的 stop-gradient 线性组合
# ============================================================================
def make_frozen_bundle(frozen_sms, frozen_params, v0, ha):
    """构造冻结态 φ_0 的「快照」表示与 |φ_0|² 采样 machine。

    冻结态不是稠密向量，而是：
        φ_0(x) = Σ_k v0[k] · ψ_k(x)
    其中 ψ_k 是 Stage-0 收敛的 gauge-fixed 单列 machine（参数 frozen_params 冻结），
    v0 是广义本征问题 M v = λ S v 的基态旋转系数（冻结的复向量）。

    Parameters
    ----------
    frozen_sms    : list[callable] 长度 K，Stage-0 的 gauge-fixed 单列 machine
    frozen_params : pytree，Stage-0 收敛参数（会被 stop_gradient）
    v0            : (K,) complex，基态旋转系数（会被 stop_gradient）
    ha            : NetKet 离散哈密顿量（只用 get_conn，不建矩阵）

    Returns
    -------
    bundle : dict
        'frozen_logs' : jitted (x (...,n_spin)) -> (..., K) 复 logψ_k(x)（冻结）
        'v0'          : (K,) complex（冻结）
        'logphi'      : jitted (x (n_spin,)) -> 标量 复 log φ_0(x)（稳定 log-sum-exp）
        'logphi_walker': jitted (x (N,K,n_spin)) -> (N,K) 复 log φ_0(x_{n,i})
        'hphi'        : jitted (x (N,K,n_spin)) -> (N,K) 复 (Hφ_0)(x_{n,i})（get_conn 作用）
        'machine'     : (params, sigma) -> 复 log φ_0(sigma)（供 NetKet 采样器）
        'frozen_sms'  : 原样持有（供 overlap 估计复用）
        'frozen_params': 原样持有
        'ha'          : 原样持有
    """
    K = len(v0)
    v0 = jax.lax.stop_gradient(jnp.asarray(v0, dtype=jnp.complex64))
    frozen_params = jax.lax.stop_gradient(frozen_params)

    # 单列 logψ_k。SingleStateAnsatz.__call__ 用 jnp.squeeze，会在批维为 1 时把
    # 批维压缩掉，因此这里对「单个 NES walker」显式按行计算，再 vmap，避免形状坍缩。
    def _col_logs(xw):
        # 单个 NES walker xw (K, n_spin) → (K, K)：[i, k] = logψ_k(x_{w, i})
        cols = [frozen_sms[k](frozen_params["single_ansatz_list"][k], xw)
                for k in range(K)]
        return jnp.stack(cols, axis=-1)

    def _logsumexp_combo(logs):
        # logs (..., K) → (...)：log φ_0 = log Σ_k v0[k]·ψ_k（稳定 log-sum-exp）
        shift = jax.lax.stop_gradient(jnp.max(logs.real, axis=-1))
        phi = jnp.sum(v0 * jnp.exp(logs - shift[..., None]), axis=-1)
        return jnp.log(phi) + shift

    @jax.jit
    def frozen_logs(x):
        # x：单个 config (n_spin,) 或 batch (N, n_spin) → (..., K)
        cols = [frozen_sms[k](frozen_params["single_ansatz_list"][k], x)
                for k in range(K)]
        return jnp.stack(cols, axis=-1)

    @jax.jit
    def logphi(x):
        # x：单个 config (n_spin,) → 标量 log φ_0
        return _logsumexp_combo(frozen_logs(x))

    @jax.jit
    def logphi_walker(x):
        # x：NES walker 批 (N, K, n_spin) → (N, K)：log φ_0(x_{n, i})（稳定 log-sum-exp）
        return jax.vmap(lambda xw: _logsumexp_combo(_col_logs(xw)))(x)

    @jax.jit
    def hphi(x):
        # x：NES walker 批 (N, K, n_spin) → (N, K)：(Hφ_0)(x_{n, i})（完整矩阵自由 H 作用）
        def _one(xw):
            x_primes, mels = ha.get_conn_padded(xw)   # (K,n_conn,n_spin),(K,n_conn)
            phi_primes = jax.vmap(
                lambda xp: jnp.exp(_logsumexp_combo(_col_logs(xp))))(x_primes)
            return jnp.sum(mels * phi_primes, axis=-1)   # (K,)
        return jax.vmap(_one)(x)

    def machine(params, sigma):
        return logphi(sigma)

    return {
        "frozen_logs": frozen_logs,
        "v0": v0,
        "logphi": logphi,
        "logphi_walker": logphi_walker,
        "hphi": hphi,
        "machine": machine,
        "frozen_sms": frozen_sms,
        "frozen_params": frozen_params,
        "ha": ha,
    }


# ============================================================================
# overlap 标量估计：c_j, h_j, E_0（用 |φ_0|² 采样器做重要性采样）
# ============================================================================
def estimate_overlaps(frozen_bundle, active_sms, active_params, x_frozen):
    """从 |φ_0|² 的样本估计 deflation 所需的全局标量。

        c_j = ⟨φ_0|ψ_j⟩/⟨φ_0|φ_0⟩   = E_{|φ_0|²}[ ψ_j(x)/φ_0(x) ]
        h_j = ⟨φ_0|H|ψ_j⟩/⟨φ_0|φ_0⟩ = E_{|φ_0|²}[ (Hψ_j)(x)/φ_0(x) ]
        E_0 = ⟨φ_0|H|φ_0⟩/⟨φ_0|φ_0⟩ = E_{|φ_0|²}[ (Hφ_0)(x)/φ_0(x) ]

    ratio estimator 的分子分母用统一 per-sample shift 稳定（shift 在比值中约分），
    且 |φ_0(x)| 过小的样本被掩码剔除（|φ_0|² 分布下极少发生）。

    Parameters
    ----------
    frozen_bundle : make_frozen_bundle 返回值
    active_sms    : list[callable] 长度 K，Stage-1 的 active（可训）gauge-fixed 单列 machine
    active_params : pytree，Stage-1 当前参数
    x_frozen      : (N, n_spin) 整数构型，来自 |φ_0|² 采样

    Returns（均为 host 侧 numpy，stop_gradient 常量）
    -------
    c     : (K,) complex128
    h     : (K,) complex128
    E0    : scalar complex128
    n_eff : int  有效样本数
    """
    ha = frozen_bundle["ha"]
    K = len(active_sms)
    v0 = frozen_bundle["v0"]
    frozen_sms = frozen_bundle["frozen_sms"]
    frozen_params = frozen_bundle["frozen_params"]

    x_f = jnp.asarray(x_frozen)

    logs_F = frozen_bundle["frozen_logs"](x_f)                              # (N,K)
    logs_A = jnp.stack(
        [active_sms[j](active_params["single_ansatz_list"][j], x_f) for j in range(K)],
        axis=-1,
    )                                                                       # (N,K)

    # 统一 shift（仅数值稳定；比值中约分）。停止梯度，避免经 shift 泄漏到 active 参数。
    shift = jax.lax.stop_gradient(
        jnp.max(jnp.concatenate([logs_F.real, logs_A.real], axis=-1), axis=-1)
    )                                                                       # (N,)

    phi = jnp.sum(v0[None, :] * jnp.exp(logs_F - shift[:, None]), axis=-1)  # (N,)  φ_0·e^-shift
    psi = jnp.exp(logs_A - shift[:, None])                                   # (N,K) ψ_j·e^-shift

    Hpsi = jnp.stack(
        [Ham_psi_scaled(ha, active_sms[j], active_params["single_ansatz_list"][j],
                        x_f, shift) for j in range(K)],
        axis=-1,
    )                                                                       # (N,K) (Hψ_j)·e^-shift
    Hpsi_F = jnp.stack(
        [Ham_psi_scaled(ha, frozen_sms[k], frozen_params["single_ansatz_list"][k],
                        x_f, shift) for k in range(K)],
        axis=-1,
    )                                                                       # (N,K)
    Hphi = jnp.sum(v0[None, :] * Hpsi_F, axis=-1)                            # (N,)  (Hφ_0)·e^-shift

    mag = jnp.abs(phi)
    valid = mag > 1e-8
    valid_f = jax.lax.stop_gradient(valid.astype(jnp.float32))
    n_eff = jnp.maximum(jnp.sum(valid_f), 1.0)

    def _ratio_mean_2d(num):                       # num: (N,K) -> (K,)
        r = jnp.where(valid[:, None], num / phi[:, None], 0.0)
        return jnp.sum(r, axis=0) / n_eff

    def _ratio_mean_1d(num):                       # num: (N,) -> scalar
        r = jnp.where(valid, num / phi, 0.0)
        return jnp.sum(r) / n_eff

    c = _ratio_mean_2d(psi)                        # (K,)
    h = _ratio_mean_2d(Hpsi)                       # (K,)
    E0 = _ratio_mean_1d(Hphi)                      # scalar

    c = jax.lax.stop_gradient(c)
    h = jax.lax.stop_gradient(h)
    E0 = jax.lax.stop_gradient(E0)

    return (np.asarray(c), np.asarray(h), float(np.real(np.asarray(E0))),
            int(np.asarray(n_eff)))


# ============================================================================
# Deflation 前向：matrix-free 的 machine / local-energy（r=0 退化为标准 NES）
# ============================================================================
def make_deflation_bundle_v2(active_sms, frozen_bundle, ha, K, n_spin):
    """构造 deflated NES 的 machine / local-energy / forward。

    当 frozen_bundle 为 None（r=0）时，c=h=E0 恒 0，A^F=ψ、Z=Hψ，退化标准 NES-VMC。

    Returns
    -------
    machine_single : jitted (params, sigma_single(32,), c, h, E0) -> scalar logΨ^F
    machine        : jitted (params, sigma(N,32), c, h, E0) -> (N,) logΨ^F  （采样器用）
    el             : jitted (params, sigma(N,32), c, h, E0) -> (N,K,K) E_L^F
    forward        : jitted (params, sigma(N,32), c, h, E0) -> (logΨ^F, E_L^F)
    """
    has_frozen = frozen_bundle is not None

    def _core(params, sigma_flat, c, h, E0):
        N = sigma_flat.shape[0]
        x = sigma_flat.reshape(N, K, n_spin)                            # (N,K,n_spin)

        # 主动列 logψ_j(x_i)。逐 walker vmap，规避 SingleStateAnsatz.squeeze 批维坍缩。
        logP = jnp.stack(
            [jax.vmap(lambda xw: active_sms[j](params["single_ansatz_list"][j], xw))(x)
             for j in range(K)],
            axis=-1,
        ).astype(jnp.complex64)                                         # (N,K,K)

        # per-walker 安全 shift（stop_gradient），类比参考实现的 max Re(L)。冻结态
        # log φ_0 也一并对齐，保证 exp(logP - s) 与 exp(logφ_0 - s) 都不溢出。
        if has_frozen:
            logphi0 = frozen_bundle["logphi_walker"](x)                 # (N,K)
            s = jax.lax.stop_gradient(jnp.maximum(
                logP.real.max(axis=(-2, -1)), logphi0.real.max(axis=-1)))
        else:
            s = jax.lax.stop_gradient(logP.real.max(axis=(-2, -1)))     # (N,)

        M = jnp.exp(logP - s[:, None, None])                            # (N,K,K) 稳定
        HM = Ham_Psi_scaled(ha, active_sms, params, x, s).astype(jnp.complex64)  # e^{-s} Hψ

        if not has_frozen:
            A_F, Z = M, HM                                              # 退化标准 NES-VMC
        else:
            c64 = jnp.asarray(c, dtype=jnp.complex64)                   # (K,)
            h64 = jnp.asarray(h, dtype=jnp.complex64)                   # (K,)
            E0c = jnp.asarray(E0, dtype=jnp.complex64)                  # scalar
            phi0s = jnp.exp(logphi0 - s[:, None])                       # (N,K) = φ_0·e^{-s}
            Hphi0s = frozen_bundle["hphi"](x) * jnp.exp(-s[:, None])    # (N,K) = Hφ_0·e^{-s}
            A_F = M - c64[None, None, :] * phi0s[:, :, None]
            Z = HM - c64[None, None, :] * Hphi0s[:, :, None] \
                 - phi0s[:, :, None] * (h64 - E0c * c64)[None, None, :]

        # log det(A_F) = log det(e^{s}·A_F^{shift}) = K·s + slogdet(A_F^{shift})
        sign, log_abs = jnp.linalg.slogdet(A_F)
        log_psi = log_abs + 1j * jnp.angle(sign) + K * s
        E_L = jnp.linalg.solve(A_F, Z)
        return log_psi, E_L

    forward = jax.jit(_core)

    @jax.jit
    def machine(params, sigma, c, h, E0):
        return _core(params, sigma, c, h, E0)[0]

    @jax.jit
    def el(params, sigma, c, h, E0):
        return _core(params, sigma, c, h, E0)[1]

    @jax.jit
    def machine_single(params, sigma_single, c, h, E0):
        return _core(params, sigma_single[None, :], c, h, E0)[0][0]

    return machine_single, machine, el, forward


# ============================================================================
# Deflated 梯度（复刻 NES-VMC 梯度公式：Ψ→Ψ^F, E_L→E_L^F）
# ============================================================================
def make_deflated_grad_fn_v2(forward, min_valid_ratio=0.25):
    """构造 deflated NES-VMC 梯度函数。

        loss_n      = Re tr(E_L^F_n)
        grad        = mean_n [ tr(E_L^F_n − mean) · conj(∇_holo logΨ^F_n) ]

    冻结态为 stop_gradient 常量 ⇒ 自动满足文档 §28 的投影切空间 ∂(Qψ)=Q∂ψ。
    """
    @jax.jit
    def grad_fn(params, sigma_flat, c, h, E0):
        log_psi, E_L = forward(params, sigma_flat, c, h, E0)

        loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))            # (N,)
        valid = (
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(E_L), axis=(-2, -1))
            & jnp.isfinite(jnp.real(log_psi))
        )
        valid_f = jax.lax.stop_gradient(valid.astype(jnp.float32))
        n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)

        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        E_L_mean = jnp.sum(E_L_safe * valid_f.reshape(-1, 1, 1), axis=0) / n_valid

        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)
        w = jax.lax.stop_gradient(tr_centered)
        vf_c = valid_f.astype(jnp.complex64)

        def objective(p):
            lp, _ = forward(p, sigma_flat, c, h, E0)
            return jnp.sum(jnp.conj(w) * lp * vf_c)

        g = jax.grad(objective, holomorphic=True)(params)
        grad = jax.tree.map(
            lambda leaf: jnp.conj(leaf).astype(jnp.complex64) / n_valid, g
        )

        loss_safe = jnp.where(jnp.isfinite(loss), loss, 0.0)
        loss_mean = jnp.sum(loss_safe * valid_f) / n_valid
        valid_ratio = n_valid / sigma_flat.shape[0]

        grad_flat, _ = ravel_pytree(grad)
        aux = {
            "valid_ratio": valid_ratio,
            "n_valid": n_valid,
            "grad_finite": jnp.all(jnp.isfinite(grad_flat)),
            "loss_finite": jnp.isfinite(loss_mean),
            "should_skip": ~(
                jnp.all(jnp.isfinite(grad_flat))
                & jnp.isfinite(loss_mean)
                & (valid_ratio >= min_valid_ratio)
            ),
        }
        return grad, loss_mean, E_L_mean, aux

    return grad_fn


# ============================================================================
# Deflated QGT / 自然梯度
# ============================================================================
def make_deflated_qgt_fn_v2(machine_single):
    """基于 deflated machine 的量子几何张量（S + diag_shift·I）。"""
    grad_logpsi = jax.grad(machine_single, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0, None, None, None))

    @jax.jit
    def qgt_fn(params, sigma, c, h, E0, diag_shift):
        n = sigma.shape[0]
        O = flatten_batched_pytree(
            vmap_grad_logpsi(params, sigma, c, h, E0), n)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        S = (jnp.conj(O_centered).T @ O_centered) / n
        S = 0.5 * (S + jnp.conj(S.T))
        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I

    return qgt_fn