"""
方案 A' —— 真 Matrix-Free deflation 工具箱（冻结 K=4 + 剩余子空间 K=2，2×2 χ 行列式）

对应计划：`NES_VMC/计划文档/1016-方案A转MatrixFree-A-prime-计划.md`

与精确版方案 A（`NES_VMC_deflation.make_deflation_bundle`）的唯一区别：
投影系数不再用全空间内积 c = F†A，而是用辅助 MCMC 的蒙特卡洛 Gram 估计。
训练全程不接触 2^N 维全空间数组（稠密 H / 全空间振幅仅允许出现在诊断/单测）。

核心数学（Z 抵消引理）：
  参考密度 q(y) = Σ_i|φ̃_i(y)|² + ε·Σ_m|ψ̃_m(y)|²（未归一，Z = Σ_y q 未知），
  y_k ~ p = q/Z。同一批样本、同一 q 估计的一切 Gram 量都带同一个 1/Z 因子：
      Ĝ = ⟨Φ̃|Φ̃⟩/Z,  M̂ = ⟨Φ̃|Ψ̃⟩/Z,  ĤGG = ⟨Φ̃|HΦ̃⟩/Z,  M̂2 = ⟨Φ̃|HΨ̃⟩/Z
  于是 c = G⁻¹M 与 c2 = G⁻¹(M2 − HGG·c) 中 Z 严格抵消 ⇒ 无需知道归一化。

尺度约定（防漂移爆炸，0917/方案A 的冻结分量漂移教训）：
  冻结列 φ_i = Σ_j ψ_j^old·v_rot[j,i]（旧网络逐点求值，常量参数），缩放 φ̃ = φ·e^{−f}；
  active 列 ψ̃_m = ψ_m·e^{−g_m}。f、g_m 由每批 aux 样本的 max 做 EMA 更新
  （beta=1 即 fresh 归一），保证 q 的两项同为 O(1)、Ψ̃/A_χ 元素 O(1)。
  χ̃ = Ψ̃ − Φ̃·c 与 det 的列缩放 e^{−(g0+g1)} 为实常量（stop-grad），
  不影响 E_L、∇logΨ_F，仅使 |Ψ_F| 采样分布差一常量因子（采样器自动归一）。

前向（全部逐点，walker x = (K, n_spin)）：
  A_χ = Ψ̃ − Φ̃c            （2×2）
  Z   = (HΨ̃ − HΦ̃·c) − Φ̃·c2   （project→H→project，与精确 bundle 同构）
  E_L = A_χ⁻¹ Z，logΨ_F = slogdet(A_χ)
  Hψ̃_m = Ham_psi_scaled(shift=g_m)，Hφ̃_i = Σ_j Ham_psi_scaled(frozen_j, shift=f)·v_rot[j,i]

梯度/QGT 与 NES_VMC_deflation_real 同构（实参数 Wirtinger + w∝|Ψ_F| 重要性加权），
但 c、c2、f、g 必须作为**显式参数**传入 jit 函数（闭包常量会被 jit 焊死）。

主采样器 parameters pytree = (total_params, c, c2, f, g)；
辅助采样器 parameters pytree = (total_params, f, g)，machine = log q，machine_pow=1。
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from NES_VMC_V1 import flatten_batched_pytree, Ham_psi_scaled
from NES_VMC_deflation_real import importance_weights_from_logpsi


def make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms,
                   n_spin=8, eps=0.25, gram_ridge=1e-12, min_valid_ratio=0.25,
                   pinv_rcond=1e-8):
    """构造 A' matrix-free deflation bundle。

    Parameters
    ----------
    ha           : netket 稀疏哈密顿量（get_conn_padded 路径，禁止 to_dense）
    frozen_sms   : list[r] gauge-fixed 旧网络 machine（常量参数）
    frozen_params: list[r] 对应参数（不参与梯度）
    v_rot        : (r, r) 列约定 φ_i = Σ_j ψ_j^old·v_rot[j,i]
    sms          : list[K] gauge-fixed active 单列 machine（可训练）
    eps          : q 中 active 混合项系数（P1 参考分布）
    gram_ridge   : Ĝ 对角正则
    pinv_rcond   : 监控量广义本征的伪逆截断阈值（丢弃 s < pinv_rcond·s_max 的方向）

    Returns
    -------
    dict: aux_logq / estimate_grams / machine_single / machine_batch /
          forward / grad_fn / qgt_fn / cols
          （estimate_grams 的 aux 里含唯一监控量 "lam_uniform"：单副本均匀测度
            Ritz 值，用截断伪逆求解；另有 "pinv_n_keep"/"pinv_cond" 供健康诊断）
    """
    r = len(frozen_sms)
    K = len(sms)
    V = jnp.asarray(v_rot, dtype=jnp.complex128)

    # ---------------- 逐点列求值 ----------------
    def _cols(p, sigma, f, g):
        """sigma (n, n_spin) -> Phi_tilde (n,r), Psi_tilde (n,K)。"""
        Lf = jnp.stack([frozen_sms[j](frozen_params[j], sigma) for j in range(r)], axis=-1)
        La = jnp.stack([sms[m](p["single_ansatz_list"][m], sigma) for m in range(K)], axis=-1)
        Phi = (jnp.exp(Lf) @ V) * jnp.exp(-f)
        Psi = jnp.exp(La - g[None, :])
        return Phi, Psi

    def _h_cols(p, sigma, f, g):
        """HPhi_tilde (n,r), HPsi_tilde (n,K)。"""
        Hpsi = jnp.stack(
            [Ham_psi_scaled(ha=ha, single_machine=sms[m],
                            params=p["single_ansatz_list"][m], x=sigma, shift=g[m])
             for m in range(K)], axis=-1)
        Hphi = jnp.stack(
            [Ham_psi_scaled(ha=ha, single_machine=frozen_sms[j],
                            params=frozen_params[j], x=sigma, shift=f)
             for j in range(r)], axis=-1) @ V
        return Hphi, Hpsi

    def _one_log(p, xw, c, f, g):
        """单 walker xw (K, n_spin) -> logΨ_F 标量。"""
        Phi, Psi = _cols(p, xw, f, g)
        A = Psi - Phi @ c
        sign, logabs = jnp.linalg.slogdet(A)
        return logabs + 1j * jnp.angle(sign)

    def _one_az(p, xw, c, c2, f, g):
        """单 walker -> (A, Z)：A=Ψ̃−Φ̃c（K×K 振幅矩阵），Z=(HΨ̃−HΦ̃c)−Φ̃c2。"""
        Phi, Psi = _cols(p, xw, f, g)
        Hphi, Hpsi = _h_cols(p, xw, f, g)
        A = Psi - Phi @ c
        Z = (Hpsi - Hphi @ c) - Phi @ c2
        return A, Z

    def _one_full(p, xw, c, c2, f, g):
        """单 walker -> (logΨ_F, E_L, valid)。"""
        A, Z = _one_az(p, xw, c, c2, f, g)
        E_L = jnp.linalg.solve(A, Z)
        sign, logabs = jnp.linalg.slogdet(A)
        log_psi = logabs + 1j * jnp.angle(sign)
        valid = (jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(Z))
                 & jnp.all(jnp.isfinite(E_L)) & jnp.isfinite(logabs)
                 & jnp.isfinite(jnp.real(log_psi)))
        return log_psi, E_L, valid

    # ---------------- 辅助采样器 machine：log q ----------------
    def aux_logq(params_aux, sigma):
        """params_aux = (total_params, f, g)；sigma (n, n_spin) -> (n,) log q。"""
        p, f, g = params_aux
        Phi, Psi = _cols(p, sigma, f, g)
        q = jnp.sum(jnp.abs(Phi) ** 2, axis=-1) + eps * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)
        return jnp.log(q)

    # ---------------- MC Gram 估计器 ----------------
    @jax.jit
    def estimate_grams(p, sigma_aux, f, g):
        """aux 批 (M, n_spin) -> (c, c2, max_phi2, max_psi, aux)。

        c (r,K)、c2 (r,K)：缩放基 (Φ̃, Ψ̃) 下的投影系数（stop-gradient 使用）。
        max_phi2/max_psi 供 EMA 尺度更新（update_scales）。
        """
        M = sigma_aux.shape[0]
        Phi, Psi = _cols(p, sigma_aux, f, g)
        Hphi, Hpsi = _h_cols(p, sigma_aux, f, g)
        q = jnp.sum(jnp.abs(Phi) ** 2, axis=-1) + eps * jnp.sum(jnp.abs(Psi) ** 2, axis=-1)
        wi = 1.0 / (q * M)
        cPhi = jnp.conj(Phi)
        Gm = cPhi.T @ (wi[:, None] * Phi) + gram_ridge * jnp.eye(r, dtype=jnp.complex128)
        Mm = cPhi.T @ (wi[:, None] * Psi)
        HG = cPhi.T @ (wi[:, None] * Hphi)
        M2 = cPhi.T @ (wi[:, None] * Hpsi)
        c = jnp.linalg.solve(Gm, Mm)
        c2 = jnp.linalg.solve(Gm, M2 - HG @ c)

        # ---- 单副本均匀测度 Ritz 值（唯一监控量；截断伪逆解广义本征）----
        # 物理能级 = 单副本希尔伯特空间计数测度下的 Ritz 值：
        #   S_ij = Σ_σ χ_i(σ)* χ_j(σ)，M_ij = Σ_σ χ_i(σ)* (Hχ_j)(σ)
        # 这里 q 上采样 + 1/(M·q) 自归一化权重 w 恰好还原该均匀测度；χ=Ψ̃−Φ̃c，
        # Hχ=HΨ̃−HΦ̃c（与收尾提取完全同构）。1/(qM) 与 1/q 只差公共常数，广义本征不变。
        #
        # 求解方式：S = U diag(s) U†，只保留 s > pinv_rcond·s_max 的方向：
        #   W = S⁺^{1/2} = U diag(1/√s_keep) U†，C = W M W，λ = eig(C)。
        #   被截断方向在 W 的零空间，C 对其作用为 0 ⇒ 本征值恒为 0；
        #   用与 S 尺度无关的判据识别：v 在“丢弃子空间”的投影占比
        #     drop = ‖(I − P_keep)v‖²，P_keep = U diag(keep) U†
        #   drop ≈ 1 ⇒ 该方向已被截断（填 NaN）；drop ≈ 0 ⇒ 可信（含真实的 λ=0）。
        #   注意不能拿 ‖Wv‖² 判方向：它带 1/s 的尺度，会随 S 整体缩放漂移。
        # 这样 S 近奇异（首步随机初始化列近共线）时不会把噪声方向放大成 ~1e10 的假能级。
        chi = Psi - Phi @ c
        hchi = Hpsi - Hphi @ c
        S_u = jnp.einsum("n,ni,nj->ij", wi, jnp.conj(chi), chi)
        M_u = jnp.einsum("n,ni,nj->ij", wi, jnp.conj(chi), hchi)
        S_u = 0.5 * (S_u + jnp.conj(S_u.T))
        M_u = 0.5 * (M_u + jnp.conj(M_u.T))
        s_all, U_all = jnp.linalg.eigh(S_u)                 # s 升序（实）
        s_max = jnp.maximum(s_all[-1], 1e-300)
        keep = s_all > pinv_rcond * s_max
        s_safe = jnp.where(keep, s_all, 1.0)
        inv_sqrt = jnp.where(keep, 1.0 / jnp.sqrt(s_safe), 0.0)
        Uc = jnp.conj(U_all.T)
        W = (U_all * inv_sqrt[None, :]) @ Uc               # S⁺^{1/2}（Hermitian）
        P_keep = (U_all * keep.astype(U_all.real.dtype)[None, :]) @ Uc
        C_tmp = W @ M_u @ W
        C_u = 0.5 * (C_tmp + jnp.conj(C_tmp.T))
        lam_raw, vec_all = jnp.linalg.eigh(C_u)
        resid = vec_all - P_keep @ vec_all                  # 丢弃子空间分量
        drop_frac = jnp.sum(jnp.abs(resid) ** 2, axis=0)    # ‖v‖=1（eigh 输出正交）
        lam_uniform = jnp.where(drop_frac < 0.5, lam_raw, jnp.nan)
        n_keep = jnp.sum(keep.astype(jnp.int32))
        s_min_keep = jnp.min(jnp.where(keep, s_all, jnp.inf))
        cond_pinv = jnp.where(
            n_keep > 0,
            s_max / jnp.maximum(s_min_keep, 1e-300),
            jnp.inf,
        )

        u = 1.0 / q
        u = u / jnp.sum(u)
        aux = {
            "w_ess": 1.0 / jnp.maximum(jnp.sum(u * u), 1e-30),
            "max_w_ratio": jnp.max(u) * M,          # 权重峰值/均值（爆炸预警）
            "grams_finite": (jnp.all(jnp.isfinite(c)) & jnp.all(jnp.isfinite(c2))
                             & jnp.all(jnp.isfinite(q)) & jnp.all(q > 0.0)),
            "lam_uniform": lam_uniform,             # (K,) 升序；NaN=被伪逆截断的方向
            "pinv_n_keep": n_keep,                  # 保留方向数（<K 说明该步秩亏）
            "pinv_cond": cond_pinv,                 # 保留子空间条件数（病态预警）
        }
        max_phi2 = jax.lax.stop_gradient(jnp.max(jnp.sum(jnp.abs(Phi) ** 2, axis=-1)))
        max_psi = jax.lax.stop_gradient(jnp.max(jnp.abs(Psi), axis=0))
        return c, c2, max_phi2, max_psi, aux

    # ---------------- 主采样 / 前向 ----------------
    def _machine_batch(p, sigma_flat, c, f, g):
        x = sigma_flat.reshape(-1, K, n_spin)
        return jax.vmap(lambda xw: _one_log(p, xw, c, f, g))(x)

    def _forward(p, sigma_flat, c, c2, f, g):
        x = sigma_flat.reshape(-1, K, n_spin)
        log_psi, E_L, valid = jax.vmap(lambda xw: _one_full(p, xw, c, c2, f, g))(x)
        return log_psi, E_L, valid

    machine_single = jax.jit(_one_log)
    machine_batch = jax.jit(_machine_batch)
    forward = jax.jit(_forward)

    # ---------------- 实参数梯度 / QGT（同构 NES_VMC_deflation_real） ----------------
    def _dlogpsi(p, xw, c, f, g):
        re_g = jax.grad(lambda q: jnp.real(_one_log(q, xw, c, f, g)))(p)
        im_g = jax.grad(lambda q: jnp.imag(_one_log(q, xw, c, f, g)))(p)
        return jax.tree.map(lambda a, b: a + 1j * b, re_g, im_g)

    @jax.jit
    def grad_fn(p, sigma_flat, c, c2, f, g):
        x = sigma_flat.reshape(-1, K, n_spin)
        # A、Z 一次算好并复用：E_L/logψ 用于梯度、QGT 与 Loss
        A, Z = jax.vmap(lambda xw: _one_az(p, xw, c, c2, f, g))(x)
        E_L = jnp.linalg.solve(A, Z)
        sign, logabs = jnp.linalg.slogdet(A)
        log_psi = logabs + 1j * jnp.angle(sign)
        valid = (jnp.all(jnp.isfinite(A), axis=(-2, -1))
                 & jnp.all(jnp.isfinite(Z), axis=(-2, -1))
                 & jnp.all(jnp.isfinite(E_L), axis=(-2, -1))
                 & jnp.isfinite(logabs) & jnp.isfinite(jnp.real(log_psi)))

        loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))
        valid = valid & jnp.isfinite(loss) & jnp.isfinite(jnp.real(log_psi))
        valid_f = jax.lax.stop_gradient(valid.astype(jnp.float64))
        n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)
        valid_ratio = jnp.sum(valid_f) / x.shape[0]

        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(log_psi, valid))

        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        E_L_mean = jnp.sum(E_L_safe * wv.reshape((-1, 1, 1)), axis=0)
        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)

        dlog = jax.vmap(lambda xw: _dlogpsi(p, xw, c, f, g))(x)

        def weight_and_weighted_mean(comp):
            comp_safe = jnp.where(jnp.isfinite(comp), comp, 0.0)
            w = (wv * tr_centered).reshape((-1,) + (1,) * (comp.ndim - 1))
            return 2.0 * jnp.real(jnp.sum(w * jnp.conj(comp_safe), axis=0))

        grad = jax.tree.map(weight_and_weighted_mean, dlog)

        loss_safe = jnp.where(jnp.isfinite(loss), loss, 0.0)
        loss_mean = jnp.sum(wv * loss_safe)

        grad_flat, _ = ravel_pytree(grad)
        ess = 1.0 / jnp.maximum(jnp.sum(wv * wv), 1e-30)
        aux = {
            "valid_ratio": valid_ratio,
            "n_valid": n_valid,
            "ess": ess,
            "grad_finite": jnp.all(jnp.isfinite(grad_flat)),
            "loss_finite": jnp.isfinite(loss_mean),
            "should_skip": ~(
                jnp.all(jnp.isfinite(grad_flat))
                & jnp.isfinite(loss_mean)
                & (valid_ratio >= min_valid_ratio)
            ),
        }
        return grad, loss_mean, E_L_mean, aux

    @jax.jit
    def qgt_fn(p, sigma_flat, diag_shift, c, c2, f, g):
        x = sigma_flat.reshape(-1, K, n_spin)
        n_samples = x.shape[0]
        log_psi, _, valid = jax.vmap(lambda xw: _one_full(p, xw, c, c2, f, g))(x)
        grad_tree_batch = jax.vmap(lambda xw: _dlogpsi(p, xw, c, f, g))(x)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)  # (N,P) complex

        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(log_psi, valid))
        O_safe = jnp.where(jnp.isfinite(O), O, 0.0)
        O_mean = jnp.sum(wv[:, None] * O_safe, axis=0, keepdims=True)
        O_centered = O_safe - O_mean
        S = (jnp.conj(O_centered) * wv[:, None]).T @ O_centered
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        return S_real + diag_shift * jnp.eye(S_real.shape[0], dtype=S_real.dtype)

    return {
        "aux_logq": aux_logq,
        "estimate_grams": estimate_grams,
        "machine_single": machine_single,
        "machine_batch": machine_batch,
        "forward": forward,
        "grad_fn": grad_fn,
        "qgt_fn": qgt_fn,
        "cols": _cols,
        "K": K,
        "r": r,
    }


def update_scales(f, g, max_phi2, max_psi, beta=0.9):
    """EMA 尺度更新（numpy 侧，在 jit 外调用）。

    目标：max Σ|φ̃|² ≈ 1 且 max|ψ̃_m| ≈ 1。beta=1 为 fresh 每批重标。
    """
    max_phi2 = max(float(np.asarray(max_phi2).real), 1e-30)
    max_psi = np.maximum(np.asarray(max_psi, dtype=np.float64), 1e-30)
    f_new = float(f) + 0.5 * beta * np.log(max_phi2)
    g_new = np.asarray(g, dtype=np.float64) + beta * np.log(max_psi)
    return f_new, g_new
