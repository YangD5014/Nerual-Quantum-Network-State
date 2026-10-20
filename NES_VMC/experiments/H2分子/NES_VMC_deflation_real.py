"""
实参数（RealFFNN：实参数 + 复输出 logψ）版 NES-VMC Deflation 工具箱。

背景：`NES_VMC_deflation.py` 的 deflated 梯度 / QGT 使用
`jax.grad(..., holomorphic=True)`，只适用于**复参数**拟设；当前 K=4 训练
（`H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.py`）用的是「实参数 + 复输出」的
SingleStateAnsatz_SharedBackbone，holomorphic 路径直接报错。

本模块提供（对应文档 [GPT1014]NES_VMC_Matrix_Free_Subspace_Extension.md 方案 A）：
  1. `load_and_orthonormalize_frozen`：从 npz 载入冻结态 φ_0..φ_3 → Gram-Schmidt
     正交归一 F (16,r)，含基矢顺序与残差双重断言；
  2. `deflation_oracle`：对角化 Q H Q，给出剩余子空间最低本征值 ν（= 冻结近似
     本征态导致的**方法偏差下限**，训练不可能优于它，用于区分方法误差 vs 训练误差）；
  3. `make_deflated_grad_fn_real`：实参数力梯度（Wirtinger 拆分 ∇logΨ=∇Re+i∇Im，
     取实部 ×2）+ 重要性加权（配合 machine_pow=1 按 |Ψ_F| 采样，w∝|Ψ_F|），
     公式与已验证的 nes_vmc_gradient_stable_gauge_real 逐项同构，仅把
     (machine, E_L) 换成 deflation bundle 的 (machine_single, forward)；
  4. `make_deflated_qgt_fn_real`：实参数 QGT，度量取 Re(S)，w 加权中心化。

约定：
  - bundle 的 forward/machine 输入 sigma_flat 形状 (N, K*n_spin)，K=2 时即 (N,16)；
  - 冻结矩阵 F 为 float64 常量（bundle 内部已 stop_gradient），投影切空间
    ∂χ = Q∂A 由 autodiff 对 χ = A − F(F†A) 自动给出，无需手工修正项；
  - 梯度 pytree 叶子为 float（实参数），不做复数转换。
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from NES_VMC_V1 import flatten_batched_pytree


# ============================================================================
# 冻结态载入 + 正交归一（含防御性断言）
# ============================================================================
def load_and_orthonormalize_frozen(npz_path, hi, min_residual=0.1):
    """从 npz 载入 φ_0..φ_{r-1} 并做升序 Gram-Schmidt 正交归一。

    Parameters
    ----------
    npz_path : str        含键 'phi_vecs' (16,r) 与 'all_states' (16,8)
    hi       : netket Hilbert   当前计算所用 hi（基矢序必须与 npz 一致）
    min_residual : float  GS 每步残差范数下限（防止近零向量混入 F）

    Returns
    -------
    F       : (16, r) complex128 正交归一冻结矩阵
    phi_raw : (16, r) complex128 原始（列归一、非严格正交）载入向量，供重叠诊断
    residuals : (r,) float 每步 GS 残差范数
    """
    data = np.load(npz_path)
    phi_raw = np.asarray(data["phi_vecs"], dtype=np.complex128)
    if phi_raw.ndim == 1:
        phi_raw = phi_raw[:, None]
    all_states_npz = np.asarray(data["all_states"])
    all_states_hi = np.asarray(hi.all_states())
    assert all_states_npz.shape == all_states_hi.shape and np.array_equal(
        all_states_npz, all_states_hi
    ), "npz 基矢顺序与 hi.all_states() 不一致，禁止继续（投影会张冠李戴）"

    # 自行 GS 以记录残差（NES_VMC_deflation.orthonormalize_frozen 在近零分支
    # 仍会 append 未归一向量，这里用断言堵住该坑）
    vecs, residuals = [], []
    for m in range(phi_raw.shape[1]):
        p = phi_raw[:, m].astype(np.complex128).copy()
        for q in vecs:
            p = p - q * (np.conj(q) @ p)
        n = float(np.linalg.norm(p))
        residuals.append(n)
        assert n > min_residual, f"φ_{m} GS 残差范数 {n:.3e} 过小，冻结空间近退化"
        vecs.append(p / n)
    F = np.stack(vecs, axis=1)

    # 正交归一自检
    G = F.conj().T @ F
    assert np.allclose(G, np.eye(F.shape[1]), atol=1e-10), "F 非正交归一"
    return F, phi_raw, np.asarray(residuals)


# ============================================================================
# Oracle：QHQ 压缩谱（冻结偏差下限）
# ============================================================================
def deflation_oracle(F, H_dense, n_keep=2):
    """对角化 Q H Q（Q = I − F F†），返回剩余子空间最低 n_keep 本征值与向量。

    ν_0..ν_{n_keep-1} 是「在冻结 span 的正交补内做变分」的**理论下限**：
    冻结态只是近似本征态时 ν 与真实 E_4,E_5 存在偏差，训练收敛到 ν 即方法饱和。

    注意：必须先把 H 压缩到 F 的正交补的 12 维基 B 上再对角化。若直接对
    16×16 的 QHQ 求谱，span(F) 贡献 r 个**精确零模**，当目标本征值为正时
    （如 E7=+0.024 > 0）零模会混进最低 n_keep 个，导致 ν 排序错误。

    Returns
    -------
    nu      : (n_keep,) float 升序
    u_vecs  : (16, n_keep) complex128 对应本征矢（已归一，落在 Q 空间）
    all_evals : (D-r,) float 压缩算子 B†HB 全谱（升序）
    """
    Qf, _ = np.linalg.qr(F.astype(np.complex128), mode="complete")
    B = Qf[:, F.shape[1]:]                      # (D, D-r) 正交补标准正交基
    Hb = B.conj().T @ (H_dense @ B)
    Hb = 0.5 * (Hb + Hb.conj().T)
    evals, evecs = np.linalg.eigh(Hb)
    return evals[:n_keep].real, B @ evecs[:, :n_keep], evals.real


# ============================================================================
# 实参数辅助：∇logΨ（Wirtinger）与重要性权重 w ∝ |Ψ_F|
# ============================================================================
def grad_logpsi_real_single(machine, params, x_single):
    """单样本 ∇logΨ = ∇Re(logΨ) + i·∇Im(logΨ)（实参数、复输出）。"""
    re_grad = jax.grad(lambda p: jnp.real(machine(p, x_single)))(params)
    im_grad = jax.grad(lambda p: jnp.imag(machine(p, x_single)))(params)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def importance_weights_from_logpsi(log_psi, valid):
    """w_i ∝ |Ψ_F(σ_i)|（valid 上归一，Σw=1），配合 p∝|Ψ_F| 采样（machine_pow=1）。"""
    log_w = jnp.real(log_psi)
    valid_f = jnp.asarray(valid, dtype=jnp.float64)
    log_w_max = jnp.max(jnp.where(valid_f > 0, log_w, -jnp.inf))
    log_w_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(log_w_max), log_w_max, 0.0))
    w = jnp.exp(log_w - log_w_max)
    wv = w * valid_f
    return wv / jnp.maximum(jnp.sum(wv), 1e-12)


# ============================================================================
# Deflated 力梯度（实参数 + 重要性加权）
# ============================================================================
def make_deflated_grad_fn_real(machine_single, forward, min_valid_ratio=0.25):
    """deflated NES-VMC 实参数梯度函数。

    machine_single: (params, sigma_single (K*n_spin,)) -> scalar logΨ_F
    forward       : (params, sigma_flat (N, K*n_spin)) -> (logΨ_F (N,), E_L^F (N,K,K))

    公式（与 nes_vmc_gradient_stable_gauge_real 同构，Ψ→Ψ_F、E_L→E_L^F、去 g）：
        loss_n   = Re tr(E_L^F_n)
        wv       ∝ |Ψ_F|（valid 归一）
        E_L_mean = Σ wv_n E_L^F_n
        grad     = 2·Re Σ_n wv_n · tr(E_L^F_n − E_L_mean) · conj(∇logΨ_F_n)
    """

    @jax.jit
    def grad_fn(total_params, sigma_flat):
        log_psi, E_L = forward(total_params, sigma_flat)  # (N,), (N,K,K)

        loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))  # (N,)
        valid = (
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(E_L), axis=(-2, -1))
            & jnp.isfinite(jnp.real(log_psi))
        )
        valid_f = jax.lax.stop_gradient(valid.astype(jnp.float64))
        n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)
        valid_ratio = jnp.sum(valid_f) / sigma_flat.shape[0]

        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(log_psi, valid))

        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        E_L_mean = jnp.sum(E_L_safe * wv.reshape((-1, 1, 1)), axis=0)

        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)

        dlogPsi_batch = jax.vmap(
            lambda s: grad_logpsi_real_single(machine_single, total_params, s)
        )(sigma_flat)

        def weight_and_weighted_mean(grad_component):
            grad_safe = jnp.where(jnp.isfinite(grad_component), grad_component, 0.0)
            weights = (wv * tr_centered).reshape((-1,) + (1,) * (grad_component.ndim - 1))
            weighted = weights * jnp.conj(grad_safe)
            return 2.0 * jnp.real(jnp.sum(weighted, axis=0))  # float 叶子

        grad = jax.tree.map(weight_and_weighted_mean, dlogPsi_batch)

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

    return grad_fn


# ============================================================================
# Deflated QGT（实参数，度量 Re(S)，重要性加权）
# ============================================================================
def make_deflated_qgt_fn_real(machine_single, machine_batch):
    """machine_batch: (params, sigma_flat (N,K*n_spin)) -> logΨ_F (N,)（算权重用）。

    返回 qgt_fn(params, sigma_flat, diag_shift) -> Re(S_w) + diag_shift·I（float）。
    """

    @jax.jit
    def qgt_fn(total_params, sigma_flat, diag_shift):
        n_samples = sigma_flat.shape[0]

        grad_tree_batch = jax.vmap(
            lambda s: grad_logpsi_real_single(machine_single, total_params, s)
        )(sigma_flat)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)  # (N,P) complex

        log_psi = machine_batch(total_params, sigma_flat)
        finite_row = jnp.isfinite(jnp.real(log_psi))
        wv = jax.lax.stop_gradient(importance_weights_from_logpsi(log_psi, finite_row))

        O_safe = jnp.where(jnp.isfinite(O), O, 0.0)
        O_mean = jnp.sum(wv[:, None] * O_safe, axis=0, keepdims=True)
        O_centered = O_safe - O_mean

        S = (jnp.conj(O_centered) * wv[:, None]).T @ O_centered
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        return S_real + diag_shift * jnp.eye(S_real.shape[0], dtype=S_real.dtype)

    return qgt_fn
