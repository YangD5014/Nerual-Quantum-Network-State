"""
NES-VMC 「矩阵自由（Matrix-Free）Deflation 冻结方案」实现
================================================================

对应文档：`NES_VMC/文档/0923-冻结方案策略.md`

核心思路（详情见 0923-冻结方案策略.md）：

1. **Stage 0**：常规 NES-VMC（K 个扩展副本，joint determinant 采样）收敛，
   得到物理基态 φ_0（通过广义本征问题 M v = λ S v 从 ψ_i 旋转得到）。

2. **冻结 φ_0^***：把收敛的物理基态保存为一个**冻结方向**（本系统 Hilbert 维数
   只有 16，因此直接把 φ_0^* 存成完整的 16 维复向量；这正是文档里"冻结态是一个
   冻结的波函数快照"在小系统中的等价形式，去掉了 stop-gradient 冻结 NN 的重求值开销）。

3. **构造投影子**：Q_0 = I − |φ_0^*⟩⟨φ_0^*|（多个冻结态时 Q_F = I − Φ(Φ†Φ)^{−1}Φ†）。

4. **Matrix-Free 施加投影**：从不显式构造 Q_0 H Q_0 ∈ C^{16×16} 之外的任何大矩阵，
   而是在每次 local-energy 计算时对 active 列做  project → H → project 三步：
       χ_j   = Q_0 ψ_j
       y_j   = H χ_j
       z_j   = Q_0 y_j    (= Q_0 H Q_0 ψ_j)
   determinant 保持 K×K：A^F[i,j] = χ_j(x_i)。

5. **Deflated 局域能量**：E_L^F = (A^F)^{−1} (Q_0 H A^F)，loss = Re tr(E_L^F)。

本文件只提供「工具箱」：
    - 全空间 staging（H_dense、all_states）
    - 全空间振幅矩阵 A (16×K) 求值
    - 冻结态提取（广义本征问题）
    - deflated machine（采样用 logΨ_F / 梯度用 ∇logΨ_F / QGT 用）
    - deflated local-energy 与梯度

训练循环放在 notebook 里（与 `h2-6-31G-K4.ipynb` 的结构一致）。

关键实现约定：
    - 系统维数 D = 16（H2 / 6-31G，SpinOrbitalFermions n_orbitals=4, (1,1)，8 个自旋轨道）。
    - `all_states` (16, 8) 与 `H_dense` (16, 16) 的基矢顺序一致（均为 hi.all_states()）。
    - 冻结态 φ_0^* 用一个「正交归一的冻结矩阵」 F ∈ C^{16×r} 表示（r 个冻结态）。
    - 投影所用振幅统一用 gauge-fixed 单列 machine：ψ_j(x) = exp(logψ_j(x) − logψ_j(ref))。
    - 所有投影/内积在 complex128 下进行；梯度最终转回 complex64 与参数 dtype 一致。
"""

import numpy as np
import jax
import jax.numpy as jnp

from scipy.linalg import eigh as scipy_eigh

from NES_VMC_V1 import (
    flatten_batched_pytree,
    ravel_pytree,
)


# ============================================================================
# 全空间 staging：H_dense 与 all_states（基矢顺序一致）
# ============================================================================
def build_full_stage(ha, hi):
    """返回全 Hilbert 空间 staging 数据。

    Parameters
    ----------
    ha : netket DiscreteOperator
        系统的哈密顿量（作用于 hi，本系统 16×16）。
    hi : netket Hilbert
        SpinOrbitalFermions。

    Returns
    -------
    H_dense   : (16, 16) complex128 numpy array
        在 hi.all_states() 基矢顺序下的完整哈密顿量矩阵。
    all_states: (16, 8) int numpy array
        全部 Fock 构型（行 = 基矢下标）。
    """
    H_dense = np.asarray(ha.to_dense()).astype(np.complex128)
    all_states = np.asarray(hi.all_states())
    return H_dense, all_states


def states_to_idx_np(all_states, configs):
    """把构型数组映射成 all_states 的行下标（numpy 侧，供训练循环使用）。

    configs: (..., 8) int -> 返回 (...,) int。
    """
    configs = np.asarray(configs)
    eq = (configs[..., None, :] == all_states).all(axis=-1)  # (..., 16)
    return eq.argmax(axis=-1)


# ============================================================================
# 全空间振幅矩阵 A (16×K)
# ============================================================================
def amplitudes_full(single_machine_list, total_params, all_states_j):
    """在全部 16 个构型上求值 K 个单列拟设的振幅 ψ_j。

    Parameters
    ----------
    single_machine_list : list[callable], 长度 K
        gauge-fixed 单列 machine：machine_j(params_j, sigma) = logψ_j(sigma) − logψ_j(ref)。
    total_params : pytree
        总参数（含 'single_ansatz_list'）。
    all_states_j : (16, 8) jnp int
        全部构型。

    Returns
    -------
    A : (16, K) complex128
        A[i, j] = ψ_j(all_states[i])。
    """
    K = len(single_machine_list)
    cols = []
    for j in range(K):
        log_vals = single_machine_list[j](
            total_params["single_ansatz_list"][j], all_states_j
        )  # (16,) complex
        cols.append(jnp.exp(log_vals.astype(jnp.complex128)))
    return jnp.stack(cols, axis=1)  # (16, K)


# ============================================================================
# 冻结态提取（广义本征问题 M v = λ S v，在 ψ_i 张成的 K 维子空间内）
# ============================================================================
def extract_frozen_state(single_machine_list, total_params, all_states, H_dense):
    """从当前 NES 参数里提取物理基态作为「待冻结方向」。

    在 K 列张成的子空间上：
        S = ⟨ψ_i|ψ_j⟩,  M = ⟨ψ_i|H|ψ_j⟩  （全空间精确求值，16 维）
    解广义本征问题 M v = λ S v，取最低能本征矢：
        φ_0 = Σ_j ψ_j v[j, 0]，再归一化。

    Returns
    -------
    phi0 : (16,) complex128 归一化物理基态向量
    E0   : float  ⟨φ_0|H|φ_0⟩
    lam  : (K,) float  K 个子空间本征值（升序）
    v    : (K, K) complex 旋转系数
    A    : (16, K) complex 原始振幅矩阵（供重建其它物理态）
    """
    A = np.asarray(
        amplitudes_full(
            single_machine_list,
            total_params,
            jnp.asarray(all_states),
        )
    )  # (16, K) complex128

    S = A.conj().T @ A                       # (K, K)
    M = A.conj().T @ (H_dense @ A)           # (K, K)

    lam, v = scipy_eigh(M, S)                # 复 Hermitian 广义本征问题
    order = np.argsort(lam.real)
    lam = lam[order]
    v = v[:, order]

    phi0 = A @ v[:, 0]                       # (16,)
    phi0 = phi0 / np.linalg.norm(phi0)
    E0 = float((phi0.conj() @ (H_dense @ phi0)).real)
    return phi0, E0, lam.real, v, A


# ============================================================================
# 正交归一化冻结矩阵（多冻结态时消除 Monte Carlo/数值导致的非正交）
# ============================================================================
def orthonormalize_frozen(phi_list):
    """把一组（可能近似正交的）冻结态做成正交归一矩阵 F ∈ C^{16×r}。

    phi_list: list of (16,) complex128；返回 (16, r) complex128（标准 Gram-Schmidt）。
    """
    if len(phi_list) == 0:
        return np.zeros((1, 0), dtype=np.complex128)
    vecs = []
    for p in phi_list:
        p = np.asarray(p, dtype=np.complex128).copy()
        for q in vecs:
            p = p - q * (np.conj(q) @ p)
        n = np.linalg.norm(p)
        if n > 1e-12:
            p = p / n
        vecs.append(p)
    return np.stack(vecs, axis=1)          # (16, r)


# ============================================================================
# Deflation 前向：构造 machine（采样 / ∇logΨ / QGT 共用）与 local-energy
# ============================================================================
def make_deflation_bundle(single_machine_list, all_states, H_dense, frozen_vecs):
    """构造 deflated 波函数 machine 与 local-energy 算子。

    Parameters
    ----------
    single_machine_list : list[callable] 长度 K（gauge-fixed 单列 machine）
    all_states : (16, 8) int
    H_dense    : (16, 16) complex128
    frozen_vecs: (16, r) complex128（已正交归一；r=0 表示无冻结 = 退化为标准 NES）

    Returns
    -------
    machine_single : jitted (params, sigma_single (32,)) -> scalar logΨ_F
    machine        : jitted (params, sigma (N, 32))      -> (N,) logΨ_F   （采样器用）
    el             : jitted (params, sigma (N, 32))      -> (N, K, K) E_L^F
    forward        : jitted (params, sigma (N, 32))      -> (logΨ_F, E_L^F)
    """
    K = len(single_machine_list)
    n_states = all_states.shape[0]      # 16
    n_spin = all_states.shape[1]        # 8

    all_states_j = jnp.asarray(all_states)
    H_j = jnp.asarray(H_dense).astype(jnp.complex128)
    F = jnp.asarray(frozen_vecs).astype(jnp.complex128)   # (16, r)，常量（stop_gradient）
    F = jax.lax.stop_gradient(F)
    r = F.shape[1] if frozen_vecs.ndim == 2 and frozen_vecs.size else 0

    def _amplitudes(total_params):
        return amplitudes_full(single_machine_list, total_params, all_states_j)  # (16, K)

    def _project(A):
        """project → H → project，返回 (chi=A_f, z=QHQ·ψ)。"""
        if r == 0:
            return A, H_j @ A
        c = (jnp.conj(F).T) @ A          # (r, K)   c_mj = ⟨φ_m|ψ_j⟩
        chi = A - F @ c                  # (16, K)  χ_j = ψ_j − Σ_m φ_m ⟨φ_m|ψ_j⟩
        Hchi = H_j @ chi                 # (16, K)
        c2 = (jnp.conj(F).T) @ Hchi      # (r, K)
        z = Hchi - F @ c2                # (16, K)  z_j = Q H Q ψ_j
        return chi, z

    def _idx(x):
        # x: (..., 8) -> (...,) 基矢下标
        eq = (x[..., None, :] == all_states_j).all(-1)   # (..., 16)
        return eq.argmax(-1)

    def _single(total_params, sigma_single):
        x = sigma_single.reshape(K, n_spin)              # (K, 8)
        A = _amplitudes(total_params)                    # (16, K)
        chi, z = _project(A)
        idx = _idx(x)                                    # (K,)
        A_F = chi[idx]                                   # (K, K)
        Z = z[idx]                                       # (K, K)
        sign, log_abs = jnp.linalg.slogdet(A_F)
        return log_abs + 1j * jnp.angle(sign)            # scalar logΨ_F

    def _batch(total_params, sigma_flat):
        N = sigma_flat.shape[0]
        x = sigma_flat.reshape(N, K, n_spin)             # (N, K, 8)
        A = _amplitudes(total_params)                    # (16, K)（N 共享）
        chi, z = _project(A)
        idx = _idx(x)                                    # (N, K)
        A_F = chi[idx]                                   # (N, K, K)
        Z = z[idx]                                       # (N, K, K)
        E_L = jnp.linalg.solve(A_F, Z)                   # (N, K, K)
        sign, log_abs = jnp.linalg.slogdet(A_F)
        log_psi = log_abs + 1j * jnp.angle(sign)         # (N,)
        return log_psi, E_L

    machine_single = jax.jit(_single)
    machine = jax.jit(jax.vmap(_single, in_axes=(None, 0)))
    el = jax.jit(lambda params, sigma: _batch(params, sigma)[1])
    forward = jax.jit(_batch)

    return machine_single, machine, el, forward


# ============================================================================
# Deflated 梯度（复刻 nes_vmc_gradient_stable 的公式，把 Ψ→Ψ_F、E_L→E_L^F）
# ============================================================================
def make_deflated_grad_fn(forward, min_valid_ratio=0.25):
    """构造 deflated NES-VMC 梯度函数。

    梯度公式（与 NES_VMC_tool.nes_vmc_gradient_stable 完全一致，仅替换 Ψ、E_L）：
        loss_n      = Re tr(E_L^F_n)
        E_L_mean    = masked mean(E_L^F)
        tr_centered = tr(E_L^F − E_L_mean)
        grad        = mean_n [ tr_centered_n · conj(∇_holo logΨ_F_n) ]   (有效样本平均)

    冻结态 F 为 stop_gradient 常量，故自动满足文档 §28 的「投影切空间」
    ∂_µ χ = Q ∂_µ ψ（即 O_µ^F = Q ∂_µ ψ / Q ψ）。
    """
    @jax.jit
    def grad_fn(total_params, sigma_flat):
        log_psi, E_L = forward(total_params, sigma_flat)   # (N,), (N, K, K)

        loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))     # (N,)
        valid = (
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(E_L), axis=(-2, -1))
            & jnp.isfinite(jnp.real(log_psi))
        )
        valid_f = jax.lax.stop_gradient(valid.astype(jnp.float32))
        n_valid = jnp.maximum(jnp.sum(valid_f), 1.0)

        # masked mean（3D 与 1D 分别处理）
        E_L_safe = jnp.where(jnp.isfinite(E_L), E_L, 0.0)
        E_L_mean = jnp.sum(E_L_safe * valid_f.reshape(-1, 1, 1), axis=0) / n_valid

        tr_centered = jnp.trace(E_L_safe - E_L_mean, axis1=-2, axis2=-1)
        tr_centered = jnp.where(valid, tr_centered, 0.0)

        w = jax.lax.stop_gradient(tr_centered)             # (N,) complex 权重
        vf_c = valid_f.astype(jnp.complex128)              # (N,)

        def objective(p):
            lp, _ = forward(p, sigma_flat)
            return jnp.sum(jnp.conj(w) * lp * vf_c)        # 复标量

        g = jax.grad(objective, holomorphic=True)(total_params)
        # conj(∂_holo obj) / n_valid == mean_n[ tr_centered_n · conj(∇logΨ_n) ]
        grad = jax.tree.map(
            lambda leaf: jnp.conj(leaf).astype(jnp.complex64) / n_valid,
            g,
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
def make_deflated_qgt_fn(machine_single):
    """基于 deflated machine 的量子几何张量（与 NES_VMC_V1.make_qgt_fn 同构）。

    machine_single: (params, sigma_single) -> scalar logΨ_F（单样本）。
    返回 jitted (params, sigma, diag_shift) -> S + diag_shift·I。
    """
    grad_logpsi = jax.grad(machine_single, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift):
        n_samples = sigma.shape[0]
        grad_tree_batch = vmap_grad_logpsi(params, sigma)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))
        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I

    return qgt_fn


# ============================================================================
# 从 deflated 基重建物理态（Stage 1 收敛后抽取 φ_1 等）
# ============================================================================
def extract_physical_states(chi_full, H_dense):
    """在（已投影的）χ 列张成的子空间上解广义本征问题，返回物理本征态。

    Parameters
    ----------
    chi_full : (16, K) complex   Q_0 ψ 的全空间振幅
    H_dense  : (16, 16) complex128

    Returns
    -------
    phis : list[(16,) complex128]  从低到高排序的归一化物理态
    lams : (K,) float              能量（升序）
    """
    chi = np.asarray(chi_full).astype(np.complex128)
    S = chi.conj().T @ chi
    M = chi.conj().T @ (H_dense @ chi)
    lam, v = scipy_eigh(M + 1e-14 * S, S)   # 加微小位移稳住近奇异的 S
    order = np.argsort(lam.real)
    lam = lam[order]
    v = v[:, order]
    phis = []
    for k in range(v.shape[1]):
        p = chi @ v[:, k]
        n = np.linalg.norm(p)
        if n > 1e-12:
            p = p / n
        phis.append(p)
    return phis, lam.real