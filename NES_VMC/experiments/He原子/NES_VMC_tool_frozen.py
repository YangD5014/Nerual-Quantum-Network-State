"""
LiH / STO-3G / K=4 NES-VMC 最小训练脚本——P8（双侧一致列规范 + 周期性 gauge reset）+ 分段冻结

本文件是 NES_VMC_train_min120_gauge_reset copy.py 的"分段冻结"扩展：
    在 P8 基础上，当某个能级 j 达到化学精度（< 1.6 mHa）并稳定后，
    将子 ansatz single_ansatz_list[j] 的参数冻结：
        1) 冻结列在 L 矩阵中施加 stop_gradient —— 反向图中该子网络 VJP 被剪掉（真减负）；
        2) QGT 的 O 矩阵按活跃索引切片，线性求解规模从 P^3 降到 P_a^3；
        3) 采样、能量估计、gauge reset 全部保持 K 列完整（前向物理上必需，不动）。

设计文档：NES_VMC/文档/基于分段冻结的 NES-VMC 训练策略-实施方案.md

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min120_gauge_reset_frozen.py
"""
import logging
import os
import time
from collections import deque

import pickle

import numpy as np

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from scipy.sparse.linalg import eigsh

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    NESTotalAnsatz,
    create_single_machine_gauge_fixed,
    Ham_Psi_scaled,
    flatten_batched_pytree,
    NESFermionHopRule,
    ravel_pytree,
)


# ====================== P8 核心：双侧一致列规范 + gauge 可重置 + 可选列冻结 ======================
def create_gauge_reset_total_machines(total_model, ref_state, frozen_mask=()):
    """
    与 NES_VMC_V1.create_gauge_fixed_total_machines 完全一致，区别：
        1) _compute_L_centered_single 末尾追加 L_centered -= sg(g)（按状态=列）；
        2) 新增 total_matrix_machine_raw：不施加 gauge 的原始 L，供 reset 边界测量漂移；
        3) 【分段冻结】frozen_mask: 长度 K 的 Python tuple（静态，trace 期常量），
           冻结列在 L 列输出处施加 stop_gradient，前向不变、反向被剪枝。

    g 作为**显式动态参数**传入所有 jit 函数（避免 JAX jit 闭包捕获 list 的陷阱：
    闭包中 gauge_holder[0] 只在首次编译时被固化，Python 侧更新永远进不了计算图）。

    frozen_mask 说明：它作为闭包 Python 常量被捕获，jax.jit 编译时视为静态量；
    冻结配置变化时**重新调用本工厂**（重新 trace/compile），成本为每事件一次。

    返回:
        total_machine, total_matrix_machine, total_max_machine, total_matrix_machine_raw,
        graphdef, state
    """
    graphdef, state = nnx.split(total_model)

    K_ = total_model.K
    n_spin = total_model.n_spin
    flat_size = K_ * n_spin

    ref_state = jnp.asarray(ref_state)

    # frozen_mask 归一化为布尔 tuple（长度 K_），静态 Python 值
    frozen_mask = tuple(bool(f) for f in frozen_mask)
    assert len(frozen_mask) == K_, f"frozen_mask 长度 {len(frozen_mask)} != K={K_}"

    def _compute_L_centered_single(m, x_single, g, with_gauge):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            log_col = ansatz_j(x_single)          # (K,)
            log_ref = ansatz_j(ref_state)         # scalar
            log_col_centered = log_col - log_ref  # 列规范（与基线一致，不停梯度）
            if frozen_mask[j]:
                # ---- 分段冻结：前向值不变，反向在该列输出处被阻断 ----
                log_col_centered = jax.lax.stop_gradient(log_col_centered)
            cols.append(log_col_centered)
        L_centered = jnp.stack(cols, axis=1)      # (K, K)

        if with_gauge:
            # ---- P8：减去全局列规范 g（stop_gradient，动态参数，非训练参数）----
            L_centered = L_centered - jax.lax.stop_gradient(g)[None, :]
            # --------------------------------------------------------------------------
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
    """QGT：与基线 make_qgt_fn 一致，区别：
    1) 多一个显式 gauge 参数 g（动态传入）；
    2) 多一个活跃索引 idx_active（动态 int 数组）：O 矩阵按活跃列切片，
       冻结列参数不参与 O^H O 与线性求解 —— 这是"真减负"的关键（P^2→P_a^2, P^3→P_a^3）。"""
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0, None))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g, idx_active):
        n_samples = sigma.shape[0]

        grad_tree_batch = vmap_grad_logpsi(params, sigma, g)

        O = flatten_batched_pytree(grad_tree_batch, n_samples)   # (n, P)

        O = O[:, idx_active]                                     # (n, P_a) 冻结列裁剪

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
        """raw L 的列均值向量 (K,)，复数：供 reset 边界做 gauge 吸收。
        注意：冻结列也必须继续吸收（活跃参数会改变采样分布，冻结列 raw L 均值随分布漂移）。"""
        m = nnx.merge(graphdef, params)

        def _offset(x_single):
            cols = []
            for j in range(K_):
                ansatz_j = m.single_ansatz_list[j]
                cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
            return jnp.mean(jnp.stack(cols, axis=1), axis=0)  # (K,)

        return jnp.mean(jax.vmap(_offset)(x_batch), axis=0)

    return gauge_fn, col_mean_fn


# ====================== 【分段冻结】活跃索引构建 ======================
def build_active_index(total_params, frozen_set):
    """按 ravel_pytree 扁平化顺序，取活跃参数的全局索引（冻结能级参数除外）。

    ravel_pytree 按 tree_flatten 叶子顺序 concat（无重排），flatten_batched_pytree
    亦按 jax.tree.leaves 顺序展平 —— 两者顺序一致，因此本索引可同时用于：
        - O[:, idx_active]   （QGT 列裁剪，flatten_batched_pytree 的列序）
        - grad_raw_flat[idx_active] 与 .at[idx_active].set(...)（ravel_pytree 的 flat 序）

    返回: jnp int32 数组，shape (P_a,)
    """
    K_ = len(total_params["single_ansatz_list"])
    sizes = []
    for j in range(K_):
        flat_j, _ = ravel_pytree(total_params["single_ansatz_list"][j])
        sizes.append(int(flat_j.size))

    offsets = np.concatenate([[0], np.cumsum(sizes)])

    # 防御性校验：分块尺寸之和必须等于整树 ravel 长度（保证顺序一致）
    full_size = int(ravel_pytree(total_params)[0].size)
    assert offsets[-1] == full_size, \
        f"per-level sizes {sizes} 之和 {offsets[-1]} != 整树 ravel {full_size}"

    active = [j for j in range(K_) if j not in frozen_set]
    if not active:
        # 全部能级冻结（P_a = 0）：返回空索引（训练循环随即 break）
        return jnp.zeros(0, dtype=jnp.int32)

    idx = np.concatenate([
        np.arange(offsets[j], offsets[j] + sizes[j]) for j in active
    ])
    assert idx.size == full_size - sum(sizes[j] for j in frozen_set)

    return jnp.asarray(idx, dtype=jnp.int32)


# ====================== 【分段冻结】冻结控制器 ======================
class FreezeController:
    """分段冻结控制器：判据（精度+持续+平稳+数值健康+预热/冷却）与漂移回退（迟滞）。

    设计依据：NES_VMC/文档/基于分段冻结的 NES-VMC 训练策略-实施方案.md 第 1 节。
    """

    def __init__(
        self,
        K,
        eigvals_exact,
        chem_acc: float = 1.6e-3,
        freeze_window: int = 10,
        unfreeze_window: int = 5,
        warmup: int = 30,
        cooldown: int = 10,
        min_valid_ratio: float = 0.25,
        frozen_set=None,
        last_freeze_step=None,
    ):
        self.K = K
        self.eigvals_exact = np.asarray(eigvals_exact, dtype=np.float64).real
        self.chem_acc = chem_acc
        self.freeze_window = freeze_window
        self.unfreeze_window = unfreeze_window
        self.warmup = warmup
        self.cooldown = cooldown
        self.min_valid_ratio = min_valid_ratio

        self.frozen_set = set(frozen_set) if frozen_set else set()
        self.last_freeze_step = (
            last_freeze_step if last_freeze_step is not None else -10**9
        )
        self.err_hist = {j: deque(maxlen=freeze_window) for j in range(K)}
        self.unfreeze_hist = {j: deque(maxlen=unfreeze_window) for j in range(K)}
        self.events = []          # (step, j, 'freeze'|'unfreeze')

    @property
    def frozen_mask(self):
        return tuple(j in self.frozen_set for j in range(self.K))

    def update(self, step, eig_vals_re, valid_ratio, grad_finite):
        """每步调用。返回冻结集合是否发生变化（触发重建/重编译）。"""
        eig_vals_re = np.asarray(eig_vals_re, dtype=np.float64).real
        errors = np.abs(eig_vals_re - self.eigvals_exact)
        changed = False

        # ---- 回退检查：冻结后漂移超 2×化学精度，连续 unfreeze_window 步则解冻 ----
        for j in list(self.frozen_set):
            self.unfreeze_hist[j].append(float(errors[j]))
            if (
                len(self.unfreeze_hist[j]) == self.unfreeze_window
                and all(e > 2.0 * self.chem_acc for e in self.unfreeze_hist[j])
            ):
                self.frozen_set.discard(j)
                self.unfreeze_hist[j].clear()
                self.events.append((step, j, "unfreeze"))
                logger.info(f"[Unfreeze@{step}] 能级 {j} 漂移超 2×化学精度，解冻回退")
                changed = True

        # ---- 冻结检查：预热已过 + 冷却期已过 + 梯度健康 ----
        if (
            step >= self.warmup
            and (step - self.last_freeze_step) >= self.cooldown
            and grad_finite
        ):
            for j in range(self.K):
                if j in self.frozen_set:
                    continue
                self.err_hist[j].append(float(errors[j]))
                e = list(self.err_hist[j])
                if len(e) == self.freeze_window and all(v < self.chem_acc for v in e):
                    # C3 平稳性 / C4 数值健康 / C6 近简并保护
                    stable = float(np.std(e)) < 0.5 * self.chem_acc
                    healthy = valid_ratio >= self.min_valid_ratio
                    gap_left = (
                        j == 0
                        or abs(eig_vals_re[j] - eig_vals_re[j - 1]) > 2.0 * self.chem_acc
                    )
                    gap_right = (
                        j == self.K - 1
                        or abs(eig_vals_re[j + 1] - eig_vals_re[j]) > 2.0 * self.chem_acc
                    )
                    if stable and healthy and gap_left and gap_right:
                        self.frozen_set.add(j)
                        self.err_hist[j].clear()
                        self.last_freeze_step = step
                        self.events.append((step, j, "freeze"))
                        logger.info(
                            f"[Freeze@{step}] 能级 {j} 冻结 | err={errors[j]:.2e} Ha"
                        )
                        changed = True
                        break   # 每次只冻结一个，冷却期内观察影响

        return changed

