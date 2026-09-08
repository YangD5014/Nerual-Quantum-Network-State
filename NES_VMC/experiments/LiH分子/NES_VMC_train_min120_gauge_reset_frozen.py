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


if __name__ == "__main__":

    # ====================== 日志配置 ======================
    time_str = time.strftime("%y-%d-%H-%M")   # 输出文件名统一时间前缀 YY-DD-HH-MM
    logger = logging.getLogger("NES_VMC_K4_gauge_reset_frozen")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_LiH_STO-3G_min120_gauge_reset_frozen.log"
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(simple_formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # ====================== 超参配置 ======================
    N_CHAINS = 16
    N_SAMPLES_PER_CHAIN = 200
    SWEEP_SIZE = 10
    N_ITER = 200
    Natural_Grad = True
    clip_norm = 20.0
    lr = 0.1
    qgt_diag_shift = 0.1
    RESET_PERIOD = 10          # 每隔多少步做一次 gauge reset
    SAVE_INTERVAL = 20         # 每多少步保存/追加一次 pickle
    HISTORY_FILE = "./data/history_natural_gradient_LiH_molecule_K4_frozen.pkl"
    os.makedirs("./data", exist_ok=True)

    # ---- 分段冻结超参（对应实施方案 1.2 节）----
    CHEM_ACC = 1.6e-3          # 化学精度 1.6 mHa
    FREEZE_WINDOW = RESET_PERIOD  # 连续达标步数
    UNFREEZE_WINDOW = 5
    WARMUP = 30
    COOLDOWN = FREEZE_WINDOW
    MIN_VALID_RATIO = 0.25

    # ====================== 模型与采样器 ======================
    total_ansatz = NESTotalAnsatz(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(11),
    )

    g_current = jnp.zeros(K, dtype=jnp.complex64)   # 全局列规范 g，非训练参数，动态传入 jit

    # 采样/监控/能量估计用的机器：不施加 mask（前向不变，永不重编译）
    (
        total_machine,
        total_matrix_machine,
        total_max_machine,
        total_matrix_machine_raw,
        total_graphdef,
        total_params,
    ) = create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    def build_grad_machines(frozen_mask):
        """按当前冻结配置重建梯度/自然梯度所用机器（含 mask，冻结事件后重编译一次）。"""
        tm, tmm, tmmax, _, _, _ = create_gauge_reset_total_machines(
            total_ansatz, Hatree_Fock, frozen_mask=frozen_mask
        )
        grad_fn = make_grad_fn_gauge(
            ha, tmm, tmmax, tm, single_machine_list
        )
        qgt_fn = make_qgt_fn_gauge(tm)
        return grad_fn, qgt_fn

    grad_fn, qgt_fn = build_grad_machines(frozen_mask=())

    gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext,
        rule=nes_rule,
        n_chains=N_CHAINS,
        sweep_size=SWEEP_SIZE,
    )

    # FCI 精确参考能量
    eigvals, _ = eigsh(ha.to_sparse(), k=K, which="SA", tol=1e-10)

    # ====================== 分段冻结控制器 ======================
    freeze_ctrl = FreezeController(
        K=K,
        eigvals_exact=eigvals,
        chem_acc=CHEM_ACC,
        freeze_window=FREEZE_WINDOW,
        unfreeze_window=UNFREEZE_WINDOW,
        warmup=WARMUP,
        cooldown=COOLDOWN,
        min_valid_ratio=MIN_VALID_RATIO,
    )
    idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)   # 初始 = 全参数

    # ====================== 优化器 ======================
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=lr),
    )
    opt_state = optimizer.init(total_params)

    sampler_rng = jax.random.PRNGKey(21)


    def sample_machine(params, sigma):
        """双参数封装：采样只需要 |Ψ|² 的转移率，全局列规范 g 是常数平移，不影响 ratio。
        闭包每次调用读取 g_current 的最新值（g 作为动态参数传入 jit）。"""
        return total_machine(params, sigma, g_current)


    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    # ====================== 训练循环 ======================
    logger.info("\n" + "=" * 60)
    logger.info("开始多链 NES-VMC 训练 | P8: 双侧一致列规范 + 周期性 gauge reset + 分段冻结")
    logger.info("=" * 60)
    logger.info(
        f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
        f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
    )
    logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
    logger.info(
        f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}, "
        f"RESET_PERIOD={RESET_PERIOD}"
    )
    logger.info(
        f"分段冻结：CHEM_ACC={CHEM_ACC:.1e} Ha, FREEZE_WINDOW={FREEZE_WINDOW}, "
        f"UNFREEZE_WINDOW={UNFREEZE_WINDOW}, WARMUP={WARMUP}, COOLDOWN={COOLDOWN}"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loss_history = []
    logpsi_history = []
    steps_history = []
    logpsi_mean_history = []
    logpsi_min_history = []
    logpsi_max_history = []
    grad_norm_raw_history = []
    grad_norm_nat_history = []
    E_L_real_history = []
    E_L_imag_history = []
    err_history = {j: [] for j in range(K)}        # 各能级误差随时间
    frozen_history = []                            # 各步已冻结能级数

    start_time = time.time()

    # ====================== 初始化/加载 pickle 历史文件 ======================
    # 先尝试只读探测已有数据，支持断点续训（含冻结状态）
    first_step = None
    last_step = None
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "rb") as f:
                history = pickle.load(f)
            if history.get("steps"):
                last_step = int(history["steps"][-1])
                first_step = int(history["steps"][0])
                logger.info(f"检测到已有历史文件，上次保存 step={last_step}，将从 step={last_step + 1} 继续")
            else:
                logger.info("检测到历史文件但为空，将从头开始")

            # 恢复冻结状态并重建 mask / 活跃索引
            if history.get("frozen_set"):
                restored = set(int(j) for j in history["frozen_set"])
                freeze_ctrl.frozen_set = restored
                freeze_ctrl.last_freeze_step = history.get("last_freeze_step", -10**9)
                logger.info(f"恢复冻结状态：frozen_set={sorted(restored)}")
                grad_fn, qgt_fn = build_grad_machines(freeze_ctrl.frozen_mask)
                idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
        except Exception as e:
            logger.warning(f"读取历史文件失败（可能损坏），将覆盖写入: {e}")
    # ====================== 初始化结束 ======================

    for step in range(N_ITER):
        # 1. 采样
        samples_raw, sampler_state = nes_sampler.sample(
            machine=sample_machine,
            parameters=total_params,
            state=sampler_state,
            chain_length=N_SAMPLES_PER_CHAIN,
        )
        samples = samples_raw.reshape(-1, hi_ext.size)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)

        # 2. 梯度与损失（分段计时，用于验证"真减负"）
        t0_grad = time.perf_counter()
        grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current)
        t_grad = time.perf_counter() - t0_grad

        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
        grad_update = grad_raw

        has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
        if has_nan or grad_norm_raw > 5000.0:
            logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

        # 3. QGT 自然梯度（活跃子矩阵切片求解）
        t0_qgt = time.perf_counter()
        if Natural_Grad:
            qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current, idx_active)
            grad_a = grad_raw_flat[idx_active]                       # (P_a,)
            ng_a = jnp.linalg.solve(qgt_reg_mat, grad_a)             # 仅活跃规模求解
            ng_flat = grad_raw_flat.at[idx_active].set(ng_a)         # 回填，冻结位保持 0
            grad_update = unravel_fn(ng_flat)
            grad_norm_natural = jnp.linalg.norm(ng_flat)
        else:
            grad_norm_natural = grad_norm_raw
        t_qgt = time.perf_counter() - t0_qgt

        # 4. 优化器内部完成梯度裁剪 + SGD 更新（冻结分量梯度为 0 ⇒ 参数逐位不动）
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        grad_norm_clipped = min(float(grad_norm_natural), clip_norm)

        # 5. 能量监控：E_L 矩阵本征值
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

        # ---- 6. 分段冻结判定（新增）----
        eig_vals_re = np.asarray(eig_vals.real)
        errors_j = np.abs(eig_vals_re - np.asarray(eigvals).real)
        for j in range(K):
            err_history[j].append(float(errors_j[j]))

        frozen_changed = freeze_ctrl.update(
            step,
            eig_vals_re,
            valid_ratio=float(aux["valid_ratio"]),
            grad_finite=not has_nan,
        )
        if frozen_changed:
            t0_fz = time.perf_counter()
            grad_fn, qgt_fn = build_grad_machines(freeze_ctrl.frozen_mask)
            idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
            t_rebuild = time.perf_counter() - t0_fz
            logger.info(
                f"[重建@{step}] frozen_set={sorted(freeze_ctrl.frozen_set)} "
                f"| P_a/P = {int(idx_active.size)}/{int(ravel_pytree(total_params)[0].size)} "
                f"| 重建耗时 {t_rebuild:.2f}s（含编译，一次性）"
            )
        frozen_history.append(len(freeze_ctrl.frozen_set))

        if len(freeze_ctrl.frozen_set) == K:
            logger.info("全部 K 个能级已冻结（均达到化学精度），训练提前终止")
            break

        # 7. 波函数与Ψ矩阵条件数监控（含 gauge 的值）
        log_Psi_batch = total_machine(total_params, x_batch, g_current)
        x_single = x_batch[0:1, ...]
        psi_mat = total_matrix_machine(total_params, x_single, g_current)[0]
        psi_cond = jnp.linalg.cond(psi_mat)

        gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))          # RAW L 的行规范坐标
        g_abs = float(jnp.linalg.norm(g_current))

        # 8. 日志（格式与基线对齐，附加冻结信息与分段计时）
        logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm}) | t_grad={t_grad*1e3:.0f}ms t_qgt={t_qgt*1e3:.0f}ms")
        logger.info(f"原始规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | 累计|g| = {g_abs:.4f}")
        logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
        logger.info(
            f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
            f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
        )
        logger.info(
            f"误差(Ha): " + " | ".join(f"e{j}={errors_j[j]:.2e}" for j in range(K))
            + f" | frozen={sorted(freeze_ctrl.frozen_set)}"
        )
        logger.info("#-----------------------------------------#")

        # 记录曲线数据（reset 发生在 step+1 边界，曲线上可直接观察回退）
        loss_history.append(float(jnp.real(loss_mean)))
        logpsi_history.append(float(jnp.real(log_Psi_batch.mean())))
        steps_history.append(step)

        # 记录监控数据（用于 pickle 保存）
        logpsi_mean_history.append(float(jnp.real(log_Psi_batch.mean())))
        logpsi_min_history.append(float(jnp.real(log_Psi_batch.min())))
        logpsi_max_history.append(float(jnp.real(log_Psi_batch.max())))
        grad_norm_raw_history.append(float(grad_norm_raw))
        grad_norm_nat_history.append(float(grad_norm_natural))
        E_L_real_history.append(float(jnp.real(jnp.trace(E_L_mean))))
        E_L_imag_history.append(float(jnp.imag(jnp.trace(E_L_mean))))

        # 每 SAVE_INTERVAL 步保存/追加一次 pickle
        if (step + 1) % SAVE_INTERVAL == 0:
            history = {
                "steps": [int(s) for s in steps_history],
                "logpsi_mean": logpsi_mean_history,
                "logpsi_min": logpsi_min_history,
                "logpsi_max": logpsi_max_history,
                "grad_norm_raw": grad_norm_raw_history,
                "grad_norm_natural": grad_norm_nat_history,
                "E_L_real": E_L_real_history,
                "E_L_imag": E_L_imag_history,
                "loss": loss_history,
                "err": {j: err_history[j] for j in range(K)},
                "frozen": frozen_history,
                "frozen_set": sorted(freeze_ctrl.frozen_set),
                "last_freeze_step": freeze_ctrl.last_freeze_step,
                "first_step": first_step if first_step is not None else 0,
                "save_interval": SAVE_INTERVAL,
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → pickle 文件已更新 ({len(steps_history)} 条记录)")

        # 9. 周期性 gauge reset：把 raw L 的列均值吸收进全局 g（双侧规范，物理不变）
        #    冻结列也必须继续吸收（列均值随采样分布漂移，见实施方案 2.1 表）
        if (step + 1) % RESET_PERIOD == 0:
            new_col_mean = col_mean_fn(total_params, x_batch)   # (K,) 复数：当前列均值
            g_current = new_col_mean                # 模块级变量直接更新
            absorbed = float(jnp.sum(jnp.real(new_col_mean)))
            logger.info(
                f"[GaugeReset@{step + 1}] 吸收 Δg = Σ Re(col_mean) = {absorbed:+.4f} "
                f"| 新|g| = {float(jnp.linalg.norm(g_current)):.4f}"
            )

    end_time = time.time()
    logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

    # 训练结束后做最后一次写入确保数据完整
    history = {
        "steps": [int(s) for s in steps_history],
        "logpsi_mean": logpsi_mean_history,
        "logpsi_min": logpsi_min_history,
        "logpsi_max": logpsi_max_history,
        "grad_norm_raw": grad_norm_raw_history,
        "grad_norm_natural": grad_norm_nat_history,
        "E_L_real": E_L_real_history,
        "E_L_imag": E_L_imag_history,
        "loss": loss_history,
        "err": {j: err_history[j] for j in range(K)},
        "frozen": frozen_history,
        "frozen_set": sorted(freeze_ctrl.frozen_set),
        "last_freeze_step": freeze_ctrl.last_freeze_step,
        "first_step": first_step if first_step is not None else 0,
        "save_interval": SAVE_INTERVAL,
    }
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"[保存] 训练结束，pickle 最终写入 {len(steps_history)} 条记录 → {HISTORY_FILE}")

    # ====================== 保存曲线图：Loss-step 与 logΨ mean-step ======================
    def _save_curve(y_values, ylabel, title, fig_path):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(steps_history, y_values, marker="o", markersize=3, linewidth=1.2)
        for r in range(RESET_PERIOD, N_ITER + 1, RESET_PERIOD):
            ax.axvline(r - 0.5, color="red", linestyle="--", alpha=0.6, label="gauge reset")
        for (s, j, kind) in freeze_ctrl.events:
            ax.axvline(s, color="green" if kind == "freeze" else "orange",
                       linestyle="--", alpha=0.6,
                       label=f"{kind} j={j}@{s}")
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    loss_fig_path = _save_curve(
        loss_history,
        "Loss",
        f"NES-VMC LiH K=4 Loss curve (frozen strategy, N_ITER={N_ITER})",
        f"./日志/{time_str}_loss_curve_gauge_reset_frozen.png",
    )
    logger.info(f"Loss 曲线已保存: {loss_fig_path}")

    logpsi_fig_path = _save_curve(
        logpsi_history,
        "log Psi mean (Re)",
        f"NES-VMC LiH K=4 logΨ mean curve (frozen strategy, N_ITER={N_ITER})",
        f"./日志/{time_str}_logPsi_mean_curve_gauge_reset_frozen.png",
    )
    logger.info(f"logΨ mean 曲线已保存: {logpsi_fig_path}")

    # 各能级误差随步数的曲线（观察冻结事件的减负与漂移回退）
    err_fig_path = "./日志/" + time_str + "_err_curve_gauge_reset_frozen.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(K):
        ax.plot(steps_history, err_history[j], marker="o", markersize=2, linewidth=1.0, label=f"E{j} err")
    ax.axhline(CHEM_ACC, color="red", linestyle="--", alpha=0.6, label="化学精度")
    ax.axhline(2 * CHEM_ACC, color="orange", linestyle="--", alpha=0.6, label="2×化学精度(回退阈值)")
    for (s, j, kind) in freeze_ctrl.events:
        ax.axvline(s, color="green" if kind == "freeze" else "orange",
                   linestyle="--", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("|E - E_exact| (Ha)")
    ax.set_yscale("log")
    ax.set_title("NES-VMC LiH K=4 能级误差曲线（分段冻结）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(err_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"能级误差曲线已保存: {err_fig_path}")
