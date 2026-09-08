"""
LiH / STO-3G / K=4 NES-VMC 训练脚本 —— 对照组：完全不进行列规范

与 NES_VMC_train_min120_gauge_reset.py（实验组）的唯一区别：
    1) total machine 直接使用原始 L：既不减参考态（列规范），也没有全局 g；
    2) loss 中 HΨ 侧同样使用原始 single machine（不除以 ψ_j(ref)），
       保证 E_L = Ψ_raw^{-1}·H·Ψ_raw 仍是对角化 Ψ 矩阵的正确形式；
    3) 无 gauge reset 逻辑，无 g 相关监控量。

目的：作为有列规范版本的对照，观察 logΨ 漂移 / 数值稳定性 / 训练曲线的差异。

其余部分（采样器、优化器、超参、pickle 保存、时间前缀日志与图片）与实验组完全一致。

运行方式：
    cd experiments/LiH分子 && python NES_VMC_train_min120_no_gauge_control.py
"""
import logging
import os
import time

import pickle

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from scipy.sparse.linalg import eigsh

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz,
    create_single_machine,
    Ham_Psi_scaled,
    flatten_batched_pytree,
    NESFermionHopRule,
    ravel_pytree,
)


# ====================== 对照组核心：原始 total machine（无任何列规范） ======================
def create_raw_total_machines(total_model):
    """
    与 create_gauge_reset_total_machines 结构一致，但：
        - 不减参考态（无列规范）；
        - 无全局 g 参数；
    所有 jit 函数签名统一为 (params, sigma)。

    返回:
        total_machine, total_matrix_machine, total_max_machine, graphdef, state
    """
    graphdef, state = nnx.split(total_model)

    K_ = total_model.K
    n_spin = total_model.n_spin
    flat_size = K_ * n_spin

    def _compute_L_single(m, x_single):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            cols.append(ansatz_j(x_single))       # 原始输出，不做任何规范
        L = jnp.stack(cols, axis=1)               # (K, K)
        return L

    def _stable_from_L(L):
        shift = jnp.max(jnp.real(L))
        shift = jax.lax.stop_gradient(shift)

        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_det_stable = log_abs_det + 1j * jnp.angle(sign)

        log_det_centered = log_det_stable + K_ * shift

        return log_det_centered, L_stable, shift

    def _one_from_matrix(m, x_single):
        L = _compute_L_single(m, x_single)
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
    def total_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single)[0]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single)[0]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[0])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[0])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_matrix_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single)[1]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[1])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[1])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    @jax.jit
    def total_max_machine(params, sigma):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            x_single = _as_single_matrix(sigma)
            return _one_from_matrix(m, x_single)[2]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                x_single = _as_single_matrix(sigma)
                return _one_from_matrix(m, x_single)[2]
            else:
                x_batch = _as_batch_matrix(sigma)
                return jax.vmap(lambda x: _one_from_matrix(m, x)[2])(x_batch)
        elif sigma.ndim == 3:
            x_batch = _as_batch_matrix(sigma)
            return jax.vmap(lambda x: _one_from_matrix(m, x)[2])(x_batch)
        else:
            raise ValueError(f"Unsupported sigma.shape={sigma.shape}")

    return (
        total_machine,
        total_matrix_machine,
        total_max_machine,
        graphdef,
        state,
    )


# ====================== 对照组 loss：两侧均为原始值（无列规范） ======================
def NES_loss_energy_stable_raw(
    ha,
    total_matrix_machine,
    total_max_machine,
    single_machine_list,
    total_params,
    x,
    return_aux: bool = False,
):
    L_stable = total_matrix_machine(total_params, x)
    shift = total_max_machine(total_params, x)

    Psi_Matrix_stable = jnp.exp(L_stable)

    # HΨ 侧使用原始 single machine（未除以 ψ_j(ref)），与 Ψ 侧保持一致
    HPsi_stable = Ham_Psi_scaled(
        ha=ha,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x,
        shift=shift,
    )

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


def nes_vmc_gradient_stable_raw(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
    total_params,
    x_batch,
    min_valid_ratio: float = 0.25,
    return_aux: bool = False,
):
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable_raw(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_max_machine=total_max_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x=x_batch,
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
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

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


def make_grad_fn_raw(
    ha,
    total_matrix_machine,
    total_max_machine,
    total_machine,
    single_machine_list,
):
    @jax.jit
    def grad_fn(total_params, x_batch):
        return nes_vmc_gradient_stable_raw(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_max_machine=total_max_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
            return_aux=True,
        )

    return grad_fn


def make_qgt_fn_raw(machine):
    """QGT：与实验组一致，仅无显式 gauge 参数 g。"""
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
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


if __name__ == "__main__":

    # ====================== 日志配置 ======================
    time_str = time.strftime("%y-%d-%H-%M")   # 输出文件名统一时间前缀 YY-DD-HH-MM
    logger = logging.getLogger("NES_VMC_K4_no_gauge_control")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_LiH_STO-3G_min120_no_gauge_control.log"
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
    SAVE_INTERVAL = 20         # 每多少步保存/追加一次 pickle
    HISTORY_FILE = "./data/history_natural_gradient_LiH_molecule_K4_no_gauge_control.pkl"
    os.makedirs("./data", exist_ok=True)

    # ====================== 模型与采样器 ======================
    total_ansatz = NESTotalAnsatz(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(11),
    )

    (
        total_machine,
        total_matrix_machine,
        total_max_machine,
        total_graphdef,
        total_params,
    ) = create_raw_total_machines(total_ansatz)

    # 对照组：使用原始 single machine（未除以 ψ_j(ref)），与 total machine 两侧一致
    single_machine_list = [
        create_single_machine(ansatz)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    grad_fn = make_grad_fn_raw(
        ha,
        total_matrix_machine,
        total_max_machine,
        total_machine,
        single_machine_list,
    )
    qgt_fn = make_qgt_fn_raw(total_machine)

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


    def sample_machine(params, sigma):
        """双参数封装：采样只需要 |Ψ|² 的转移率。"""
        return total_machine(params, sigma)


    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    # ====================== 训练循环 ======================
    logger.info("\n" + "=" * 60)
    logger.info("开始多链 NES-VMC 训练 | 对照组：完全不进行列规范")
    logger.info("=" * 60)
    logger.info(
        f"精确CAS基准：基态={eigvals[0]:.8f} Ha | 1激发={eigvals[1]:.8f} Ha"
        f"|2激发={eigvals[2]:.8f} Ha|3激发={eigvals[3]:.8f} Ha|"
    )
    logger.info(f"理论 Loss 上限：{sum(eigvals[0:K]):.8f} ")
    logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}")

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

    start_time = time.time()

    # ====================== 初始化/加载 pickle 历史文件 ======================
    # 先尝试只读探测已有数据，支持断点续训
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

        # 5. 能量监控：E_L 矩阵本征值
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]

        # 6. 波函数与Ψ矩阵条件数监控（原始值）
        log_Psi_batch = total_machine(total_params, x_batch)
        x_single = x_batch[0:1, ...]
        psi_mat = total_matrix_machine(total_params, x_single)[0]
        psi_cond = jnp.linalg.cond(psi_mat)

        # 7. 日志（格式与实验组对齐，便于直接对比）
        logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm})")
        logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
        logger.info(
            f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
            f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
        )
        logger.info("#-----------------------------------------#")

        # 记录曲线数据
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
                "first_step": first_step if first_step is not None else 0,
                "save_interval": SAVE_INTERVAL,
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → pickle 文件已更新 ({len(steps_history)} 条记录)")

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
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    loss_fig_path = _save_curve(
        loss_history,
        "Loss",
        f"NES-VMC LiH K=4 Loss curve (no gauge control, N_ITER={N_ITER})",
        f"./日志/{time_str}_loss_curve_no_gauge_control_min120.png",
    )
    logger.info(f"Loss 曲线已保存: {loss_fig_path}")

    logpsi_fig_path = _save_curve(
        logpsi_history,
        "log Psi mean (Re)",
        f"NES-VMC LiH K=4 logΨ mean curve (no gauge control, N_ITER={N_ITER})",
        f"./日志/{time_str}_logPsi_mean_curve_no_gauge_control_min120.png",
    )
    logger.info(f"logΨ mean 曲线已保存: {logpsi_fig_path}")
