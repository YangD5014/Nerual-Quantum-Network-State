"""
He原子 / cc-pVDZ / K=4 NES-VMC 训练脚本 —— P8（双侧一致列规范 + 周期性 gauge reset）+ 分段冻结

本文件基于：
    - experiments/He原子/NES_VMC_tool_frozen.py （冻结机制：create_gauge_reset_total_machines 的
      frozen_mask 列冻结 + make_qgt_fn_gauge 的 idx_active QGT 裁剪 + build_active_index +
      FreezeController 单向冻结判据）
    - experiments/He原子/He_atom_K4.ipynb （He 原子主训练循环，N_ITER=300）

分段冻结要点（详见 NES_VMC/文档/基于分段冻结的 NES-VMC 训练策略-实施方案.md）：
    1) 冻结列在 L 矩阵输出处施加 stop_gradient —— 反向图中该子网络 VJP 被真正剪掉；
    2) QGT 的 O 矩阵按活跃索引 idx_active 切片，线性求解规模从 P^3 降到 P_a^3；
    3) 采样、能量估计、gauge reset 仍保持 K 列完整（前向物理上必需）。

本脚本每轮记录并保存：
    n_total_params   —— 总变分参数数量 P（恒定）
    n_active_params  —— 本轮实际参与梯度/QGT 的活跃参数数 P_a（随冻结下降）
    qgt_size         —— 本轮 QGT 矩阵尺寸 P_a × P_a
    frozen_set       —— 本轮冻结的能级集合
    各能级误差、梯度范数、QGT 求解耗时等

运行：
    cd experiments/He原子 && /opt/miniconda3/envs/Netket/bin/python He_atom_K4_train_frozen.py
"""
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

from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    flatten_batched_pytree,
    NESFermionHopRule,
    ravel_pytree,
)
from NES_VMC_tool import (
    NES_loss_energy_stable_gauge,
    nes_vmc_gradient_stable_gauge,
    make_grad_fn_gauge,
    make_gauge_fn,
)
from He_ccpvD import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock, hi, E_fcis


# ====================== 冻结机制（源自 NES_VMC_tool_frozen.py）======================
def create_gauge_reset_total_machines(total_model, ref_state, frozen_mask=()):
    """P8 双侧一致列规范 + 周期性 gauge reset + 可选列冻结。

    frozen_mask：长度 K 的 Python tuple（静态，trace 期常量）。冻结列在 L 列输出处施加
    stop_gradient —— 前向不变、反向被剪枝。冻结配置变化时重新调用本工厂（每事件重编译一次）。
    """
    graphdef, state = nnx.split(total_model)

    K_ = total_model.K
    n_spin = total_model.n_spin
    flat_size = K_ * n_spin

    ref_state = jnp.asarray(ref_state)
    frozen_mask = tuple(bool(f) for f in frozen_mask)
    if not frozen_mask:   # 空元组（默认）⇒ 全不冻结
        frozen_mask = tuple(False for _ in range(K_))
    assert len(frozen_mask) == K_, f"frozen_mask 长度 {len(frozen_mask)} != K={K_}"

    def _compute_L_centered_single(m, x_single, g, with_gauge):
        cols = []
        for j in range(K_):
            ansatz_j = m.single_ansatz_list[j]
            log_col = ansatz_j(x_single)
            log_ref = ansatz_j(ref_state)
            log_col_centered = log_col - log_ref
            if frozen_mask[j]:
                # ---- 分段冻结：前向值不变，反向在该列输出处被阻断 ----
                log_col_centered = jax.lax.stop_gradient(log_col_centered)
            cols.append(log_col_centered)
        L_centered = jnp.stack(cols, axis=1)
        if with_gauge:
            L_centered = L_centered - jax.lax.stop_gradient(g)[None, :]
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
            assert sigma.shape[0] == flat_size
            return sigma.reshape(K_, n_spin)
        elif sigma.ndim == 2:
            assert sigma.shape == (K_, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to single walker")

    def _as_batch_matrix(sigma):
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 2:
            assert sigma.shape[-1] == flat_size
            return sigma.reshape(-1, K_, n_spin)
        elif sigma.ndim == 3:
            assert sigma.shape[1:] == (K_, n_spin)
            return sigma
        else:
            raise ValueError(f"Cannot convert sigma.shape={sigma.shape} to batch walkers")

    @jax.jit
    def total_machine(params, sigma, g):
        m = nnx.merge(graphdef, params)
        sigma = jnp.asarray(sigma)
        if sigma.ndim == 1:
            return _one_from_matrix(m, _as_single_matrix(sigma), g)[0]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                return _one_from_matrix(m, _as_single_matrix(sigma), g)[0]
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
            return _one_from_matrix(m, _as_single_matrix(sigma), g)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                return _one_from_matrix(m, _as_single_matrix(sigma), g)[1]
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
            return _one_from_matrix(m, _as_single_matrix(sigma), g)[2]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                return _one_from_matrix(m, _as_single_matrix(sigma), g)[2]
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
            return _one_from_matrix(m, _as_single_matrix(sigma), None, with_gauge=False)[1]
        elif sigma.ndim == 2:
            if sigma.shape == (K_, n_spin):
                return _one_from_matrix(m, _as_single_matrix(sigma), None, with_gauge=False)[1]
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


def make_qgt_fn_gauge_frozen(machine):
    """QGT（冻结版）：O 矩阵按活跃索引 idx_active 切片，冻结列不参与 O^H O 与线性求解。"""
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0, None))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g, idx_active):
        n_samples = sigma.shape[0]
        grad_tree_batch = vmap_grad_logpsi(params, sigma, g)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)   # (n, P)
        O = O[:, idx_active]                                     # (n, P_a)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))
        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I

    return qgt_fn


def build_active_index(total_params, frozen_set):
    """按 ravel_pytree 扁平化顺序，取活跃参数的全局索引（冻结能级参数除外）。

    ravel_pytree 按 tree_flatten 叶子顺序 concat（无重排），flatten_batched_pytree 亦按
    jax.tree.leaves 顺序展平 —— 顺序一致，因此本索引可同时用于 O[:, idx_active] 与
    grad_raw_flat[idx_active] / .at[idx_active].set(...)。返回 jnp int32，shape (P_a,)。
    """
    K_ = len(total_params["single_ansatz_list"])
    sizes = []
    for j in range(K_):
        flat_j, _ = ravel_pytree(total_params["single_ansatz_list"][j])
        sizes.append(int(flat_j.size))

    offsets = np.concatenate([[0], np.cumsum(sizes)])
    full_size = int(ravel_pytree(total_params)[0].size)
    assert offsets[-1] == full_size, \
        f"per-level sizes {sizes} 之和 {offsets[-1]} != 整树 ravel {full_size}"

    active = [j for j in range(K_) if j not in frozen_set]
    if not active:
        return jnp.zeros(0, dtype=jnp.int32)

    idx = np.concatenate([
        np.arange(offsets[j], offsets[j] + sizes[j]) for j in active
    ])
    assert idx.size == full_size - sum(sizes[j] for j in frozen_set)
    return jnp.asarray(idx, dtype=jnp.int32)


class FreezeController:
    """分段冻结控制器（简化判据）：**只要达到化学精度就立即冻结**，无其他判据。

    不再要求持续 N 步、平稳性、采样有效率、能级间隔、预热/冷却——全部去掉。
    冻结是**单向**的：能级一旦达到化学精度被冻结，参数永久不再参与
    梯度/QGT（stop_gradient + 活跃索引切片），**永不激活**。
    """

    def __init__(self, K, eigvals_exact, chem_acc: float = 1.6e-3, frozen_set=None):
        self.K = K
        self.eigvals_exact = np.asarray(eigvals_exact, dtype=np.float64).real
        self.chem_acc = chem_acc
        self.frozen_set = set(frozen_set) if frozen_set else set()
        self.events = []

    @property
    def frozen_mask(self):
        return tuple(j in self.frozen_set for j in range(self.K))

    def update(self, step, eig_vals_re):
        """每步调用。返回冻结集合是否发生变化（触发重建/重编译）。

        判据：仅当 $|E_j - E_j^{FCI}| < chem_acc$（化学精度）即立刻冻结该能级。
        单向冻结：只增不减，永不激活。
        """
        eig_vals_re = np.asarray(eig_vals_re, dtype=np.float64).real
        errors = np.abs(eig_vals_re - self.eigvals_exact)
        changed = False

        for j in range(self.K):
            if j in self.frozen_set:
                continue
            if errors[j] < self.chem_acc:
                self.frozen_set.add(j)
                self.events.append((step, j, "freeze"))
                logger.info(
                    f"[Freeze@{step}] 能级 {j} 冻结 | err={errors[j]:.2e} Ha"
                    f" (< {self.chem_acc:.1e})"
                )
                changed = True

        return changed


# ====================== 主程序 ======================
if __name__ == "__main__":

    # ====================== 日志配置 ======================
    time_str = time.strftime("%y-%m-%d-%H-%M")
    logger = logging.getLogger("NES_VMC_K4_He_frozen")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

    os.makedirs("./日志", exist_ok=True)
    log_path = f"./日志/{time_str}_nes_vmc_K4_He_atom_ccpvD_frozen.log"
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
    N_ITER = 300                       # 轮次（用户指定）
    Natural_Grad = True
    clip_norm = 20.0
    lr = 0.1
    qgt_diag_shift = 0.1
    RESET_PERIOD = 10
    SAVE_INTERVAL = 20
    HISTORY_FILE = f"./data/{time_str}_history_natural_gradient_He_atom_K4_frozen.pkl"
    os.makedirs("./data", exist_ok=True)

    # ---- 分段冻结超参（单向冻结 + 立即冻结判据：仅达化学精度即冻结）----
    CHEM_ACC = 1.6e-3

    # ====================== 模型与采样器 ======================
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(11),
    )

    g_current = jnp.zeros(K, dtype=jnp.complex64)

    # 采样/监控/能量估计用机器：不施加 mask（前向不变，永不重编译）
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
        """按当前冻结配置重建梯度/自然梯度机器（含 mask，冻结事件后重编译一次）。"""
        tm, tmm, tmmax, _, _, _ = create_gauge_reset_total_machines(
            total_ansatz, Hatree_Fock, frozen_mask=frozen_mask
        )
        grad_fn = make_grad_fn_gauge(ha, tmm, tmmax, tm, single_machine_list)
        qgt_fn = make_qgt_fn_gauge_frozen(tm)
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

    # FCI 精确参考能量（He_ccpvD 提供）
    exact_eigvals = np.asarray(E_fcis).real

    # ====================== 分段冻结控制器 ======================
    freeze_ctrl = FreezeController(
        K=K,
        eigvals_exact=exact_eigvals,
        chem_acc=CHEM_ACC,
    )
    idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)   # 初始 = 全参数
    N_TOTAL = int(ravel_pytree(total_params)[0].size)                        # 总参数数 P（恒定）

    # ====================== 优化器 ======================
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=lr),
    )
    opt_state = optimizer.init(total_params)

    sampler_rng = jax.random.PRNGKey(21)


    def sample_machine(params, sigma):
        """双参数封装：采样只需要 |Ψ|² 的转移率，全局列规范 g 是常数平移，不影响 ratio。"""
        return total_machine(params, sigma, g_current)


    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    # ====================== 训练循环 ======================
    logger.info("\n" + "=" * 60)
    logger.info("开始多链 NES-VMC 训练 | P8 双侧列规范 + 周期性 gauge reset + 分段冻结")
    logger.info("=" * 60)
    logger.info(f"精确CAS基准：E0={exact_eigvals[0]:.8f} | E1={exact_eigvals[1]:.8f} | "
                f"E2={exact_eigvals[2]:.8f} | E3={exact_eigvals[3]:.8f} Ha")
    logger.info(f"理论 Loss 上限：{sum(exact_eigvals[0:K]):.8f}")
    logger.info(f"总变分参数 P = {N_TOTAL}（每能级 {N_TOTAL // K}）")
    logger.info(f"超参：clip_norm={clip_norm}, lr={lr}, QGT diag_shift={qgt_diag_shift}, "
                f"RESET_PERIOD={RESET_PERIOD}, N_ITER={N_ITER}")
    logger.info(f"分段冻结（单向 + 立即冻结判据）：CHEM_ACC={CHEM_ACC:.1e} Ha（仅达化学精度即冻结）")

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
    Energy_levels_history = []
    err_history = {j: [] for j in range(K)}
    # ---- 分段冻结指标（每轮保存）----
    n_active_params_history = []     # 本轮实际参与训练（梯度/QGT）的活跃参数数 P_a
    qgt_size_history = []            # 本轮 QGT 矩阵尺寸 P_a × P_a
    frozen_set_history = []          # 本轮冻结能级集合
    frozen_count_history = []        # 本轮已冻结能级数
    t_grad_history = []              # 本轮 grad_fn 耗时 (s)
    t_qgt_history = []               # 本轮 QGT 构建+求解耗时 (s)

    start_time = time.time()

    # ====================== 初始化/加载 pickle 历史文件 ======================
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
            if history.get("frozen_set"):
                restored = set(int(j) for j in history["frozen_set"])
                freeze_ctrl.frozen_set = restored
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

        # 2. 梯度与损失（分段计时）
        t0_grad = time.perf_counter()
        grad_raw, loss_mean, E_L_mean, _ = grad_fn(total_params, x_batch, g_current)
        t_grad = time.perf_counter() - t0_grad

        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)
        grad_update = grad_raw

        has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
        if has_nan or grad_norm_raw > 5000.0:
            logger.warning(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw_grad_norm={grad_norm_raw:.2f}")

        # 3. QGT 自然梯度（活跃子矩阵切片求解）
        #    本轮实际参与训练的活跃参数数 / QGT 尺寸：用本轮 idx_active
        n_active = int(idx_active.size)
        qgt_size = n_active * n_active
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
        eig_vals_re = np.asarray(eig_vals.real)
        errors_j = np.abs(eig_vals_re - exact_eigvals)
        for j in range(K):
            err_history[j].append(float(errors_j[j]))

        # ---- 6. 分段冻结判定：达到化学精度立即冻结（影响下一轮起计算的规模）----
        frozen_before = sorted(freeze_ctrl.frozen_set)
        frozen_changed = freeze_ctrl.update(step, eig_vals_re)
        if frozen_changed:
            t0_fz = time.perf_counter()
            grad_fn, qgt_fn = build_grad_machines(freeze_ctrl.frozen_mask)
            idx_active = build_active_index(total_params, freeze_ctrl.frozen_set)
            t_rebuild = time.perf_counter() - t0_fz
            logger.info(
                f"[重建@{step}] frozen_set={sorted(freeze_ctrl.frozen_set)} "
                f"| P_a/P = {int(idx_active.size)}/{N_TOTAL} "
                f"| 重建耗时 {t_rebuild:.2f}s（含编译，一次性）"
            )

        # 7. 波函数与Ψ矩阵条件数监控（含 gauge 的值）
        log_Psi_batch = total_machine(total_params, x_batch, g_current)
        x_single = x_batch[0:1, ...]
        psi_mat = total_matrix_machine(total_params, x_single, g_current)[0]
        psi_cond = jnp.linalg.cond(psi_mat)

        gauge_now = float(jnp.real(gauge_fn(total_params, x_batch)))
        g_abs = float(jnp.linalg.norm(g_current))

        # 8. 日志（含冻结状态与分段计时）
        logger.info(f"[Step {step:3d}] logΨ: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        logger.info(f"梯度监控 | raw={grad_norm_raw:.4f} | natural={grad_norm_natural:.4f} | clipped={grad_norm_clipped:.4f}(上限{clip_norm}) | t_grad={t_grad*1e3:.0f}ms t_qgt={t_qgt*1e3:.0f}ms")
        logger.info(f"冻结状态 | frozen_set={sorted(freeze_ctrl.frozen_set)} | P_a/P={n_active}/{N_TOTAL} | QGT={qgt_size} | 本轮实际训练参数={n_active}")
        logger.info(f"原始规范坐标 G(Re Σrow_mean) = {gauge_now:+.4f} | 累计|g| = {g_abs:.4f}")
        logger.info(f"Ψ矩阵条件数 cond(Ψ) = {psi_cond:.2e}")
        logger.info(
            f"Loss={loss_mean:.6f} | E0={eig_vals[0]:.8f} | E1={eig_vals[1]:.8f}"
            f"|E2={eig_vals[2]:.8f}|E3={eig_vals[3]:.8f}"
        )
        logger.info(
            f"误差(Ha): " + " | ".join(f"e{j}={errors_j[j]:.2e}" for j in range(K))
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
        Energy_levels_history.append(eig_vals[:K])

        # ---- 每轮保存分段冻结指标（本轮实际使用的 P_a / QGT 尺寸 / 冻结集合）----
        n_active_params_history.append(n_active)
        qgt_size_history.append(qgt_size)
        frozen_set_history.append(frozen_before)          # 本轮实际计算所用冻结集合
        frozen_count_history.append(len(frozen_before))
        t_grad_history.append(t_grad)
        t_qgt_history.append(t_qgt)

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
                "Energy_levels": Energy_levels_history,
                "err": {j: err_history[j] for j in range(K)},
                # ---- 分段冻结指标 ----
                "n_total_params": N_TOTAL,
                "n_active_params": n_active_params_history,
                "qgt_size": qgt_size_history,
                "frozen_set": sorted(freeze_ctrl.frozen_set),
                "frozen_set_history": frozen_set_history,
                "frozen_count": frozen_count_history,
                "freeze_events": freeze_ctrl.events,
                "t_grad": t_grad_history,
                "t_qgt": t_qgt_history,
            }
            with open(HISTORY_FILE, "wb") as f:
                pickle.dump(history, f)
            logger.info(f"[保存] Step {step} → pickle 文件已更新 ({len(steps_history)} 条记录)")

        # 9. 周期性 gauge reset：把 raw L 的列均值吸收进全局 g（冻结列也继续吸收）
        if (step + 1) % RESET_PERIOD == 0:
            new_col_mean = col_mean_fn(total_params, x_batch)
            g_current = new_col_mean
            absorbed = float(jnp.sum(jnp.real(new_col_mean)))
            logger.info(
                f"[GaugeReset@{step + 1}] 吸收 Δg = Σ Re(col_mean) = {absorbed:+.4f} "
                f"| 新|g| = {float(jnp.linalg.norm(g_current)):.4f}"
            )

        # 全部能级冻结：本轮各指标已记录完毕，提前终止
        if len(freeze_ctrl.frozen_set) == K:
            logger.info("全部 K 个能级已冻结（均达到化学精度），训练提前终止")
            break

    end_time = time.time()
    logger.info(f"训练耗时：{end_time - start_time:.2f} 秒")
    logger.info("\n" + "=" * 60)
    logger.info("训练完成!")
    logger.info("=" * 60)

    # ====================== 分段冻结结果汇总 ======================
    logger.info("\n===== 分段冻结结果汇总 =====")
    logger.info(f"冻结事件（step, 能级, 类型）：{freeze_ctrl.events}")
    logger.info(f"最终 frozen_set：{sorted(freeze_ctrl.frozen_set)}")
    logger.info(f"总参数 P = {N_TOTAL} | 初始活跃 P_a = {N_TOTAL} | 最终活跃 P_a = {int(idx_active.size)}")
    logger.info(f"最终 QGT 尺寸 = {int(idx_active.size)} x {int(idx_active.size)}（初始 {N_TOTAL} x {N_TOTAL}）")
    final_err = np.abs(np.asarray(Energy_levels_history[-1]).real - exact_eigvals) * 1e3
    logger.info("最终各能级误差(mHa): " + " | ".join(f"E{j}={final_err[j]:.2f}" for j in range(K)))

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
        "Energy_levels": Energy_levels_history,
        "err": {j: err_history[j] for j in range(K)},
        "n_total_params": N_TOTAL,
        "n_active_params": n_active_params_history,
        "qgt_size": qgt_size_history,
        "frozen_set": sorted(freeze_ctrl.frozen_set),
        "frozen_set_history": frozen_set_history,
        "frozen_count": frozen_count_history,
        "freeze_events": freeze_ctrl.events,
        "t_grad": t_grad_history,
        "t_qgt": t_qgt_history,
    }
    with open(HISTORY_FILE, "wb") as f:
        pickle.dump(history, f)
    logger.info(f"[保存] 训练结束，pickle 最终写入 {len(steps_history)} 条记录 → {HISTORY_FILE}")

    # ====================== 保存曲线图 ======================
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
        f"NES-VMC He K=4 Loss curve (frozen, N_ITER={N_ITER})",
        f"./日志/{time_str}_loss_curve_frozen.png",
    )
    logger.info(f"Loss 曲线已保存: {loss_fig_path}")

    # 活跃参数数 P_a 与 QGT 尺寸随步数（观察分段冻结减负）
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_history, n_active_params_history, marker="o", markersize=3,
            linewidth=1.2, color="#1f77b4", label="P_a (active params)")
    ax.axhline(N_TOTAL, color="gray", linestyle="--", alpha=0.6, label=f"P total = {N_TOTAL}")
    for (s, j, kind) in freeze_ctrl.events:
        ax.axvline(s, color="green" if kind == "freeze" else "orange",
                   linestyle="--", alpha=0.6, label=f"{kind} j={j}@{s}")
    ax.set_xlabel("Step")
    ax.set_ylabel("活跃参数数 P_a")
    ax.set_title("分段冻结：实际参与训练的参数数 P_a 随步数（真减负）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(f"./日志/{time_str}_active_params_curve_frozen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"活跃参数数曲线已保存: ./日志/{time_str}_active_params_curve_frozen.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps_history, qgt_size_history, marker="o", markersize=3,
            linewidth=1.2, color="#ff7f0e", label="QGT size (P_a^2)")
    ax.axhline(N_TOTAL * N_TOTAL, color="gray", linestyle="--", alpha=0.6,
               label=f"P^2 = {N_TOTAL * N_TOTAL}")
    for (s, j, kind) in freeze_ctrl.events:
        ax.axvline(s, color="green" if kind == "freeze" else "orange",
                   linestyle="--", alpha=0.6, label=f"{kind} j={j}@{s}")
    ax.set_xlabel("Step")
    ax.set_ylabel("QGT 矩阵元素数 P_a^2")
    ax.set_title("分段冻结：QGT 矩阵规模随步数")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(f"./日志/{time_str}_qgt_size_curve_frozen.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"QGT 尺寸曲线已保存: ./日志/{time_str}_qgt_size_curve_frozen.png")

    # 各能级误差曲线
    err_fig_path = f"./日志/{time_str}_err_curve_frozen.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    for j in range(K):
        ax.plot(steps_history, err_history[j], marker="o", markersize=2,
                linewidth=1.0, label=f"E{j} err")
    ax.axhline(CHEM_ACC, color="red", linestyle="--", alpha=0.6, label="化学精度 1.6 mHa")
    for (s, j, kind) in freeze_ctrl.events:
        ax.axvline(s, color="green", linestyle="--", alpha=0.6)
    ax.set_xlabel("Step")
    ax.set_ylabel("|E - E_FCI| (Ha)")
    ax.set_yscale("log")
    ax.set_title("NES-VMC He K=4 能级误差曲线（分段冻结）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(err_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"能级误差曲线已保存: {err_fig_path}")
