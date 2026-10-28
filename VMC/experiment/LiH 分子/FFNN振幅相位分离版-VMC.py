# -*- coding: utf-8 -*-
"""
FFNN 振幅-相位分离版 VMC（共享骨干）—— 脚本版
================================================
与同级目录 `H2 分子/FFNN振幅相位分离版-VMC.py` 同一套算法与流程，
只把数据源换成本目录的 LiH_STO3G_8Qbit.py。

系统：LiH / STO-3G（4 轨道活性空间 CAS(4e,4o)）
    SINGLE_SIZE = hi.size = 8（4 轨道 × 2 自旋）
    物理空间维度 36，FCI E0 = -7.86364558 Ha
"""
import argparse
import logging
import os
import pickle
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import numpy as np
import optax
from jax import flatten_util

jax.config.update("jax_enable_x64", True)

# 从本目录的 LiH 分子数据模块导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from LiH_STO3G_8Qbit import (  # noqa: E402
    SINGLE_SIZE,
    ha,
    hi,
    E_fcis,
    hf_ground_energy,
    single_rule,
)


# ====================== 日志配置 ======================
# 与参考脚本一致：日志/ data / graph 三个同级目录都在脚本所在目录下
time_str = time.strftime("%y-%m-%d-%H-%M")
logger = logging.getLogger("FFNN_amp_phase_VMC_LiH")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()
simple_formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%y-%d-%H-%M")

this_file_dir = os.path.dirname(os.path.abspath(__file__))
log_dir = os.path.join(this_file_dir, "日志")
data_dir = os.path.join(this_file_dir, "data")
graph_dir = os.path.join(this_file_dir, "graph")
os.makedirs(log_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(graph_dir, exist_ok=True)

log_path = f"{log_dir}/{time_str}_ffnn_amp_phase_LiH.log"
file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
file_handler.setFormatter(simple_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)


class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        # 全部网络参数依旧 float 实数，不使用复数权重
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        # ========== 修改点：替换原来 head_logA、head_phi → head_u、head_v 两个实输出头 ==========
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        # 两路实输出 u, v
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        # ========== 修改点：不再 logA + iφ；拼装复数 z = u + iv，返回 log(z) ==========
        # 小epsilon防止u+iv=0造成log奇点
        z = (u + 1e-12) + 1j * (v + 1e-12)
        return jnp.log(z)

# ===================== machine 包装 =====================
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)

    return machine, graphdef, state


# ===================== 局部能量 =====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


# ===================== 每样本梯度：实参数网络，拆实部/虚部求导 =====================
def _grad_logpsi_real_params(machine, params, sigma):
    """
    实参数、复输出的 logψ 梯度：
        ∇ logψ(σ) = ∇ Re logψ(σ) + i · ∇ Im logψ(σ)
    （holomorphic=False 时手写的正确版本，绕开 jax.grad 的实输出限制）
    """
    re_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).real)(params)
    )(sigma)
    im_grad = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s).imag)(params)
    )(sigma)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


# ===================== 重要性加权（配合 machine_pow=1 采样 p ∝ |ψ|） =====================
# 按 |ψ|² 采样时，一旦 |ψ|² 塌缩到单个组态（如 HF），其 w·E_loc·∇logψ 的激发组态贡献
# 永远采不到 → force 梯度恒 0 → 冻结在本征态。改按 |ψ| 采样并对估计器加权
# w(σ) = |ψσ|²/p(σ) ∝ |ψσ|，稀有组态访问率提升数个量级，其 w·E_loc·∇logψ 贡献
# 恰好恢复精确公式（w·E_loc ~ ψη·H 为有限量），这是通用的重要性加权 VMC。

@partial(jax.jit, static_argnames=("machine",))
def _sample_weights(machine, params, sigma):
    """w_i = |ψ(σ_i)|（未归一化），配合 p ∝ |ψ| 的采样分布"""
    logpsi = machine(params, sigma)
    return jnp.exp(jnp.real(logpsi))


def _weighted_mean_std(x, w):
    w_sum = jnp.sum(w)
    mean = jnp.sum(w * x) / w_sum
    var = jnp.sum(w * jnp.abs(x - mean) ** 2) / w_sum
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    重要性加权 force 梯度（实参数网络，取实部）：
        ∇⟨E⟩ = 2·Re⟨(E_loc − ⟨E⟩) (∇log ψ)*⟩_w
    返回的 grad 已含因子 2（即完整 ∇⟨E⟩）。
    """
    w = _sample_weights(machine, params, sigma)
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = _weighted_mean_std(O_loc, w)
    O_centered = O_loc - O_mean

    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)

    def weight_and_mean(grad_component):
        wc = (w * O_centered).reshape((w.shape[0],) + (1,) * (grad_component.ndim - 1))
        # 实参数：取实部并乘因子 2（与参考脚本 2.0 * jnp.real(...) 一致）
        return 2.0 * (jnp.sum(wc * jnp.conj(grad_component), axis=0) / jnp.sum(w)).real

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    return O_mean, O_std, grad


def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    加权 QGT / F 矩阵（实参数网络，取实部）。
    实参数位移 dθ 为实向量时 ⟨|dlogψ|²⟩ = dθᵀ Re(S) dθ，
    虚部是反对称部分、对实二次型无贡献，度量取 Re(S)。
    """
    n_samples = sigma.shape[0]
    w = _sample_weights(machine, params, sigma)
    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    w_mean = jnp.sum(w[:, None] * grad_flat, axis=0, keepdims=True) / jnp.sum(w)
    grad_centered = grad_flat - w_mean
    qw = (w[:, None] * jnp.conj(grad_centered)).T @ grad_centered / jnp.sum(w)
    qgt = qw.real  # 实参数度量
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    return qgt_reg, unravel_fn


# ===================== 只看能量的轻量评估（供回溯线搜索用） =====================
@partial(jax.jit, static_argnames=("machine",))
def energy_only(machine, params, sigma):
    """同一批样本上的重要性加权能量估计（不计算梯度），用于判断试探步是否让能量上升。"""
    w = _sample_weights(machine, params, sigma)
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = _weighted_mean_std(O_loc, w)
    return O_mean, O_std


# ===================== nnx.split 参数的 npz 存取 =====================
# 与参考脚本 NES-VMC K4/H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.py 一致：
# 逐叶子存数组，并保存 __order__（叶子顺序）与 __treedef__（pickle 的结构）以便无损还原。
def save_params_npz(params, npz_path):
    """把 nnx.split 的参数树（State）逐叶子存进 npz，返回叶子数。"""
    leaves_with_path, treedef = jax.tree_util.tree_flatten_with_path(params)
    arrays = {f"p{i:04d}": np.asarray(leaf) for i, (_, leaf) in enumerate(leaves_with_path)}
    arrays["__order__"] = np.array(
        [jax.tree_util.keystr(p) for p, _ in leaves_with_path], dtype=str
    )
    arrays["__treedef__"] = np.frombuffer(pickle.dumps(treedef), dtype=np.uint8)
    np.savez(npz_path, **arrays)
    return len(leaves_with_path)


def load_params_npz(npz_path):
    """与 save_params_npz 配对：读回参数树（State）。读回前务必先开启 x64。"""
    z = np.load(npz_path, allow_pickle=False)
    treedef = pickle.loads(z["__treedef__"].tobytes())
    leaves = [z[f"p{i:04d}"] for i in range(len(z["__order__"]))]
    return jax.tree_util.tree_unflatten(treedef, leaves)


def parse_args():
    parser = argparse.ArgumentParser(
        description="FFNN 振幅-相位分离共享骨干 VMC（LiH / STO-3G）"
    )
    # LiH 比 H2 多 2 个电子（4e/4o），原始配置（lr=0.1 / diag_shift=0.01 / 无裁剪）
    # 会出现大量能量尖峰与长时间平台。实测稳定配置：
    #   diag_shift=0.1 + per_param_clip=0.5 + clip_norm=5 + warmup=100 + 每 10 步 reset
    #   + 回溯线搜索（拦截过冲步）
    # 尖峰只在最初的下降段（前 ~200 步）出现，之后平滑收敛到 ~5e-4 Ha 误差。
    parser.add_argument("--n-iter", type=int, default=2000, help="训练迭代步数")
    parser.add_argument("--seed", type=int, default=21, help="模型初始化随机种子")
    parser.add_argument("--lr", type=float, default=0.05, help="学习率")
    parser.add_argument("--hidden", type=int, default=None,
                        help="骨干隐藏维度（默认 SINGLE_SIZE*2+4）")
    parser.add_argument("--qbits", type=int, default=4, help="共享骨干输出维度")
    parser.add_argument("--diag-shift", type=float, default=0.1, help="QGT 对角正则化")
    parser.add_argument("--n-chains", type=int, default=150, help="并行马尔可夫链数")
    parser.add_argument("--sweep-size", type=int, default=32, help="sweep_size")
    parser.add_argument("--chain-length", type=int, default=20, help="每步采样链长")
    parser.add_argument("--save-interval", type=int, default=20,
                        help="每多少步把历史数据写入 pickle")
    parser.add_argument("--clip-norm", type=float, default=5.0,
                        help="自然梯度全局范数裁剪上限（参考脚本为 20，LiH 上 20 仍会尖峰，实测 5 才稳）")
    parser.add_argument("--per-param-clip", type=float, default=0.5,
                        help="QGT 求解前原始梯度 per-param 范数裁剪上限")
    parser.add_argument("--reset-period", type=int, default=10,
                        help="每多少步 reset 一次采样链（其余步复用链状态连续采样）")
    parser.add_argument("--warmup", type=int, default=100,
                        help="学习率线性 warmup 步数（0 表示不 warmup）")
    parser.add_argument("--lr-end", type=float, default=None,
                        help="余弦衰减终止学习率；给定则整个训练做 warmup+cosine 衰减")
    parser.add_argument("--max-backtracks", type=int, default=4,
                        help="回溯线搜索最多折半次数（0 表示关闭线搜索）")
    parser.add_argument("--out-png", type=str, default=None, help="训练曲线保存路径")
    parser.add_argument("--no-plot", action="store_true", help="不绘制/保存训练曲线")
    return parser.parse_args()


def main():
    args = parse_args()
    hidden_dim = args.hidden if args.hidden is not None else SINGLE_SIZE * 2 + 4

    # 输出文件路径（目录已在模块级日志配置中创建）
    history_file = os.path.join(data_dir, f"{time_str}_history_FFNN_amp_phase_LiH.pkl")
    params_npz = os.path.join(data_dir, f"{time_str}_params_FFNN_amp_phase_LiH.npz")

    model = SingleStateAnsatz_SharedBackbone(
        n_spin_orbitals=SINGLE_SIZE,
        hidden_dim=hidden_dim,
        rngs=nnx.Rngs(args.seed),
    )
    # machine_pow=1: 按 |ψ|（而非 |ψ|²）采样，配合重要性加权估计器，
    # 避免波函数塌缩后梯度信号丢失（见 _sample_weights 注释）
    sampler = nk.sampler.MetropolisSampler(
        hi, rule=single_rule, n_chains=args.n_chains,
        sweep_size=args.sweep_size, machine_pow=1,
    )
    machine, _graphdef, params = create_machine(model)
    sampler_state = sampler.init_state(machine, params, seed=1)

    # 与参考脚本一致：自然梯度全局范数裁剪后再做 SGD；
    # 学习率 warmup（可选 cosine 衰减）抑制初期过冲与后期在极小值附近的震荡尖峰
    if args.lr_end is not None:
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0, peak_value=args.lr, warmup_steps=args.warmup,
            decay_steps=args.n_iter, end_value=args.lr_end,
        )
    elif args.warmup > 0:
        lr_schedule = optax.linear_schedule(
            init_value=0.0, end_value=args.lr, transition_steps=args.warmup
        )
    else:
        lr_schedule = args.lr
    optimizer = optax.chain(
        optax.clip_by_global_norm(args.clip_norm),
        optax.sgd(learning_rate=lr_schedule),
    )
    opt_state = optimizer.init(params)

    history = {
        'step': [],
        'energy': [],
        'energy_std': [],
        'error': [],
        'logpsi_mean': [],
        'logpsi_min': [],
        'logpsi_max': [],
        'grad_norm_raw': [],
        'grad_norm_per_param': [],
        'grad_norm_natural': [],
        'lr_scale': [],
    }

    def save_history():
        # params 只保留最新一份（滚动覆盖），避免 pkl 随步数线性膨胀
        payload = dict(history)
        payload['params'] = [params]
        payload['fci_energy'] = float(E_fcis[0])
        payload['hf_energy'] = float(hf_ground_energy)
        payload['save_interval'] = args.save_interval
        with open(history_file, 'wb') as f:
            pickle.dump(payload, f)

    logger.info("\n" + "=" * 60)
    logger.info("开始纯 JAX VMC 训练 (自然梯度下降法, 重要性加权)")
    logger.info(
        f"超参数配置:\n"
        f"N_ITER = {args.n_iter}\n"
        f"n_chains = {args.n_chains} | sweep_size = {args.sweep_size} | chain_length = {args.chain_length}\n"
        f"machine_pow = 1 (|ψ| 采样 + 重要性加权 w∝|ψ|)\n"
        f"hidden_dim = {hidden_dim} | seed = {args.seed}\n"
        f"lr = {args.lr} | qgt_diag_shift = {args.diag_shift} | SAVE_INTERVAL = {args.save_interval}\n"
        f"clip_norm = {args.clip_norm} | per_param_clip = {args.per_param_clip} "
        f"| RESET_PERIOD = {args.reset_period} | warmup = {args.warmup} | lr_end = {args.lr_end}"
    )
    logger.info(f"FCI 基准：E0 = {E_fcis[0]:.8f} Ha | HF = {hf_ground_energy:.8f} Ha")
    logger.info("=" * 60)

    samples = None
    t0 = time.time()
    for step in range(args.n_iter):
        # 参考脚本做法：每 RESET_PERIOD 步才 reset 一次，其余步复用链状态连续采样，
        # 避免每步 reset 导致链无法热化、样本质量差
        if step % args.reset_period == 0:
            sampler_state = sampler.reset(machine, params, sampler_state)
        samples, sampler_state = sampler.sample(
            machine, params, state=sampler_state, chain_length=args.chain_length
        )
        samples = samples.reshape(-1, hi.size)

        # 1. force-based 能量和梯度（grad 已是完整 ∇⟨E⟩，实数）
        energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)

        # 1.5 梯度/波函数监控量
        grad_flat_raw, grad_unravel_fn = flatten_util.ravel_pytree(grad)
        grad_norm_raw = float(jnp.linalg.norm(grad_flat_raw))
        n_params = grad_flat_raw.size
        grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)

        # 参考脚本做法：per-param 梯度范数超上限时整体等比缩放（发生在 QGT 求解之前），
        # 防止 S^{-1} 把过大的原始梯度放大成踢飞参数的巨步
        clipped_per_param = False
        if grad_norm_per_param > args.per_param_clip:
            grad_scale = args.per_param_clip / grad_norm_per_param
            grad_flat_raw = grad_flat_raw * grad_scale
            grad = grad_unravel_fn(grad_flat_raw)
            grad_norm_raw = float(jnp.linalg.norm(grad_flat_raw))
            grad_norm_per_param = grad_norm_raw / np.sqrt(n_params)
            clipped_per_param = True

        logpsi_vals = jnp.real(machine(params, samples))

        # 2. 自然梯度 = S^{-1} * grad
        qgt_reg, _ = compute_qgt(machine, params, samples, diag_shift=args.diag_shift)
        natural_grad = jnp.linalg.solve(qgt_reg, grad_flat_raw)
        grad_norm_natural = float(jnp.linalg.norm(natural_grad))
        grad = grad_unravel_fn(natural_grad)

        # 3. 更新参数（optimizer 内含全局范数裁剪 clip_norm）
        #    回溯线搜索：若试探步使同一批样本上的能量明显上升（过冲），
        #    把这一步整体折半再试，避免自然梯度大步把波函数“推塌”成近均匀态造成尖峰
        updates, opt_state = optimizer.update(grad, opt_state, params)
        step_scale = 1.0
        n_backtracks = 0
        tol = max(3.0 * float(energy_std), 1e-4)
        e_ref = float(energy.real)
        candidate = params
        while True:
            candidate = optax.apply_updates(
                params, jax.tree_util.tree_map(lambda u: step_scale * u, updates)
            )
            e_try, _ = energy_only(machine, candidate, samples)
            if float(e_try.real) <= e_ref + tol or n_backtracks >= args.max_backtracks:
                break
            step_scale *= 0.5
            n_backtracks += 1
        params = candidate

        # 4. 每一轮都记录历史（能量为该轮采样估计，误差相对 FCI）
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        history['logpsi_mean'].append(float(jnp.mean(logpsi_vals)))
        history['logpsi_min'].append(float(jnp.min(logpsi_vals)))
        history['logpsi_max'].append(float(jnp.max(logpsi_vals)))
        history['grad_norm_raw'].append(grad_norm_raw)
        history['grad_norm_per_param'].append(grad_norm_per_param)
        history['grad_norm_natural'].append(grad_norm_natural)
        history['lr_scale'].append(step_scale)
        logger.info(
            f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} "
            f"| FCI: {E_fcis[0]:.8f} | Error: {error:.6f}"
        )
        logger.info(
            f"梯度监控 | raw={grad_norm_raw:.4f} | per_param={grad_norm_per_param:.4f} "
            f"| natural={grad_norm_natural:.4f} "
            f"| clipped={'per_param' if clipped_per_param else '-'} "
            f"| logΨ mean/min/max = {jnp.mean(logpsi_vals):.3f}/"
            f"{jnp.min(logpsi_vals):.3f}/{jnp.max(logpsi_vals):.3f}"
        )
        if n_backtracks > 0:
            logger.info(f"线搜索 | 过冲拦截，步长缩放 ×{step_scale:.4f}（折半 {n_backtracks} 次）")

        # 5. 每 save_interval 步落盘一次
        if (step + 1) % args.save_interval == 0:
            save_history()
            logger.info(f"[保存] Step {step} → {history_file} ({len(history['step'])} 条记录)")

    # 最终结果
    final_energy, final_std, _ = forces_expect_hermitian(machine, params, samples)
    final_error = jnp.abs(final_energy.real - E_fcis[0])
    logger.info("\n" + "=" * 60)
    logger.info(f"训练完成! 用时 {time.time() - t0:.1f}s")
    logger.info(f"最终能量：{final_energy.real:.8f} ± {final_std:.6f} Ha")
    logger.info(f"FCI 基准：{E_fcis[0]:.8f} Ha")
    logger.info(f"绝对误差：{final_error:.6f} Ha")
    logger.info(f"相对误差：{final_error / jnp.abs(E_fcis[0]) * 100:.4f}%")
    logger.info("=" * 60)

    # 训练结束：最终一次历史落盘 + 参数另存 npz
    save_history()
    n_leaf = save_params_npz(params, params_npz)
    logger.info(f"[保存] 训练结束，pickle 共 {len(history['step'])} 条记录 → {history_file}")
    logger.info(f"[保存] 最终轮次 params 另存 npz（{n_leaf} 个叶子）→ {params_npz}")

    if not args.no_plot:
        import matplotlib.pyplot as plt

        steps = np.asarray(history['step'])
        energy_arr = np.asarray(history['energy'])
        error_arr = np.asarray(history['error'])
        fci_e0 = float(E_fcis[0])

        fig, axs = plt.subplots(4, 1, figsize=(10, 18), sharex=True)

        # 子图 0：能量曲线 + FCI / HF 基准
        axs[0].errorbar(steps, energy_arr, yerr=np.asarray(history['energy_std']),
                        fmt='o-', ms=2.5, lw=1.0, capsize=2, label='VMC energy')
        axs[0].axhline(fci_e0, color='r', ls='--', label=f'FCI E0 = {fci_e0:.6f} Ha')
        axs[0].axhline(hf_ground_energy, color='g', ls='--',
                       label=f'HF energy = {hf_ground_energy:.6f} Ha')
        axs[0].set_ylabel('Energy (Ha)')
        axs[0].set_title('FFNN amplitude-phase VMC (LiH/STO-3G)')
        axs[0].legend(fontsize=8)

        # 子图 1：能级差 |E - E_FCI|（log 轴 + 化学精度带）
        mk = np.isfinite(error_arr) & (error_arr > 0)
        axs[1].semilogy(steps[mk], error_arr[mk], lw=1.2, color='#1f77b4',
                        label='|E - E_FCI|')
        axs[1].axhspan(1e-9, 0.0016, facecolor='lightblue', alpha=0.3, zorder=0,
                       label='chemical accuracy (1.6 mHa)')
        axs[1].set_ylabel('|E - E_FCI| (Ha)')
        axs[1].set_yscale('log')
        axs[1].legend(fontsize=8)

        # 子图 2：log|Ψ| 采样统计
        axs[2].plot(steps, history['logpsi_mean'], lw=1.2, label='mean')
        axs[2].plot(steps, history['logpsi_min'], lw=0.8, ls=':', label='min')
        axs[2].plot(steps, history['logpsi_max'], lw=0.8, ls=':', label='max')
        axs[2].set_ylabel('log|Ψ|')
        axs[2].legend(fontsize=8)

        # 子图 3：grad_norm_per_param 监控（每参数梯度范数）
        axs[3].plot(steps, history['grad_norm_per_param'], lw=1.2, color='#1f77b4',
                    label='grad_norm_per_param')
        axs[3].set_ylabel('grad_norm_per_param')
        axs[3].set_yscale('log')
        axs[3].set_xlabel('iteration')
        axs[3].legend(fontsize=8)

        fig.tight_layout()

        out_png = args.out_png or os.path.join(
            graph_dir, f'FFNN振幅相位分离版-VMC_{time_str}.png'
        )
        fig.savefig(out_png, dpi=150)
        logger.info(f"已保存图: {out_png}")

    return params, history


if __name__ == "__main__":
    main()
