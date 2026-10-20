"""QE-NES-VMC：量子增强 NES-VMC 训练脚本（H2/6-31G，K=4，方案 A）。

按《Quantum_Enhanced_NES_VMC方案总结.md》实现实验路线 §12-2：
「QE-NES-VMC：NN–PQC 混合 Ansatz，保留原采样策略」。

组成：
  1. 单态混合拟设 ψ_i = φ_i^NN × φ_i^QC（qe_nes_ansatz.py：FFNN 振幅 + PQC 相位）；
  2. 联合行列式 D(X)=det[ψ_i(x_j)]，复用 NES_VMC_tool 双侧一致列规范 + gauge reset；
  3. 采样保留原策略：hi_ext（4×8=32 bit）上 NESFermionHopRule Metropolis，
     machine_pow=1 按 |D| 采样 + 重要性加权 w∝|D|（防零方差冻结）；
  4. 实参数 SR：Wirtinger 拆 ∇Re/∇Im，力梯度带因子 2，度量 Re(S)+diag_shift。

与 K4 RealFFNN 基线（H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.ipynb）的差异仅在
单态拟设：logA+iφ 的 φ 由经典线性头换成 PQC；梯度/QGT/loss/采样/gauge 全同。

用法：
    /opt/miniconda3/envs/Netket/bin/python H2_631G_K4_QE_NES_VMC.py \
        [learning_rate=0.1] [n_iter=300] [n_chains=16] [samples_per_chain=200] [tag=...]

产物：日志、训练曲线 PNG、history.pkl、收敛参数 npz（同目录）。
"""

import os
import sys
import time
import pickle
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, ".."))

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import flax.nnx as nnx
import netket as nk
import optax
from jax.flatten_util import ravel_pytree

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    Ham_Psi_scaled,
    NESFermionHopRule,
    flatten_batched_pytree,
)
from NES_VMC_tool import (
    create_gauge_reset_total_machines,
    NES_loss_energy_stable_gauge,
    make_gauge_fn,
)
from H2_6_31G_8Qbit_K4 import (
    SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, E_fcis, Hatree_Fock,
)
from qe_nes_ansatz import QETotalAnsatz

jax.config.update("jax_enable_x64", True)


# ============================================================
# 实参数 / 复输出 logΨ 的梯度兼容层（与 K4 RealFFNN notebook 相同）
#   ∇logΨ = ∇Re + i·∇Im（Wirtinger 手写，绕开 holomorphic 限制）
#   力梯度  ∇⟨L⟩ = 2·Re⟨ (trE_L − ⟨trE_L⟩)(∇logΨ)* ⟩_w
#   QGT     度量取 Re(S)；均值全部用重要性权重 w∝|Ψ|（machine_pow=1 采样）
# ============================================================
def _grad_logpsi_real_single(total_machine, params, x_single, g):
    re_grad = jax.grad(lambda p: jnp.real(total_machine(p, x_single, g)))(params)
    im_grad = jax.grad(lambda p: jnp.imag(total_machine(p, x_single, g)))(params)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def _importance_weights(machine, params, x_batch, g, valid):
    """w_i ∝ |Ψ(σ_i)|（valid 上归一化），配合 p ∝ |Ψ| 采样（machine_pow=1）。"""
    log_psi = machine(params, x_batch, g)
    log_w = jnp.real(log_psi)
    valid_f = jnp.asarray(valid, dtype=jnp.float64)
    log_w_max = jnp.max(jnp.where(valid_f > 0, log_w, -jnp.inf))
    log_w_max = jax.lax.stop_gradient(jnp.where(jnp.isfinite(log_w_max), log_w_max, 0.0))
    w = jnp.exp(log_w - log_w_max)
    wv = w * valid_f
    return wv / jnp.maximum(jnp.sum(wv), 1e-12)


def nes_vmc_gradient_stable_gauge_real(
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

    wv = _importance_weights(total_machine, total_params, x_batch, g, valid)

    E_L_safe = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0)
    E_L_mean = jnp.sum(E_L_safe * wv.reshape((-1,) + (1,) * (E_L_safe.ndim - 1)), axis=0)

    E_L_centered = E_L_safe - E_L_mean
    tr_centered = jnp.trace(E_L_centered, axis1=-2, axis2=-1)
    tr_centered = jnp.where(valid, tr_centered, 0.0)

    dlogPsi_batch = jax.vmap(
        lambda x: _grad_logpsi_real_single(total_machine, total_params, x, g)
    )(x_batch)

    def weight_and_weighted_mean(grad_component):
        grad_safe = jnp.where(jnp.isfinite(grad_component), grad_component, 0.0)
        weights = (wv * tr_centered).reshape((-1,) + (1,) * (grad_component.ndim - 1))
        weighted = weights * jnp.conj(grad_safe)
        return 2.0 * jnp.real(jnp.sum(weighted, axis=0))

    grad = jax.tree.map(weight_and_weighted_mean, dlogPsi_batch)

    loss_safe = jnp.where(jnp.isfinite(loss_batch), loss_batch, 0.0)
    loss_mean = jnp.sum(wv * loss_safe)

    # 重要性采样健康度（方案总结 §7）：ESS 与最大权重
    ess = 1.0 / jnp.maximum(jnp.sum(wv ** 2), 1e-30)
    w_max = jnp.max(wv) * batch_size

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
        "ess": ess,
        "w_max": w_max,
        "loss_aux": loss_aux,
    }

    if return_aux:
        return grad, loss_mean, E_L_mean, aux
    return grad, loss_mean, E_L_mean


def make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list):
    @jax.jit
    def grad_fn(total_params, x_batch, g):
        return nes_vmc_gradient_stable_gauge_real(
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


def make_qgt_fn_gauge_real(machine):
    """实参数版 QGT：度量 Re(S)，均值/协方差用重要性权重 w∝|Ψ|。"""

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g):
        n_samples = sigma.shape[0]
        grad_tree_batch = jax.vmap(
            lambda s: _grad_logpsi_real_single(machine, params, s, g)
        )(sigma)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)   # (N, P) 复
        finite_row = jnp.all(jnp.isfinite(O), axis=1)
        wv = _importance_weights(machine, params, sigma, g, finite_row)
        O_safe = jnp.where(jnp.isfinite(O), O, 0.0)
        O_mean = jnp.sum(wv[:, None] * O_safe, axis=0, keepdims=True)
        O_centered = O_safe - O_mean
        S = (jnp.conj(O_centered) * wv[:, None]).T @ O_centered
        S = 0.5 * (S + jnp.conj(S.T))
        S_real = jnp.real(S)
        return S_real + diag_shift * jnp.eye(S_real.shape[0], dtype=S_real.dtype)

    return qgt_fn


# ============================================================
# 训练主函数
# ============================================================
def main(learning_rate=0.1, diag_shift=0.1, seed=11, n_iter=300,
         n_chains=16, samples_per_chain=200, sweep_size=20,
         hidden=12, layers=2, pqc_scale=0.1,
         clip_norm=20.0, per_param_clip=0.5, reset_period=10,
         tag="", log_every=10):

    time_str = time.strftime("%y-%m-%d-%H-%M")
    print("=" * 64)
    print(f"QE-NES-VMC 训练开始 | H2/6-31G K={K} | FFNN振幅×PQC相位（方案A）")
    print(f"lr={learning_rate} diag_shift={diag_shift} seed={seed} n_iter={n_iter} "
          f"n_chains={n_chains}×{samples_per_chain} hidden={hidden} layers={layers} tag={tag}")
    print("=" * 64, flush=True)

    # ---------- 模型 / 机器 ----------
    total_ansatz = QETotalAnsatz(
        n_spin_orbitals=SINGLE_SIZE, n_states=K, hidden_dim=hidden,
        n_layers=layers, pqc_scale=pqc_scale, rngs=nnx.Rngs(seed),
    )
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(
        nnx.split(total_ansatz)[1]))
    print(f"总参数量：{n_params}（K={K} 个独立 FFNN+PQC 单态）", flush=True)

    g_current = jnp.zeros(K, dtype=jnp.complex128)

    (total_machine, total_matrix_machine, total_max_machine,
     total_matrix_machine_raw, total_graphdef, total_params) = \
        create_gauge_reset_total_machines(total_ansatz, Hatree_Fock)

    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    grad_fn = make_grad_fn_gauge_real(
        ha, total_matrix_machine, total_max_machine, total_machine, single_machine_list)
    qgt_fn = make_qgt_fn_gauge_real(total_machine)
    gauge_fn, col_mean_fn = make_gauge_fn(total_ansatz, Hatree_Fock)

    # ---------- 采样器（保留原策略）----------
    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    nes_sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext, rule=nes_rule,
        n_chains=n_chains, sweep_size=sweep_size, machine_pow=1,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.sgd(learning_rate=learning_rate),
    )
    opt_state = optimizer.init(total_params)

    sampler_rng = jax.random.PRNGKey(21)

    def sample_machine(params, sigma):
        return total_machine(params, sigma, g_current)

    sampler_state = nes_sampler.init_state(sample_machine, total_params, sampler_rng)

    print(f"精确 CAS 基准：E0={E_fcis[0]:.8f} E1={E_fcis[1]:.8f} "
          f"E2={E_fcis[2]:.8f} E3={E_fcis[3]:.8f} | Loss 理论值 ΣE={sum(E_fcis[:K]):.8f}",
          flush=True)

    # ---------- 历史 ----------
    hist = {k: [] for k in ["step", "loss", "E_levels", "grad_norm_raw",
                            "grad_norm_nat", "valid_ratio", "ess", "w_max",
                            "cond_Psi", "gauge_abs", "time"]}
    flat_ref, unravel = ravel_pytree(total_params)

    t0 = time.time()
    for step in range(n_iter):
        # 1. 采样
        samples_raw, sampler_state = nes_sampler.sample(
            machine=sample_machine, parameters=total_params,
            state=sampler_state, chain_length=samples_per_chain,
        )
        samples = samples_raw.reshape(-1, hi_ext.size)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)

        # 2. 梯度
        grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch, g_current)

        grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
        grad_norm_raw = jnp.linalg.norm(grad_raw_flat)

        # per-param 裁剪（QGT 之前）
        if float(grad_norm_raw) / np.sqrt(grad_raw_flat.size) > per_param_clip:
            scale = per_param_clip * np.sqrt(grad_raw_flat.size) / float(grad_norm_raw)
            grad_raw_flat = grad_raw_flat * scale
            grad_raw = unravel_fn(grad_raw_flat)

        has_nan = bool(jnp.any(jnp.isnan(grad_raw_flat)))
        if has_nan or float(grad_norm_raw) > 5000.0:
            print(f"【Step {step} 告警】梯度异常！nan={has_nan}, raw={float(grad_norm_raw):.2f}", flush=True)

        # 3. SR 自然梯度
        qgt_reg = qgt_fn(total_params, x_batch, diag_shift, g_current)
        ng_flat = jnp.linalg.solve(qgt_reg, ravel_pytree(grad_raw)[0])
        grad_update = unravel_fn(ng_flat)
        grad_norm_nat = float(jnp.linalg.norm(ng_flat))

        # 4. 更新
        updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        # 5. 监控
        eig_vals, _ = jnp.linalg.eig(E_L_mean)
        eig_vals = eig_vals[jnp.argsort(eig_vals.real)]
        log_Psi_batch = total_machine(total_params, x_batch, g_current)
        psi_mat = total_matrix_machine(total_params, x_batch[0:1], g_current)[0]
        psi_cond = float(jnp.linalg.cond(psi_mat))

        if step % log_every == 0 or step == n_iter - 1:
            errs = [float(abs(eig_vals[j].real - E_fcis[j]) * 1000) for j in range(K)]
            print(f"[Step {step:3d}] Loss={float(loss_mean):.6f} "
                  f"| E0..E3 err(mHa)={errs[0]:.2f}/{errs[1]:.2f}/{errs[2]:.2f}/{errs[3]:.2f} "
                  f"| grad raw={float(grad_norm_raw):.3f} nat={grad_norm_nat:.3f} "
                  f"| valid={float(aux['valid_ratio']):.2f} ESS={float(aux['ess']):.0f}/{x_batch.shape[0]} "
                  f"maxw/avg={float(aux['w_max']):.2f} cond(Ψ)={psi_cond:.1e} "
                  f"|g|={float(jnp.linalg.norm(g_current)):.3f}", flush=True)

        for k, v in [("step", step), ("loss", float(loss_mean)),
                     ("E_levels", np.array([ev.real for ev in eig_vals[:K]])),
                     ("grad_norm_raw", float(grad_norm_raw)),
                     ("grad_norm_nat", grad_norm_nat),
                     ("valid_ratio", float(aux["valid_ratio"])),
                     ("ess", float(aux["ess"])), ("w_max", float(aux["w_max"])),
                     ("cond_Psi", psi_cond),
                     ("gauge_abs", float(jnp.linalg.norm(g_current))),
                     ("time", time.time() - t0)]:
            hist[k].append(v)

        # 6. 周期性 gauge reset
        if (step + 1) % reset_period == 0:
            g_current = col_mean_fn(total_params, x_batch)

    dur = time.time() - t0
    print("=" * 64)
    print(f"训练完成 用时 {dur:.1f}s（{dur/max(n_iter,1):.2f}s/步）")
    final_errs = [abs(hist["E_levels"][-1][j] - E_fcis[j]) * 1000 for j in range(K)]
    for j in range(K):
        print(f"  E{j}: {hist['E_levels'][-1][j]:.8f} Ha  误差 {final_errs[j]:.3f} mHa "
              f"(FCI {E_fcis[j]:.8f})")
    print(f"  最终 Loss={hist['loss'][-1]:.6f}  Σ误差={sum(final_errs):.3f} mHa")
    print("=" * 64, flush=True)

    # ---------- 保存产物 ----------
    out_png = os.path.join(HERE, f"QE_NES_VMC_K{K}{tag}.png")
    out_pkl = os.path.join(HERE, f"QE_NES_VMC_K{K}{tag}_history.pkl")
    out_npz = os.path.join(HERE, f"QE_NES_VMC_K{K}{tag}_params.npz")

    fig, axs = plt.subplots(3, 1, figsize=(10, 13))
    axs[0].plot(hist["step"], np.array(hist["E_levels"]))
    for e in E_fcis[:K]:
        axs[0].axhline(float(e), color="gray", ls="--", alpha=0.4)
    axs[0].set_ylabel("Energy (Ha)")
    axs[0].set_title(f"QE-NES-VMC (FFNN amp x PQC phase) K={K} H2/6-31G {tag}")
    axs[1].plot(hist["step"], hist["loss"])
    axs[1].set_ylabel("Loss")
    axs[1].axhline(float(sum(E_fcis[:K])), color="r", ls="--", alpha=0.5)
    for j in range(K):
        axs[2].plot(hist["step"], np.array(hist["E_levels"])[:, j] - E_fcis[j], label=f"E{j} err")
    axs[2].set_ylabel("Energy error (Ha)")
    axs[2].set_xlabel("step")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"曲线已保存：{out_png}")

    with open(out_pkl, "wb") as f:
        pickle.dump(hist, f)
    flat, _ = ravel_pytree(total_params)
    np.savez(out_npz, flat=np.asarray(flat))
    print(f"历史/参数已保存：{out_pkl} / {out_npz}")
    return total_ansatz, total_machine, total_params, hist


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        if k in ("tag",):
            kw[k] = v
        elif k in ("hidden", "layers", "n_iter", "n_chains", "samples_per_chain",
                   "sweep_size", "seed", "reset_period", "log_every"):
            kw[k] = int(v)
        else:
            kw[k] = float(v)
    main(**kw)
