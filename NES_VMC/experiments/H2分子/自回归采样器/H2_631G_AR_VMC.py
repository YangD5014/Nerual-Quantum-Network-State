"""H2/6-31G 基态 VMC —— 自回归采样器版本。

基于典藏 notebook「FFNN振幅相位分离实数版-典藏.ipynb」，把 Metropolis MCMC
采样器换成**因果掩码自回归 NQS 的精确顺序采样器**（见 ar_nqs.py）。

关键区别（对比典藏 notebook）：
  1. Ansatz：FFNN 全局 logA(x)+iφ(x) → 因果掩码自回归网络，逐位条件分布
     π̃_i(x_i|x_<i) 在粒子数约束扇区上严格归一化，|ψ|²=Ππ̃_i 无需配分函数。
  2. 采样：MetropolisSampler(150 chains × 20 步) → 自回归顺序采样，每步
     只做 n_bits 次前向，直接得到 i.i.d. 精确样本（无自相关、无 reset）。
  3. 主算法全程 Matrix-Free：只用 ha.get_conn_padded 取局部能量，不做任何
     2^N 全空间枚举。全空间仅用于最后的诊断（重叠/方差/能量核对）。

沿用典藏 notebook 的实参数 SR 推导（Wirtinger 拆 Re/Im、度量取 Re(S)、
machine_pow=1 按 |ψ| 采样 + 重要性加权 w∝|ψ| 防塌缩冻结）。
注意：本脚本的 force 公式带因子 2（∇E=2Re⟨…⟩，典藏版代码漏乘），故默认
lr=0.05 与典藏版有效步长（0.1×½）一致。pow=2 消融实验证实：即使 AR 精确
采样，纯 |ψ|² 采样仍会零方差冻结在 HF，重要性加权不可省。

用法：
    /opt/miniconda3/envs/Netket/bin/python H2_631G_AR_VMC.py
产物：训练曲线 PNG、收敛能量/重叠打印、可选 history.npz。
"""

import sys
import time
import pickle
import warnings

warnings.filterwarnings("ignore")

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optax
from flax import nnx
from jax import flatten_util
from functools import partial

sys.path.append("..")
from H2_6_31G_8Qbit_K2 import (
    SINGLE_SIZE, ha, hi, E_fcis, hf_ground_energy,
)
from ar_nqs import (
    CausalMaskedARNQS, SegmentLayout, create_machine, make_ar_sampler,
)

jax.config.update("jax_enable_x64", True)

N_BITS = SINGLE_SIZE          # 8 自旋轨道
SEG_SPECS = [(0, 4, 1), (4, 8, 1)]   # 上旋段/下旋段各恰 1 电子


# ===================== 局部能量（Matrix-Free）=====================
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)


# ===================== 每样本梯度（实参数、复输出，拆 Re/Im）=====================
def _grad_logpsi_real_params(machine, params, sigma):
    re_grad = jax.vmap(lambda s: jax.grad(lambda p: machine(p, s).real)(params))(sigma)
    im_grad = jax.vmap(lambda s: jax.grad(lambda p: machine(p, s).imag)(params))(sigma)
    return jax.tree_util.tree_map(lambda r, i: r + 1j * i, re_grad, im_grad)


def _weighted_mean_std(x, w):
    w_sum = jnp.sum(w)
    mean = jnp.sum(w * x) / w_sum
    var = jnp.sum(w * jnp.abs(x - mean) ** 2) / w_sum
    return mean, jnp.sqrt(var / x.shape[0])


@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma, logw):
    """重要性加权 force：∇⟨E⟩ = 2·Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩_w，w=exp(logw)。"""
    w = jnp.exp(logw - jax.lax.stop_gradient(jnp.max(logw)))
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = _weighted_mean_std(O_loc, w)
    O_centered = O_loc - O_mean
    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)

    def weight_and_mean(g):
        wc = (w * O_centered).reshape((w.shape[0],) + (1,) * (g.ndim - 1))
        return 2.0 * (jnp.sum(wc * jnp.conj(g), axis=0) / jnp.sum(w)).real

    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    return O_mean, O_std, grad


def compute_qgt(machine, params, sigma, logw, diag_shift=0.01):
    """加权 QGT（实参数度量 Re(S)）。"""
    w = jnp.exp(logw - jax.lax.stop_gradient(jnp.max(logw)))
    grad_matrix = _grad_logpsi_real_params(machine, params, sigma)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    n = sigma.shape[0]
    grad_flat = grad_flat.reshape(n, -1)
    w_sum = jnp.sum(w)
    w_mean = jnp.sum(w[:, None] * grad_flat, axis=0, keepdims=True) / w_sum
    gc = grad_flat - w_mean
    qw = (w[:, None] * jnp.conj(gc)).T @ gc / w_sum
    qgt = qw.real
    return qgt + diag_shift * jnp.eye(qgt.shape[0]), unravel_fn


# ===================== 全空间诊断（仅用于验证，非主算法）=====================
def build_H_dense():
    """诊断专用：在 16 个基矢上 get_conn 组装稠密 H（主算法不使用）。"""
    states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
    st = np.asarray(states)
    key = {tuple(row): i for i, row in enumerate(st)}
    H = np.zeros((16, 16), dtype=complex)
    eta_all, H_all = ha.get_conn_padded(states)
    eta_all = np.asarray(eta_all); H_all = np.asarray(H_all)
    for i in range(16):
        for e, h in zip(eta_all[i], H_all[i]):
            H[key[tuple(e)], i] += h
    return states, jnp.asarray(H)


def exact_grad(machine, params, states, H):
    """全空间 Rayleigh 商 autodiff → 精确能量与梯度范数（诊断）。"""
    def energy_full(p):
        lp = machine(p, states)
        psi = jnp.exp(lp - jnp.max(lp.real))
        return (jnp.vdot(psi, H @ psi) / jnp.vdot(psi, psi)).real
    E, g = jax.value_and_grad(energy_full)(params)
    gf, _ = flatten_util.ravel_pytree(g)
    return float(E), float(jnp.linalg.norm(gf))


def full_space_diag(machine, params, H_bundle=None):
    states, H = H_bundle if H_bundle is not None else build_H_dense()
    lp = machine(params, states)
    psi = np.exp(np.asarray(lp))
    psi = psi / np.linalg.norm(psi)
    E = np.vdot(psi, H @ psi).real
    E2 = np.vdot(H @ psi, H @ psi).real
    var = E2 - E ** 2
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    overlap = abs(np.vdot(gs, psi)) ** 2
    return E, var, overlap, evals[0]


def main(learning_rate=0.05, diag_shift=0.01, seed=21, n_iter=400,
         n_walkers=3000, machine_pow=1.0, tag="", load_params=""):

    layout = SegmentLayout(N_BITS, SEG_SPECS)
    model = CausalMaskedARNQS(N_BITS, hidden_per_site=4, rngs=nnx.Rngs(seed))
    machine, net_out, _graphdef, params = create_machine(model, layout)
    if load_params:
        flat = jnp.asarray(np.load(load_params)["flat"])
        _, unr = flatten_util.ravel_pytree(params)
        params = unr(flat)
        print(f"[续训] 载入参数：{load_params}")
    sample = make_ar_sampler(net_out, layout, N_BITS)

    optimizer = optax.sgd(learning_rate)
    opt_state = optimizer.init(params)

    rng = jax.random.PRNGKey(1)
    history = {"step": [], "energy": [], "energy_std": [], "error": [],
               "diag_E": [], "diag_var": [], "diag_overlap": []}
    hb = build_H_dense()

    print("=" * 64)
    print("H2/6-31G 基态 VMC — 自回归精确采样器 (Matrix-Free 主算法)")
    print(f"n_bits={N_BITS}  n_walkers={n_walkers}  machine_pow={machine_pow}  "
          f"lr={learning_rate}  diag_shift={diag_shift}  tag={tag}")
    print("=" * 64)

    sigma = None
    logw = None
    t0 = time.time()
    for step in range(n_iter):
        # 自回归精确采样（i.i.d.，无 MCMC reset）
        rng, key = jax.random.split(rng)
        sigma, logw = sample(params, key, n_walkers, machine_pow)

        energy, energy_std, grad = forces_expect_hermitian(machine, params, sigma, logw)

        qgt_reg, _ = compute_qgt(machine, params, sigma, logw, diag_shift)
        grad_flat, unravel = flatten_util.ravel_pytree(grad)
        nat = jnp.linalg.solve(qgt_reg, grad_flat)
        grad = unravel(nat)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 50 == 0 or step == n_iter - 1:
            err = abs(float(energy.real) - E_fcis[0])
            dE, dvar, dov, e0exact = full_space_diag(machine, params, hb)
            for k, v in [("step", step), ("energy", float(energy.real)),
                         ("energy_std", float(energy_std)), ("error", err),
                         ("diag_E", float(dE)), ("diag_var", float(dvar)),
                         ("diag_overlap", float(dov))]:
                history[k].append(v)
            print(f"Step {step:3d} | MC E: {energy.real:.8f}±{energy_std:.6f} "
                  f"| 全空间 E: {dE:.8f} Var: {dvar:.2e} Ovl: {dov:.6f} "
                  f"| err(FCI): {err:.6f}")

    dur = time.time() - t0
    dE, dvar, dov, e0exact = full_space_diag(machine, params, hb)
    Eex, gn = exact_grad(machine, params, hb[0], hb[1])
    print("=" * 64)
    print(f"训练完成 用时 {dur:.1f}s")
    print(f"MC 能量：{history['energy'][-1]:.8f} Ha")
    print(f"全空间基态能量：{dE:.8f} Ha  (精确 E0={e0exact:.8f})")
    print(f"全空间方差：{dvar:.3e}   与基态重叠：{dov:.6f}")
    print(f"精确梯度范数 |∇E|：{gn:.3e}")
    print(f"FCI 基准 E0：{E_fcis[0]:.8f} Ha")
    print(f"绝对误差：{abs(dE - E_fcis[0]):.6f} Ha")
    print("=" * 64)

    # 绘图
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(history["step"], history["energy"], yerr=history["energy_std"],
                fmt="o-", capsize=3, label="VMC energy (AR sampler)")
    ax.axhline(E_fcis[0], color="r", ls="--", label=f"FCI E0 = {E_fcis[0]:.6f}")
    ax.axhline(hf_ground_energy, color="g", ls="--", label=f"HF = {hf_ground_energy:.6f}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title("Autoregressive-sampler VMC (H2/6-31G)")
    ax.legend()
    fig.tight_layout()
    out_png = f"H2_631G_AR_VMC{tag}.png"
    fig.savefig(out_png, dpi=150)
    print(f"训练曲线已保存：{out_png}")

    flat, unravel = flatten_util.ravel_pytree(params)
    np.savez(f"H2_631G_AR_VMC_params{tag}.npz", flat=np.asarray(flat))
    with open(f"H2_631G_AR_VMC_history{tag}.pkl", "wb") as f:
        pickle.dump(history, f)
    return machine, params, history


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=", 1)
        if k in ("tag", "load_params"):
            kw[k] = v
        else:
            kw[k] = float(v) if "." in v else int(v)
    main(**kw)
