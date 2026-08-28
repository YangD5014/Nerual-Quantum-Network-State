"""
诊断脚本：归因 rowfix 失败机制。

目标问题：
1. rowfix 的 logΨ 梯度是否真的无行方向分量（数学成立 → 实现可能泄露）
2. 为什么 rowfix 下 loss 卡在 -24.6（梯度结构差异？）
3. logΨ 漂移是否由"行方向补偿项 correction"驱动

方法：
A. 同一批样本、同一参数：对比 orig / rowfix 的 logΨ, grad_raw, loss, E_L, cond
B. 各自训练 3 步：追踪 correction(Σrow_mean)、logΨ mean、loss、raw grad
"""
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
from jax.flatten_util import ravel_pytree

from LiH import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_gauge_fixed_total_machines,
    create_single_machine_gauge_fixed,
    make_grad_fn,
    make_qgt_fn,
    NESFermionHopRule,
)


def create_rowfix_debug(total_model, ref_state):
    """rowfix total_machine 的插桩版：
    返回 (machine, corr_machine, graphdef, state)——machine 单返回值（供采样/梯度），
    corr_machine 单独输出 Σrow_mean 补偿项以追踪规范漂移。
    """
    graphdef, state = nnx.split(total_model)
    K = total_model.K
    n_spin = total_model.n_spin
    ref_state = jnp.asarray(ref_state)

    def _L(m, x_single):
        cols = []
        for j in range(K):
            ansatz_j = m.single_ansatz_list[j]
            cols.append(ansatz_j(x_single) - ansatz_j(ref_state))
        return jnp.stack(cols, axis=1)

    def _one(m, x_single):
        L = _L(m, x_single)
        row_mean = jnp.mean(L, axis=1, keepdims=True)
        L_rowfixed = L - row_mean
        shift = jax.lax.stop_gradient(jnp.max(jnp.real(L_rowfixed)))
        Psi_stable = jnp.exp(L_rowfixed - shift)
        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_det_stable = log_abs_det + 1j * jnp.angle(sign)
        correction = jax.lax.stop_gradient(jnp.sum(row_mean[:, 0]))
        val = log_det_stable + K * shift + correction
        return val, jnp.sum(row_mean[:, 0])

    def _make(selector):
        @jax.jit
        def machine(params, sigma):
            m = nnx.merge(graphdef, params)
            sigma = jnp.asarray(sigma)
            if sigma.ndim == 1:
                val, corr = _one(m, sigma.reshape(K, n_spin))
                return selector(val, corr)
            elif sigma.ndim == 2 and sigma.shape != (K, n_spin):
                xs = sigma.reshape(-1, K, n_spin)
                vals, corrs = jax.vmap(lambda x: _one(m, x))(xs)
                return selector(vals, corrs)
            elif sigma.ndim == 2:
                val, corr = _one(m, sigma)
                return selector(val, corr)
            else:
                xs = sigma.reshape(-1, K, n_spin)
                vals, corrs = jax.vmap(lambda x: _one(m, x))(xs)
                return selector(vals, corrs)
        return machine

    machine = _make(lambda v, c: v)
    corr_machine = _make(lambda v, c: c)
    return machine, corr_machine, graphdef, state


def build():
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE, n_states=K,
        hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(11),
    )
    (total_machine_orig, total_matrix_machine, total_max_machine,
     total_graphdef, total_params) = create_gauge_fixed_total_machines(
        total_ansatz, Hatree_Fock)
    (total_machine_row, corr_machine_row, _, _) = create_rowfix_debug(total_ansatz, Hatree_Fock)
    graphdef_total, _ = nnx.split(total_ansatz)

    single_machine_list = [
        create_single_machine_gauge_fixed(ans, Hatree_Fock)[0]
        for ans in total_ansatz.single_ansatz_list
    ]
    grad_fn_orig = make_grad_fn(ha, total_matrix_machine, total_max_machine,
                                total_machine_orig, single_machine_list)
    grad_fn_row = make_grad_fn(ha, total_matrix_machine, total_max_machine,
                               total_machine_row, single_machine_list)
    qgt_fn_orig = make_qgt_fn(total_machine_orig)
    qgt_fn_row = make_qgt_fn(total_machine_row)

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    sampler = nk.sampler.MetropolisSampler(hilbert=hi_ext, rule=nes_rule,
                                           n_chains=16, sweep_size=30)
    return (total_ansatz, graphdef_total, total_params, total_machine_orig,
            total_machine_row, corr_machine_row, total_matrix_machine,
            single_machine_list, grad_fn_orig, grad_fn_row,
            qgt_fn_orig, qgt_fn_row, sampler)


def build_L(graphdef, params, x):
    m = nnx.merge(graphdef, params)
    cols = []
    ref = jnp.asarray(Hatree_Fock)
    for j in range(K):
        ans_j = m.single_ansatz_list[j]
        cols.append(ans_j(x) - ans_j(ref))
    return jnp.stack(cols, axis=1)


def slogdet_vs_tr(graphdef, params, x):
    L = build_L(graphdef, params, x)
    shift = jnp.max(jnp.real(L))
    Psi = jnp.exp(L - shift)
    sign, la = jnp.linalg.slogdet(Psi)
    logdet = la + 1j * jnp.angle(sign) + K * shift
    return L, logdet


def rowfix_from_L(L):
    row_mean = jnp.mean(L, axis=1, keepdims=True)
    L_rf = L - row_mean
    shift = jnp.max(jnp.real(L_rf))
    Psi = jnp.exp(L_rf - shift)
    sign, la = jnp.linalg.slogdet(Psi)
    return la + 1j * jnp.angle(sign) + K * shift + jnp.sum(row_mean[:, 0])


def sample_batch(sampler, machine, params, rng_key):
    s_state = sampler.init_state(machine, params, rng_key)
    samples, _ = sampler.sample(machine, params, state=s_state, chain_length=200)
    return samples.reshape(-1, K, SINGLE_SIZE)


# ---------------- 主流程 ----------------
(ansatz, gdef, params, tm_orig, tm_row, corr_row, tmm, singles,
 grad_fn_orig, grad_fn_row, qgt_orig, qgt_row, sampler) = build()

print("=" * 70)
print("A. 同一批样本、同一参数下 orig vs rowfix 静态对比")
print("=" * 70)

x_batch = sample_batch(sampler, tm_orig, params, jax.random.PRNGKey(0))
print(f"x_batch.shape = {x_batch.shape}")

logPsi_o = tm_orig(params, x_batch)
logPsi_r = tm_row(params, x_batch)
corr_r = corr_row(params, x_batch)
print(f"\n[logΨ] orig mean={logPsi_o.real.mean():.4f} | "
      f"rowfix mean={logPsi_r.real.mean():.4f} | "
      f"correction(Σrow_mean) mean={corr_r.real.mean().item():+.4f}")
print(f"[logΨ] orig [min,max]={logPsi_o.real.min():.3f},{logPsi_o.real.max():.3f} | "
      f"rowfix={logPsi_r.real.min():.3f},{logPsi_r.real.max():.3f}")

x0 = x_batch[0]
L0, logdet0 = slogdet_vs_tr(gdef, params, x0)
print(f"\n[恒等式] L[0] 对角实部 tr(L) = {jnp.trace(L0):+.4f}")
print(f"[恒等式] slogdet(exp(L)) 实部 = {logdet0:+.4f}  (实部应与 tr 实部一致)")

grad_o, loss_o, E_L_o, aux_o = grad_fn_orig(params, x_batch)
grad_r, loss_r, E_L_r, aux_r = grad_fn_row(params, x_batch)

gof, _ = ravel_pytree(grad_o)
grf, _ = ravel_pytree(grad_r)
print(f"\n[grad] orig norm={jnp.linalg.norm(gof):.4f} | rowfix norm={jnp.linalg.norm(grf):.4f}")
print(f"[grad] |gr-go|/|go| = {jnp.linalg.norm(grf - gof) / (jnp.linalg.norm(gof) + 1e-30):.4e}")
print(f"[loss] orig={loss_o:.6f} | rowfix={loss_r:.6f}  (应相同 -> loss 路径未受影响)")
print(f"[E_L trace] orig={jnp.trace(E_L_o, axis1=-2, axis2=-1).mean():.4f} | "
      f"rowfix={jnp.trace(E_L_r, axis1=-2, axis2=-1).mean():.4f}")
cond_aux = aux_o["loss_aux"]["cond_Psi"]
print(f"[cond(Ψ)] orig aux = {cond_aux}")

L_shifted = L0 + jnp.array([1.0, -2.0, 3.0, 0.5])[:, None]
print(f"\n[行平移不变性] logdet(L)={rowfix_from_L(L0):+.6f} | "
      f"logdet(L+row_shift)={rowfix_from_L(L_shifted):+.6f}  (应严格相等)")

# ================= B 部分：3 步训练对比 =================
print("\n" + "=" * 70)
print("B. orig / rowfix 各训练 3 步，追踪 correction 与 logΨ mean")
print("=" * 70)

import optax


def run_few_steps(name, tm, grad_fn, qgt_fn, params0, corr_fn=None, n_steps=3):
    params = params0
    optimizer = optax.chain(optax.clip_by_global_norm(20.0), optax.sgd(learning_rate=0.1))
    opt_state = optimizer.init(params)
    print(f"\n--- {name} ---")
    key = jax.random.PRNGKey(21)
    s_state = sampler.init_state(tm, params, key)
    for step in range(n_steps):
        samples, s_state = sampler.sample(tm, params, state=s_state, chain_length=200)
        x_batch = samples.reshape(-1, K, SINGLE_SIZE)
        grad_raw, loss_mean, E_L_mean, aux = grad_fn(params, x_batch)
        grf, unravel_fn = ravel_pytree(grad_raw)
        gnorm = float(jnp.linalg.norm(grf))
        qgt = qgt_fn(params, x_batch, 0.1)
        ng_flat = jnp.linalg.solve(qgt, grf)
        ng = unravel_fn(ng_flat)
        updates, opt_state = optimizer.update(ng, opt_state, params)
        params = optax.apply_updates(params, updates)

        lp = tm(params, x_batch)
        if corr_fn is None:
            print(f"[Step {step}] loss={loss_mean:.4f} | raw_grad={gnorm:.3f} | "
                  f"logΨ mean={float(jnp.mean(lp.real)):.3f}")
        else:
            corr = corr_fn(params, x_batch)
            print(f"[Step {step}] loss={loss_mean:.4f} | raw_grad={gnorm:.3f} | "
                  f"logΨ mean={float(jnp.mean(lp.real)):.3f} | "
                  f"corr mean={float(jnp.mean(corr.real)):+.3f}")
    return params


run_few_steps("orig", tm_orig, grad_fn_orig, qgt_orig, params)
run_few_steps("rowfix", tm_row, grad_fn_row, qgt_row, params, corr_fn=corr_row)

print("\n诊断完成")