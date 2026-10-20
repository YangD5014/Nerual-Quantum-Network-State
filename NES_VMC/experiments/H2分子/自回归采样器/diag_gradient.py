"""诊断：MC 重要性加权梯度 vs 全空间精确梯度（仅诊断，非主算法）。

方法（沿用项目「全空间 16 维稠密 H 对照」方法论）：
  1. 全空间：ψ(x;θ)=exp(logψ)，E(θ)=⟨ψ|H|ψ⟩/⟨ψ|ψ⟩ 对 16 基矢 autodiff → ∇E_exact。
  2. MC：自回归采样 + forces_expect_hermitian（pow=1 加权）→ ∇E_MC。
  3. 比较 cosine、范数比；再比较两者经同一 SR 度量后的自然梯度方向。
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "..")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax import flatten_util

import H2_631G_AR_VMC as M
from ar_nqs import CausalMaskedARNQS, SegmentLayout, create_machine, make_ar_sampler
from H2_6_31G_8Qbit_K2 import hi, ha

layout = SegmentLayout(8, [(0, 4, 1), (4, 8, 1)])
model = CausalMaskedARNQS(8, 4, rngs=nnx.Rngs(21))
machine, net_out, gd, params = create_machine(model, layout)

states = jnp.asarray(hi.all_states(), dtype=jnp.int32)
st = np.asarray(states)
key = {tuple(r): i for i, r in enumerate(st)}
eta_all, H_all = ha.get_conn_padded(states)
eta_all = np.asarray(eta_all); H_all = np.asarray(H_all)
H = np.zeros((16, 16), dtype=complex)
for i in range(16):
    for e, h in zip(eta_all[i], H_all[i]):
        H[key[tuple(e)], i] += h
H = jnp.asarray(H)

# ---- 全空间精确能量与梯度 ----
def energy_full(p):
    lp = machine(p, states)
    psi = jnp.exp(lp - jnp.max(lp.real))
    return (jnp.vdot(psi, H @ psi) / jnp.vdot(psi, psi)).real

E_exact, g_exact = jax.value_and_grad(energy_full)(params)
gf, unr = flatten_util.ravel_pytree(g_exact)
print(f"EXACT E = {float(E_exact):.8f}  |g_exact| = {float(jnp.linalg.norm(gf)):.6e}")

# ---- MC 梯度（与训练完全一致的估计器）----
sample = make_ar_sampler(net_out, layout, 8)
sigma, logw = sample(params, jax.random.PRNGKey(0), 3000, 1.0)
Emc, std, g_mc = M.forces_expect_hermitian(machine, params, sigma, logw)
mf, _ = flatten_util.ravel_pytree(g_mc)
print(f"MC    E = {float(Emc.real):.8f} ± {float(std):.6f}  |g_mc| = {float(jnp.linalg.norm(mf)):.6e}")
cos = float(jnp.vdot(mf, gf).real / (jnp.linalg.norm(mf) * jnp.linalg.norm(gf)))
print(f"COSINE(g_mc, g_exact) = {cos:.4f}")

# ---- 多次 MC 的噪声底：两次独立 MC 梯度的 cosine ----
sigma2, logw2 = sample(params, jax.random.PRNGKey(7), 3000, 1.0)
_, _, g_mc2 = M.forces_expect_hermitian(machine, params, sigma2, logw2)
m2, _ = flatten_util.ravel_pytree(g_mc2)
print(f"COSINE(g_mc1, g_mc2) = {float(jnp.vdot(m2, mf).real / (jnp.linalg.norm(m2)*jnp.linalg.norm(mf))):.4f}")

# ---- 全空间精确 E_loc（诊断）：验证 compute_local_energies 一致性 ----
El_mc = np.asarray(M.compute_local_energies(machine, params, states))
El_exact = np.asarray(H @ jnp.exp(machine(params, states)) / jnp.exp(machine(params, states)))
print("E_loc 全空间 vs get_conn 最大偏差:", np.abs(El_mc - El_exact).max())
print("done")
