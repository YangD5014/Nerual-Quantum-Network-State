"""QE-NES-VMC 单元验证（接口 / 形状 / 行列式梯度恒等式 / 采样器合法性）。

覆盖《Quantum_Enhanced_NES_VMC方案总结.md》§9 的行列式微分恒等式
    ∂log D = Tr[A⁻¹ ∂A]
与现有 NES_VMC_tool 流水线的接口兼容性检查。

运行：/opt/miniconda3/envs/Netket/bin/python test_qe_nes_vmc.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, ".."))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import flax.nnx as nnx
import netket as nk
from jax.flatten_util import ravel_pytree

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed, Ham_Psi_scaled, NESFermionHopRule,
    flatten_batched_pytree,
)
from NES_VMC_tool import (
    create_gauge_reset_total_machines, NES_loss_energy_stable_gauge, make_gauge_fn,
)
from H2_6_31G_8Qbit_K4 import SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock
from qe_nes_ansatz import QETotalAnsatz

ok = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name} {detail}")
    ok.append(cond)
    return cond


# ---------- 构建 ----------
model = QETotalAnsatz(SINGLE_SIZE, n_states=K, hidden_dim=12, n_layers=2,
                     pqc_scale=0.1, rngs=nnx.Rngs(11))
(total_machine, total_matrix_machine, total_max_machine,
 total_matrix_machine_raw, graphdef, params) = create_gauge_reset_total_machines(
    model, Hatree_Fock)
single_machines = [
    create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
    for a in model.single_ansatz_list
]
g0 = jnp.zeros(K, dtype=jnp.complex128)

states = jnp.asarray(np.array(hi.all_states()), dtype=jnp.int32)   # (16, 8)
rng = np.random.default_rng(0)
# 无放回抽样：避免重复组态使 det=0 → logΨ=-inf（真实采样器天然避开重复块）
walk = states[rng.choice(16, size=K, replace=False)]               # (K, 8) 联合 walker
batch = np.stack([states[rng.choice(16, size=K, replace=False)]
                  for _ in range(64)])                              # (64, K, 8)

# ---------- 1. 单态前向形状 ----------
s0 = single_machines[0]
p0 = params["single_ansatz_list"][0]
v1 = s0(p0, walk[0])          # (8,) → 标量
v2 = s0(p0, walk)             # (4,8) → (4,)
check("单态前向 1D→标量", v1.shape == () and jnp.isfinite(v1), f"{v1.shape}")
check("单态前向 2D→(K,)", v2.shape == (K,) and jnp.all(jnp.isfinite(v2)), f"{v2.shape}")

# ---------- 2. total machine 各形状 ----------
m_flat = total_machine(params, np.asarray(walk).reshape(-1), g0)
m_mat = total_machine(params, walk, g0)
m_b2 = total_machine(params, batch.reshape(64, -1), g0)
m_b3 = total_machine(params, batch, g0)
check("total_machine (32,)→标量", m_flat.shape == () and jnp.isfinite(m_flat))
check("total_machine (K,8)→标量", m_mat.shape == () and jnp.isfinite(m_mat))
check("total_machine (64,32)→(64,)", m_b2.shape == (64,) and jnp.all(jnp.isfinite(m_b2)))
check("total_machine (64,K,8)→(64,)", m_b3.shape == (64,))
check("形状间一致性", abs(m_flat - m_mat) < 1e-12 and
      jnp.allclose(m_b3[:8], m_b2[:8], atol=1e-12))

# ---------- 3. 行列式梯度恒等式 ∂logD = Tr[A⁻¹∂A]（方案 §9）----------
# A[i,j] = exp(single_j(x_i))（规范固定列，不含 g/shift —— 二者对参数为常数，
# 在 Tr[A⁻¹∂A] 中严格消掉，故恒等式在该规范下成立）
A = jnp.stack([jnp.exp(single_machines[j](params["single_ansatz_list"][j], walk))
               for j in range(K)], axis=1)          # (K 行=构型, K 列=态)
Ainv = jnp.linalg.inv(A)

# autodiff 全参数梯度（复输出拆 Re/Im）
gd_re = ravel_pytree(jax.grad(lambda p: jnp.real(total_machine(p, walk, g0)))(params))[0]
gd_im = ravel_pytree(jax.grad(lambda p: jnp.imag(total_machine(p, walk, g0)))(params))[0]
gd_ad = gd_re + 1j * gd_im

# 恒等式梯度：对每个态列 m，∇_m = Σ_n (A⁻¹)[m,n]·A[n,m]·∇logψ_m(x_n)
id_parts = []
for mcol in range(K):
    coeffs = Ainv[mcol, :] * A[:, mcol]              # (K,) over 行 n
    gs = []
    for nrow in range(K):
        gr = ravel_pytree(jax.grad(lambda p: jnp.real(single_machines[mcol](p, walk[nrow])))(
            params["single_ansatz_list"][mcol]))[0]
        gi = ravel_pytree(jax.grad(lambda p: jnp.imag(single_machines[mcol](p, walk[nrow])))(
            params["single_ansatz_list"][mcol]))[0]
        gs.append(gr + 1j * gi)
    G = jnp.stack(gs)                                 # (K 行=n, P_m)
    id_parts.append(jnp.sum(coeffs[:, None] * G, axis=0))
id_flat = jnp.concatenate(id_parts)

sizes = [x.size for x in id_parts]
check("恒等式：ravel 长度匹配", id_flat.size == gd_ad.size,
      f"id={id_flat.size} ad={gd_ad.size}")
if id_flat.size == gd_ad.size:
    rel = jnp.linalg.norm(id_flat - gd_ad) / jnp.linalg.norm(gd_ad)
    check("恒等式 Tr[A⁻¹∂A] == autodiff", rel < 1e-8, f"rel_err={float(rel):.2e}")

# ---------- 4. PQC 参数梯度非零（相位支路真的在训练）----------
off = 0
for mcol in range(K):
    leaves, _ = jax.tree_util.tree_flatten(params["single_ansatz_list"][mcol])
    keys, _ = jax.tree_util.tree_flatten_with_path(params["single_ansatz_list"][mcol])
    sz = [x.size for x in leaves]
    qnn_idx = [i for i, n in enumerate(keys) if "qnn_params" in str(n)][0]
    a = off + sum(sz[:qnn_idx]); b = a + sz[qnn_idx]
    if mcol == 0:
        qnn_g0 = gd_ad[a:b]
    off += sum(sz)
check("PQC 变分参数梯度非零", float(jnp.linalg.norm(qnn_g0)) > 1e-6,
      f"|g_qnn|={float(jnp.linalg.norm(qnn_g0)):.3e}")

# ---------- 5. Ham_Psi_scaled / loss / E_L ----------
shift = total_max_machine(params, walk, g0)
HP = Ham_Psi_scaled(ha, single_machines, params, walk, shift)
check("Ham_Psi_scaled (K,K) 有限", HP.shape == (K, K) and jnp.all(jnp.isfinite(HP)))
loss_b, EL_b = NES_loss_energy_stable_gauge(
    ha, total_matrix_machine, total_max_machine, single_machines, params, batch, g0)
check("NES loss (64,) 有限", loss_b.shape == (64,) and jnp.all(jnp.isfinite(loss_b)),
      f"loss mean={float(jnp.mean(loss_b)):.6f}")
check("E_L 矩阵 (64,K,K)", EL_b.shape == (64, K, K))

# ---------- 6. 梯度/QGT 全流水线（小批量）----------
import H2_631G_K4_QE_NES_VMC as TR

grad_fn = TR.make_grad_fn_gauge_real(ha, total_matrix_machine, total_max_machine,
                                     total_machine, single_machines)
qgt_fn = TR.make_qgt_fn_gauge_real(total_machine)
grad, loss_mean, E_L_mean, aux = grad_fn(params, batch, g0)
gf, _ = ravel_pytree(grad)
check("grad_fn：梯度全有限", bool(jnp.all(jnp.isfinite(gf))),
      f"|g|={float(jnp.linalg.norm(gf)):.3e} ESS={float(aux['ess']):.0f}/64")
S = qgt_fn(params, batch, 0.1, g0)
ev = np.linalg.eigvalsh(np.asarray(S))
check("QGT：对称 + 正则后正定", float(np.abs(S - S.T).max()) < 1e-10 and ev.min() > 0,
      f"λmin={ev.min():.3e} λmax={ev.max():.3e}")
ng = jnp.linalg.solve(S, gf)
check("SR 解有限", bool(jnp.all(jnp.isfinite(ng))))

# ---------- 7. 采样器合法性 ----------
rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
sampler = nk.sampler.MetropolisSampler(hilbert=hi_ext, rule=rule,
                                       n_chains=4, sweep_size=10, machine_pow=1)
st = sampler.init_state(lambda p, s: total_machine(p, s, g0), params, jax.random.PRNGKey(0))
smp, st = sampler.sample(lambda p, s: total_machine(p, s, g0), params, state=st,
                         chain_length=5)
smp = np.asarray(smp)
check("采样器输出形状 (4,5,32)", smp.shape == (4, 5, K * SINGLE_SIZE))
blocks = smp.reshape(-1, K, SINGLE_SIZE)
up = blocks[..., :4].sum(-1); dn = blocks[..., 4:].sum(-1)
check("每副本粒子数约束 (1,1)", bool((up == 1).all() and (dn == 1).all()))

print("\n" + ("全部通过 ✓" if all(ok) else "存在失败 ✗"))
sys.exit(0 if all(ok) else 1)
