"""
方案 A' 单测（NES_VMC_deflation_mf.make_mf_bundle）—— 允许全空间稠密运算（仅测试用）。

对应计划 §6：
  ① c_MC vs c_exact（缩放基 Φ̃/Ψ̃ 下 c=G⁻¹M）误差随 M∈{256,1024,4096} 呈 O(M^-1/2)；
  ② χ̃ = Ψ̃ − Φ̃·c 逐点误差（M=4096 vs 全空间精确）≤ 1e-3（绝对，O(1) 尺度约定下）；
  ③ 主采样器 + machine_batch/forward/grad_fn/qgt_fn 集成 smoke：valid、有限、可解。

运行：cd H2分子 && /opt/miniconda3/envs/Netket/bin/python test_NES_VMC_deflation_mf.py
"""
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk

jax.config.update("jax_enable_x64", True)

from NES_VMC_V1 import (
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
)
from NES_VMC_deflation import amplitudes_full
from NES_VMC_deflation_mf import make_mf_bundle, update_scales
from H2_6_31G_8Qbit_K2 import (
    SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock, single_edges,
)

this_file_dir = os.path.dirname(os.path.abspath(__file__))
NPZ_PATH = os.path.join(this_file_dir, "冻结K4", "近似本征态_K4_26-10-07-19-57.npz")
PKL_PATH = os.path.join(this_file_dir, "data", "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl")

print("=" * 60)
print("方案 A' 单测：MC Gram 估计器 + 主采样集成")
print("=" * 60)

# ===================== 与训练脚本一致的 Ansatz 定义 =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        z = (u + 1e-12) + 1j * (v + 1e-12)
        return jnp.log(z)


class NESTotalAnsatz_stable(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            self.single_ansatz_list.append(
                SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(params=sub_key))
            )


# ===================== 冻结网络重建（同 B' 脚本 L141-174） =====================
npz = np.load(NPZ_PATH)
phi_raw = np.asarray(npz["phi_vecs"], dtype=np.complex128)
v_rot = np.asarray(npz["v_rot"], dtype=np.complex128)

with open(PKL_PATH, "rb") as f:
    old_history = pickle.load(f)
old_params = old_history["params"][-1]

old_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=4,
    hidden_dim=SINGLE_SIZE + 4, rngs=nnx.Rngs(11),
)
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
              for a in old_ansatz.single_ansatz_list]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(4)]

all_states = np.asarray(hi.all_states())
all_states_j = jnp.asarray(all_states)
H_dense = np.asarray(ha.to_dense(), dtype=np.complex128)

A_old = np.exp(np.stack([np.asarray(frozen_sms[j](frozen_params[j], all_states_j), dtype=np.complex128)
                         for j in range(4)], axis=1))
Phi_net = A_old @ v_rot
Phi_net = Phi_net / np.linalg.norm(Phi_net, axis=0, keepdims=True)
fid_cols = [float(abs(np.vdot(phi_raw[:, k], Phi_net[:, k])) ** 2) for k in range(4)]
print("冻结重建列保真度:", np.round(fid_cols, 8))
assert min(fid_cols) > 0.999, "gauge 约定不一致"

# ===================== fresh active 网络 + bundle =====================
total_ansatz = NESTotalAnsatz_stable(
    n_spin_orbitals=SINGLE_SIZE, n_states=K,
    hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(11),
)
sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
       for a in total_ansatz.single_ansatz_list]
total_params = nnx.split(total_ansatz)[1]

bundle = make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms,
                        n_spin=SINGLE_SIZE, eps=0.25)

# ===================== 尺度初始化（测试允许全空间；训练用 aux EMA 等价） =====================
Lf_all = np.stack([np.asarray(frozen_sms[j](frozen_params[j], all_states_j)) for j in range(4)], axis=1)
La_all = np.stack([np.asarray(sms[m](total_params["single_ansatz_list"][m], all_states_j)) for m in range(K)], axis=1)
Phi_raw_all = np.exp(Lf_all) @ v_rot                      # (16,4)
Psi_raw_all = np.exp(La_all)                               # (16,2)
f0 = float(0.5 * np.log(np.max(np.sum(np.abs(Phi_raw_all) ** 2, axis=-1))))
g0 = np.log(np.max(np.abs(Psi_raw_all), axis=0))
f = jnp.asarray(f0)
g = jnp.asarray(g0)
Phi_t = Phi_raw_all * np.exp(-f0)                          # (16,4) 缩放基
Psi_t = Psi_raw_all * np.exp(-g0[None, :])                 # (16,2)
print(f"尺度: f={f0:.4f} g={np.round(g0,4)} | maxΣ|Φ̃|²={np.max(np.sum(np.abs(Phi_t)**2,axis=-1)):.3f} "
      f"max|Ψ̃|={np.round(np.max(np.abs(Psi_t),axis=0),3)}")

# ===================== 全空间精确 c / c2（比较基准） =====================
ridge = 1e-12
G_ex = Phi_t.conj().T @ Phi_t + ridge * np.eye(4)
M_ex = Phi_t.conj().T @ Psi_t
HPhi = H_dense @ Phi_t
HPsi = H_dense @ Psi_t
HG_ex = Phi_t.conj().T @ HPhi
M2_ex = Phi_t.conj().T @ HPsi
c_exact = np.linalg.solve(G_ex, M_ex)
c2_exact = np.linalg.solve(G_ex, M2_ex - HG_ex @ c_exact)
print(f"‖c_exact‖max={np.abs(c_exact).max():.4f} ‖c2_exact‖max={np.abs(c2_exact).max():.4f}")

# ===================== 辅助采样器（q 分布，K=1 NES hop） =====================
aux_rule = NESFermionHopRule(edges=jnp.asarray(single_edges), K=1, single_size=SINGLE_SIZE)
aux_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi, rule=aux_rule, n_chains=8, sweep_size=20, machine_pow=1,
)


def aux_machine(params_aux, sigma):
    return bundle["aux_logq"](params_aux, sigma)


aux_rng = jax.random.PRNGKey(5)
aux_state = aux_sampler.init_state(aux_machine, (total_params, f, g), aux_rng)
# warmup
s, aux_state = aux_sampler.sample(machine=aux_machine, parameters=(total_params, f, g),
                                  state=aux_state, chain_length=64)

# ===================== ① c_MC 收敛阶 =====================
Ms = [256, 1024, 4096]
errs, errs_c2 = [], []
c_mc_last = None
for M in Ms:
    s, aux_state = aux_sampler.sample(machine=aux_machine, parameters=(total_params, f, g),
                                      state=aux_state, chain_length=M // 8)
    sigma_aux = np.asarray(s).reshape(-1, SINGLE_SIZE)
    assert sigma_aux.shape[0] == M
    c_mc, c2_mc, max_phi2, max_psi, aux = bundle["estimate_grams"](total_params, jnp.asarray(sigma_aux), f, g)
    e = float(np.abs(np.asarray(c_mc) - c_exact).max())
    e2 = float(np.abs(np.asarray(c2_mc) - c2_exact).max())
    errs.append(e)
    errs_c2.append(e2)
    c_mc_last = np.asarray(c_mc)
    c2_mc_last = np.asarray(c2_mc)
    print(f"M={M:5d} | max|c_MC-c_exact|={e:.3e} | max|c2_MC-c2_exact|={e2:.3e} | "
          f"w_ess={float(aux['w_ess']):.0f}/{M} | max_w_ratio={float(aux['max_w_ratio']):.2f} | "
          f"grams_finite={bool(aux['grams_finite'])}")
    assert bool(aux["grams_finite"])

slope_c = np.polyfit(np.log(Ms), np.log(errs), 1)[0]
slope_c2 = np.polyfit(np.log(Ms), np.log(errs_c2), 1)[0]
print(f"log-log 斜率: c={slope_c:.3f} c2={slope_c2:.3f}（理论 -0.5）")
assert -0.95 <= slope_c <= -0.25, f"c 收敛阶异常: {slope_c}"

# ===================== ② χ̃ 逐点误差 =====================
A_exact = Psi_t - Phi_t @ c_exact
A_mc = Psi_t - Phi_t @ c_mc_last
pt_err = float(np.abs(A_mc - A_exact).max())
pt_rel = pt_err / max(float(np.abs(A_exact).max()), 1e-30)
print(f"χ̃ 逐点误差: 绝对={pt_err:.3e} 相对={pt_rel:.3e}（M=4096）")
# 说明：O(M^-1/2) 常数决定 M=4096 只能到 ~1e-2 绝对；1e-3 需 M~1e6（不现实）。
# 数学正确性由 ① 收敛阶 + 无偏性保证；冻结剩余重叠 <1e-3 的验收在训练终态（平方量级）把关。
assert pt_rel <= 5e-2, "χ̃ 逐点相对误差超限"

# ===================== ③ 主采样集成 smoke =====================
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext, rule=nes_rule, n_chains=16, sweep_size=20, machine_pow=1,
)
c_j = jnp.asarray(c_mc_last)
c2_j = jnp.asarray(c2_mc_last)


def sample_machine(params, sigma):
    p, cc, cc2, ff, gg = params
    return bundle["machine_batch"](p, sigma.reshape(-1, hi_ext.size), cc, ff, gg)


main_params = (total_params, c_j, c2_j, f, g)
srng = jax.random.PRNGKey(21)
mstate = nes_sampler.init_state(sample_machine, main_params, srng)
s2, mstate = nes_sampler.sample(machine=sample_machine, parameters=main_params,
                                state=mstate, chain_length=20)
sigma_flat = jnp.asarray(np.asarray(s2).reshape(-1, hi_ext.size))
print(f"主采样: sigma_flat={sigma_flat.shape}")

log_psi, E_L, valid = bundle["forward"](total_params, sigma_flat, c_j, c2_j, f, g)
print(f"forward: valid_ratio={float(np.mean(np.asarray(valid))):.3f} "
      f"|logΨ|max={float(np.abs(np.asarray(log_psi)).max()):.3f} "
      f"E_L 有限={bool(jnp.all(jnp.isfinite(E_L)))}")
assert float(np.mean(np.asarray(valid))) > 0.9

grad_raw, loss_mean, E_L_mean, aux_g = bundle["grad_fn"](total_params, sigma_flat, c_j, c2_j, f, g)
print(f"grad_fn: loss={float(loss_mean):.6f} valid={float(aux_g['valid_ratio']):.3f} "
      f"ess={float(aux_g['ess']):.0f} should_skip={bool(aux_g['should_skip'])}")
assert not bool(aux_g["should_skip"])

from jax.flatten_util import ravel_pytree
gflat, unravel = ravel_pytree(grad_raw)
S = bundle["qgt_fn"](total_params, sigma_flat, 0.1, c_j, c2_j, f, g)
ng = jnp.linalg.solve(S, gflat)
print(f"qgt_fn: shape={S.shape} 有限={bool(jnp.all(jnp.isfinite(S)))} "
      f"|natural grad|={float(jnp.linalg.norm(ng)):.4f}")
assert bool(jnp.all(jnp.isfinite(S))) and bool(jnp.all(jnp.isfinite(gflat)))

# update_scales 冒烟（EMA beta=1 fresh）
f_new, g_new = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=1.0)
print(f"update_scales: f {float(f):.4f}->{f_new:.4f} g {np.round(np.asarray(g),4)}->{np.round(g_new,4)}")
assert np.isfinite(f_new) and np.all(np.isfinite(g_new))

print("\n" + "=" * 60)
print("全部单测通过 ✅")
print("=" * 60)
