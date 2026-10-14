"""
复原 300 步训练后哈密顿量各能级 (K=4) 的近似本征态 φ_0..φ_3 及对应特征值。

流程（参考 手动验证.ipynb）：
  1. 重建同架构 NES 拟设 + gauge-fixed 单列 machine；
  2. 载入训练参数 history['params'][-1] 与 walker 样本 history['samples'][-2]；
  3. 样本版 MV=SVA：compute_lam_v_from_samples → λ_k（有偏监控值）与旋转矩阵 v
     （列 = 能级，φ_k = Σ_j ψ_j v[j,k]，与 NES_VMC_deflation.extract_frozen_state 一致）；
  4. 在全部 16 个 Fock 构态基矢上重建 φ_k 振幅向量，并做变分验收（仅诊断，
     不进入训练/估计器）：Rayleigh 商（真变分上界）、局域能量方差、
     与 H_dense 精确本征矢的重叠；
  5. 保存 φ_k、v、λ 到 npz。
"""
import numpy as np
import pickle
import sys
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from pathlib import Path

# 获取当前文件所在目录与上一级目录（NES_VMC_V1/NES_VMC_tool/H2_6_31G_8Qbit_K4 在上一级）
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from NES_VMC_V1 import create_single_machine_gauge_fixed
from NES_VMC_tool import make_MS_estimator_fn, compute_lam_v_from_samples
from H2_6_31G_8Qbit_K4 import SINGLE_SIZE, ha, hi, K, Hatree_Fock, E_fcis


# ===================== Ansatz：双实输出拼装复数（与 手动验证.ipynb 完全一致） =====================
class SingleStateAnsatz_SharedBackbone(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_so = n_spin_orbitals
        # 全部网络参数依旧 float 实数，不使用复数权重
        self.backbone1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.backbone2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.head_u = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)
        self.head_v = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.backbone1(x))
        h = nnx.tanh(self.backbone2(h))
        u = jnp.squeeze(self.head_u(h))
        v = jnp.squeeze(self.head_v(h))
        # 拼装复数 z = u + iv，返回 log(z)；小 epsilon 防止 log 奇点
        z = (u + 1e-12) + 1j * (v + 1e-12)
        return jnp.log(z)


class NESTotalAnsatz_stable(nnx.Module):
    """NES 总拟设：K 个「实参数 + 复输出」单态拟设组成的行列式波函数。"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()

        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz_SharedBackbone(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=jnp.complex128)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))

            shift = jax.lax.stop_gradient(jnp.max(jnp.real(L)))
            L_stable = L - shift
            Psi_stable = jnp.exp(L_stable)
            sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi_stable, L_stable, shift

        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin * self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f"不支持的输入形状: {x.shape}")


def main():
    # ---------- 1. 载入训练历史 ----------
    pkl_path = current_dir / ".." / "data" / "26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl"
    with open(pkl_path, "rb") as f:
        history = pickle.load(f)

    params = history["params"][-1]          # 第 300 步（Step 299）参数
    samples = np.asarray(history["samples"][-2])   # (N, K*SINGLE_SIZE) 整数构型
    N = samples.shape[0]
    x_batch = samples.reshape(N, K, SINGLE_SIZE)

    # ---------- 2. 重建拟设与 gauge-fixed 单列 machine ----------
    total_ansatz = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K,
        hidden_dim=SINGLE_SIZE + K,
        rngs=nnx.Rngs(11),
    )
    single_machine_list = [
        create_single_machine_gauge_fixed(ansatz, Hatree_Fock)[0]
        for ansatz in total_ansatz.single_ansatz_list
    ]

    # ---------- 3. 样本版 MV=SVA：λ（有偏）+ 旋转矩阵 v ----------
    estimator = make_MS_estimator_fn(ha, single_machine_list)
    lam_real, v, n_valid, order = compute_lam_v_from_samples(
        ha=ha,
        single_machine_list=single_machine_list,
        total_params=params,
        x_batch=x_batch,
        estimator=estimator,
    )
    v = np.asarray(v)                        # (K,K) 列 = 能级（升序），φ_k = Σ_j ψ_j v[j,k]
    # 注：compute_lam_v_from_samples 的 docstring 写成行约定，但 scipy_eigh 返回本征矢在列，
    # 且已验证的 NES_VMC_deflation.extract_frozen_state 用 A @ v[:, 0]，故此处采用列约定。
    print(f"有效 walker 数 n_valid = {n_valid} / {N}")
    print(f"样本版 MV=SVA λ (有偏监控值) = {np.asarray(lam_real)}")
    print(f"FCI 精确能级 (E_fcis[:K])      = {np.asarray(E_fcis[:K])}")

    # ---------- 4. 全空间重建 φ_k（仅诊断用，不进入训练/估计器） ----------
    H_dense = np.asarray(ha.to_dense()).astype(np.complex128)   # (16,16)，基矢序 = all_states
    all_states = np.asarray(hi.all_states())                    # (16, 8)
    all_states_j = jnp.asarray(all_states)

    # A[i,j] = ψ_j(all_states[i])（gauge-fixed：ψ_j(HF)=1）
    cols = []
    for j in range(K):
        log_vals = single_machine_list[j](params["single_ansatz_list"][j], all_states_j)
        cols.append(np.exp(np.asarray(log_vals).astype(np.complex128)))
    A = np.stack(cols, axis=1)                                  # (16,K)

    Phi = A @ v                                                 # Phi[:,k] = Σ_j A[:,j]·v[j,k]（列约定）
    Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)      # 逐列归一

    # 精确对照（同一基矢序下对角化，避免 pyscf/netket 基矢排序不一致问题）
    E_exact, U = np.linalg.eigh(H_dense)

    print("\n================ φ_k 变分验收（真 Rayleigh 商 ≥ 精确值） ================")
    print(f"{'level':>5} | {'λ_k(样本MV=SVA)':>16} | {'E_k=⟨φ|Hφ⟩':>13} | {'E_k(精确)':>12} | "
          f"{'Var[E_loc]':>10} | {'最佳重叠(精确矢)':>15} | {'重叠值':>8}")
    E_ray = np.zeros(K)
    Var = np.zeros(K)
    Ov = np.zeros(K)
    best_m = np.zeros(K, dtype=int)
    for k in range(K):
        phi = Phi[:, k]
        Hphi = H_dense @ phi
        E_ray[k] = float(np.vdot(phi, Hphi).real)
        Var[k] = float(np.vdot(Hphi, Hphi).real - E_ray[k] ** 2)
        ov = np.abs(U.conj().T @ phi) ** 2
        best_m[k] = int(ov.argmax())
        Ov[k] = float(ov[best_m[k]])
        print(f"{k:>5} | {lam_real[k]:>16.8f} | {E_ray[k]:>13.8f} | {E_fcis[k]:>12.8f} | "
              f"{Var[k]:>10.2e} | {best_m[k]:>15d} | {Ov[k]:>8.5f}")

    # 每个 φ_k 的前 3 大分量（按 |振幅|）
    print("\n---- φ_k 在 all_states 基矢下的前 3 大分量 ----")
    for k in range(K):
        idx = np.argsort(np.abs(Phi[:, k]))[::-1][:3]
        comp = ", ".join(
            f"{all_states[i].tolist()}:{Phi[i, k]:.3f}" for i in idx
        )
        print(f"φ_{k}: {comp}")

    # ---------- 5. 保存 ----------
    out_path = current_dir / "近似本征态_K4_26-10-07-19-57.npz"
    np.savez(
        out_path,
        phi_vecs=Phi,                # (16,K) 复振幅，基矢序 = all_states，列已归一
        all_states=all_states,       # (16,8) 基矢
        lam_sample=lam_real,         # 样本版 MV=SVA 有偏 λ
        E_rayleigh=E_ray,            # φ_k 的真实 Rayleigh 商（变分上界）
        E_exact=E_fcis[:K],          # FCI 精确能级
        var_eloc=Var,                # 局域能量方差（本征性判据）
        overlap=Ov,                  # 与最佳精确本征矢重叠
        v_rot=v,                     # 旋转矩阵（行=能级，φ_k=Σ_j ψ_j v[k,j]）
    )
    print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
