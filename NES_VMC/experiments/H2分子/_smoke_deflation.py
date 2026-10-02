import time
import numpy as np
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax

from H2_631G import SINGLE_SIZE, ha, hi_ext, ext_edges, K, Hatree_Fock, hi, E_fcis
from NES_VMC_V1 import (
    NESTotalAnsatz_stable,
    create_single_machine_gauge_fixed,
    NESFermionHopRule,
    ravel_pytree,
)
import NES_VMC_deflation as dfl

H_dense, all_states = dfl.build_full_stage(ha, hi)
print("H_dense", H_dense.shape, "all_states", all_states.shape)
print("exact E0..E3:", E_fcis[:4])


def make_ansatz(seed):
    total = NESTotalAnsatz_stable(
        n_spin_orbitals=SINGLE_SIZE, n_states=K,
        hidden_dim=SINGLE_SIZE + K, rngs=nnx.Rngs(seed),
    )
    graphdef, params = nnx.split(total)
    sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
           for a in total.single_ansatz_list]
    return total, params, sms


def run_stage(params, sms, frozen_vecs, n_iter, seed=21):
    machine_single, machine, el, forward = dfl.make_deflation_bundle(
        sms, all_states, H_dense, frozen_vecs
    )
    grad_fn = dfl.make_deflated_grad_fn(forward)
    qgt_fn = dfl.make_deflated_qgt_fn(machine_single)

    nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
    sampler = nk.sampler.MetropolisSampler(
        hilbert=hi_ext, rule=nes_rule, n_chains=16, sweep_size=20,
    )
    sampler_state = sampler.init_state(machine, params, jax.random.PRNGKey(seed))

    optimizer = optax.chain(optax.clip_by_global_norm(20.0), optax.sgd(0.1))
    opt_state = optimizer.init(params)

    hist = []
    for step in range(n_iter):
        samples, sampler_state = sampler.sample(
            machine=machine, parameters=params, state=sampler_state, chain_length=400,
        )
        x_batch = samples.reshape(-1, hi_ext.size)

        grad, loss, EL_mean, aux = grad_fn(params, x_batch)
        qgt = qgt_fn(params, x_batch, 0.1)
        gflat, unravel = ravel_pytree(grad)
        ngflat = jnp.linalg.solve(qgt, gflat)
        updates = unravel(ngflat)
        updates, opt_state = optimizer.update(updates, opt_state, params)
        params = optax.apply_updates(params, updates)

        ev = jnp.linalg.eigvals(EL_mean)
        ev = jnp.sort(ev.real)
        hist.append((float(loss), [float(e) for e in ev]))
        if step % 5 == 0 or step == n_iter - 1:
            print(f"  step {step:3d} loss={float(loss):.5f} "
                  f"E=[{ev[0]:.5f}, {ev[1]:.5f}, {ev[2]:.5f}, {ev[3]:.5f}]")
    return params, hist


print("\n=== Stage 0: 空冻结集（退化为标准 NES-VMC）===")
t0 = time.time()
total0, params0, sms0 = make_ansatz(11)
params0, hist0 = run_stage(params0, sms0, np.zeros((16, 0)), n_iter=25, seed=21)
print("Stage 0 耗时 %.1fs" % (time.time() - t0))

# 提取 φ_0（冻结）
phi0, E0, lam0, v0, A0 = dfl.extract_frozen_state(sms0, params0, all_states, H_dense)
print("\n冻结态 φ_0: E0 = %.8f (目标 -1.05434745)  偏差 %.2e" % (E0, E0 - E_fcis[0]))
# 与 FCI 基态重叠
_, fci_vecs = np.linalg.eigh(H_dense)
ov0 = float(np.abs(fci_vecs[:, 0].conj() @ phi0) ** 2)
print("|⟨φ0|ψ0^FCI⟩|^2 =", ov0)

# 检查投影干净：Q0 φ0 ≈ 0
Qphi = phi0 - phi0 * (phi0.conj() @ phi0)
print("||Q0 φ0|| =", float(np.linalg.norm(Qphi)))

print("\n=== Stage 1: 冻结 φ_0，训练 deflated NES（fresh init）===")
t1 = time.time()
total1, params1, sms1 = make_ansatz(123)
F = dfl.orthonormalize_frozen([phi0])
print("frozen matrix shape", F.shape)
params1, hist1 = run_stage(params1, sms1, F, n_iter=40, seed=99)
print("Stage 1 耗时 %.1fs" % (time.time() - t1))

# 抽取 Stage 1 物理态（在 χ_full 投影子空间上）
A_full = np.asarray(dfl.amplitudes_full(sms1, params1, jnp.asarray(all_states)))
chi_full = A_full - F @ (F.conj().T @ A_full)   # Q0 ψ
phis, lams = dfl.extract_physical_states(chi_full, H_dense)
print("\nStage 1 重建物理态能量（应在 Q0H 内，最低 ≈ E1=-0.95790573）:")
for k in range(K):
    ov = float(np.abs(phi0.conj() @ phis[k]) ** 2)
    print(f"  φ{k}: E={lams[k]:.8f}  |⟨φ0|φ{k}⟩|²={ov:.2e}")

print("\nDone.")