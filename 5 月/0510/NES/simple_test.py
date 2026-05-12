"""
简化的 NES-VMC 测试脚本
直接测试基本功能，不涉及复杂的采样
"""

import jax
import jax.numpy as jnp
import netket.experimental as nkx
from pyscf import gto, scf, fci
from flax import nnx


bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()

print("FCI 基准能量:")
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

ha = nkx.operator.from_pyscf_molecule(mol)


class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        return jnp.squeeze(self.output(h))


class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42+i))
            for i in range(n_states)
        ]

    def __call__(self, x_batch):
        K = self.n_states
        M = []
        for i in range(K):
            for j in range(K):
                M.append(self.single_ansatz_list[j](x_batch[i]))
        M_matrix = jnp.stack(M).reshape(K, K)
        return jnp.linalg.det(M_matrix), M_matrix


def compute_local_energy_matrix(ha, total_ansatz, x_batch):
    psi_total, M_matrix = total_ansatz(x_batch)
    det_M = jnp.linalg.det(M_matrix)

    K = total_ansatz.n_states
    H_Psi = []
    for i in range(K):
        row = []
        for j in range(K):
            x_primes, mels = ha.get_conn(x_batch[i])
            psi_values = jax.vmap(lambda x: total_ansatz.single_ansatz_list[j](x))(x_primes)
            H_psi = jnp.sum(mels * psi_values)
            row.append(H_psi)
        H_Psi.append(row)

    H_Psi_matrix = jnp.array(H_Psi).reshape(K, K)

    eps = 1e-8
    if jnp.abs(det_M) < eps:
        M_reg = M_matrix + eps * jnp.eye(K)
    else:
        M_reg = M_matrix

    E_L = jnp.linalg.solve(M_reg, H_Psi_matrix.T).T
    return E_L, psi_total, det_M


K = 2
n_spin_orbitals = 4

total_ansatz = NESTotalAnsatz(n_spin_orbitals, K, hidden_dim=16, rngs=nnx.Rngs(42))

print(f"\n测试不同的样本...")

valid_count = 0
E_L_list = []

for trial in range(10):
    x_batch = jnp.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    E_L, psi_total, det_M = compute_local_energy_matrix(ha, total_ansatz, x_batch)

    abs_det = jnp.abs(det_M)
    is_finite = jnp.isfinite(E_L).all()

    print(f"\nTrial {trial}:")
    print(f"  det(M) = {det_M}")
    print(f"  |det| = {abs_det}")
    print(f"  E_L finite = {is_finite}")
    print(f"  E_L = {E_L}")
    print(f"  Tr(E_L) = {jnp.trace(E_L)}")

    if is_finite and abs_det > 1e-8:
        valid_count += 1
        E_L_list.append(E_L)
        eigenvalues = jnp.linalg.eigvalsh(E_L)
        print(f"  本征值: {eigenvalues}")

if valid_count > 0:
    E_L_mean = jnp.mean(jnp.stack(E_L_list), axis=0)
    eigenvalues = jnp.linalg.eigvalsh(E_L_mean)
    print(f"\n平均局域能量矩阵的本征值:")
    for i, ev in enumerate(eigenvalues):
        print(f"  E_{i} = {ev:.8f} Ha")
else:
    print("\n警告：所有样本都无效！")
