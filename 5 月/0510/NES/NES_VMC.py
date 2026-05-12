"""
NES-VMC (Natural Excited State Variational Monte Carlo) 算法实现

本文件实现基于原生 JAX 和部分 NetKet 的 NES-VMC 算法，用于计算量子多体系统的激发态能量。
"""

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial
from jax import flatten_util


class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)


class NESTotalAnsatz(nnx.Module):
    """NES-VMC 总 Ansatz：K 个单态 Ansatz 的行列式"""

    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals

        key = jax.random.key(42)
        keys = jax.random.split(key, n_states)

        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(keys[i]))
            for i in range(n_states)
        ]

    def __call__(self, x_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        if x_batch.shape[0] != self.n_states:
            raise ValueError(f"x_batch.shape[0] != {self.n_states}")

        K = self.n_states
        M = []
        for i in range(K):
            for j in range(K):
                psi_i_xj = self.single_ansatz_list[j](x_batch[i])
                M.append(psi_i_xj)

        M = jnp.stack(M, axis=0)
        M_matrix = M.reshape(K, K)
        psi_total = jnp.linalg.det(M_matrix)
        return psi_total, M_matrix


def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])


def Ham_psi(ha, model, x):
    """计算 Hψ(x)"""
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def Ham_Psi(ha, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    k = total_ansatz.n_states
    if x.shape[0] != k:
        raise ValueError(f"Input array must have shape ({k},) but got shape {x.shape}")

    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi


def compute_local_energy_matrix_safe(ha, total_ansatz, x_batch, eps=1e-8):
    """
    安全计算局域能量矩阵，带数值稳定性保护
    E_L(x) = Ψ^{-1}(x) HΨ(x)
    """
    psi_total, M_matrix = total_ansatz(x_batch)

    det_M = jnp.linalg.det(M_matrix)
    abs_det = jnp.abs(det_M)

    M_reg = M_matrix
    if abs_det < eps:
        M_reg = M_matrix + eps * jnp.eye(M_matrix.shape[0])
        det_M = jnp.linalg.det(M_reg)

    H_Psi = Ham_Psi(ha, total_ansatz, x_batch)

    E_L = jnp.linalg.solve(M_reg, H_Psi.T).T

    return E_L, psi_total, det_M


def compute_local_energy_matrix_from_params_safe(ha, graphdef, params, x_batch, n_states, eps=1e-8):
    """从参数计算局域能量矩阵，带数值稳定性保护"""
    merged_model = nnx.merge(graphdef, params)

    K = n_states
    M = []
    for i in range(K):
        for j in range(K):
            psi_i_xj = merged_model.single_ansatz_list[j](x_batch[i])
            M.append(psi_i_xj)

    M_matrix = jnp.stack(M).reshape(K, K)

    det_M = jnp.linalg.det(M_matrix)
    abs_det = jnp.abs(det_M)

    M_reg = M_matrix
    if abs_det < eps:
        M_reg = M_matrix + eps * jnp.eye(K)

    H_Psi = []
    for i in range(K):
        row = []
        for j in range(K):
            x_primes, mels = ha.get_conn(x_batch[i])
            psi_values = jax.vmap(lambda x: merged_model.single_ansatz_list[j](x))(x_primes)
            H_psi = jnp.sum(mels * psi_values)
            row.append(H_psi)
        H_Psi.append(row)

    H_Psi_matrix = jnp.array(H_Psi).reshape(K, K)

    E_L = jnp.linalg.solve(M_reg, H_Psi_matrix.T).T
    psi_total = jnp.linalg.det(M_matrix)

    return E_L, psi_total, det_M


def sample_nes_batches(sampler, machine_fn, params, sampler_state, n_samples, K, n_spin_orbitals):
    """从扩展希尔伯特空间采样 NES 批次"""
    samples_list = []

    for _ in range(n_samples):
        sample, sampler_state = sampler.sample(machine_fn, params, state=sampler_state)
        samples_list.append(sample.reshape(-1, n_spin_orbitals))

    all_samples = jnp.stack(samples_list).reshape(-1, n_spin_orbitals)

    x_batches = []
    for i in range(n_samples // K):
        indices = jnp.arange(i * K, (i + 1) * K)
        batch = all_samples[indices]
        x_batches.append(batch)

    return jnp.stack(x_batches), sampler_state


def compute_nes_loss_and_grad(ha, graphdef, params, sigma_batches, n_states, eps=1e-8):
    """计算 NES-VMC 的损失函数和梯度，带数值稳定性保护"""
    E_L_list = []
    valid_count = 0

    for i in range(sigma_batches.shape[0]):
        E_L, psi, det_M = compute_local_energy_matrix_from_params_safe(
            ha, graphdef, params, sigma_batches[i], n_states, eps
        )

        abs_det = jnp.abs(det_M)
        is_valid = jnp.isfinite(E_L).all() and abs_det > eps

        if is_valid:
            E_L_list.append(E_L)
            valid_count += 1

    if valid_count == 0:
        print("Warning: No valid samples!")
        return jnp.array(np.nan), jax.tree.map(lambda x: jnp.zeros_like(x), params), jnp.array([])

    E_L_batch = jnp.stack(E_L_list)
    loss = jnp.trace(jnp.mean(E_L_batch, axis=0))

    grads = []
    for i in range(len(E_L_list)):
        def loss_per_sample(p):
            E_L_s, _, _ = compute_local_energy_matrix_from_params_safe(
                ha, graphdef, p, sigma_batches[i], n_states, eps
            )
            return jnp.trace(E_L_s)

        grad_i = jax.grad(loss_per_sample, holomorphic=True)(params)
        grads.append(grad_i)

    grad = jax.tree.map(
        lambda *x: jnp.mean(jnp.stack(x), axis=0),
        *grads
    )

    return loss, grad, E_L_batch


def extract_eigenvalues(E_L_matrices):
    """从局域能量矩阵提取本征值（激发态能量）"""
    if len(E_L_matrices) == 0:
        return jnp.array([jnp.nan, jnp.nan]), None, None

    E_L_mean = jnp.mean(E_L_matrices, axis=0)

    if not jnp.isfinite(E_L_mean).all():
        return jnp.array([jnp.nan, jnp.nan]), None, E_L_mean

    eigenvalues, eigenvectors = jnp.linalg.eigh(E_L_mean)
    return eigenvalues, eigenvectors, E_L_mean


if __name__ == "__main__":
    print("=" * 60)
    print("NES-VMC 算法测试")
    print("=" * 60)

    bond_length = 1.4
    geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
    mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)

    cisolver = fci.FCI(mf)
    cisolver.nroots = 4
    E_fcis, fcivec = cisolver.kernel()

    print("\nFCI 基准能量:")
    for i, e in enumerate(E_fcis):
        exc = (e - E_fcis[0]) * 27.2114
        print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

    ha = nkx.operator.from_pyscf_molecule(mol)

    K = 2
    n_spin_orbitals = 4
    hidden_dim = 16

    total_ansatz = NESTotalAnsatz(
        n_spin_orbitals=n_spin_orbitals,
        n_states=K,
        hidden_dim=hidden_dim,
        rngs=nnx.Rngs(42)
    )

    print(f"\nNES-VMC 模型已创建:")
    print(f"  - K (态数量): {K}")
    print(f"  - n_spin_orbitals: {n_spin_orbitals}")
    print(f"  - hidden_dim: {hidden_dim}")

    x_batch = jnp.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ])

    psi_total, M_matrix = total_ansatz(x_batch)
    print(f"\n测试样本 x_batch[0]: {x_batch[0]}")
    print(f"测试样本 x_batch[1]: {x_batch[1]}")
    print(f"M 矩阵:\n{M_matrix}")
    print(f"行列式 ψ_total: {psi_total}")

    E_L, psi, det_M = compute_local_energy_matrix_safe(ha, total_ansatz, x_batch)
    print(f"\n局域能量矩阵 E_L:\n{E_L}")
    print(f"迹 Tr(E_L): {jnp.trace(E_L)}")
    print(f"行列式 det_M: {det_M}")

    eigenvalues, eigenvectors = extract_eigenvalues(jnp.stack([E_L]))
    print(f"\n本征值（提取的激发态能量）:")
    for i, ev in enumerate(eigenvalues):
        print(f"  E_{i} = {ev:.8f} Ha")

    print("\n" + "=" * 60)
    print("基础测试完成!")
    print("=" * 60)
