"""
NES-VMC 完整训练脚本
包含完整的训练循环和可视化
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
import matplotlib.pyplot as plt
from tqdm import tqdm


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


def Ham_psi(ha, model, x):
    """计算 Hψ(x)"""
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def Ham_Psi(ha, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    k = total_ansatz.n_states
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)

    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi


def compute_local_energy_matrix(ha, total_ansatz, x_batch):
    """计算局域能量矩阵 E_L(x)"""
    psi_total, M_matrix = total_ansatz(x_batch)
    H_Psi = Ham_Psi(ha, total_ansatz, x_batch)
    E_L = jnp.linalg.solve(M_matrix, H_Psi.T).T
    return E_L, psi_total


def compute_local_energy_matrix_from_params(ha, total_ansatz, params, x_batch):
    """从参数计算局域能量矩阵"""
    graphdef, state = nnx.split(total_ansatz)
    merged_model = nnx.merge(graphdef, params)

    K = total_ansatz.n_states
    M = []
    for i in range(K):
        for j in range(K):
            psi_i_xj = merged_model.single_ansatz_list[j](x_batch[i])
            M.append(psi_i_xj)

    M_matrix = jnp.stack(M).reshape(K, K)

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

    E_L = jnp.linalg.solve(M_matrix, H_Psi_matrix.T).T
    psi_total = jnp.linalg.det(M_matrix)

    return E_L, psi_total


def compute_nes_loss_and_grad(ha, total_ansatz, params, sigma):
    """计算 NES-VMC 的损失函数和梯度"""
    E_L_list = []

    for i in range(sigma.shape[0]):
        E_L, _ = compute_local_energy_matrix_from_params(ha, total_ansatz, params, sigma[i])
        E_L_list.append(E_L)

    E_L_batch = jnp.stack(E_L_list)
    loss = jnp.trace(jnp.mean(E_L_batch, axis=0))

    grads = []
    for i in range(sigma.shape[0]):
        def loss_per_sample(p):
            E_L_s, _ = compute_local_energy_matrix_from_params(ha, total_ansatz, p, sigma[i])
            return jnp.trace(E_L_s)

        grad_i = jax.grad(loss_per_sample, holomorphic=True)(params)
        grads.append(grad_i)

    grad = jax.tree.map(
        lambda *x: jnp.mean(jnp.stack(x), axis=0),
        *grads
    )

    return loss, grad, E_L_batch


def extract_eigenvalues(E_L_matrices):
    """从局域能量矩阵提取本征值"""
    E_L_mean = jnp.mean(E_L_matrices, axis=0)
    eigenvalues, eigenvectors = jnp.linalg.eigh(E_L_mean)
    return eigenvalues, eigenvectors, E_L_mean


def sample_batch(sampler, machine, params, sampler_state, n_samples, K):
    """从扩展希尔伯特空间采样"""
    samples_list = []
    for _ in range(n_samples):
        samples_single, sampler_state = sampler.sample(
            machine, params, state=sampler_state
        )
        samples_list.append(samples_single.reshape(-1, 4))

    all_samples = jnp.stack(samples_list)

    indices = jnp.arange(K)
    batch_indices = indices[:, None] + jnp.arange(K) * 0
    batches = []
    for i in range(K):
        batch_samples = all_samples[i::K][:n_samples // K]
        batches.append(batch_samples)

    x_batches = jnp.stack([b[0] for b in batches])
    for b in batches[1:]:
        for j in range(len(b)):
            x_batches = jnp.concatenate([
                x_batches[:j*K+i],
                b[j:j+1],
                x_batches[j*K+i:]
            ], axis=0)
            if j*K+i+1 < len(all_samples):
                break

    return all_samples[:n_samples], sampler_state


def main():
    print("=" * 60)
    print("NES-VMC 算法完整训练")
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

    hi = nkx.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1, 1)
    )

    edges = [(0, 1), (2, 3)]
    g = nk.graph.Graph(edges=edges)
    single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
    sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)

    K = 2
    n_spin_orbitals = 4
    hidden_dim = 16
    learning_rate = 0.01
    n_iterations = 100
    n_samples = 1008

    print(f"\n训练参数:")
    print(f"  - K (态数量): {K}")
    print(f"  - 学习率: {learning_rate}")
    print(f"  - 迭代次数: {n_iterations}")
    print(f"  - 样本数: {n_samples}")

    total_ansatz = NESTotalAnsatz(
        n_spin_orbitals=n_spin_orbitals,
        n_states=K,
        hidden_dim=hidden_dim,
        rngs=nnx.Rngs(42)
    )

    graphdef, params = nnx.split(total_ansatz)

    @jax.jit
    def machine_fn(params, sigma):
        model = nnx.merge(graphdef, params)
        return model(sigma)

    sampler_state = sampler.init_state(machine_fn, params, seed=42)

    optimizer = optax.sgd(learning_rate=learning_rate)
    opt_state = optimizer.init(params)

    history = {
        'step': [],
        'loss': [],
        'eigenvalues': [],
        'error_0': [],
        'error_1': []
    }

    print("\n" + "=" * 60)
    print("开始训练")
    print("=" * 60)

    for step in tqdm(range(n_iterations)):
        sampler_state = sampler.reset(machine_fn, params, sampler_state)

        samples_list = []
        for _ in range(n_samples // sampler.n_chains):
            samples_single, sampler_state = sampler.sample(
                machine_fn, params, state=sampler_state
            )
            samples_list.append(samples_single)

        samples = jnp.concatenate(samples_list, axis=0)
        samples = samples.reshape(-1, n_spin_orbitals)

        loss, grad, E_L_batch = compute_nes_loss_and_grad(
            ha, total_ansatz, params, samples
        )

        eigenvalues, eigenvectors, E_L_mean = extract_eigenvalues(E_L_batch)

        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)

        if step % 10 == 0:
            history['step'].append(step)
            history['loss'].append(float(loss.real))
            history['eigenvalues'].append([float(e) for e in eigenvalues])
            history['error_0'].append(float(abs(eigenvalues[0] - E_fcis[0])))
            history['error_1'].append(float(abs(eigenvalues[1] - E_fcis[1])))

            print(f"\nStep {step:3d} | Loss: {loss.real:.8f}")
            print(f"  本征值: E_0 = {eigenvalues[0]:.8f} (FCI: {E_fcis[0]:.8f}, Error: {abs(eigenvalues[0] - E_fcis[0]):.6f})")
            print(f"          E_1 = {eigenvalues[1]:.8f} (FCI: {E_fcis[1]:.8f}, Error: {abs(eigenvalues[1] - E_fcis[1]):.6f})")

    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history['step'], history['loss'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Iteration Step')
    axes[0, 0].set_ylabel('Loss (Tr(E_L))')
    axes[0, 0].set_title('NES-VMC Loss Convergence')
    axes[0, 0].grid(True, alpha=0.3)

    eigenvalues_array = jnp.array(history['eigenvalues'])
    for i in range(K):
        axes[0, 1].plot(history['step'], eigenvalues_array[:, i],
                        label=f'E_{i}', linewidth=2)
    axes[0, 1].axhline(y=E_fcis[0], color='r', linestyle='--',
                       label=f'FCI E0: {E_fcis[0]:.4f}')
    axes[0, 1].axhline(y=E_fcis[1], color='g', linestyle='--',
                       label=f'FCI E1: {E_fcis[1]:.4f}')
    axes[0, 1].set_xlabel('Iteration Step')
    axes[0, 1].set_ylabel('Energy (Ha)')
    axes[0, 1].set_title('Eigenvalue Convergence')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(history['step'], history['error_0'], 'b-',
                         linewidth=2, label='Error E_0')
    axes[1, 0].semilogy(history['step'], history['error_1'], 'r-',
                         linewidth=2, label='Error E_1')
    axes[1, 0].set_xlabel('Iteration Step')
    axes[1, 0].set_ylabel('Absolute Error (Ha)')
    axes[1, 0].set_title('Error vs FCI Reference')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, which='both')

    axes[1, 1].bar(['FCI E0', 'NES E0', 'FCI E1', 'NES E1'],
                   [E_fcis[0], eigenvalues_array[-1, 0],
                    E_fcis[1], eigenvalues_array[-1, 1]],
                   color=['red', 'blue', 'red', 'blue'],
                   alpha=0.7)
    axes[1, 1].set_ylabel('Energy (Ha)')
    axes[1, 1].set_title('Final Energy Comparison')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('nes_vmc_training_results.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n结果图表已保存为 'nes_vmc_training_results.png'")


if __name__ == "__main__":
    main()
