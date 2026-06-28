"""
NES-VMC: Neural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

H2分子激发态计算案例
"""

import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
import netket.experimental as nkx
import jax
import jax.numpy as jnp
from flax import nnx
import optax
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from netket.hilbert import SpinOrbitalFermions
except ImportError:
    from netket.experimental.hilbert import SpinOrbitalFermions

try:
    from netket.sampler import MetropolisFermionHop
except (ImportError, AttributeError):
    try:
        from netket.experimental.sampler import MetropolisFermionHop
    except (ImportError, AttributeError):
        from netket.sampler import MetropolisExchange as MetropolisFermionHop


class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz: 复数值前馈神经网络
    
    每个单态 Ansatz 独立参数化一个量子态的波函数
    """
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
    """NES-VMC 总 Ansatz: K个单态 Ansatz 的行列式
    
    总波函数: Ψ(x^1, ..., x^K) = det[ψ_i(x^j)]
    这确保了不同态之间的正交性
    """
    def __init__(self, n_spin_orbitals: int, n_states: int = 3, hidden_dim: int = 12, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals
        
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42+i))
            for i in range(n_states)
        ]

    def __call__(self, x_batch: jax.Array) -> tuple[jax.Array, jax.Array]:
        """计算总波函数值和矩阵
        
        Args:
            x_batch: 形状 [K, n_spin_orbitals] 的批量状态
            
        Returns:
            psi_total: 行列式值 (标量)
            M_matrix: K×K 矩阵
        """
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


def ham_psi_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算哈密顿量作用在单个波函数上: (Hψ_j)(x)
    
    Args:
        ha: 哈密顿量算符
        model: 单态 Ansatz
        x: 单个组态
        
    Returns:
        Hψ(x) 的值
    """
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def ham_psi_matrix(ha: nk.operator.DiscreteOperator, total_ansatz: NESTotalAnsatz, x_batch: jnp.array) -> jnp.array:
    """计算哈密顿量矩阵 HΨ
    
    (HΨ)_{ij} = (Hψ_j)(x_i)
    
    Args:
        ha: 哈密顿量算符
        total_ansatz: NES总 Ansatz
        x_batch: K个组态的批量
        
    Returns:
        K×K 矩阵
    """
    k = total_ansatz.n_states
    if x_batch.shape[0] != k:
        raise ValueError(f"Input array must have shape ({k}, n_spin_orbitals) but got shape {x_batch.shape}")
    
    H_psi_matrix = []
    for i in range(k):
        row = []
        for j in range(k):
            ele = ham_psi_single(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(ele)
        H_psi_matrix.append(row)
    
    return jnp.array(H_psi_matrix)


def compute_local_energy(ha: nk.operator.DiscreteOperator, total_ansatz: NESTotalAnsatz, x_batch: jnp.array) -> jnp.array:
    """计算局部能量
    
    E_L = Tr[Ψ^{-1} HΨ]
    
    Args:
        ha: 哈密顿量算符
        total_ansatz: NES总 Ansatz
        x_batch: K个组态的批量
        
    Returns:
        局部能量 (标量)
    """
    psi_total, psi_matrix = total_ansatz(x_batch)
    H_psi_matrix = ham_psi_matrix(ha, total_ansatz, x_batch)
    trace_matrix = jnp.linalg.inv(psi_matrix) @ H_psi_matrix
    return jnp.trace(trace_matrix)


class NESVMCSampler:
    """NES-VMC 采样器
    
    使用单个采样器同时采样 K 个组态
    """
    def __init__(self, hilbert, n_states: int, graph):
        self.hilbert = hilbert
        self.n_states = n_states
        self.graph = graph
        
        self.sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=n_states)
    
    def init_state(self, model, seed: int = 0):
        """初始化采样器状态"""
        model_state = nnx.split(model)
        
        def forward_fn(state, x):
            model_obj = nnx.merge(*state)
            if x.ndim == 1:
                x = x[jnp.newaxis, :]
            psi_total, _ = model_obj(x)
            return psi_total
        
        sampler_state = self.sampler.init_state(forward_fn, model_state, seed=seed)
        sampler_state = self.sampler.reset(forward_fn, model_state, sampler_state)
        
        return model_state, sampler_state
    
    def sample(self, model, model_state, sampler_state, chain_length: int = 100):
        """采样 K 个组态
        
        Returns:
            x_batch: 形状 [K, n_spin_orbitals] 的批量
        """
        def forward_fn(state, x):
            model_obj = nnx.merge(*state)
            psi_total, _ = model_obj(x)
            return psi_total
        
        samples, new_sampler_state = self.sampler.sample(
            forward_fn, 
            model_state, 
            state=sampler_state,
            chain_length=chain_length
        )
        
        x_batch = samples[-1]
        
        return x_batch, new_sampler_state


class NESVMCTrainer:
    """NES-VMC 训练器
    
    完整的训练流程,包括:
    - 采样
    - 损失计算
    - 参数优化
    - 能量监控
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 3, hidden_dim: int = 12):
        self.bond_length = bond_length
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        self._setup_molecule()
        self._setup_model()
        self._setup_optimizer()
        
    def _setup_molecule(self):
        """设置分子系统和参考计算"""
        geometry = [
            ('H', (0., 0., 0.)),
            ('H', (self.bond_length, 0., 0.)),
        ]
        
        print("=" * 60)
        print("H2分子系统设置")
        print("=" * 60)
        print(f"键长: {self.bond_length} 埃")
        
        mol = gto.M(atom=geometry, basis='STO-3G')
        mf = scf.RHF(mol).run(verbose=0)
        self.E_hf = mf.e_tot
        print(f"\nHartree-Fock能量: {self.E_hf:.8f} Ha")
        
        cisolver = fci.FCI(mf)
        cisolver.nroots = 4
        self.E_fcis, _ = cisolver.kernel()
        
        print(f"\nFCI能级 (参考值):")
        print("-" * 50)
        for i, e in enumerate(self.E_fcis):
            exc_energy = (e - self.E_fcis[0]) * 27.2114
            if i == 0:
                print(f"  E{i} (基态)     = {e:.8f} Ha")
            else:
                print(f"  E{i} (第{i}激发态) = {e:.8f} Ha  激发能: {exc_energy:.4f} eV")
        
        self.ha = nkx.operator.from_pyscf_molecule(mol)
        
        self.hi = SpinOrbitalFermions(
            n_orbitals=2,
            s=1/2,
            n_fermions_per_spin=(1, 1)
        )
        
        self.graph = nk.graph.Graph(edges=[(0,1), (2,3)])
        
        print(f"\nHilbert空间维度: {self.hi.n_states}")
        print(f"计算态数: {self.n_states}")
        
    def _setup_model(self):
        """设置神经网络模型和采样器"""
        self.model = NESTotalAnsatz(
            n_spin_orbitals=4,
            n_states=self.n_states,
            hidden_dim=self.hidden_dim,
            rngs=nnx.Rngs(42)
        )
        
        self.sampler = NESVMCSampler(
            self.hi, 
            self.n_states, 
            self.graph
        )
        
    def _setup_optimizer(self, learning_rate: float = 0.01):
        """设置优化器"""
        self.optimizer = optax.adam(learning_rate)
        self.model_state = nnx.split(self.model)
        self.opt_state = self.optimizer.init(self.model_state)
        
    def compute_loss_and_grad(self, model_state, x_batch):
        """计算损失函数和梯度
        
        损失函数: L = ⟨|Ψ|² E_L⟩ / ⟨|Ψ|²⟩
        """
        def loss_fn(state):
            model = nnx.merge(*state)
            psi_total, psi_matrix = model(x_batch)
            local_energy = compute_local_energy(self.ha, model, x_batch)
            
            log_prob = jnp.log(jnp.abs(psi_total) + 1e-10)
            loss = jnp.real(log_prob) * jnp.real(local_energy)
            
            return loss, (local_energy, psi_total)
        
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
        return loss, grads, aux
    
    def train_step(self, model_state, sampler_states, opt_state):
        """单步训练"""
        model = nnx.merge(*model_state)
        x_batch, new_sampler_states = self.sampler.sample(
            model, model_state, sampler_states, chain_length=10
        )
        
        loss, grads, (local_energy, psi_total) = self.compute_loss_and_grad(
            model_state, x_batch
        )
        
        updates, new_opt_state = self.optimizer.update(grads, opt_state)
        new_model_state = optax.apply_updates(model_state, updates)
        
        return new_model_state, new_sampler_states, new_opt_state, local_energy, x_batch
    
    def train(self, n_iterations: int = 500, n_therm: int = 50, save_every: int = 10):
        """训练循环
        
        Args:
            n_iterations: 总迭代次数
            n_therm: 热化步数
            save_every: 保存频率
        """
        print("\n" + "=" * 60)
        print("开始 NES-VMC 训练")
        print("=" * 60)
        print(f"总迭代次数: {n_iterations}")
        print(f"热化步数: {n_therm}")
        
        model_state, sampler_states = self.sampler.init_state(self.model, seed=42)
        
        print("\n热化阶段...")
        for i in tqdm(range(n_therm), desc="Thermalization"):
            model = nnx.merge(*model_state)
            x_batch, sampler_states = self.sampler.sample(
                model, model_state, sampler_states, chain_length=10
            )
        
        energy_history = []
        print("\n训练阶段...")
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for step in pbar:
            model_state, sampler_states, self.opt_state, local_energy, x_batch = self.train_step(
                model_state, sampler_states, self.opt_state
            )
            
            energy_history.append(float(jnp.real(local_energy)))
            
            if step % save_every == 0:
                avg_energy = np.mean(energy_history[-save_every:]) if len(energy_history) >= save_every else np.mean(energy_history)
                pbar.set_postfix({'E': f'{avg_energy:.6f}'})
        
        self.model_state = model_state
        self.energy_history = energy_history
        
        print("\n训练完成!")
        self._print_results()
        
    def _print_results(self):
        """打印最终结果"""
        print("\n" + "=" * 60)
        print("最终结果")
        print("=" * 60)
        
        final_energy = np.mean(self.energy_history[-50:])
        final_std = np.std(self.energy_history[-50:])
        
        print(f"\nNES-VMC 平均能量: {final_energy:.8f} ± {final_std:.6f} Ha")
        print(f"FCI 基态能量:     {self.E_fcis[0]:.8f} Ha")
        print(f"Hartree-Fock能量: {self.E_hf:.8f} Ha")
        print(f"\n相对 FCI 误差:    {abs(final_energy - self.E_fcis[0]):.6f} Ha")
        print(f"相对 HF 改进:     {abs(self.E_hf - self.E_fcis[0]) - abs(final_energy - self.E_fcis[0]):.6f} Ha")
        
    def plot_convergence(self, save_path: str = None):
        """绘制能量收敛曲线"""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.energy_history, 'b-', alpha=0.5, label='NES-VMC')
        plt.axhline(y=self.E_fcis[0], color='r', linestyle='--', label=f'FCI基态 ({self.E_fcis[0]:.6f} Ha)')
        plt.axhline(y=self.E_hf, color='g', linestyle=':', label=f'Hartree-Fock ({self.E_hf:.6f} Ha)')
        
        final_energy = np.mean(self.energy_history[-50:])
        plt.axhline(y=final_energy, color='b', linestyle='-', alpha=0.8, 
                   label=f'NES-VMC平均 ({final_energy:.6f} Ha)')
        
        plt.xlabel('迭代步数', fontsize=12)
        plt.ylabel('能量 (Ha)', fontsize=12)
        plt.title('NES-VMC 能量收敛曲线 - H2分子', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存至: {save_path}")
        
        plt.show()
        
    def compute_excitation_energies(self, n_samples: int = 100):
        """计算激发能
        
        通过对角化局部能量矩阵来获得各个态的能量
        """
        print("\n" + "=" * 60)
        print("计算激发态能量")
        print("=" * 60)
        
        model = nnx.merge(*self.model_state)
        model_state, sampler_states = self.sampler.init_state(self.model, seed=123)
        
        energy_matrices = []
        
        for _ in range(n_samples):
            x_batch, sampler_states = self.sampler.sample(
                model, model_state, sampler_states, chain_length=10
            )
            
            _, psi_matrix = model(x_batch)
            H_psi_matrix = ham_psi_matrix(self.ha, model, x_batch)
            
            H_eff = jnp.linalg.inv(psi_matrix) @ H_psi_matrix
            energy_matrices.append(np.array(H_eff))
        
        avg_H_eff = np.mean(energy_matrices, axis=0)
        eigenvalues = np.linalg.eigvalsh(avg_H_eff)
        
        print("\n计算得到的能级:")
        print("-" * 50)
        for i, e in enumerate(sorted(eigenvalues)):
            exc_energy = (e - sorted(eigenvalues)[0]) * 27.2114
            if i == 0:
                print(f"  E{i} (基态)     = {e:.8f} Ha")
            else:
                print(f"  E{i} (第{i}激发态) = {e:.8f} Ha  激发能: {exc_energy:.4f} eV")
        
        print("\nFCI 参考能级:")
        print("-" * 50)
        for i, e in enumerate(self.E_fcis[:len(eigenvalues)]):
            exc_energy = (e - self.E_fcis[0]) * 27.2114
            if i == 0:
                print(f"  E{i} (基态)     = {e:.8f} Ha")
            else:
                print(f"  E{i} (第{i}激发态) = {e:.8f} Ha  激发能: {exc_energy:.4f} eV")
        
        return eigenvalues


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NES-VMC: H2分子激发态计算")
    print("=" * 60)
    
    trainer = NESVMCTrainer(
        bond_length=1.4,
        n_states=3,
        hidden_dim=12
    )
    
    trainer.train(
        n_iterations=500,
        n_therm=50,
        save_every=10
    )
    
    trainer.plot_convergence()
    
    trainer.compute_excitation_energies(n_samples=50)


if __name__ == "__main__":
    main()
