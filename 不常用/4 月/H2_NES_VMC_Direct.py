"""
NES-VMC: Neural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

H2分子激发态计算 - 直接优化版本
K=2: 计算基态和第一激发态

对于小系统（Hilbert空间维度小），直接优化波函数系数比神经网络更有效
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


class DirectWaveFunction:
    """直接参数化的波函数
    
    对于小系统，直接优化每个组态的波函数系数
    """
    def __init__(self, n_states: int, seed: int = 42):
        self.n_states = n_states
        self.key = jax.random.PRNGKey(seed)
        
        self.coefficients = jax.random.normal(
            self.key, 
            shape=(n_states,), 
            dtype=complex
        )
        
        norm = jnp.sqrt(jnp.sum(jnp.abs(self.coefficients) ** 2))
        self.coefficients = self.coefficients / norm
        
    def __call__(self, state_index: int) -> jnp.array:
        """返回第 state_index 个组态的波函数值"""
        return self.coefficients[state_index]
    
    def get_all_coefficients(self) -> jnp.array:
        """返回所有系数"""
        return self.coefficients
    
    def set_coefficients(self, coeffs: jnp.array):
        """设置系数"""
        self.coefficients = coeffs


class NESVMCDirect:
    """NES-VMC 直接优化版本
    
    对于小系统，直接优化波函数系数
    使用变分原理和正交约束
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 2):
        self.bond_length = bond_length
        self.n_states = n_states
        
        self._setup_molecule()
        self._setup_wavefunctions()
        
    def _setup_molecule(self):
        """设置分子系统"""
        geometry = [
            ('H', (0., 0., 0.)),
            ('H', (self.bond_length, 0., 0.)),
        ]
        
        print("=" * 70)
        print("H2分子系统设置 (直接优化版本, K=2)")
        print("=" * 70)
        print(f"键长: {self.bond_length} 埃")
        
        mol = gto.M(atom=geometry, basis='STO-3G')
        mf = scf.RHF(mol).run(verbose=0)
        self.E_hf = mf.e_tot
        print(f"\nHartree-Fock能量: {self.E_hf:.8f} Ha")
        
        cisolver = fci.FCI(mf)
        cisolver.nroots = 4
        self.E_fcis, self.fci_vectors = cisolver.kernel()
        
        print(f"\nFCI能级 (参考值):")
        print("-" * 70)
        for i, e in enumerate(self.E_fcis[:self.n_states]):
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
        
        self.all_states = self.hi.all_states()
        self.n_basis = len(self.all_states)
        
        print(f"\nHilbert空间维度: {self.hi.n_states}")
        print(f"基组大小: {self.n_basis}")
        print(f"计算态数: {self.n_states}")
        
        self.H_matrix = self._construct_hamiltonian_matrix()
        print(f"\n哈密顿量矩阵形状: {self.H_matrix.shape}")
        
    def _construct_hamiltonian_matrix(self) -> jnp.array:
        """构造哈密顿量矩阵"""
        H = np.zeros((self.n_basis, self.n_basis), dtype=complex)
        
        for i, state_i in enumerate(self.all_states):
            x_primes, mels = self.ha.get_conn(state_i)
            
            for j, state_j in enumerate(x_primes):
                for k, basis_state in enumerate(self.all_states):
                    if np.array_equal(state_j, basis_state):
                        H[i, k] += mels[j]
                        break
        
        return jnp.array(H)
    
    def _setup_wavefunctions(self):
        """设置波函数"""
        self.wavefunctions = []
        
        for i in range(self.n_states):
            wf = DirectWaveFunction(self.n_basis, seed=42 + i * 100)
            
            if i == 0:
                hf_state = np.array([0, 1, 0, 1])
                for j, state in enumerate(self.all_states):
                    if np.array_equal(state, hf_state):
                        init_coeffs = jnp.zeros(self.n_basis, dtype=complex)
                        init_coeffs = init_coeffs.at[j].set(1.0)
                        wf.set_coefficients(init_coeffs)
                        break
            
            self.wavefunctions.append(wf)
    
    def compute_energy(self, coeffs: jnp.array) -> jnp.array:
        """计算能量期望值: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩"""
        norm_sq = jnp.sum(jnp.abs(coeffs) ** 2)
        energy = jnp.conj(coeffs) @ self.H_matrix @ coeffs / norm_sq
        return energy
    
    def compute_overlap(self, coeffs1: jnp.array, coeffs2: jnp.array) -> jnp.array:
        """计算重叠: ⟨ψ₁|ψ₂⟩"""
        return jnp.conj(coeffs1) @ coeffs2
    
    def gram_schmidt(self, coeffs_list: list) -> list:
        """Gram-Schmidt 正交化"""
        orthogonal_coeffs = []
        
        for i, coeffs in enumerate(coeffs_list):
            v = coeffs
            
            for prev_coeffs in orthogonal_coeffs:
                overlap = self.compute_overlap(prev_coeffs, v)
                v = v - overlap * prev_coeffs
            
            norm = jnp.sqrt(jnp.sum(jnp.abs(v) ** 2))
            v = v / norm
            
            orthogonal_coeffs.append(v)
        
        return orthogonal_coeffs
    
    def train_state(self, coeffs: jnp.array, prev_coeffs_list: list,
                   n_iterations: int = 1000, lr: float = 0.01,
                   overlap_weight: float = 100.0) -> tuple:
        """训练单个态"""
        
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=200,
            decay_rate=0.95
        )
        
        optimizer = optax.adam(schedule)
        opt_state = optimizer.init(coeffs)
        
        energy_history = []
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for step in pbar:
            def loss_fn(c):
                energy = self.compute_energy(c)
                
                overlap_loss = 0.0
                for prev_c in prev_coeffs_list:
                    overlap = self.compute_overlap(prev_c, c)
                    overlap_loss = overlap_loss + jnp.abs(overlap) ** 2
                
                total_loss = jnp.real(energy) + overlap_weight * overlap_loss
                return total_loss, (energy, overlap_loss)
            
            (loss, (energy, overlap_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(coeffs)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            coeffs = optax.apply_updates(coeffs, updates)
            
            norm = jnp.sqrt(jnp.sum(jnp.abs(coeffs) ** 2))
            coeffs = coeffs / norm
            
            energy_history.append(float(jnp.real(energy)))
            
            if step % 50 == 0:
                overlap_val = float(jnp.real(overlap_loss)) if prev_coeffs_list else 0.0
                pbar.set_postfix({
                    'E': f'{float(jnp.real(energy)):.6f}',
                    'Overlap': f'{overlap_val:.6f}'
                })
        
        return coeffs, energy_history
    
    def train(self, n_iterations: int = 1000, lr: float = 0.01, overlap_weight: float = 100.0):
        """训练所有态"""
        print("\n" + "=" * 70)
        print("开始 NES-VMC 训练 (直接优化)")
        print("=" * 70)
        print(f"训练参数:")
        print(f"  迭代次数: {n_iterations}")
        print(f"  初始学习率: {lr}")
        print(f"  正交约束权重: {overlap_weight}")
        
        self.energy_histories = []
        self.final_energies = []
        self.final_coeffs = []
        
        for i in range(self.n_states):
            print(f"\n{'='*70}")
            print(f"训练第 {i} 个态 {'(基态)' if i == 0 else '(第1激发态)'}")
            print(f"{'='*70}")
            
            prev_coeffs = self.final_coeffs
            
            coeffs = self.wavefunctions[i].get_all_coefficients()
            
            coeffs, energy_history = self.train_state(
                coeffs,
                prev_coeffs,
                n_iterations=n_iterations,
                lr=lr,
                overlap_weight=overlap_weight
            )
            
            self.energy_histories.append(energy_history)
            
            final_energy = np.mean(energy_history[-100:])
            self.final_energies.append(final_energy)
            self.final_coeffs.append(coeffs)
            
            print(f"\n第 {i} 个态训练完成:")
            print(f"  最终能量: {final_energy:.8f} Ha")
            print(f"  FCI 参考: {self.E_fcis[i]:.8f} Ha")
            print(f"  误差: {abs(final_energy - self.E_fcis[i]):.6f} Ha")
            
            if i > 0:
                overlap = float(jnp.abs(self.compute_overlap(self.final_coeffs[0], coeffs)))
                print(f"  与基态重叠: {overlap:.6f}")
        
        self._print_summary()
        
    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 70)
        print("训练总结")
        print("=" * 70)
        
        print("\n能级比较:")
        print("-" * 70)
        print(f"{'态':<15} {'NES-VMC (Ha)':<20} {'FCI (Ha)':<20} {'误差':<15}")
        print("-" * 70)
        
        for i, (nes_energy, fci_energy) in enumerate(zip(self.final_energies, self.E_fcis[:self.n_states])):
            error = abs(nes_energy - fci_energy)
            state_name = f"E{i} (基态)" if i == 0 else f"E{i} (激发态)"
            print(f"{state_name:<15} {nes_energy:<20.8f} {fci_energy:<20.8f} {error:<15.6f}")
        
        if self.n_states >= 2:
            print("\n激发能:")
            print("-" * 70)
            exc_nes = (self.final_energies[1] - self.final_energies[0]) * 27.2114
            exc_fci = (self.E_fcis[1] - self.E_fcis[0]) * 27.2114
            exc_error = abs(exc_nes - exc_fci)
            print(f"E1-E0: {exc_nes:.4f} eV (NES-VMC) vs {exc_fci:.4f} eV (FCI)")
            print(f"激发能误差: {exc_error:.4f} eV ({exc_error/exc_fci*100:.2f}%)")
        
        print("\n态正交性检查:")
        print("-" * 70)
        for i in range(self.n_states):
            for j in range(i+1, self.n_states):
                overlap = float(jnp.abs(self.compute_overlap(self.final_coeffs[i], self.final_coeffs[j])))
                print(f"|⟨E{i}|E{j}⟩| = {overlap:.6f}")
        
        print("\n波函数系数:")
        print("-" * 70)
        for i, coeffs in enumerate(self.final_coeffs):
            print(f"\n态 E{i}:")
            print("组态      系数 (实部+虚部)      |ψ|²")
            print("-" * 50)
            for j, state in enumerate(self.all_states):
                c = coeffs[j]
                prob = float(jnp.abs(c) ** 2)
                print(f"{state}  {c.real:+.6f}{c.imag:+.6f}j  {prob:.6f}")
    
    def plot_results(self, save_path: str = None):
        """绘制结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        for i, history in enumerate(self.energy_histories):
            label = f'E{i} (基态)' if i == 0 else f'E{i} (激发态)'
            ax1.plot(history, label=f'{label} NES-VMC', linewidth=2)
            ax1.axhline(y=self.E_fcis[i], color=f'C{i}', linestyle='--', 
                       alpha=0.5, linewidth=1.5, label=f'{label} FCI')
        
        ax1.set_xlabel('迭代步数', fontsize=12)
        ax1.set_ylabel('能量', fontsize=12)
        ax1.set_title('能量收敛曲线', fontsize=14)
        ax1.legend(fontsize=9, loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        states = [f'E{i}\n({"基态" if i == 0 else "激发态"})' for i in range(self.n_states)]
        x = np.arange(len(states))
        width = 0.35
        
        nes_energies = np.array(self.final_energies)
        fci_energies = np.array(self.E_fcis[:self.n_states])
        
        bars1 = ax2.bar(x - width/2, nes_energies, width, label='NES-VMC', 
                       alpha=0.8, color='steelblue')
        bars2 = ax2.bar(x + width/2, fci_energies, width, label='FCI', 
                       alpha=0.8, color='coral')
        
        ax2.set_xlabel('量子态', fontsize=12)
        ax2.set_ylabel('能量', fontsize=12)
        ax2.set_title('NES-VMC vs FCI 能量对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(states)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.annotate(f'{height:.4f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存至: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("NES-VMC: H2分子激发态计算 (直接优化版本, K=2)")
    print("=" * 70)
    
    trainer = NESVMCDirect(
        bond_length=1.4,
        n_states=2
    )
    
    trainer.train(
        n_iterations=1500,
        lr=0.02,
        overlap_weight=200.0
    )
    
    trainer.plot_results()


if __name__ == "__main__":
    main()
