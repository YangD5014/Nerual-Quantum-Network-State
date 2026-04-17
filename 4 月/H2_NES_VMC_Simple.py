"""
NES-VMC: Neural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

H2分子激发态计算案例 - 简化实现版本
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


class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz: 复数值前馈神经网络"""
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


def ham_psi_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算哈密顿量作用在单个波函数上"""
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def compute_local_energy_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算单个态的局部能量"""
    psi_x = model(x)
    H_psi_x = ham_psi_single(ha, model, x)
    return H_psi_x / psi_x


class NESVMCTrainer:
    """NES-VMC 训练器 - 简化版本
    
    使用顺序优化方法:
    1. 先优化基态
    2. 然后优化激发态,保持与之前态的正交性
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 3, hidden_dim: int = 12):
        self.bond_length = bond_length
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        self._setup_molecule()
        self._setup_models()
        
    def _setup_molecule(self):
        """设置分子系统"""
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
        
        print(f"\nHilbert空间维度: {self.hi.n_states}")
        print(f"计算态数: {self.n_states}")
        
    def _setup_models(self):
        """设置神经网络模型"""
        self.models = [
            SingleStateAnsatz(
                n_spin_orbitals=4,
                hidden_dim=self.hidden_dim,
                rngs=nnx.Rngs(42 + i)
            )
            for i in range(self.n_states)
        ]
        
        self.all_states = self.hi.all_states()
        
    def compute_overlap(self, model1: SingleStateAnsatz, model2: SingleStateAnsatz):
        """计算两个态之间的重叠"""
        overlaps = []
        for state in self.all_states:
            psi1 = model1(state)
            psi2 = model2(state)
            overlaps.append(jnp.conj(psi1) * psi2)
        
        return jnp.sum(jnp.array(overlaps))
    
    def compute_energy_expectation(self, model: SingleStateAnsatz):
        """计算能量期望值"""
        energies = []
        weights = []
        
        for state in self.all_states:
            psi = model(state)
            E_L = compute_local_energy_single(self.ha, model, state)
            weight = jnp.abs(psi) ** 2
            energies.append(E_L * weight)
            weights.append(weight)
        
        energies = jnp.array(energies)
        weights = jnp.array(weights)
        
        return jnp.sum(energies) / jnp.sum(weights)
    
    def orthogonalize(self, model: SingleStateAnsatz, prev_models: list, lr: float = 0.01, n_steps: int = 100):
        """正交化模型参数,使其与之前的态正交"""
        if not prev_models:
            return
        
        print("  正交化处理...")
        
        for step in range(n_steps):
            overlaps = []
            for prev_model in prev_models:
                overlap = self.compute_overlap(model, prev_model)
                overlaps.append(overlap)
            
            total_overlap = sum(abs(o) for o in overlaps)
            
            if total_overlap < 1e-6:
                break
            
            def loss_fn(model_state):
                model_obj = nnx.merge(*model_state)
                total_loss = 0.0
                for prev_model in prev_models:
                    overlap = self.compute_overlap(model_obj, prev_model)
                    total_loss = total_loss + jnp.abs(overlap) ** 2
                return total_loss
            
            model_state = nnx.split(model)
            loss, grads = jax.value_and_grad(loss_fn)(model_state)
            
            updates = optax.adam(lr).update(grads, optax.adam(lr).init(model_state))[0]
            new_model_state = optax.apply_updates(model_state, updates)
            
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, new_model_state[1])
        
        print(f"  正交化完成,总重叠: {total_overlap:.6f}")
    
    def train_single_state(self, model: SingleStateAnsatz, prev_models: list, 
                          n_iterations: int = 200, lr: float = 0.01):
        """训练单个态"""
        optimizer = optax.adam(lr)
        model_state = nnx.split(model)
        opt_state = optimizer.init(model_state)
        
        energy_history = []
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for step in pbar:
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                
                energy = self.compute_energy_expectation(model_obj)
                
                overlap_loss = 0.0
                for prev_model in prev_models:
                    overlap = self.compute_overlap(model_obj, prev_model)
                    overlap_loss = overlap_loss + jnp.abs(overlap) ** 2
                
                total_loss = jnp.real(energy) + 10.0 * overlap_loss
                
                return total_loss, energy
            
            (loss, energy), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            energy_history.append(float(jnp.real(energy)))
            
            if step % 10 == 0:
                pbar.set_postfix({'E': f'{energy:.6f}'})
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, model_state[1])
        
        return model, energy_history
    
    def train(self, n_iterations: int = 200, lr: float = 0.01):
        """训练所有态"""
        print("\n" + "=" * 60)
        print("开始 NES-VMC 训练 (顺序优化)")
        print("=" * 60)
        
        self.energy_histories = []
        self.final_energies = []
        
        for i in range(self.n_states):
            print(f"\n{'='*60}")
            print(f"训练第 {i} 个态")
            print(f"{'='*60}")
            
            prev_models = self.models[:i]
            
            if i > 0:
                self.orthogonalize(self.models[i], prev_models)
            
            self.models[i], energy_history = self.train_single_state(
                self.models[i], 
                prev_models,
                n_iterations=n_iterations,
                lr=lr
            )
            
            self.energy_histories.append(energy_history)
            
            final_energy = np.mean(energy_history[-20:])
            self.final_energies.append(final_energy)
            
            print(f"\n第 {i} 个态训练完成:")
            print(f"  最终能量: {final_energy:.8f} Ha")
            print(f"  FCI 参考: {self.E_fcis[i]:.8f} Ha")
            print(f"  误差: {abs(final_energy - self.E_fcis[i]):.6f} Ha")
        
        self._print_summary()
        
    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        
        print("\n能级比较:")
        print("-" * 60)
        print(f"{'态':<10} {'NES-VMC (Ha)':<20} {'FCI (Ha)':<20} {'误差 (Ha)':<15}")
        print("-" * 60)
        
        for i, (nes_energy, fci_energy) in enumerate(zip(self.final_energies, self.E_fcis[:self.n_states])):
            error = abs(nes_energy - fci_energy)
            print(f"E{i:<9} {nes_energy:<20.8f} {fci_energy:<20.8f} {error:<15.6f}")
        
        print("\n激发能:")
        print("-" * 60)
        for i in range(1, self.n_states):
            exc_nes = (self.final_energies[i] - self.final_energies[0]) * 27.2114
            exc_fci = (self.E_fcis[i] - self.E_fcis[0]) * 27.2114
            print(f"E{i}-E0: {exc_nes:.4f} eV (NES-VMC) vs {exc_fci:.4f} eV (FCI)")
    
    def plot_results(self, save_path: str = None):
        """绘制结果"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1 = axes[0]
        for i, history in enumerate(self.energy_histories):
            ax1.plot(history, label=f'E{i} (NES-VMC)')
            ax1.axhline(y=self.E_fcis[i], color=f'C{i}', linestyle='--', alpha=0.5, 
                       label=f'E{i} FCI参考')
        
        ax1.set_xlabel('迭代步数', fontsize=12)
        ax1.set_ylabel('能量 (Ha)', fontsize=12)
        ax1.set_title('各态能量收敛曲线', fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        states = [f'E{i}' for i in range(self.n_states)]
        x = np.arange(len(states))
        width = 0.35
        
        nes_energies = np.array(self.final_energies)
        fci_energies = np.array(self.E_fcis[:self.n_states])
        
        ax2.bar(x - width/2, nes_energies, width, label='NES-VMC', alpha=0.8)
        ax2.bar(x + width/2, fci_energies, width, label='FCI', alpha=0.8)
        
        ax2.set_xlabel('量子态', fontsize=12)
        ax2.set_ylabel('能量 (Ha)', fontsize=12)
        ax2.set_title('NES-VMC vs FCI 能量对比', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(states)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n图像已保存至: {save_path}")
        
        plt.show()


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NES-VMC: H2分子激发态计算 (简化版)")
    print("=" * 60)
    
    trainer = NESVMCTrainer(
        bond_length=1.4,
        n_states=3,
        hidden_dim=12
    )
    
    trainer.train(
        n_iterations=200,
        lr=0.01
    )
    
    trainer.plot_results()


if __name__ == "__main__":
    main()
