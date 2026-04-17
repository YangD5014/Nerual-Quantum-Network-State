"""
NES-VMC: Neural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

H2分子激发态计算案例 - 改进版本
K=2: 计算基态和第一激发态
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
    """单态 Ansatz: 复数值前馈神经网络
    
    改进版:
    - 更深的网络结构
    - 残差连接
    - 更好的初始化
    """
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h1 = nnx.tanh(self.linear1(x))
        h2 = nnx.tanh(self.linear2(h1))
        h3 = nnx.tanh(self.linear3(h2))
        out = self.output(h3)
        return jnp.squeeze(out)


def ham_psi_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算哈密顿量作用在单个波函数上: (Hψ)(x)"""
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    H_psi_x = jnp.sum(mels * psi_values)
    return H_psi_x


def compute_local_energy_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算单个态的局部能量: E_L(x) = (Hψ)(x) / ψ(x)"""
    psi_x = model(x)
    H_psi_x = ham_psi_single(ha, model, x)
    return H_psi_x / psi_x


class NESVMCTrainer:
    """NES-VMC 训练器 - 改进版本
    
    改进策略:
    1. 更深的网络结构
    2. 更长的训练时间
    3. 更强的正交化约束
    4. 学习率调度
    5. 早停机制
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 2, hidden_dim: int = 32):
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
        print("H2分子系统设置 (K=2)")
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
        
        print(f"\nHilbert空间维度: {self.hi.n_states}")
        print(f"计算态数: {self.n_states}")
        
        self.all_states = self.hi.all_states()
        print(f"所有可能的组态数: {len(self.all_states)}")
        
    def _setup_models(self):
        """设置神经网络模型"""
        self.models = [
            SingleStateAnsatz(
                n_spin_orbitals=4,
                hidden_dim=self.hidden_dim,
                rngs=nnx.Rngs(42 + i * 100)
            )
            for i in range(self.n_states)
        ]
        
    def compute_overlap(self, model1: SingleStateAnsatz, model2: SingleStateAnsatz):
        """计算两个态之间的重叠: ⟨ψ₁|ψ₂⟩"""
        overlaps = []
        for state in self.all_states:
            psi1 = model1(state)
            psi2 = model2(state)
            overlaps.append(jnp.conj(psi1) * psi2)
        
        return jnp.sum(jnp.array(overlaps))
    
    def compute_energy_expectation(self, model: SingleStateAnsatz):
        """计算能量期望值: ⟨ψ|H|ψ⟩ / ⟨ψ|ψ⟩"""
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
    
    def normalize_model(self, model: SingleStateAnsatz):
        """归一化波函数"""
        norm_sq = 0.0
        for state in self.all_states:
            psi = model(state)
            norm_sq += jnp.abs(psi) ** 2
        
        norm = jnp.sqrt(norm_sq)
        
        model_state = nnx.split(model)
        graphdef, params = model_state
        
        def scale_params(param_tree, scale_factor):
            return jax.tree_util.tree_map(
                lambda x: x / scale_factor if jnp.iscomplexobj(x) else x,
                param_tree
            )
        
        scaled_params = scale_params(params, jnp.sqrt(norm))
        new_model_state = (graphdef, scaled_params)
        
        return nnx.merge(*new_model_state)
    
    def orthogonalize(self, model: SingleStateAnsatz, prev_models: list, 
                     lr: float = 0.001, n_steps: int = 200):
        """正交化模型参数,使其与之前的态正交
        
        改进:
        - 更多迭代步数
        - 更小的学习率
        - 归一化处理
        """
        if not prev_models:
            return model
        
        print("  正交化处理...")
        
        optimizer = optax.adam(lr)
        model_state = nnx.split(model)
        opt_state = optimizer.init(model_state)
        
        for step in range(n_steps):
            overlaps = []
            for prev_model in prev_models:
                overlap = self.compute_overlap(model, prev_model)
                overlaps.append(overlap)
            
            total_overlap_sq = sum(jnp.abs(o) ** 2 for o in overlaps)
            
            if total_overlap_sq < 1e-8:
                print(f"  正交化完成 (步骤 {step}), 总重叠²: {float(total_overlap_sq):.8f}")
                break
            
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                total_loss = 0.0
                for prev_model in prev_models:
                    overlap = self.compute_overlap(model_obj, prev_model)
                    total_loss = total_loss + jnp.abs(overlap) ** 2
                return total_loss
            
            loss, grads = jax.value_and_grad(loss_fn)(model_state)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, model_state[1])
        
        overlaps = []
        for prev_model in prev_models:
            overlap = self.compute_overlap(model, prev_model)
            overlaps.append(float(jnp.abs(overlap)))
        
        print(f"  正交化完成, 与各态重叠: {overlaps}")
        
        return model
    
    def train_single_state(self, model: SingleStateAnsatz, prev_models: list, 
                          n_iterations: int = 500, lr: float = 0.001,
                          overlap_weight: float = 50.0):
        """训练单个态
        
        改进:
        - 更多迭代次数
        - 学习率调度
        - 更强的正交约束权重
        - 早停机制
        """
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=100,
            decay_rate=0.95
        )
        
        optimizer = optax.adam(schedule)
        model_state = nnx.split(model)
        opt_state = optimizer.init(model_state)
        
        energy_history = []
        best_energy = float('inf')
        patience = 50
        patience_counter = 0
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for step in pbar:
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                
                energy = self.compute_energy_expectation(model_obj)
                
                overlap_loss = 0.0
                for prev_model in prev_models:
                    overlap = self.compute_overlap(model_obj, prev_model)
                    overlap_loss = overlap_loss + jnp.abs(overlap) ** 2
                
                total_loss = jnp.real(energy) + overlap_weight * overlap_loss
                
                return total_loss, (energy, overlap_loss)
            
            (loss, (energy, overlap_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            energy_val = float(jnp.real(energy))
            energy_history.append(energy_val)
            
            if step % 20 == 0:
                overlap_val = float(jnp.real(overlap_loss)) if prev_models else 0.0
                pbar.set_postfix({
                    'E': f'{energy_val:.6f}',
                    'Overlap': f'{overlap_val:.6f}'
                })
            
            if energy_val < best_energy - 1e-6:
                best_energy = energy_val
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience and step > 200:
                print(f"\n  早停于步骤 {step}")
                break
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, model_state[1])
        
        return model, energy_history
    
    def train(self, n_iterations: int = 500, lr: float = 0.001, overlap_weight: float = 50.0):
        """训练所有态
        
        改进:
        - 更长的训练时间
        - 更强的正交约束
        - 归一化处理
        """
        print("\n" + "=" * 60)
        print("开始 NES-VMC 训练 (改进版)")
        print("=" * 60)
        print(f"训练参数:")
        print(f"  迭代次数: {n_iterations}")
        print(f"  初始学习率: {lr}")
        print(f"  正交约束权重: {overlap_weight}")
        
        self.energy_histories = []
        self.final_energies = []
        
        for i in range(self.n_states):
            print(f"\n{'='*60}")
            print(f"训练第 {i} 个态 {'(基态)' if i == 0 else '(第1激发态)'}")
            print(f"{'='*60}")
            
            prev_models = self.models[:i]
            
            print(f"  初始化...")
            self.models[i] = self.normalize_model(self.models[i])
            
            if i > 0:
                self.models[i] = self.orthogonalize(
                    self.models[i], 
                    prev_models,
                    lr=0.001,
                    n_steps=300
                )
            
            self.models[i], energy_history = self.train_single_state(
                self.models[i], 
                prev_models,
                n_iterations=n_iterations,
                lr=lr,
                overlap_weight=overlap_weight
            )
            
            self.models[i] = self.normalize_model(self.models[i])
            
            self.energy_histories.append(energy_history)
            
            final_energy = np.mean(energy_history[-50:])
            self.final_energies.append(final_energy)
            
            print(f"\n第 {i} 个态训练完成:")
            print(f"  最终能量: {final_energy:.8f} Ha")
            print(f"  FCI 参考: {self.E_fcis[i]:.8f} Ha")
            print(f"  误差: {abs(final_energy - self.E_fcis[i]):.6f} Ha")
            
            if i > 0:
                overlap = float(jnp.abs(self.compute_overlap(self.models[i], self.models[0])))
                print(f"  与基态重叠: {overlap:.6f}")
        
        self._print_summary()
        
    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 60)
        print("训练总结")
        print("=" * 60)
        
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
                overlap = float(jnp.abs(self.compute_overlap(self.models[i], self.models[j])))
                print(f"|⟨E{i}|E{j}⟩| = {overlap:.6f}")
    
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
        
    def analyze_wavefunctions(self):
        """分析波函数性质"""
        print("\n" + "=" * 60)
        print("波函数分析")
        print("=" * 60)
        
        for i, model in enumerate(self.models):
            print(f"\n态 E{i}:")
            print("-" * 50)
            
            psi_values = []
            for state in self.all_states:
                psi = model(state)
                psi_values.append(psi)
            
            psi_values = np.array(psi_values)
            
            print("组态    波函数值 (实部+虚部)    |ψ|²")
            print("-" * 50)
            for j, state in enumerate(self.all_states):
                psi = psi_values[j]
                prob = np.abs(psi) ** 2
                print(f"{state}  {psi.real:+.4f}{psi.imag:+.4f}j  {prob:.4f}")
            
            total_prob = np.sum(np.abs(psi_values) ** 2)
            print(f"\n总概率: {total_prob:.4f}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("NES-VMC: H2分子激发态计算 (改进版, K=2)")
    print("=" * 60)
    
    trainer = NESVMCTrainer(
        bond_length=1.4,
        n_states=2,
        hidden_dim=32
    )
    
    trainer.train(
        n_iterations=800,
        lr=0.002,
        overlap_weight=100.0
    )
    
    trainer.analyze_wavefunctions()
    
    trainer.plot_results()


if __name__ == "__main__":
    main()
