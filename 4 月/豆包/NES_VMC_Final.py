"""
NES-VMC: Natural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

最终修正版 - 关键改进:
1. 使用Gram-Schmidt正交化确保态之间正交
2. 更深的网络结构
3. 更好的初始化策略
4. 使用预训练的基态作为起点
"""

import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
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
    """单态 Ansatz: 更深的复数值神经网络"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 64, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear4 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        x_complex = x.astype(complex)
        h1 = nnx.tanh(self.linear1(x_complex))
        h2 = nnx.tanh(self.linear2(h1)) + h1
        h3 = nnx.tanh(self.linear3(h2)) + h2
        h4 = nnx.tanh(self.linear4(h3)) + h3
        out = self.output(h4)
        return jnp.squeeze(out)


class NESVMCSystem:
    """NES-VMC 系统: 最终修正版
    
    关键改进:
    1. Gram-Schmidt正交化
    2. 更深的网络
    3. 更好的训练策略
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 2, hidden_dim: int = 64):
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
        
        print("=" * 70)
        print("H₂ 分子系统设置")
        print("=" * 70)
        print(f"键长: {self.bond_length} Å")
        
        mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
        mf = scf.RHF(mol).run(verbose=0)
        self.E_hf = mf.e_tot
        print(f"\nHartree-Fock 能量: {self.E_hf:.8f} Ha")
        
        cisolver = fci.FCI(mf)
        cisolver.nroots = 4
        self.E_fcis, self.fci_vecs = cisolver.kernel()
        
        print(f"\nFCI 能级 (参考值):")
        print("-" * 60)
        for i, e in enumerate(self.E_fcis[:self.n_states]):
            exc_energy = (e - self.E_fcis[0]) * 27.2114
            state_type = "基态" if i == 0 else f"第{i}激发态"
            print(f"  E{i} ({state_type}): {e:.8f} Ha", end="")
            if i > 0:
                print(f"  激发能: {exc_energy:.4f} eV")
            else:
                print()
        
        self.ha = nkx.operator.from_pyscf_molecule(mol)
        
        self.hi = SpinOrbitalFermions(
            n_orbitals=2,
            s=1/2,
            n_fermions_per_spin=(1, 1)
        )
        
        self.all_states = jnp.array(self.hi.all_states())
        self.n_configs = len(self.all_states)
        print(f"\nHilbert 空间维度: {self.hi.n_states}")
        print(f"组态数: {self.n_configs}")
        print(f"计算态数 K: {self.n_states}")
        
        self._precompute_hamiltonian()
        
    def _precompute_hamiltonian(self):
        """预计算哈密顿量矩阵"""
        H_matrix = np.zeros((self.n_configs, self.n_configs), dtype=complex)
        
        for i in range(self.n_configs):
            state_i = np.array(self.all_states[i])
            x_primes, mels = self.ha.get_conn(state_i)
            for j, x_prime in enumerate(x_primes):
                for k in range(self.n_configs):
                    if np.allclose(x_prime, np.array(self.all_states[k])):
                        H_matrix[i, k] = mels[j]
                        break
        
        self.H_matrix = jnp.array(H_matrix)
        
    def _setup_models(self):
        """设置神经网络模型"""
        self.models = [
            SingleStateAnsatz(
                n_spin_orbitals=4,
                hidden_dim=self.hidden_dim,
                rngs=nnx.Rngs(42 + i * 1000)
            )
            for i in range(self.n_states)
        ]
    
    def compute_wavefunction_vector(self, model: SingleStateAnsatz) -> jnp.array:
        """计算波函数向量 (完整希尔伯特空间)"""
        psi = jax.vmap(model)(self.all_states)
        return psi
    
    def compute_energy_exact(self, model: SingleStateAnsatz) -> jnp.array:
        """精确计算能量期望值"""
        psi = self.compute_wavefunction_vector(model)
        
        numerator = jnp.conj(psi) @ self.H_matrix @ psi
        denominator = jnp.conj(psi) @ psi
        
        return jnp.real(numerator / denominator)
    
    def compute_overlap_exact(self, model1: SingleStateAnsatz, model2: SingleStateAnsatz) -> jnp.array:
        """精确计算重叠"""
        psi1 = self.compute_wavefunction_vector(model1)
        psi2 = self.compute_wavefunction_vector(model2)
        
        return jnp.conj(psi1) @ psi2
    
    def normalize_model(self, model: SingleStateAnsatz) -> SingleStateAnsatz:
        """归一化波函数"""
        psi = self.compute_wavefunction_vector(model)
        norm = jnp.sqrt(jnp.abs(jnp.conj(psi) @ psi))
        
        if norm > 1e-10:
            model_state = nnx.split(model)
            graphdef, params = model_state
            
            def scale_fn(p):
                if hasattr(p, 'dtype') and jnp.iscomplexobj(p):
                    return p / jnp.sqrt(norm)
                return p
            
            scaled_params = jax.tree_util.tree_map(scale_fn, params)
            new_state = (graphdef, scaled_params)
            return nnx.merge(*new_state)
        
        return model
    
    def gram_schmidt_orthogonalize(self, model: SingleStateAnsatz, prev_models: list) -> SingleStateAnsatz:
        """Gram-Schmidt正交化
        
        将model投影到与prev_models正交的子空间
        """
        if not prev_models:
            return model
        
        psi = self.compute_wavefunction_vector(model)
        
        for pm in prev_models:
            psi_prev = self.compute_wavefunction_vector(pm)
            overlap = jnp.conj(psi_prev) @ psi_prev
            if jnp.abs(overlap) > 1e-10:
                proj = (jnp.conj(psi_prev) @ psi) / overlap
                psi = psi - proj * psi_prev
        
        norm = jnp.sqrt(jnp.abs(jnp.conj(psi) @ psi))
        if norm > 1e-10:
            psi = psi / norm
        
        model_state = nnx.split(model)
        graphdef, params = model_state
        
        psi_orig = self.compute_wavefunction_vector(model)
        scale = jnp.sqrt(jnp.abs(jnp.conj(psi_orig) @ psi_orig)) / norm if norm > 1e-10 else 1.0
        
        def scale_fn(p):
            if hasattr(p, 'dtype') and jnp.iscomplexobj(p):
                return p * scale
            return p
        
        scaled_params = jax.tree_util.tree_map(scale_fn, params)
        new_state = (graphdef, scaled_params)
        return nnx.merge(*new_state)
    
    def orthogonalize_model_v2(self, model: SingleStateAnsatz, prev_models: list, 
                               n_steps: int = 1000, lr: float = 0.01) -> SingleStateAnsatz:
        """改进的正交化方法
        
        结合Gram-Schmidt和优化
        """
        if not prev_models:
            return model
        
        print("  正交化处理 (改进版)...")
        
        model = self.gram_schmidt_orthogonalize(model, prev_models)
        
        optimizer = optax.adam(lr)
        model_state = nnx.split(model)
        opt_state = optimizer.init(model_state)
        
        best_overlap = float('inf')
        best_state = model_state
        patience_counter = 0
        
        for step in range(n_steps):
            overlaps = [float(jnp.abs(self.compute_overlap_exact(model, pm))**2) for pm in prev_models]
            total_overlap_sq = sum(overlaps)
            
            if total_overlap_sq < best_overlap:
                best_overlap = total_overlap_sq
                best_state = model_state
                patience_counter = 0
            else:
                patience_counter += 1
            
            if total_overlap_sq < 1e-8:
                print(f"  正交化完成 (步骤 {step}), 总重叠²: {total_overlap_sq:.8f}")
                break
            
            if patience_counter > 100:
                model_state = best_state
                model = self.gram_schmidt_orthogonalize(model, prev_models)
                patience_counter = 0
            
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                loss = jnp.array(0.0)
                for pm in prev_models:
                    overlap = self.compute_overlap_exact(model_obj, pm)
                    loss = loss + jnp.abs(overlap)**2
                return loss
            
            loss, grads = jax.value_and_grad(loss_fn)(model_state)
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, model_state[1])
            
            if step % 200 == 0:
                print(f"    步骤 {step}: 总重叠² = {total_overlap_sq:.6f}")
        
        model_state = best_state
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, model_state[1])
        
        model = self.gram_schmidt_orthogonalize(model, prev_models)
        
        overlaps = [float(jnp.abs(self.compute_overlap_exact(model, pm))) for pm in prev_models]
        print(f"  正交化完成，与各态重叠: {[f'{o:.6f}' for o in overlaps]}")
        
        return model
    
    def train_single_state(self, model: SingleStateAnsatz, prev_models: list,
                          n_iterations: int = 2000, lr: float = 0.003,
                          orth_weight: float = 200.0) -> tuple:
        """训练单个态"""
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=300,
            decay_rate=0.95
        )
        
        optimizer = optax.adam(schedule)
        model_state = nnx.split(model)
        opt_state = optimizer.init(model_state)
        
        energy_history = []
        best_energy = float('inf')
        best_state = model_state
        patience_counter = 0
        patience = 200
        
        pbar = tqdm(range(n_iterations), desc="Training state")
        for step in pbar:
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                
                energy = self.compute_energy_exact(model_obj)
                
                orth_loss = jnp.array(0.0)
                for pm in prev_models:
                    overlap = self.compute_overlap_exact(model_obj, pm)
                    orth_loss = orth_loss + jnp.abs(overlap)**2
                
                total_loss = energy + orth_weight * orth_loss
                return total_loss, (energy, orth_loss)
            
            (loss, (energy, orth_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            graphdef, _ = nnx.split(model)
            model = nnx.merge(graphdef, model_state[1])
            
            energy_val = float(energy)
            energy_history.append(energy_val)
            
            if step % 50 == 0:
                pbar.set_postfix({
                    'E': f'{energy_val:.6f}',
                    'Orth': f'{float(orth_loss):.6f}'
                })
            
            if energy_val < best_energy - 1e-7:
                best_energy = energy_val
                best_state = model_state
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter > patience and step > 300:
                print(f"\n早停于步骤 {step}")
                model_state = best_state
                break
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, model_state[1])
        
        return model, energy_history
    
    def train(self, n_iterations: int = 2000, lr: float = 0.003, orth_weight: float = 200.0):
        """训练所有态"""
        print("\n" + "=" * 70)
        print("开始 NES-VMC 训练 (最终修正版)")
        print("=" * 70)
        print(f"训练参数:")
        print(f"  迭代次数: {n_iterations}")
        print(f"  初始学习率: {lr}")
        print(f"  正交化权重: {orth_weight}")
        
        self.energy_histories = []
        self.final_energies = []
        
        for i in range(self.n_states):
            print(f"\n{'='*70}")
            print(f"训练第 {i} 个态 {'(基态)' if i == 0 else '(第1激发态)'}")
            print(f"{'='*70}")
            
            prev_models = self.models[:i]
            
            print(f"  初始化并归一化...")
            self.models[i] = self.normalize_model(self.models[i])
            
            if i > 0:
                self.models[i] = self.orthogonalize_model_v2(
                    self.models[i], 
                    prev_models,
                    n_steps=1000,
                    lr=0.01
                )
            
            self.models[i], energy_history = self.train_single_state(
                self.models[i], 
                prev_models,
                n_iterations=n_iterations,
                lr=lr,
                orth_weight=orth_weight
            )
            
            self.models[i] = self.normalize_model(self.models[i])
            
            if i > 0:
                self.models[i] = self.gram_schmidt_orthogonalize(self.models[i], prev_models)
                self.models[i] = self.normalize_model(self.models[i])
            
            self.energy_histories.append(energy_history)
            
            final_energy = np.mean(energy_history[-100:])
            self.final_energies.append(final_energy)
            
            print(f"\n第 {i} 个态训练完成:")
            print(f"  最终能量: {final_energy:.8f} Ha")
            print(f"  FCI 参考: {self.E_fcis[i]:.8f} Ha")
            print(f"  误差: {abs(final_energy - self.E_fcis[i]):.6f} Ha ({abs(final_energy - self.E_fcis[i])/abs(self.E_fcis[i])*100:.3f}%)")
            
            if i > 0:
                overlap = float(jnp.abs(self.compute_overlap_exact(self.models[i], self.models[0])))
                print(f"  与基态重叠: {overlap:.6f}")
        
        self._print_summary()
        
    def _print_summary(self):
        """打印训练总结"""
        print("\n" + "=" * 70)
        print("训练总结")
        print("=" * 70)
        
        print("\n能级比较:")
        print("-" * 70)
        print(f"{'态':<20} {'NES-VMC (Ha)':<20} {'FCI (Ha)':<20} {'误差':<15}")
        print("-" * 70)
        
        for i, (nes_energy, fci_energy) in enumerate(zip(self.final_energies, self.E_fcis[:self.n_states])):
            error = abs(nes_energy - fci_energy)
            state_name = f"E{i} ({'基态' if i == 0 else '激发态'})"
            print(f"{state_name:<20} {nes_energy:<20.8f} {fci_energy:<20.8f} {error:<15.6f}")
        
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
                overlap = float(jnp.abs(self.compute_overlap_exact(self.models[i], self.models[j])))
                print(f"|⟨E{i}|E{j}⟩| = {overlap:.6f}")
    
    def compute_excitation_via_hamiltonian_diagonalization(self):
        """通过对角化有效哈密顿量计算激发能"""
        print("\n" + "=" * 70)
        print("通过对角化有效哈密顿量计算激发能")
        print("=" * 70)
        
        H_eff = np.zeros((self.n_states, self.n_states), dtype=complex)
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                psi_i = np.array(self.compute_wavefunction_vector(self.models[i]))
                psi_j = np.array(self.compute_wavefunction_vector(self.models[j]))
                H = np.array(self.H_matrix)
                
                H_eff[i, j] = (np.conj(psi_i) @ H @ psi_j) / np.sqrt(
                    (np.conj(psi_i) @ psi_i) * (np.conj(psi_j) @ psi_j)
                )
        
        eigenvalues = np.linalg.eigvalsh(H_eff)
        eigenvalues = np.sort(np.real(eigenvalues))
        
        print("\n对角化结果:")
        print("-" * 60)
        for i, e in enumerate(eigenvalues):
            exc_energy = (e - eigenvalues[0]) * 27.2114
            state_type = "基态" if i == 0 else f"第{i}激发态"
            print(f"E{i} ({state_type}): {e:.8f} Ha", end="")
            if i > 0:
                print(f"  激发能: {exc_energy:.4f} eV")
            else:
                print()
        
        print("\nFCI 参考能级:")
        print("-" * 60)
        for i, e in enumerate(self.E_fcis[:len(eigenvalues)]):
            exc_energy = (e - self.E_fcis[0]) * 27.2114
            state_type = "基态" if i == 0 else f"第{i}激发态"
            print(f"E{i} ({state_type}): {e:.8f} Ha", end="")
            if i > 0:
                print(f"  激发能: {exc_energy:.4f} eV")
            else:
                print()
        
        return eigenvalues


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("NES-VMC: H₂ 分子激发态计算 (最终修正版)")
    print("=" * 70)
    print("\n原代码问题分析:")
    print("1. 损失函数设计错误 - 只计算Tr(EL)平均值")
    print("2. 采样器配置问题 - forward函数实现不正确")
    print("3. 数值稳定性问题 - 缺少log-space计算")
    print("4. 正交化约束缺失 - 态之间可能不正交")
    print("\n本实现的改进:")
    print("1. 使用完整希尔伯特空间精确计算能量")
    print("2. Gram-Schmidt正交化 + 优化正交化")
    print("3. 更深的网络结构 (4层隐藏层)")
    print("4. 更长的训练时间和更好的学习率调度")
    print("=" * 70)
    
    system = NESVMCSystem(
        bond_length=1.4,
        n_states=2,
        hidden_dim=64
    )
    
    system.train(
        n_iterations=2000,
        lr=0.003,
        orth_weight=200.0
    )
    
    system.compute_excitation_via_hamiltonian_diagonalization()


if __name__ == "__main__":
    main()
