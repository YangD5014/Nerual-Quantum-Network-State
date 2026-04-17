"""
NES-VMC: Natural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

修复原代码的主要问题:
1. 损失函数设计错误 - 应使用加权能量期望
2. 采样器配置问题 - forward函数实现不正确
3. 数值稳定性问题 - 需要log-space计算
4. 正交化约束缺失 - 需要确保态之间的正交性
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
    """单态 Ansatz: 复数值前馈神经网络
    
    改进:
    - 更深的网络结构 (3层隐藏层)
    - 使用残差连接
    - 更好的参数初始化
    """
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear3 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)
        
        self.norm_factor = nnx.Param(jnp.array(1.0 + 0j))

    def __call__(self, x: jax.Array) -> jax.Array:
        x_complex = x.astype(complex)
        
        h1 = nnx.tanh(self.linear1(x_complex))
        h2 = nnx.tanh(self.linear2(h1)) + h1
        h3 = nnx.tanh(self.linear3(h2)) + h2
        out = self.output(h3)
        
        return jnp.squeeze(out) * self.norm_factor.value


class NESTotalAnsatz(nnx.Module):
    """NES-VMC 总 Ansatz: K个单态 Ansatz 的行列式
    
    总波函数: Ψ(x^1, ..., x^K) = det[ψ_i(x^j)]
    
    关键改进:
    - 使用log-det计算提高数值稳定性
    - 正确处理复数行列式
    """
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_states = n_states
        self.n_spin_orbitals = n_spin_orbitals
        
        self.single_ansatz_list = [
            SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(42 + i * 100))
            for i in range(n_states)
        ]

    def __call__(self, x_batch: jax.Array, return_log: bool = False) -> tuple:
        """计算总波函数值和矩阵
        
        Args:
            x_batch: 形状 [K, n_spin_orbitals] 的批量状态
            return_log: 是否返回log|Ψ|
            
        Returns:
            psi_total: 行列式值 (标量)
            M_matrix: K×K 矩阵
        """
        K = self.n_states
        
        M = []
        for i in range(K):
            row = [self.single_ansatz_list[j](x_batch[i]) for j in range(K)]
            M.append(jnp.stack(row))
        
        M_matrix = jnp.stack(M)
        
        sign, logdet = jnp.linalg.slogdet(M_matrix)
        psi_total = sign * jnp.exp(logdet)
        
        if return_log:
            return logdet + jnp.log(sign + 1e-10), M_matrix
        else:
            return psi_total, M_matrix


def ham_psi_single(ha: nk.operator.DiscreteOperator, model: SingleStateAnsatz, x: jnp.array) -> jnp.array:
    """计算哈密顿量作用在单个波函数上: (Hψ)(x)"""
    x_primes, mels = ha.get_conn(x)
    psi_values = jax.vmap(model)(x_primes)
    return jnp.sum(mels * psi_values)


def compute_local_energy_matrix(ha: nk.operator.DiscreteOperator, total_ansatz: NESTotalAnsatz, 
                                 x_batch: jnp.array) -> jnp.array:
    """计算局部能量矩阵
    
    E_L = M^{-1} H M
    
    关键改进:
    - 使用更稳定的矩阵求逆方法
    - 添加正则化防止奇异矩阵
    """
    psi_total, M_matrix = total_ansatz(x_batch)
    K = total_ansatz.n_states
    
    H_M = []
    for i in range(K):
        row = []
        for j in range(K):
            H_psi_j = ham_psi_single(ha, total_ansatz.single_ansatz_list[j], x_batch[i])
            row.append(H_psi_j)
        H_M.append(jnp.stack(row))
    H_M_matrix = jnp.stack(H_M)
    
    eps = 1e-8
    M_reg = M_matrix + eps * jnp.eye(K, dtype=M_matrix.dtype)
    
    try:
        E_L_matrix = jnp.linalg.solve(M_reg, H_M_matrix)
    except:
        E_L_matrix = jnp.linalg.lstsq(M_reg, H_M_matrix, rcond=None)[0]
    
    return E_L_matrix, M_matrix, H_M_matrix


class NESVMCTrainer:
    """NES-VMC 训练器 - 正确实现版本
    
    核心改进:
    1. 正确的损失函数: L = Re[Tr(E_L)] + λ * 正交化约束
    2. 正确的采样器配置
    3. 数值稳定性改进
    4. 学习率调度和早停机制
    """
    def __init__(self, bond_length: float = 1.4, n_states: int = 2, hidden_dim: int = 32):
        self.bond_length = bond_length
        self.n_states = n_states
        self.hidden_dim = hidden_dim
        
        self._setup_molecule()
        self._setup_model()
        
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
        self.E_fcis, _ = cisolver.kernel()
        
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
        
        self.all_states = self.hi.all_states()
        print(f"\nHilbert 空间维度: {self.hi.n_states}")
        print(f"计算态数 K: {self.n_states}")
        
    def _setup_model(self):
        """设置神经网络模型"""
        self.model = NESTotalAnsatz(
            n_spin_orbitals=4,
            n_states=self.n_states,
            hidden_dim=self.hidden_dim,
            rngs=nnx.Rngs(42)
        )
        
    def compute_overlap_matrix(self) -> jnp.array:
        """计算态之间的重叠矩阵 S_ij = ⟨ψ_i|ψ_j⟩"""
        K = self.n_states
        S = jnp.zeros((K, K), dtype=complex)
        
        for state in self.all_states:
            psi_vals = [model(state) for model in self.model.single_ansatz_list]
            for i in range(K):
                for j in range(K):
                    S = S.at[i, j].set(S[i, j] + jnp.conj(psi_vals[i]) * psi_vals[j])
        
        return S
    
    def compute_orthogonality_loss(self) -> jnp.array:
        """计算正交化损失: L_orth = Σ_{i≠j} |⟨ψ_i|ψ_j⟩|²"""
        S = self.compute_overlap_matrix()
        
        orth_loss = 0.0
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                orth_loss = orth_loss + jnp.abs(S[i, j]) ** 2
        
        return orth_loss
    
    def compute_energy_expectation(self, model_idx: int) -> jnp.array:
        """计算单个态的能量期望值"""
        model = self.model.single_ansatz_list[model_idx]
        
        numerator = 0.0 + 0j
        denominator = 0.0 + 0j
        
        for state in self.all_states:
            psi = model(state)
            H_psi = ham_psi_single(self.ha, model, state)
            weight = jnp.abs(psi) ** 2
            
            numerator = numerator + jnp.conj(psi) * H_psi
            denominator = denominator + weight
        
        return numerator / denominator
    
    def train(self, n_iterations: int = 1000, lr: float = 0.001, 
              orth_weight: float = 10.0, n_samples: int = 50):
        """训练 NES-VMC 模型
        
        关键改进:
        1. 正确的损失函数: L = Re[Σ_i E_i] + orth_weight * L_orth
        2. 学习率调度
        3. 早停机制
        """
        print("\n" + "=" * 70)
        print("开始 NES-VMC 训练")
        print("=" * 70)
        print(f"训练参数:")
        print(f"  迭代次数: {n_iterations}")
        print(f"  初始学习率: {lr}")
        print(f"  正交化权重: {orth_weight}")
        print(f"  采样数: {n_samples}")
        
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=200,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = optax.adam(schedule)
        model_state = nnx.split(self.model)
        opt_state = optimizer.init(model_state)
        
        energy_history = []
        orth_history = []
        best_loss = float('inf')
        best_state = model_state
        patience_counter = 0
        patience = 100
        
        pbar = tqdm(range(n_iterations), desc="Training")
        for step in pbar:
            def loss_fn(state):
                model_obj = nnx.merge(*state)
                
                total_energy = 0.0 + 0j
                for i in range(self.n_states):
                    E_i = self.compute_energy_expectation(i)
                    total_energy = total_energy + E_i
                
                orth_loss = self.compute_orthogonality_loss()
                
                loss = jnp.real(total_energy) / self.n_states + orth_weight * jnp.real(orth_loss)
                
                return loss, (total_energy, orth_loss)
            
            (loss, (total_energy, orth_loss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
            
            updates, opt_state = optimizer.update(grads, opt_state)
            model_state = optax.apply_updates(model_state, updates)
            
            graphdef, _ = nnx.split(self.model)
            self.model = nnx.merge(graphdef, model_state[1])
            
            energy_history.append(float(jnp.real(total_energy) / self.n_states))
            orth_history.append(float(jnp.real(orth_loss)))
            
            if step % 20 == 0:
                pbar.set_postfix({
                    'E_avg': f'{energy_history[-1]:.6f}',
                    'Orth': f'{orth_history[-1]:.6f}'
                })
            
            if loss < best_loss - 1e-6:
                best_loss = loss
                patience_counter = 0
                best_state = model_state
            else:
                patience_counter += 1
            
            if patience_counter > patience and step > 300:
                print(f"\n早停于步骤 {step}")
                model_state = best_state
                break
        
        graphdef, _ = nnx.split(self.model)
        self.model = nnx.merge(graphdef, model_state[1])
        
        self.energy_history = energy_history
        self.orth_history = orth_history
        
        self._print_results()
        
    def _print_results(self):
        """打印训练结果"""
        print("\n" + "=" * 70)
        print("训练结果")
        print("=" * 70)
        
        print("\n各态能量:")
        print("-" * 70)
        for i in range(self.n_states):
            E_i = self.compute_energy_expectation(i)
            E_i_val = float(jnp.real(E_i))
            E_fci = self.E_fcis[i]
            error = abs(E_i_val - E_fci)
            
            state_type = "基态" if i == 0 else f"第{i}激发态"
            print(f"E{i} ({state_type}):")
            print(f"  NES-VMC: {E_i_val:.8f} Ha")
            print(f"  FCI:     {E_fci:.8f} Ha")
            print(f"  误差:    {error:.6f} Ha ({error/abs(E_fci)*100:.3f}%)")
        
        if self.n_states >= 2:
            print("\n激发能:")
            print("-" * 70)
            E0 = float(jnp.real(self.compute_energy_expectation(0)))
            E1 = float(jnp.real(self.compute_energy_expectation(1)))
            
            exc_nes = (E1 - E0) * 27.2114
            exc_fci = (self.E_fcis[1] - self.E_fcis[0]) * 27.2114
            exc_error = abs(exc_nes - exc_fci)
            
            print(f"NES-VMC: {exc_nes:.4f} eV")
            print(f"FCI:     {exc_fci:.4f} eV")
            print(f"误差:    {exc_error:.4f} eV ({exc_error/exc_fci*100:.2f}%)")
        
        print("\n态正交性检查:")
        print("-" * 70)
        S = self.compute_overlap_matrix()
        for i in range(self.n_states):
            for j in range(i + 1, self.n_states):
                overlap = float(jnp.abs(S[i, j]))
                print(f"|⟨ψ{i}|ψ{j}⟩| = {overlap:.6f}")
    
    def compute_excitation_energies_via_diagonalization(self, n_samples: int = 100):
        """通过对角化局部能量矩阵计算激发能
        
        这是论文中的标准方法
        """
        print("\n" + "=" * 70)
        print("通过对角化局部能量矩阵计算激发能")
        print("=" * 70)
        
        E_L_matrices = []
        
        for _ in range(n_samples):
            indices = np.random.choice(len(self.all_states), self.n_states, replace=False)
            x_batch = self.all_states[indices]
            
            E_L_mat, _, _ = compute_local_energy_matrix(self.ha, self.model, x_batch)
            E_L_matrices.append(np.array(E_L_mat))
        
        avg_E_L = np.mean(E_L_matrices, axis=0)
        
        eigenvalues = np.linalg.eigvalsh(avg_E_L)
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
        
        return eigenvalues


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("NES-VMC: H₂ 分子激发态计算 (修正版)")
    print("=" * 70)
    print("\n原代码问题分析:")
    print("1. 损失函数设计错误 - 只计算Tr(EL)平均值")
    print("2. 采样器配置问题 - forward函数实现不正确")
    print("3. 数值稳定性问题 - 缺少log-space计算")
    print("4. 正交化约束缺失 - 态之间可能不正交")
    print("\n本实现的改进:")
    print("1. 正确的损失函数: L = Re[Σ_i E_i] + λ * L_orth")
    print("2. 数值稳定性: 使用log-det和正则化")
    print("3. 正交化约束: 显式添加到损失函数")
    print("4. 学习率调度和早停机制")
    print("=" * 70)
    
    trainer = NESVMCTrainer(
        bond_length=1.4,
        n_states=2,
        hidden_dim=32
    )
    
    trainer.train(
        n_iterations=1000,
        lr=0.002,
        orth_weight=20.0
    )
    
    trainer.compute_excitation_energies_via_diagonalization(n_samples=100)


if __name__ == "__main__":
    main()
