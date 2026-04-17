"""
NES-VMC: Natural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

简化版 - 使用NetKet标准框架
对于小系统，使用完整希尔伯特空间计算更稳定
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


class SimpleAnsatz(nnx.Module):
    """简单的复数值神经网络"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 32, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        return jnp.squeeze(self.output(h))


def compute_hamiltonian_matrix(ha, all_states):
    """计算完整哈密顿量矩阵"""
    n_configs = len(all_states)
    H = np.zeros((n_configs, n_configs), dtype=complex)
    
    for i, state_i in enumerate(all_states):
        x_primes, mels = ha.get_conn(np.array(state_i))
        for j, x_prime in enumerate(x_primes):
            for k, state_k in enumerate(all_states):
                if np.allclose(x_prime, np.array(state_k)):
                    H[i, k] = mels[j]
                    break
    
    return H


def compute_energy_and_gradient(H, psi, model_state):
    """计算能量和梯度"""
    psi_conj = jnp.conj(psi)
    norm = psi_conj @ psi
    
    H_psi = H @ psi
    energy = jnp.real((psi_conj @ H_psi) / norm)
    
    return energy


def train_state_direct(H, all_states, n_spin_orbitals, hidden_dim, seed, 
                       prev_psi_list=None, n_iterations=2000, lr=0.01, orth_weight=100.0):
    """直接优化单个态
    
    使用完整希尔伯特空间，避免采样噪声
    """
    model = SimpleAnsatz(n_spin_orbitals, hidden_dim, rngs=nnx.Rngs(seed))
    
    def get_psi(model):
        return jax.vmap(model)(all_states)
    
    def normalize(psi):
        norm = jnp.sqrt(jnp.abs(jnp.conj(psi) @ psi))
        return psi / norm if norm > 1e-10 else psi
    
    def project_orthogonal(psi, prev_psi_list):
        """投影到与之前态正交的子空间"""
        for prev_psi in prev_psi_list:
            overlap = jnp.conj(prev_psi) @ psi
            psi = psi - overlap * prev_psi
        return normalize(psi)
    
    optimizer = optax.adam(lr)
    model_state = nnx.split(model)
    opt_state = optimizer.init(model_state)
    
    energy_history = []
    best_energy = float('inf')
    best_state = model_state
    patience = 0
    max_patience = 300
    
    pbar = tqdm(range(n_iterations), desc="Training")
    for step in pbar:
        psi = get_psi(model)
        psi = normalize(psi)
        
        if prev_psi_list:
            psi_proj = project_orthogonal(psi, prev_psi_list)
            if jnp.abs(jnp.conj(psi_proj) @ psi_proj) > 1e-6:
                psi = psi_proj
        
        psi_conj = jnp.conj(psi)
        norm = psi_conj @ psi
        H_psi = H @ psi
        energy = jnp.real((psi_conj @ H_psi) / norm)
        
        orth_loss = jnp.array(0.0)
        if prev_psi_list:
            for prev_psi in prev_psi_list:
                overlap = jnp.conj(prev_psi) @ psi
                orth_loss = orth_loss + jnp.abs(overlap)**2
        
        def loss_fn(state):
            model_obj = nnx.merge(*state)
            psi_new = jax.vmap(model_obj)(all_states)
            psi_new = normalize(psi_new)
            
            if prev_psi_list:
                psi_new = project_orthogonal(psi_new, prev_psi_list)
            
            psi_new_conj = jnp.conj(psi_new)
            norm_new = psi_new_conj @ psi_new
            H_psi_new = H @ psi_new
            energy_new = jnp.real((psi_new_conj @ H_psi_new) / norm_new)
            
            orth_loss_new = jnp.array(0.0)
            if prev_psi_list:
                for prev_psi in prev_psi_list:
                    overlap = jnp.conj(prev_psi) @ psi_new
                    orth_loss_new = orth_loss_new + jnp.abs(overlap)**2
            
            return energy_new + orth_weight * orth_loss_new, (energy_new, orth_loss_new)
        
        (loss, (e, o)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model_state)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        model_state = optax.apply_updates(model_state, updates)
        
        graphdef, _ = nnx.split(model)
        model = nnx.merge(graphdef, model_state[1])
        
        energy_val = float(energy)
        energy_history.append(energy_val)
        
        if step % 50 == 0:
            pbar.set_postfix({'E': f'{energy_val:.6f}', 'Orth': f'{float(orth_loss):.6f}'})
        
        if energy_val < best_energy - 1e-7:
            best_energy = energy_val
            best_state = model_state
            patience = 0
        else:
            patience += 1
        
        if patience > max_patience and step > 500:
            print(f"\n早停于步骤 {step}")
            model_state = best_state
            break
    
    graphdef, _ = nnx.split(model)
    model = nnx.merge(graphdef, model_state[1])
    
    final_psi = get_psi(model)
    final_psi = normalize(final_psi)
    
    if prev_psi_list:
        final_psi = project_orthogonal(final_psi, prev_psi_list)
    
    return model, final_psi, energy_history


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("NES-VMC: H₂ 分子激发态计算 (简化版)")
    print("=" * 70)
    
    bond_length = 1.4
    geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
    
    mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
    mf = scf.RHF(mol).run(verbose=0)
    
    print(f"键长: {bond_length} Å")
    print(f"Hartree-Fock 能量: {mf.e_tot:.8f} Ha")
    
    cisolver = fci.FCI(mf)
    cisolver.nroots = 4
    E_fcis, fci_vecs = cisolver.kernel()
    
    print(f"\nFCI 能级 (参考值):")
    print("-" * 60)
    for i, e in enumerate(E_fcis[:2]):
        exc = (e - E_fcis[0]) * 27.2114
        print(f"E{i} = {e:.8f} Ha", end="")
        if i > 0:
            print(f"  激发能: {exc:.4f} eV")
        else:
            print()
    
    ha = nkx.operator.from_pyscf_molecule(mol)
    hi = SpinOrbitalFermions(n_orbitals=2, s=1/2, n_fermions_per_spin=(1, 1))
    
    all_states = jnp.array(hi.all_states())
    n_configs = len(all_states)
    print(f"\n组态数: {n_configs}")
    
    print("\n计算哈密顿量矩阵...")
    H = jnp.array(compute_hamiltonian_matrix(ha, all_states))
    
    print("\n" + "=" * 70)
    print("训练基态")
    print("=" * 70)
    
    model_gs, psi_gs, energy_history_gs = train_state_direct(
        H, all_states, 4, 32, seed=42,
        prev_psi_list=None,
        n_iterations=3000,
        lr=0.01,
        orth_weight=0.0
    )
    
    E_gs = float(jnp.real((jnp.conj(psi_gs) @ H @ psi_gs) / (jnp.conj(psi_gs) @ psi_gs)))
    print(f"\n基态能量: {E_gs:.8f} Ha")
    print(f"FCI 参考: {E_fcis[0]:.8f} Ha")
    print(f"误差: {abs(E_gs - E_fcis[0]):.6f} Ha")
    
    print("\n" + "=" * 70)
    print("训练第一激发态")
    print("=" * 70)
    
    model_ex, psi_ex, energy_history_ex = train_state_direct(
        H, all_states, 4, 32, seed=142,
        prev_psi_list=[psi_gs],
        n_iterations=3000,
        lr=0.01,
        orth_weight=200.0
    )
    
    E_ex = float(jnp.real((jnp.conj(psi_ex) @ H @ psi_ex) / (jnp.conj(psi_ex) @ psi_ex)))
    overlap = float(jnp.abs(jnp.conj(psi_gs) @ psi_ex))
    
    print(f"\n激发态能量: {E_ex:.8f} Ha")
    print(f"FCI 参考: {E_fcis[1]:.8f} Ha")
    print(f"误差: {abs(E_ex - E_fcis[1]):.6f} Ha")
    print(f"与基态重叠: {overlap:.6f}")
    
    print("\n" + "=" * 70)
    print("最终结果")
    print("=" * 70)
    
    exc_nes = (E_ex - E_gs) * 27.2114
    exc_fci = (E_fcis[1] - E_fcis[0]) * 27.2114
    
    print(f"\n能级比较:")
    print(f"  E0: {E_gs:.8f} Ha (NES-VMC) vs {E_fcis[0]:.8f} Ha (FCI)")
    print(f"  E1: {E_ex:.8f} Ha (NES-VMC) vs {E_fcis[1]:.8f} Ha (FCI)")
    print(f"\n激发能:")
    print(f"  {exc_nes:.4f} eV (NES-VMC) vs {exc_fci:.4f} eV (FCI)")
    print(f"  误差: {abs(exc_nes - exc_fci):.4f} eV ({abs(exc_nes - exc_fci)/exc_fci*100:.2f}%)")


if __name__ == "__main__":
    main()
