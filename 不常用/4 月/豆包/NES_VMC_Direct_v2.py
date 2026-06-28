"""
NES-VMC: Natural Excited States Variational Monte Carlo
基于论文: Pfau et al. 2024 - Accurate Computation of Quantum Excited States with Neural Networks

直接变分法 - 对于小系统最有效
"""

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto, scf, fci
import netket as nk
import netket.experimental as nkx
import optax
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from netket.hilbert import SpinOrbitalFermions
except ImportError:
    from netket.experimental.hilbert import SpinOrbitalFermions


def compute_hamiltonian_matrix(ha, all_states):
    """计算完整哈密顿量矩阵"""
    n = len(all_states)
    H = np.zeros((n, n), dtype=complex)
    
    for i, state_i in enumerate(all_states):
        x_primes, mels = ha.get_conn(np.array(state_i))
        for j, x_prime in enumerate(x_primes):
            for k, state_k in enumerate(all_states):
                if np.allclose(x_prime, np.array(state_k)):
                    H[i, k] = mels[j]
                    break
    
    return H


def normalize_coeffs(coeffs):
    """归一化系数"""
    norm = jnp.sqrt(jnp.abs(jnp.conj(coeffs) @ coeffs))
    return coeffs / norm if norm > 1e-10 else coeffs


def project_orthogonal(coeffs, prev_coeffs_list):
    """投影到与之前态正交的子空间"""
    for prev_coeffs in prev_coeffs_list:
        overlap = jnp.conj(prev_coeffs) @ coeffs
        coeffs = coeffs - overlap * prev_coeffs
    return normalize_coeffs(coeffs)


def train_variational_state(H, n_configs, prev_states=None, n_iterations=5000, lr=0.05, orth_weight=100.0, seed=42):
    """训练变分态"""
    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)
    
    coeffs_real = jax.random.normal(key1, (n_configs,)) * 0.1
    coeffs_imag = jax.random.normal(key2, (n_configs,)) * 0.1
    coeffs = coeffs_real + 1j * coeffs_imag
    coeffs = normalize_coeffs(coeffs)
    
    if prev_states:
        coeffs = project_orthogonal(coeffs, prev_states)
    
    optimizer = optax.adam(lr)
    params = {'real': jnp.real(coeffs), 'imag': jnp.imag(coeffs)}
    opt_state = optimizer.init(params)
    
    energy_history = []
    best_energy = float('inf')
    best_params = params
    patience = 0
    
    pbar = tqdm(range(n_iterations), desc="Training")
    for step in pbar:
        coeffs = params['real'] + 1j * params['imag']
        coeffs = normalize_coeffs(coeffs)
        
        if prev_states:
            coeffs = project_orthogonal(coeffs, prev_states)
        
        coeffs_conj = jnp.conj(coeffs)
        norm = coeffs_conj @ coeffs
        
        H_coeffs = H @ coeffs
        energy = jnp.real((coeffs_conj @ H_coeffs) / norm)
        
        orth_loss = jnp.array(0.0)
        if prev_states:
            for prev_coeffs in prev_states:
                overlap = coeffs_conj @ prev_coeffs
                orth_loss = orth_loss + jnp.abs(overlap)**2
        
        def loss_fn(p):
            c = p['real'] + 1j * p['imag']
            c = normalize_coeffs(c)
            
            if prev_states:
                c = project_orthogonal(c, prev_states)
            
            c_conj = jnp.conj(c)
            n = c_conj @ c
            H_c = H @ c
            e = jnp.real((c_conj @ H_c) / n)
            
            o_loss = jnp.array(0.0)
            if prev_states:
                for prev_c in prev_states:
                    ov = c_conj @ prev_c
                    o_loss = o_loss + jnp.abs(ov)**2
            
            return e + orth_weight * o_loss, (e, o_loss)
        
        (loss, (e, o)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        c = params['real'] + 1j * params['imag']
        c = normalize_coeffs(c)
        params = {'real': jnp.real(c), 'imag': jnp.imag(c)}
        
        energy_val = float(energy)
        energy_history.append(energy_val)
        
        if step % 100 == 0:
            pbar.set_postfix({'E': f'{energy_val:.6f}', 'Orth': f'{float(orth_loss):.6f}'})
        
        if energy_val < best_energy - 1e-8:
            best_energy = energy_val
            best_params = {k: v for k, v in params.items()}
            patience = 0
        else:
            patience += 1
        
        if patience > 500 and step > 1000:
            print(f"\n早停于步骤 {step}")
            params = best_params
            break
    
    final_coeffs = params['real'] + 1j * params['imag']
    final_coeffs = normalize_coeffs(final_coeffs)
    
    if prev_states:
        final_coeffs = project_orthogonal(final_coeffs, prev_states)
    
    return final_coeffs, energy_history


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("NES-VMC: H₂ 分子激发态计算 (直接变分法)")
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
    
    all_states = hi.all_states()
    n_configs = len(all_states)
    print(f"\n组态数: {n_configs}")
    
    print("\n计算哈密顿量矩阵...")
    H = jnp.array(compute_hamiltonian_matrix(ha, all_states))
    
    print("\n" + "=" * 70)
    print("训练基态")
    print("=" * 70)
    
    psi_gs, history_gs = train_variational_state(
        H, n_configs, prev_states=None, n_iterations=5000, lr=0.05, seed=42
    )
    
    E_gs = float(jnp.real((jnp.conj(psi_gs) @ H @ psi_gs) / (jnp.conj(psi_gs) @ psi_gs)))
    
    print(f"\n基态能量: {E_gs:.8f} Ha")
    print(f"FCI 参考: {E_fcis[0]:.8f} Ha")
    print(f"误差: {abs(E_gs - E_fcis[0]):.6f} Ha ({abs(E_gs - E_fcis[0])/abs(E_fcis[0])*100:.3f}%)")
    
    print("\n" + "=" * 70)
    print("训练第一激发态")
    print("=" * 70)
    
    psi_ex, history_ex = train_variational_state(
        H, n_configs, prev_states=[psi_gs], n_iterations=5000, lr=0.05, orth_weight=100.0, seed=142
    )
    
    E_ex = float(jnp.real((jnp.conj(psi_ex) @ H @ psi_ex) / (jnp.conj(psi_ex) @ psi_ex)))
    overlap = float(jnp.abs(jnp.conj(psi_gs) @ psi_ex))
    
    print(f"\n激发态能量: {E_ex:.8f} Ha")
    print(f"FCI 参考: {E_fcis[1]:.8f} Ha")
    print(f"误差: {abs(E_ex - E_fcis[1]):.6f} Ha ({abs(E_ex - E_fcis[1])/abs(E_fcis[1])*100:.3f}%)")
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
