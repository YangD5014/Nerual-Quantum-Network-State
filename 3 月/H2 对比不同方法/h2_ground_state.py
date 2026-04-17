"""
H2分子基态计算 - OpenFermion + PySCF 实现

输入H2分子的键长间隔，输出基态能量和各组态的概率分布
"""

import numpy as np
from scipy.linalg import eigh
import openfermion
from openfermion.chem import MolecularData
from openfermionpyscf import run_pyscf
from openfermion.transforms import jordan_wigner


def calculate_h2_ground_state(bond_length: float, basis: str = 'sto-3g') -> dict:
    """
    计算H2分子基态能量和波函数
    
    参数:
        bond_length: H-H键长 (Angstrom)
        basis: 基组名称，默认 'sto-3g'
    
    返回:
        dict: 包含基态能量、各组态概率等信息
    """
    geometry = [('H', (0, 0, 0)), ('H', (0, 0, bond_length))]
    
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        description=f'{bond_length}_angstrom'
    )
    
    molecule = run_pyscf(molecule, run_scf=True, run_fci=True)
    
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    fermion_hamiltonian = openfermion.get_fermion_operator(molecular_hamiltonian)
    qubit_hamiltonian = jordan_wigner(fermion_hamiltonian)
    
    n_qubits = molecule.n_qubits
    hamiltonian_sparse = openfermion.get_sparse_operator(qubit_hamiltonian, n_qubits=n_qubits)
    hamiltonian_matrix = hamiltonian_sparse.toarray()
    eigenvalues, eigenvectors = eigh(hamiltonian_matrix)
    
    exact_energy = eigenvalues[0]
    exact_state = eigenvectors[:, 0]
    
    n_states = 2 ** n_qubits
    probabilities = []
    binary_strings = [f"{i:0{n_qubits}b}" for i in range(n_states)]
    
    for i in range(n_states):
        basis_state = np.zeros(n_states, dtype=complex)
        basis_state[i] = 1.0
        probability = np.abs(np.vdot(exact_state, basis_state)) ** 2
        probabilities.append(probability)
    
    configurations = []
    for bs, prob in zip(binary_strings, probabilities):
        if prob > 1e-10:
            configurations.append({
                'state': bs,
                'probability': prob
            })
    
    return {
        'bond_length': bond_length,
        'n_orbitals': molecule.n_orbitals,
        'n_electrons': molecule.n_electrons,
        'n_qubits': n_qubits,
        'hf_energy': molecule.hf_energy,
        'fci_energy': molecule.fci_energy,
        'exact_energy': exact_energy,
        'configurations': configurations
    }


def print_results(result: dict):
    """打印计算结果"""
    print("=" * 60)
    print(f"H2分子基态计算结果 (键长: {result['bond_length']} Å)")
    print("=" * 60)
    print(f"分子轨道数: {result['n_orbitals']}")
    print(f"电子数: {result['n_electrons']}")
    print(f"量子比特数: {result['n_qubits']}")
    print("-" * 60)
    print(f"Hartree-Fock能量: {result['hf_energy']:.10f} Ha")
    print(f"FCI能量: {result['fci_energy']:.10f} Ha")
    print(f"对角化能量: {result['exact_energy']:.10f} Ha")
    print("-" * 60)
    print("基态波函数组态分布:")
    print(f"{'组态':<10} {'概率':<15} {'百分比':<10}")
    print("-" * 60)
    for config in result['configurations']:
        print(f"|{config['state']}>    {config['probability']:.6f}        {config['probability']*100:.2f}%")
    print("=" * 60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='计算H2分子基态能量和组态分布')
    parser.add_argument('-b', '--bond_length', type=float, default=2.2,
                        help='H-H键长 (Angstrom), 默认2.2')
    parser.add_argument('--basis', type=str, default='sto-3g',
                        help='基组名称, 默认sto-3g')
    
    args = parser.parse_args()
    
    result = calculate_h2_ground_state(args.bond_length, args.basis)
    print_results(result)
    
    return result


if __name__ == '__main__':
    main()
