import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx
from tqdm import tqdm

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

bond_lengths = np.linspace(0.4, 3.0, 14)

E_hf_list = []
E_fci_list = []
E_vmc_list = []
E_vmc_err_list = []

n_iterations = 300
n_samples = 512
learning_rate = 0.05
alpha = 1

print("=" * 60)
print("H2 分子键长扫描 - RBM 方法")
print("=" * 60)
print(f"键长范围: {bond_lengths[0]:.2f} - {bond_lengths[-1]:.2f} Å")
print(f"键长点数: {len(bond_lengths)}")
print(f"VMC 迭代次数: {n_iterations}")
print("=" * 60)

for bond_length in tqdm(bond_lengths, desc="扫描键长"):
    geometry = [
        ('H', (0., 0., 0.)),
        ('H', (bond_length, 0., 0.)),
    ]
    
    mol = gto.M(atom=geometry, basis='STO-3G')
    
    mf = scf.RHF(mol).run(verbose=0)
    E_hf = mf.e_tot
    E_hf_list.append(E_hf)
    
    cisolver = fci.FCI(mf)
    E_fci, fcivec = cisolver.kernel()
    E_fci_list.append(E_fci)
    
    ha = nkx.operator.from_pyscf_molecule(mol)
    
    hi = nk.hilbert.SpinOrbitalFermions(
        n_orbitals=2,
        s=1/2,
        n_fermions_per_spin=(1, 1)
    )
    
    g = nk.graph.Graph(edges=[(0,1),(2,3)])
    sa = nk.sampler.MetropolisFermionHop(
        hi, graph=g, n_chains=16, spin_symmetric=True, sweep_size=64
    )
    
    ma = nk.models.RBM(alpha=alpha, param_dtype=complex, use_visible_bias=False)
    vs = nk.vqs.MCState(sa, ma, n_discard_per_chain=10, n_samples=n_samples)
    
    opt = nk.optimizer.Sgd(learning_rate=learning_rate)
    sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)
    
    gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)
    
    exp_name = f"temp_h2_bond_{bond_length:.2f}"
    gs.run(n_iterations, out=exp_name)
    
    with open(f"{exp_name}.log") as f:
        data = json.load(f)
    
    E_vmc = data["Energy"]["Mean"]['real'][-1]
    E_vmc_list.append(E_vmc)
    
    E_vmc_err = abs(E_vmc - E_fci)
    E_vmc_err_list.append(E_vmc_err)
    
    tqdm.write(f"键长 {bond_length:.2f} Å: HF={E_hf:.6f}, FCI={E_fci:.6f}, VMC={E_vmc:.6f}, 误差={E_vmc_err:.2e} Ha")

print("\n" + "=" * 60)
print("计算完成！")
print("=" * 60)

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

ax1 = axes[0]
ax1.plot(bond_lengths, E_hf_list, 'b-o', label='Hartree-Fock', markersize=6, linewidth=2)
ax1.plot(bond_lengths, E_fci_list, 'r-s', label='FCI (Exact)', markersize=6, linewidth=2)
ax1.plot(bond_lengths, E_vmc_list, 'g-^', label='VMC (RBM)', markersize=6, linewidth=2)
ax1.set_xlabel('H-H Bond Length (Å)', fontsize=12)
ax1.set_ylabel('Energy (Ha)', fontsize=12)
ax1.set_title('H2 Molecule Energy vs Bond Length', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.semilogy(bond_lengths, E_vmc_err_list, 'g-o', label='VMC-FCI Error', markersize=6, linewidth=2)
ax2.set_xlabel('H-H Bond Length (Å)', fontsize=12)
ax2.set_ylabel('Energy Error (Ha)', fontsize=12)
ax2.set_title('VMC Energy Error vs FCI', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('H2_bond_scan_RBM.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n结果已保存到 H2_bond_scan_RBM.png")

print("\n" + "=" * 60)
print("数值结果汇总:")
print("=" * 60)
print(f"{'键长(Å)':<10} {'HF(Ha)':<12} {'FCI(Ha)':<12} {'VMC(Ha)':<12} {'误差(Ha)':<12}")
print("-" * 60)
for i, bl in enumerate(bond_lengths):
    print(f"{bl:<10.2f} {E_hf_list[i]:<12.6f} {E_fci_list[i]:<12.6f} {E_vmc_list[i]:<12.6f} {E_vmc_err_list[i]:<12.2e}")

idx_min = np.argmin(E_fci_list)
print(f"\n平衡键长 (FCI): {bond_lengths[idx_min]:.2f} Å, 能量: {E_fci_list[idx_min]:.6f} Ha")
print(f"平衡键长 (VMC): {bond_lengths[idx_min]:.2f} Å, 能量: {E_vmc_list[idx_min]:.6f} Ha")

results = {
    'bond_lengths': bond_lengths.tolist(),
    'E_hf': E_hf_list,
    'E_fci': E_fci_list,
    'E_vmc': E_vmc_list,
    'E_vmc_err': E_vmc_err_list
}

with open('H2_bond_scan_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n数值结果已保存到 H2_bond_scan_results.json")
