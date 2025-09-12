import sys
sys.path.append('..')
from molecule_ham import build_H4
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx
from tqdm import tqdm

H4_bond_distance = np.linspace(0.5, 2.5, 25)
clusters = [(i, j) for i in range(4) for j in range(i+1,4 )]
clusters.extend([(i, j) for i in range(4,8) for j in range(i+1,8 )])
g = nk.graph.Graph(edges=clusters)
opt = nk.optimizer.Sgd(learning_rate=0.05)
sr = nk.optimizer.SR(
    diag_shift=0.05,
    holomorphic=True,
    solver=nk.optimizer.solver.cholesky
)
ma = nk.models.RBM(alpha=1, param_dtype=complex, use_visible_bias=False)

E_fci_list = []
E_vmc_list = []


for bond_distance in tqdm(H4_bond_distance, desc="计算H4分子在不同键长下的能量"):
    H4_ham,E_fci = build_H4(bond_distance)
    hi = H4_ham.hilbert
    
    sa = nk.sampler.MetropolisFermionHop(
    hi,
    graph=g,
    n_chains=64,
    spin_symmetric=True,
    sweep_size=hi.size * 4,
    reset_chains=True)
    # 创建VMC驱动器
    
    
    vs = nk.vqs.MCState(
        sa,
        ma,
        n_discard_per_chain=100,
        n_samples=2000,
        seed=42
    )

    gs = nk.driver.VMC(H4_ham, opt, variational_state=vs, preconditioner=sr)
    exp_name = "./VMC遍历H4/h4_molecule_RBM_alpha_1_R"+str(bond_distance)
    gs.run(120, out=exp_name)
    E_fci_list.append(E_fci)
    E_vmc_list.append(np.real(gs.energy.mean.item(0)))

# 将能量数据保存为npz文件
np.savez('./VMC遍历H4/h4_energy_results_RBM_alpha1.npz', 
         bond_distance=H4_bond_distance, 
         E_fci=E_fci_list, 
         E_vmc=E_vmc_list)
print("能量数据已保存到 h4_energy_results_RBM_alpha1.npz")
    

    
    
    
    
    
    

    
    
    

