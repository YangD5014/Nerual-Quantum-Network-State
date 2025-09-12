import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx

# 设置H4分子的几何构型
bond_length = 0.74
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
    ('H', (2*bond_length, 0., 0.)),
    ('H', (3*bond_length, 0., 0.))
]

# 创建分子对象
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)

# 获取原子坐标
coords = mol.atom_coords()

# Hartree-Fock计算
mf = scf.RHF(mol).run(verbose=0)
E_hf = mf.e_tot
print(f"Hartree-Fock能量: {E_hf:.8f} Ha")

# FCI计算
cisolver = fci.FCI(mf)
E_fci, fcivec = cisolver.kernel()
print(f"FCI能量: {E_fci:.8f} Ha")

# 使用NetKet创建哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)

# 创建希尔伯特空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=4,
    s=1/2,
    n_fermions_per_spin=(2, 2)
)

# 基于距离构建图
# 对于STO-3G基组，每个原子贡献一个轨道
# 轨道中心近似为原子核位置
edges = []
distance_threshold = 2.0 * bond_length  # 设置距离阈值

# α自旋轨道之间的连接（前4个轨道）
for i in range(4):
    for j in range(i+1, 4):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < distance_threshold:
            edges.append((i, j))

# β自旋轨道之间的连接（后4个轨道）
for i in range(4):
    for j in range(i+1, 4):
        dist = np.linalg.norm(coords[i] - coords[j])
        if dist < distance_threshold:
            edges.append((i+4, j+4))

print(f"构建的图包含 {len(edges)} 条边")

g = nk.graph.Graph(edges=[
    (0, 1), (0, 2), (0, 3),  # 轨道0与其他所有轨道连接
    (1, 2), (1, 3),          # 轨道1与剩余轨道连接
    (2, 3)                   # 轨道2与轨道3连接
])

# 创建采样器
sa = nk.sampler.MetropolisFermionHop(
    hi,
    graph=g,
    n_chains=64,
    spin_symmetric=True,
    sweep_size=hi.size * 4,
    reset_chains=True
)

# 使用Slater模型确保反对称性
ma = nk.models.RBM(alpha=2, param_dtype=complex, use_visible_bias=False)

# 创建变分量子态
vs = nk.vqs.MCState(
    sa,
    ma,
    n_discard_per_chain=200,
    n_samples=2000,
    seed=42
)

# 设置优化器
opt = nk.optimizer.Sgd(learning_rate=0.01)
sr = nk.optimizer.SR(
    diag_shift=0.05,
    holomorphic=True,
    solver=nk.optimizer.solver.cholesky
)

# 创建VMC驱动器
gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)

# 运行优化
exp_name = "h4_molecule_physical_graph"
gs.run(300, out=exp_name)

############## 绘图 #################

# 获取精确对角化能量（FCI能量）
ed_energies = np.array([E_fci])  # H2只有一个基态能量

# 读取日志数据
with open(f"{exp_name}.log") as f:
    data = json.load(f)

x = data["Energy"]["iters"]
y = data["Energy"]["Mean"]["real"]

# 绘制能量收敛曲线
plt.figure(figsize=(10, 6))
plt.axhline(ed_energies[0], color="red", linestyle="--", label=f"FCI能量 = {E_fci:.6f} Ha")
plt.plot(x, y, 'b-', label="VMC能量")
plt.xlabel("迭代步数")
plt.ylabel("能量 (Ha)")
plt.title("H2分子能量收敛")
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果
print(f"\n最终VMC能量: {y[-1]:.8f} Ha")
print(f"与FCI能量误差: {abs(y[-1] - E_fci):.8f} Ha")