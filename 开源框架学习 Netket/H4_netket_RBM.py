import netket as nk
import numpy as np
import matplotlib.pyplot as plt
import json
from pyscf import gto, scf, fci
import netket.experimental as nkx

# 设置H4分子的几何构型（线性排列，等间距）
bond_length = 0.74  # H-H键长（埃）
geometry = [
    ('H', (0., 0., 0.)),
    ('H', (bond_length, 0., 0.)),
    ('H', (2*bond_length, 0., 0.)),
    ('H', (3*bond_length, 0., 0.))
]

# 创建分子对象，使用STO-3G基组
mol = gto.M(atom=geometry, basis='STO-3G')

# 进行Hartree-Fock计算
mf = scf.RHF(mol).run(verbose=0)
E_hf = mf.e_tot
print(f"Hartree-Fock能量: {E_hf:.8f} Ha")

# 进行FCI计算作为参考
cisolver = fci.FCI(mf)
E_fci, fcivec = cisolver.kernel()
print(f"FCI能量: {E_fci:.8f} Ha")

# 使用NetKet创建哈密顿量
ha = nkx.operator.from_pyscf_molecule(mol)

# 创建希尔伯特空间
# H4分子在STO-3G基组下有4个空间轨道，8个自旋轨道
# 总电子数=4，自旋向上和向下各2个
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=4,  # 总空间轨道数
    s = 1/2,  # 自旋轨道的轨道角动量
    n_fermions_per_spin=(2, 2)  # 每种自旋的电子数
)

# 创建采样器 - 使用费米子跳跃采样器
# 对于分子系统，我们使用完全连接的图
g = nk.graph.Graph(edges=[(i, j) for i in range(hi.size) for j in range(i+1, hi.size)])
sa = nk.sampler.MetropolisFermionHop(
    hi, graph=g, n_chains=32, spin_symmetric=True, sweep_size=128
)

# 创建变分量子态 - 使用更复杂的RBM模型
ma = nk.models.RBM(
    alpha=2,  # 增加隐藏单元数量
    param_dtype=complex,
    use_visible_bias=False,
    use_hidden_bias=True
)
vs = nk.vqs.MCState(sa, ma, n_discard_per_chain=100, n_samples=2048)

# 设置优化器
opt = nk.optimizer.Sgd(learning_rate=0.03)  # 降低学习率
sr = nk.optimizer.SR(diag_shift=0.01, holomorphic=True)

# 创建VMC驱动器
gs = nk.driver.VMC(ha, opt, variational_state=vs, preconditioner=sr)

# 运行优化
exp_name = "h4_molecule"
gs.run(300, out=exp_name)

############## 绘图 #################

# 获取精确对角化能量（FCI能量）
ed_energies = np.array([E_fci])  # H4只有一个基态能量

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
plt.title("H4分子能量收敛")
plt.legend()
plt.grid(True)
plt.show()

# 打印最终结果
print(f"\n最终VMC能量: {y[-1]:.8f} Ha")
print(f"与FCI能量误差: {abs(y[-1] - E_fci):.8f} Ha")
print(f"相对误差: {abs(y[-1] - E_fci)/abs(E_fci)*100:.4f}%")
