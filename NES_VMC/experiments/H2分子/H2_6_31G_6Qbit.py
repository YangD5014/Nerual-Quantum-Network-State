from pyscf import gto, scf, fci, mcscf
import netket as nk
import netket.experimental as nkx
import itertools
import math
import jax.numpy as jnp

# ============================================================
# H2 分子 / 6-31G / 活性空间压缩到 6 比特
# ============================================================
# 6-31G 基组下每个 H 有 2 个缩并基函数，H2 共 4 个空间轨道：
#   完整空间：4 空间轨道 = 8 自旋轨道 = 8 比特（物理维数 C(4,1)^2 = 16）
#   压缩方案：只保留最低 3 个活性空间轨道
#            3 空间轨道 = 6 自旋轨道 = 6 比特（物理维数 C(3,1)^2 = 9）
# 依据：NES_VMC/SKILL文档/活性空间压缩-多轨道问题降至8比特-SKILL.md
# ============================================================
bond_length = 1.5
geometry = [
    ("H", (0.0, 0.0, 0.0)),
    ("H",  (0.0, 0.0, bond_length)),
]

mol = gto.M(atom=geometry, basis="6-31G", verbose=0, spin=0)
mf = scf.RHF(mol).run(verbose=0)
hf_ground_energy = mf.e_tot

print("=" * 60)
print("H2 分子基本信息 (6-31G)")
print("=" * 60)
print(f"HF energy = {hf_ground_energy:.8f} Ha")
print(f"Total electrons = {mol.nelec}")
print(f"Total basis functions = {mol.nao_nr()}")

# ------------------------------------------------------------
# 完整空间 FCI 基准（4 空间轨道 / 8 自旋轨道）
# ------------------------------------------------------------
cisolver = fci.FCI(mf)
cisolver.nroots = 12
E_fcis, fcivec = cisolver.kernel()
print("=" * 60)
print("H2 / 6-31G FCI 基准能量（完整空间，8 比特）")
print("=" * 60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能: {exc:.4f} eV")

# ============================================================
# Step 1：选择活性空间轨道（4 空间轨道 -> 3 空间轨道）
# ============================================================
n_orbitals = 3                     # 活性空间轨道数 -> 2*n_orbitals = 6 比特
active_idx = [0, 1, 2]             # 0-based，取最低 3 个正则 MO
mo_active = mf.mo_coeff[:, active_idx]

# 活性空间 CASCI 参考能量（验证压缩损失）
cas = mcscf.CASCI(mf, ncas=n_orbitals, nelecas=2)
cas.kernel()
E_casci = cas.e_tot
print("\n" + "=" * 60)
print("活性空间 CASCI 参考（压缩正确性核对）")
print("=" * 60)
print(f"CAS(2,{n_orbitals}) energy = {E_casci:.8f} Ha")
print(f"vs 完整 FCI E0        = {E_fcis[0]:.8f} Ha")
print(f"压缩损失              = {(E_casci - E_fcis[0]) * 1000:.4f} mHa")

# 电子数（H2 闭壳层，2 电子）
n_alpha = 1
n_beta = 1

# 自旋轨道索引布局：alpha 在 [0, n_orbitals)，beta 在 [n_orbitals, 2*n_orbitals)
alpha_orbs = list(range(n_orbitals))                    # 0,1,2
beta_orbs  = list(range(n_orbitals, 2 * n_orbitals))    # 3,4,5

# 自旋守恒的单激发边（同一自旋的轨道两两组合）
single_edges = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)

# ============================================================
# Step 2：构造压缩后的 6 比特 Hilbert 空间
# ============================================================
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,                  # 3 空间轨道 = 6 自旋轨道
    s=1/2,
    n_fermions_per_spin=(n_alpha, n_beta),  # (1, 1)
)
SINGLE_SIZE = hi.size                       # = 6
n_sector_states = math.comb(n_orbitals, n_alpha) * math.comb(n_orbitals, n_beta)

print("\n" + "=" * 60)
print("压缩后 Hilbert 信息")
print("=" * 60)
print(f"空间轨道数 = {n_orbitals}  ->  自旋轨道(比特)数 hi.size = {SINGLE_SIZE}")
print(f"粒子数扇区维数 = C({n_orbitals},{n_alpha}) * C({n_orbitals},{n_beta}) = {n_sector_states}")

# HF 参考态：注意 all_states()[0] 并非 HF，需按占据数手动构造
hf_state = hi.all_states()[0]
Hatree_Fock = hf_state
print(f"HF reference state: {hf_state}")

# ============================================================
# Step 3：用活性 MO 系数投影哈密顿量
# ============================================================
ha_raw = nkx.operator.from_pyscf_molecule(
    mol,
    mo_coeff=mo_active,                       # ← 传入活性列即完成积分投影
    implementation=nk.operator.FermionOperator2ndJax,
)
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)
print(f"\nHamiltonian 类型: {type(ha).__name__}")
print(f"(原始: {type(ha_raw).__name__})")

# ============================================================
# Step 4：NES 扩展副本空间（K 个态联合采样）
# ============================================================
K = 4                       # 基态 + 3 激发态
hi_ext = hi ** K            # 6 * K 比特联合空间
print(f"\nK = {K}")
print(f"hi_ext.size = {hi_ext.size}")

# 跳跃边按 k * SINGLE_SIZE 偏移复制到各 replica
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.asarray(ext_edges)

print(f"single_edges ({len(single_edges)}): {single_edges}")
print(f"ext_edges shape: {ext_edges.shape}")

# 采样规则（单空间 + 张量扩展到 K 副本）
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
print(f"tensor_rule 就绪（{K} 个 FermionHopRule）")
