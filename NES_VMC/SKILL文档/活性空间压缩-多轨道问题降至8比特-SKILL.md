# Skill：活性空间压缩 —— 把多空间轨道问题降到 8 比特以内

> 来源脚本：`NES_VMC/experiments/LiH分子/LiH.py`（LiH / STO-3G）
> 适用场景：任何 PySCF + NetKet 的 VMC 问题，当完整基组的自旋轨道数 N 过大
> （Hilbert 空间 ~ C(N, Nα)·C(N, Nβ) 爆炸）时，用活性空间（Active Space / CAS）
> 截断把问题压缩到可精确对角化、可全求和的规模。

---

## 1. 核心思想

LiH 在 STO-3G 下有 **6 个空间轨道 = 12 个自旋轨道**（Li: 1s, 2s, 2p_x/y/z 共 5 个，H: 1s 共 1 个），4 个电子（2α + 2β）。
直接做 VMC 的组态空间为 C(12,2)² = 4356 维，太大。

压缩分两步：

1. **轨道截断（CAS(N_e, N_o)）**：只保留对化学键/激发最重要的少数空间轨道，
   电子数不变。LiH 取最低的 4 个正则 HF 分子轨道（1σ ≈ Li1s、2σ ≈ Li2s/H1s 成键、
   3σ、1p_z），构成 **CAS(4,4)**。
2. **比特计数**：每个空间轨道对应 α、β 两个自旋轨道（2 个比特），
   故 4 空间轨道 → **8 比特**；物理 Hilbert 空间 = C(4,2)·C(4,2) = 6² = **36 维**，
   可以精确稠密对角化、FullSumState 全求和。

一句话：**问题规模由"活性空间轨道数"决定，而不是由"基组轨道数"决定。**

---

## 2. 操作步骤（可复用模板）

### Step 1：PySCF 求 RHF 与 FCI 基准

```python
from pyscf import gto, scf, fci, mp

mol = gto.M(atom=geometry, basis="STO-3G", verbose=0, spin=0)
mf = scf.RHF(mol).run()
E_RHF = mf.e_tot

# 完整空间 FCI 基准（多 roots，作激发态参考）
cisolver = fci.FCI(mf)
cisolver.nroots = 10
E_fcis, fcivec = cisolver.kernel()
```

### Step 2：选择活性轨道索引

```python
n_orbitals = 4                      # 活性空间轨道数 → 2*n_orbitals = 8 比特
active_idx = [0, 1, 2, 3]           # 0-based，取最低 4 个正则 MO
mo_active = mf.mo_coeff[:, active_idx]
```

正则 MO 已按轨道能升序排列，`[0,1,2,3]` 即"占据层 + 最低虚轨道"，
对应 LiH 的 CAS(4,4)（含冻结核 1σ，4 电子全部放进 4 轨道）。

### Step 3：用活性 MO 系数投影哈密顿量（关键 API）

```python
import netket as nk
import netket.experimental as nkx

ha_raw = nkx.operator.from_pyscf_molecule(
    mol,
    mo_coeff=mo_active,             # ← 传入活性系数矩阵即完成积分投影
    implementation=nk.operator.FermionOperator2ndJax,
)
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)
```

原理：`from_pyscf_molecule` 内部把 AO 积分 `h_μν, (μν|λσ)` 用 `mo_coeff`
变换到 MO 基 `h_pq, (pq|rs)` 并构造二次量子化哈密顿量。
**只传活性列 → 自动得到活性空间有效哈密顿量**（1 电子项 + 2 电子项均在活性轨道内），
无需手写积分变换。

### Step 4：构造压缩后的 Hilbert 空间

```python
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,                      # 4 空间轨道 = 8 自旋轨道
    s=1/2,
    n_fermions_per_spin=(n_alpha, n_beta),      # (2, 2)
)
# hi.size == 8 比特；物理维数 == C(4,2)^2 == 36
```

粒子数扇区约束使采样天然只遍历合法组态，比朴素 8 比特全空间（256 维）再小 7 倍。

### Step 5：NES 多态扩展空间（K 个态联合采样）

```python
K = 4                              # 基态 + 3 激发态
hi_ext = hi ** K                   # 8*K 比特联合空间
SINGLE_SIZE = hi.size              # = 8

# 跳跃边只在各 replica 内部（自旋守恒单激发 i→j）
alpha_orbs = list(range(n_orbitals))           # 0..3
beta_orbs  = list(range(n_orbitals, 2*n_orbitals))  # 4..7
single_edges_full = (
    list(itertools.combinations(alpha_orbs, 2))
    + list(itertools.combinations(beta_orbs, 2))
)
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    ext_edges += [(i + offset, j + offset) for i, j in single_edges_full]
```

要点：扩展空间的边按 `k * SINGLE_SIZE` 偏移复制，保证第 k 个 replica 的
α/β 轨道索引与单空间定义一致（α 在 [0,4)，β 在 [4,8)）。

---

## 3. 验证清单（压缩是否正确）

| 检查项 | 期望 |
|---|---|
| `hi.size` | == 2 × n_orbitals（本例 8） |
| `hi.physical_size` / 扇区维数 | == C(4,2)² = 36，可全对角化 |
| 活性空间 FCI 能量 vs 完整 FCI | CAS(4,4) 下二者应一致；若截断更狠则 E_CAS 高于 E_FCI，差值应在化学精度量级内 |
| 哈密顿量对角元 vs 组态 HF 能 | 取 HF 组态（注意：`all_states()[0]` 不是 HF，见坑①）算 ⟨σ|H|σ⟩ 应等于对应滑行列式能量 |
| `ha` 类型 | 应成功转成 `ParticleNumberAndSpinConservingFermioperator2nd`（采样高效实现） |

---

## 4. 已知坑（本项目实测）

1. **`SpinOrbitalFermions.all_states()[0]` 不是 HF 行列式**——枚举从最高轨道开始占；
   必须按占据数手动构造 HF 组态（前 n_alpha 个 α 位 + 对应偏移的 β 位 = 1）。
2. **轨道索引布局**：`SpinOrbitalFermions` 中 α 轨道占 `[0, n_orbitals)`，
   β 占 `[n_orbitals, 2*n_orbitals)`；构造边、NES 偏移、算符都要按这个布局对齐。
3. **活性空间选择影响激发态保真度**：垂直激发若涉及被截掉的虚轨道，
   CAS 内无法描述；需要更高激发时把 `active_idx` 换成自然轨道
   （MP2/FCI 约化密度矩阵对角化）或手动纳入关键虚轨道。
4. **`from_pyscf_molecule` 默认用完整 `mf.mo_coeff`**——忘记传 `mo_coeff=mo_active`
   会静默得到 12 自旋轨道的大空间，压缩失效。
5. 若冻结内层（如 Li 1s），需同时改 `n_fermions_per_spin` 与活性轨道，
   并在能量里加回冻结核贡献（本项目 LiH 基准未冻结，CAS(4,4) 含 1σ）。

---

## 5. 规模速查（LiH/STO-3G）

| 方案 | 空间轨道 | 比特 | 物理维数 | 可行性 |
|---|---|---|---|---|
| 完整基组 | 6 | 12 | 4356 | 采样勉强，FullSum 不可 |
| **CAS(4,4)（本脚本）** | **4** | **8** | **36** | **精确对角化 + 全求和 + NES K=4（32 比特）** |
| CAS(2,2)（注释备选） | 2 | 4 | 9 | 玩具级，丢失 σ→p 激发 |

压缩的本质收益：36 维让"投影哈密顿量 / 冻结方案 / 精确重叠"等
所有小系统验证手段（见 `H2_631G_projection.py`、`NES_VMC_deflation.py`）
在 LiH 上同样可用。
