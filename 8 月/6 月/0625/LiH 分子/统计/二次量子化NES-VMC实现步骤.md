# 二次量子化下 NES-VMC 的实现步骤（LiH / STO-3G / CAS(2,2) 实证）

> 配套代码：
> - [LiH.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py)  ← 二次量子化算符 + Hilbert 构造
> - [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py)  ← Ansatz + 采样器 + 机器 + 损失 + QGT
> - [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb)  ← 训练主循环
>
> 参考：
> - [NES-VMC算法详解-背景公式与代码实现.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC算法详解-背景公式与代码实现.md)  ← 第一性原理与公式速查

本文档按"读代码→改代码→跑代码"的顺序，把 LiH 实测代码 [LiH.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py) + [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py) + [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb) 的实现步骤拆成 10 步, 每步都附**真实代码片段 + 公式对照**。

---

## 第 0 步：背景 —— 为什么"二次量子化 + 神经网络 + 行列式"是 NES-VMC 的最优组合

### 0.1 二次量子化的两个核心便利

对于 K 条正交归一态的**联合**变分, 二次量子化提供了两个不可替代的好处：

**便利 ①：自旋轨道占据数 = 0/1 直接做神经网络的输入。**

$$
|\mathbf{x}\rangle = |x_1, x_2, \dots, x_{N_\text{orb}}\rangle, \quad x_i \in \{0, 1\}
$$

不需要手算 CI 系数, 不需要构造 Slater 行列式, 神经网络直接吃一个 $N_\text{orb}$ 维的 0/1 向量就能输出 $\log\psi(\mathbf{x})$。

**便利 ②：哈密顿量的二体算符自动被 PySCF+NetKet 处理, 配合 `get_conn_padded` 即可用。**

$$
H = \sum_{pq} h_{pq}\, a_p^\dagger a_q \;+\; \tfrac{1}{2}\sum_{pqrs} g_{pqrs}\, a_p^\dagger a_q^\dagger a_s a_r
$$

NetKet 的 `FermionOperator2ndJax` 在 Jax 内核下用 `get_conn_padded(x)` 一次返回 $(x', m_{x,x'})$ 列表, 直接喂给 VMC 能量计算, 完全免去手写矩阵元。

### 0.2 行列式结构 = K 条态自动正交

K 个独立副本 $(\mathbf{x}_1, \dots, \mathbf{x}_K)$ 在**扩展 Hilbert** $\mathcal{H}^\otimes K$ 上, 把它们的 K 个子 Ansatz 拼成 K×K 矩阵取行列式：

$$
\Psi_\theta(\mathbf{x}_1, \dots, \mathbf{x}_K) = \det\!\left[\psi_j(\mathbf{x}_i)\right]_{i,j=1}^{K}
$$

只要任意两子组态 $\mathbf{x}_i \neq \mathbf{x}_j$, 这个 K×K 矩阵**几乎处处可逆**, K 条 NES 态自动线性无关 → 自动正交归一（详见 2.2 节）。

### 0.3 LiH 的算例规模

| 量 | 值 |
|----|----|
| 几何 | Li-(0,0,0), H-(0,0,1.58 Å) |
| 基组 | STO-3G (6 个 AOs) |
| 冻结/活性 | 取前 4 个 canonical MOs (HOMO-1, HOMO, LUMO, LUMO+1) |
| 活性电子 | (2α, 2β) = 共 4e |
| 自旋轨道数 $N_\text{spin}$ | 8（4 α + 4 β）|
| Hilbert 维度 | $2^8 = 256$（CAS(2,2) 还可降到 36, 但全空间够用）|
| 同时拟合的态数 $K$ | 4（基态 + 3 激发态）|
| 扩展 Hilbert 维度 | $256^4 = 4.3 \times 10^9$, 但**采样只走 allowed 子空间** |

---

## 第 1 步：从分子几何到二次量子化算符

### 1.1 PySCF 算 H、轨道、FCI 参考

代码：[LiH.py:11-79](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L11-L79)

```python
from pyscf import gto, scf, fci
mol = gto.M(atom=[("Li", (0,0,0)), ("H", (0,0,1.58))], basis="STO-3G", verbose=0)
mf  = scf.RHF(mol).run(verbose=0)            # RHF
my_mp = mp.MP2(mf).run()                     # MP2 拿 natural orbitals
cisolver = fci.FCI(mf); cisolver.nroots = 6
E_fcis, fcivec = cisolver.kernel()          # FCI 6 条态
```

这一步 PySCF 已经在内部完成了：
1. 计算 STO-3G 基组积分（重叠矩阵 $S_{μν}$、动能 $T_{μν}$、核吸引 $V_{μν}$、双电子 $(\mu\nu|\lambda\sigma)$）；
2. RHF 解出 canonical 系数矩阵 $C_\text{MO}$，给出 RHF 占据数与轨道能；
3. FCI 在 Fock 空间上做精确对角化，给 6 条态的精确能量作为基准。

对应公式：

$$
E_\text{HF} = \sum_i h_{ii} + \tfrac{1}{2}\sum_{ij}\big(2 J_{ij} - K_{ij}\big), \quad
h_{ij} = \int \phi_i^*(r)\!\left(-\tfrac{1}{2}\nabla^2 - \sum_A \tfrac{Z_A}{|r-R_A|}\right)\phi_j(r)\,dr
$$

### 1.2 选 Active Space 并构造 MOs 系数

代码：[LiH.py:90-122](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L90-L122)

```python
n_orbitals = 4                          # 取 4 个 MOs: HOMO-1, HOMO, LUMO, LUMO+1
n_alpha_electrons = 2
n_beta_electrons  = 2
active_idx = [0, 1, 2, 3]
mo_active  = mf.mo_coeff[:, active_idx]  # shape (n_ao, 4)
```

### 1.3 二次量子化算符的两种 NetKet 实现

代码：[LiH.py:118-126](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L118-L126)

```python
ha_raw = nkx.operator.from_pyscf_molecule(
    mol, mo_coeff=mo_active,
    implementation=nk.operator.FermionOperator2ndJax,   # ② 二体算符的 Jax 内核
)
# 第一步：拿到完整 fermion operator
ha = nkx.operator.ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd(
    ha_raw
)  # 第二步：切成"保 N + 保 S_z" 的稀疏表示
```

`FermionOperator2ndJax` 的核心方法就是后面 VMC 反复调用的 `get_conn_padded(x)`：

```python
x_primes, mels = ha.get_conn_padded(x)
# x:       (n_spin,) 0/1
# x_primes: (n_conn, n_spin) 与 x 差 1~2 个电子的组态
# mels:     (n_conn,) 对应矩阵元
```

公式：

$$
\langle x'|H|x\rangle = h_{ij}\,\delta_{x', a_i^\dagger a_j x} + \tfrac{1}{2} g_{ijkl}\,\delta_{x', a_i^\dagger a_j^\dagger a_l a_k x}
$$

> 详尽推导见 [NES-VMC算法详解-背景公式与代码实现.md §2.3](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC算法详解-背景公式与代码实现.md)

---

## 第 2 步：构造 Hilbert + NES 扩展

### 2.1 单子 Hilbert（自旋轨道占据空间）

代码：[LiH.py:103-107](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L103-L107)

```python
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=n_orbitals,
    s=1/2,
    n_fermions_per_spin=(n_alpha_electrons, n_beta_electrons),
)
```

效果：
- $\text{hi.size} = 2 \cdot n_\text{orbs} = 8$（α 占 0-3, β 占 4-7）
- 满足 $\hat{N}_\alpha = 2$、$\hat{N}_\beta = 2$ 的合法组态有 $\binom{4}{2}^2 = 36$ 个

### 2.2 扩展 Hilbert（NES 的核心）

代码：[LiH.py:109-110](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L109-L110)

```python
K = 4       # 同时拟合 4 条态
hi_ext = hi ** K
```

`hi ** K` 在 NetKet 里就是 K 个单子的张量积：

$$
\mathcal{H}_\text{ext} = \mathcal{H}^{\otimes K}, \quad
|\mathbf{X}\rangle = |\mathbf{x}_1\rangle \otimes |\mathbf{x}_2\rangle \otimes \cdots \otimes |\mathbf{x}_K\rangle
$$

扩展 Hilbert 的"样本" $\mathbf{X}$ 编码为 `jnp.array` 长度 $K \cdot \text{hi.size} = 32$, 排列是 $(\mathbf{x}_1, \mathbf{x}_2, \mathbf{x}_3, \mathbf{x}_4)$。

### 2.3 NES 约束：K 个子组态互不相同

物理上要求：

$$
\forall\, i \neq j: \quad \mathbf{x}_i \neq \mathbf{x}_j
$$

否则行列式 $\Psi(\mathbf{x}_1, \dots, \mathbf{x}_K) \equiv 0$, 整链死掉。这正是为什么 `NESFermionHopRule.transition` ([NES_VMC.py:1110-1125](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1110-L1125)) 在每次跳跃后要调 `_check_duplicate`, 失败就回退到旧态。

### 2.4 Sampler 跳跃边 ext_edges

代码：[LiH.py:154-167](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L154-L167)

```python
single_edges_full = list(itertools.combinations(alpha_orbs, 2)) \
                  + list(itertools.combinations(beta_orbs, 2))    # α/β 内部两两交换
ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges_full:
        ext_edges.append((i + offset, j + offset))                # 每个副本一份边
ext_edges = jnp.asarray(ext_edges)
```

每条边 $(i, j)$ 在第 $k$ 个子组态上**交换**第 $i$ 个自旋轨道的电子到第 $j$ 个自旋轨道, 这就实现了"单激发跃迁" $a_j^\dagger a_i$。代码里直接 `sigma.at[i].set(sigma[j])` + `sigma.at[j].set(sigma[i])`, 等价于一次自旋轨道跳跃。

### 2.5 Hatree-Fock 参考态

代码：[LiH.py:143](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L143)

```python
Hatree_Fock = hi.all_states()[0]
```

`hi.all_states()[0]` 是自旋轨道占据数按字典序最小的合法组态, 即 RHF 的占据: $\alpha$ 占 0,1, $\beta$ 占 4,5, 形如 `[1,1,0,0, 1,1,0,0]`。

该 HF 态是后续 **gauge-fixing** 的基准（详见第 4 步）。

---

## 第 3 步：复值 FFNN Ansatz + Gauge-Fixing 行列式式

### 3.1 单态 Ansatz：复值三层 FFNN

代码：[NES_VMC.py:26-40](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L26-L40)

```python
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, param_dtype=complex, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, param_dtype=complex, rngs=rngs)
        self.output  = nnx.Linear(hidden_dim, 1, param_dtype=complex, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
```

公式：

$$
\log\psi_j(\mathbf{x}) \;=\; W_3 \, \tanh\!\big(W_2 \tanh(W_1 \mathbf{x} + b_1) + b_2\big) + b_3
$$

其中 $W_\ell, b_\ell$ 都是**复数**矩阵/向量, 整个 logψ 是个复标量, 这是为了正确表达 NES 复值局域能量（见第 7 步）。

### 3.2 NESTotalAnsatz 的 K×K 矩阵

代码：[NES_VMC.py:110-160](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L110-L160)

原版（不带数值稳定）：

```python
for i in range(K):
    for j in range(K):
        L = L.at[i, j].set(self.single_ansatz_list[j](x_single[i]))
sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
log_Psi = log_abs_det + 1j * jnp.angle(sign)
```

公式：

$$
L_{ij} = \log\psi_j(\mathbf{x}_i), \quad
\log\Psi = \log\det(e^L) = \mathrm{slogdet}(e^L)
$$

直接 `jnp.exp(L)` 在 $|L|$ 大时立刻溢出（logψ 的实部可达 ~30+），所以用 `slogdet(e^L) = K·max\,\text{Re}(L) + \text{slogdet}(e^{L - K·max})$ 的等价改写 —— 这正是下一步的稳定化。

### 3.3 Gauge-Fixed 版：NESTotalAnsatz_gauge_stable

代码：[NES_VMC.py:236-322](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L236-L322)

```python
class NESTotalAnsatz_gauge_stable(nnx.Module):
    def __init__(self, n_spin_orbitals, n_states, hidden_dim, ref_state, *, rngs):
        ...
        self.ref_state = jnp.asarray(ref_state, dtype=jnp.complex64)
        self.single_ansatz_list = nnx.List([... K 个 SingleStateAnsatz ...])

    def _forward_single(self, x_single):                       # x_single: (K, n_spin)
        # 1) ref 列：stop_gradient 包, 只用一次
        ref_logs = jnp.array([
            stop_gradient(ans(self.ref_state)) for ans in self.single_ansatz_list
        ])                                                    # (K,)

        # 2) gauge-fixed 矩阵: L_ij = logψ_j(x_i) - logψ_j(ref)
        L = jnp.zeros((K, K), dtype=jnp.complex64)
        for i in range(K):
            for j in range(K):
                L = L.at[i, j].set(
                    self.single_ansatz_list[j](x_single[i]) - ref_logs[j]
                )

        # 3) 数值稳定化: 减 max Re(L), 阻止 exp 溢出
        shift  = stop_gradient(jnp.max(jnp.real(L)))
        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        # 4) 稳定 log det
        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
        log_Psi_gauge  = log_Psi_stable + K * shift           # 给 sampler 的真实 log|Ψ|

        return log_Psi_gauge, L_stable, shift, L             # 4 个返回值
```

公式：

$$
\begin{aligned}
L_{ij}^\text{(gauge)} &= \log\psi_j(\mathbf{x}_i) - \log\psi_j(\mathbf{x}_\text{HF}) \\
\text{shift} &= \max_{i,j}\,\text{Re}\,L_{ij}^\text{(gauge)} \\
L_{ij}^\text{(stable)} &= L_{ij}^\text{(gauge)} - \text{shift} \\
\Psi_\text{stable} &= \exp(L^\text{(stable)}) \\
\log\Psi_\text{gauge} &= \mathrm{slogdet}(\Psi_\text{stable}) + K\cdot\text{shift}
\end{aligned}
$$

> 注意 **gauge-fixed** 与 **stable** 是两件事：
> - **gauge-fixing**（减 $\log\psi_j(\text{HF})$）：消除 $\psi_j$ 的整体相位任意性, 防止 K 条态在训练中"互相绕圈子"（gauge drift）。
> - **stable**（减 shift）：阻止 `jnp.exp(L)` 在大数上溢出。

为什么 `ref_state` 用 `stop_gradient` 包住？因为 ref 态在反向时只需要它的 logψ 数值参与 E_L, 不应该让"ref 态对应的参数"被梯度穿过 —— 否则会改变 sub-ansatz 内部"以谁为锚" 的设定。

---

## 第 4 步：自定义采样器（NES 硬约束）

### 4.1 为什么 NetKet 默认 Metropolis 采样器不够

NetKet 的 `MetropolisSampler` 会对 `hi_ext` 上任意一个 `spin` 做翻转/跳跃, 但**不保证** K 个子组态互不相同。任意一对子组态 $\mathbf{x}_i = \mathbf{x}_j$ 会让 $\Psi \equiv 0$, 整链概率密度归零 → 死链。

### 4.2 NESFermionHopRule：约束式 Metropolis 规则

代码：[NES_VMC.py:1083-1151](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1083-L1151)

```python
@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    edges: jnp.ndarray
    K: int = nk.utils.struct.static_field()
    single_size: int = nk.utils.struct.static_field()

    def _check_duplicate(self, sigma_ext):
        # sigma_ext: (batch, K*single_size)  →  reshape (batch, K, single_size)
        sub = sigma_ext.reshape((-1, self.K, self.single_size))
        # pair_equal: (batch, K, K)  True 表示两子组态完全相同
        pair_equal = jnp.all(sub[:, :, None, :] == sub[:, None, :], axis=-1)
        diag_mask  = jnp.eye(self.K, dtype=jnp.bool_)[None, :, :]
        off_diag_dup = jnp.where(diag_mask, False, pair_equal)   # 排除对角
        return jnp.any(off_diag_dup, axis=(-2, -1))              # (batch,)

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        # 1) 随机选一条边
        batch_size = sigma.shape[0]
        e_idx = jax.random.randint(rng, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]                                # (batch, 2)
        i, j = sel_e[:, 0], sel_e[:, 1]

        # 2) 在 (batch, K*single_size) 上做一次交换: σ[i] ↔ σ[j]
        sigma_cand = sigma.at[jnp.arange(batch_size), i].set(sigma[jnp.arange(batch_size), j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size), j].set(sigma[jnp.arange(batch_size), i])

        # 3) 检查新组态是否产生重复子组态；是就回退
        invalid    = self._check_duplicate(sigma_cand)
        new_sigma  = jnp.where(invalid[:, None], sigma, sigma_cand)
        return new_sigma, None
```

公式（一次 transition 相当于 Metropolis 提议）：

$$
\sigma' \;=\; \begin{cases}
T_\text{hop}(\sigma), & \text{若}\ \text{all\_diff}(\sigma') \\
\sigma, & \text{否则}
\end{cases}
\quad,\quad
T_\text{hop}: \sigma[i] \leftrightarrow \sigma[j]
$$

### 4.3 用法

[LiH 规范漂移-1.ipynb:170-174](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L170-L174)

```python
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext, rule=nes_rule, n_chains=16, sweep_size=30
)
```

`sweep_size=30` 表示每次 `sample(chain_length=200)` 调用时, 每条链上跑 30×200 = 6000 步 Metropolis 提议, 然后每 30 步取一次样本 → 每条链 200 个样本, 16 条链共 3200 个样本。

---

## 第 5 步：机器函数（machine）包装

### 5.1 机器函数的设计契约

`NetKet` 的 `Sampler.sample(machine=...)` 期待 `machine(params, sigma) -> log_psi`；`Hamiltonian.get_conn_padded` 期待单态机器; 损失/QGT 期望不同的"内部" 输出。 因此把 `NESTotalAnsatz_gauge_stable` 拆成**4 个**机器闭包, 每个 jit 化以吃 `total_params`：

| 机器 | 输入 | 输出 | 用途 | 代码 |
|------|------|------|------|------|
| `total_machine_log_Psi_gauge` | `(params, σ)` | `logΨ_gauge` | 喂给 NetKet 采样器 | [1549-1568](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1549-L1568) |
| `total_machine_L_stable` | `(params, σ)` | `L_stable = L - shift` | 构造 $\Psi_\text{stable}=\exp(L_\text{stable})$ | [1571-1585](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1571-L1585) |
| `total_machine_shift` | `(params, σ)` | `shift` | 给 HPsi 做 $\exp(\log\psi - \text{shift})$ | [1588-1602](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1588-L1602) |
| `create_single_machine_gauge_fixed` | `(params_j, σ)` | `logψ_j(σ) - logψ_j(ref)` | 计算 HPsi 中的单态 | [1525-1544](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1525-L1544) |

最后用 `create_machine_gauge_synthesis` ([1620-1625](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1620-L1625)) 一站式拿到 4 个机器。

### 5.2 形状兼容

所有机器都要正确处理 4 种 $\sigma$ 形状：

```text
(flat_size,)              single_flat       = (K * n_spin,)
(K, n_spin)               single_matrix
(batch, flat_size)        batch_flat        ← NetKet 采样器输出
(batch, K, n_spin)        batch_matrix
```

具体 `_as_single_matrix` / `_as_batch_matrix` 见 [NES_VMC.py:1232-1282](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1232-L1282)。

---

## 第 6 步：哈密顿作用 HPsi（数值稳定版）

### 6.1 单态 HPsi：Ham_psi_scaled

代码：[NES_VMC.py:474-533](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L474-L533)

```python
def Ham_psi_scaled(ha, single_machine, params, x, shift):
    """
    e^{-shift} Hψ(x) = Σ_{x'} H[x, x'] · exp(logψ(x') - shift)
    """
    x_primes, mels = ha.get_conn_padded(x)                # (n_conn, n_spin), (n_conn,)
    log_psi_vals   = single_machine(params, x_primes)     # (n_conn,)
    psi_vals_scaled = jnp.exp(log_psi_vals - shift)       # ★ 关键：先减 shift 再 exp
    return jnp.sum(mels * psi_vals_scaled)
```

公式：

$$
\big[\widetilde{H\psi}\big](x) \;=\; \sum_{x'} \langle x|H|x'\rangle\, \psi(x')\, e^{-\text{shift}}
$$

**为什么要先减 shift？** 因为 $\log\psi(x')$ 实部可达 +20~30, 直接 exp 会 inf。**减完 shift 再 exp 后, 最大 exp(0)=1, 数值永远安全**。最后乘 $e^{+K\cdot\text{shift}}$ 把常数乘回来, 与 $\Psi_\text{stable}$ 里的 $K\cdot\text{shift}$ 对消, 整体守恒。

### 6.2 K×K 矩阵：Ham_Psi_scaled

代码：[NES_VMC.py:536-607](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L536-L607)

```python
def Ham_Psi_scaled(ha, single_machine_list, total_params, x, shift):
    K = len(single_machine_list)
    HPsi = jnp.zeros((K, K), dtype=jnp.complex64)
    for i in range(K):
        xi = x_single[i]
        for j in range(K):
            params_j = total_params["single_ansatz_list"][j]
            val = Ham_psi_scaled(ha, single_machine_list[j], params_j, xi, shift)
            HPsi = HPsi.at[i, j].set(val)
    return HPsi
```

公式（每条 $(i, j)$ 元）：

$$
\widetilde{H\Psi}_{ij} = \sum_{x'} \langle \mathbf{x}_i|H|x'\rangle \, \psi_j(x')\, e^{-\text{shift}}
$$

整体 `Ham_Psi_scaled` 仍只是数值上的"前向", 与冻结设计兼容（详见第二部分的方案 ❷/❸）。

---

## 第 7 步：局域能量矩阵 E_L 与 NES 损失

### 7.1 局域能量矩阵

代码：[NES_VMC.py:609-713](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L609-L713)

```python
def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                            single_machine_list, total_params, x, return_aux=False):
    # 1) 取出 L_stable 和 shift
    L_stable = total_matrix_machine(total_params, x)        # (K, K) 或 (batch, K, K)
    shift    = total_max_machine(total_params, x)           # () 或 (batch,)

    # 2) 构造稳定化后的 Ψ 矩阵
    Psi_Matrix_stable = jnp.exp(L_stable)

    # 3) 构造稳定化后的 HPsi 矩阵
    HPsi_stable = Ham_Psi_scaled(ha, single_machine_list, total_params, x, shift)

    # 4) 解 Ψ E_L = HPsi 得局域能量矩阵
    E_L_matrix = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)

    # 5) loss = Re·Tr(E_L)
    loss_batch = jnp.real(jnp.trace(E_L_matrix, axis1=-2, axis2=-1))
    ...
```

公式（这一步是核心!）：

$$
\widetilde{E}_L \;=\; \Psi_\text{stable}^{-1}\, \widetilde{H\Psi}_\text{stable} \;=\; \Psi^{-1} H \Psi, \quad
\mathcal{L} \;=\; \mathrm{Re}\,\mathrm{Tr}\big(\widetilde{E}_L\big) \;=\; \sum_{k=1}^{K} E_k
$$

**为什么"trace = K 条态能量之和"？** 因为 $\Psi$ 把 K 个 Ansatz 拼成可逆矩阵, $\Psi^{-1} H \Psi$ 是把 H 用 K 条 Ansatz 做基底对角化, 对角元就是 K 个本征值, trace 就是 K 个本征值之和。**最小化 trace = 联合最小化 K 个态的能量**。这正是 NES-VMC 能"同时" 优化多个态的核心。

### 7.2 数值稳定性的完整链路

```text
logψ 数值 ~ +30
  → ψ = exp(logψ) 溢出 ❌

gauge-fixed: L_ij = logψ_j(x_i) - logψ_j(HF)  → 实部范围 [-30, +30]
  → ψ = exp(L) 仍可能溢出 ❌

减 shift: L_stable = L - max(Re L)   → 实部范围 (-∞, 0]
  → ψ_stable = exp(L_stable) 范围 (0, 1]   ✅
  → slogdet(ψ_stable) 稳定
  → 还原真实 logΨ = log|slogdet| + j·angle + K·shift
```

全程乘/除 $e^{\pm\text{shift}}$ 完全对消, 数学上**与原版 bit-by-bit 等价**。

### 7.3 valid mask（NaN/inf walker 屏蔽）

代码：[NES_VMC.py:695-711](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L695-L711)

```python
valid = (
    jnp.all(jnp.isfinite(Psi_Matrix_stable), axis=(-2, -1))
    & jnp.all(jnp.isfinite(HPsi_stable),       axis=(-2, -1))
    & jnp.all(jnp.isfinite(E_L_matrix),       axis=(-2, -1))
    & jnp.isfinite(loss_batch)
)
```

`_masked_mean_batch` ([NES_VMC.py:715-737](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L715-L737)) 用 `valid` 屏蔽：

$$
\bar{x} = \frac{\sum_{b} x_b \cdot \text{valid}_b}{\max(\sum_b \text{valid}_b,\, \epsilon)}
$$

---

## 第 8 步：NES-VMC 梯度

### 8.1 nes_vmc_gradient_stable 全流程

代码：[NES_VMC.py:740-904](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L740-L904)

```python
def nes_vmc_gradient_stable(ha, total_matrix_machine, total_max_machine,
                            total_machine, single_machine_list, total_params, x_batch, ...):
    # 1) E_L_batch, loss_batch（含 valid mask）
    loss_batch, E_L_batch, loss_aux = NES_loss_energy_stable(...)
    valid = loss_aux["valid"]

    # 2) E_L 在 batch 维上求 mean（masked）
    E_L_mean = _masked_mean_batch(E_L_batch, valid)

    # 3) 中心化 E_L 矩阵
    E_L_safe     = jnp.where(jnp.isfinite(E_L_batch), E_L_batch, 0.0)
    E_L_centered = E_L_safe - E_L_mean
    tr_centered  = jnp.trace(E_L_centered, axis1=-2, axis2=-1)
    tr_centered  = jnp.where(valid, tr_centered, 0.0)         # 非法 walker 屏蔽

    # 4) 对 total_machine 沿参数 0 求 holomorphic 梯度
    grad_logPsi   = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)  # (batch, ...) PyTree

    # 5) 加权平均: 权重 = tr(E_L_centered), 然后沿 batch 求 mean
    def weight_and_masked_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,)*(grad_component.ndim-1))
        valid_f = valid.astype(jnp.float32).reshape((-1,) + (1,)*(grad_component.ndim-1))
        return jnp.sum(weights * jnp.conj(grad_component) * valid_f, axis=0) \
               / jnp.maximum(n_valid, 1.0)
    grad = jax.tree.map(weight_and_masked_mean, dlogPsi_batch)

    # 6) loss_mean 同样 masked
    loss_mean = _masked_mean_batch(loss_batch, valid)
    return grad, loss_mean, E_L_mean
```

公式推导：

$$
\begin{aligned}
\mathcal{L} &= \mathrm{Re}\,\mathrm{Tr}\!\left(\Psi^{-1} H \Psi\right) = \mathrm{Re}\sum_{ij} (\Psi^{-1} H \Psi)_{ii} \\
&= \mathrm{Re}\sum_{ij} \sum_{k\ell} \Psi^{-1}_{ik} H_{k\ell} \Psi_{\ell j} \delta_{ij} \\
&= \mathrm{Re}\sum_{k\ell} H_{k\ell}\, (\Psi^{-1})_{k\ell}\, \Psi_{\ell k}
\end{aligned}
$$

取对 $\theta$ 的导, 注意 $\Psi$ 各列都依赖 $\theta$：

$$
\frac{\partial \mathcal{L}}{\partial \theta}
\;=\; \mathbb{E}_{\mathbf{X} \sim |\Psi|^2}\!\left[\,\mathrm{tr}\!\big(E_L - \bar{E}_L\big)\cdot \nabla_\theta \log\Psi(\mathbf{X})^*\,\right]
$$

代码中 $\bar{E}_L$ 就是 `E_L_mean`, $\mathrm{tr}(E_L - \bar{E}_L)$ 就是 `tr_centered`, $\nabla_\theta \log\Psi^*$ 就是 `jax.grad(..., holomorphic=True)`, 加权平均就是 `weight_and_masked_mean`。

### 8.2 make_grad_fn 闭包化

代码：[NES_VMC.py:1501-1521](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1501-L1521)

```python
def make_grad_fn(ha, total_matrix_machine, total_max_machine,
                 total_machine, single_machine_list):
    @jax.jit
    def grad_fn(total_params, x_batch):
        return nes_vmc_gradient_stable(
            ha, total_matrix_machine, total_max_machine,
            total_machine, single_machine_list, total_params, x_batch,
            return_aux=True,
        )
    return grad_fn
```

`@jax.jit` 让 5 个机器 + ha 都被闭包捕获, 重编译时**只按 total_params 的 PyTree 结构 hash**, 改 `hidden_dim` 等才会触发重新 trace。

---

## 第 9 步：量子几何张量（QGT）与自然梯度

### 9.1 QGT 公式

量子几何张量（也称 Fisher 信息矩阵）是 VMC 中"按物理距离而非欧氏距离"做下降的关键：

$$
S_{ij} \;=\; \langle \partial_i \log\Psi^*\, \partial_j \log\Psi \rangle \;-\; \langle \partial_i \log\Psi^*\rangle \langle \partial_j \log\Psi \rangle
$$

写成样本平均：

$$
S = \tfrac{1}{N}\sum_{b=1}^{N} (O_b - \bar{O})(O_b - \bar{O})^\dagger, \quad
O_b = \nabla_\theta \log\Psi(\mathbf{X}_b)
$$

代码：[NES_VMC.py:1435-1498](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1435-L1498)

```python
def make_qgt_fn(machine):
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift):
        n_samples = sigma.shape[0]
        # 1) 沿样本求 ∂logΨ
        grad_tree_batch = vmap_grad_logpsi(params, sigma)
        # 2) PyTree → (n_samples, n_params) 矩阵
        O = flatten_batched_pytree(grad_tree_batch, n_samples)
        # 3) 中心化
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        # 4) S = (1/N) O_centered^\dagger O_centered
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))                # 强制 Hermitian
        return S + diag_shift * jnp.eye(S.shape[0])  # ★ Tikhonov 正则
    return qgt_fn
```

### 9.2 数值技巧

- `flatten_batched_pytree` ([1420-1433](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1420-L1433)) 替代 `ravel_pytree`, 避免 PyTree reshape 时的内存拷贝。
- `0.5 * (S + S^\dagger)` 强制 Hermitian, 避免浮点误差导致特征值虚部。
- `diag_shift * I` (Tikhonov 正则) 保证 S_reg 可逆, 经验值 0.01~0.5, LiH K4 用 0.1。

### 9.3 自然梯度更新

$$
\delta\theta \;=\; S_\text{reg}^{-1}\, \nabla_\theta \mathcal{L}, \quad
\theta \leftarrow \theta - \eta\, \delta\theta
$$

代码：[LiH 规范漂移-1.ipynb:968-974](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L968-L974)

```python
qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)   # S_reg: (N, N)
ng_flat     = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)    # δθ ∈ C^N
grad_update = unravel_fn(ng_flat)                             # PyTree
```

> QGT 求逆是 LiH K4 训练中最贵的步骤, N~1832 时 `jnp.linalg.solve` 约 8ms/step。冻结浅层 sub-ansatz 后 N_active = (3/4)·N, 求解时间按平方缩减到 ~56%。

---

## 第 10 步：训练主循环

### 10.1 初始化（[LiH 规范漂移-1.ipynb:841-914](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L841-L914)）

```python
# 超参
N_CHAINS, N_WARMUP, N_SAMPLES_PER_CHAIN, SWEEP_SIZE = 16, 200, 200, 30
N_ITER, Natural_Grad = 500, True
clip_norm, lr, qgt_diag_shift = 20.0, 0.1, 0.1

# 模型
total_ansatz = NESTotalAnsatz_gauge_stable(
    SINGLE_SIZE, K, SINGLE_SIZE + K, ref_state=Hatree_Fock, rngs=nnx.Rngs(11)
)
# 4 个机器闭包
(machine_log_Psi, machine_L_stable, machine_shift, machine_L_gauge) = \
    create_machine_gauge_synthesis(total_ansatz)
# K 个单态 machine（HPsi 用）
single_machine_list = [
    create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
    for a in total_ansatz.single_ansatz_list
]

# 梯度 & QGT
grad_fn = make_grad_fn(ha, machine_L_stable, machine_shift,
                       machine_log_Psi, single_machine_list)
qgt_fn  = make_qgt_fn(machine_log_Psi)

# 采样器 & 优化器
nes_sampler = nk.sampler.MetropolisSampler(hi_ext, nes_rule, N_CHAINS, SWEEP_SIZE)
optimizer   = optax.chain(optax.clip_by_global_norm(clip_norm), optax.sgd(lr))
opt_state   = optimizer.init(total_params)
sampler_state = nes_sampler.init_state(machine_log_Psi, total_params, jax.random.PRNGKey(21))
```

### 10.2 训练主循环（[LiH 规范漂移-1.ipynb:944-1029](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L944-L1029)）

```python
history = {k: [] for k in [
    'step', 'loss', 'E_Lmatrix', 'samples', 'log_Psi_mean', 'log_Psi_min', 'log_Psi_max',
    'grad_norm_raw', 'grad_norm_natural', 'grad_norm_clipped', 'psi_cond',
    'energy_0st', 'energy_1st', 'energy_2st', 'energy_3st', 'params'
]}

for step in range(N_ITER):
    # ── 1. 采样 ───────────────────────────────────────────────
    samples_raw, sampler_state = nes_sampler.sample(
        machine=machine_log_Psi, parameters=total_params,
        state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
    x_batch = samples_raw.reshape(-1, hi_ext.size).reshape(-1, K, SINGLE_SIZE)

    # ── 2. NES 梯度（含 Loss / E_L）────────────────────────────
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)
    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)

    # ── 3. QGT 自然梯度（可选）──────────────────────────────────
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)
        ng_flat     = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_update = grad_raw
        grad_norm_natural = grad_norm_raw

    # ── 4. 裁剪 + 更新 ───────────────────────────────────────
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    clip_transform = optax.clip_by_global_norm(clip_norm)
    clipped_grad, _ = clip_transform.update(grad_update, opt_state[0], total_params)
    grad_norm_clipped = jnp.linalg.norm(ravel_pytree(clipped_grad)[0])

    if grad_norm_clipped == 0.0:
        logger.info(f"【Step {step}】梯度范数为 0, 结束迭代")
        break

    history['params'].append(total_params)               # 先存旧参数
    total_params = optax.apply_updates(total_params, updates)

    # ── 5. 监控 ──────────────────────────────────────────────
    psi_cond = jnp.linalg.cond(machine_L_stable(total_params, x_batch[0:1])[0])
    log_Psi_batch = machine_log_Psi(total_params, x_batch)
    eig_vals, _   = jnp.linalg.eig(E_L_mean)
    sort_idx      = jnp.argsort(eig_vals.real)
    eig_vals      = eig_vals[sort_idx]

    history['step'].append(step)
    history['loss'].append(loss_mean)
    history['E_Lmatrix'].append(E_L_mean)
    history['samples'].append(samples_raw)
    history['log_Psi_mean'].append(log_Psi_batch.mean())
    history['log_Psi_min'].append(log_Psi_batch.min())
    history['log_Psi_max'].append(log_Psi_batch.max())
    history['grad_norm_raw'].append(grad_norm_raw)
    history['grad_norm_natural'].append(grad_norm_natural)
    history['grad_norm_clipped'].append(grad_norm_clipped)
    history['psi_cond'].append(psi_cond)
    history['energy_0st'].append(eig_vals[0])
    history['energy_1st'].append(eig_vals[1])
    history['energy_2st'].append(eig_vals[2])
    history['energy_3st'].append(eig_vals[3])

    # ── 6. 异常保护 ─────────────────────────────────────────
    if jnp.any(jnp.isnan(grad_raw_flat)) or grad_norm_raw > 5000.0:
        logger.warning(f"【Step {step} 告警】梯度异常")
```

### 10.3 关键量纲与监控

| 监控量 | 物理意义 | 期望范围 |
|--------|----------|----------|
| `log_Psi_mean/min/max` | 整体 logΨ 的尺度 | 0~40 |
| `psi_cond = cond(Ψ)` | Ψ 矩阵条件数 | 1 ~ 50（>1000 → Ψ 接近奇异）|
| `grad_norm_raw` | QGT 前原始梯度范数 | 0.01 ~ 10 |
| `grad_norm_natural` | 自然梯度范数 | 0.01 ~ 2（被 QGT 压住）|
| `grad_norm_clipped` | clip 后真实更新 | ≤ `clip_norm` (20) |
| `E_L[i,i]` (4 条) | K 条态能量之和 = Loss | 由 -7.7×4 = -30.8 渐近 |

---

## 实现总览：一图流

```text
[1] LiH.py                          # 分子 → H、HF、FCI、Hilbert、ext_edges
     │
[2] 二次量子化算符
     FermionOperator2ndJax(ha_raw)
     → ParticleNumberAndSpinConservingFermioperator2nd(ha)
     │
     ↓
[3] NetKet sampling
     NESFermionHopRule(edges, K, single_size)
     MetropolisSampler(hi_ext, rule, n_chains, sweep_size)
     │
     ↓
[4] Ansatz
     SingleStateAnsatz          (复值 FFNN,  ~229 复数参数 / 个)
     NESTotalAnsatz_gauge_stable(K 个,  + ref_state)
     │
[5] Machine 包装
     create_machine_gauge_stable         → logΨ_gauge   (sampler 用)
     create_machine_matrix_gauge_stable  → L_stable     (loss 构造 Ψ)
     create_machine_max_gauge_stable     → shift        (loss 构造 HPsi)
     create_single_machine_gauge_fixed   → logψ_j - logψ_j(ref)  (HPsi 用)
     │
[6] Ham_Psi_scaled           # HPsi 矩阵（带 shift）
     │
[7] NES_loss_energy_stable   # E_L = Ψ⁻¹·H·Ψ,  loss = Re·Tr(E_L)
     │
[8] nes_vmc_gradient_stable  # ∂Loss/∂θ = E[(Tr(E_L) - Tr(Ē_L))·∇logΨ*]
     │
[9] QGT  S = (1/N)·∇logΨ†·∇logΨ + λI
     natural_gradient = solve(S, ∇Loss)
     │
[10] optax.clip + sgd + apply_updates
     ↓
   下一 step 重新采样, 循环 N_ITER=500 次
```

---

## 总结：实现 NES-VMC 的 10 步检查表

| 步骤 | 核心问题 | 关键代码 | 公式 |
|------|----------|----------|------|
| 1 | 分子/基组/参考 | [LiH.py:11-79](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L11-L79) | $E_\text{HF},\,E_\text{FCI}$ |
| 2 | 二次量子化算符 | [LiH.py:118-126](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L118-L126) | $H = \sum h_{pq} a_p^\dagger a_q + \tfrac{1}{2}\sum g_{pqrs} a_p^\dagger a_q^\dagger a_s a_r$ |
| 3 | Hilbert + 扩展 | [LiH.py:103-110](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py#L103-L110) | $\mathcal{H}_\text{ext} = \mathcal{H}^{\otimes K}$ |
| 4 | 采样器 | [NES_VMC.py:1083-1151](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1083-L1151) | $\sigma \to T_\text{hop}(\sigma)$, 拒绝重复子组态 |
| 5 | Ansatz | [NES_VMC.py:26-40](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L26-L40) | $\log\psi_j = W_3 \tanh(W_2 \tanh(W_1 x + b_1) + b_2) + b_3$ |
| 6 | Gauge-Fixing | [NES_VMC.py:236-322](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L236-L322) | $L_{ij} = \log\psi_j(x_i) - \log\psi_j(x_\text{HF})$ |
| 7 | 数值稳定 | [NES_VMC.py:236-322](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L236-L322) | $L_\text{stable} = L - \max\text{Re}\,L$ |
| 8 | 机器包装 | [NES_VMC.py:1525-1625](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1525-L1625) | 4 个 `machine(params, σ)` 闭包 |
| 9 | HPsi 矩阵 | [NES_VMC.py:474-607](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L474-L607) | $\widetilde{H\Psi}_{ij} = \sum_{x'} \langle x_i|H|x'\rangle\, \psi_j(x')\, e^{-\text{shift}}$ |
| 10 | 损失 | [NES_VMC.py:609-713](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L609-L713) | $\mathcal{L} = \mathrm{Re}\,\mathrm{Tr}\big(\Psi^{-1} H \Psi\big) = \sum_k E_k$ |
| 11 | 梯度 | [NES_VMC.py:740-904](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L740-L904) | $\partial_\theta \mathcal{L} = \mathbb{E}\big[(\mathrm{Tr}\,E_L - \mathrm{Tr}\,\bar{E}_L) \nabla_\theta \log\Psi^*\big]$ |
| 12 | QGT | [NES_VMC.py:1435-1498](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1435-L1498) | $S = \frac{1}{N}\sum (\nabla\log\Psi^* - \bar{O})(\nabla\log\Psi - \bar{O})^\dagger + \lambda I$ |
| 13 | 自然梯度 | [LiH 规范漂移-1.ipynb:968-974](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L968-L974) | $\delta\theta = S^{-1} \nabla \mathcal{L}$ |
| 14 | 训练循环 | [LiH 规范漂移-1.ipynb:944-1029](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L944-L1029) | 采样 → grad → QGT → clip → apply |

---

## 附：与已有文档的差异化定位

- [NES-VMC算法详解-背景公式与代码实现.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC算法详解-背景公式与代码实现.md) 偏"算法导论 + H₂ 算例", 用 H₂ 1.4 Å 简版说明, 不涉及 LiH / CAS 拆分 / gauge-fixing / 数值稳定化。
- 本文档聚焦**LiH 实测代码**, 涵盖**二次量子化算符的 NetKet 落地 + Gauge-Fixing + 数值稳定化 + QGT/自然梯度**全链路, 并直接引用 [LiH.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py) + [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py) + [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb) 的真实行号。
- 配套阅读：
  - [NES-VMC数值稳定性策略详解.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC数值稳定性策略详解.md) — 数值稳定化深入
  - [NES-VMC数值稳定性与GaugeFixed策略详解.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC数值稳定性与GaugeFixed策略详解.md) — gauge drift 现象
  - [NES-VMC诊断版训练代码函数详解.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/文档/NES-VMC诊断版训练代码函数详解.md) — 诊断版函数
  - [冻结子Ansatz方案.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/冻结子Ansatz方案.md) / [冻结子Ansatz_跳过梯度版.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/冻结子Ansatz_跳过梯度版.md) — 浅层冻结方案（mask + stop_gradient + 物理分离）
