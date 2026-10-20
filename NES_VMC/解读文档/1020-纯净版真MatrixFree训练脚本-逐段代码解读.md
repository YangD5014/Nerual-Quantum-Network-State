# 纯净版真 Matrix-Free 训练脚本 · 逐段代码解读

- 日期：2026-10-20
- 解读对象：`experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py`（443 行）
- 组织方式：严格按脚本执行顺序，每个代码块给出「做什么 → 输入/输出 → 数学原理 → 工程注意点」。全文以本脚本自身为对象，不依赖其他脚本。

---

## 0. 这个脚本在算什么（30 秒背景）

H2/6-31G 分子，8 个自旋轨道（8 bit 组态串），2 个电子。已有一个训练好的 K=4 NES-VMC 解（4 条列函数张成的子空间 $\mathrm{span}(\Phi)=\mathrm{span}(\varphi_0..\varphi_3)$，含近似基态与三个低激发态）。本脚本的任务：**冻结这 4 个态，在与它们正交的剩余子空间里训练 2 条新列函数，求出第 4、5 激发态**。

数学骨架三句话：

1. **投影靠行列式**：新列 $\psi_m$ 不直接优化，而是优化 $\chi_m=\psi_m-\Phi\,c_m$（$\Phi c_m$ 是 $\psi_m$ 在冻结 span 上的投影），$\chi_m$ 自动与冻结子空间正交。
2. **投影系数 $c$ 用蒙特卡洛 Gram 估计**：$c_m=G^{-1}M_m$，其中 $G_{ij}=\langle\varphi_i|\varphi_j\rangle$、$M_{m,j}=\langle\varphi_j|\psi_m\rangle$ 都是全空间内积——但本脚本从不物化这些向量，而是在一批辅助采样上按重要性权重估计它们（§6 阶段 1）。
3. **一切运算逐点**：网络对单个组态 $x$ 求值、哈密顿量用 `get_conn` 局部连接、行列式是 $2\times2$、广义本征是 $2\times2$——没有任何一步的代价依赖希尔伯特空间维数 $D=16$（或真实分子里的 $10^{10}$）。

脚本的完整流程：

```
读冻结源(旧网络参数+v_rot) → fresh 初始化 2 条 active 列
→ [循环 N_ITER 步]
     aux 采样(单组态链) → MC Gram 估 c/c2 → EMA 尺度更新(同步变换 c)
     → 主采样(2-walker 联合链) → 梯度 → SR 预处理 → 裁剪 → 参数更新
     → matrix-free 监控 (Loss, λ̂=eig⟨E_L⟩, 健康度)
→ 物理态提取 (MC Gram 广义本征 → λ̂, v_active)
→ 保存 npz/pkl → 画 4 联监控图
```

---

## 1. 头部、import 与全局配置（L1–L55）

[L1–27](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L1-L27) 是 docstring：算法定位、运行命令、环境变量（`NES_N_ITER`/`NES_AUX_M`/`NES_SCALE_BETA`/`NES_INIT_SEED`）。

[L28–52](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L28-L52) import 三组：

| 来源 | 成员 | 在本脚本中的角色 |
|---|---|---|
| `NES_VMC_V1` | `create_single_machine_gauge_fixed` | 把列网络包装成「单组态 logψ 机器」并做 gauge 固定 $\psi(x_{HF})=1$ |
| | `NESFermionHopRule` | MCMC 移动规则：在 K 份副本内各做一次保粒子数的 Hop 跳跃 |
| | `Ham_psi_scaled` | 逐点算 $e^{-s}(H\psi)(x)=e^{-s}\sum_{x'}H[x,x']\psi(x')$，只用 `get_conn` |
| `NES_VMC_deflation_mf` | `make_mf_bundle` | 造出整套前向/梯度/QGT/Gram 估计函数（§5） |
| | `update_scales` | 尺度 EMA 更新（§6 阶段 2） |
| `H2_6_31G_8Qbit_K2` | `ha`/`hi`/`hi_ext`/`ext_edges`/`single_edges`/`Hatree_Fock`/`K`/`SINGLE_SIZE` | pyscf 积分 → netket 费米子哈密顿量、单组态希尔伯特空间 `hi`(8bit)、联合 walker 空间 `hi_ext`(16bit=2×8bit)、Hop 边表、HF 参考组态、K=2、单份比特数 8 |

`scipy_eigh`（L39）只用于 $2\times2$ 广义本征（§7）。

[L54–55](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L54-L55)：`jax_enable_x64` 是硬要求（float32 下 QGT 病态会出负特征值导致梯度爆炸）；断言脚本按 K=2/8bit 设计。

---

## 2. Ansatz：列网络的定义（L58–L87）

### `SingleStateAnsatz_SharedBackbone`（L58–73）

一条列函数 $\psi_j(x)$ 的实参数复输出 FFNN：

```
x (8,) → Linear(8→16) → tanh → Linear(16→16) → tanh → 两个头
head_u → u(x)   head_v → v(x)
输出 logψ = log[(u+1e-12) + i(v+1e-12)]
```

- **输入**：单个 8-bit 组态（float）；**输出**：一个复数 `logψ`。
- **为什么存 logψ 而不是 ψ**：VMC 里处处用到 $\nabla\log\psi$ 与 $\log|\Psi|$；行列式机器内部再取 exp。
- **`+1e-12`**：防止 $u=v=0$ 时 `log(0)` 出 -inf。
- **`param_dtype=float`**：参数严格实数——复相位由 `head_v` 表达（激发态必须复输出，实正定波函数表示不了）。

### `NESTotalAnsatz_stable`（L76–87）

K 条上述列网络的容器（`nnx.List`），每条用 `jax.random.split` 出的独立子种子初始化——**列与列初始化互不相同**，否则行列式恒零。同一个类被实例化两次：`n_states=4`（冻结旧网络，L140）与 `n_states=K`（active，L158）。

---

## 3. 日志与超参（L90–L132）

[L90–109](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L90-L109)：日志同时写 `日志/` 文件与终端，文件名带启动时间戳 `time_str`，后续所有产物共用这个 ts。

[L111–127](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L111-L127) 超参分三组：

| 组 | 参数 | 含义 |
|---|---|---|
| 主采样 | `N_CHAINS=16 × N_SAMPLES_PER_CHAIN=200`，`SWEEP_SIZE=20` | 每步 3200 个 16-bit 联合样本 |
| 优化 | `lr=0.1`，`qgt_diag_shift=0.1`，`clip_norm=20`（全局）/`per_param_clip=0.5`（逐参数） | SGD + SR 自然梯度，双层裁剪 |
| aux/Gram | `AUX_M=1024`（8 链×128），`EPS_MIX=0.25`，`SCALE_BETA=0.9` | Gram 估计批大小、混合分布权重、尺度 EMA |
| 其他 | `N_ITER=1500`（env 可覆盖），`INIT_SEED=11`，`SAVE_INTERVAL=20` | |

[L129–132](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L129-L132) 冻结源路径：npz（K=4 训练的物理态提取结果）与 pkl（K=4 训练的完整历史）。**本脚本只从 npz 取 `v_rot`，从 pkl 取 `params[-1]`**——见下节为什么。

---

## 4. 步骤 0：冻结列的「逐点表示」（L134–L154）

[L135–138](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L135-L138)：

```python
v_rot = npz["v_rot"]                      # (4,4)
old_params = pkl["params"][-1]            # 4 条列网络的最终参数
```

K=4 训练收敛后，物理近似本征态与网络列的关系是**列约定** $\varphi_k=\sum_j\psi_j^{old}\,v_{rot}[j,k]$。npz 里同时存了 `phi_vecs`（16 维向量）和 `v_rot`；本脚本**只需要 4×4 的 `v_rot`**，因为冻结列不存向量、只存「怎么在任何组态 $x$ 上逐点算出 $\varphi_i(x)$」的配方：

$$\varphi_i(x)=\sum_{j=0}^{3}\psi_j^{old}(x;\theta^{old})\,v_{rot}[j,i]$$

[L140–146](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree_Pure.py#L140-L146)：

```python
old_ansatz = NESTotalAnsatz_stable(SINGLE_SIZE, n_states=4, hidden_dim=SINGLE_SIZE+4, rngs=nnx.Rngs(11))
frozen_sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in old_ansatz...]
frozen_params = [old_params["single_ansatz_list"][j] for j in range(4)]
```

- `hidden_dim=12` 必须与当年 K=4 训练逐字一致——`create_single_machine_gauge_fixed` 用这个新结构的 graphdef 去 merge 旧参数，结构不符直接报错（免费的结构校验）。
- `create_single_machine_gauge_fixed(a, Hatree_Fock)` 把每条列网络归一到 $\psi_j^{old}(x_{HF})=1$ 规范。**为什么必须做这一步**：K=4 训练时列就是在这个规范下收敛的，`v_rot` 的系数是针对这个规范定义的；换规范则列约定断掉，$\varphi_i$ 就拼错了。`Hatree_Fock` 只是单个 8-bit 参考组态（除法用的基准点），不是全空间枚举。
- `frozen_sms`/`frozen_params` 从此以**闭包常量**身份进入 bundle：它们从不接受 `jax.grad` 追踪 ⇒ 冻结天然成立（stop-gradient），不需要任何额外机制。

---

## 5. active 初始化、bundle 与采样器（L156–L216）

### 5.1 fresh active（L157–163）

```python
total_ansatz = NESTotalAnsatz_stable(SINGLE_SIZE, n_states=K, hidden_dim=SINGLE_SIZE+K, rngs=nnx.Rngs(seed))
sms = [create_single_machine_gauge_fixed(a, Hatree_Fock)[0] for a in ...]
total_params = nnx.split(total_ansatz)[1]
```

2 条 active 列全新随机初始化（seed=11）。**为什么必须 fresh、不能沿用旧列**：投影 $\chi=\psi-\Phi c$ 之后，若 $\psi$ 本来就在 $\mathrm{span}(\Phi)$ 里，则 $\chi\approx 0$，$2\times2$ 行列式奇异、训练立刻病态。fresh 随机列投影后几乎必然非零。

### 5.2 bundle（L167–168）

```python
bundle = make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms, n_spin=8, eps=EPS_MIX)
```

一次装配，返回 7 个逐点函数（内部实现属 `NES_VMC_deflation_mf.py`，这里讲**本脚本怎么用、每个在算什么**）：

| 成员 | 输入 → 输出 | 计算的数学对象 |
|---|---|---|
| `cols(p, σ, f, g)` | (M,8) 组态 → `Φ̃ (M,4)`, `Ψ̃ (M,2)` | **缩放基**：$\tilde\varphi_i=e^{-f}\varphi_i$、$\tilde\psi_m=e^{-g_m}\psi_m$，逐点网络求值 + `v_rot` 组合 |
| `aux_logq(p, σ)` | (M,8) → `(M,)` 实数 | $\log q=\log\big(\sum_i|\tilde\varphi_i|^2+\varepsilon\sum_m|\tilde\psi_m|^2\big)$：Gram 估计用的辅助分布（未归一，$q/Z$ 才是分布） |
| `estimate_grams(p, σ_aux, f, g)` | aux 样本 → `c, c2, max_phi2, max_psi, aux` | MC Gram（§6 阶段 1 详解） |
| `machine_batch(p, Σ, c, f, g)` | (B,16) 联合 walker → `(B,)` 复数 | $\log\Psi_F=\mathrm{slogdet}\,[\,\Phi̱\,|\,\Chī\,]$：$\tilde\chi_m=\tilde\psi_m-\tilde\Phi c_m$ 后做 $2\times2$ 行列式。**行列式反对称性使 $\Psi_F$ 对 $\chi_m$ 的冻结 span 分量不敏感——投影在数学上自动实现** |
| `grad_fn(p, σ_flat, c, c2, f, g)` | 主样本 → `grad_raw, loss_mean, E_L_mean, aux` | 力梯度 + 监控量（§6 阶段 4/5） |
| `qgt_fn(p, σ_flat, η, c, c2, f, g)` | 主样本 → `S (P,P)` | 实部 QGT + diag_shift（§6 阶段 4） |
| `h_cols(...)` | — | 本脚本未直接使用（提取段自己调 `Ham_psi_scaled` 重算 $H$ 作用，§7） |

### 5.3 两个采样器（L174–190）

```python
def aux_machine(params_aux, sigma):  return bundle["aux_logq"](params_aux, sigma)
def sample_machine(params, sigma):   p,cc,cc2,ff,gg = params; return bundle["machine_batch"](...)
```

- **aux 采样器**（L183–186）：希尔伯特空间 `hi`（8 bit），移动规则 `NESFermionHopRule(single_edges, K=1)`——单组态内的 Hop（把一个电子从轨道跳到空轨道，保自旋保粒子数）。`machine_pow=1` ⇒ 按 $\sqrt{q}$ 而非 $q$ 采样。
- **主采样器**（L187–190）：`hi_ext`（16 bit = 2 份 8 bit 副本拼接），`NESFermionHopRule(ext_edges, K=2)`——每步对 2 份副本各尝试 Hop，副本间不交换。按 $|\Psi_F|$ 采样（`machine_pow=1`）。
- **为什么都 `machine_pow=1`**：按 $|\Psi|^2$ 精确采样时，轨迹一旦滑进 HF 盆地，稀有双激发组态出现频率≈0，其梯度信号消失，训练「冻死」（零方差冻结）。按 $|\Psi|$ 采样后，梯度端乘 $w\propto|\Psi|$ 权重修正，加权期望精确还原 $|\Psi|^2$ 公式，且稀有组态贡献从 $\propto\psi\eta^2$ 恢复为 $\propto\psi\eta$。

### 5.4 优化器与预热（L192–207）

`optax.chain(clip_by_global_norm(20), sgd(0.1))`（L192–195）：全局范数裁剪 + 朴素 SGD（自然梯度方向在循环里手工做，见 §6 阶段 4）。

L198–202：aux 链初始化后**先空跑一轮**——MCMC 初始状态在 $\tilde\varphi$ 未缩放（f=g=0）时分布偏斜，预热避免首步 Gram 拖尾偏差。

L204–207：主链 RNG key=21；`c_cur/c2_cur` 置零阵（首步 `estimate_grams` 立刻覆盖；行列式在 c=0 时退化为 $\det[\Phi̱|\Psī]$，有限非零，采样合法）。

---

## 6. 训练循环（L219–L337）——每步五件事

### 阶段 1：aux 采样 + MC Gram 估计 c/c2（L222–238）

```python
s_aux, aux_state = aux_sampler.sample(...)              # (8,128,8) → sigma_aux (1024,8)
c_mc, c2_mc, max_phi2, max_psi, aux_g = bundle["estimate_grams"](total_params, sigma_aux, f, g)
```

**要算的东西**：$c=G^{-1}M$，其中（省略缩放号）$G_{ij}=\langle\tilde\varphi_i|\tilde\varphi_j\rangle=\sum_x\overline{\tilde\varphi_i(x)}\,\tilde\varphi_j(x)$（16 维求和），$M_{mi}=\langle\tilde\varphi_i|\tilde\psi_m\rangle$。$c_2$ 是它的「哈密顿版」：$c_2=G^{-1}(M_2-\hat G\,\hat H\,G^{-1}M)$ 型组合，供 $E_L$ 里 $H\chi$ 的表示用。

**MC 估计（Z 抵消引理）**：在 $q$ 的样本上按 $1/q$ 加权，

$$\hat G_{ij}=\frac1{\mathcal M}\sum_k\frac{\overline{\tilde\varphi_i(y_k)}\,\tilde\varphi_j(y_k)}{q(y_k)}\;\longrightarrow\;\frac1Z\langle\tilde\varphi_i|\tilde\varphi_j\rangle .$$

所有 Gram 块带**同一个** $1/Z$，于是 $\hat G^{-1}\hat M\to(G/Z)^{-1}(M/Z)=G^{-1}M$——**归一化常数严格抵消，不需要算 $\|\Psi\|^2$**。这就是「全空间内积可以在采样组态上逐点估计」的全部原理。

- `machine_pow=1` 的第二个作用：$q$ 含 $\varepsilon\sum|\tilde\psi|^2$ 分量，保证 active 列的组态也被采到（否则 $M$ 块对 $\psi$ 无覆盖）。
- `max_phi2/max_psi`：本批 $\max_i|\tilde\varphi_i|^2$、$\max_m|\tilde\psi_m|^2$，供阶段 2。
- `aux_g`：`grams_finite`（非有限则整步作废，L229–238）、`w_ess`/`max_w_ratio`（权重健康度：$1/q$ 加权的有效样本数与最大权重比）。
- **为什么混合分布要含 ε 项**：纯按 $\sum|\tilde\varphi|^2$ 采样则 $\tilde\psi$ 支撑覆盖差，$M$ 块噪声爆炸；ε=0.25 是覆盖/方差的折中。

### 阶段 2：EMA 尺度更新 + c 同步变换（L240–249）

```python
f_np, g_np = update_scales(float(f), np.asarray(g), max_phi2, max_psi, beta=beta)
trans = np.exp(d_f - d_g)                    # 每列 e^{Δf−Δg_m}
c_cur = c_mc * trans ;  c2_cur = c2_mc * trans
```

**问题**：列函数的绝对尺度是规范自由度（$\psi_m\to e^{c}\psi_m$ 不改物理），放任漂移则 $|\tilde\psi|$ 可到 $10^{3}$，$1/q$ 权重退化、行列式溢出。

**做法**：`update_scales` 把每列的 log 尺度钉在「批最大值」的 EMA 上：$f\leftarrow\mathrm{EMA}(\tfrac12\log\max|\tilde\varphi|^2)_{\text{等价}}$，$g_m$ 同理（首步 β=1 直接吸收）。尺度一变，$\tilde\varphi\to e^{-\Delta f}\tilde\varphi$、$\tilde\psi_m\to e^{-\Delta g_m}\tilde\psi_m$，则 $\chi_m=\psi_m-\sum_ic_{mi}\varphi_i$ 要保持**方向不变**，系数必须同步变换 $c_{mi}\to c_{mi}\,e^{\Delta f-\Delta g_m}$——这就是 `trans`。首步 β=1.0：初值 f=g=0 与真实尺度差几个数量级，EMA 慢收敛会拖坏前几十步。

### 阶段 3：主采样（L251–262）

2-walker 联合链采 3200 个样本。`sample_machine` 内部：16-bit 串拆成 2 个 8-bit 组态 → `machine_batch` 算 $\chi$、拼 $2\times2$、`slogdet`。`sigma_flat (3200,16)` 是后续梯度/QGT 的输入。

### 阶段 4：梯度 / SR（L264–299）

```python
grad_raw, loss_mean, E_L_mean, aux = bundle["grad_fn"](total_params, sigma_flat, c_cur, c2_cur, f, g)
```

**`grad_fn` 在算什么**（内部细节属工具箱，这里给数学）：

- 每个 walker 上：$A=e^{-s}\,[\,\Phi̱\,|\,\Chī\,]$（$2\times2$），$Z_{nj}=e^{-s}(H\psi_j)(x_n)$ 用 `get_conn` 逐点算 → $E_L=A^{-1}Z$（$2\times2$ 局域能量矩阵）。
- $\mathrm{loss\_mean}=\mathrm{Re\,tr}\langle E_L\rangle_w$：active 块两列的 Rayleigh 商之和（冻结部分贡献是常数，不在 Loss 里）。
- 力：$g=2\,\mathrm{Re}\,\big\langle\mathrm{tr}(E_L-\langle E_L\rangle_w)\,\overline{\nabla_\theta\log\Psi_F}\big\rangle_w$，梯度只流向 active 列（冻结参数是闭包常量；且 autodiff 过 `slogdet` 自动得到投影切空间 $\partial\chi=Q\partial\psi$）。
- 实参数复输出 ⇒ Wirtinger：$\nabla\log\Psi=\nabla\mathrm{Re}\log\Psi+i\nabla\mathrm{Im}\log\Psi$，两次实值 `jax.grad`。
- 重要性加权：样本按 $|\Psi_F|$ 采，梯度均值乘 $w\propto|\Psi_F|$ 归一化权重还原 $|\Psi_F|^2$ 期望。
- `aux` 里的 `valid_ratio`/`ess`/`should_skip`：$A$ 阵条件数监控——`should_skip=True` 表示本批样本里行列式大面积病态，梯度不可信。

**外层防线**（L268–299）：

- `ravel_pytree` 把参数梯度展平 → 逐参数范数超 0.5 则整体缩放（L273–280，`n_grad_clip` 计数）；
- NaN / raw>5000 只告警不拦截（L282–284，拦截靠 should_skip）；
- SR：`qgt_fn` 给 $S_w=\mathrm{Re\,Cov}_w(\nabla\log\Psi_F)+0.1I$（$\approx290\times290$，参数空间，与 $D$ 无关），`solve` 得自然梯度，再过 optax（全局裁剪 20 + SGD 0.1）。**度量与期望必须同用 Re(S)**——对复 S solve 后丢虚部是错的（$\mathrm{Re}(S^{-1}g)\neq\mathrm{Re}(S)^{-1}\mathrm{Re}(g)$）。

### 阶段 5：matrix-free 监控（L301–316）

```python
E_h = 0.5*(E_L_mean + conj(E_L_mean.T))
lam_hat = np.sort(np.linalg.eigvalsh(np.asarray(E_h)).real)
```

**原理**：$\langle E_L\rangle$（$2\times2$）的特征值就是两个 active 态能量 $\lambda_4,\lambda_5$ 的蒙特卡洛实时估计——收敛时应趋近 $(-0.02803,\,-0.01450)$。**symmetrize** 是因为 $\langle E_L\rangle$ 理论 Hermitian、数值上反对称部全是采样噪声，`eigvalsh` 需要 Hermitian 输入。**注意**：这是含噪监控（偶发 ±10 mHa 尖峰属正常），最终值以 §7 提取为准。

日志四行（L306–316）：Loss/λ̂/尺度 → 采样健康（valid/ESS/aux_ess/aux_maxw/accept）→ 梯度三范数（raw/per_param/natural）→ 分隔线。natural 范数是 SR 后更新方向的大小，膨胀是失稳前兆。

### 历史追加与中间保存（L318–337）

每步把 loss/λ̂/diag/参数全量入列（L318–328；`params_history` 每步存全量参数——400–1500 份 ×~290 参数，float64 下仅几 MB，换来的是**事后逐步重算任何全空间量**的能力）。每 20 步落盘一次中间 pkl（L330–337，`N_ITER>5` 条件让 smoke 测试不产生中间文件；中间版只留最近 15 份参数防文件膨胀）。

---

## 7. 物理态提取：MC Gram 广义本征（L342–L377）——本脚本的数学高潮

训练收敛后，active 两列 $\chi_1,\chi_2$ 张成剩余子空间里能量最高的变分平面，但**物理本征态是它们的特定线性组合**。要找组合系数 $v$ 与能级 $\lambda$，需解广义本征

$$M_\chi\,v=\lambda\,S_\chi\,v,\qquad (S_\chi)_{mn}=\langle\chi_m|\chi_n\rangle,\quad (M_\chi)_{mn}=\langle\chi_m|H|\chi_n\rangle .$$

全空间做法是物化 $\chi$（16 维）再算——本脚本全程不物化，全部在采样组态上估计：

**L345–350 重新采一批 aux 样本并估 c**：

```python
s_aux, aux_state = aux_sampler.sample(..., (total_params, f, g), ...)
c_f, *_ = bundle["estimate_grams"](total_params, sigma_ext, f, g)
```

为什么重采：提取批必须与加权用的 $q$ **同一尺度**（采样分布=加权分布，Z 抵消引理的适用前提）。训练循环里样本按旧尺度采、尺度随后更新，直接复用最后一批配新尺度权重会引入 $q_{old}/q_{new}$ 比值偏差。链状态 `aux_state` 延续，不重新预热。`grams_finite` 失败则退回训练末 `c_cur`（L351–353）。

**L355–357 构造 χ̃**：`Phi, Psi = bundle["cols"](...)` → `chi = Psi - Phi @ c_f`，形状 (1024, 2)——**每个采样组态上的 χ̃ 值**，不是 16 维向量。

**L358–364 构造 Hχ̃**：

```python
Hpsi[m] = Ham_psi_scaled(sms[m], x=sigma_ext, shift=g[m])          # (1024,) 每列
Hphi[j] = Ham_psi_scaled(frozen_sms[j], x=sigma_ext, shift=f) @ V  # (1024,4)
Hchi = Hpsi - Hphi @ c_f
```

$H\psi$ 逐点 = $\sum_{x'}H[x,x']\psi(x')$，只用 `get_conn`（每组态 ≤13 连接）。$H\varphi_i=\sum_j v_{rot}[j,i]\,H\psi_j^{old}$ 由 $H$ 线性在采样点组合。$\chi$ 的 $H$ 作用同理线性展开。**这里 shift 用 `g[m]`/`f` 与 cols 一致 ⇒ $H\tilde\chi=e^{-g_m}(H\chi_m)$，与 L357 的 $\tilde\chi$ 同基**。

**L365–368 加权 Gram**：

```python
q = Σ|Phi|² + EPS_MIX·Σ|Psi|² ;  wi = 1/(q·M)
S_chi = χ† diag(wi) χ ;  M_chi = χ† diag(wi) Hχ
```

$\frac1M\sum_k\overline{\tilde\chi_m(y_k)}\,\tilde\chi_n(y_k)/q(y_k)\to\frac1Z\langle\tilde\chi_m|\tilde\chi_n\rangle$，$M_\chi$ 同带 $1/Z$ ⇒ **广义本征两边同乘 Z 不变，λ 与 v 无偏**。这就是 Z 抵消引理第二次、也是更重要的一次应用（第一次在 `estimate_grams` 算 c）。

**L369–373 数值防线与求解**：对称化（理论 Hermitian，反对称部是噪声）+ $S$ 加 $10^{-12}I$（防 MC 噪声下近奇异）+ $M+10^{-14}S$（ridge）→ `scipy_eigh(M,S)` 解 $2\times2$ 广义本征 → 按 λ 实部升序排。**成本**：一次 aux 采样 + 两次 `estimate_grams` 量级的逐点运算 + 一个 $2\times2$ 本征，与 $D$ 无关。

**输出的物理含义**：$\hat\varphi_k=\sum_m\tilde\chi_m\,v_{active}[m,k]$——第 $4{+}k$ 激发态的**表示配方**（不是向量）：在任何组态 $x$ 上，用旧网络 + active 网络 + `c_f/f/g/v_rot/v_active` 即可逐点求值。这正是大体系里波函数唯一可行的存储形态。

---

## 8. 保存（L379–L400）

- **npz**（L380–391）：`lam_extract`（最终能级）、`v_active`（提取旋转）、`c_extract`、`v_rot_frozen`、最终尺度 `scales_f/g`、`seed`、耗时/跳过/裁剪计数、`loss_final`。配合 pkl 里的最终参数，完整重建 $\hat\varphi_{4,5}$。
- **history pkl**（L392–399）：每步全量参数 + loss + λ̂ + diag + 尺度。任何后续诊断（包括用全空间重算精度曲线）都从这份文件出发，**不需要重训**。

## 9. 绘图（L402–L443）

4 联图，全部是 matrix-free 监控量（图内标签英文，避免中文字体缺 glyph）：

1. λ̂₀/λ̂₁ vs step（eig⟨E_L⟩，含噪）；
2. Loss vs step（NaN 步先掩码再画——连续 NaN 会把线整段切断）；
3. 采样健康三线（valid_ratio、主 ESS 占比、aux w_ess 占比）；
4. 梯度范数 raw/natural（semilogy，NaN 与非正值掩码）。

## 10. 工程注意点汇总

- **冻结是闭包级的**：`frozen_sms/frozen_params/v_rot` 从不进入 `jax.grad` 追踪，无需显式 stop-gradient。
- **gauge 一致性是隐形契约**：`v_rot` 只对「gauge-fixed 机器 + hidden_dim=12 + 同一 `Hatree_Fock` 参考点」成立；三者任一改变，冻结列就拼错，且**本脚本不会报错**（无自检）——更换冻结源时必须先另行验证重建保真度。
- **`trans` 同步变换不能漏**：尺度更新后若不同步 c，`machine_batch` 用的 $\chi$ 与 Gram 用的 $\chi$ 差每列一个实因子，投影失效、冻结剩余回升。
- **提取批必须新采**（§7 第一条）：复用训练末批 + 更新后的尺度会破坏 Z 抵消前提。
- **λ̂ 监控 vs 提取值的分工**：前者每步、3200 样本、有尖峰；后者一次、1024 样本、block 广义本征，是最终答案。两者差 ~0.1 mHa 属噪声预期。
- **Loss 只含 active 块**：收敛目标是 $\nu_4+\nu_5\approx-0.04254$（剩余子空间最低两本征值之和），不是全部六态能量之和。
- **aux_ess 偏低（几十/1024）是常态**：$1/q$ 加权方差大，实测不危害训练；更大体系需复查此项。
