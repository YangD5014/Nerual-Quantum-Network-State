# 方案 A' 训练脚本逐函数解读：Matrix-Free 的准确边界与 `build_full_stage` 的角色

- 日期：2026-10-19
- 解读对象：`NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py`（方案 A' 训练脚本）
- 依据方法文档：`1016-NES-VMC冻结式子空间扩展-方法全解与进展报告.md`（下称「方法报告」，引用记作 §n）
- 配套实现解读（数学细节）：`1016-方案A-prime-MC-Gram投影真MatrixFree-实现解读.md`
- 涉及工具箱：`NES_VMC_deflation_mf.py`（训练引擎）、`NES_VMC_deflation.py` / `NES_VMC_deflation_real.py`（staging/oracle）、`NES_VMC_V1.py`（gauge/采样规则/Hψ 原语）、`H2_6_31G_8Qbit_K2.py`（体系定义）

---

## 0. 一页总览：代码地图与「更新路径」审计

脚本按执行顺序分为 9 个区块。第三列标注该区块是否位于**训练更新路径**（采样 → 前向/梯度 → QGT/SR → 参数更新）上——这是 Matrix-Free 主张的管辖范围：

| 区块 | 行号 | 内容 | 在更新路径上？ | 接触 $2^N=16$ 全空间？ |
|---|---|---|---|---|
| ① Ansatz 类 | L57–86 | 实参数复输出 FFNN、K 列容器 | 是（前向） | 否（逐点） |
| ② 日志/超参 | L89–131 | 采样规模、SR、aux、尺度、冻结源路径 | — | — |
| ③ Step 0 预检 | L133–172 | `build_full_stage`、$F$、oracle $\nu$、冻结重建保真度断言 | **否**（启动诊断） | **是** |
| ④ 初始化预检 | L174–197 | fresh active 列 + $\mathrm{cond}(\chi)$ 选 seed | **否**（启动诊断） | **是** |
| ⑤ bundle+采样器 | L199–239 | `make_mf_bundle`、aux/主双 MCMC、optax | 装配 | 否 |
| ⑥ 训练循环 1–4 段 | L261–351 | aux Gram → 尺度 → 主采样 → 梯度/SR 更新 | **是** | **否** |
| ⑦ 训练循环 5 段 | L353–411 | 全空间监控（c_err、λ_sub、冻结剩余…） | **否**（旁路诊断） | **是** |
| ⑧ 提取与验收 | L416–440 | `extract_physical_states` → φ̂₄/φ̂₅ 验收表 | **否**（收尾诊断） | **是** |
| ⑨ 保存与绘图 | L442–532 | npz/pkl/6 联图 | **否** | **是** |

**一句话结论**：更新路径（①⑤⑥）全程逐点、维数无关；全空间稠密运算只出现在③④⑦⑧⑨五处**诊断白名单**里，全部删掉训练照样收敛，只是失去一切验证能力。这正是方法报告 §7 「全空间稠密运算只存在于循环外/旁路」主张在本脚本中的体现（详见 §1）。

---

## 1. 核心问题：说好 Matrix-Free，为什么还有 `build_full_stage`？

### 1.1 Matrix-Free 的准确定义（方法报告 §7）

「Matrix-Free」的管辖对象是**训练更新路径的每步计算量**：采样、局域能量 $E_L$、力梯度、QGT、SR 求解、参数更新——这些环节的成本只依赖（样本数 × 每组态连接数 × 参数数），与希尔伯特空间维数 $D=2^N$ 无关。它**不是一句「整个文件禁止出现稠密数组」的语法禁令**。

方法报告 §7 的逐操作审计表针对的就是更新路径；§0.2 同时指出：$D=16$ 的小系统「小到可以整稠密对角化，这使本项目拥有金标准」——稠密 $H$ 恰恰是用来**证明** matrix-free 训练正确性的参照物。

### 1.2 本脚本中稠密/全空间对象的全部出场位置

`build_full_stage(ha, hi)`（[NES_VMC_deflation.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/NES_VMC_deflation.py#L59-L78)）返回 `H_dense` (16,16) 与 `all_states` (16,8)。追踪这两个对象的全部下游，恰好是五处，且**没有任何一条边指向 `grad_fn` / `qgt_fn` / 采样 machine / `optimizer.update`**：

1. **启动 oracle 预检**（L137–138）：`deflation_oracle(F, H_dense)` 算压缩谱最低两本征值 $\nu$——方法报告 §2.3 的 Loss 理论目标 $\mathcal{L}^\ast=\sum_i E_{\mathrm{ray}}(\varphi_i)+\nu_4+\nu_5$ 与「方法偏差下限」就来自这里。没有 $\nu$，训练收敛到什么值才算「饱和」都无从判定。
2. **启动冻结重建保真度断言**（L154–159）：旧网络在 `all_states` 上逐点求值成 16 维列，与 npz 里存的 $\varphi_k$ 比对（>0.999），验证 gauge 约定（方法报告 §4.3）没接错。
3. **初始化 $\mathrm{cond}(\chi)$ 预检**（L174–197）：fresh active 列全空间投影后条件数检查，选出不使行列式病态的 seed（方法报告 §5 工程纪律）。
4. **每步旁路监控**（L353–402）：精确重投影 $\chi$、对比 `c_err = |c_{MC}-c_{exact}|`、块广义本征 $\lambda_{sub}$、冻结剩余重叠、$\max|A|$ 漂移告警。这些量**只写进日志和 history，不回流任何训练状态**。
5. **最终验收与绘图**（L416–440、L471–532）：`extract_physical_states` 在 $\chi$ 张成的子空间上解 $2\times2$ 广义本征得 $\hat\varphi_{4,5}$，与 FCI 本征矢算重叠、与 $\nu$ 算偏差——方法报告 §10 的「验收四线」全部依赖稠密参照。

### 1.3 为什么这五处对验证方案 A' 是必需的

方案 A' 的全部风险集中在一个替换上：把精确内积 $c=F^\dagger A$ 换成蒙特卡洛 Gram 估计 $c=\hat G^{-1}\hat M$（辅助 MCMC，见 §9.4）。要证明这个替换成立，必须同时持有「MC 值」和「精确值」做对照：

- `c_err` 曲线证明 MC Gram 的误差量级（实测相对误差 ~1e-4）；
- `frozen_res`（$|F^\dagger\chi_n|$）证明投影严格成立（实测 ~1e-34）；
- $\lambda_{sub}$ 与 $\nu$、$E^{FCI}$ 三线对照证明能量收敛目标正确；
- FCI 重叠证明提取出的物理态本身好，而不只是能量对。

这些都是**小系统特权**（方法报告 §0.2）：正因为 $D=16$ 存得下、算得起，才能给 matrix-free 主张提供金标准证据。方案 B' 用同样的策略（稠密 $H$ 仅每 25 步诊断），本脚本把监控做成了每步——只是诊断频率的差异，不改变「更新路径维数无关」的定性。

### 1.4 推到真实分子时怎么处理

$D\sim10^{6}\text{–}10^{20}$ 时 `build_full_stage` 的 `ha.to_dense()` 根本存不下。届时：③④⑦⑧⑨五处代码整体删除或降级（oracle 换成抽样估计、c_err 失去参照、验收改用局部观测量），训练路径 ①⑤⑥ **一行不改**——这才是「真 Matrix-Free」的含义：不是文件里没有稠密矩阵，而是**算法本身不消费稠密矩阵**。方法报告 §8 第 5 条也强调偏差核算需要 oracle，大体系上会退化为近似 oracle，但结构不变。

---

## 2. 导入与环境（L19–54）

| 导入 | 来源 | 作用 | 对应方法报告 |
|---|---|---|---|
| `create_single_machine_gauge_fixed` | [NES_VMC_V1.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/NES_VMC_V1.py#L1674-L1693) | 把单列网络归一为 $\psi_j(x_{ref})=1$ 的 machine（输出 $\log\psi$） | §4.1 gauge 第一层 |
| `NESFermionHopRule` | NES_VMC_V1.py L1233 | 自定义 Metropolis 跃迁规则：副本内 Hop + 副本去重 | §1.2 walker 结构 |
| `build_full_stage` / `amplitudes_full` / `extract_physical_states` | NES_VMC_deflation.py | 全空间 staging / 16×K 振幅 / 子空间本征提取——**仅诊断** | §7 循环外运算 |
| `load_and_orthonormalize_frozen` / `deflation_oracle` | NES_VMC_deflation_real.py | npz→正交归一 $F$；$B^\dagger H B$ 压缩谱 $\nu$ | §6.1、§2.3 |
| `make_mf_bundle` / `update_scales` | NES_VMC_deflation_mf.py | **A' 训练引擎本体** | §6.1 的 MC 化 |
| `SINGLE_SIZE, ha, hi_ext, ext_edges, K, hi, Hatree_Fock, single_edges` | [H2_6_31G_8Qbit_K2.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/H2_6_31G_8Qbit_K2.py) | 体系定义（见 §2.1） | §0.2、§1.1 |
| `jax_enable_x64`（L53） | — | float64 硬要求 | §3.4（0917 事故教训） |

L54 断言 `K==2 and SINGLE_SIZE==8`：脚本按 H2/6-31G「冻结 4 + 扩展 2」写死。

### 2.1 体系模块 `H2_6_31G_8Qbit_K2.py` 提供的对象

- `mol/mf/fci`：pyscf 建 H2（键长 1.5 Å）、6-31G 基组、RHF 与 FCI（`nroots=12`）基准——FCI 谱即金标准（§0.2）。
- `ha = nkx.operator.from_pyscf_molecule(mol)`：二次量子化哈密顿量（§1.1），以**局域连接**接口暴露（`get_conn_padded`），是 matrix-free 的根基。
- `hi = SpinOrbitalFermions(n_orbitals=4, s=1/2, n_fermions_per_spin=(1,1))`：8 自旋轨道、每自旋 1 电子 → $D=16$。
- `Hatree_Fock = hi.all_states()[0]`：gauge 锚点参考组态（注意：netket 枚举序下它未必是物理 HF 行列式，但作为**固定参考点**足够——gauge 固定只要求所有网络用同一个锚，Stage0/A/A' 全链一致，保真度断言 L159 会兜底验证一致性）。
- `hi_ext = hi ** K`：walker 希尔伯特空间（$K\times8=16$ bit，§1.2）。
- `single_edges`：α 块 {0..3}、β 块 {4..7} 内的轨道对——Hop 移动的候选边（保粒子数）。
- `ext_edges`：把 `single_edges` 按副本偏移 $k\times8$ 复制到拼接串上，实现「副本内跳、副本间不交换」。

---

## 3. Ansatz 两个类（L57–86）—— 方法报告 §1.2 的代码化

### `SingleStateAnsatz_SharedBackbone`（L57–72）

一条「列函数」$\psi_j$：`backbone1/backbone2` 两层 tanh 隐层（`n_so=8 → hidden → hidden`），`head_u/head_v` 两个实值输出头。前向：

```python
z = (u + 1e-12) + 1j * (v + 1e-12)
return jnp.log(z)
```

- **实参数、复输出** $\log\psi_j = u_j + i v_j$（§1.2）；`1e-12` 偏移防 $u=v=0$ 时 `log(0)`。
- `param_dtype=float` 显式声明参数为实数——后续 Wirtinger 拆分（§3.1）的前提。
- 实例化时隐层宽度：旧 K4 网络 `hidden_dim=SINGLE_SIZE+4=12`（与基线 8→12→12→1 一致，L147），active K=2 网络 `hidden_dim=SINGLE_SIZE+K=10`（L187）。

### `NESTotalAnsatz_stable`（L75–86）

$K$ 条独立单态网络的容器（`nnx.List`），每条用分裂出的子 key 独立初始化。注意它只是**参数容器**：行列式组装不在这里发生，而在 bundle 的 `_one_log`（§9.3）。`nnx.split(total_ansatz)[1]` 取出纯参数 pytree（L190），供 jax 函数以 `params` 形式消费——这也是「冻结参数是闭包常量、不进梯度追踪」（§3.5 机制一）的实现方式。

---

## 4. 超参区块（L110–131）—— 方法报告 §3.3/§3.4/§5 的数值化

| 变量 | 值 | 含义 | 报告出处 |
|---|---|---|---|
| `N_CHAINS×N_SAMPLES_PER_CHAIN` | 16×200=3200 | 主采样批大小 | §2.1 |
| `SWEEP_SIZE` | 20 | 每链扫描数 | — |
| `N_ITER` | 1500（env 可覆盖） | 训练步数 | §6.1 |
| `lr / qgt_diag_shift / clip_norm / per_param_clip` | 0.1 / 0.1 / 20 / 0.5 | SR 更新与数值防线 | §3.4 逐字对应 |
| `AUX_M / AUX_CHAINS / AUX_SWEEP` | 1024 / 8 / 20 | 辅助 MCMC 批（估 Gram 用） | A' 特有 |
| `EPS_MIX` | 0.25 | 参考密度 $q$ 中 active 混合系数 $\varepsilon$ | §9.4 |
| `SCALE_BETA` | 0.9 | 尺度 EMA 系数 | §9.6 |
| `MAX_INIT_COND` | 1e6 | 初始 $\mathrm{cond}(\chi)$ 上限 | §5 工程纪律 |
| `NPZ_PATH / PKL_PATH` | 冻结 K4 产物 | 26-10-07-19-57 的 300 步 Stage0 成果 | §5 |

`NPZ_PATH` 提供 $\varphi_{0..3}$ 向量与 `v_rot`；`PKL_PATH` 提供旧网络最终参数 `params[-1]`——「冻结 = 参数闭包常量 + 列约定 $\varphi_k=\sum_j\psi_j^{old}v_{rot}[j,k]$」的两块原料（§5）。

---

## 5. Step 0：冻结重建 + oracle（L133–172）—— 全部是启动诊断

```python
H_dense, all_states = build_full_stage(ha, hi)
```

`ha.to_dense()` 物化 16×16 稠密阵 + 枚举 16 个组态。**用途见 §1.2 白名单，此处不进入任何梯度/采样函数**。

- L136 `load_and_orthonormalize_frozen(NPZ_PATH, hi)`（[NES_VMC_deflation_real.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/NES_VMC/experiments/H2分子/NES_VMC_deflation_real.py#L38-L79)）：读 npz 的 `phi_vecs` (16,4)，先断言 `all_states` 基矢序与 `hi.all_states()` 逐行一致（防「投影张冠李戴」），再做升序 Gram-Schmidt 得正交归一 $F$ (16,4)，并记录每步残差范数（>0.1 断言，防近退化冻结空间）。返回 `phi_raw`（列归一原始向量，供重叠诊断）。对应报告 §6.1 第一条。
- L137 `deflation_oracle(F, H_dense, n_keep=2)`（NES_VMC_deflation_real.py L85–106）：对 $F$ 做完整 QR 取正交补基 $B$（16×12），压缩 $H_b=B^\dagger H B$ 对角化，返回最低两本征值 $\nu$、对应本征矢、全谱。docstring 特别警告：不能直接对 16×16 的 $QHQ$ 求谱——$\mathrm{span}(F)$ 贡献的精确零模会混进最低本征值（本项目目标能级为正时必错）。$\nu$ 即 Loss 理论目标的可核算部分（§2.3）。
- L138 `E_exact, U_exact = np.linalg.eigh(H_dense)`：FCI 金标准谱与本征矢（验收对照用）。
- L140–151 **冻结列的逐点表示**：读 `v_rot`、旧参数；重建 4 条 gauge-fixed 旧 machine（`frozen_sms`）与参数 list（`frozen_params`）。注意 `old_ansatz` 在这里只是**结构模板**（提供 graphdef），真正前向靠 `frozen_sms[j](frozen_params[j], sigma)`。这 4 列从不出现在 `jax.grad` 追踪里 ⇒ 天然 stop-gradient（§3.5 机制一）。
- L154–159 **保真度断言**：`A_old = np.exp(...)` 把旧 machine 输出（$\log\psi$）取指数成振幅矩阵（16×4，memory 里记录的坑），`Phi_net = A_old @ v_rot` 按列约定合成 $\varphi_0..\varphi_3$，逐列归一后与 `phi_raw` 算保真度，断言 >0.999——gauge 约定接错的最后一道闸（§4.3）。
- L161–162 `Phi_raw_all` / `Gphi_ex`：未缩放的原始冻结基与其 Gram，仅供第 5 段监控算精确 $c$ 参照。

---

## 6. fresh active 初始化 + 投影可逆性预检（L174–197）

```python
def _chi_full_np(params):
    A = np.asarray(amplitudes_full(sms, params, all_states_j))...
    c = F.conj().T @ A
    return A, c, A - F @ c
```

这是**精确版方案 A 的投影公式**（$\chi=A-FF^\dagger A$，§6.1），在 A' 里只用于两处诊断：本节的 seed 预检和第 5 段监控。while 循环反复 fresh 初始化 active 网络（L185–190），对每个 seed 算全空间 $\chi$ 的条件数，直到 $\mathrm{cond}(\chi)<10^6$（L191–196）。动机是 §5 的工程纪律：active 列若恰好落入冻结 span 的低维残余，行列式奇异/病态；小系统上直接全空间验一遍最稳妥。真实大体系上这一步会换成 B' 式的采样点 Gram 检查（§6.2 的 `cond(B_6列)` 预检思路）。

---

## 7. bundle 构造与双采样器（L199–239）

- L200 `make_mf_bundle(ha, frozen_sms, frozen_params, v_rot, sms, n_spin=8, eps=0.25)`：把训练引擎全部函数装进 dict（内部见 §9）。**关键**：传入的是 `ha`（稀疏算符），不是 `H_dense`——bundle 内部只走 `get_conn_padded` 路径，稠密阵在类型上就进不了更新路径。
- L203–204 `f=0, g=zeros(2)`：尺度变量初值（§9.6）。
- L207–208 `aux_machine`：辅助链的 machine = `bundle["aux_logq"]`，参数 pytree `(total_params, f, g)`。
- L211–213 `sample_machine`：主链 machine = `bundle["machine_batch"]`，参数 pytree `(p, c, c2, f, g)`——注意 `sigma.reshape(-1, hi_ext.size)` 把 walker 摊平成 16 bit。
- L216–219 `aux_sampler`：作用在 `hi`（单组态 8 bit）、`K=1` 的 Hop 规则、`machine_pow=1`。因 `log q` 的指数 $q>0$，pow=1 即按 $q/Z$ 精确采样——辅助链不需要重要性加权（它估的是 $1/q$ 加权 Gram，定义里就带着它）。
- L220–223 `nes_sampler`：作用在 `hi_ext`（16 bit 联合 walker）、`ext_edges`、`machine_pow=1`——按 $|\Psi_F|$ 而非 $|\Psi_F|^2$ 采样，防零方差冻结（§3.3；梯度端补 $w\propto|\Psi_F|$）。
- L225–229 `optimizer`：`clip_by_global_norm(20)` + `sgd(0.1)`。注意 SR 的「自然梯度」求解在循环里手工做（`solve(S, g)`），optax 链只负责裁剪 + 步长——与 §3.4 公式 $\theta\leftarrow\theta-\mathrm{clip}(S^{-1}g)\times0.1$ 一致。
- L231–235 aux 预热：正式循环前先跑一整批（128/链）辅助采样，让链脱离 burn-in，首步 Gram 不拖尾偏。

---

## 8. 训练循环五段（L261–411）—— 逐段对应

### 第 1 段（L264–284）：aux 采样 → MC Gram

`s_aux` 取 8 链×128=1024 个单组态；`bundle["estimate_grams"](total_params, sigma_aux, f, g)` 返回 `(c_mc, c2_mc, max_phi2, max_psi, aux_g)`。`grams_finite` 不成立 ⇒ 整步跳过（记 NaN 占位，保持历史数组对齐）。

**这一步取代了方案 A 的 $c=F^\dagger A$**——A' 的全部新数学都在 `estimate_grams` 里（§9.4），是「唯一被替换的数学对象」。

### 第 2 段（L286–295）：EMA 尺度更新 + $c$ 同步变换

`update_scales(f, g, max_phi2, max_psi, beta)` 把 $\tilde\varphi=\varphi e^{-f}$、$\tilde\psi_m=\psi_m e^{-g_m}$ 的尺度钉在 $O(1)$（首步 beta=1 fresh）。尺度一变，上一批用旧尺度算出的 $c,c2$ 必须做 $c[:,m]\leftarrow c[:,m]e^{\Delta f-\Delta g_m}$（L291–293 `trans`）才能保持 $\tilde\chi=\tilde\Psi-\tilde\Phi c$ 方向不变、$E_L$ 严格不变（恒等变形见配套解读 §3）。这是防 0917/方案A「冻结分量漂移 → max|A| 爆到 1e3」的显式机制。

### 第 3 段（L297–310）：主采样

联合 2-walker、16 链×200 步，machine 按 `slogdet(Ψ̃−Φ̃c)` 给 $\log\Psi_F$。`c_cur/c2_cur/f/g` 作为**显式参数**传入——若走闭包会被 `jax.jit` 焊死成常量、每步换值触发重编译（A' 实现的关键约束）。

### 第 4 段（L312–351）：梯度 → 裁剪 → SR → 更新

- `grad_fn` 返回力梯度、`loss_mean`（= Re tr$\langle E_L\rangle$，监控值与 $\mathcal{L}^\ast$ 对照）、`E_L_mean` (2×2)、`aux`（valid_ratio/ESS/should_skip）。公式 = §3.1–3.3 的矩阵推广（逐函数见 §9.5）。
- L323–330：per-param 裁剪（阈值 0.5，按范数等比缩放）；L332–334：NaN/‖g‖>5000 告警（只告警不拦截，拦截靠 should_skip）。
- L337–339：`should_skip`（valid_ratio < 0.25 或非有限）⇒ 跳过更新。
- L341–348：`qgt_fn` 组 $\mathrm{Re}(S_w)+0.1I$（§3.4），`jnp.linalg.solve` 出自然梯度，再过 optax（全局裁剪 20 → lr 0.1）。**float64 下 solve 是安全的**——这正是 0917 float32 事故的修复线。

### 第 5 段（L353–402）：全空间监控（旁路，不进回路）

- `_chi_full_np(total_params)`：精确重投影（16 维）。
- `c_mc_raw = c_cur·e^{g−f}`：把缩放基下的 MC 系数换回原始基；`c_ref_raw = solve(Gphi_ex, Φ†A)`：原始基下的精确系数（$F$ 正交归一时即 $F^\dagger A$）。`c_err` = 两者最大差——**MC Gram 正确性的逐步审计线**（注意其绝对值随 max|A| 漂移而涨，看相对值，见 §13）。
- `lam_sub`：在 $\chi$ 列张成的 2 维子空间上解广义本征 $(M_\chi+10^{-14}S_\chi)v=\lambda S_\chi v$——**block 分析**，只对 active 块，绝不碰完整 6×6（§2.2 冻结纪律）。
- `frozen_res = \max|F^\dagger\chi_n|$：冻结剩余重叠，自动/显式投影严格性的数值证据。
- `max_absA>1e3` 漂移告警；`aux_g` 的 `w_ess/max_w_ratio` 监控辅助链权重退化。
- L372–382 日志三段（Loss/λ、诊断、梯度）；L404–411 每 20 步落盘 history。

---

## 9. 训练引擎 `NES_VMC_deflation_mf.py` 逐函数

### 9.1 `_cols(p, sigma, f, g)`（L69–75）：逐点列求值

输入 `sigma` (n,8)（任意 n：1 个 walker 的行、或 aux 批）。输出缩放后的冻结列 $\tilde\Phi$ (n,4) 与 active 列 $\tilde\Psi$ (n,2)：

```python
Phi = (jnp.exp(Lf) @ V) * jnp.exp(-f)   # φ_i = Σ_j ψ_j^old v_rot[j,i]，再乘 e^{-f}
Psi = jnp.exp(La - g[None, :])          # ψ̃_m = ψ_m e^{-g_m}
```

`Lf/La` 是 gauge-fixed machine 的 $\log\psi$ 输出——**必须取 exp**（§4.3 的坑）。全程逐点，$n$ 可以是任何数，没有 16 维数组。

### 9.2 `_h_cols(p, sigma, f, g)`（L77–87）：$H$ 作用的逐点化

active 列走 `Ham_psi_scaled(ha, sms[m], ..., shift=g[m])`；冻结列先对**旧网络逐列**算 `Ham_psi_scaled(..., shift=f)`，再右乘 `V` 合成 $H\varphi_i=\sum_j v_{rot}[j,i]\,H\psi_j^{old}$——$H$ 线性把「冻结列的 H 作用」化为旧网络的逐点求值（§5 第三条）。`Ham_psi_scaled`（NES_VMC_V1.py L622–681）内部是 `ha.get_conn_padded(x)` 取 ≤13 个连接、$\sum_{x'}H[x,x']e^{\log\psi(x')-\mathrm{shift}}$——**稀疏局域，从不物化 H**（§1.1、§7 审计表第三行）。

### 9.3 `_one_log` / `_one_full`（L89–108）：单 walker 前向

`xw` (K,8) → `_cols` 得每副本每列的振幅 → $A_\chi=\tilde\Psi-\tilde\Phi c$ (2×2)：

- `_one_log`：`slogdet(A_χ)` → $\log\Psi_F$（采样器只要这个）。
- `_one_full`：再算 $Z=(H\tilde\Psi-H\tilde\Phi c)-\tilde\Phi c_2$（**project→H→project** 的 MC 版，对应精确 bundle 的 $\chi\to H\chi\to QH\chi$），$E_L=A_\chi^{-1}Z$ (2×2)（§2.1），并给 `valid` 有限性标志。

行列式结构 $\det[\tilde\Phi|\tilde\Psi]=\det[\tilde\Phi|Q_{\tilde\Phi}\tilde\Psi]$（§6.2 恒等式）在这里的表现是：**投影显式写进 $A_\chi$，但投影系数 $c$ 由 MC 给出而非全空间内积**——这是 A' 与 B'（完全不写投影、纯靠 det 列变换）的中间路线。

### 9.4 `aux_logq` / `estimate_grams`（L111–149）：Z 抵消引理的实现

`aux_logq`：$\log q(y)=\log\big(\sum_i|\tilde\varphi_i|^2+\varepsilon\sum_m|\tilde\psi_m|^2\big)$，即参考密度（未归一）。

`estimate_grams`（jit）核心四行（L131–137）：

```python
wi = 1.0 / (q * M)
Gm = conj(Phi).T @ (wi[:,None] * Phi) + ridge*I     # Ĝ = ⟨Φ̃|Φ̃⟩/Z
Mm = conj(Phi).T @ (wi[:,None] * Psi)               # M̂ = ⟨Φ̃|Ψ̃⟩/Z
HG = conj(Phi).T @ (wi[:,None] * Hphi)              # ĤGG = ⟨Φ̃|HΦ̃⟩/Z
M2 = conj(Phi).T @ (wi[:,None] * Hpsi)              # M̂2 = ⟨Φ̃|HΨ̃⟩/Z
c  = solve(Gm, Mm);  c2 = solve(Gm, M2 - HG @ c)
```

逐样本估计 $\frac1M\sum_k \tilde\varphi_i^*(y_k)\tilde\psi_j(y_k)/q(y_k)\to \frac1Z\langle\tilde\varphi_i|\tilde\psi_j\rangle$：$1/q$ 恰好抵消未归一 $q$ 带来的 $Z$，所以 $c=\hat G^{-1}\hat M$ 里 $Z$ 严格消去、**不需要配分函数**（配套解读 §2）。$c_2$ 是二次投影系数。`gram_ridge=1e-12` 防 $\hat G$ 奇异。`aux` 返回 `w_ess`（$1/\sum u^2$，辅助链有效样本数）与 `max_w_ratio`（权重峰值预警）；`max_phi2/max_psi` 是批 max（stop-gradient），供尺度更新。

### 9.5 `grad_fn`（L171–215）：力梯度 = §3.1–3.3 的逐项落地

- `loss = Re tr(E_L)` 逐 walker；`wv = importance_weights_from_logpsi(log_psi, valid)`（NES_VMC_deflation_real.py L119–127：$w\propto|\Psi_F|$、valid 上归一、stop-gradient）。
- $E_L^{mean}=\sum_n w_n E_{L,n}$（矩阵加权均值，§2.1 的 $\langle E_L\rangle_w$）。
- `tr_centered = tr(E_L − E_L_mean)`，梯度 `grad = 2·Re Σ_n w_n · tr_centered_n · conj(∇logΨ_F,n)`（§3.2 矩阵中心化力公式）。
- $\nabla\log\Psi_F$ 由 `_dlogpsi`（L166–169）给：两次实值 `jax.grad` 拼 $\nabla\mathrm{Re}+i\nabla\mathrm{Im}$（Wirtinger，§3.1）。
- `aux.should_skip`：梯度/loss 非有限或 `valid_ratio<0.25` 时置真（§3.4 数值防线之一）。
- 全程 $c,c2,f,g$ 是显式入参、对 `p` 求导时视为常量 ⇒ **梯度只流向 active 列**（§3.5）。

### 9.6 `qgt_fn`（L217–232）与 `update_scales`（L248–257）

`qgt_fn`：vmap 出逐样本 $\nabla\log\Psi_F$ → `flatten_batched_pytree` 拼 (N,P) 复矩阵 $O$ → 按 $w$ 加权中心化 → $S=(\overline{O_c}\,w)^T O_c$ → 厄米化取 $\mathrm{Re}(S)+\eta I$。$\mathrm{Re}(S)$ 而非复 $S$ 是 0925 修复线（§3.2 要点②）。

`update_scales`（numpy 侧、jit 外）：$f\mathrel{+}=\frac{\beta}{2}\ln\max_y\sum_i|\tilde\varphi_i|^2$、$g_m\mathrel{+}=\beta\ln\max_y|\tilde\psi_m|$——目标是缩放后批 max ≈1，使 $q$ 两项同量级、$A_\chi$ 元素 $O(1)$。

### 9.7 返回 dict（L234–245）

`aux_logq / estimate_grams / machine_single / machine_batch / forward / grad_fn / qgt_fn / cols / K / r`——脚本第 7、8 节消费的就是这些句柄；`machine_single/forward` 主要供单测与诊断复用。

---

## 10. 复用工具箱函数（精确版遗留，全部只服务诊断/预检）

| 函数 | 位置 | 做什么 | 在 A' 中的用途 | 报告出处 |
|---|---|---|---|---|
| `build_full_stage` | NES_VMC_deflation.py L59 | `ha.to_dense()` + `all_states` | 启动 staging | §7（循环外） |
| `amplitudes_full` | 同 L94 | 16 组态上逐列求 $\psi_j$ → (16,K) | seed 预检、每步监控、验收 | §7 |
| `extract_physical_states` | 同 L364 | $\chi$ 子空间广义本征 → 归一物理态 | 收尾提取 φ̂₄/φ̂₅ | §2.2、§6.2 |
| `load_and_orthonormalize_frozen` | deflation_real L38 | npz→$F$ + 基矢序/残差断言 | 冻结空间表示（oracle/监控参照） | §6.1 |
| `deflation_oracle` | deflation_real L85 | $B^\dagger H B$ 谱 → $\nu$ | Loss 目标、方法偏差下限 | §2.3 |
| `create_single_machine_gauge_fixed` | V1 L1674 | machine 输出 $\log\psi(\sigma)-\log\psi(ref)$ | 冻结重建 + active 前向 | §4.1 |
| `NESFermionHopRule` | V1 L1233 | 副本内 Hop（保粒子数）+ 副本重复拒绝；对称提案无 KK 修正 | 主/aux 两采样器 | §1.2 |
| `Ham_psi_scaled` | V1 L622 | $e^{-s}H\psi$ 经 `get_conn_padded` | 前向 $Z$、$H\Phi/H\Psi$（**在更新路径上，matrix-free**） | §1.1、§7 |
| `importance_weights_from_logpsi` | deflation_real L119 | $w\propto|\Psi_F|$ 归一 | grad/qgt 加权 | §3.3 |

注意区分：`NES_VMC_deflation.make_deflation_bundle`（精确版 bundle，每步接触 16 维）在 A' 中**没有被使用**——A' 用的是 `make_mf_bundle`；只有它的三个纯 staging 函数被借来当诊断参照。

---

## 11. 提取、验收、保存、绘图（L416–532）

- L417–418：终态 `_chi_full_np` → `extract_physical_states(chi, H_dense)` 得 $\hat\varphi_4,\hat\varphi_5$（block 广义本征，§2.2 纪律）。
- L421–435 验收表：每能级报 $\lambda_{\mathrm{Rayleigh}}$、$\nu$（oracle）、$E^{FCI}$、误差 mHa、$\mathrm{Var}[E_{loc}]=\langle H^2\rangle-E^2$、FCI 重叠 $|\langle U_{exact,4+k}|\phi\rangle|^2$、对冻结 span 的最大剩余重叠 $|\phi_{raw}^\dagger\phi|^2$。
- L438–440 验收线：误差<2 mHa ∧ 重叠>0.99 ∧ Var<1e-4 ∧ 冻结剩余<1e-3 ∧ 耗时<500 s。实测（ts=26-10-16-02-23）前四项全过、608 s 超耗时线 ⇒ 脚本报 FAIL，但结论判定为成功（比 B' 快 1.73×，见验证报告）。
- L443–469 保存：npz（φ̂ 向量、`chi_full`、active 块旋转 `v_rot=v2`、$\hat\lambda$、$\nu$、$E^{FCI}$、Var、重叠、cond、seed、耗时）+ history pkl（含 `nu_oracle` 与尺度终值）。
- L471–532 六联图：λ 收敛（含 $\nu$/$E^{FCI}$ 参考线）、Loss（目标线 $\sum\nu$）、能级误差（symlog + 化学精度带）、采样/投影健康（valid/ESS/aux w_ess/冻结剩余）、`c_err`、梯度与 max|A| 漂移。全部采用「有限值掩码后连线」的两段式画法——稀疏诊断序列直接 plot 会因相邻 NaN 断线（§10 已修坑）。

---

## 12. 代码 ↔ 方法报告章节速查

| 方法报告 | 代码落点 |
|---|---|
| §1.1 局域连接 H | `Ham_psi_scaled` ← `ha.get_conn_padded` |
| §1.2 K 列行列式 | `_one_log` 的 `slogdet(Psi − Phi@c)` |
| §2.1 $E_L=A^{-1}Z$、Loss=Re tr | `_one_full` + `grad_fn.loss` |
| §2.2 block 本征提取 | `extract_physical_states`、监控 `lam_sub` |
| §2.3 $\mathcal{L}^\ast=\sum E_{ray}+\nu$ | `deflation_oracle` → 日志目标线、图 1 参考线 |
| §3.1 Wirtinger | `_dlogpsi` 两次实值 grad |
| §3.2 力梯度 + Re(S) | `grad_fn` 的 `2*real(...)`、`qgt_fn` 的 `real(S)` |
| §3.3 machine_pow=1 + w∝\|Ψ\| | `nes_sampler` + `importance_weights_from_logpsi` |
| §3.4 SR/裁剪/float64 | `qgt_fn` diag_shift、optax 链、L53 x64 |
| §3.5 梯度只流向 active | 冻结参数 list 闭包 + c/c2/f/g 显式入参 |
| §4.1 gauge 三层 | `create_single_machine_gauge_fixed` + `f/g` 尺度管理 |
| §5 冻结 = pkl+npz 合成常量列 | L140–159；fresh 初始化纪律 L184–197 |
| §6.1 方案 A 显式投影 | `_chi_full_np`（降级为诊断参照） |
| §7 Matrix-Free 审计 | 本文 §1.2 白名单表 |
| §8 正交约束零成本/增量谱学 | `estimate_grams` 替代 Gram 求逆；冻结继承 K4 |

---

## 13. 已知现象与阅读脚本时的注意点

1. **`c_err` 绝对值「爆炸」是假象**：它随 max|A| 漂移同步涨（末段 ~2.9），但相对误差稳定 ~1e-4；看 `c_err/max|A|`。漂移本身是「$\psi+\sum\alpha\varphi$ 对 $\Psi_F$ 不可见」的规范方向（方案 A 同款），float64 下无害。
2. **aux 链健康度要盯**：`aux_ess` 长期 20–50/1024、`aux_maxw~140` 本轮未构成危害，但外推大体系需复查（缓解：升 `AUX_M` 或 $\varepsilon$ 自适应）。
3. **TIMING 分量不可用于成本归因**：`jax.block_until_ready(0)` 不阻塞实际异步计算，grad/qgt 显示值是派发计时；以总墙钟为准。
4. **验收 PASS/FAIL 看耗时线**：500 s 线是针对「比 B' 快」的预设目标，未达不否定算法正确性（质量四项全过）。
5. **`build_full_stage` 的命名误导**：它来自精确版工具箱（方案 A 的核心 staging），A' 借它纯粹因为「小系统存得下稠密 H 是验证特权」。判断某段代码是否破坏 matrix-free 主张的准绳只有一条：**它的输出是否流入 `sample_machine` / `grad_fn` / `qgt_fn` / `optimizer.update`**。本脚本五处全空间运算无一流入。
