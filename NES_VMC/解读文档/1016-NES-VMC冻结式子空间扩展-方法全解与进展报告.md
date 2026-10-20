# NES-VMC 冻结式子空间扩展：方法全解与进展报告

- 日期：2026-10-16
- 主题：从二次量子化定义到「$K=6$ = 冻结 $K=4$ + 扩展 $K=2$」的完整算法链条，含真 Matrix-Free 实现
- 关联脚本：
  - K=4 基线训练：`H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.py`（300 步产物即冻结源）
  - 精确版扩展（方案 A）：`H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py` + 工具箱 `NES_VMC_deflation_real.py`
  - Matrix-Free 版扩展（方案 B'）：`H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py`
- 关联解读：`1015-冻结K4剩余子空间K2扩展-实现解读.md`、`1015-真MatrixFree冻结块K6-行列式自动投影-实现解读.md`

---

# 第一部分 背景

## 0.1 问题设定

目标是用电神经网络（NN）变分蒙特卡洛（VMC）求解分子电子结构的**基态与低激发态谱**。传统 VMC 一次训练只优化一个态；激发态要么逐态加正交约束（penalty/Gram，数值脆弱），要么同时优化多个态（收敛难、能级排序漂移）。

NES-VMC（Neural Encoded States VMC）的核心思想：**用一个 $K$ 列行列式拟设同时表示 $K$ 个电子态**，一次联合采样 + 一次广义本征提取，同时得到 $K$ 个近似本征态 $\varphi_0,\dots,\varphi_{K-1}$ 与 $K$ 个能级 $\lambda_0,\dots,\lambda_{K-1}$。

本项目已完成的验证链条（时间序）：

| 里程碑 | 内容 | 结论 |
|---|---|---|
| 09-16 | 投影哈密顿量方案（$H_\perp$ 显式矩阵）H2/6-31G 全谱验证 | VMC 可精确收敛到第一激发态；激发态需复参数/复输出 |
| 09-23 | 矩阵自由两阶段 deflation（Stage0 收敛 $\varphi_0$ → 冻结 → Stage1 收敛 $\varphi_1$） | 冻结流形可行 |
| 09-25 | 实参数复输出 $\log\psi$ 的三处修复（Wirtinger 梯度、$\mathrm{Re}(S)$ 度量、$\vert\psi\vert$ 采样+重要性加权） | 解决「滑向 HF 零方差冻结」 |
| 10-01 | Route B 冻结流形（$K=4$，1 冻结 + 3 active）对照实验 | 冻结保证「不变差」成立 |
| 10-07 | K=4 RealFFNN 300 步训练产物（本报告的冻结源） | $\varphi_0..\varphi_3$ Rayleigh 误差进入 mHa 量级 |
| 10-15 | **方案 A**：冻结 K=4 + 剩余子空间 K=2（显式 16 维投影，精确版） | $\hat\varphi_4/\hat\varphi_5$ 误差 0.0005/0.018 mHa |
| 10-15 | **方案 B'**：同任务的**真 Matrix-Free** 版（$6\times6$ 行列式自动投影） | 误差 0.0061/0.018 mHa，精度持平 |

## 0.2 物理系统

H2 分子，6-31G 基组 → 4 个空间轨道 × 2 自旋 = **8 个自旋轨道**；2 个电子（1 $\alpha$ + 1 $\beta$）→ 物理 Fock 空间维数 $D=\binom{4}{1}\times\binom{4}{1}=16$。8 个自旋轨道占据数即 8 bit 组态串，这就是「8 量子比特」映射（Jordan-Wigner 后为 qubit 哈密顿量，netket 中由 `SpinOrbitalFermions` + `FermionOperator2nd` 直接处理费米子表示）。

$D=16$ 小到可以整稠密对角化（FCI 精确解），这使本项目拥有**金标准**：一切近似（冻结偏差、训练误差）都能被逐项拆开验证——这是方法论研究的理想试车场，也是「精确版方案 A」得以存在的原因。但真正的算法目标（见 §7、§8）不依赖 $D$ 小。

---

# 第二部分 NES-VMC 方法全解

## 1. 二次量子化下的定义

### 1.1 哈密顿量

分子电子结构在自旋轨道 $\{a_p^\dagger,a_p\}$ 下的标准二次量子化形式：

$$
H=\sum_{pq}h_{pq}\,a_p^\dagger a_q+\frac{1}{2}\sum_{pqrs}(pq|rs)\,a_p^\dagger a_q^\dagger a_s a_r
$$

$h_{pq}$ 为单电子积分，$(pq|rs)$ 为双电子积分（pyscf 生成）。netket 将其表示为 `SpinOrbitalFermions(n_spin_orbitals=8, num_fermions=1, num_spin_up=1)` 希尔伯特空间上的 `LocalOperator`，关键能力是**只暴露局域连接**：对任意组态 $x$，`ha.get_conn(x)` 返回 $\{x'\}$ 与矩阵元 $H[x,x']$（H2/6-31G 下每组态 $\le 13$ 个非零连接）。这是后文一切 matrix-free 性质的根基——算法从接口上就不需要 $H$ 的整张矩阵。

### 1.2 K 列行列式拟设（NES 结构）

拟 $K$ 个「单电子态样」的列函数 $\psi_j\;(j=0,\dots,K-1)$，每个是一条**实参数、复输出**的 FFNN：

$$
\psi_j(x)=\exp\!\big(\log\psi_j(x)\big),\qquad
\log\psi_j(x)=u_j(x)+\mathrm{i}\,v_j(x),
$$

其中 $u_j,v_j$ 为 $8\to12\to12\to1$ 的 tanh-FFNN（两个实参数头）。

对 $K$ 个电子组态 $x_1,\dots,x_K$（一个「walker」= $K$ 份副本），总波函数定义为 Slater 型行列式：

$$
\Psi(x_1,\dots,x_K)=\det A,\qquad A_{nj}=\psi_j(x_n)\in\mathbb{C}\ \ (K\times K).
$$

walker 空间是 $K$ 份 8-bit 副本拼接（$K=4$ 时 32 bit，$K=6$ 时 48 bit），MCMC 在每份副本内部做 Hop 移动（把某轨道的占据跳到另一轨道，保粒子数），副本间不交换——因此 $\Psi$ 定义在「$K$ 个组态的有序组」上，$|\Psi|^2$ 是这 $K$ 个组态的联合分布。

### 1.3 为什么行列式能同时装下 K 个态

行列式拟设的变分空间包含所有形如

$$
\varphi_k=\sum_{j}v_{jk}\,\psi_j\qquad(\text{列的固定线性组合})
$$

的 $K$ 个态；当网络收敛到 $K$ 个（近似）本征态张成的空间时，存在唯一的旋转矩阵 $v_{\mathrm{rot}}$ 使 $\{\varphi_k\}$ 成为近似本征态组，对应的能量由 $K\times K$ 局域能量矩阵的本征值给出（§2.2）。**一次采样、一次本征提取 = $K$ 个能级 + $K$ 个态**，且行列式结构天然保证不同列张成的态互不相同（列线性相关时 $\det=0$，采样分布自动惩罚退化）。

## 2. 损失函数的定义

### 2.1 K×K 局域能量矩阵

对每个 walker，用同一组连接计算：

$$
A_{nj}=e^{-s}\psi_j(x_n),\qquad
Z_{nj}=e^{-s}(H\psi_j)(x_n)=\sum_{x'}H[x_n,x']\,e^{-s}\psi_j(x')\ \leftarrow\ \text{只用 get\_conn},
$$

$$
E_L=A^{-1}Z\in\mathbb{C}^{K\times K},
$$

其中 $s$ 为 per-walker 数值稳定化 shift。$E_L$ 是「矩阵值的局域能量」：若列函数恰为真本征态，$E_L$ 退化为对角常数阵。变分目标为

$$
\mathcal{L}=\mathrm{Re}\,\mathrm{tr}\,\langle E_L\rangle_w
$$

（$\langle\cdot\rangle_w$ 表示按采样分布的重要性加权期望）。可以证明 $\mathrm{Re}\,\mathrm{tr}\langle E_L\rangle$ 正是这 $K$ 个态能量期望之和

$$
\mathcal{L}=\sum_{k}\frac{\langle\varphi_k|H|\varphi_k\rangle}{\langle\varphi_k|\varphi_k\rangle}
$$

的蒙特卡洛估计——**损失函数就是 $K$ 个 Rayleigh 商之和**，最小化它同时把 $K$ 个列推向本征态。

训练监控中每步对 3200 样本的 $E_L$ 求均值得 $6\times6$ 矩阵 $\langle E_L\rangle_{\mathrm{mean}}$，其特征值即 $K$ 个能级的实时估计（matrix-free，只涉及 $6\times6$ 小矩阵）。

### 2.2 物理态与能级的提取（广义本征问题）

采样收敛后，在经验协方差意义下解 $K\times K$ 本征问题：

$$
\langle E_L\rangle\,v_k=\lambda_k\,S\,v_k,\qquad
S_{nm}=\langle\psi_n|\psi_m\rangle_{\mathrm{samples}},
$$

得 $(\lambda_k,\,v_{\mathrm{rot}}[:,k])$，物理近似本征态 $\varphi_k=\sum_j\psi_j\,v_{\mathrm{rot}}[j,k]$。
**冻结块扩展时的关键纪律**（Route B 教训）：提取 $\hat\varphi_4/\hat\varphi_5$ 时只对 active $2\times2$ 块做本征（block 分析），绝不重新对角化完整 $6\times6$——那会把冻结列 $\varphi_0..\varphi_3$ 重新旋转混合，破坏冻结。

### 2.3 冻结扩展后的损失目标（≠ ΣE_FCI！）

方案 A/B' 中，4 列冻结、2 列可训练，损失分解为

$$
\mathcal{L}=\sum_{i=0}^{3}E_{\mathrm{ray}}(\varphi_i)+\mathrm{tr}\,\langle E_L\rangle_{\mathrm{active\,block}},
$$

冻结部分是常量偏置。本项目实测 $E_{\mathrm{ray}}(\varphi_0..\varphi_3)=[-1.05355584,\,-0.95784918,\,-0.66900976,\,-0.55194186]$（与 FCI $E_0..E_3$ 差 0.8~0.5 mHa），因此 **Loss 的收敛目标是**

$$
\mathcal{L}^\ast=\sum_{i=0}^{3}E_{\mathrm{ray}}(\varphi_i)+\nu_4+\nu_5=-3.27489293,
$$

而不是 $\sum_k E_k^{\mathrm{FCI}}$。其中 $\nu$ 是压缩算子 $B^\dagger H B$（$B$ = $\mathrm{span}(\Phi)$ 正交补的标准正交基，$Q_\Phi=BB^\dagger=I-FF^\dagger$）的最低两个本征值，即 oracle：

$$
\nu=\big(-0.02803284,\,-0.01450344\big).
$$

$\nu$ 与真实 $E_4,E_5$ 的偏差是**冻结近似本征态导致的方法偏差下限**（本项目实测 $\nu-E^{\mathrm{FCI}}\approx-0.0001/\,-0.0000$ mHa，即 Stage0 冻结时刻精度已极高，方法偏差近乎消失）。训练收敛到 $\nu$ 附近即「方法饱和」；验收时必须先算 oracle，才能把误差拆成「方法偏差 + 训练误差」两项。

## 3. 梯度计算的方式

### 3.1 实参数复输出的 Wirtinger 拆分

列网络参数 $\theta\in\mathbb{R}^P$，$\log\Psi(\theta)$ 为复数。`jax.grad` 不接受复输出，拆两次实值梯度：

$$
\nabla_\theta\log\Psi=\nabla_\theta\mathrm{Re}\log\Psi+\mathrm{i}\,\nabla_\theta\mathrm{Im}\log\Psi.
$$

### 3.2 力梯度（含矩阵中心化）

$$
g=2\,\mathrm{Re}\,\Big\langle\mathrm{tr}\big(E_L-\langle E_L\rangle_w\big)\;\overline{\nabla_\theta\log\Psi}\Big\rangle_w .
$$

要点：① 实参数 VMC 的力必须取实部（$\nabla E=2\,\mathrm{Re}\,\langle(E_{\mathrm{loc}}-\langle E\rangle)\overline{\nabla\log\psi}\rangle$ 的矩阵推广）；② 度量与期望必须同用 $\mathrm{Re}(S)$，不能对复数 $S$ solve 后丢虚部（$\mathrm{Re}(S^{-1}g)\neq\mathrm{Re}(S)^{-1}\mathrm{Re}(g)$）——这两条是 09-25 验证过的修复。

### 3.3 重要性加权（防零方差冻结）

采样用 `machine_pow=1`（按 $|\Psi|$ 而非 $|\Psi|^2$ 抽样），梯度端乘以 $w\propto|\Psi|$ 归一化权重。原因：按 $|\psi|^2$ 采样时，轨迹一旦滑向 HF 行列式附近，稀有双激发组态出现频率 $\approx 0$，其梯度信号 $\propto\psi\,\eta^2$ 消失，训练「冻死」在 HF 盆地（09-25 与 10-06 两次独立踩坑）。按 $|\psi|$ 采样 + 加权后稀有组态贡献恢复 $\propto\psi\,\eta$，加权期望精确还原 $\langle\cdot\rangle_{|\psi|^2}$ 公式。

### 3.4 SR（自然梯度）与数值防线

$$
S_w=\mathrm{Re}\,\mathrm{Cov}_w\!\big(\nabla_\theta\log\Psi\big)+\eta I,\qquad \eta=\texttt{diag\_shift}=0.1,\ \ \mathbb{R}^{P\times P},\ \texttt{float64},
$$

$$
\delta\theta=S_w^{-1}g\ \xrightarrow{\ \texttt{clip\_by\_global\_norm}(20)\ }\ \xrightarrow{\ \text{per-param clip }0.5\ }\ \theta\leftarrow\theta-0.1\,\delta\theta .
$$

float64 是硬要求：09-17 的 K4 梯度爆炸事故中，float32 下 $\mathrm{cond}(S)\ge 5\times10^4$ 出现负特征值，一步更新撬动 $\log\psi$ 尺度 $-16.6$。本次 B' 全程 0 次 should_skip、0 次裁剪触发。

### 3.5 梯度只流向 active 列（两条独立机制）

- **冻结参数是闭包常量**：Python list 持有的旧网络参数从不进入任何 `jax.grad` 追踪 ⇒ 天然 stop-gradient。
- **行列式自动投影给出切空间**：$\nabla_\theta\det[\Phi|C]=\nabla_\theta\det[\Phi|Q_\Phi C]$，autodiff 过 `slogdet` 自动得到 $\partial\chi=Q_\Phi\,\partial A$ 的投影梯度，无需手工修正项。

## 4. Gauge 问题与解决

NES 行列式拟设有两层规范自由度，处理不好就是 09-17 事故的根源：

### 4.1 列的绝对尺度（每列一个复常数）

$\psi_j\to c_j\psi_j\ (c_j\in\mathbb{C})$ 被 det 吸收（$A$ 每列乘 $c_j$ ⇒ $\Psi$ 乘 $\prod_j c_j$），对 $|\Psi|^2$ 物理无影响，但让参数空间出现**零恢复力的平坦方向**：列范数可以无声漂移，$|O|/\lambda_{\mathrm{eff}}$ 膨胀、QGT 软方向被 diag_shift 放大 → 梯度爆炸（09-17 诊断：220 步崩溃，$E_1$ 误差 1.3 → 19.5 mHa）。

**解决（三层）**：

1. **gauge-fixed 单列机器**：`create_single_machine_gauge_fixed` 把每条列函数归一为 $\psi_j(x_{\mathrm{HF}})=1$（除以参考组态上的输出），冻结网络重建与 active 初始化都在此规范下进行；
2. **列规范 $g[m]$**：训练时 active 列输出减常数 $g[m]$（$A$ 列乘 $e^{-g_m}$），每 10 步把 $g$ 重置为 $\log\psi_m$ 在样本上的均值，把列尺度钉住；$g$ 全程 `stop_gradient`，不进梯度；
3. **数值防线兜底**：float64 + diag_shift + 全局/逐参数裁剪（§3.4）。

### 4.2 列的混合旋转（K×K 酉/一般线性）

$A\to AU\ (U\in\mathbb{C}^{K\times K})$ 改变各列但 $\det\to\det A\cdot\det U$；$|\Psi|^2$ 在 $U$ 酉时严格不变，一般 $U$ 时只改整体尺度。这是**物理上无关的方向**，但也是提取本征态的**必要通道**：物理态由 $v_{\mathrm{rot}}$ 定义（§2.2），训练只需把列空间推向本征子空间，具体哪组基无所谓。

**解决**：训练中不显式对抗列旋转（行列式结构使其无害化：旋转方向上 $\mathcal{L}$ 是平的，SR 度量自然把它压到零空间）；**验收/提取时**才用 block 广义本征固定 $v_{\mathrm{rot}}$，且冻结块扩展时只动 active 块（§2.2 纪律）。

### 4.3 冻结列的规范衔接

npz 里存的 $\varphi_k$ 与旧网络 $\psi_j$ 的关系是列约定 $\varphi_k=\sum_j\psi_j\,v_{\mathrm{rot}}[j,k]$。重建冻结列必须沿用**同一 gauge-fixed 机器 + 同一 $v_{\mathrm{rot}}$**，否则保真度断言（§6.2）失败。这也是「machine 输出是 $\log\psi$、重建全空间列要先取 $\exp$」这个坑的来源。

## 5. K=4 训练 300 轮后如何冻结

Stage 0（`H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.py`，300 步）产出两样东西：

1. **参数**：`data/26-10-07-19-57_history_..._K4__realFFNN.pkl` 的 `history['params'][-1]`（4 条 gauge-fixed 列网络的最终参数 $\theta^{\mathrm{old}}$）；
2. **物理态**：`冻结K4/近似本征态_K4_26-10-07-19-57.npz`（`phi_vecs` $(16,4)$、`v_rot` $(4,4)$、`E_rayleigh`、`all_states` 基矢序）。

冻结 = 把这两样合成为**常量列**：

$$
\varphi_i(x)=\sum_{j=0}^{3}\psi_j^{\mathrm{old}}\!\big(x;\theta^{\mathrm{old}}\big)\,\big(v_{\mathrm{rot}}\big)_{ji}\qquad(\text{stop-gradient 闭包}).
$$

要点与纪律：

- **冻结保证的是「不再变差」，不是「更好」**：$\varphi_0..\varphi_3$ 的精度永远停留在冻结时刻（本项目 Rayleigh 误差 0.5~0.8 mHa，已在化学精度内）。所以 Stage 0 必须先训到化学精度（$\approx1.6$ mHa）再冻结——项目铁律。
- **active 列必须全新随机初始化**：若沿用 Stage0 收敛列，投影后 2 列会落入冻结 span 的低维残余，行列式奇异/病态（工程惯例，0923 起）。B' 初始化预检实测 $\mathrm{cond}(B_{6\text{列}})=3.53$，健康。
- 冻结列的 $H$ 作用不需要 $H\varphi$ 的显式矩阵：$H$ 线性 ⇒ $H\varphi_i=\sum_j(v_{\mathrm{rot}})_{ji}\,H\psi_j$，逐点用 get_conn 算（§7）。

## 6. 基于已知 K=4 解再跑 K=2：新能级与新本征态

冻结后问题变成：**在 $\mathrm{span}(\varphi_0..\varphi_3)$ 的正交补里做变分，找最低 2 个本征对**。两个实现：

### 6.1 方案 A（精确版，小系统特权）

- 从 npz 载入 $\varphi$ 向量做 Gram-Schmidt 得 $F\in\mathbb{C}^{16\times4}$（含基矢序断言、残差断言）；
- active 列振幅向量 $A\in\mathbb{C}^{16\times2}$ 显式投影：$\chi=A-FF^\dagger A$；$\Psi=\det[\chi_1|\chi_2]$（$2\times2$）；
- $K=2$ 联合采样（32 bit walker）+ 同一套 Wirtinger/$\mathrm{Re}(S)$/重要性加权梯度（`NES_VMC_deflation_real.py`）；
- 投影切空间 $\partial\chi=Q\,\partial A$（$Q=I-FF^\dagger$）由 autodiff 过 $\chi=A-FF^\dagger A$ 自动给出；
- 1500 步/167 s：$\hat\lambda=(-0.02803225,\,-0.01448578)$，误差 $(0.0005,\,0.018)$ mHa，FCI 重叠 $(0.99999,\,0.99813)$，$\mathrm{Var}\sim4\times10^{-7}$，冻结剩余重叠 $\sim10^{-32}$。

注意：A 版训练循环**每步接触 16 维**（$F$、$A$、$\chi$ 都是全空间向量），是「精确极限」参照物，不是 matrix-free。

### 6.2 方案 B'（真 Matrix-Free，行列式自动投影）

- 冻结列以**旧网络逐点求值 + $4\times4$ $v_{\mathrm{rot}}$** 进入 $6\times6$ 行列式（§5），active 2 列 fresh FFNN：

$$
\Psi(x_1,\dots,x_6)=\det\big[\;\varphi_0\;\varphi_1\;\varphi_2\;\varphi_3\;\chi_1\;\chi_2\;\big],
$$

- 数学核心：$\det[\Phi|C]=\det[\Phi|Q_\Phi C]$ 逐点成立 ⇒ **投影由行列式不变性自动实现**，不需要 $F$、Gram、$\|\psi\|^2$；
- $K=6$ 联合采样（48 bit）+ `machine_pow=1` + $w\propto|\Psi|$；梯度/QGT/gauge 与 §3、§4 相同；
- 1500 步/1054 s：$\hat\lambda=(-0.02802664,\,-0.01448541)$，误差 $(0.0061,\,0.0180)$ mHa，FCI 重叠 $(0.99985,\,0.99820)$，$\mathrm{Var}\approx2\times10^{-6}$，active 对冻结 span 剩余重叠 $\sim10^{-30}$（自动投影严格成立的数值证据）；
- 新近似本征态 $\hat\varphi_4/\hat\varphi_5$：诊断步把 active 列全空间振幅 $B_a\in\mathbb{C}^{16\times2}$ 投影到 $F^\perp$（$B_p=B_a-FF^\dagger B_a$）后解 $2\times2$ 广义本征（仅诊断），存 npz。

### 6.3 两版对照

| | 方案 A（显式投影） | 方案 B'（det 自动投影） |
|---|---|---|
| 训练循环内 16 维运算 | 有（$\chi=A-FF^\dagger A$） | **无** |
| $\hat\varphi_4$ 误差/重叠 | 0.0005 mHa / 0.99999 | 0.0061 mHa / 0.99985 |
| $\hat\varphi_5$ 误差/重叠 | 0.018 mHa / 0.99813 | 0.018 mHa / 0.99820 |
| 1500 步耗时 | 167 s | 1054 s（前向多 4 列冻结网络） |
| 可推广到 $D\gg16$ | 否（依赖全空间向量） | **是** |

结论：**自动投影在真实训练中完全替代了显式投影矩阵，精度持平（均贴近 oracle $\nu$）**。

## 7. 为什么是 Matrix-Free

逐操作审计（训练循环，B' 版）：

| 操作 | 计算对象/规模 | 接触 $2^N=16$？ |
|---|---|---|
| MCMC 采样 | 48-bit walker，局域 Hop 移动 | 否 |
| 前向 $\log\Psi$ / $E_L$ | 3200 样本 × $6\times6$ `slogdet`/`solve`，网络逐点 | 否 |
| 哈密顿量作用 | `get_conn_padded` 每组态 $\le13$ 连接 | 否（稀疏局域） |
| 冻结列 $\varphi_i$ | 旧网络逐点求值 + $4\times4$ $v_{\mathrm{rot}}$ 组合 | 否（从不物化 16 维向量） |
| 力梯度 / QGT | vmap over 样本；$O$ 阵 $3200\times P$，$P\approx290$ | 否 |
| SR solve / optax | $P\times P$ 参数空间 | 否 |
| 列规范重置 / $\mathrm{eig}(E_L)$ | 样本均值 / $6\times6$ 特征值 | 否 |

全空间稠密运算只存在于**循环外**：启动 oracle 预检、初始化 $\mathrm{cond}(B)$ 检查、每 25 步 block 诊断、最终验收、绘图——全删不影响训练本身。

「Matrix-Free」的准确含义：**每步计算量与希尔伯特空间维数 $D=2^N$ 无关**，只依赖（样本数 × 每组态连接数 × 参数数）。对 H2/6-31G 这是「也能跑」；对真实分子（$D\sim10^{6}\text{–}10^{20}$）这是「只有这么跑」——方案 A 的 $F\in\mathbb{C}^{D\times4}$ 根本存不下。

## 8. 算法因何高效

1. **正交约束零成本**：与冻结 span 正交由行列式反对称性自动实现（§6.2 核心恒等式），无需 penalty 项、Gram 求逆或 Lagrange 乘子——这三者在多激发态 VMC 里正是数值事故高发区。
2. **已知解不重算**：$E_0..E_3$ 已收敛到 mHa 级，冻结后梯度维度只剩 2 列（$P_{\mathrm{active}}\approx290$ vs 全量 6 列 $\approx870$）；K=4 的采样、优化、方差缩减成果被「免费」继承。
3. **增量谱学**：要 $E_4/E_5$ 不需要重训整个 6 列问题。反例即基线：**K=6 全量同时训 300 步，$E_4/E_5$ 误差 59.7/493 mHa**；冻结 + 2 列扩展 1500 步达 0.006/0.018 mHa——两个数量级差距，源于「同时收敛 6 个态」的优化难度远高于「在已知子空间补全 2 维」。
4. **一次采样覆盖全部态**：联合行列式分布 $|\det[\Phi|C]|^2$ 同时给冻结块与 active 块提供梯度信号，避免逐态多次运行。
5. **偏差可控可测**：冻结偏差有 oracle 下限 $\nu=\lambda_{\min,1:2}(B^\dagger H B)$ 可精确核算（本项目 $\nu-E^{\mathrm{FCI}}\approx0.0001$ mHa），训练误差与方法误差彻底分离。

## 9. K=6 拆成 K=4 + K=2 的意义

- **数值代数对应物**：这是特征值 deflation / 子空间扩张（Sleijpen–van der Vorst 传统）在 VMC 里的实例——收敛的近似本征对 $(\varphi_i,E_i)$ 被「扣除」，后续只在正交补 $\mathcal{H}\ominus\mathrm{span}(\Phi)$ 里搜索。区别在于扣除不靠显式矩阵，而靠采样分布的行列式结构。
- **变分结构意义**：$K$ 列拟设的空间是嵌套的——$K=4$ 的列空间 $\subset$ $K=6$ 的列空间。拆分让训练沿「先低能 4 维子空间收敛 → 再补 2 维」的**课程（curriculum）**推进，每一步都是良态的低维问题。
- **物理意义**：支持**增量激发谱**——分子的光谱兴趣往往散布在很多激发态上，逐批扩展（$K=4\to6\to8\to\cdots$）比一次 $K$ 取大更稳、可并行、可断点续训。
- **工程意义**：冻结块是「检查点」——Stage 产物（pkl 参数 + npz 态）可版本化、可复用（本项目同一份 K=4 产物同时喂养了方案 A 与方案 B'）。
- **方法论意义（最重要）**：B' 证明扩展步骤**不需要全空间表示**，因此该拆分范式原则上直接适用于 $D$ 巨大的真实分子体系；方案 A 则给出小系统上的「精确极限」标尺，两者互为验证。

---

# 第三部分 现状与进展

## 10. 当前状态（截至 2026-10-16）

**已完成并验证**：

- H2/6-31G 全链：K=4 基线（300 步，$\varphi_0..\varphi_3$ Rayleigh 误差 0.5–0.8 mHa）→ 冻结 → K=2 扩展双实现（A/B'）；
- 最终谱（B'，1500 步）六能级 $\mathrm{eig}(\langle E_L\rangle_{\mathrm{mean}})$：$[-1.054349,\,-0.957906,\,-0.668952,\,-0.551402,\,-0.028042,\,-0.014474]$，前四与 FCI 差 $\le3\times10^{-6}$ Ha，后二与 FCI 差 $\le0.018$ mHa；
- 验收四线全过：误差 $<2$ mHa ✓、$\mathrm{Var}<10^{-4}$ ✓、FCI 重叠 $>0.99$ ✓、冻结剩余重叠 $<10^{-6}$ ✓（实测 $\sim10^{-30}$）；
- 训练稳定性：0 skip、0 裁剪、valid_ratio $=1.0$、ESS $\approx1200/3200$；
- 产物齐备：npz（$\hat\varphi_{4,5}$/$v_{\mathrm{rot}}$/$\hat\lambda$/$\nu$/重叠）、history pkl、5 联诊断图、日志、两份 1015 解读文档。

**已知边界与注意**：

- 冻结精度 = 冻结时刻精度，扩展结果的绝对下限由 Stage 0 决定（化学精度铁律）；
- Loss 目标必须按 $\mathcal{L}^\ast=\sum_i E_{\mathrm{ray}}(\varphi_i)+\sum_k\nu_k$ 核对，不能对 $\sum E^{\mathrm{FCI}}$；
- $\mathrm{eig}(E_L)$ 每步监控含采样噪声（偶发 $\pm10$ mHa 尖峰），验收以每 25 步 block 精确值为准；
- 稀疏诊断序列绘图需「连续监控线 + 诊断散点」两段式（NaN 断线坑，已修）；
- B' 每步成本高于 A（多 4 列冻结网络前向），换来的是维数无关性——$D$ 越大这笔账越划算。

## 11. 下一步

1. **规模验证**：把 B' 骨架推到 LiH/更大基组（物理空间 $>16$ 维，方案 A 失效、只能 matrix-free），检验 $D$ 无关性主张；
2. **迭代扩展**：$K=4\to6\to8$ 连续 deflation 的累积偏差行为（每轮 oracle 下限核算）；
3. **冻结表示简化**：探索不载旧网络的冻结（采样组态上查表/低秩压缩），进一步降前向成本；
4. **自适应 SR**：diag_shift 自适应（$\eta\propto\lambda_{\max}(S)$）与信任域，替代固定 $0.1$，为更大参数空间护航；
5. **active 列数与批次策略**：$K_{\mathrm{active}}$ 选取、扩展步数预算与化学精度目标的经验律。
