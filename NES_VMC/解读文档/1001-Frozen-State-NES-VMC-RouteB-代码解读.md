# Frozen-State NES-VMC（Route B 冻结流形）— 代码解读

> 方案依据：`NES_VMC/文档/NES-VMC冻结策略/1001-冻结方案.md`
> 核心代码：`NES_VMC/experiments/H2分子/NES_VMC_frozen_routeB.py`
> 复现 notebook：`NES_VMC/experiments/H2分子/H2_631G_K4_frozen_routeB_RealFFNN.ipynb`
> Ansatz / 梯度 / QGT 复用：`H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.ipynb`（实参数 FFNN + 振幅-相位分离）

## 0. 与 0923「矩阵自由 deflation」的关系与区别

两者都实现「冻结 φ₀ 后继续优化激发态」，但机制不同：

| | 0923 deflation（Route A） | 本文 Route B（冻结流形） |
|---|---|---|
| 核心思想 | 显式投影：局域能量用 **Q₀HQ₀**，active 列被替换为 Q₀ψⱼ | **改变变分流形**：把 φ₀ 作为行列式第一列，`det[φ₀,ψ₁',ψ₂',ψ₃']` |
| 采样分布 | `det[Q₀ψ₁, …]`（投影后列） | `det[φ₀, ψ₁', ψ₂', ψ₃']`（φ₀ 直接进联合采样） |
| 正交性来源 | 显式 Q₀ 算子 | 行列式自动消除与 φ₀ **线性相关**的成分（implicit，文档 §14） |
| 维度 | K=4 → 有效 3 | K=4 保持（文档 §17：冻结 ≠ 降阶） |
| 额外采样器 | 需要 `|φ₀|²` 采样器估 overlap 标量 c/h/E₀ | **不需要**——φ₀ 是列 NN 的 stop-gradient 组合，直接进行列式 |

Route B 的最大优点：**不引入任何 overlap penalty、不显式构造投影算子、不需要第二个采样器**，完全靠 NES-VMC 自身的多态行列式结构实现「隐式投影」。

## 1. 总体流程

```
Stage 0（标准 NES-VMC, K=4 列）
    │  训练 300 步，φ₀ 收敛
    ▼
提取 φ₀：MCMC 样本上解 4×4 广义本征 M v = λ S v → v₀，φ₀ = Σⱼ v₀ⱼ ψⱼ（stop-gradient）
    │
    ├────────────────────────┐
    ▼                        ▼
Control（对照组）          Frozen（Route B）
4 列全部继续训练 200 步     流形换成 [φ₀, ψ₁', ψ₂', ψ₃']
（φ₀ 会漂移）              只训练 3 个 fresh active 列 200 步
                           （φ₀ 构造性严格不变）
```

两条分支共享同一 Stage 0 产物，保证对比公平（同种子列、同步数、同超参）。

## 2. 关键函数解读

### 2.1 `make_frozen_bundle`（≈ 文档 §2「冻结的是 φ₀ 而不是 v₀」）

把 Stage-0 的 K 个 gauge-fixed 单列 machine + 收敛参数 + 基态旋转系数 v₀ 打包成常量。
冻结参数**不进入任何 `jax.grad` 的追踪路径**（Python 侧常量闭包），比 `stop_gradient` 更彻底。

### 2.2 `make_frozen_forward` —— Route B 的心脏（≈ 文档 §12-13）

对单个 NES walker `xw (K, n_spin)` 构造冻结流形矩阵：

```
Lf[n,j] = logψ_j^frozen(x_n)        （gauge-fixed 列，ψ_j(ref)=1）
La[n,m] = logψ_m^active(x_n) − g_m   （active 列 + 列规范）
s       = max Re(Lf ∪ La)            （per-walker 稳定化 shift，stop-gradient）

φ₀ 列：  e^{-s}·φ₀(x_n) = Σⱼ v₀ⱼ · e^{Lf[n,j]−s}     ← H 线性 ⇒ Hφ₀ = Σ v₀ⱼ Hψⱼ
A = [φ₀_scaled | exp(La−s)]           （K×K，第一列是冻结态！）
Z = [Hφ₀_scaled | HPa·e^{-g}]         （K×K）

logΨ = K·s + slogdet(A)              （行列式回到对数域）
E_L  = solve(A, Z)                   （矩阵局域能量）
loss = Re tr(E_L)
```

对应文档 §13：采样分布 `p(X) ∝ |det[φ₀,ψ₁,ψ₂,ψ₃]|²`。
文档 §14 的性质在这里自动成立：若 active 列含 a·φ₀ 成分，行列式第一列与它线性相关 ⇒ 该成分被消掉 ⇒ **冗余方向被 suppress**，无需显式投影。

数值稳定化沿用 notebook 的 shift 技巧：`exp(logP − s)` 防 complex 溢出；`s` 取冻结列与 active 列的联合最大实部；`E_L = solve(A,Z)` 对 shift 不变（A、Z 同乘 e^{-s}）。

### 2.3 `make_frozen_grad_fn` / `make_frozen_qgt_fn`（实参数 Wirtinger 版）

与 notebook 的 `make_grad_fn_gauge_real` / `make_qgt_fn_gauge_real` 同构，只是 machine 换成冻结流形 forward：

- `∇logΨ = ∇Re(logΨ) + i·∇Im(logΨ)`（实参数、复输出）
- 力梯度 `∇E = 2·Re⟨ tr(E_L − ⟨E_L⟩) · conj(∇logΨ) ⟩`
- QGT 度量取 `Re(S)`，`S = O†_c O_c / N`，加 `diag_shift·I`
- 梯度只流向 3 个 active 列的参数——φ₀/v₀/冻结参数是闭包常量，`∂(det)/∂θ_frozen ≡ 0`，
  这正是文档 §7-8「人为定义 frozen state 不参与 rotation」的实现形式。

### 2.4 `make_frozen_MS_fn` —— 冻结流形上的 4×4 广义本征（≈ 文档 §17）

列 0 是 φ₀、列 1..3 是 active 列，在 MCMC 样本上估计 `M_B、S_B`（与 `NES_VMC_tool.make_MS_estimator_fn` 同法：逐 walker Frobenius 归一化后样本平均）。

**注意（文档 §5 的坑）**：直接对完整 4×4 问题 `scipy_eigh(M_B, S_B)` 会重新旋转 φ₀，得到
`ãφ₀ = aφ₀ + bψ₁ + …`，破坏冻结。因此 Route B 的**物理输出**不能取完整对角化的第一本征矢；
完整 4×4 本征只用于监控/对照（检验块结构 `|V_B[0,0]| ≈ 1` 是否成立），真正的态提取见 §2.6。

### 2.5 gauge reset（active 列规范）

active 列每 `reset_period` 步做一次列均值吸收：`g ← mean over samples of logψ_m`（仅 active 列）。
冻结列的 gauge 已在 Stage 0 固定（ψ_j(ref)=1），不再动。φ₀ 作为常量列其整体尺度由 v₀ 携带，
行列式对公共尺度不敏感（被 Rayleigh 商/slogdet 归一吸收）。

### 2.6 `exact_frozen_block_analysis` —— block 结构的精确诊断（≈ 文档 §7、§9-11）

小系统特权：16 维全空间稠密 H。Route B 的正确态提取是**分块**的：

```
冻结块：  E_φ₀ = ⟨φ₀|H|φ₀⟩ / ⟨φ₀|φ₀⟩          （Rayleigh 商，φ₀ 原样保留）
active 块：投影基 P₀ψ_m = ψ_m − c_m φ₀，c_m = ⟨φ₀|ψ_m⟩/⟨φ₀|φ₀⟩
           在 {P₀ψ₁, P₀ψ₂, P₀ψ₃} 上解 3×3 广义本征 → E₁'..E₃'
```

这就是文档 §11 的 **implicit matrix-free projection** 的「精确极限」版本：
训练用行列式隐式做，诊断用显式投影验证。两个块合起来 = 块对角 `V_B = diag(1, V_active)`
（文档 §7），φ₀ 不参与 rotation。

### 2.7 `gauge_fixed_basis_vectors` —— 一个容易踩的坑

多列基向量必须**共享同一个 shift** 再 `exp(logL − s)`。若逐列各自归一化，列间相对尺度被破坏，
`φ₀ = Σ v₀ⱼ ψⱼ` 的系数含义就错了（v₀ 是相对原始列定义的）。广义本征分析本身对逐列缩放不变，
但 φ₀ 构造会变——这是调试中实际修掉的一个 bug。

## 3. 与 notebook 参考实现的对应

| notebook（RealFFNN） | 本脚本 | 变化 |
|---|---|---|
| `SingleStateAnsatz_SharedBackbone` | 同名 | 不变 |
| `NESTotalAnsatz_stable` | 同名（Stage0/control 用） | 不变 |
| `make_grad_fn_gauge_real` | Stage0/control 复用 | 不变 |
| — | `make_frozen_grad_fn` | 新增：machine 换成冻结流形 |
| `NESFermionHopRule` 联合采样 | 同一规则 | 不变（walker 结构 K×n_spin 不变） |
| `create_gauge_reset_total_machines` | Stage0/control 复用 | 不变 |
| — | `make_frozen_forward` | 新增：det[φ₀, ψ'] 前向 |
| `compute_lam_v_from_samples` | Stage0 提取 v₀ | 复用 |
| — | `exact_frozen_block_analysis` | 新增：block 精确诊断 |

## 4. 运行方式

```bash
cd NES_VMC/experiments/H2分子
# 完整实验（Stage0 300 + Control 200 + Frozen 200，约 5 分钟）
/opt/miniconda3/envs/Netket/bin/python NES_VMC_frozen_routeB.py --tag full
# 冒烟
/opt/miniconda3/envs/Netket/bin/python NES_VMC_frozen_routeB.py \
    --n-iter0 40 --n-iter1 30 --n-samples 60 --n-chains 4 --tag smoke
```

结果 pickle 落在 `./data/`，日志落在 `./日志/`。

## 5. 机制验证要点（冒烟实验已确认）

1. **构造性冻结**：Frozen 阶段 φ₀ 对 FCI 基态保真度逐步严格恒定（同一向量，零漂移）；
   Control 阶段 φ₀ 保真度随训练波动（漂移）。
2. **行列式隐式投影有效**：active 列随机初始化时 block E₁ ≈ -0.14，训练后收敛到 ≈ -0.95
   （FCI E₁ = -0.958），且提取态与 FCI 第一激发态重叠 ≈ 0.99、与 φ₀ 重叠 ≈ 0（正交性由
   NES 广义本征 + 行列式结构产生，文档 §16 的完整逻辑链成立）。
3. **块结构成立**：冻结流形 4×4 广义本征的最低本征矢以 φ₀ 为绝对主导分量。
4. **冻结 ≠ 降阶**（文档 §17-18）：K 保持 4，行列式保持 4×4，只省掉了 1 列的参数与梯度。
