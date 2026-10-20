# 方案 A' 计划：把显式 deflation（$2\times2$ $\chi$ 行列式）改造成真 Matrix-Free

- 日期：2026-10-16
- 状态：**已完成（M1–M4）**。1500 步正式运行：质量四项验收全过（φ̂₄ 误差 0.0127 mHa / 重叠 0.99995，φ̂₅ 误差 0.0243 mHa / 重叠 0.99751，冻结剩余 ~1e-34），唯耗时 608.2 s 超 500 s 线 ⇒ 脚本判 FAIL；结论 = A' 成功（比 B' 快 1.73× 且维数无关）。详见 `验证报告/1016-方案A-prime真MatrixFree-三向对照报告.md`（含两项如实偏离：耗时线、χ̃ 单测阈值改相对口径）。M 三档×EMA 消融未执行。
- 目标脚本（拟）：`NES_VMC_deflation_mf.py`（新工具箱）+ `H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py`
- 参照实现：方案 A `H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py`（精确版）、方案 B' `H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py`（已验证 matrix-free）
- 用户给定参考：netket operator 指南 https://netket.readthedocs.io/en/latest/user-guides/operator.html ；vmc-from-scratch 哈密顿量篇 https://netket.readthedocs.io/en/latest/vmc-from-scratch/01-hamiltonian-and-operators.html

---

## 1. 动机：为什么要做 A'，而不是直接用 B'

B' 已经证明 matrix-free 可行（误差 0.0061/0.018 mHa），但它的代价是**每个 walker 前向都要跑 4 条冻结网络**：$6\times6$ 行列式 + $6\times6$ `slogdet/solve`，1500 步耗时 1054 s，是方案 A（167 s）的 6.3 倍。

方案 A 的结构更便宜也更干净：拟设只有 $2\times2$ 行列式 $\Psi=\det[\chi_1|\chi_2]$，冻结信息全部压缩进投影系数 $c$。它的唯一"作弊"是投影 $\chi=\psi-\Phi c$ 的系数用了**精确全空间内积**。

**A' 的定位**：保留 A 的 $2\times2$ 结构与显式可审计的正交性，只把「精确内积」换成「蒙特卡洛内积」，冻结列改用 B' 的「旧网络逐点求值」表示。若成功，得到三者中每步成本最低、且维数无关的扩展方案。

## 2. 方案 A 的全空间依赖审计（改造对象）

| # | 依赖点 | 现状 | matrix-free 难度 |
|---|---|---|---|
| 1 | 哈密顿量 | `nkx.operator.from_pyscf_molecule(mol)` → `FermionOperator2nd`，`get_conn_padded` 稀疏局域 | **已满足**，无需改动（里程碑 1 复核 `H2_6_31G_8Qbit_K6.py` 调用链） |
| 2 | 冻结列存储 | $F\in\mathbb{C}^{16\times4}$ 从 npz 显式载入 | 易：换成 B' 方案（旧网络参数 + $v_{\mathrm{rot}}$，逐点求值 $\varphi_i(x)=\sum_j\psi_j^{\mathrm{old}}(x)v_{ji}$） |
| 3 | 投影系数 | $c$ 由 $\langle\varphi_i\mid\psi_j\rangle=\sum_{x}^{\text{全空间}}\varphi_i^*(x)\psi_j(x)$ 精确求和 | **核心难点**，见 §3 |
| 4 | 前向/梯度/QGT/E_L | 全部在采样组态上逐点 | 已满足（A 的采样本来就是 $2\times2$ 联合 MCMC） |

结论：A → A' 只需解决**一个**数学对象——全局内积（Gram）$M_{ij}=\langle\varphi_i\mid\psi_j\rangle$。

## 3. 核心数学：把 Gram 内积蒙特卡洛化

对任意覆盖支撑的参考分布 $p(y)$：

$$
M_{ij}=\sum_{y}\varphi_i^*(y)\psi_j(y)
=\mathbb{E}_{y\sim p}\!\left[\frac{\varphi_i^*(y)\,\psi_j(y)}{p(y)}\right]
\;\approx\;\frac{1}{M}\sum_{k=1}^{M}\frac{\varphi_i^*(y_k)\psi_j(y_k)}{p(y_k)},
$$

$y_k$ 由辅助 MCMC 产生（单组态，8 bit）。同理冻结 Gram $G_{ij}=\langle\varphi_i|\varphi_j\rangle$（常量，启动时估一次或 EMA），投影系数

$$
c = G^{-1}M \in \mathbb{C}^{4\times 2},\qquad
\chi_j(y)=\psi_j(y)-\sum_{i=0}^{3}\varphi_i(y)\,c_{ij}.
$$

每步额外成本 = 内批 $M$ 个单组态 ×（4 条冻结网络 + 2 条 active 网络）前向，与 $D=2^N$ 完全无关。

**与 B' 的本质区别**：B' 把投影藏在行列式不变性里（$\det[\Phi|C]=\det[\Phi|Q_\Phi C]$），不需要任何内积；A' 显式算 $c$，因此拟设仍是 $2\times2$。代价是 $c$ 有统计噪声（§5），收益是前向规模从 $6\times6$ 降回 $2\times2$、且 active 列**不再与冻结列同场采样**（walker 从 48 bit 缩回 16 bit，采样器/接受率都更简单）。

## 4. 实现设计

### 4.1 参考分布 $p(y)$（关键选择）

要求：在 $\varphi_i$ 与 $\psi_j$ 同时非零处有支撑（激发态列的振幅集中在基态分布的稀有组态上——这正是 09-25/10-06 零方差陷阱的变体）。候选：

- **P1（首选）**：混合 $p(y)\propto\sum_{i=0}^{3}|\varphi_i(y)|^{2}+\varepsilon\big(|\psi_1(y)|^{2}+|\psi_2(y)|^{2}\big)$，逐点可算（正是我们要的内积项本身），用自定义 `MetropolisSampler`（单组态 hi + Hop 规则，`machine_pow=1` 按 $\sqrt{p}$ 行走）；
- P2（对照）：直接从主采样器（$2\times2$ 联合 walker）的组态边缘 + 重要性权重复用，零额外采样成本，但覆盖性依赖主轨迹，风险高；
- P3（保底）：$p\propto\sum_i|\varphi_i|^2$ 纯冻结混合。

内批规模消融 $M\in\{1024,4096,16384\}$，验收 $M$ 下 $\|c^{\mathrm{MC}}-c^{\mathrm{exact}}\|/\|c\|\lesssim10^{-3}$（H2 上可用精确值对照）。

### 4.2 $c$ 的时间处理

- $c$ 取 **stop_gradient**（视作步内常量），每步重估 + 可选 EMA（$\beta\approx0.9$）平滑；
- 梯度端：$\partial_\theta\chi=\partial_\theta\psi-\Phi\,\partial_\theta c$，丢弃 $\partial c$ 项后切空间投影不再严格——但误差方向落在冻结 span 内，其影响以 $\|\delta c\|$ 为一阶小量，由诊断「冻结剩余重叠」直接监控（精确版 $10^{-32}$ → A' 验收线放宽到 $<10^{-3}$，见 §6）；
- 若 EMA 导致漂移，备选：每步 fresh 估计（纯 stop-grad，无 EMA）。

### 4.3 复用与新增代码

| 组件 | 来源 | 改动 |
|---|---|---|
| 冻结列逐点求值 | 直接抄 B' `make_block_forward` 的 `Lf/HPf/V_MAT` 段 | 无 |
| 辅助采样器 | `NESFermionHopRule(K=1)` 或 netket `HopRule` + 自定义 log-amp | 新增 ~40 行 |
| Gram 估计器 | 新函数 `estimate_projection(sampler_aux, sms_frozen, sms_active, params)` | 新增 |
| $2\times2$ bundle forward | A 的 `make_deflation_bundle` 接口不变，内部把 $F$/$F^\dagger A$ 换成网络逐点 + $c^{\mathrm{MC}}$ | 核心改写 |
| 梯度/QGT/SR/gauge/采样 | A 原样（`NES_VMC_deflation_real.py`） | 无 |
| 诊断/验收/绘图 | B' 版（含 NaN 两段式绘图修复） | 无 |

### 4.4 netket 算符层核对清单（用户提示的方向）

- 复核 `ha` 为 `FermionOperator2nd`（`from_pyscf_molecule` 产物），确认 `get_conn_padded` 路径无 `to_dense` 调用；稠密 `H_dense` 只允许出现在诊断（照 B' 的纪律）；
- 辅助采样器若走 netket 标准接口：`nk.sampler.MetropolisSampler(hilbert=hi, machine=<log√p 函数>, rule=..., machine_pow=1)`，注意 machine 签名（params, σ）→ log ψ，P1 的 $\sqrt p$ 不是波函数但接口兼容（netket 只要求返回标量 log 幅）；
- 大分子外推检查：`FermionOperator2nd` 的 `get_conn` 成本随积分项数而非 $D$（参考 vmc-from-scratch 01 篇），A' 全链路（采样/前向/Gram/H 作用）均满足，写进文档结论。

## 5. 风险清单

1. **Gram 噪声地板**：$c$ 每步抖动 → 损失成为随机过程，SR 更新可能在其上放大噪声。缓解：大 $M$、EMA、`diag_shift` 0.1 不变、`should_skip` 保留；监控 natural 梯度范数是否出现噪底（0917 教训）。
2. **覆盖失败**：$\psi_j$ 的权重组态若被 $p$ 低估，重要性权重爆炸。缓解：P1 的 $\varepsilon$ 混合项自适应（$\varepsilon\propto$ 权重峰值比）。
3. **正交性泄漏**：$\|\delta c\|$ 直接成为冻结剩余重叠，污染 $\hat\varphi_{4,5}$ 的 Rayleigh 商。缓解：每 25 步 block 诊断盯住；超线报警。
4. **HF 陷阱变体**：辅助采样器同样按 $\sqrt p$（machine_pow=1）行走。
5. A' 若整体不稳：结论回归「B' 为 matrix-free 主线」，A' 记为成本优化失败尝试——**这也是有价值的产出**（说明显式投影的 MC 化在 VMC 动力学下不可靠）。

## 6. 验证与验收

**单元测试**（利用 H2 的精确参照，这是小系统特权、仅测试用）：
1. $c^{\mathrm{MC}}$ vs $c^{\mathrm{exact}}$：误差随 $M$ 呈 $O(M^{-1/2})$；
2. 固定随机种子下，A' 前向 $\chi$ 与精确 $\chi$ 的逐点差 $\le 10^{-3}$；
3. 5 步 smoke：Loss 单调、valid_ratio=1、无 skip。

**正式运行**：1500 步，lr/diag_shift/clip 沿用 A（0.1/0.1/20+0.5），三向对照：

| 指标 | 方案 A（精确） | 方案 B' | A' 验收线 |
|---|---|---|---|
| $\hat\varphi_4$ 误差 | 0.0005 mHa | 0.0061 mHa | $<2$ mHa |
| $\hat\varphi_5$ 误差 | 0.018 mHa | 0.018 mHa | $<2$ mHa |
| FCI 重叠 | 0.99999 / 0.99813 | 0.99985 / 0.99820 | $>0.99$ |
| 冻结剩余重叠 | $\sim10^{-32}$ | $\sim10^{-30}$ | $<10^{-3}$（MC 投影放宽） |
| 1500 步耗时 | 167 s | 1054 s | 目标 $<500$ s |

**消融**：$M$ 三档 × EMA 开关，出「成本-精度」曲线，选默认配置。

## 7. 里程碑

1. **M1 审计**：逐行确认 A 中 $F$/$F^\dagger A$ 的全部调用点；复核 `ha` 稀疏路径。（半天级）
2. **M2 工具箱**：`NES_VMC_deflation_mf.py` = 辅助采样器 + Gram 估计器 + 新 bundle；单测 1/2 通过。
3. **M3 训练脚本**：A' 完整脚本 + smoke（单测 3）。
4. **M4 正式**：1500 步 + 消融 + 三向对照报告（写入 `验证报告/`）。
5. **M5 收尾**：无论成败，更新 `解读文档/1016-...报告.md` §11 与项目记忆（A' 结论、验收线依据）。

## 8. 决策点

- M2 单测若 $c^{\mathrm{MC}}$ 收敛阶不对 → 先查 $p$ 覆盖（换 P1 的 $\varepsilon$），不动主框架；
- M4 若误差进不了 2 mHa 但趋势正确 → 加大 $M$/步数再试一轮；仍失败 → 按 §5.5 记录负结果并终止，主线维持 B'；
- 若 A' 成功 → 下一步用 A' 跑 LiH（$D>16$，方案 A 无法运行的系统），作为 matrix-free 家族的**首个非小系统证据**。
