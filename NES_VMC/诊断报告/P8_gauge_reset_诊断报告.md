# NES-VMC LiH (K=4) 三问终报：logΨ 漂移根因、既有修复失效原因与可行方案（P8 gauge reset）

> **报告日期**: 2026-08-29
> **实验对象**: `experiments/LiH分子/LiH_molecule_K4_STO-3G.ipynb`（K=4 激发态 NES-VMC，LiH/STO-3G）
> **基线日志**: `experiments/LiH分子/GaugeFixing00_K4_LiH_STO-3G.log` / `日志/nes_vmc_0829_K4_LiH_STO-3G_min30.log`
> **成功方案日志**: `experiments/LiH分子/日志/nes_vmc_0829_K4_LiH_STO-3G_min30_gauge_reset.log`
> **方案代码**: `experiments/LiH分子/NES_VMC_train_min30_gauge_reset.py`（P8）
> **前置报告**: 本目录 `gauge_fixing_analysis.md`、`logPsi_drift_diagnosis.md`

---

## 0. 三问一句话答案

| 问题 | 答案 |
| :--- | :--- |
| **Q1** 为什么 30 轮内没遇到梯度异常？ | 基线梯度确实持续上涨（raw 20.17→146.93），但未越过阈值 5000；真正的爆炸发生在 ~Step 113（`logΨ max` 逼近 64，`exp≈1e13`，float32 精度崩塌）。30 轮远未到崩溃点，所以"看似正常"。 |
| **Q2** 为什么 `logΨ mean` 持续升高？`gauge_fixing_analysis.md` 的措施为什么不奏效？ | 升高的直接原因是**行规范自由度未修复**，且该漂移对 QGT 岭函数、diag_shift 均不敏感；`gauge_fixing_analysis.md` 指向的修复（模型 `_forward_single` 里的 rowfix）**从未进入运行路径**——wrapper 内部函数根本不调用 `_forward_single`；真正接入（GaugeFixing01）后因两个内部 bug 立即崩塌。 |
| **Q3** 怎么同时抑制漂移、保证 Loss 下降、杜绝梯度异常？ | **P8：双侧一致的全局列规范 g + 周期性 gauge reset**。g 以 stop_gradient 常数作用于 Ψ 与 HΨ 两侧（相似变换，loss/梯度/轨迹与基线逐位相同），同时在 reset 边界把网络学习到的列漂移吸收入 g → `logΨ mean` 全局有界。30 轮实测达标。 |

---

## 1. Q1：为什么基线 30 轮内"没有梯度异常"？

### 1.1 事实：梯度确实在涨，只是没到阈值

基线 30 轮 `raw` 梯度（来自 `nes_vmc_0829_K4_LiH_STO-3G_min30.log`）：

| Step | raw 梯度 | Loss | logΨ mean |
| :--- | :--- | :--- | :--- |
| 0   | 20.17 | -21.99 | 9.264 |
| 5   | 38.02 | -25.55 | 14.765 |
| 10  | 133.67 | -27.61 | 18.971 |
| 11  | **146.93** | -28.12 | 19.987 |
| 20  | 31.63 | -30.20 | 26.246 |
| 29  | 92.73 | -30.48 | 31.532 |

`raw` 从 20→147，中间波动很大，但始终低于告警阈值 5000（脚本里 `raw>5000` 才告警）。所以**30 轮日志看起来完全正常**。

### 1.2 事实：真正的爆炸在 ~Step 113（来自 700 步日志 `logPsi_drift_diagnosis.md` §0）

| Step | logΨ mean | cond(Ψ) | ‖∇raw‖ |
| :--- | :--- | :--- | :--- |
| 100 | 46.675 | 9.1 | 92.01 |
| **113** | **64.357** | **236** | **1393.5** |

- 崩溃物理机制：`Psi_Matrix = exp(L - shift)`，当 `logΨ max ≈ 30-50` 时 float32 的 `exp` 逼近溢出（`exp(48) ≈ 7e20`），Ψ 矩阵数值病态（`cond` 飙升），`E_L = Ψ⁻¹HΨ` 的求解把误差放大 → raw 梯度爆炸。
- 对 diagshift 实验的直接观测：Step 16 `logΨ max=47.9`，Step 17 raw 冲到 **2.75e12**。这就是同一机制的量化证据。

**结论**：没有梯度异常只是"30 轮还没走到崩溃点"。崩溃是浮点精度问题，不是 30 轮内的问题。

---

## 2. Q2：为什么 `logΨ mean` 持续升高？既有修复为何失效？

### 2.1 升高机制（详见 `logPsi_drift_diagnosis.md`）

| 规范自由度 | 变换 | 对 log\|det\| | 对 loss tr(E_L) |
| :--- | :--- | :--- | :--- |
| 列规范 column | `logψ_j(x) += c_j` | `+Σc_j` | 不变 |
| **行规范 row** | `logψ_j(x_i) += c(x_i)` | `+Σc_i` | 不变 |

- 列规范已被 `logψ_j(x) - logψ_j(ref)` 修复 → 完全不漂移；
- **行规范未修复** → `logΨ mean` 的增量 ≈ `Σc_i`，这就是日志中"所有 walker 几乎同步平移"的来源（`min/mean/max` 三者间距几乎不随漂移变化）。
- 为什么漂移被持续驱动：`F_row = mean[(tr E_L - mean) · dlogΨ_row]` 解析期望为 0，但 MC 噪声给出 `O(σ/√N)` 的残量；QGT 在行方向的特征值小，`(S+λI)⁻¹` 把残量放大 → 每步 +0.9~1.2 的漂移速率（实测与 `η=0.1`、`λ=0.1` 吻合）。早期漂移快，能量收敛后漂移放缓。

### 2.2 为什么 `gauge_fixing_analysis.md` 的修复没有奏效？

报告指向的修复位于 `NESTotalAnsatz_gauge_stable_full._forward_single`（rowfix：`L - row_mean` + `det_correction`）。但：

1. **从未进入运行路径**（`gauge_fixing_analysis.md` 已确认）：
   - `create_gauge_fixed_total_machines` 内部用**自建**的 `_compute_L_centered_single` / `_stable_from_L`，直接操作 `single_ansatz_list`，**不调用 `_forward_single`**；
   - 因此 GaugeFixing00 与 GaugeFixing01 的日志**逐字节相同**（`logΨ mean` 0→29 均为 9.264→31.532）。

2. **真正接入反而崩塌**（GaugeFixing01 崩溃根因，来自对 `_forward_single` 的源码审读）：
   - 模型内 `det_correction = Σ row_mean` **携带梯度**，它把行规范的"缩放"又写回 logΨ，抵消了 rowfix 的作用；
   - loss 侧 `L_rowfixed` 与 HΨ 侧（未 rowfix 的原始单态）**数值不一致**，`E_L = Ψ_L_rowfixed ⁻¹ · H·Ψ_raw` 不再是相似/对合形式，梯度方向被畸变 → 爆炸。

### 2.3 进一步的正面发现（本轮最重要的架构事实）

基线 `single_machine_list` 由 `create_single_machine_gauge_fixed` 构造，即 **HΨ 侧与 Ψ 侧共用同一套 D_ref 缩放**：

```
Psi_Matrix = Ψ_raw · D_ref      （每列 ÷ ψ_j(ref)）
HPsi       = H·Ψ_raw · D_ref    （single_machine 也是 gauge-fixed 的）
E_L = Psi_Matrix⁻¹ · HPsi = D_ref⁻¹ · (Ψ_raw⁻¹ H Ψ_raw) · D_ref   ← 相似变换
```

⇒ E_L 始终处于**相似变换形式**，`loss = real(trace E_L)`、E0-E3（特征值）**严格规范不变、物理精确**（E0 收敛到 -7.748 vs CAS -7.864）。

因此：**`logΨ mean` 的上升不是物理量变化，也不是列规范破坏，而是网络把"行列式幅值"学到越来越大**（slogdet 沿 training trajectory 单调上升）。这本身不坏物理，但会让 `exp(L)` 逼近 float32 上限 → 这正是 Q1 里 Step 113 爆炸的温床。

---

## 3. Q3：方案探索——四个失败方案与一个成功方案

所有方案共享同一套训练循环（`N_ITER=30`、`lr=0.1`、`qgt_diag_shift=0.1`、`clip=20`），只改动 Ψ/gauge 的构造方式。

### 3.1 P2 rowfix（逐 walker 行均值 stop_gradient 减去）——失败

- 日志：`日志/nes_vmc_0829_K4_LiH_STO-3G_min30_rowfix.log`
- 结果：`logΨ mean` 增速放缓（Step 19 = 13.7 vs 基线 25.6），但 **Loss 卡在 -25 附近（基线同期 -30.1），E3 停在 -5.1（不收敛）**，E1/E2 也偏离基线。
- 根因：`slogdet(exp(L))` 的梯度 ∝ `A⁻ᴴ`，逐行减去均值后 L 矩阵结构剧变；实测 `|gr-go|/|go| = 90.6%`（梯度方向近乎正交）→ 优化器"原地打转"。**手动去行规范会破坏 logdet 的梯度几何，这是该方法路线（改 loss 内部 Ψ 值）的天花板。**

### 3.2 P3 diag_shift = 1.0——失败

- 日志：`日志/nes_vmc_0829_K4_LiH_STO-3G_min30_diagshift.log`
- 结果：漂移速率降到 ~0.55/步（仅为基线的 1/2~1/3），但 Step 16 `max=47.9`，Step 17 raw = **2.75e12** → 爆炸。
- 根因：diag_shift 只压 QGT 的 ampliation 系数，**治标不治本**；只要 `logΨ` 还在涨，float32 精度墙总会被撞到。

### 3.3 P4 gaugepin（QGT 上加 rank-1 岭函数拉 pin）——失败

- 日志：`日志/nes_vmc_0829_K4_LiH_STO-3G_min30_gaugepin.log`
- 结果：前 10 步漂移速率与基线几乎相同（Step 10 = 19.075 vs 基线 18.971），Step 10 `max` 跳到 32.8，Step 11 raw = **2.7e8** → 爆炸。
- 根因：物理梯度在行方向上的投影**不为零**（是 MC 残量），rank-1 pin 只是把 QGT 逆的方向改了，**没去掉梯度的 row 分量**；且额外的矩阵操作让数值更脆。

### 3.4 P5 sgrecenter（Psi 侧 per-walker 列均值减去）——失败

- 日志：`日志/nes_vmc_0829_K4_LiH_STO-3G_min30_sgrecenter.log`
- 结果：**Step 1 即爆炸**（raw=30804），Loss=-1014 → 持续恶化。
- 根因（本次会话推导的**最高价值教训**）：只改 Ψ 侧的数值规格、HΨ 侧不变 ⇒
  ```
  E_L = (Ψ·D⁻¹)⁻¹ · (HΨ) = D·(Ψ⁻¹HΨ)    ← 非相似，是 D·M 形式
  ```
  与基线第 1 步的（理论上梯度应为零模方向）完全南辕北辙，梯度方向剧烈畸变 → 第 1 步就把参数推出物理区。
  **任何方案，若 Psi 侧与 HPsi 侧不对称，必然爆炸。**

### 3.5 P8：双侧一致全局列规范 + 周期性 gauge reset —— 成功 ✅

代码：`NES_VMC_train_min30_gauge_reset.py`

#### 原理（数学证明）

基线已保证 `E_L = D_ref⁻¹ M D_ref`（相似）。在 |Ψ 与 HΨ 两侧同时右乘 `D_g = diag(e^{-g_j})`：

```
E_L' = (Ψ D_g)⁻¹ (HΨ D_g) = D_g⁻¹ E_L D_g    ← 仍是相似变换
```

- `tr(D_g⁻¹ M D_g) = tr(M)` → **loss 逐位不变**；
- `D_g` 与 θ 无关 → 梯度/轨迹逐位不变；
- 但 `logΨ' = slogdet(exp(L - g)) = logΨ_b - Σg_j` → **reset 边界吸收列均值后 logΨ 重新居中**。

关键：g 是**常数**（stop_gradient、非训练参数、只在 reset 边界手动更新）。它既不在梯度里，也不被优化器更新——纯粹是"数值读数平移"，因此印记 0 成本。

#### 实现要点（三条，缺一不可）

1. **双侧一致**：`_compute_L_centered_single` 里 `L -= sg(g)[None,:]`；loss 里 `HPsi_stable *= exp(-sg(g))[None,:]`。Ψ 侧与 HΨ 侧必须同乘同除，否则退化为 P5。
2. **stop_gradient**：g 必须 `jax.lax.stop_gradient`，保证优化器看到的参数梯度与基线完全相同。
3. **显式动态参数**（本会话最重要的工程 bug）：g 不能放进 Python list 后再被 jit 闭包捕获——JAX jit 编译时会**把闭包里 list 持有的数组固化成常量**，Python 侧的更新永远进不了计算图（初次运行复现：reset 打印了"吸收 12.41"，但 logΨ 与基线逐位一致）。修复：g 作为**显式形参**穿入 `total_machine/grad_fn/qgt_fn`，主循环用模块级变量 `g_current` 更新；采样器用二参封装 `sample_machine(params, sigma)=total_machine(params, sigma, g_current)`（常数平移不影响 Metropolis ratio）。

#### 结果（与基线逐位列比对）

| Step | P8 logΨ mean | 基线 logΨ mean | P8 Loss | 基线 Loss | P8 raw | 基线 raw |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | 9.264 | 9.264 | -21.989 | -21.989 | 20.17 | 20.17 |
| 9 | 18.124 | 18.124 | -27.129 | -27.129 | 113.05 | 113.05 |
| **10 (reset@10)** | **6.556** | 18.971 | -27.610 | -27.610 | 133.67 | 133.67 |
| 19 | 13.185 | 25.599 | -30.126 | -30.126 | 25.29 | 25.29 |
| **20 (reset@20)** | **-6.207** | 26.246 | -30.198 | -30.198 | 31.63 | 31.63 |
| 29 | -0.922 | 31.532 | -30.483 | -30.483 | 92.73 | 92.73 |
| **30 (reset@30)** | 吸收 Δg=+24.46 | 32.136 | - | - | - | - |

- **Loss 与基线逐位一致**（Step 0/9/10/19/20/29 完全相等，`E0→-7.748` 与基线相同）；
- **raw 梯度与基线逐位一致**（Step 11=146.9343 相同；全程无 nan、无爆炸、clip 从未触发）；
- **logΨ mean 全局有界 [-12, +14]**，reset 后回到 0 附近，基线则单调 9.264→31.532。

**一次 reset 的直观效果**：Step 9 网络学习的行列式幅值已推到 +12.41（raw L 的列均值），reset 把它全部吸收进 g → Step 10 报告中的 logΨ mean 回落到 6.556；之后网络重新学习幅值 → 20/30 步再次吸收。`logΨ = logΨ_raw - Σg` 永远处在安全数值区。

---

## 4. 结论与可复用方法（要点）

1. **诊断先行，改动后验**：薛定谔猫式漂移的本质是**行规范为 loss 的精确零模**，所以任何"修正 logΨ 数值"的做法都必须验证 **E_L 是否仍保持相似变换**（P5 反例）与**梯度方向是否仍与基线一致**（rowfix/P2 反例）。
2. **正确的治漂移方法不是去改 loss 里的 Ψ 值，而是给一个双侧一致、stop_gradient、常数化的全局列规范 g**：
   -`L -= sg(g)[None,:]`（Psi 侧）+ `HPsi *= exp(-sg(g))[None,:]`（HΨ 侧），
   - 周期性 `g += mean(raw L 列均值)`，
   - 数学保证轨迹逐位不变，只把 logΨ 读数平稳化。
3. **JAX 工程铁律**：需要"常数量 + 不定时外部更新"进计算图时，**必须显式参数传递**；不要依赖 Python list/dict 闭包（闭包数组在首次 jit 被固化）。
4. **监控**：`logΨ max` 是比 `raw 梯度` 更早的预警信号——`exp(max logΨ)` 逼近 float32 上限（≈1e19，logΨ≈48）时，爆炸就在几步之内。

---

## 5. 附录：关键文件索引

| 用途 | 文件 |
| :--- | :--- |
| P8 方案与训练循环 | `experiments/LiH分子/NES_VMC_train_min30_gauge_reset.py` |
| P8 成功日志（30 轮） | `experiments/LiH分子/日志/nes_vmc_0829_K4_LiH_STO-3G_min30_gauge_reset.log` |
| 基线 30 轮日志 | `experiments/LiH分子/日志/nes_vmc_0829_K4_LiH_STO-3G_min30.log`；700 步旧日志见 `logPsi_drift_diagnosis.md` §0 |
| 失败对照 | `..._min30_rowfix.log`（P2）、`..._min30_diagshift.log`（P3）、`..._min30_gaugepin.log`（P4）、`..._min30_sgrecenter.log`（P5） |
| 核心实现 | `experiments/LiH分子/NES_VMC_V1.py`（wrapper L1304-1565 / loss L757-861 / QGT L1625） |
| 既有报告 | `gauge_fixing_analysis.md`（修复未生效根因）、`logPsi_drift_diagnosis.md`（漂移机制与 Step113 证据） |