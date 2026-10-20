# 方案 A'：MC Gram 投影系数的真 Matrix-Free deflation NES-VMC — 实现解读

- 日期：2026-10-16
- 工具箱：`NES_VMC/experiments/H2分子/NES_VMC_deflation_mf.py`
- 训练脚本：`NES_VMC/experiments/H2分子/H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py`
- 单测：`NES_VMC/experiments/H2分子/test_NES_VMC_deflation_mf.py`
- 计划：`NES_VMC/计划文档/1016-方案A转MatrixFree-A-prime-计划.md`
- 验证报告：`NES_VMC/验证报告/1016-方案A-prime真MatrixFree-三向对照报告.md`
- 结果：1500 步 / 608.2 s，φ̂₄ 误差 **0.0127 mHa** / 重叠 **0.99995**，φ̂₅ 误差 **0.0243 mHa** / 重叠 **0.99751**，Var 1.5e-5 / 1.4e-6，冻结剩余 ~1e-34，0 skip；质量四项验收全过（唯预设 500 s 耗时线未达）
- 对照：方案 A（精确投影，167 s，0.0005/0.018 mHa）、方案 B'（行列式自动投影，1054 s，0.0061/0.018 mHa）

---

## 1. 三个方案一张表：A' 改了什么

| | 方案 A（精确） | 方案 B'（隐式） | **方案 A'（MC 显式）** |
|---|---|---|---|
| 拟设行列式 | 2×2：det[χ₁\|χ₂] | 6×6：det[φ₀..φ₃\|χ₁χ₂] | 2×2：det[χ̃₁\|χ̃₂] |
| 投影方式 | 显式 c=F†A（全空间内积） | det 列变换不变性自动投影 | 显式 c=Ĝ⁻¹M̂（**蒙特卡洛 Gram**） |
| 冻结列表示 | npz 里的 16×4 矩阵 F | 旧网络逐点求值 + v_rot | 旧网络逐点求值 + v_rot |
| walker | 2 组态（16 bit） | 6 组态（48 bit） | 2 组态（16 bit） |
| 训练回路接触 2^N？ | **是** | 否 | 否 |
| 每步额外成本 | 16 维矩阵乘 | 每 walker 多跑 4 条冻结网络 | 每步 M=1024 个单组态 aux 前向 |

方案 A 唯一的"作弊"是投影系数 $c=F^\dagger A$ 用了全空间内积（依赖 16 维向量存储）。
A' 只替换这一个数学对象：**把 Gram 内积蒙特卡洛化**，其余结构（2×2 χ 行列式、实参数
Wirtinger 梯度、Re(S) QGT、SR、重要性加权）与 A 逐行同构。

## 2. 数学核心一：Z 抵消引理（为什么不需要知道归一化）

取参考密度（未归一）

$$q(y)=\sum_{i=0}^{3}|\tilde\varphi_i(y)|^2+\varepsilon\sum_{m}\big|\tilde\psi_m(y)\big|^2,\qquad p(y)=q(y)/Z,\ Z=\text{未知},$$

$y_k\sim p$ 由辅助 MCMC 产生（单组态，8 bit，`machine_pow=1` 按 $\sqrt q$ 行走）。
同一批样本、用同一个 $q$ 估计的一切 Gram 量都带同一个 $1/Z$ 因子：

$$\hat G=\langle\tilde\Phi|\tilde\Phi\rangle/Z,\quad \hat M=\langle\tilde\Phi|\tilde\Psi\rangle/Z,\quad \hat{HG}=\langle\tilde\Phi|H\tilde\Phi\rangle/Z,\quad \hat{M_2}=\langle\tilde\Phi|H\tilde\Psi\rangle/Z$$

（逐样本估计式 $\hat M_{ij}=\frac1k\sum_k \tilde\varphi_i^*(y_k)\tilde\psi_j(y_k)/q(y_k)$，$\hat G$ 同理）。
于是投影系数

$$c=\hat G^{-1}\hat M,\qquad c_2=\hat G^{-1}\big(\hat{M_2}-\hat{HG}\,c\big)$$

中 $Z$ **严格抵消**——重要性权重里出现的正是 $1/q$，与 Gram 定义里的 $1/Z$ 相乘自动补齐归一化。
这就是 `estimate_grams` 里 `wi = 1/(q·M)` 一行注释的由来。$c_2$ 是"二次投影"
（project→H→project，对应 Z 矩阵里 $(H\Psi̇-H\Phi̇ c)-\Phi̇ c_2$），与精确版 bundle 同构。

$\hat G$ 加 `gram_ridge=1e-12` 对角正则防奇异。

## 3. 数学核心二：尺度管理与 c 的同步变换（防漂移爆炸）

0917/方案 A 的教训：冻结分量漂移方向（$\psi+\sum\alpha\varphi$ 对 $\Psi_F$ 不可见）会让
振幅无界增长。A' 显式引入两个实尺度并每步校准：

$$\tilde\varphi_i=\varphi_i\cdot e^{-f},\qquad \tilde\psi_m=\psi_m\cdot e^{-g_m}$$

- 每步 aux 批给出 $\max_y\sum_i|\tilde\varphi_i|^2$ 与 $\max_y|\tilde\psi_m|$（stop-grad），
  `update_scales` 做 EMA：$f\mathrel{+}=\tfrac{\beta}{2}\ln(\max|\tilde\Phi|^2\text{批})$、
  $g_m\mathrel{+}=\beta\ln\max|\tilde\psi_m|$，$\beta=0.9$（首步 $\beta=1$ 即 fresh 归一）。
  目标：$q$ 的两项同为 $O(1)$，$\tilde\Psi/A_\chi$ 元素 $O(1)$。
- 尺度一变，已算出的 $c,c_2$ 必须同步变换才能保持 $\tilde\chi=\tilde\Psi-\tilde\Phi c$ 方向不变：
  $$c[:,m]\ \leftarrow\ c[:,m]\cdot e^{\Delta f-\Delta g_m}$$
  （训练脚本第 2 段 `trans = np.exp(d_f - d_g)`）。这是恒等变形：$\tilde\Phi e^{\Delta f}\cdot c\,e^{\Delta f-\Delta g}=\tilde\Phi c\,e^{-\Delta g}$，
  与 $\tilde\Psi e^{-\Delta g}$ 匹配，$E_L$ 严格不变。

det 的列缩放 $e^{-(g_0+g_1)}$ 是实常量（stop-grad），不影响 $E_L,\nabla\log\Psi_F$，
只使采样分布差一常量因子（采样器自动归一）。

## 4. 每步训练回路（`H2_631G_8Qbit_K2_frozen4_deflation_MatrixFree.py` 主循环）

```
1. aux 采样：8 链×128=1024 个单组态 ~ q，machine_pow=1
   → estimate_grams → c_mc, c2_mc, max_phi2, max_psi, 诊断(w_ess, max_w_ratio)
   （grams 非有限 ⇒ 整步跳过）
2. EMA 尺度更新 f,g + c/c2 同步变换（§3）
3. 主采样：hi_ext 联合 2-walker，16 链×200，machine_pow=1 按 |Ψ_F|
   machine 输入 pytree = (total_params, c, c2, f, g)
4. grad_fn（Wirtinger ∇Re+i∇Im + w∝|Ψ_F| 重要性加权）→ per-param clip 0.5
   → qgt_fn（Re(S)，权重同上）→ solve(S+0.1I, g) → optax sgd lr=0.1
5. 全空间监控（仅诊断）：精确重投影 χ、c_err、冻结剩余、λ_sub、max|A|
```

关键实现约束（netket/jax 坑）：**c、c2、f、g 必须作为显式参数传入 jit 函数**——
若走闭包会被 `jax.jit` 焊死成常量，每步换新值需要重新编译。`NES_VMC_deflation_mf.py`
的 `grad_fn/qgt_fn/machine_batch/forward` 签名都带全这四个量。

稠密 `H_dense`、全空间振幅 `amplitudes_full` 只出现在 Step0 预检、每步监控、最终验收
三处（小系统特权，诊断专用）；训练四步全部逐点、维数无关。

## 5. 冻结列的 matrix-free 表示（继承 B'）

- 旧 K4 参数：`data/26-10-07-19-57_history_..._realFFNN.pkl` 的 `params[-1]`；
  列约定 $\varphi_k=\sum_j\psi_j^{old}\cdot v_{rot}[j,k]$（npz `v_rot`）。
- 重建 4 个 gauge-fixed 旧 machine（`create_single_machine_gauge_fixed`），参数为
  常量 list，不被 grad 追踪 ⇒ 天然 stop-gradient。
- 预检断言：重建列 vs npz `phi_vecs` 保真度 >0.999（实测 =1.00000000）。
  注意 machine 输出 **logψ**，比较前必须 `np.exp`。
- $H\varphi_i=\sum_j v_{rot}[j,i]\,H\psi_j^{old}$（H 线性），用 `Ham_psi_scaled(shift=f)` 逐点算。

## 6. 单元测试结论（`test_NES_VMC_deflation_mf.py`，全部通过）

1. **收敛阶**：$\|c^{MC}-c^{exact}\|$ 随 $M$ 的 log-log 斜率 **-0.476 / -0.495**（理论 -0.5）✓
2. **χ̃ 精度**：M=4096 时相对误差 1.2e-2；绝对 1e-3 需 M~1e6 ⇒ 断言改为**相对 ≤5e-2**
   （对计划 §6 阈值的已记录偏离，正式运行按相对口径复核 ~1e-4）
3. **集成**：主采样/grad/qgt 全链路 valid=1。

## 7. 正式运行观察到的现象与解读

- **c_err 绝对值"爆炸"是假象**：从 2.6e-3（早期）涨到 ~2.9（末段），但同期 max|A| 从 3
  漂到 1.2e4（冻结分量漂移，float64 无害，方案 A 同款现象）；**相对误差稳定 ~1e-4**，
  且冻结剩余、能量、重叠全部无恙。看 c_err 必须除以 max|A|。
- **aux_ess 长期 20–50/1024、aux_maxw ~140**：active 列收敛后 q 尾部权重变重。
  本轮未构成危害（Gram 噪声没进 natural 梯度噪底，末段 natural=0.002–0.009），
  但**外推大体系需复查此项**（缓解：升 M 或 ε 自适应）。
- **5 次 per-param 裁剪全在末段 90 步**（scale 0.69–0.95，轻微）；0 skip；Loss 单调无震荡
  → 计划 §5.1 担心的"Gram 噪声地板放大"未发生。
- **TIMING 分量不可信**：`jax.block_until_ready(0)` 不阻塞实际计算，grad/qgt 显示 2ms/77ms
  是异步派发计时；成本归因以总墙钟 608.2 s（0.405 s/步）为准，实测最大单项是主采样 ~218 ms/步。
- **耗时 608 s > 500 s 线**：aux 采样 + 编译启动吃掉预算；相对结论不变——比 B' 快 1.73×。

## 8. 结论与适用边界

1. A' 证明**显式投影的 MC 化在 VMC 动力学下可靠**（计划 §5.5 失败预案不启用）。
2. matrix-free 家族现有两成员：**A'（首选）**——2×2、16-bit walker、投影系数显式可审计、
   成本最低；B'——零内积估计、严格自动投影，但 6×6/48-bit 更贵。
3. 精度排序 A < B' < A'（0.0005 < 0.0061 < 0.0127 mHa）符合"投影噪声"直觉，
   但三者均远低于化学精度 1.6 mHa，排序无实际意义差异。
4. 下一步：A' 跑 LiH（$D>16$，方案 A 无法运行），作 matrix-free 家族首个非小系统证据；
   届时 c_err 监控口径改为相对值、并复核 aux_ess。

## 9. 产物清单（ts=26-10-16-02-23）

- npz：`冻结K4/近似本征态_K4冻结_K2扩展_MatrixFree_26-10-16-02-23.npz`
- history：`data/26-10-16-02-23_history_K2_frozen4_deflation_MatrixFree.pkl`
- 日志：`日志/26-10-16-02-23_nes_vmc_K2_frozen4_deflation_MatrixFree.log`
- 6 联图：`冻结K4/训练曲线_K2冻结4_MatrixFree_26-10-16-02-23.png`
- 环境变量：`NES_N_ITER / NES_AUX_M(1024) / NES_SCALE_BETA(0.9) / NES_INIT_SEED / NES_TIMING`
