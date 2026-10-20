# 真 Matrix-Free 冻结块 NES-VMC（方案 B'：行列式自动投影）— 实现解读

- 日期：2026-10-15
- 脚本：`NES_VMC/experiments/H2分子/H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py`
- 系统：H2 / 6-31G / 8 量子比特（物理空间 2^8=16 维）/ K=6 = 冻结 4 + 可训练 2
- 结果：1500 步（1054 s）收敛，φ̂₄ 误差 0.0061 mHa / 重叠 0.99985，φ̂₅ 误差 0.0180 mHa / 重叠 0.99820，Var≈2e-6，冻结剩余重叠 ~1e-30
- 对照：精确版方案 A（`H2_631G_8Qbit_K2_frozen4_deflation_RealFFNN.py`，显式 16 维投影矩阵 F），误差 (0.0005, 0.018) mHa —— 精度基本持平

---

## 1. 与方案 A 的本质区别

方案 A（精确版）在**训练循环内部**每步都要接触 16 维全空间：

```
χ = A − F F† A          # A 是 active 列的 16 维振幅向量，F 是 (16,4) 冻结正交归一矩阵
```

即把 active 列显式投影到冻结 span 的正交补上，Ψ 是 2×2 行列式（只含 χ）。这依赖"物理空间只有 16 维、可以把整列振幅存成向量"这一小系统特权，**不是** matrix-free。

方案 B' 的目标：**训练全程不构造任何 2^N 维向量/矩阵**。做法是把冻结列和 active 列放进同一个 6×6 行列式：

```
Ψ(x) = det[ φ0 | φ1 | φ2 | φ3 | χ1 | χ2 ]      （每列在该 walker 的 6 个组态上逐点求值）
```

投影不再显式计算，而是由行列式的数学性质**自动实现**（见 §2）。

---

## 2. 数学核心：det 列变换不变性 ⇒ 自动投影

设 Φ 为冻结 4 列张成的空间，Q_Φ = I − P_Φ 为其正交补投影。对任意 active 列矩阵 C：

```
det[Φ | C] = det[Φ | Q_Φ C]
```

**证明要点**：把 C 的每一列分解 c = Qc + Φa（Φ 标准正交时 a = Φ†c）。行列式对每列是多重线性且 Φ 列重复出现时为零，故 Φa 分量对 det 的贡献恒为 0，只剩 Qc。逐点（每个组态 x）成立。

两个推论，正好覆盖 VMC 需要的两件事：

1. **采样分布自动正交**：|Ψ|² 只依赖 Q_Φ C，active 列里与冻结 span 平行的分量对波函数完全不可见——训练不会去拟合冗余方向，也**不需要**显式投影系数 a、Gram 矩阵或 ‖ψ‖² 绝对范数。
2. **梯度自动切空间投影**：∇_θ log det[Φ|C] = ∇_θ log det[Φ|Q_ΦC]，autodiff 过 `slogdet` 自动给出 ∂χ = Q_Φ ∂A 的切空间投影，无需手工修正项。

数值验证：训练后 active 列对冻结 span 的最大剩余重叠 ~1e-30（float64 机器精度），即自动投影在实现层面严格成立。

---

## 3. 冻结列如何进入训练（不载 16 维向量）

冻结列 φ_i 不以任何 16 维向量形式进入训练，而是**旧网络 + 4×4 常量矩阵**的逐点组合：

1. 载入 K=4 训练产物：
   - 参数：`data/26-10-07-19-57_history_natural_gradient_H2_molecule_K4__realFFNN.pkl` 的 `history['params'][-1]`；
   - 旋转系数：`冻结K4/近似本征态_K4_26-10-07-19-57.npz` 的 `v_rot` (4,4)，列约定 **φ_k = Σ_j ψ_j^old · v_rot[j,k]**。
2. 用旧架构 `NESTotalAnsatz_stable(8, 4, hidden=12, rngs=Rngs(11))` 重建 4 个 gauge-fixed 单列 machine（`create_single_machine_gauge_fixed`，ψ_j(HF)=1），参数作为**闭包常量**（Python list，不被任何 `jax.grad` 追踪 ⇒ 天然 stop-gradient）。
3. 训练每步对采样 walker 的 6 个组态 x_n 逐点算 `Lf[n,j] = logψ_j^old(x_n)`，再在 6×4 的小块上乘 `V_MAT=v_rot` 得到 `Φ_scaled[n,i] = e^{-s}·φ_i(x_n)`。**唯一接触过的"向量"是 6×4 块，与 2^N 无关。**

注意坑（已修）：machine 输出的是 **logψ 不是 ψ**，全空间一致性检查时必须 `np.exp` 后再比较，否则保真度断言假阳性/假阴性。

预检断言（脚本 L166-174）：冻结网络重建列 vs npz `phi_vecs` 列保真度 >0.999 才允许继续，确保 gauge 约定吻合。

---

## 4. 前向实现：`make_block_forward`（L188-230）

对单个 walker `xw (6,8)`（6 个组态 × 8 自旋轨道）：

```python
Lf[n,j] = logψ_j^old(x_n)                    # (6,4) 冻结网络，常量参数
La[n,m] = logχ_m(x_n) - g[m]                 # (6,2) active fresh 列 + 列规范 g
s = stop_gradient(max Re(Lf ∪ La))           # per-walker 数值稳定 shift
A = [ exp(Lf-s) @ v_rot | exp(La-s) ]        # (6,6) 流形矩阵 B（缩放后）
Z = [ HPf @ v_rot          | HPa · e^{-g} ]  # (6,6)，H 线性 ⇒ Hφ_i = Σ_j v_rot[j,i]·Hψ_j
logΨ = 6·s + slogdet(A) + i·argdet(A)
E_L  = solve(A, Z)                           # (6,6) 局域能量矩阵
loss = Re tr(E_L)
```

- `Ham_psi_scaled`（NES_VMC_V1.py L623）对每个组态用 `ha.get_conn_padded` 取局域连接（H2/6-31G 每组态 ≤13 个 x'），在连接组态上逐点评估网络求 Σ_{x'} H[x,x']·ψ(x')。**这是全算法唯一接触哈密顿量的地方，纯稀疏、纯逐点。**
- 冻结列的 H 作用同样逐点：HPf 是 6×4 的 Hψ_j^old 块，乘 4×4 的 v_rot 即得 Hφ。
- `valid` 掩码检查 A/Z/E_L 全有限，坏 walker 整条剔除。

---

## 5. 采样：联合 NES 行走（L338-356）

- `NESFermionHopRule(edges=ext_edges, K=6, single_size=8)`：48 bit 状态（6 副本 × 8 轨道），每副本内做 (0,1)(0,2)…(6,7) 的 Hop 移动——局域移动，无全空间枚举。
- `MetropolisSampler(hilbert=hi_ext, n_chains=16, sweep_size=20, machine_pow=1)`：
  - **machine_pow=1 按 |Ψ| 采样**（而非 |Ψ|²），配合梯度端的**重要性加权 w ∝ |Ψ|**（`importance_weights_from_logpsi`）。这是 10-06 修复的延续：防止轨迹滑向零方差组态后梯度信号消失（稀有组态的贡献 ∝ ψη 而非 ψη²，加权后恢复精确期望公式）。
- 每步 chain_length=200 → 3200 样本。

## 6. 梯度与 QGT（L239-298）

实参数 + 复输出 logΨ 的 Wirtinger 拆分（与已验证的 RealFFNN 惯例一致）：

```
∇logΨ = ∇Re(logΨ) + i·∇Im(logΨ)             # dlogpsi_one，对 active 参数
力梯度  g_O = 2·Re ⟨ tr(E_L − ⟨E_L⟩_w) · conj(∇logΨ) ⟩_w     # w 加权期望
度量    S_w = Re( 加权中心化协方差(∇logΨ) ) + diag_shift·I    # diag_shift=0.1，float64
更新    δθ = solve(S_w, g_O)  → optax.clip_by_global_norm(20) → per-param clip 0.5 → sgd lr=0.1
```

- `⟨E_L⟩_w` 是 w 加权的 6×6 矩阵均值，tr 中心化后只取迹——与方案 A 的 deflated 梯度逐项同构，只是 (machine, E_L) 换成 block bundle。
- `should_skip`：valid_ratio < 0.25 或梯度/loss 非有限时跳过该步更新（本次 1500 步 0 次触发）。
- **列规范重置**：每 10 步把 g[m] 设为 active 列 logψ 在样本上的均值（仅 active 列；冻结列 gauge 已固定在 HF）。g 全部 `stop_gradient`，不进入梯度。

## 7. Matrix-Free 审计：训练循环每一步用了什么

| 训练循环操作 | 计算对象/规模 | 是否接触 2^N=16 全空间 |
|---|---|---|
| MCMC 采样 | 48-bit walker，局域 Hop 移动 | 否 |
| `core` 前向（logΨ、E_L） | 3200 样本 × 6×6 slogdet/solve；网络逐点求值 | 否 |
| `Ham_psi_scaled` | `get_conn_padded` 每组态 ≤13 连接 | 否（稀疏局域） |
| 冻结列 φ_i | 旧网络逐点求值 + 4×4 v_rot 组合 | 否（无 16 维向量） |
| 力梯度 / QGT | vmap over 3200 样本；O 矩阵 (3200, P_active) | 否 |
| SR solve / optax | (P_active×P_active)，P_active=2×~145 | 否 |
| 列规范重置 | 样本上均值 | 否 |
| `eig(E_L_mean)` 监控 | 6×6 特征值 | 否 |

**结论：训练主算法（采样→前向→梯度→QGT→更新）没有任何 16 维向量、16×16 矩阵或全空间枚举。** 每一步的矩阵规模至多 3200×P（样本×参数）或 6×6（行列式块）。

### 全空间运算出现的位置（全部是预检/诊断/验收，不进梯度）

| 位置 | 内容 | 频率 |
|---|---|---|
| 步骤 0 预检（L141-182） | `hi.all_states()`、`H_dense`、`eigh`、oracle ν、冻结网络保真度断言 | 启动时一次 |
| 初始化预检（L301-332） | `active_basis_full` 全空间振幅 → cond(B_6列) < 1e6 才开训 | 每个 seed 一次 |
| block 诊断（L408-417） | Ba (16,2) → Bp=Ba−FF†Ba → 2×2 广义本征 λ̂45 | 每 25 步 |
| 最终验收（L449-480） | φ̂₄₅ 提取、Var、FCI/冻结重叠 | 结束时一次 |
| 绘图 | 历史曲线 | 结束时一次 |

即使把这三处全空间诊断全部删掉，训练本身照常运行（只是失去监控）；这与方案 A 形成对照——方案 A 的 F (16,4) 投影在训练循环内部，删掉就不是 deflation 了。

## 8. 结果

| level | λ̂ (Ha) | ν oracle | E(FCI) | λ̂−E (mHa) | Var | FCI 重叠 | 冻结剩余重叠 |
|---|---|---|---|---|---|---|---|
| 4 | −0.02802664 | −0.02803284 | −0.02803278 | 0.0061 | 2.8e-6 | 0.99985 | 9.8e-31 |
| 5 | −0.01448541 | −0.01450344 | −0.01450343 | 0.0180 | 1.5e-6 | 0.99820 | 6.1e-31 |

- 冻结列对 FCI 重叠 [0.9984, 0.9998, 0.9998, 0.9989] 构造上严格不变（冻结保证"不变差"，精度=冻结时刻精度）。
- 最终 6×6 `E_L_mean` 本征值一次性给出 E0–E5 全部六能级：[−1.054349, −0.957906, −0.668952, −0.551402, −0.028042, −0.014474]，前四与 FCI 差 ≤3e-6 Ha。
- Loss 收敛目标 = ΣE_ray(frozen) + Σν = −3.274893（不是 ΣE！冻结列只是近似本征态，验收必须先算 oracle 区分方法偏差与训练误差）；最终 −3.27512±2e-4，block λ̂ 与 oracle 差 ≤0.018 mHa。
- 训练稳定性：0 次 should_skip、0 次 per-param 裁剪、valid_ratio=1.0、ESS≈1200/3200、natural 梯度范数单调降至 ~2e-2。

## 9. 已知坑与现象

1. **machine 输出 logψ**：重建全空间列必须 `np.exp`；smoke 前曾因漏 exp 导致保真度断言失败。
2. **稀疏诊断曲线绘图 NaN 坑**（本次踩中）：`lam_hist` 每 25 步才有限，直接 `plot(steps, lam_arr)` 因相邻点间全 NaN 而无任何线段可画，能级曲线整条隐形。修复：每步 `eig(E_L)` 监控量画连续线 + 诊断步散点 marker 叠加。
3. **eig(E_L) 监控有采样噪声**（如 step ~450 的 −10 mHa 尖峰），验收以每 25 步 block 精确值为准。
4. **两套冻结基并存**：det 投影用网络重建的 Phi_net（逐点），block 诊断用 npz φ 的 GS 正交归一 F（16 维）；两者张成同一空间（列保真度=1.0），诊断结论对训练成立。
5. 冻结分量漂移（active 列沿 ψ+Σαφ 方向滑动）对 Ψ 不可见 ⇒ 参数上可能缓慢漂移，float64 下无害（方案 A 同款现象）。

## 10. 运行与产物

```bash
# 完整 1500 步（~18 分钟）
python H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py
# smoke
NES_N_ITER=5 python H2_631G_8Qbit_K6_frozen4_block_MatrixFree.py
```

产物（ts=26-10-15-13-36）：
- `冻结K4/近似本征态_K6冻结块_MatrixFree_26-10-15-13-36.npz`（φ̂₄₅、v_rot、lam、ν、重叠、Var）
- `data/26-10-15-13-36_history_K6_frozen4_block_MatrixFree.pkl`（steps/loss/lam_block/eig_EL/params/diag）
- `data/H2_631G_8Qbit_K6_frozen4_block_MatrixFree_26-10-15-13-36.png`（5 联图）
- `日志/26-10-15-13-36_nes_vmc_K6_frozen4_block_MatrixFree.log`

## 11. 三向对照（2026-10-16 更新：A' 已验证）

10-16 方案 A'（MC Gram 投影系数，保留 A 的 2×2 χ 行列式，把精确内积 c=F†A 换成辅助
MCMC 蒙特卡洛 Gram 估计）验证成功，B' 的定位相应更新。同冻结源、同超参、1500 步：

| 指标 | 方案 A（精确投影） | **方案 B'（本文，det 自动投影）** | 方案 A'（MC Gram 投影） |
|---|---|---|---|
| φ̂₄ 误差 / 重叠 | 0.0005 mHa / 0.99999 | **0.0061 mHa / 0.99985** | 0.0127 mHa / 0.99995 |
| φ̂₅ 误差 / 重叠 | 0.018 mHa / 0.99813 | **0.018 mHa / 0.99820** | 0.0243 mHa / 0.99751 |
| Var[E_loc] | ~4e-7 | **~2e-6** | ~1.5e-5 / 1.4e-6 |
| 冻结剩余重叠 | ~1e-32 | **~1e-30** | ~1e-34 |
| 1500 步耗时 | 167 s（非 matrix-free） | **1054 s** | 608.2 s |
| 行列式 / walker | 2×2 / 16 bit | **6×6 / 48 bit** | 2×2 / 16 bit |
| 投影严格性 | 精确 | **严格（det 不变性，零估计）** | MC 估计（相对误差 ~1e-4） |
| 维数无关 | ❌ | ✅ | ✅ |

- **B' 的优势**：投影由行列式数学性质严格实现，无任何估计噪声、无辅助采样器、无尺度管理，
  实现与诊断最干净；φ̂₅ 精度与 A 完全持平。
- **B' 的代价**：每个 walker 前向要跑全部 6 条网络（4 冻结 + 2 active）+ 6×6 slogdet/solve，
  采样空间 48 bit；A' 用 M=1024 的 aux 批换回 2×2/16-bit，成本降 1.73×。
- **家族定位结论**：matrix-free 主线首选 A'（便宜、显式可审计），B' 作为"零估计噪声"的
  严格对照与小系统基准保留。详见 `验证报告/1016-方案A-prime真MatrixFree-三向对照报告.md`
  与 `解读文档/1016-方案A-prime-MC-Gram投影真MatrixFree-实现解读.md`。
