# 量子增强 NES-VMC（QE-NES-VMC）H2/6-31G K=4 — 算法总结与实现解读

> 日期：2026-10-18
> 代码目录：`NES_VMC/experiments/H2分子/Quantum Enhanced NES-VMC/`
> 算法蓝本：`Quantum Enhanced NES-VMC/Quantum_Enhanced_NES_VMC方案总结.md`（§3 方案 A、§4 联合行列式、§9 行列式微分恒等式、§12 实验路线第 2 项）
> 参考实现：`H2_631G_8Qbit_K4_Ansatz_A_RealFFNN.ipynb`（K4 基线）、`量子-经典混合式 Ansatz/FFNN振幅_PQC相位混合版-VMC.py`（已验证 1.971 mHa 的混合单态）、`自回归采样器/ar_nqs.py`

---

## 1. 一句话概括

把 NES-VMC 的每个单态波函数从「纯 FFNN」升级为「FFNN 振幅 × PQC 相位」的量子增强形式
ψ_i(x) = φ_i^NN(x) · φ_i^QC(x)，实现为 **logψ_i = logA_i^FFNN + i·φ_i^PQC**（实参数、复输出），
K=4 个独立单态经行列式 𝒟(X)=det[ψ_i(x_j)] 组装成反对称多体态，训练引擎完整沿用已验证的
「扩展空间 Metropolis + machine_pow=1 重要性加权 + 实参数 Wirtinger SR + 双侧一致列规范/gauge reset」框架。

## 2. 算法结构

### 2.1 单态混合 Ansatz（方案 A：各态独立 FFNN+PQC）

```
σ(8 bit) → backbone1: Linear(8→12) tanh → backbone2: Linear(12→12) tanh
振幅头：logA = Linear(12→1)                     # 经典，读 full hidden
相位支路：z = tanh(Linear(12→4))                # 压缩到 4 维瓶颈
相位头：φ = W·PQC(z) + b                        # PennyLane PQC + 线性读出
输出：logψ = logA + i·φ                          # 实参数、复输出
```

- **PQC**：`default.qubit` 4 qubits；编码层 RX(π·z_i)；2 层变分（CNOT 链 0-1/1-2/2-3 + 每比特 RY/RZ，(2,8) 个变分参数）；特征读出 4 个 ⟨Z_i⟩ → Linear(4→1) 得相位 φ。
- QNode 放在**模块级**（历史坑：`nnx.split` 不能携带设备对象）；`diff_method="backprop"`，jax.grad 直通。
- 每个单态 350 参数（backbone 108+156、head_logA 13、phase_proj 52、qnn 16、readout 5），K=4 共 **1400** 参数。

### 2.2 联合行列式与稳定化

walker σ 为 K×8=32 bit（hi_ext = hi**4），矩阵 L[i,j] = ψ_j(x_i)（行=walker 内 K 个组态、列=单态），
logΨ = slogdet(exp(L−shift)) + K·shift，shift=max Re(L)（stop_gradient）。接口与 `NESTotalAnsatz_stable` 完全一致，
因此 `NES_VMC_tool.create_gauge_reset_total_machines` / `NES_loss_energy_stable_gauge` / `Ham_Psi_scaled` 全部直接复用。

### 2.3 行列式微分恒等式（§9 验证）

∂log𝒟 = Tr[A⁻¹ ∂A]（A 为 K×K 矩阵）。单元测试将逐列 ∂logψ_j 的 autodiff 梯度与恒等式重建的
全参数梯度逐元素对比：**rel_err = 1.31e-15**（1400 维），确认梯度链路正确。

### 2.4 训练引擎（§12 实验路线第 2 项：保留原采样策略）

- **采样**：`NESFermionHopRule`（扩展空间单粒子跳跃，保持每副本 (1,1) 粒子数）+ `MetropolisSampler`，16 chains × 200 samples，machine_pow=1 按 |Ψ| 采样。
- **重要性加权**：w ∝ |Ψ| 修正回 |Ψ|² 期望（防零方差冻结，见 1006 文档）；aux 新增 **ESS=1/Σw̃²** 与 max w/avg 监控（§7）。
- **梯度**：实参数 Wirtinger 拆 ∇Re + i∇Im；力梯度带因子 2：∇E = 2Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩。
- **SR**：度量 Re(S) + diag_shift=0.1（float64 svd 路线，规避 netket 默认 SR 的 jax 分派冲突）。
- **列规范**：双侧一致（Ψ 侧与 HΨ 侧同减 e^{-g}）+ 每 10 步 gauge reset，与 K4 RealFFNN 基线一致。
- 梯度保护：per-param clip 0.5（QGT 前）、总范数告警阈值 5000。

## 3. 文件清单

| 文件 | 角色 |
|---|---|
| `qe_nes_ansatz.py` | `QESingleStateAnsatz`（FFNN 振幅 + PQC 相位单态）与 `QETotalAnsatz`（K 行列式总拟设，输出与 NESTotalAnsatz_stable 一致） |
| `H2_631G_K4_QE_NES_VMC.py` | 训练脚本：实参数兼容层（grad/qgt/loss，含 ESS）+ 主循环 + 曲线/history/params 产物；支持 `k=v` argv 覆盖 |
| `test_qe_nes_vmc.py` | 18 项单元测试（见 §4.1） |
| `train_main.log` | 正式 300 步训练日志 |
| `QE_NES_VMC_K4_main.png / _history.pkl / _params.npz` | 正式训练产物（tag=`_main`） |
| `QE_NES_VMC_K4_smoke.*` | 冒烟（20 步）产物 |

## 4. 验证与结果

### 4.1 单元测试：18/18 全部通过

单态前向形状（1D→标量、2D→(K,)）、total_machine 四种输入形状及一致性、恒等式 Tr[A⁻¹∂A]（1.31e-15）、
PQC 变分参数梯度非零（|g_qnn|=0.98，相位支路真的在训练）、Ham_Psi_scaled/loss/E_L 形状与有限性、
grad_fn 全有限 + ESS 报告、QGT 对称且正则后正定（λmin=diag_shift）、SR 解有限、
采样器输出形状与每副本粒子数约束 (1,1)。

### 4.2 冒烟（20 步）

3.21 s/步，梯度全有限、valid=1.00，E0 误差 544→7 mHa。

### 4.3 正式训练（300 步，600.7s，2.0 s/步）

lr=0.1 diag_shift=0.1 seed=11 16×200 样本，与基线 notebook 配置逐项对齐。误差演化（mHa，E0/E1/E2/E3）：

| Step | Loss | E0 | E1 | E2 | E3 | cond(Ψ) | ‖g‖ |
|---|---|---|---|---|---|---|---|
| 0 | 0.789 | 512.0 | 1037.0 | 942.0 | 1530.1 | 9.6 | 0 |
| 10 | -1.790 | 34.1 | 443.2 | 215.9 | 749.9 | 58 | 2.3 |
| 50 | -3.204 | **3.0** | 5.9 | 9.9 | 9.7 | 290 | 9.1 |
| 100 | -3.214 | 3.2 | 3.1 | 7.4 | 4.5 | 89 | 9.7 |
| 200 | -3.218 | 5.2 | 1.5 | 4.1 | 3.4 | 260 | 10.8 |
| 299 | -3.221 | 6.4 | **1.5** | **1.9** | **2.3** | 360 | 11.7 |

最终：E0=-1.04795499（6.39 mHa）、E1=-0.95640354（1.50）、E2=-0.66702498（1.93）、E3=-0.54913932（2.26）；
Σ误差=12.1 mHa；Loss=-3.220523（理论 ΣE=-3.23260738）。全程 valid=1.00、ESS≈1300–1400/3200、max w/avg≈2.4–2.7、无 NaN、无梯度爆炸。

### 4.4 与 K4 RealFFNN 基线对照

| 方案 | 300 步各级误差 | 备注 |
|---|---|---|
| 基线 RealFFNN（同配置） | ~0.01–0.02 mHa | Loss 收敛到理论值 |
| **QE（本文）** | 6.4 / 1.5 / 2.0 / 2.3 mHa | E1–E3 进入化学精度内；E0 停在 mHa 平台 |

## 5. 观察与结论

1. **算法链路正确**：恒等式、梯度、QGT、采样约束全部通过；PQC 相位支路确实在参与训练（变分参数梯度非零）。
2. **收敛形态**：E1–E3 稳定收敛到 ~2 mHa；**E0 在 step 50 达到最优 3.0 mHa 后缓慢回漂到 ~6 mHa**，与列规范软方向漂移特征吻合——‖g‖ 单调涨到 11.7、cond(Ψ) 升到 10²–10³（基线诊断报告 0917 的同一机制，float64 下无害但反映参数在平坦方向游走）。
3. **相对纯 FFNN 基线**：本实现未复现基线的 0.01 mHa 级精度。可能因素：PQC 相位支路引入额外噪声/瓶颈（12→4 压缩 + 4 qubit 读出维度有限），且相位头对 E0 的振幅精度无直接贡献；混合单态此前在「单态 VMC」上验证过 1.971 mHa，量级与本次 NES 结果一致。
4. **Matrix-Free 合规**：主算法全程无 16 维全空间枚举，仅 FCI 基准与误差监控使用精确值（诊断用途）。

## 6. 复现命令

```bash
cd "NES_VMC/experiments/H2分子/Quantum Enhanced NES-VMC"
PY=/opt/miniconda3/envs/Netket/bin/python

# 单元测试（18 项）
PYTHONUNBUFFERED=1 $PY test_qe_nes_vmc.py

# 冒烟（20 步）
PYTHONUNBUFFERED=1 $PY H2_631G_K4_QE_NES_VMC.py n_iter=20 n_chains=8 samples_per_chain=50 tag=_smoke

# 正式（300 步，约 10 分钟）
PYTHONUNBUFFERED=1 $PY H2_631G_K4_QE_NES_VMC.py tag=_main
```

任意 `k=v` 可覆盖超参（lr、diag_shift、seed、n_iter、hidden、layers、pqc_scale、reset_period 等）。

## 7. 下一步建议

- **E0 回漂治理**：diag_shift 自适应（η·λ_max）或对 gauge ‖g‖ 加弱约束拉回列均值（0917 诊断报告的修复优先级）；亦可对 E0 取 step 50 早停对照。
- **消融**：`pqc_scale=0`（相位头退化为纯 FFNN）应复现基线精度，可分离「PQC 相位是否有害/有益」；以及 phase_proj 瓶颈宽度 4→8。
- **延长训练**：1000 步观察 E0 是否继续回漂或重新下降。
