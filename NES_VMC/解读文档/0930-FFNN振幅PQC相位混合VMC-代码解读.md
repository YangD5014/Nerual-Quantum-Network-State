# FFNN 振幅头 + PennyLane PQC 相位头 混合式 VMC — 代码解读

日期：2026-09-30
任务：把 `FFNN振幅相位分离版-VMC-3.ipynb` 中的相位头 `self.head_phi = nnx.Linear(Qbits, 1, ...)` 替换为 PennyLane 包装的 PQC，参考 `QNN-VMC.ipynb`，要求 VMC 400 步迭代收敛接近 FCI。
结果：**400 步收敛到 1.971 mHa（相对误差 0.1869%），能量仍在下降，任务成功。**

代码文件：

- 脚本：`NES_VMC/experiments/H2分子/量子-经典混合式 Ansatz/FFNN振幅_PQC相位混合版-VMC.py`
- 复现 notebook：`NES_VMC/experiments/H2分子/量子-经典混合式 Ansatz/FFNN振幅_PQC相位混合版-VMC.ipynb`
- 训练曲线：`NES_VMC/experiments/H2分子/量子-经典混合式 Ansatz/FFNN振幅_PQC相位混合版-VMC.png`

---

## 1. 整体结构

系统：H2 / 6-31G，4 个自旋轨道（σ ∈ {0,1}⁴，物理空间 16 维），FCI E0 = -1.05434745 Ha，HF = -0.99749729 Ha。

Ansatz 为实参数、复输出波函数 logψ = logA + i·φ，两条支路共享一个经典骨干：

```text
σ(4) → Linear(4→12) → tanh → Linear(12→12) → tanh → h
   ├─ 振幅支路（经典）： logA = Linear(12→1)(h)
   └─ 相位支路（量子）： z = tanh(Linear(12→4)(h))
                        f = PQC(z) ∈ ℝ⁴      ← PennyLane 4-qubit 线路
                        φ = Linear(4→1)(f)    ← 经典线性读出
logψ = logA + i·φ
```

与 VMC-3 的唯一改动点：`head_phi`（经典 Linear(Qbits→1)）→ PQC + readout；振幅头 `head_logA` 从"读 4 维瓶颈 z"改为"读完整 hidden h"（这是收敛成功的关键，见 §4）。

## 2. PennyLane PQC 部分（脚本 §1）

```python
_DEV = qml.device("default.qubit", wires=4)
PQC_NODE = qml.QNode(pqc_circuit, _DEV, interface="jax", diff_method="backprop")
```

线路结构（`pqc_circuit`，L=2 层）：

| 层 | 内容 |
|---|---|
| 编码层 | 每比特 `RX(π·z_i)`，把经典特征 z_i ∈ [-1,1] 映射为旋转角 |
| 变分层 ×L | CNOT 链 (0-1, 1-2, 2-3) 纠缠 + 每比特 `RY(a)`/`RZ(b)` |
| 测量 | 4 个特征 `⟨Z_0⟩…⟨Z_3⟩`（各为 (B,) 批量实数） |

三个设计要点：

1. **QNode 放模块级、不放 nnx.Module 内**。`nnx.split(model)` 会把模块属性全部拆进状态树，设备/QNode 对象不可序列化——所以 `PQC_NODE` 是全局常量，`qnn_params`（形状 (L, 2·4) = (2,8)）才是 nnx 参数，`__call__` 里显式传入。
2. **批量接口**。PennyLane 的 `qml.RX(h[:, i] * np.pi, ...)` 用切片参数天然支持 batch：一次传入 (B,4) 特征，返回 4 个 (B,) 期望值，`jnp.stack(feats, axis=1)` 拼成 (B,4)。VMC 里 local energy 的 `get_conn_padded` 返回 3D eta (N, n_conn, size)，必须先展平成 2D 再进 PQC（见 §3 引擎）。
3. **梯度直通**。`interface="jax"` + `diff_method="backprop"`（default.qubit 态矢量反传，16 维希尔伯特空间极小），使 `jax.grad` 可以对 `qnn_params` 和上游经典层参数精确求导，无需参数移位规则。

`qnn_params` 初始化为 `uniform × pqc_scale(0.1) × π`——继承 0925 进度文档的教训：PQC 参数小初始化（接近"纯编码层"起点）明显优于大初始化（scale=0.5 时平台期更早更高）。

## 3. VMC 引擎（脚本 §3，沿用已验证框架）

全部为实参数，绕开 netket/jax 的 `holomorphic=True` 限制：

- **∇logψ 拆分**：`_grad_logpsi_real_params` 对每个样本分别 `jax.grad(machine(...).real)` 和 `.imag`，再组合 `r + 1j·i`。
- **force 梯度**：`forces_expect_hermitian` 计算 ∇⟨E⟩ = 2·Re⟨(E_loc−⟨E⟩)(∇logψ)*⟩_w，权重 w = |ψ|（重要性加权）。
- **采样防冻结**：`MetropolisSampler(machine_pow=1)` 按 |ψ| 采样 + 加权 w ∝ |ψ|。这是 0925 结论里最关键的修复：若按 |ψ|² 采样，轨迹一旦滑向 HF 本征态会零方差冻结（梯度严格为 0）；按 |ψ| 采样使稀有组态的梯度信号 ∝ ψη 而非 ψη²。
- **QGT/自然梯度**：`compute_qgt` 构造 Re(S)（实参数下度量必须取实部，不能复数 solve 后丢虚部），加 `diag_shift=0.01` 正则后 `jnp.linalg.solve` 得自然梯度，`optax.sgd` 应用。
- **ravel 一致性**：注意 `compute_qgt` 返回的 `unravel_fn` 来自批量 pytree，不能用于还原单样本 grad 的展平结果——`run()` 里对 `grad` 自己 `ravel_pytree` 并用其专属 `grad_unravel_fn`（开发中踩过的 `ValueError: Sum of sizes ...` 即源于此）。
- **收敛判据**：`exact_energy()` 对全空间 16 组态做精确求和（16 维太小，无需采样），每 50 步打印，消除采样噪声对"是否接近 FCI"判断的干扰。

## 4. 调试历程：配置 A 失败 → 配置 B 成功

| 配置 | 振幅头输入 | 相位头 | 400 步结果 |
|---|---|---|---|
| A（最初实现） | 4 维瓶颈 z（与相位支路共享） | PQC | 卡在 **56.9 mHa**（≈HF 盆地） |
| B（最终方案） | 完整 hidden h（12 维） | PQC | **1.971 mHa**，仍在下降 |

根因：配置 A 中 `logA` 与 `φ` 都只能通过 4 维 `z = tanh(Linear(h→4))` 获取信息，振幅表达力被瓶颈截断——HF 附近波函数需要 16 个独立振幅值，4 维 tanh 瓶颈无法分辨。修复即把振幅头改回读完整 hidden，让 PQC 只承担相位支路（相位对表达力要求低，4 qubit 足够）。

对照历史结果：

| 方案 | 步数 | 精确误差 |
|---|---|---|
| 纯 PQC（0925 QNN振幅相位分离版） | 800 | 6.9–8.0 mHa（平台期） |
| 实参数 FFNN 双头（VMC-1） | 400 | 3.5 mHa |
| **FFNN 振幅 + PQC 相位（本方案）** | **400** | **1.971 mHa** |

本方案是目前该体系混合式 Ansatz 的最好结果，且 400 步时能量仍单调下降（Step 300: 2.587 → Step 350: 2.357 → Step 399: 1.971 mHa）。

## 5. 运行方式与实测数据

```bash
# 脚本（支持全部超参 CLI）
/opt/miniconda3/envs/Netket/bin/python FFNN振幅_PQC相位混合版-VMC.py \
    --n-iter 400 --seed 21 --lr 0.1 --layers 2 --hidden 12 \
    --pqc-scale 0.1 --diag-shift 0.01

# notebook：依次执行全部 cell（末 cell 即 400 步 + 绘图）
```

实测（seed=21，默认超参，150 chains × sweep 32 × chain 20 = 3000 样本/步）：

```text
Step 399 | E_sample: -1.05226164 ± 0.001154 | E_exact: -1.05237680 | 精确误差: 1.971 mHa
FCI 基准：-1.05434745 Ha
绝对误差：1.9706 mHa | 相对误差：0.1869%
用时：78.5 s
```

采样能量与全空间精确能量一致（差 <0.1 mHa），说明收敛瓶颈不在采样/梯度，而在优化步数——继续训练误差仍会下降。

## 6. notebook 说明

`FFNN振幅_PQC相位混合版-VMC.ipynb` 与 py 同源（由脚本按分节标记切分生成），共 12 个 cell（6 markdown + 6 code），分节：

1. imports + 路径（notebook 环境无 `__file__`，已加 `os.getcwd()` 回退逻辑）
2. §1 PennyLane PQC
3. §2 Ansatz
4. §3 VMC 引擎
5. §4 run() 训练
6. §5 执行 400 步 + 绘图（末 cell 直接调用 `run(n_iter=400, ...)`，无 argparse）

已通过端到端冒烟验证（n_iter=3 全 cell 执行）与完整 400 步复现。

## 7. 关键经验（复用清单）

1. **PennyLane QNode 不要放进 nnx.Module 属性**——设备对象会破坏 `nnx.split/merge`；参数用 `nnx.Param` 单独持有，QNode 作模块级常量。
2. **`get_conn_padded` 的 3D eta 必须先展平再进 PQC**（PennyLane 切片批量只支持单批维），算完再 reshape 回去。
3. **混合式 Ansatz 中经典头与量子头不要共享过窄瓶颈**——振幅支路保留完整表达力，量子支路负责相位即可。
4. **PQC 变分参数小初始化（pqc_scale=0.1）** 延续 0925 结论有效。
5. 实参数复输出 VMC 三件套不变：Wirtinger 拆分梯度、Re(S) 度量、machine_pow=1 + 重要性加权防零方差冻结。
