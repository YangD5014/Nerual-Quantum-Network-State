# 诊断报告：NES-VMC LiH (K=4) 训练中 `logΨ mean` 持续单调上升现象

> 实验：`experiments/LiH分子/LiH_molecule_K4_STO-3G.ipynb`
> 日志：`experiments/LiH分子/日志/nes_vmc_0701_K4_LiH_STO-3G.log` (700 步)
> 关键代码：`experiments/LiH分子/NES_VMC.py::create_gauge_fixed_total_machines`、`functions.py::nes_vmc_gradient_stable`、`models.py::NESTotalAnsatz_stable`

---

## 0. 现象速览

| 监控量 | Step 0 | Step 50 | Step 100 | Step 113 | Step 200 | Step 700 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `logΨ mean` (实部) | 9.264 | 33.986 | 46.675 | **64.357** | 64.070 | 67.294 |
| `min(logΨ)` | 3.413 | 27.163 | 43.453 | 44.349 | 61.154 | 62.976 |
| `max(logΨ)` | 11.034 | 34.520 | 46.948 | 64.578 | 64.278 | 67.352 |
| E0 (Ha) | -5.9017 | -7.7962 | -7.8097 | **-7.7699** | -7.8375 | -7.8629 |
| Loss | -21.99 | -30.82 | -30.89 | **-30.73** | -30.88 | -30.90 |
| `cond(Ψ)` | 11.5 | 7.4 | 9.1 | **236** | 11.0 | 9.97 |
| `‖∇raw‖` | 20.17 | 29.55 | 92.01 | **1393.5** | 2.96 | 0.05 |

**观察结论：**

1. **`logΨ mean` 在 700 步内单调地从 9.26 漂移到 67.29**，跨度 ~58 个自然单位，没有回落迹象。
2. `min` 始终比 `mean` 低约 4，`max` 与 `mean` 几乎重合 → 所有 walker 的 `logΨ` 几乎被"平移"同一常数（典型 row-shift 形态）。
3. 能量 E0 在 ~Step 100 之后稳定在 -7.8629，**最终非常接近 FCI 精确值 -7.8636**，说明漂移在收敛前并没有"毁掉"物理结果。
4. **Step 113 出现一次显著的"gauge 模式爆发"**：`‖∇raw‖` 从 ~90 跳到 1393.5，`cond(Ψ)` 从 ~10 跳到 236，`E0` 从 -7.83 退化到 -7.77（且 E0/E2/E3 都长出 10⁻² 量级的虚部）。这是漂移累积触顶的标志事件。

---

## 1. 根因：列规范 (column gauge) 修了，行规范 (row gauge) 没修

### 1.1 NES 行列式 ansatz 的双重规范自由度

`NESTotalAnsatz_stable` 把 K 个 `SingleStateAnsatz` 的输出拼成 L 矩阵：

```
L[i, j] = log ψ_j(x_i)
Ψ[i, j] = exp(L[i, j])
log Ψ(X) = log|det Ψ| + i·arg(det Ψ)
```

对这个矩阵有两类"不影响物理量"的乘法规范自由度：

| 规范 | 变换 | 对 `L[i,j]` 的效果 | 对 `log|det Ψ|` 的效果 | 对 loss `tr(E_L)` 的效果 |
| :--- | :--- | :--- | :--- | :--- |
| **列规范** (column) | $\log \psi_j(x) \rightarrow \log \psi_j(x) + c_j$ | $L[i,j] += c_j$ | $+\text{Re}(\sum_j c_j)$ | 不变（列缩放 = 右乘对角阵） |
| **行规范** (row) | $\log \psi_j(x_i) \rightarrow \log \psi_j(x_i) + c(x_i)$ | $L[i,j] += c_i$ | $+\text{Re}(\sum_i c_i)$ | 不变（行缩放 = 左乘对角阵） |

两者本质都是把 Ψ 乘一个对角矩阵，det 整体放缩，但 `Ψ⁻¹HΨ` 的迹（也就是 K 条能级的求和 loss）**完全不变**。

### 1.2 当前实现只修了列，没修行

`create_gauge_fixed_total_machines` (NES_VMC.py#L1191) 做的事：

```python
# 减去参考态 logψ_j(ref)，干掉列规范
L[i,j] = logψ_j(x_i) - logψ_j(ref)   # column-gauge invariant
shift = max(Re(L))                     # 稳定化
log_det = log|det(exp(L - shift))| + K·shift
```

- `L[i,j] - logψ_j(ref)` 在 $\log\psi_j \rightarrow \log\psi_j + c_j$ 下**完全不变** → **列规范确实被杀死**。
- 但**行规范是 per-sample 偏移 $c(x_i)$**：参考态 `ref` 只有一个点，根本无法约束"对所有 x 同时加 c(x)"这种自由。
- 数学上可以严格证明（见下文 §1.3），对"行规范"下的 $c_i$：

$$
\log\_det_{centered}' = \log\_det_{centered} + \sum_i c_i \quad (\text{当 } c_i \text{ 实数})
$$

也就是说 `logΨ mean` 的漂移量 ≈ $\sum_i c_i$，**完全没受列规范修复影响**。

### 1.3 解析验证（手工推导）

设 $L_{centered} = L - \log\psi(\text{ref})$，列规范 $L \rightarrow L + c\cdot\mathbf{1}^T$ 不改变 $L_{centered}$，所以列方向死透了 ✓。

行规范 $L \rightarrow D\cdot L$（$D = \text{diag}(\exp(c_1),...,\exp(c_K))$）：

```
L_centered' = D·L − logψ(ref)        # 注：参考态那一项不会被行规范影响
             = D·(L_centered + logψ(ref)) − logψ(ref)
             = D·L_centered + (D − I)·logψ(ref)
```

而行规范下 $L_{centered}'$ 的 shift 变 `shift' = shift + max(c_i)`，展开

$$
\log\_det_{centered}' = \log\_det_{centered} + \sum_i c_i \quad (\text{实数 } c_i)
$$

→ `logΨ mean` 增加 $\sum_i c_i$，正是我们在 log 里看到的"全局平移"。

---

## 2. 为什么自然梯度会驱动 row gauge 漂移？

直觉上：loss $\text{tr}(\Psi^{-1}H\Psi)$ 在行规范下是不变量 → 梯度 $F$ 应该在行方向上严格为 0，SR 更新也应该是 0。**为什么实际还是漂了？**

### 2.1 Monte-Carlo 噪声在 row gauge 方向上的非零残量

`nes_vmc_gradient_stable` 用批量样本估计：

$$
\hat{F} = \text{mean}_i\left(\text{tr}(E_L)_i - \langle\text{tr}(E_L)\rangle\right) \cdot \frac{\partial\log\Psi_i}{\partial\theta}
$$

- **列方向**：$\frac{\partial\log\Psi}{\partial\theta} \equiv 0$（logΨ 在列规范下不变），所以 $\hat{F}_{col}$ 在 *数学上* 和 *浮点上* 都是 0。
- **行方向**：$\frac{\partial\log\Psi}{\partial\theta} \neq 0$，但 $\hat{F}_{row} = E[c \cdot \text{constant}] = \text{constant} \cdot (E[\text{tr}(E_L)] - E[\text{tr}(E_L)]) = 0$（**解析上**为零）。

  但 MC 估计量不是解析值。$\hat{F}_{row}$ 的方差 ≈ $\text{Var}[(\text{tr}(E_L) - \langle\text{tr}(E_L)\rangle) \cdot d\log\Psi_{row}] / N_{batch}$。当 $d\log\Psi_{row}$（行方向的 logΨ 梯度）在样本间非恒定时，**这个方差非零**，$\hat{F}_{row}$ 就有一个 $\mathcal{N}(0, \sigma^2/N)$ 的随机残量。

### 2.2 diag_shift 正则化放大残量

`make_qgt_fn` 计算 $S = \frac{1}{N}\cdot O^H O$，并在 $S + \lambda I$ 上求逆（$\lambda=0.1$）。

- 在列方向 $S_{col} = 0$，所以 $(S+\lambda I)^{-1}|_{col} = 1/\lambda$。
- 在行方向 $S_{row} > 0$（因为 dlogΨ 在行方向上随样本变化），但**往往很小**（仅当所有 walker 的 row shift 完全一致时 $S_{row}$ 才会接近 0；实际中并不完全一致）。

  SR 更新：

  $$
  \Delta\theta_{row} = -\eta \cdot (S_{row} + \lambda)^{-1} \cdot \hat{F}_{row}
  $$

  即使 $\hat{F}_{row}$ 是个 ~10⁻² 量级的 MC 残量，**只要 $(S_{row} + \lambda)^{-1}$ 显著大于 $1/\lambda$ 区域，这一项就以 ~0.01~0.1/step 的速度持续累加**。

### 2.3 漂移速率与 logΨ 增长吻合

把 `logΨ mean` 每步增量做统计（来自日志前 100 步，无 step 113 爆发）：

- step 0→1: +0.89；step 1→2: +1.16；step 2→3: +1.22；...
- 早期大约 +1.0~+1.2/step（与 `η=0.1`、`λ=0.1`、MC 残量量级一致）。
- 100 步后参数进入能量极小值附近，$\hat{F}_{row}$ 残量也跟着变小（因为 $\text{tr}(E_L) - \langle\text{tr}(E_L)\rangle$ 的样本方差变小），漂移速率随之放缓 → 与 log 中 step 100 后曲线趋于平缓相吻合。
- step 113 出现"gauge 爆发"：`‖∇raw‖` 跳到 1393（注意 natural 梯度只有 0.12，被 `clip_norm=20` 截断）→ 说明参数已经在 row 方向上偏离得相当远，以至于原始空间中的梯度变得很大；同时 `cond(Ψ)=236` 提示 Ψ 矩阵在数值上已经接近奇异。

---

## 3. 这个漂移对 sampler 的影响

### 3.1 列规范方向（被修）：无影响

因为 `total_machine` 返回的是 `log_det_centered = log_det_stable + K·shift`，对列规范严格不变，**所以 $|\Psi|^2$ 在列规范下严格不变，Metropolis 接受率、采样分布完全不受影响**。

### 3.2 行规范方向（未修）：对 sampler 的影响

Metropolis 接受步用 $|\Psi(x')|^2 / |\Psi(x)|^2$。在行规范下：

$$
\frac{|\Psi(x')|^2}{|\Psi(x)|^2} \rightarrow \frac{\exp(2 c(x'))}{\exp(2 c(x))} = \exp(2(c(x') - c(x)))
$$

- 如果 $c(x)$ 对所有 x 都一样（"均匀 row shift"），接受率不变。
- 如果 $c(x)$ 因 x 而异，**采样分布的相对权重会被重新校准**，相当于引入了一个非物理的偏置。

### 3.3 实证：从 `min` 和 `max` 看漂移是不是"近似均匀"

观察数据：

| step | min | mean | max | (mean − min) | (max − mean) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 0   | 3.413 | 9.264 | 11.034 | 5.85 | 1.77 |
| 100 | 43.453 | 46.675 | 46.948 | 3.22 | 0.27 |
| 400 | 63.230 | 66.690 | 66.735 | 3.46 | 0.05 |
| 700 | 62.976 | 67.294 | 67.352 | 4.32 | 0.06 |

- `mean − min` 始终在 3~6 之间（≈ 4），**几乎不随 `logΨ` 整体水平变化**。
- `max − mean` 一直在 0~2 之间。

→ **行规范 $c(x)$ 对所有样本几乎相同**，即 walker 之间没有明显的相对权重再校准；**sampler 的统计性质（接受率、混合、采样分布）实际上没有显著恶化**。`mean` 的整体抬升本质上是"对所有 walker 同时乘了一个 $\exp(2\delta)$ 的常数"，这是一个 column shift 而不是 row shift 的效果。

> 但这并不意味着没有问题。$c(x) \approx \text{常数}$ 的事实，**反过来证明**参数在 row gauge 上的漂移是"近均匀 row shift"——也就是网络对**所有典型构型**都加了同一个 log ψ 偏移。这是行规范漂移的最坏情形之一：它不会立刻让 sampler 出问题，但会**让 $|\Psi|^2$ 越来越大**，最终在数值上撑爆（step 113 爆发就是临界点）。

### 3.4 总结：sampler 是不是"坏了"？

- **从能量估计角度**：没有坏。E0 在 step 400 之后稳在 -7.8629，离 FCI -7.8636 不到 1 mHa。这跟我们的分析完全一致——因为 $\text{tr}(E_L)$ 在 row gauge 下不变。
- **从数值稳定性角度**：有隐患。`cond(Ψ)=236`、原始梯度 1393 这种事件已经发生过一次（step 113）。如果继续训练或在更复杂的体系（H₂O、更多轨道、更深网络）上跑，**这种 row gauge 漂移迟早会触发 NaN/Inf**。
- **从偏差角度**：在当前的 K=4 LiH 体系下没有可见偏差；但**理论上一旦 row shift 在样本间变得不均匀，sampler 就会引入系统性偏差**（行规范不再只是全局常数因子）。

---

## 4. 与代码的对应关系

| 文件 / 行 | 角色 | 关键点 |
| :--- | :--- | :--- |
| `models.py::NESTotalAnsatz_stable` (~L96) | 模型本体 | 给出 $\log\Psi = \log\|\det \Psi\|$，**没有任何 gauge fixing** |
| `models.py::NESTotalAnsatz_gauge_stable` (~L169) | 备用模型 | 同样做 $L_{ij} = \log\psi_j(x_i) - \log\psi_j(\text{ref})$，**仍然是只修列** |
| `NES_VMC.py::create_gauge_fixed_total_machines` (L1191) | 当前使用的 wrappers | 在 `_compute_L_centered_single` 里只减 `logψ_j(ref)`，**没有加行约束** |
| `functions.py::nes_vmc_gradient_stable` (L206) | 梯度 | `jax.grad(total_machine)` 用的就是上面那个**列规范已修但行规范未修**的 `total_machine` |
| `NES_VMC.py::make_qgt_fn` (L1512) | QGT | $S$ 在列方向严格为 0，在行方向非零但量级小 |
| `NES_VMC.py::make_grad_fn` (L1536) | 入口 | 训练循环就是被这个 `grad_fn` 驱动 |

`LiH_molecule_K4_STO-3G.ipynb` Cell 4 里直接用 `create_gauge_fixed_total_machines(NESTotalAnsatz_stable, Hatree_Fock)`——**只跑了列规范修复**。

---

## 5. 修复建议（按推荐度从高到低）

### 5.1 [推荐] 真正 gauge-fix：把行规范也杀掉

在 `L_centered` 上再加一行约束，例如强制"每列减掉一个样本均值"：

```python
# 在 _compute_L_centered_single 里追加：
row_mean = jnp.mean(L_centered, axis=0, keepdims=True)  # shape (1, K)
L_centered = L_centered - row_mean                      # 杀掉行规范
```

或者用更标准的方式——把 L 投影到 $\text{tr}(L\cdot\mathbf{1}) = 0$（列向求和为零）的子空间。注意投影后要保持 L 仍是 K×K 方阵并保持 determinant 与原问题一致（这种投影会改变 det 的大小，所以要重新校准）。

一种更"安全"的做法：在网络结构层加**"每列的 logψ 在 ref 之外再减去一个常数"**，使 column shift 在训练中不增加自由度——但这只解决列，不解决行。

### 5.2 [次推荐] 周期性 re-centering

不动模型结构，每 N 步强行把"全局 log|\Psi|"重新拉回某个目标值：

```python
# 每 50 步，对所有 single_ansatz 的输出 bias 做一次减法，使 logΨ mean 回到 target
delta = target_logPsi - current_logPsi_mean
for ans in total_ansatz.single_ansatz_list:
    # 通过 stop_grad 的方式减去 delta，不影响梯度
    ans.output.bias.value -= delta
```

优点：实现极简，**不需要重写 ansatz**；缺点：是个"补丁"，理论上仍可能与优化器互相干扰。

### 5.3 [可选] 把 diag_shift 调大 / 监控 QGT 谱

- 把 `qgt_diag_shift` 从 0.1 调到 1.0 或更大：会**抑制** row gauge 方向的更新（因为 $1/(S_{row} + \lambda)$ 变小），但也会让非 gauge 方向的学习变慢。
- 在 `grad_fn` 里加一个 $\text{eigvalsh}(S)$ 的返回值，**把 QGT 谱画出来**。如果看到非常接近 0 但非零的特征值（数量应该恰好是 K）持续存在，那就是 row gauge 漂移的直接信号。

### 5.4 [可选] 用 `NESTotalAnsatz_gauge_stable_frozen` 做对比实验

`models.py` 里这个类（~L262）支持对某些 ansatz 做"梯度截断"，可以临时把 K 个 ansatz 之一冻结，看 row gauge 漂移是否消失。这是验证 row-gauge 假说的最直接实验。

---

## 6. 验证假说的小实验（建议跑一下）

1. **打印 row shift 的大小**：在 `_stable_from_L` 里把 `shift - logψ_j(ref)` 也输出（per-j），观察 4 列之间是否同步漂移。如果 4 列的 log 漂移量级一致 → 确认是"近似均匀 row shift"。
2. **打印 QGT 最小 K 个特征值**：在 `make_qgt_fn` 里加 `return_aux=True`，观察那 K 个最小特征值是否稳定在 0 附近。如果它们始终 ~0，列规范修得很干净；如果发现 K 个"接近 0 但非零"的特征值（量级 ~10⁻³~10⁻²），那就是 row gauge 的指纹。
3. **人为锁住 row shift**：每步之后把 `logψ_j(x_i)` 的样本均值强行减一个常数（用 stop_grad），重复训练，看 `logΨ mean` 是否被压平、E0/E1 是否能收敛到更低值。

---

## 7. 一句话结论

`logΨ mean` 单调上升是 **NES 行列式 ansatz 的"行规范"未被修复**导致的：它本身是物理 loss 的精确零模（不会破坏 E0/E1/E2/E3），但在 `qgt_diag_shift=0.1` 的正则化下被 MC 噪声缓慢驱动，最终在 step 113 触发了一次"gauge 模式爆发"（`cond(Ψ)=236`、`‖∇raw‖=1393`），并把 E0 从 -7.83 暂时打回 -7.77。**sampler 在当前 K=4 LiH 下还能正常工作**（行 shift 对所有 walker 近似均匀），但这是数值上的"苟延残喘"，不是真稳。**建议尽快在 `create_gauge_fixed_total_machines` 里加上 row gauge 修复（§5.1）**，或者至少在训练循环里加每 50 步 re-center（§5.2）。
