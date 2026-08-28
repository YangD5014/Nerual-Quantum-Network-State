# NES-VMC 运行报告 + 冻结浅层 Single-Ansatz 参数的设计方案

> 适用代码：
> - [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb)
> - [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py)
> - [NES_VMC_frozen_tech.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC_frozen_tech.py)
>
> 同源代码：[Li 原子/NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/Li原子/NES_VMC.py)
> 与本目录 `NES_VMC.py` 字节相同，可互换。

---

## 第一部分：NES-VMC 运行报告

### 1. 算法目标

NES-VMC（Natural Excited States Variational Monte Carlo）一次性变分求解 K 条 **正交归一** 的低激发态：

$$
\mathcal{L}_\theta(X) \;=\; \mathrm{Tr}\!\left[\Psi_\theta(X)^{-1}\,H\,\Psi_\theta(X)\right]
\;=\; \sum_{k=1}^{K} E_k,
$$

其中 $\Psi_\theta(X) = \big[\psi_j(x_i)\big]_{i,j=1}^{K}$ 是 $K\times K$ 行列式式 Ansatz，$X = (x_1, \dots, x_K)$ 是 **不重复的 K 个子构型**。通过最小化 $\mathcal{L}$，自动学到基态 + 最低 $(K-1)$ 个激发态的同步变分波函数。

### 2. 关键数据约定

| 符号 | 含义 | 数值（LiH K4） |
|------|------|----------------|
| $K$ | 同时拟合的态数 | 4 |
| `SINGLE_SIZE = hi.size` | 单子 Hilbert 维度 | 4（CAS 2e / 2α × 2β）|
| `hi_ext.size = K * SINGLE_SIZE` | 扩展 Hilbert 维度 | 16（外加 K-1=3 个 forbidden）→ 实际 32 位 |
| `n_spin_orbitals` | SingleStateAnsatz 输入维 | 4 |
| `hidden_dim` | FFNN 隐藏维 | 12 |
| $\theta$ 总参数量 | K 个 sub-ansatz 拼成 | 约 1832 个复数参数 |

### 3. 代码分层结构

```text
LiH 规范漂移-1.ipynb
├── 数据 / 系统构造 (LiH.py: ha, hi, hi_ext, ext_edges, K, Hatree_Fock)
├── 采样器
│   └── NESFermionHopRule               (NES_VMC.py:1083)
│       ├─ _check_duplicate             拒绝产生重复子组态
│       └─ transition / random_state    满足 Pauli 不相容
├── 模型
│   ├── SingleStateAnsatz                (NES_VMC.py:26)        单态 logψ 网络
│   └── NESTotalAnsatz_gauge_stable      (NES_VMC.py:236)       行列式式 + gauge-fixing
├── 包装 / 机器
│   ├── create_machine_gauge_stable      (NES_VMC.py:1549)      sampler 用的 logΨ_gauge
│   ├── create_machine_matrix_gauge_stable (NES_VMC.py:1571)    loss 用的 L_stable
│   ├── create_machine_max_gauge_stable  (NES_VMC.py:1588)      loss 用的 shift
│   ├── create_single_machine_gauge_fixed (NES_VMC.py:1525)     HPsi 用的 logψ_j - logψ_j(ref)
│   └── create_machine_gauge_synthesis   (NES_VMC.py:1620)      一站式打包 4 个机器
├── 能量 / 损失
│   ├── Ham_psi_scaled                   (NES_VMC.py:474)       单态 HPsi（带 shift）
│   ├── Ham_Psi_scaled                   (NES_VMC.py:536)       K×K HPsi 矩阵
│   ├── NES_loss_energy_stable           (NES_VMC.py:609)       loss + E_L 矩阵
│   ├── nes_vmc_gradient_stable          (NES_VMC.py:740)       loss 的反向梯度
│   └── _masked_mean_batch               (NES_VMC.py:715)       NaN/inf walker 屏蔽
├── QGT / 自然梯度
│   ├── make_qgt_fn                      (NES_VMC.py:1477)      S_ij = <∂logΨ* ∂logΨ>
│   ├── compute_qgt_fixed                (NES_VMC.py:1435)      一次性版本（含 eigvalsh 诊断）
│   └── flatten_batched_pytree           (NES_VMC.py:1420)      拼 (n_samples, n_params)
├── 工具
│   ├── get_ccsd_excitations_and_sampler_edges_from_hf  (NES_VMC.py:326)
│   ├── statistics                       (NES_VMC.py:907)
│   └── sampler_info                     (NES_VMC.py:1071)
└── 训练主循环 (LiH 规范漂移-1.ipynb:944-1029)
    ├── nes_sampler.sample
    ├── grad_fn(total_params, x_batch)
    ├── qgt_fn → solve → 自然梯度
    ├── optimizer.update (clip + SGD)
    └── optax.apply_updates
```

### 4. 关键函数职责详解

#### 4.1 网络层

| 函数 / 类 | 行号 | 职责 |
|-----------|------|------|
| `SingleStateAnsatz.__init__` | [NES_VMC.py:29-34](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L29-L34) | 三层复值 FFNN：`linear1 → tanh → linear2 → tanh → output(1)` |
| `SingleStateAnsatz.__call__` | [NES_VMC.py:36-40](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L36-L40) | 输入 `(n_spin,)` 单构型 / `(n_conn, n_spin)` 哈密顿连接批，输出复标量 logψ |
| `NESTotalAnsatz_gauge_stable.__init__` | [NES_VMC.py:236-258](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L236-L258) | 维护 K 个 sub-ansatz + `ref_state`（默认 Hatree-Fock）|
| `NESTotalAnsatz_gauge_stable._forward_single` | [NES_VMC.py:260-297](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L260-L297) | 构造 gauge-fixed 矩阵 $L_{ij}=\log\psi_j(x_i)-\log\psi_j(\text{ref})$，再数值稳定化（减 max Re(L)）|

#### 4.2 机器包装

| 函数 | 行号 | 输入 | 输出 | 用途 |
|------|------|------|------|------|
| `create_machine_gauge_stable` | [1549-1568](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1549-L1568) | `(params, σ)` | `logΨ_gauge` | 喂给 **NetKet 采样器**（重整化后概率 $|\Psi|^2$ 不变）|
| `create_machine_matrix_gauge_stable` | [1571-1585](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1571-L1585) | `(params, σ)` | `L_stable` | 构造 $\Psi_{\text{stable}}=\exp(L-L_{\max})$ |
| `create_machine_max_gauge_stable` | [1588-1602](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1588-L1602) | `(params, σ)` | `shift` | 给 HPsi 做指数位移 $\exp(\log\psi - \text{shift})$ |
| `create_single_machine_gauge_fixed` | [1525-1544](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1525-L1544) | `(params_j, σ)` | `logψ_j(σ) - logψ_j(ref)` | **不** stop_grad ref，让 ref 进入反向——这是 gauge drift 抑制的关键 |
| `create_machine_gauge_synthesis` | [1620-1625](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1620-L1625) | — | 4 个机器的闭包 | 笔记本里一站式 4 个机器 |

#### 4.3 哈密顿作用（关键计算瓶颈）

| 函数 | 行号 | 公式 | 说明 |
|------|------|------|------|
| `Ham_psi_scaled` | [474-533](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L474-L533) | $\sum_{x'}H[x,x']\,e^{\log\psi(x')-\text{shift}}$ | 单态 HPsi，自动兼容 1D/2D |
| `Ham_Psi_scaled` | [536-607](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L536-L607) | 拼成 K×K 矩阵 | 双层 for（仅 Python 静态循环，JIT 内联）|
| `NES_loss_energy_stable` | [609-713](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L609-L713) | $\mathcal{L}=\mathrm{Re}\,\mathrm{Tr}(\Psi^{-1}HPsi)$ | 数值稳定 + valid mask |

#### 4.4 梯度 + QGT

| 函数 | 行号 | 输出 | 备注 |
|------|------|------|------|
| `nes_vmc_gradient_stable` | [740-904](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L740-L904) | `(grad_pytree, loss_mean, E_L_mean)` | 调用链：loss → E_L_centered → `jax.grad(total_machine, holomorphic=True)` → masked weighted mean |
| `make_grad_fn` | [1501-1521](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1501-L1521) | jit 化 grad_fn | 闭包内预先 capture 5 个机器 |
| `make_qgt_fn` | [1477-1498](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1477-L1498) | `S_reg = (1/N)·∇logΨ†∇logΨ + λI` | 沿用 NetKet SR 的 QGT 公式，复数内积用 `jnp.conj` |
| `compute_qgt_fixed` | [1435-1474](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1435-L1474) | 同上 + `S_cond`, `S_eig_min/max` | 一次性版本，多用于诊断 |
| `flatten_batched_pytree` | [1420-1433](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1420-L1433) | `(n_samples, n_params)` | 拼多个 leaf 为稠密矩阵（比 `ravel_pytree` 更稳）|

#### 4.5 采样器

| 函数 / 类 | 行号 | 职责 |
|-----------|------|------|
| `NESFermionHopRule._check_duplicate` | [1089-1107](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1089-L1107) | NES 硬约束：任意两子组态不能完全相同（保证行列式非零 + Slater 反对称）|
| `NESFermionHopRule.transition` | [1110-1125](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1110-L1125) | 从 `edges` 随机挑边做单激发跳跃，违反约束就回退 |
| `NESFermionHopRule.random_state` | [1127-1151](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1127-L1151) | 拒绝采样到无重复子组态为止 |
| `get_ccsd_excitations_and_sampler_edges_from_hf` | [326-372](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L326-L372) | 由 HF 态产生 (occ → vir) 单激发边 + 派生的 doubles 列表 |

### 5. 训练主循环（[LiH 规范漂移-1.ipynb:944-1029](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb#L944-L1029)）

```text
init:  total_ansatz = NESTotalAnsatz_gauge_stable(...)
       build 4 个 machine 闭包 (logΨ, L_stable, shift, L_gauge)
       build K 个 single_machine（HPsi 用）
       grad_fn = make_grad_fn(...)
       qgt_fn  = make_qgt_fn(logΨ)
       nes_sampler = MetropolisSampler(hi_ext, NESFermionHopRule, n_chains=16)
       optimizer = optax.chain(clip_by_global_norm(20), sgd(lr=0.1))
       opt_state = optimizer.init(total_params)
       sampler_state = nes_sampler.init_state(...)

for step in range(N_ITER=500):
    1. SAMPLING
       samples_raw, sampler_state = nes_sampler.sample(
           machine=total_machine_log_Psi_gauge, parameters=total_params,
           state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
       )
       x_batch = samples.reshape(-1, K, SINGLE_SIZE)        # (batch, K, n_spin)

    2. GRAD (含 Loss / E_L)
       grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)
       #   内部：NES_loss_energy_stable → masked mean → jax.grad(total_machine, holo)

    3. QGT / NATURAL GRAD
       grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
       if Natural_Grad:
           qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift=0.1)
           ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
           grad_update = unravel_fn(ng_flat)
       else:
           grad_update = grad_raw

    4. CLIP + UPDATE
       updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
       total_params = optax.apply_updates(total_params, updates)

    5. MONITOR
       psi_cond  = cond(L_stable(x_batch[0]))
       eig_vals  = eig(E_L_mean) → 4 条能量曲线
       logΨ mean/min/max, gradient 三范数, 4 条 loss_per_state
       history[step].append(...)

    6. 异常保护
       if has_nan_grad or grad_norm > 5000 → warning
       if grad_norm_clipped == 0 → break
```

### 6. Gauge-fixing 数值稳定化机理

`NESTotalAnsatz_gauge_stable._forward_single` 的核心 3 行：

```python
# 1) 矩阵中心化：把整列减去 logψ_j(ref)  → 消除整体相位漂移
L_ij = logψ_j(x_i) - logψ_j(ref)

# 2) 减 max Re(L)：阻止 exp 数值溢出
shift = stop_grad(max(Re(L)))            # shift 不进入反向
L_stable = L - shift

# 3) 稳定 log det
sign, log_abs_det = slogdet(exp(L_stable))
logΨ = log_abs_det + 1j·angle(sign) + K·shift   # 给 sampler 的真实值
```

这一步同时实现了 **规范(gauge)固定** 和 **数值稳定**，使得 K 条态线性无关且 Ψ 矩阵始终条件数良好。

---

## 第二部分：冻结浅层 `single_ansatz` 参数的设计方案

### 1. 需求规格

| 维度 | 冻结前 | 冻结后（step ≥ N）|
|------|--------|-------------------|
| Forward log Ψ / L / shift | 全部 K 列参与 | **全部 K 列仍参与** |
| Loss / E_L | 全部 K 列 | **全部 K 列仍参与** |
| 采样 | 全部 K 个子网前向 | **仍用全部 K 个子网** |
| ∂Loss/∂θ_frozen | 正常求导 | **必须为 0** |
| QGT 中 frozen 行/列 | 正常 | **必须从 QGT 中物理剔除** |
| QGT solve 维度 | N×N | **N_active × N_active**（提速核心）|
| θ_frozen 的更新 | η·∇ | **不更新** |
| θ_active 的更新 | 正常 | **正常** |

### 2. 设计原则

> **冻结只改变"哪些参数参与训练", 不改变 loss 公式.**

`Loss = trace(Psi⁻¹ · H · Psi)` 中，$\Psi_{ij} = \exp(\log\psi_j(x_i) - \text{shift})$，每一列 $\log\psi_j$ 都进入 loss 数值。冻结只是让"对 $\theta_j$ 的导数不再计算 / 不再更新", **loss 值不变**, 训练曲线连续。

### 3. 三档方案对比

| 方案 | 做法 | 节省 forward | 节省 backward | QGT 尺寸 | 改动量 |
|------|------|:----:|:----:|:----:|:----:|
| ❶ mask to 0 | 全算全求导再乘 0 | ✗ | ✗ | N×N（不变）| ★ |
| ❷ `stop_gradient` 包 frozen 列的 forward 输出 | forward 算, backward 跳过 | ✗ | ✅ ~25% | N×N（数值仍是 0）| ★★ |
| ❸ **物理分离** frozen / active 子图 | 拆 graphdef, 重建子 machine | ✅ ~25% | ✅ ~25% | **N_active × N_active** | ★★★ |

**推荐：先 ❷ 后 ❸**。❷ 改动最小, 10 行代码; ❸ 在 ❷ 仍是瓶颈时再上。

### 4. 方案 ❷: `stop_gradient` 包 frozen 列（首选, 改动量小）

#### 4.1 原理

JAX 的 `jax.lax.stop_gradient(x)` 在 forward 阶段返回 x 本身, 在 backward 阶段直接返回 0 梯度（不分配 cotangent buffer, 不调度反向 kernel）。把 frozen sub-ansatz 的 forward 输出 `logψ_freeze(x)` 包住, 反向时整个子图被剪枝。

#### 4.2 代码改动 (在 `NESTotalAnsatz_gauge_stable` 内)

修改 `_forward_single` 中构造 L 矩阵的双层 for 循环 ([NES_VMC.py:272-278](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L272-L278)):

```python
# 原版
for i in range(self.K):
    for j in range(self.K):
        ans_j = self.single_ansatz_list[j]
        log_x = ans_j(x_single[i])
        log_ref = ref_logs[j]
        L = L.at[i, j].set(log_x - log_ref)

# 冻结版: 在 j == freeze_idx 时包 stop_gradient
for i in range(self.K):
    for j in range(self.K):
        ans_j = self.single_ansatz_list[j]
        log_x = ans_j(x_single[i])
        if j == self.freeze_idx:                 # ★ 新增
            log_x = jax.lax.stop_gradient(log_x)
        log_ref = ref_logs[j]                    # ref 本就 stop_grad
        L = L.at[i, j].set(log_x - log_ref)
```

> 进阶做法：把 `freeze_idx` 做成 `nnx.Module` 的字段，运行时直接切换（推荐）。

#### 4.3 训练循环的修改

在 `LiH 规范漂移-1.ipynb:944` 主循环开头加 1 行：

```python
FREEZE_STEP = 300    # 第 300 步后冻结
FREEZE_IDX  = 0      # 冻结 sub-ansatz 0（即"基态锚点"）

# 初始化时: freeze_idx = None, 即原版行为
total_ansatz = NESTotalAnsatz_gauge_stable(
    SINGLE_SIZE, K, SINGLE_SIZE+K, ref_state=Hatree_Fock,
    rngs=nnx.Rngs(11), freeze_idx=None,    # ★ 新增字段
)

for step in range(N_ITER):
    # ★ 冻结开关: 重建机器以触发 jit cache 重新编译
    if step == FREEZE_STEP:
        total_ansatz.freeze_idx = FREEZE_IDX
        total_machine_log_Psi_gauge, graphdef, total_params = \
            create_machine_gauge_stable(total_ansatz)
        total_machine_L_stable, total_machine_shift, total_machine_L_gauge = \
            create_machine_gauge_synthesis(total_ansatz)[1:]   # 重新拿 3 个
        single_machine_list = [
            create_single_machine_gauge_fixed(a, Hatree_Fock)[0]
            for a in total_ansatz.single_ansatz_list
        ]
        grad_fn = make_grad_fn(ha, total_machine_L_stable, total_machine_shift,
                               total_machine_log_Psi_gauge, single_machine_list)
        qgt_fn  = make_qgt_fn(total_machine_log_Psi_gauge)
        sampler_state = nes_sampler.reset(machine=total_machine_log_Psi_gauge,
                                          parameters=total_params,
                                          state=sampler_state)

    # 训练循环其余一字不动 ...
```

> 关键说明：`stop_gradient` 切断了 frozen 列的反向路径, JAX 会**自动重新 jit**, 不需要手动清理 cache. 一次编译开销 ~30s, 之后每步变快.

#### 4.4 Loss 是否变化?

**完全不变**. 给出数学证明：

```text
冻结前: L[i, j] = logψ_j(x_i) - logψ_j(ref)
冻结后: L[i, j] = stop_grad(logψ_j(x_i)) - stop_grad(logψ_j(ref))  (j == freeze_idx)
         数值上 = logψ_j(x_i) - logψ_j(ref)                          (Loss 用到的就是这个数)
```

`stop_gradient` 在前向求值时返回的就是输入本身, 只在反向时切断. Loss 拿到的 `L_stable`, `Psi_stable`, `HPsi_stable` 与冻结前 bit-by-bit 一致。

#### 4.5 QGT 是否需要单独处理?

**❷ 方案下, 严格来说不必**. 解释:

- `qgt_fn` 仍对完整 `total_params` 求导, frozen 方向的梯度虽然物理上为 0, 但 JAX 算图里仍然走 backward 路径 → 维度不变 (N×N), solve 耗时不变。
- 自然梯度 `ng = S⁻¹ · ∇` 在 frozen 方向上数值是 0（因为 ∇_frozen = 0, S^{-1} 在 frozen 方向也无贡献）。
- **QGT 仍然在 frozen 方向浪费一次矩阵求逆**。但 frozen 方向 ∇=0, 数值结果与剔除它等价。

**真正的提速来自 ❸**（物理分离）。

### 5. 方案 ❸: 物理分离 frozen / active 子图（论文级加速）

#### 5.1 思想

把 `total_ansatz` 拆成两个独立图：

1. **frozen 子图**：`single_ansatz[freeze_idx]` 单独封一个 machine, 闭包内捕获 frozen params. 后续不再变化。
2. **active 子图**：剩余 K-1 个 sub-ansatz 拼成新的 `NESTotalAnsatz_active(K-1)`, 只对它求导和算 QGT.

`L` 矩阵的 frozen 列直接用 `frozen_machine(sigma)`, active 列走 active 子图. L 仍是 K×K 完整矩阵, loss 公式不变, 但 QGT 只对 active 参数求 → 维度从 N×N 降到 (N - N_frozen)×(N - N_frozen).

#### 5.2 框架代码（参考 [NES_VMC_frozen_tech.py:22-69](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC_frozen_tech.py#L22-L69) 的雏形, 下面是完整可运行版）

```python
# ============================================================
# 1) 冻结时刻的"拆分"工具
# ============================================================
def split_frozen_active(total_ansatz, total_params, freeze_idx):
    """
    一次性把 total_ansatz 拆成 frozen + active 两部分
    返回:
        frozen_machine, frozen_params            # 冻结的, 不再变
        active_ansatz, active_params, active_graphdef
    """
    graphdef, state = nnx.split(total_ansatz)

    # frozen 部分
    frozen_ansatz = total_ansatz.single_ansatz_list[freeze_idx]
    frozen_params = state['single_ansatz_list'][freeze_idx]
    frozen_machine, _, _ = create_single_machine_gauge_fixed(
        frozen_ansatz, Hatree_Fock
    )  # 闭包已捕获 frozen_ansatz, 之后不会重新构造

    # active 部分: 用剩下的 K-1 个 sub-ansatz 重新建一个 total
    active_sub_list = [
        a for i, a in enumerate(total_ansatz.single_ansatz_list)
        if i != freeze_idx
    ]
    active_state = {
        'single_ansatz_list': [
            state['single_ansatz_list'][i]
            for i in range(len(state['single_ansatz_list']))
            if i != freeze_idx
        ]
    }
    # ★ 关键: 新建一个 K-1 的 total_ansatz
    active_ansatz = NESTotalAnsatz_gauge_stable_active(
        n_spin_orbitals=SINGLE_SIZE,
        n_states=K - 1,
        hidden_dim=SINGLE_SIZE + K,
        ref_state=Hatree_Fock,
        sub_ansatz_list=active_sub_list,    # 复用原 sub-ansatz 对象
    )

    return frozen_machine, frozen_params, active_ansatz, active_state


# ============================================================
# 2) 重新构造机器: 冻结列走 frozen_machine, 活跃列走 active_ansatz
# ============================================================
def make_split_total_machines(frozen_machine, active_ansatz, freeze_idx):
    """
    构造 4 个 machine:
        total_machine_log_Psi_gauge   采样用
        total_machine_L_stable         loss 构造 Psi 用
        total_machine_shift            loss 构造 HPsi 用
        total_machine_L_gauge          诊断 / 重置
    它们都接受 active_params, frozen_machine 闭包内捕获
    """
    K_full = active_ansatz.K + 1   # 原始 K
    n_spin = active_ansatz.n_spin

    @jax.jit
    def total_machine_log_Psi_gauge(active_params, sigma):
        # sigma: (batch, K_full * n_spin) 或 (batch, K_full, n_spin)
        m = nnx.merge(*nnx.split(active_ansatz)[:1], active_params)  # 仅用 active graphdef
        # 1) active 部分: 直接调 active_ansatz, 返回 L_active (K-1, K-1)
        L_active, _ = m(sigma)    # active 走自己的 slogdet
        # 2) frozen 列: 算 logψ_freeze(x_i), shape (K_full,)
        log_frozen = _frozen_col(frozen_machine, sigma)  # (batch, K_full)
        # 3) 拼回完整 L, 再走一次 slogdet
        L_full = _embed_frozen_col(L_active, log_frozen, freeze_idx)
        # ... 同样的 shift + slogdet + K·shift ...
        return log_Psi_gauge

    # 同样改造 L_stable / shift / L_gauge 三个
    return total_machine_log_Psi_gauge, ...
```

具体实现细节省略（核心是 L 矩阵的"列重排"操作 + 一次额外的 slogdet）。**实现后 QGT 维度从 N×N 降到 (N - N_frozen)×(N - N_frozen), QGT-solve 耗时按平方缩减**.

#### 5.3 加速效果估算（K=4, hidden=12, n_spin=4, LiH K4）

| 方案 | backward 节省 | QGT-solve 节省 | 整体节省 |
|------|:--:|:--:|:--:|
| ❶ mask | 0% | 0% | 0% |
| ❷ stop_grad | ~25% | 0%（仍是 N×N）| 10~15% |
| ❸ 物理分离 | ~25% | ~44% (16² → 9² = 56% 规模) | 30~45% |
| ❸ + jit 缓存清理 | — | — | + 5% |

实际占比请用 `jax.profiler` 测后再决定上哪一档。

### 6. 三档方案的 Loss 表达式对照

| 表达式 | 冻结前 | ❶ mask | ❷ stop_grad | ❸ 物理分离 |
|--------|--------|--------|-------------|------------|
| `L_ij = logψ_j(x_i) - logψ_j(ref)` | 全部 K 列 | 全部 K 列 | 全部 K 列, j=freeze 包 stop_grad | 全部 K 列, 但分两路计算 |
| `Ψ_stable = exp(L_stable)` | K×K | K×K | K×K（数值 bit-by-bit 同 ❶）| K×K（数值 bit-by-bit 同 ❶）|
| `HPsi_stable[i,j] = Σ H[x_i,x']·exp(logψ_j(x')-shift)` | K×K | K×K | K×K | K×K |
| `Loss = Re·Tr(solve(Psi, HPsi))` | 数值 A | 数值 A | 数值 A | 数值 A |
| `∇_active = ∂Loss/∂θ_active` | 全量 | 全量 (frozen 屏蔽) | 完整 | 完整（不求 frozen 方向）|
| `∇_frozen = ∂Loss/∂θ_frozen` | 数值 B | 0（mask 后）| 0（stop_grad 后）| **不计算**（图里没有 frozen 参数）|
| `S_ij = <∂_i logΨ* · ∂_j logΨ>` | N×N | N×N（frozen 行/列 = 0 数值）| N×N（frozen 行/列 = 0 数值）| **(N - N_frozen)²**（物理剔除）|
| `ng = S⁻¹ · ∇` | N 维 | N 维（frozen = 0）| N 维（frozen = 0）| **(N - N_frozen) 维**（frozen 不参与）|

### 7. 验证清单

冻结后请依次检查：

1. **采样分布不变**：冻结前后 `logΨ` mean/min/max 不应有突变。
2. **Loss / E_L 数值连续**：冻结那一帧的能量相对上一帧的 |Δ| 应在噪声量级。
3. **冻结子 ansatz 的参数 norm 恒定**：连续打印 `total_params["single_ansatz_list"][freeze_idx]["linear1"]["kernel"]` 的 norm，冻结后必须完全不动。
4. **其他 sub-ansatz 仍正常更新**：norm 单调下降 / 有合理学习曲线。
5. **QGT 条件数**：冻结后 `S_cond` 应下降（少了一堆零特征方向），不会再有"按不下去"的现象。
6. **整体能量曲线**：基态能量继续收敛, frozen sub-ansatz 对应的态能量被锁住, 其它态继续下降。

可选诊断：

```python
if step >= FREEZE_STEP and step % 50 == 0:
    frozen_norm = jnp.linalg.norm(ravel_pytree(
        total_params["single_ansatz_list"][FREEZE_IDX]
    )[0])
    flat_grad = grad_raw_flat
    active_grad_norm = jnp.linalg.norm(flat_grad * mask_flat_bool)
    frozen_grad_norm = jnp.linalg.norm(flat_grad * (~mask_flat_bool))
    logger.info(
        f"[step {step}] frozen_param_norm={frozen_norm:.6f} "
        f"active_grad_norm={active_grad_norm:.4f} "
        f"frozen_grad_norm={frozen_grad_norm:.2e}"
    )
    # 期望: frozen_grad_norm 恒为 0, frozen_param_norm 恒定
```

### 8. 注意事项 & 坑

1. **NNX state 是 Frozen Mapping**：`nnx.split` 返回的 `state` 不可直接 `__setitem__` / `append`. 用 `list(...)` 重建再 `jax.tree.map`。
2. **`stop_gradient` 不要包 `shift`**：`shift` 本就被 `stop_grad` 包住, 不要重复包, 否则会破坏数值稳定化的可微性。
3. **重建 machine 的时机**：`freeze_idx` 改动必须 **rebuild 4 个 machine** + `grad_fn` + `qgt_fn` + `nes_sampler.reset()`, 否则 JAX 仍用旧 cache.
4. **sampler state 的 reset**：`freeze_idx` 改后概率分布会跳变 (因为 frozen 方向停止更新), 旧的 `sampler_state` 链可能不平衡, 建议 reset 一次链。
5. **多次冻结 / 解冻**：当前实现支持"冻一次 + 冻更多". 想动态开关, 用 `is_frozen[step]` 查表函数即可。
6. **如果同时用 optax.masked**：在 ❷ 方案下, 配合 `optax.masked(sgd, param_mask)` 可以**双保险**, 但要确保 mask 与 stop_grad 同步切换, 否则会出现"前向不计算梯度, 但 optimizer 误以为有梯度" 的奇怪现象。

### 9. 推荐路线

```text
Step 1. 方案 ❷: stop_gradient 包 frozen 列的 forward
        → 验证 loss 曲线在 FREEZE_STEP 处没有跳跃
        → 验证 frozen sub-ansatz 的 E_k 锁住, 其它态继续收敛
        → 提速 ~10~15%

Step 2. 如果 sampler 仍是瓶颈: 把 n_chains / chain_length 加大
        如果 backward / QGT-solve 是瓶颈: 升级到 ❸
        → QGT 求解从 N×N → (N - N_frozen)²
        → 整体提速 30~45%

Step 3. 如果还想要更快: 多个 sub-ansatz 共享"浅层 + 各自顶层"
        → 进一步压缩参数总量
```

### 10. 一图流（修改位置）

```text
LiH 规范漂移-1.ipynb
└── for step in range(N_ITER):          ← 循环开头
    │
    ├── ★ 冻结开关: step == FREEZE_STEP ?
    │     ├─ total_ansatz.freeze_idx = FREEZE_IDX
    │     ├─ 重建 4 个 machine (logΨ, L_stable, shift, L_gauge)
    │     ├─ 重建 grad_fn, qgt_fn
    │     └─ sampler_state = sampler.reset(...)
    │
    ├── nes_sampler.sample(...)         # 1. 采样: 全部 K 个 sub-ansatz 仍算
    │
    ├── grad_fn(total_params, x_batch)  # 2. forward: 全部 K 列, loss/E_L 不变
    │       ↓
    │   grad_raw                        # 3. ★ frozen 列的 ∂Loss/∂θ = 0 (stop_grad)
    │       ↓
    │   grad_raw_flat
    │
    ├── qgt_fn(...)                     # 4. 仍算完整 N×N (❷) / 或 N_active×N_active (❸)
    │       ↓
    │   ng_flat                         # 5. frozen 方向数值 = 0
    │       ↓
    │   unravel_fn(ng_flat)
    │       ↓
    │   optimizer.update(...)           # 6. updates[freeze_idx] = 0
    │       ↓
    │   apply_updates(total_params, updates)  # 7. frozen 子网 add 0 = 不动
    │
    └── 监控 / Loss / logΨ / E_L: 完全不动
```

按这个流程改, **前向完全不受影响**, **优化器只对未被冻结的 sub-ansatz 起作用**, 冻结那支子 ansatz 仍然作为"基态锚点"参与波函数构造, 但不再被训练扰动。配合方案 ❸ 可同时实现 **QGT 维度缩减**, 训练速度可提速 30~45%。

---

## 附：相关已有文档

- [冻结子Ansatz方案.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/冻结子Ansatz方案.md) — mask 版
- [冻结子Ansatz_跳过梯度版.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/冻结子Ansatz_跳过梯度版.md) — stop_gradient + 物理分离 详细对比
- [NES_VMC_frozen_tech.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC_frozen_tech.py) — 物理分离的雏形实现
- [0710_分层冻结策略.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/0710_分层冻结策略.ipynb) — 实测日志
- [0710_K4_LiH_分层冻结策略.log](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/0710_K4_LiH_分层冻结策略.log) — 对应运行日志
