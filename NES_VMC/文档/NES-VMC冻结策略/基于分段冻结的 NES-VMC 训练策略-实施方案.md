# 基于分段冻结的 NES-VMC 训练策略：详细实施方案

> 配套构想文档：[基于分段冻结的 NES-VMC 训练策略.md](基于分段冻结的%20NES-VMC%20训练策略.md)
> 参考实现：`experiments/LiH分子/NES_VMC_train_min120_gauge_reset copy.py`（P8：双侧一致列规范 + 周期性 gauge reset）

---

## 0. 总体思路

NES-VMC 的总波函数由 K 个子波函数构成：$\Psi = \det[\psi_j(x_i)]$，即 $\Psi$ 矩阵的第 $j$ 列由子 ansatz `single_ansatz_list[j]` 给出。训练中各能级收敛速度不同——部分能级率先达到化学精度后，其对应子 ansatz 的参数继续参与训练已无收益，反而拖慢整体（梯度、QGT、线性求解的规模都由**全部**参数决定）。

**分段冻结**：当能级 $j$ 满足冻结判据后，把 `single_ansatz_list[j]` 的全部参数从"可训练集合"中移除，**单向冻结**——直至训练结束，永不激活（无解冻路径）。其余环节——采样、能量估计、gauge reset、监控——全部保持不变。

方案回答三个核心问题：

| 问题 | 一句话答案 |
|---|---|
| 1. 冻结判据？ | $E_L$ 矩阵本征值 $E_j$ 与 FCI 精确值误差 < 化学精度（1.6 mHa），且**连续 W 步**满足 + 波动稳定 + 采样有效 + 能级间隔健康，才冻结；**单向冻结**（永不激活），冻结后持续监控 $E_j$ 漂移（漂移不触发解冻，合理性论证见 §2.5） |
| 2. 如何冻结不影响算法？ | 数学上 = 约束变分（活跃参数子集上的优化），loss/采样/规范全部不变；工程上 = 对冻结列施加 `stop_gradient` + QGT 的 O 矩阵按活跃索引切片 + SGD 零梯度自然实现参数不动 |
| 3. 如何确保真减负？ | 不是简单把梯度乘 0（那是假减负），而是：`stop_gradient` 在反向图上**剪掉冻结支路**的 VJP；QGT 从 $P\times P$ 缩到 $P_a\times P_a$（solve 从 $O(P^3)$ 降到 $O(P_a^3)$）；并通过 shape 断言、jaxpr 检查、profiler 计时三重验证 |

---

## 1. 问题一：判断需要部分冻结的标准

### 1.1 监测量

现有代码每步已经算出了所需的全部量：

- 能级估计：`E_L_mean` 的本征值（[NES_VMC_train_min120_gauge_reset copy.py #L643-L644](../experiments/LiH分子/NES_VMC_train_min120_gauge_reset%20copy.py#L643-L644)）：

  ```python
  eig_vals, _ = jnp.linalg.eig(E_L_mean)
  eig_vals = eig_vals[jnp.argsort(eig_vals.real)]
  ```

- 精确参考：FCI 前K个本征值（[#L539](../experiments/LiH分子/NES_VMC_train_min120_gauge_reset%20copy.py#L539)）：

  ```python
  eigvals, _ = eigsh(ha.to_sparse(), k=K, which="SA", tol=1e-10)
  ```

定义能级 $j$ 的误差与稳定性指标：

$$
\epsilon_j^{(t)} = \big|\ \mathrm{Re}\,E_j^{(t)} - E_j^{\mathrm{exact}}\ \big|, \qquad
\sigma_j^{(t)} = \mathrm{std}\big(\{\mathrm{Re}\,E_j^{(t-W+1)},\dots,\mathrm{Re}\,E_j^{(t)}\}\big)
$$

### 1.2 冻结判据（全部满足才冻结）

| 条件 | 公式/阈值 | 理由 |
|---|---|---|
| C1 精度 | $\epsilon_j < \delta_{\mathrm{chem}} = 1.6\times10^{-3}$ Ha（化学精度 1 kcal/mol） | 构想文档的定义 |
| C2 持续性 | $\epsilon_j < \delta_{\mathrm{chem}}$ 在**连续 $W$ 步**内每步都成立，$W = 10$（= 一个 RESET_PERIOD） | 防止在误差曲线的瞬时下探处误冻 |
| C3 平稳性 | 窗口内 $\sigma_j < 0.5\,\delta_{\mathrm{chem}}$ | 区分"收敛到平台"与"恰好路过" |
| C4 数值健康 | 窗口内 `valid_ratio` 均值 ≥ `min_valid_ratio`(0.25)，且无 NaN 梯度 | 不在数值病态时做结构性决策 |
| C5 预热与冷却 | $\mathrm{step} \ge 30$ 才允许首次冻结；两次冻结事件间隔 ≥ $W$ | 留出观察单次冻结对剩余能级影响的时间 |

满足 C1–C5 后，将 $j$ 加入 `frozen_set` 并记录冻结时刻的 $\epsilon_j$。

**补充说明（能级配对问题）**：$E_j^{(t)}$ 按 $\mathrm{Re}$ 排序后与精确值按序号配对，是一种实用近似。若接近简并（$|E_j - E_{j+1}| \lesssim \delta_{\mathrm{chem}}$），序号可能在相邻步之间交换，此时暂缓冻结该相邻对（加一条 C6：目标能级与相邻能级间距 $> 2\delta_{\mathrm{chem}}$，否则跳过）。LiH/STO-3G 的前 4 个能级间距远大于化学精度，通常不触发。

### 1.3 单向冻结与冻结后的漂移监控

被冻结的 $\theta_j$ 不再更新，但 $E_L$ 的本征值仍依赖**所有**列（$E_L = \Psi^{-1}H\Psi$），活跃列继续移动可能使冻结能级 $E_j$ 的估计值缓慢漂移（这正是 §2.5 讨论的"$E_j$ 与所有列耦合"的体现）。

**策略为单向冻结（与当前代码 `He_atom_K4_train_frozen.py` 一致）**：

- 冻结后继续按步计算 $\epsilon_j$（成本为零，本来就在监控）——**只监控、不激活**；
- **没有解冻路径**：`frozen_set` 只增不减。这是有意为之：若"一漂移就解冻"，收益（反向剪枝 + QGT 降维）会被反复的 JIT 重编译与状态反复消耗掉；且 300 轮实验表明，E1 冻结后即使漂移到 ~1.4 mHa 仍在化学精度内；
- 若担心"冻错能级"，更稳妥的做法是把冻结判据收紧（§2.5 的列-态重叠判据 C7），而不是事后引入回退。

### 1.4 终止条件

若 $|$ `frozen_set` $| = K$：全部能级达标，训练终止（`break`），不再有可优化参数。

### 1.5 判据实现草图

```python
from collections import deque

CHEM_ACC = 1.6e-3        # Ha
FREEZE_WINDOW = 10       # C2：连续步数（= RESET_PERIOD）
WARMUP = 30              # C5
COOLDOWN = FREEZE_WINDOW

err_hist = {j: deque(maxlen=FREEZE_WINDOW) for j in range(K)}
frozen_set = set()                  # 冻结的能级编号（只增不减，单向冻结）
last_freeze_step = -10**9

def maybe_freeze(step, eig_vals_re, valid_ratio, grad_finite, frozen_set):
    """eig_vals_re: 长度 K 的 numpy 数组（已按 Re 排序）"""
    if len(frozen_set) == K:
        return frozen_set
    errors = jnp.abs(eig_vals_re - eigvals)          # eigvals: FCI 参考 ndarray

    # --- 冻结检查（单向：只加不减，无解冻路径）---
    if step >= WARMUP and step - last_freeze_step >= COOLDOWN and grad_finite:
        for j in range(K):
            if j in frozen_set or j == last_freeze_candidate_guard:
                continue
            err_hist[j].append(float(errors[j]))
            e = err_hist[j]
            if len(e) == FREEZE_WINDOW and all(v < CHEM_ACC for v in e):
                if jnp.std(jnp.asarray(e)) < 0.5 * CHEM_ACC \
                   and valid_ratio >= 0.25 \
                   and (j == 0 or abs(eig_vals_re[j] - eig_vals_re[j-1]) > 2 * CHEM_ACC) \
                   and (j == K-1 or abs(eig_vals_re[j+1] - eig_vals_re[j]) > 2 * CHEM_ACC):
                    frozen_set.add(j)
                    err_hist[j].clear()
                    logger.info(f"[Freeze@{step}] 能级 {j} 冻结 | err={errors[j]:.2e} Ha")
                    break   # 每次只冻结一个，冷却期内观察影响
    return frozen_set
```

> 实现注记：`last_freeze_candidate_guard` 一类细节以最终代码为准；每次冻结事件之间用 `COOLDOWN` 间隔即可，不需要复杂状态机。

---

## 2. 问题二：如何冻结部分参数而不影响整体算法

### 2.1 数学层面：冻结 = 约束变分，算法结构不变

把参数分成活跃集 $\theta_A$ 与冻结集 $\theta_F$。冻结后的优化问题从

$$
\min_{\theta} \; \mathcal{L}(\theta) = \mathrm{tr}\big(\Psi_\theta^{-1} H \Psi_\theta\big)
\quad\Longrightarrow\quad
\min_{\theta_A}\; \mathcal{L}(\theta_A, \theta_F^{(0)})
$$

即**在冻结点处的约束变分**。变分原理、NES 的梯度公式、$E_L$ 估计器、采样分布 $|\Psi|^2$ 全都不变——唯一的区别是 $\partial\mathcal{L}/\partial\theta_F \equiv 0$ 不再被计算和应用。

**逐项核对"不影响"**：

| 环节 | 是否受影响 | 说明 |
|---|---|---|
| 采样 Metropolis $|\Psi|^2$ | 否 | 前向 $\Psi$ 仍包含全部 K 列（冻结列仍参与行列式） |
| Loss / $E_L$ 矩阵 | 否 | 前向值不变；`stop_gradient` 只改反向图，前向恒等 |
| 梯度公式 | 否 | 活跃分量的梯度与不冻结时**逐位一致**（见 2.3 验证） |
| QGT / SR | 否 | 活跃块 $S_{AA}$ 与全量 $S$ 的对应子块一致；解出的是同一约束问题的自然梯度 |
| gauge reset | 否 | `col_mean_fn` 仍对全部 K 列做列均值吸收。**冻结列也必须继续吸收**：活跃参数更新会改变采样分布 $|\Psi|^2$，冻结列的 raw L 列均值会随分布漂移，不吸收会导致数值不稳定 |
| 优化器 | 否 | SGD（无动量）下零梯度 ⇒ 零更新 ⇒ 参数严格不动；`clip_by_global_norm` 对零分量不敏感（0 不贡献范数），裁剪结果与"只在活跃集上裁剪"完全一致 |

### 2.2 工程层面：三个改动点

现有代码中梯度只经过一条路径：`dlogΨ = jax.grad(total_machine)`（[#L332-L334](../experiments/LiH分子/NES_VMC_train_min120_gauge_reset%20copy.py#L332-L334)），而 `Ham_Psi_scaled` 中的 HΨ 本来就不参与反向（其结果只作为权重 `tr_centered` 的构成，权重侧已 detach）。因此**只需要在 $\Psi$ 侧动手**。

#### 改动 ①：`_compute_L_centered_single` 中对冻结列加 `stop_gradient`

`frozen_mask` 作为**静态 Python tuple** 传入（触发一次性重编译，见 2.4）：

```python
def _compute_L_centered_single(m, x_single, g, with_gauge, frozen_mask=()):
    cols = []
    for j in range(K_):
        ansatz_j = m.single_ansatz_list[j]
        log_col = ansatz_j(x_single)
        log_ref = ansatz_j(ref_state)
        log_col_centered = log_col - log_ref
        if frozen_mask[j]:
            # 冻结列：前向值不变，反向在列输出处被阻断
            log_col_centered = jax.lax.stop_gradient(log_col_centered)
        cols.append(log_col_centered)
    L_centered = jnp.stack(cols, axis=1)
    if with_gauge:
        L_centered = L_centered - jax.lax.stop_gradient(g)[None, :]
    return L_centered
```

效果：slogdet 的余切传到 $L[:, j]$ 处遇到屏障即终止，冻结子网络内部的全部 VJP **不会被生成、不会被执行**（这是真减负的一半，详见第 3 节）。

#### 改动 ②：QGT 的 O 矩阵按活跃索引切片

`make_qgt_fn_gauge` 中 O 的宽度是全参数数 $P$（含冻结参数对应的零列）。预先计算活跃索引（每次冻结配置变化时在 Python 侧算一次）：

```python
import numpy as np

def build_active_index(total_params, frozen_set):
    """按 ravel_pytree 的扁平化顺序，取活跃参数的全局索引"""
    per_level = [ravel_pytree(total_params["single_ansatz_list"][j])[0] for j in range(K)]
    sizes = [len(f) for f in per_level]
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    idx = np.concatenate([
        np.arange(offsets[j], offsets[j] + sizes[j])
        for j in range(K) if j not in frozen_set
    ])
    return jnp.asarray(idx)          # shape (P_a,)
```

`make_qgt_fn_gauge` 增加一个 `idx_active` 参数：

```python
def make_qgt_fn_gauge(machine):
    grad_logpsi = jax.grad(machine, argnums=0, holomorphic=True)
    vmap_grad_logpsi = jax.vmap(grad_logpsi, in_axes=(None, 0, None))

    @jax.jit
    def qgt_fn(params, sigma, diag_shift, g, idx_active):
        n_samples = sigma.shape[0]
        grad_tree_batch = vmap_grad_logpsi(params, sigma, g)
        O = flatten_batched_pytree(grad_tree_batch, n_samples)   # (n, P)
        O = O[:, idx_active]          # ← 唯一改动：切到活跃列 (n, P_a)
        O_mean = jnp.mean(O, axis=0, keepdims=True)
        O_centered = O - O_mean
        S = (jnp.conj(O_centered).T @ O_centered) / n_samples
        S = 0.5 * (S + jnp.conj(S.T))
        I = jnp.eye(S.shape[0], dtype=S.dtype)
        return S + diag_shift * I
    return qgt_fn
```

冻结参数的梯度列本来就是 0（被 `stop_gradient` 屏障阻断），切片**不改变** $S_{AA}$ 的数值，只缩小矩阵规模。

#### 改动 ③：训练循环中 flat 向量的切片与回填

```python
grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)      # (P,)，冻结分量已是 0
...
if Natural_Grad:
    qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift, g_current, idx_active)
    grad_a = grad_raw_flat[idx_active]                  # (P_a,)
    ng_a = jnp.linalg.solve(qgt_reg_mat, grad_a)        # O(P_a^3)
    ng_flat = grad_raw_flat.at[idx_active].set(ng_a)    # 回填，冻结位保持 0
    grad_update = unravel_fn(ng_flat)
```

`optimizer.update` / `optax.apply_updates` 不需要任何改动：冻结分量梯度为 0 ⇒ SGD 更新为 0 ⇒ `apply_updates` 加 0 后参数**逐位不变**（若要求绝对严格，可换 `optax.masked`，但对无动量 SGD 二者数值等价）。

### 2.3 不改变算法轨迹的证明要点

1. **前向恒等**：`stop_gradient` 前向是恒等映射 ⇒ loss、$E_L$、$E_j$、采样比率全部逐位不变。
2. **活跃梯度不变**：对活跃参数的余切传播路径不经过冻结列的屏障（屏障只阻断流向冻结支路的那一支），且 $\partial \mathcal{L}/\partial\theta_A$ 的解析表达式与全参数问题相同。
3. **裁剪不变**：$\|g\|_2$ 在补零后与活跃集范数相等 ⇒ `clip_by_global_norm` 缩放因子相同。
4. **更新不变**：SGD $\Delta\theta = -\eta\cdot \mathrm{clip}(g)$，冻结位 $g=0 \Rightarrow \Delta\theta=0$。
5. **经验证**：取一个未达判据的能级强制置入 `frozen_mask` 跑 10 步，与基线（空 frozen_mask）对比 loss / $E_0$–$E_3$ / 参数哈希，应**逐位一致**（这是上线路前的强制回归测试，见 3.4）。

### 2.4 JIT 与重编译

- `frozen_mask` 是静态 Python tuple ⇒ 每种冻结配置编译一次。冻结事件全程 ≤ K 次，编译开销摊销后可忽略。
- `idx_active` 形状随冻结变化 ⇒ `qgt_fn` 重编译，同样是每事件一次。
- **前向 machine（采样用）也含 `frozen_mask`**：为避免采样路径重编译，可以让 `total_machine` 保持无 mask 版本，仅给梯度路径用的 machine 传 mask。两种做法都对，推荐后者（把 mask 只加在 `make_grad_fn_gauge`/`make_qgt_fn_gauge` 用的 machine 上），采样器完全不动。
- 断点续训：把 `frozen_set`、冻结时刻、各能级误差历史一并写入现有 pickle 历史文件；加载后先重建 `frozen_mask`/`idx_active` 再继续。

### 2.5 关键辨析：E_1 由 E_L 对角化得到，直接冻结 psi_1 合理吗？

> 这是分段冻结策略**最需要诚实的部分**。用户疑问：E_1 并不是"psi_1 的能量"，而是对整块 $E_L$ 矩阵对角化、排序后取倒数第二小的本征值；那么冻结 psi_1 的参数，真的能"锁住 E_1"吗？

**结论**：这是一个**高效但近似的工程假设**，并非严格恒等。它成立有前提条件（自然规范），且有失效场景。逐条论证如下。

#### (a) 为什么 E_1 不专属 psi_1

能量估计路径（[He_atom_K4_train_frozen.py](../experiments/He原子/He_atom_K4_train_frozen.py#L570-L571)）：

```python
eig_vals, _ = jnp.linalg.eig(E_L_mean)       # 整块 K×K 平均局部能量矩阵
eig_vals = eig_vals[jnp.argsort(eig_vals.real)]   # 升序，E_1 = 倒数第二小
```

其中 $E_L = \Psi^{-1} H \Psi$，$\Psi = [\exp f_j(x_i)]$ 是**全部 K 列**构成的行列式。对本征值 $E_k$（取 Hermitize 后的对称化矩阵），
$$
\mathrm{d}E_k \;=\; v_k^\dagger\, \mathrm{d}E_L\, v_k, \qquad
\mathrm{d}E_L = \Psi^{-1}\big[ H\,\mathrm{d}\Psi - \mathrm{d}\Psi\, E_L\big],
$$
$\mathrm{d}\Psi/\mathrm{d}\theta_j$ 虽只支撑在第 $j$ 列，但左乘稠密的 $\Psi^{-1}$、右乘 $H$/$E_L$ 后，$\mathrm{d}E_L$ 是满矩阵，再被 $v_k$ 双向投影。因此**一般地**：
$$
\frac{\partial E_k}{\partial \theta_j} \neq 0 \quad (\forall\, j,k),
$$
即每个能级 $E_k$ 的导数都依赖**所有列**的参数。冻结 psi_1 只让 psi_1 的参数停止更新，E_1 仍随 psi_0、psi_2、psi_3 演化而漂移——300 轮实验已见：step 85 冻结 E1 后，E1 误差从 ~0 缓慢漂到 1.41 mHa（仍在化学精度内）。

#### (b) 何时近似成立（"冻结合理"的前提）

行列式基存在**残余 K×K 列规范自由度**：对任意与构型无关的可逆矩阵 $M$，$\Psi \to \det(M)\,\Psi$（常数因子被归一化吸收），物理态不变。因此"第 $j$ 列"本身不是规范不变的；只有在**自然规范**——列基恰为 $E_L$ 的本征基，即 $E_L$ 在列基下近似对角——时，"第 $j$ 列 ≈ 第 $j$ 个物理态"才成立。此时 $v_k \approx e_k$，于是：
- $\partial E_k/\partial \theta_k$ 占主导，冻结 psi_1 近似把态 1 钉住；
- 冻结后 E_1 的漂移是其余列非对角耦合的**二阶小量**，通常远小于化学精度。

当前实现用"按 $\mathrm{Re}$ 升序配对 + C6 能级间隔判据"来**默认**这种近对角假设成立，但并未显式验证。

#### (c) 严格化建议：冻结前验证"列-态对应"（判据 C7）

在 C1–C6 基础上追加一条可操作的重叠判据，只有当目标列确实是该能级本征向量的主导分量时才冻结：

```python
eig_vals, eig_vecs = jnp.linalg.eig(E_L_mean)      # eig_vecs: (K, K)，第 k 列 = 态 k 在列基下的系数
eig_vecs = eig_vecs[:, jnp.argsort(eig_vals.real)] # 按能量升序对齐
overlap = np.abs(eig_vecs[:, k])[j]                # 目标能级 k 在列 j 上的局域化程度
# C7：overlap > overlap_thresh（如 0.9） 才允许把"列 j"与"态 k"绑定并冻结
```

若 overlap 不足（简并或列基明显未对角化），应**不冻结**，或先把列基旋转到本征基（把单列网络重参数化为 $v_k^T\psi$ 的组合）再冻。

#### (d) 结论

1. 冻结 psi_1 的**减负收益是真实的**（反向剪枝 + QGT $P^2\to P_a^2$，已实测 25% 参数缩减）；
2. 但对 E_1 的"锁定"是**近似**：E_1 由整块 $E_L$ 对角化得到，与所有列耦合；在自然规范（列基≈本征基）下近似成立、漂移为二阶小量；
3. 工程上当前策略可保留（高效近似），但必须满足：冻结后**每步仍对完整 $E_L$ 对角化以持续监控 E_1**（当前实现天然如此——冻结只作用于梯度/QGT，前向行列式与能量估计始终用全部 K 列）；若要更严格，加入 (c) 的 C7 重叠判据。

---

## 3. 问题三：如何确保被冻结的参数"真减负"而非假减负

### 3.1 先明确什么是假减负

以下做法**看起来**冻结了参数，实际计算量一点没省：

| 假减负做法 | 为什么没省 |
|---|---|
| `grad *= 0` 或把梯度叶子乘 mask | `jax.grad` 的反向图完整执行，冻结网络的 VJP 照算，只是结果被清零 |
| `optax.masked` / `optax.multi_transform` | 只影响**优化器更新**阶段（逐叶子的 elementwise 运算），前向反向照旧 |
| QGT 不切片，靠零行列"撑大"矩阵 | $S$ 仍是 $P\times P$，`solve` 仍是 $O(P^3)$ |

### 3.2 本方案省掉的是什么：逐项成本核算

记每步成本的关键项（$n$ = walker 数，$P$ = 全参数数，$P_a$ = 活跃参数数，$m$ = 已冻结能级数，$P_a = P(1-m/K)$）：

| 环节 | 成本 | 冻结后 | 说明 |
|---|---|---|---|
| 采样（Metropolis 前向） | $O(n\cdot K\cdot C_{\mathrm{fwd}})$ | **不变** | 冻结列仍需前向——$|\Psi|^2$ 依赖全部列，无法省，也不应省 |
| L 矩阵前向 / slogdet / HΨ 前向 | $O(n\cdot K\cdot C_{\mathrm{fwd}})$ | **不变** | 同上，能量估计器需要完整 $\Psi$ 与 $H\Psi$ |
| **dlogΨ 反向（VJP）** | $O(n\cdot K\cdot C_{\mathrm{bwd}})$ | **冻结支路被剪除** | `stop_gradient` 屏障使冻结子网络的 VJP 不进入 jaxpr 反向图；活跃支路的余切传播照常 |
| **QGT 构建（$O^\dagger O$）** | $O(n P^2)$ | $O(n P_a^2)$ | O 切片后矩阵乘法规模缩小 |
| **QGT 求解（`solve`）** | $O(P^3)$ | $O(P_a^3)$ | SR 主导成本，收益最大 |
| 优化器 elementwise | $O(P)$ | 不变（可忽略） | |

**定量估算（K=4）**：冻结 1 / 2 / 3 个能级后，QGT 求解成本比例 $(1-\frac{m}{4})^3$：

| 已冻结 m | $P_a/P$ | QGT 矩阵乘 | QGT solve |
|---|---|---|---|
| 1 | 0.75 | 56% | **42%** |
| 2 | 0.50 | 25% | **12.5%** |
| 3 | 0.25 | 6% | **1.6%** |

对长训练（目标上千步）而言，QGT 构建 + 求解通常占每步 wall time 的主要部分，这是"训练越到后段越快"的来源——与构想文档的预期一致。

### 3.3 为什么 `stop_gradient` 是"真剪枝"而不是"算了再乘零"

JAX 的 `jaxpr` 是显式的：`vjp` 变换会为前向 jaxpr 中**从输出余切可达**的原始操作生成反向代码。`stop_gradient` 的 VJP 返回全零且**不再向输入方向传播余切**，因此屏障以下的全部层（冻结子网络的 dense、激活、甚至其 VJP 中代价较高的复数运算）在转置后的 jaxpr 中不可达 ⇒ 反向代码根本不会为它们生成。这与"算完再乘 0"有本质区别：后者反向代码完整存在并每次执行。

唯一仍需付出的冻结列成本是**前向**——而前向是物理上必需的（$\Psi$ 矩阵必须完整），所以本方案已经达到该算法结构下的减负上限。

### 3.4 验证清单（上线路前逐项打勾）

1. **Shape 断言**
   - `qgt_reg_mat.shape == (P_a, P_a)`；
   - `idx_active.shape[0] == P_a` 且 `set(idx_active.tolist()) ∪ 冻结索引 == 全集`（无遗漏无重叠）。

2. **逐位回归测试（轨迹不变性）**
   - 用一个随机小模型，分别以 `frozen_mask=()` 与 `frozen_mask=(0,)*K 但该能级未达标` 跑 10 步，断言每步 loss、$E_0$–$E_3$、活跃参数哈希**逐位一致**（`stop_gradient` 前向恒等 + 2.3 的推理 ⇒ 必须逐位一致，不一致说明实现有 bug）。

3. **jaxpr / 剪枝检查**
   - 对 `grad_fn` 打印 `jax.make_jaxpr`，对比空 mask 与冻结 mask 的 jaxpr：冻结配置的反向图中，冻结子网络对应的原语数量应显著减少（屏障以下消失）。
   - 或更直接：`jax.profiler` 记录一次 `grad_fn` 调用，对比冻结前后 FLOPs / op 数。

4. **Wall-time 计时对比**
   - 在日志中按段计时（采样 / grad_fn / qgt+solve / 更新），现有训练循环已天然可插入 `time.perf_counter()`。
   - 预期：冻结事件后 `qgt+solve` 段时长按 3.2 的比例下降，`grad_fn` 段小幅下降，采样段不变。
   - 注意：冻结当步有一次性编译开销，对比应取冻结后 ≥10 步的窗口均值。

5. **参数不变性断言**
   - 每步对冻结叶子做哈希（`hash(grad_raw_flat[idx_frozen].tobytes())` 级别即可），断言 `apply_updates` 后冻结参数逐位不变。

6. **单向冻结演练**
   - 人为把某能级的判据阈值调松（如 10×化学精度）强制提前冻结，观察冻结后该能级误差的漂移幅度；确认漂移仍在化学精度内、且 `frozen_set` 之后不再变化（无解冻事件）。

### 3.5 减负的边界（诚实声明）

- 冻结**不减少**采样与能量前向的成本（物理上不可省），减负只发生在**反向 + QGT**。若某天采样成为瓶颈（例如 K 很大时），需要另找办法（如冻结列的 $\psi_j$ 预查表——仅当输入构型离散且重复率高时可行），不在本方案范围内。
- JIT 重编译是每次冻结事件的一次性成本（几秒到几十秒），事件总数 ≤ K（单向冻结，无解冻），冷却期（1.2 节 C5）保证不会频繁触发。

---

## 4. 实施步骤（在 `NES_VMC_train_min120_gauge_reset copy.py` 基础上的改动清单）

1. **新增常量与状态**：`CHEM_ACC`、`FREEZE_WINDOW`、`WARMUP`、`COOLDOWN`、`frozen_set`、各能级误差 `deque`（**单向冻结**：无 `UNFREEZE_WINDOW`、无解冻路径）。
2. **`create_gauge_reset_total_machines`**：`_compute_L_centered_single` 增加 `frozen_mask` 静态参数与 `stop_gradient` 分支；返回一个**带 mask 的梯度用 machine**（或让 machine 工厂接受 mask 参数再生成一份），采样用的前向 machine 保持无 mask。
3. **`make_grad_fn_gauge` / `nes_vmc_gradient_stable_gauge`**：换用带 mask 的 machine；其余不动（冻结叶子梯度自动为 0）。
4. **`make_qgt_fn_gauge`**：增加 `idx_active` 参数并切片 O（见 2.2 改动②）。
5. **训练循环**：
   - 每步末尾调用 `maybe_freeze`（1.5 节），更新 `frozen_set`；
   - 冻结集合变化时：重建 `frozen_mask`、`idx_active`，记录日志与耗时；
   - `grad_raw_flat` 切片/回填逻辑（2.2 改动③）；
   - 日志增加：`|frozen_set|`、各能级 $\epsilon_j$、分段计时；
   - 全冻结时 `break`。
6. **持久化**：把 `frozen_set` 与冻结历史写入现有 pickle 历史文件，加载时恢复。
7. **回归测试**：执行 3.4 节清单 1–5。

---

## 5. 风险与注意事项

| 风险 | 缓解 |
|---|---|
| 冻结列的列均值随采样分布漂移 | gauge reset 对全部 K 列继续生效（2.1 表格），`col_mean_fn` 不做任何裁剪 |
| 近简并能级序号交换导致误判 | 判据 C6：与相邻能级间距 $>2\delta_{\mathrm{chem}}$ 才允许冻结 |
| 冻结后能级漂移 | 单向冻结下漂移不会被"纠正"，但漂移为二阶小量（§2.5），且 $E_j$ 每步持续监控；若担忧漂移超化学精度，改用更保守的冻结判据（C7 列-态重叠）而非回退 |
| JIT 抖动 | 单向冻结 + 冷却期（两次冻结事件间隔 ≥ W），事件数上限 K，无反复重编译 |
| 断点续训状态不一致 | `frozen_set` 必须随 pickle 一起保存/恢复，恢复后先重建 mask 与 `idx_active` |
| `stop_gradient` 误加到 ref 列以外的地方（如 g） | g 本来就是 `stop_gradient`（P8 设计），不动；mask 只作用于列输出 |
