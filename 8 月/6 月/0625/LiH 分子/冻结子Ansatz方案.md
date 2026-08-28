# 冻结第一个子 Ansatz 参数的实现方案

> 适用代码：[NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py) + [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb)

## 1. 需求拆解

| 维度 | 冻结前 | 冻结后 (step ≥ N) |
| ---- | ------ | ----------------- |
| 前向 log Ψ / L 矩阵 / shift | 全部 sub-ansatz 参与 | **全部 sub-ansatz 仍参与** |
| Loss / E_L | 全部 sub-ansatz 参与 | **全部 sub-ansatz 仍参与** |
| 采样 (NetKet sampler) | 用当前参数采样 | **用当前（已冻结）参数采样** |
| 梯度 ∂Loss/∂θ₀ | 正常计算 | **强制为 0** |
| QGT / 自然梯度 | 包含 θ₀ 行列 | **θ₀ 对应行列置零**（避免无意义求解） |
| 参数更新 | θ_i ← θ_i − η·∂ | **θ₀ 不动** |

核心思路：**梯度 → mask 为 0；QGT → mask 行列；Loss/E_L → 完全不动。**

---

## 2. 关键代码定位

| 位置 | 作用 |
| ---- | ---- |
| `NES_VMC.py:110-160` | `NESTotalAnsatz` / `NESTotalAnsatz_gauge_stable`，参数挂在 `single_ansatz_list[i]` 下 |
| `NES_VMC.py:376-413` | `create_machine*`，从 `nnx.split(model)` 得到 `(graphdef, state)`，这里 `state` 就是 `total_params` |
| `NES_VMC.py:740-904` | `nes_vmc_gradient_stable`，对全部 sub-ansatz 算梯度 |
| `NES_VMC.py:1477-1498` | `make_qgt_fn`，对全部 sub-ansatz 算 QGT |
| `LiH 规范漂移-1.ipynb:2459-2509` | 训练循环（采样 → grad → QGT → 自然梯度 → clip → apply_updates） |

`total_params` 的 PyTree 结构（简化）：

```text
total_params = {
  "single_ansatz_list": [
    { "linear1": {"kernel": (..., ...), "bias": (...)},   # ← 想冻结的 sub-ansatz 0
      "linear2": {...}, "output": {...} },
    { "linear1": {...}, ... },                              # sub-ansatz 1
    ...
  ]
}
```

冻结时只需把 `total_params["single_ansatz_list"][0]` 整棵子树 mask 成 False 即可。

---

## 3. 实现方案（推荐：mask + optax 双重保险）

### 3.1 在 `NES_VMC.py` 末尾新增两个工具函数

```python
# ============================================================
# 子 Ansatz 冻结工具
# ============================================================
def build_subansatz_mask(total_params, freeze_idx: int):
    """
    返回一个与 total_params 同结构的 bool PyTree：
    True  = 该 leaf 仍参与梯度/QGT/更新
    False = 该 leaf 已冻结（梯度、QGT 行列、optimizer 更新都跳过）

    实现：先把整棵树设为 True，再把 freeze_idx 那一支整体替换为 False。
    """
    ones = jax.tree.map(lambda x: jnp.ones(x.shape, dtype=bool), total_params)
    frozen_zeros = jax.tree.map(
        lambda x: jnp.zeros(x.shape, dtype=bool),
        total_params["single_ansatz_list"][freeze_idx],
    )
    # nnx.split 出来的 state 是 frozen Mapping，
    # 重新构造新 dict 避免 in-place 报错
    new_list = list(total_params["single_ansatz_list"])
    return {"single_ansatz_list": [
        frozen_zeros if i == freeze_idx else new_list[i]
        for i in range(len(new_list))
    ]}


def mask_pytree(tree, mask):
    """对 PyTree 乘 mask（mask=True 保留，False 置零）"""
    return jax.tree.map(lambda x, m: x * m.astype(x.dtype), tree, mask)


def zero_qgt_rows_cols(qgt_flat_dim, unravel_fn, mask_flat, diag_shift):
    """
    把展平后 QGT 中 mask=False 对应的行/列置 0，再保留 diag_shift。
    等价于"冻结子 ansatz 不参与 QGT 求解"。
    """
    n = qgt_flat_dim
    idx = jnp.arange(n)
    keep = jnp.where(mask_flat, 1.0, 0.0)            # (n,)
    row_col_mask = keep[None, :] * keep[:, None]     # (n, n)
    S = qgt_flat_dim  # 占位，下面用真实参数重写
    return S  # 见下面 main 写法
```

更直观的写法（推荐放进训练循环）：

```python
def mask_qgt_matrix(S_reg, mask_flat, diag_shift):
    """
    S_reg:   (n, n)  正则化 QGT
    mask_flat: (n,)  bool, True=活动
    返回: 把 mask=False 的行/列置 0，对角线加回 diag_shift 保证可逆
    """
    keep = mask_flat.astype(S_reg.dtype)            # (n,)
    row_col = keep[:, None] * keep[None, :]         # (n, n)

    S_masked = S_reg * row_col
    # 在被置零的对角线位置上补回 diag_shift，让子矩阵仍可求逆
    diag_shift_vec = jnp.diag(row_col)             # 被 mask 掉的位置是 0
    S_masked = S_masked + diag_shift * (1.0 - diag_shift_vec) * jnp.eye(S_masked.shape[0], dtype=S_masked.dtype)
    return S_masked
```

### 3.2 在 notebook 的训练循环里嵌入

```python
# ====================== 冻结配置 ======================
FREEZE_STEP   = 300   # 训练到第 300 步后冻结 sub-ansatz 0
FREEZE_IDX    = 0

# 一次性构造 mask，PyTree 结构与 total_params 完全一致
param_mask = build_subansatz_mask(total_params, FREEZE_IDX)
mask_flat, _ = jax.tree_util.tree_flatten(param_mask)
mask_flat_bool = jnp.concatenate(
    [m.reshape(-1) for m in mask_flat]
).astype(bool)   # (n_params,)
```

训练循环核心改动（节选自 `LiH 规范漂移-1.ipynb:2459-2509`）：

```python
for step in range(N_ITER):
    # ---------- 采样：完全不动 ----------
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine_log_Psi_gauge,
        parameters=total_params,
        state=sampler_state,
        chain_length=N_SAMPLES_PER_CHAIN,
    )
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # ---------- 1. 原始梯度 ----------
    grad_raw, loss_mean, E_L_mean, aux = grad_fn(total_params, x_batch)
    #                       ^^^^^^^^^  ↑↑↑ 注意：Loss / E_L 始终包含所有 sub-ansatz

    # ★ 新增：冻结步后把 sub-ansatz 0 的梯度置 0
    if step >= FREEZE_STEP:
        grad_raw = mask_pytree(grad_raw, param_mask)

    grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    grad_norm_raw = jnp.linalg.norm(grad_raw_flat)

    # ---------- 2. QGT 自然梯度 ----------
    if Natural_Grad:
        qgt_reg_mat = qgt_fn(total_params, x_batch, qgt_diag_shift)

        # ★ 新增：冻结步后 QGT 中 sub-ansatz 0 对应行列置零
        if step >= FREEZE_STEP:
            qgt_reg_mat = mask_qgt_matrix(qgt_reg_mat, mask_flat_bool, qgt_diag_shift)

        ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)

        # ★ 新增：自然梯度里冻结部分仍是 0（数值误差防护）
        if step >= FREEZE_STEP:
            ng_flat = ng_flat * mask_flat_bool.astype(ng_flat.dtype)

        grad_update = unravel_fn(ng_flat)
        grad_norm_natural = jnp.linalg.norm(ng_flat)
    else:
        grad_update = grad_raw
        grad_norm_natural = grad_norm_raw

    # ---------- 3. 裁剪 + 更新 ----------
    updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
    #                                          ^^^^^^^^^ grad_update 里 sub-ansatz 0 已经是 0
    #                                          所以 updates 里 sub-ansatz 0 也是 0

    # total_params 里 sub-ansatz 0 不会变（add 0），其他照常更新
    total_params = optax.apply_updates(total_params, updates)

    # 后续的 E_L、log_Psi、Ψ_cond 等监控完全不用改
```

### 3.3 替代方案：直接用 `optax.masked`（更省事，但 QGT 仍含冻结行列）

如果你**不在意** QGT 矩阵里继续保留冻结子 ansatz 对应的行列（自然梯度求解出来反正也是 0），可以用 optax 的 mask：

```python
import optax

# 在原 optimizer 定义处替换：
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    # 只对 mask=True 的叶子（未被冻结的 sub-ansatz）应用 SGD
    optax.masked(optax.sgd(learning_rate=lr), param_mask),
)
opt_state = optimizer.init(total_params)
```

加上"梯度 mask"那一步（3.2 中的 `grad_raw = mask_pytree(...)`）即可。
这种写法**比 3.2 的 QGT mask 简单**，代价是 QGT 矩阵仍然是完整尺寸，求解时 N² 复杂度没省下来，但 Frozen 子 ansatz 对应的自然梯度数值上仍然正确（=0），效果等价。

---

## 4. 为什么 Loss / E_L 不用动？

`NES_VMC.py:740-904` 的 `nes_vmc_gradient_stable` 调用链：

```text
grad_fn(params, x_batch)
  └─ NES_loss_energy_stable(...)
        ├─ L_stable       = total_matrix_machine(params, x)   # 包含 sub-ansatz 0
        ├─ shift          = total_max_machine(params, x)      # 包含 sub-ansatz 0
        ├─ HPsi_stable    = Ham_Psi_scaled(...)               # 包含 sub-ansatz 0
        ├─ Psi_stable     = exp(L_stable)                     # 包含 sub-ansatz 0
        └─ E_L = solve(Psi_stable, HPsi_stable)              # 包含 sub-ansatz 0
```

`params` 始终是**完整**的 `total_params`，所以 forward 路径上 sub-ansatz 0 仍然有数值贡献。
我们只对**梯度**和**QGT**动刀 → forward 行为不变 → Loss / E_L 数值不变。

---

## 5. 验证清单

冻结后请依次检查：

1. **采样分布不变**：冻结前后 `log_Psi` 的 mean / min / max 不应有突变。
2. **Loss / E_L 数值连续**：冻结那一帧的能量相对上一帧的 |Δ| 应在噪声量级。
3. **冻结子 ansatz 0 的参数**：连续打印 `total_params["single_ansatz_list"][0]["linear1"]["kernel"]` 的 norm，冻结后必须完全不动。
4. **其他 sub-ansatz 仍正常更新**：norm 单调下降 / 有合理学习曲线。
5. **QGT 条件数**：冻结后 `S_cond` 应当下降（少了一堆零特征方向），不会再有"按不下去"的现象。
6. **整体能量曲线**：基态能量应当继续收敛，激发态（特别是 sub-ansatz 0 对应的态）能量应在冻结后被锁住。

可选的诊断打印（放在训练循环里）：

```python
if step >= FREEZE_STEP and step % 50 == 0:
    frozen_norm = jnp.linalg.norm(ravel_pytree(
        total_params["single_ansatz_list"][FREEZE_IDX]
    )[0])
    active_grad_norm = jnp.linalg.norm(grad_raw_flat * mask_flat_bool)
    frozen_grad_norm = jnp.linalg.norm(grad_raw_flat * (~mask_flat_bool))
    logger.info(
        f"[step {step}] frozen_param_norm={frozen_norm:.6f} "
        f"active_grad_norm={active_grad_norm:.4f} "
        f"frozen_grad_norm={frozen_grad_norm:.2e}"
    )
    # 期望: frozen_grad_norm 恒为 0, frozen_param_norm 恒定
```

---

## 6. 注意事项 & 坑

1. **NNX state 是 Frozen Mapping**：`nnx.split` 返回的 `state` 不可直接 `__setitem__` / `append`。
   `build_subansatz_mask` 中用 `list(...)` 重建列表就是为绕开这点。
2. **mask 形状必须严格对齐**：`build_subansatz_mask` 用 `jax.tree.map` 在每个 leaf 上生成 mask，确保 shape 一致。
3. **bool dtype**：`optax.masked` 接收的是 `PyTree` of `bool`（或 0/1），不要传 `int` 容易在某些版本下出错。
4. **QGT mask 的对角线**：`mask_qgt_matrix` 中给被置零的对角线补回 `diag_shift`，否则该子矩阵奇异，`jnp.linalg.solve` 会 NaN。
5. **如果用 0706 / 数值稳定版 notebook**：把上面的 patch 直接复制到对应训练循环里即可，结构相同。
6. **多次冻结 / 解冻**：当前实现只支持"冻一次"。如果想动态开关，把 `if step >= FREEZE_STEP` 换成 `is_frozen[step]` 查表函数即可。

---

## 7. 一图流（修改位置）

```text
LiH 规范漂移-1.ipynb
└── for step in range(N_ITER):          ← 在循环开头
    ├── nes_sampler.sample(...)         # 1. 采样：不动
    ├── grad_fn(total_params, x_batch)  # 2. forward：不动（Loss/E_L 自动含 sub-ansatz 0）
    │       ↓
    │   grad_raw                        # 3. ★ 冻结后 mask_pytree(grad_raw, param_mask)
    │       ↓
    │   grad_raw_flat
    │
    ├── qgt_fn(...)                     # 4. ★ 冻结后 mask_qgt_matrix(S, mask_flat, diag_shift)
    │       ↓
    │   ng_flat                         # 5. ★ 冻结后 * mask_flat_bool
    │       ↓
    │   unravel_fn(ng_flat)
    │       ↓
    │   optimizer.update(...)           # 6. 已有 grad_update=0 → updates=0
    │       ↓
    │   apply_updates(total_params, updates)  # 7. sub-ansatz 0 add 0 = 不动
    │
    └── 监控 / Loss / log_Psi / E_L：完全不动
```

按这个流程改，**前向完全不受影响**，**优化器只对未被冻结的 sub-ansatz 起作用**，冻结那支子 ansatz 仍然作为"基态锚点"参与波函数构造，但不再被训练扰动。
