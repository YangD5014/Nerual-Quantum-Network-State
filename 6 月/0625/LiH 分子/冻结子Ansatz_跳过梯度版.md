# 真正"跳过"第一个子 Ansatz 的梯度计算（不仅 mask）

> 配套文档：[冻结子Ansatz方案.md](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/冻结子Ansatz方案.md)（mask 版）  
> 适用代码：[NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py) + [LiH 规范漂移-1.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH 规范漂移-1.ipynb)

## 0. 先回答你两个问题

**Q1：能实现吗？**  
能。JAX 提供 `jax.lax.stop_gradient` / `jax.lax.cond` / 自定义 `jax.grad` 三种粒度都能让 JAX 在追踪计算图时**根本不知道**被冻结参数的导数路径,从而省掉 forward 之后那部分 backward。

**Q2：Loss / 梯度表达式会变吗？**  
**Loss 表达式 = 0 变化。梯度表达式对未冻结参数 = 0 变化,对冻结参数 = 本来就为 0,只是不计算了。**  
具体解释见下表。

| 表达式 | 冻结前 | 冻结后 | 变化？ |
| ------ | ------ | ------ | ----- |
| `Loss = trace(Psi⁻¹ H Psi)` | 含 K 列 `logψ_j(x_i)` | **同 K 列**,只是 `logψ_0` 来自常量函数 | **不变** |
| `∂Loss/∂θ_j, j > 0` | 链式法则经 `det` 展开 | 同一条链式法则,只是 `logψ_0` 被视作常数 | **数值完全相同** |
| `∂Loss/∂θ_0` | 链式法则 | **不计算**(等价于 0) | 形式上不写这一项 |
| `S_ij = ⟨∂_i logΨ* ∂_j logΨ⟩` | 完整 N×N | 活动子空间 N_active×N_active | 结构变小,数值不变 |
| `ng = S⁻¹ · grad` | 完整向量 | 活动子向量 | 冻结方向 = 0 |

> **关键结论**：冻结对 **数学表达式** 没有影响,只是 **计算图 / 优化变量** 变小了。这正是我们想要的加速——计算量与参数数目的平方/立方同步下降。

---

## 1. 三种"跳过梯度"的层次

| 层次 | 做法 | 节省 forward | 节省 backward | 代码改动量 |
| ---- | ---- | ------------ | ------------- | ---------- |
| ❶ mask to 0 | 算出全梯度再乘 mask | ❌ 全部算了 | ❌ 全部算了 | ★ |
| ❷ `stop_gradient` 包输出 | forward 照算,backward 不追溯 | ❌ | ✅ 大部分 | ★★ |
| ❸ 参数物理分离 | 冻结 ansatz 改用常量 cache | ✅ | ✅ | ★★★ |
| ❹ 完全重写 grad_fn | `jax.grad` 只对活动 params 求导 | ✅ | ✅ | ★★★★ |

下面分别给出 ❷ ❸ ❹ 三档实现,选哪一档取决于你愿意改多少代码。

---

## 2. 方案 ❷：`stop_gradient` 包输出（最简单,推荐先试）

### 2.1 原理

在 `NESTotalAnsatz_gauge_stable._forward_single` 里,`L[i, j] = logψ_j(x_i) - log_ref_j`。  
当 `j == FREEZE_IDX` 时,只要让 `log_x = stop_gradient(ans_j(x_single[i]))`,JAX 就会:

- **forward**: 调用 ans_j 算出 `log_x`（Loss 用得到）  
- **backward**: 看到 `stop_gradient`,**直接返回 0 不追溯**(JAX 内部对 stop_grad 节点不分配 cotangent buffer,也不调度反向 kernel)

于是 `jax.grad` 在反向遍历时,**冻结 ansatz 的整条子图被剪掉**,只回溯活动 ansatz 的参数。这就是真正的"不计算梯度"。

### 2.2 代码改动（`NES_VMC.py:236` 起）

把 `NESTotalAnsatz_gauge_stable` 改成支持"动态冻结":

```python
class NESTotalAnsatz_gauge_freeze(nnx.Module):
    """
    与 NESTotalAnsatz_gauge_stable 等价,新增 freeze_idx 字段。
    freeze_idx = None  → 行为完全相同
    freeze_idx = k     → 第 k 个 sub-ansatz 的输出在 forward 阶段被 stop_gradient 包住
    """
    def __init__(self, n_spin_orbitals, n_states, hidden_dim, ref_state,
                 *, rngs, freeze_idx=None):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals
        self.ref_state = jnp.asarray(ref_state, dtype=jnp.complex64)
        self.freeze_idx = freeze_idx  # 新增:可冻结的索引
        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ans = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ans)

    def _forward_single(self, x_single):
        x_single = x_single.reshape(self.K, self.n_spin)

        # ref 态:本就被 stop_gradient 包住(原版行为),保持
        ref_logs = []
        for j in range(self.K):
            ans_j = self.single_ansatz_list[j]
            ref_logs.append(jax.lax.stop_gradient(ans_j(self.ref_state)))
        ref_logs = jnp.array(ref_logs, dtype=jnp.complex64)

        # 构造 L,冻结列直接 stop_gradient
        L = jnp.zeros((self.K, self.K), dtype=jnp.complex64)
        for i in range(self.K):
            for j in range(self.K):
                ans_j = self.single_ansatz_list[j]
                log_x = ans_j(x_single[i])
                if j == self.freeze_idx:
                    log_x = jax.lax.stop_gradient(log_x)
                L = L.at[i, j].set(log_x - ref_logs[j])

        shift = jnp.max(jnp.real(L), axis=(-2, -1))
        shift = jax.lax.stop_gradient(shift)
        L_stable = L - shift
        Psi_stable = jnp.exp(L_stable)

        sign, log_abs_det = jnp.linalg.slogdet(Psi_stable)
        log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
        log_Psi_gauge = log_Psi_stable + self.K * shift

        return log_Psi_gauge, L_stable, shift, L

    def __call__(self, x):
        # 与原版完全相同的形状分发
        ...
```

### 2.3 训练循环的修改（notebook 里）

冻结切换只需要 1 行——改模型的 `freeze_idx`：

```python
# ====================== 冻结配置 ======================
FREEZE_STEP = 300
FREEZE_IDX  = 0

# 初始化时不冻结
total_ansatz = NESTotalAnsatz_gauge_freeze(
    SINGLE_SIZE, K, SINGLE_SIZE + K,
    ref_state=Hatree_Fock,
    rngs=nnx.Rngs(11),
    freeze_idx=None,         # 初始不冻
)

# 后续 machine / grad_fn / qgt_fn 的构建方式完全不变
total_machine_log_Psi_gauge, graphdef, total_params = create_machine_gauge_stable(total_ansatz)
...

for step in range(N_ITER):
    # 冻结开关 —— 仅此一行！
    if step == FREEZE_STEP:
        total_ansatz.freeze_idx = FREEZE_IDX
        # 重要:NNX 改了属性后,需要让缓存的 machine 重新 jit,
        # 最简单办法是重建一组 machine(也可 force re-jit)
        total_machine_log_Psi_gauge, graphdef, total_params = \
            create_machine_gauge_stable(total_ansatz)
        grad_fn = make_grad_fn(ha, total_machine_L_stable, total_machine_shift,
                               total_machine_log_Psi_gauge, single_machine_list)
        qgt_fn  = make_qgt_fn(total_machine_log_Psi_gauge)
        # sampler 也需要重新建,因为 machine 函数签名变了
        sampler_state = nes_sampler.reset()

    # 训练循环其余部分一字不动
    ...
```

> **NCCL/缓存失效**：`freeze_idx` 改变后,JAX 的 jit cache 会因为 trace 时捕获的 Python 整数变化而**自动重新编译**。不需要手动 `block_until_ready` 之类的清理。

### 2.4 Loss 表达式是否变化？

**完全不变。** 看一眼 forward 的输出：

```text
冻结前 L[i,j] = logψ_j(x_i) - logψ_j(ref)
冻结后 L[i,0] = stop_grad(logψ_0(x_i)) - stop_grad(logψ_0(ref))
                = 数值上等于 logψ_0(x_i) - logψ_0(ref)   ← Loss/E_L 用到的就是这个数
```

`stop_gradient` 在前向求值时返回的**就是输入本身**,不修改数值,只切断反向梯度。`Loss = trace(solve(Psi, HPsi))` 拿到的 `L` 与 `Psi` 与冻结前 bit-by-bit 一致(浮点完全相同,除非冻结 ansatz 参数随后被改动——但我们不会动它)。

---

## 3. 方案 ❸：冻结 ansatz 改用"常量 cache"（进一步省 forward）

`stop_gradient` 省了 backward,但 forward 里 `ans_j(x_single[i])` 还是算了 K 次（每个 sub-ansatz 在每个 sub-walker 上都要算）。如果 frozen_idx 永远不再变,可以让**第一次 forward 把冻结列的 log 缓存下来**,后续直接读 cache。

### 3.1 思路

在 L 矩阵构造前,准备一个 `frozen_log_cache`,形状 `(K, n_spin) → (K,)`(因为 K 行,每行一个 frozen-ansatz 输出)。这个 cache 只在 `freeze_idx` 改变的瞬间刷新一次,之后所有 step 共用。

```python
class FrozenCache:
    def __init__(self):
        self.log_cache = None   # shape (K,)  logψ_freeze(x_i) for current x_batch
        self.x_id = None        # 标识当前 x_batch 是否变化

def get_frozen_log(cache, params, x_batch, frozen_ansatz):
    """
    x_batch: (batch, K, n_spin)  或 (K, n_spin)
    返回 logψ_freeze(x_i), shape (batch, K) 或 (K,)
    """
    if cache.log_cache is not None and cache.x_id is x_batch:
        return cache.log_cache
    # 计算一次并缓存
    # x_batch 可能是 (batch, K, n_spin) 或 (K, n_spin)
    if x_batch.ndim == 3:
        # batch, K, n_spin → 对每个 i, frozen_ansatz(x_batch[:, i, :])
        # 简单做法:vmap 沿 axis 1
        def per_i(xi):  # xi: (batch, n_spin)
            return jax.vmap(frozen_ansatz)(xi)
        # 注意:frozen_ansatz 是 closure,使用冻结 params(常量)
        ...
    cache.log_cache = result
    cache.x_id = x_batch
    return result
```

但注意：**x_batch 每步都变**(采样器出新的 walker),所以这个 cache 命中率低,实际加速不大。**真实使用中,方案 ❷ 才是性价比最高的**。

---

## 4. 方案 ❹：彻底分离 frozen / active 子图(论文级优化)

如果 frozen_idx 长期不变(几百 step),且你愿意重写机器函数,可以做**参数物理分离**:

```python
# 一次性拆出 frozen 和 active
graphdef, state = nnx.split(total_ansatz)
frozen_state = state["single_ansatz_list"][FREEZE_IDX]   # 整个子 ansatz 0
active_state = {
    "single_ansatz_list": [s for i, s in enumerate(state["single_ansatz_list"])
                            if i != FREEZE_IDX]
}

# frozen 子 ansatz 单独建一个 machine,输入 raw x,输出 logψ_0(x)
frozen_machine, _, _ = create_single_machine_gauge_fixed(
    total_ansatz.single_ansatz_list[FREEZE_IDX], Hatree_Fock
)
# 注意:它使用 NNX 原参数(后续也不会改),frozen_machine 闭包内捕获
```

然后构建新 L 矩阵,冻结列用 `frozen_machine`：

```python
def new_total_machine(active_params, sigma):
    # 1. frozen 列:算一次
    log_frozen = frozen_machine(frozen_state, sigma_flat)  # (batch*K,)
    log_frozen = log_frozen.reshape(batch, K)               # (batch, K)

    # 2. active 列:用 active_params 调 active ansatz
    # active 模型的 graphdef 不包含 frozen 子 ansatz
    active_model = nnx.merge(active_graphdef, active_params)
    log_active = active_model(sigma_flat)  # (batch, K, K-1)

    # 3. 拼回完整 L:把 frozen 列粘到第 0 列
    # ... 形状操作 ...

    # 4. 后续 slogdet / Loss 计算完全相同
```

**好处**:
- frozen_ansatz 的 forward 也只算一次(对于相同 x_batch)
- active 模型的 graphdef 更小,JIT 编译后的 HLO 图小很多,反向 pass 跳过的代码更多
- QGT 矩阵只对 active_params 求,尺寸直接是 (N_active, N_active)

**代价**:
- 需要手动管理 graphdef / state 的拆分 / 合并
- sampler 状态需要重建
- 调试更难(出问题时断点位置不明显)

---

## 5. 加速效果估算（K=4, hidden=12, n_spin=4, LiH K4）

每个 `SingleStateAnsatz` 的参数量级：

```text
linear1: 4 * 12 = 48 (kernel) + 12 (bias)        = 60
linear2: 12 * 12 = 144 + 12                       = 156
output:  12 * 1 = 12 + 1                          = 13
单 sub-ansatz: ≈ 229 个复数参数 → 458 个实数梯度分量
K=4 总参数 ≈ 1832
```

| 方案 | 每次 step 节省 |
| ---- | -------------- |
| ❶ mask to 0 | 0% (算完再置 0) |
| ❷ stop_gradient 包输出 | backward 减少 ~25% (1/4 参数路径被剪) |
| ❸ frozen cache | forward 减少 ~25% (前提:cache 命中) |
| ❹ 物理分离 | forward -25% + backward -25% + **QGT 求解从 N² → (3N/4)² = 56% 大小** |
| 综合 ❹ | **总训练时间 -30% ~ -45%**(QGT solve 占大头时接近 45%) |

实测时建议你打开 `jax.profiler` 看 backward / QGT-solve 的耗时占比,占比大的先优化。

---

## 6. 推荐路线

1. **先上方案 ❷**（最简,10 行代码）：`stop_gradient` 包住冻结列的 `log_x`。  
   验证 Loss 曲线在 FREEZE_STEP 处没有跳跃,冻结 ansatz 0 的 `loss_per_state[0]` 锁住,其他态继续收敛。
2. **如果 backward 是瓶颈**（即 grad_fn + qgt_fn 耗时 >> sampler）,再上方案 ❹。  
   物理分离后重新编译一次,后续每个 step 都快。
3. **不建议做方案 ❸**：x_batch 每步变,cache 命中率低,代码复杂度高,收益小。

---

## 7. 一句话总结

> 冻结**不会改变** Loss 公式 `trace(Psi⁻¹ H Psi)`,也**不会改变** `∂Loss/∂θ_active` 的公式；  
> 想真正跳过梯度计算,在 `L[i, FREEZE_IDX] = log_x - log_ref` 里对 `log_x` 加一行 `jax.lax.stop_gradient` 即可,无需改任何机器函数签名或训练循环结构。
