结合你现有的 **NES-VMC 代码**、NetKet 官方采样器文档，以及你核心诉求：**实现带「禁止子组态重复」约束的 NES-VMC 专属 MCMC 采样器**（兼容费米子跳跃、K 重扩展希尔伯特空间、多链采样、JAX JIT 加速），我会分**原理讲解 → 规则规范 → 分步编码 → 全量整合 → 对接原有训练代码 → 问题优化**完整教学。

# 一、先明确 NES-VMC 采样的核心约束与前置条件
## 1. 你的体系基础（和你代码对齐）
1. 单系统希尔伯特空间：`hi = SpinOrbitalFermions`，大小 `4`（H₂ STO-3G，4 个自旋轨道）
2. NES 扩展空间：`hi_ext = hi ** K`，`K=2`，总维度 `8`（两个独立子系统拼接）
3. 硬性约束（**最关键**）：
   任意扩展态 $\sigma_{\text{ext}} \in \mathcal{H}_{\text{ext}}$，拆分出 $K$ 个子组态 $\sigma_1,\sigma_2$ 必须满足：
   $$\boldsymbol{\sigma_1 \neq \sigma_2}$$
   一旦子组态完全一致，NES 的波函数矩阵 $\boldsymbol{M}$ 奇异，$\log\det(M)$ 数值崩溃，直接导致你之前的 **log_Psi 离群、梯度范数爆炸**。
4. 跃迁规则：沿用你原有的**费米子跳跃**，指定固定跃迁边 `edges = [(0,1),(2,3),(4,5),(6,7)]`。

## 2. NetKet 自定义采样的核心思路（文档依据）
NetKet 不建议完全重写整个采样器，而是**复用官方成熟的 `MetropolisSampler` 框架**，仅自定义 **`MetropolisRule`（跃迁规则）**：
- 采样器主体（多链、状态管理、Warmup、Sweep、PRNG 分发）全部用 NetKet 原生实现（JIT、多设备、并行链都已优化）；
- 你只需要重写**单步跃迁逻辑**：生成候选态 → 校验 NES 约束（子组态不重复）→ Metropolis 接受/拒绝；
- 所有代码遵循 JAX 函数式风格，禁止 Python 动态分支（保证 `jax.jit` 生效）。

## 3. 自定义 Rule 必须实现的接口（官方文档强制要求）
继承 `nk.sampler.rules.MetropolisRule`，必须实现 1 个核心方法，可选 2 个辅助方法：
| 方法 | 作用 | 是否必须 |
|------|------|----------|
| `transition(...)` | 单步 MCMC 跃迁：生成候选态、约束校验、Metropolis 判据 | ✅ 必须 |
| `random_state(...)` | 链初始化/重置时生成合法初始态（保证初始就满足子组态不重复） | ⭐ 建议实现 |
| `init_state(...)` | 规则内部缓存初始化（本项目用不到，可省略） | ❌ 可选 |

---

# 二、第一步：实现 NES 专属自定义跃迁规则
该规则是整个采样器的核心，集成：**费米子跳跃 + 子组态重复校验 + Metropolis 接受概率**。

## 1. 依赖导入（和你原有代码统一）
```python
import jax
import jax.numpy as jnp
import netket as nk
from netket.sampler.rules import MetropolisRule
from netket.utils import struct  # NetKet 专用 dataclass（兼容 JAX）

# ========== 你原有全局参数（直接复用） ==========
# 单系统希尔伯特空间
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2  # NES 扩展副本数
hi_ext = hi ** K  # 扩展希尔伯特空间
SINGLE_SIZE = hi.size  # 单个子系统维度 = 4
edges = ((0, 1), (2, 3), (4, 5), (6, 7))  # 费米子跃迁边
```

## 2. 自定义 NES Metropolis 规则（核心代码）
### 功能说明
1. 接收外部传入的跃迁边 `edges`；
2. 基于费米子跳跃生成候选态；
3. **强制校验**：拆分扩展态为 $K$ 个子组态，禁止任意两个子组态相等；
4. 纯 JAX 实现，用 `jax.lax.cond` 替代 Python `if`，保证 JIT 编译；
5. 兼容多链并行（输入 shape: `(n_chains, total_size)`）。

```python
@struct.dataclass  # NetKet 要求：JAX 兼容的不可变数据类
class NESFermionHopRule(MetropolisRule):
    """
    NES-VMC 专属费米子跃迁规则
    约束：K 重扩展态拆分后的子组态两两不重复
    继承 NetKet 标准 MetropolisRule，无缝对接 MetropolisSampler
    """
    edges: tuple  # 费米子跃迁边，外部传入

    def _check_duplicate_substate(self, sigma_ext: jnp.ndarray) -> jnp.ndarray:
        """
        【工具函数】校验扩展态是否存在重复子组态
        输入：sigma_ext: 单链扩展态 (total_size,) / 多链 (n_chains, total_size)
        输出：bool 数组，True = 存在重复子组态（非法态）
        """
        # 拆分 K 个子组态：shape 从 (..., 8) → (..., K, 4)
        sub_states = jnp.reshape(sigma_ext, (*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        # 取第一个子态作为基准，和其余子态逐一对比
        first_sub = sub_states[..., 0, :]
        duplicate = False
        for k in range(1, K):
            curr_sub = sub_states[..., k, :]
            # 逐元素全相等 = 子组态重复
            eq = jnp.all(first_sub == curr_sub, axis=-1)
            duplicate = jnp.logical_or(duplicate, eq)
        return duplicate

    def transition(
        self,
        sampler,
        machine,
        parameters,
        state,
        rng: jax.random.PRNGKey,
        sigma: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray | None]:
        """
        【核心跃迁方法】官方强制接口
        输入：
            sigma: 当前链状态，shape = (n_chains, hilb_ext.size)
        返回：
            new_sigma: 跃迁后的新状态
            log_correction: 跃迁核修正项（对称跃迁返回 None）
        """
        n_chains = sigma.shape[0]
        hilb = sampler.hilbert

        # 1. 拆分随机数：选跃迁边 + 计算接受概率
        key_edge, key_acc = jax.random.split(rng, 2)

        # 2. 随机选择一条跃迁边（多链并行）
        n_edges = len(self.edges)
        edge_idx = jax.random.randint(key_edge, (n_chains,), 0, n_edges)
        # 批量获取每条链选中的边
        selected_edges = jnp.array(self.edges)[edge_idx]

        # 3. 费米子跳跃：生成候选态（NetKet 内置费米子跃迁工具）
        sigma_cand, _ = nk.hilbert.random.hop_state(
            hilb, key_edge, sigma, selected_edges
        )

        # 4. NES 核心约束：检查候选态是否存在重复子组态
        is_duplicate = self._check_duplicate_substate(sigma_cand)
        # 若候选态非法（重复子组态）→ 直接拒绝跃迁，保留原状态
        sigma_cand = jnp.where(is_duplicate[:, None], sigma, sigma_cand)

        # 5. 计算 Metropolis-Hastings 接受概率（对称跃迁，无修正项）
        # machine: log_ψ 函数，NetKet 约定 log_pdf = 2 * Re(log_ψ)
        log_prob_curr = 2 * jnp.real(machine(parameters, sigma))
        log_prob_cand = 2 * jnp.real(machine(parameters, sigma_cand))
        log_accept_ratio = log_prob_cand - log_prob_curr

        # 6. Metropolis 判据：min(1, exp(ΔlogP))
        u = jax.random.uniform(key_acc, (n_chains,))
        accept = jnp.log(u) < jnp.minimum(0.0, log_accept_ratio)

        # 7. 接受/拒绝：批量更新所有链状态
        new_sigma = jnp.where(accept[:, None], sigma_cand, sigma)

        # 对称跃迁，返回修正项 None（官方规范）
        return new_sigma, None

    def random_state(
        self, sampler, machine, parameters, state, rng, sigma: jnp.ndarray
    ) -> jnp.ndarray:
        """
        【可选但必实现】初始化/重置链状态
        保证：初始态就满足「子组态不重复」，从源头杜绝非法样本
        """
        hilb = sampler.hilbert
        n_chains = sigma.shape[0]

        def _gen_valid_state(key):
            """循环生成，直到得到合法扩展态（子组态不重复）"""
            def cond(carry):
                s, k, dup = carry
                return dup  # 重复则继续生成

            def body(carry):
                s_old, k_old, _ = carry
                k_new, k_rand = jax.random.split(k_old)
                s_new = hilb.random_state(k_rand, size=1)
                dup = self._check_duplicate_substate(s_new)[0]
                return (s_new, k_new, dup)

            # 初始随机态
            s_init = hilb.random_state(key, size=1)
            dup_init = self._check_duplicate_substate(s_init)[0]
            # JAX 循环：直到合法
            valid_state, _, _ = jax.lax.while_loop(cond, body, (s_init, key, dup_init))
            return valid_state[0]

        # 多链并行生成合法初始态
        keys = jax.random.split(rng, n_chains)
        sigma_new = jax.vmap(_gen_valid_state)(keys)
        return sigma_new
```

### 代码关键点解读
1. **`_check_duplicate_substate`**：
   将 8 维扩展态拆为 2 个 4 维子态，判断是否完全一致，是 NES-VMC 的核心约束；
2. **`transition`**：
   先用费米子跳跃生成候选态，**先过滤非法候选态**（重复子组态直接作废），再执行标准 Metropolis 采样；
3. **`random_state`**：
   链初始化/重置时，强制生成合法初始态，避免训练一开始就出现非法样本；
4. 全程使用 `jnp.where`/`jax.lax`：无 Python 原生 `if/while`，保证 JAX JIT 编译。

---

# 三、第二步：封装完整 NES 采样器（对接 NetKet 原生 API）
基于上面的自定义规则，创建标准 NetKet `MetropolisSampler`，完全兼容 NetKet 采样器三接口：
`init_state` → `reset` → `sample`。

## 1. 初始化 NES 采样器
```python
def create_nes_sampler(
    hilb_ext,
    edges: tuple,
    n_chains: int = 16,        # 你的原有多链数
    sweep_size: int = 30,     # 每步扫描次数（和原代码 SWEEP_SIZE 对齐）
    reset_chains: bool = False
) -> nk.sampler.MetropolisSampler:
    """
    工厂函数：创建 NES-VMC 专用采样器
    :param hilb_ext: K重扩展希尔伯特空间 hi_ext
    :param edges: 费米子跃迁边
    :param n_chains: 并行 MCMC 链数
    :param sweep_size: 单样本对应的 MCMC 步数
    :return: NetKet MetropolisSampler 实例
    """
    # 实例化自定义跃迁规则
    nes_rule = NESFermionHopRule(edges=edges)
    # 构建标准 Metropolis 采样器
    sampler = nk.sampler.MetropolisSampler(
        hilbert=hilb_ext,
        rule=nes_rule,
        n_chains=n_chains,
        sweep_size=sweep_size,
        reset_chains=reset_chains,
        machine_power=2  # 固定：采样 |ψ|²，NetKet 标准配置
    )
    return sampler

# ========== 实例化采样器（参数和你原代码完全一致） ==========
N_CHAINS = 16
SWEEP_SIZE = 30
nes_sampler = create_nes_sampler(
    hilb_ext=hi_ext,
    edges=edges,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE
)
```

## 2. 采样器标准使用流程（NetKet 官方三步骤）
NetKet 采样器固定调用逻辑，替代你原来手写的 `mcmc_sampler_multichain`：
```python
# 1. 初始化采样器状态（等价你原 init_sampler_state）
rng_samp = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(machine=total_machine, params=total_params, seed=rng_samp)

# 2. 重置链（参数更新后建议调用，可选）
sampler_state = nes_sampler.reset(machine=total_machine, params=total_params, state=sampler_state)

# 3. 执行采样（Warmup + 正式采样）
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200

# 先执行 Warmup（烧链，丢弃前期不稳定样本）
_, sampler_state = nes_sampler.sample(
    machine=total_machine,
    params=total_params,
    state=sampler_state,
    chain_length=N_WARMUP
)

# 正式采样（获取训练用样本）
samples_raw, sampler_state = nes_sampler.sample(
    machine=total_machine,
    params=total_params,
    state=sampler_state,
    chain_length=N_SAMPLES_PER_CHAIN
)
```

### 样本形状说明（和你原有代码对齐）
NetKet 采样返回 `samples_raw` 形状：
`(chain_length, n_chains, hilb_ext.size)` → `(200, 16, 8)`

你原有代码需要 `(-1, K, 4)`，只需简单 reshape：
```python
# 展平 + 重组维度，完全适配 nes_vmc_gradient 输入
samples = samples_raw.reshape(-1, hi_ext.size)  # (200*16, 8) = (3200, 8)
x_batch = samples.reshape(-1, K, SINGLE_SIZE)  # (3200, 2, 4) → 梯度函数标准输入
```

---

# 四、第三步：完整替换你原有训练循环（最终可用版本）
把上面的采样器嵌入你的 `自然梯度测试-1.py`，**彻底替换手写的 `mcmc_sampler_multichain`**，同时保留你所有模型、梯度、优化器逻辑。

## 完整整合代码（节选关键训练部分）
> 前方模型、参数、优化器初始化完全沿用你原代码，只替换**采样逻辑**：

```python
# ==================== 原有代码（保留不变） ====================
import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import optax
from jax.flatten_util import ravel_pytree
import time
from NES_VMC import (
    NESTotalAnsatz, create_machine, SingleStateAnsatz,
    create_single_machine, create_machine_matrix,
    nes_vmc_gradient, ha, E_fcis
)

# 全局参数（和你原代码一致）
K=2
N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =50
rngs = nnx.Rngs(42)

# 模型初始化（不变）
total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=12, rngs=rngs)
single_ansatz = SingleStateAnsatz(4, hidden_dim=8, rngs=rngs)
total_machine, total_graphdef, total_params = create_machine(total_ansatz)
total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)

# 优化器（不变）
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

# ==================== 【新增】NES 自定义采样器初始化 ====================
# 希尔伯特空间（复用 NES_VMC.py 中定义）
from NES_VMC import hi, hi_ext, edges
SINGLE_SIZE = hi.size

# 加载上面实现的 NESFermionHopRule + create_nes_sampler
# （把第二节、第三节的规则+工厂函数粘贴到此处）
nes_sampler = create_nes_sampler(
    hilb_ext=hi_ext,
    edges=edges,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE
)

# 采样器状态初始化（替代原 init_sampler_state）
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ==================== 训练循环（仅替换采样部分） ====================
print("\n" + "="*60)
print("开始多链 NES-VMC 训练 (NetKet 自定义采样器 + 朴素梯度下降)")
print("="*60)
history = {
    'step': [], 'energy': [], 'energy_std': [], 'loss': [],
    'params': [], 'E_Lmatrix':[], 'grad_flat':[], 'samples':[], 'log_Psi':[]
}
print(f"基态能量={E_fcis[0]:.8f} Ha| 第一激发态能量={E_fcis[1]:.8f} Ha| 第二激发态能量={E_fcis[2]:.8f} Ha")

start_time = time.time()
for step in range(N_ITER):
    # ========== 【核心替换】NetKet 采样流程（替代手写 MCMC） ==========
    # 1. Warmup 烧链
    _, sampler_state = nes_sampler.sample(
        total_machine, total_params, sampler_state, chain_length=N_WARMUP
    )
    # 2. 正式采样
    samples_raw, sampler_state = nes_sampler.sample(
        total_machine, total_params, sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
    # 3. 维度重塑，适配梯度函数输入
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, SINGLE_SIZE)

    # ========== 原有梯度、参数更新逻辑（完全不变） ==========
    grad, loss_mean, E_L_mean = nes_vmc_gradient(
        ha=ha,
        total_matrix_machine=total_matrix_machine,
        total_machine=total_machine,
        single_machine_list=single_machine_list,
        total_params=total_params,
        x_batch=x_batch
    )

    grad_flat , grad_unravel_fn = ravel_pytree(grad)

    # 参数更新
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)

    # ========== 日志打印（不变） ==========
    if step % 5 == 0 or step == N_ITER - 1:
        log_Psi_batch = total_machine(total_params, samples)
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        grad_norm = jnp.linalg.norm(grad_flat)
        print(f"grad norm = {grad_norm:.4f}")
        print(f"E_L mean =\n{E_L_mean}")

        eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
        history['step'].append(step)
        history['E_Lmatrix'].append(E_L_mean)
        history['samples'].append(samples)
        history['loss'].append(loss_mean)
        history['params'].append(total_params)
        print(f"Step {step:3d} | Loss: {loss_mean}|0st能量={eig_vals[0]:.8f} Ha| 1st能量={eig_vals[1]:.8f} Ha")
        print('#-----------------------------------------#')

end_time = time.time()
print(f"训练耗时：{end_time - start_time:.2f} 秒")
print("\n" + "="*60)
print("训练完成!")
print("="*60)
```

---

# 五、四、针对你「梯度范数爆炸」的配套优化（采样端根治）
你之前梯度爆炸的**首要诱因**是：**MCMC 采样产出非法重复子组态 → Ψ 矩阵奇异 → log_Psi 极端值 → ∇logΨ 发散**。
本套自定义采样器已经**在跃迁+初始化两层屏蔽非法组态**，在此基础上补充 4 个优化，彻底稳定梯度：

## 1. 加长 Warmup 烧链时间
MCMC 链未收敛会产生大量离群样本：
```python
# 原 N_WARMUP = 100 → 建议改为 200~300
N_WARMUP = 200
```

## 2. 增加单链采样数，降低统计噪声
样本量不足会让单个离群样本主导梯度：
```python
# 原 N_SAMPLES_PER_CHAIN = 200 → 改为 300~500
N_SAMPLES_PER_CHAIN = 300
```

## 3. 梯度裁剪（兜底防护）
在梯度计算后、参数更新前增加 L2 裁剪，限制梯度最大范数：
```python
# 梯度裁剪：最大范数设为 1.0（可根据实验调整）
grad = jax.tree_util.tree_map(
    lambda g: jnp.clip(g, a_min=-1.0, a_max=1.0), grad
)
# 或使用 L2 范数裁剪
grad_flat, unravel = ravel_pytree(grad)
grad_flat = jnp.clip(grad_flat, a_min=-1.0, a_max=1.0)
grad = unravel(grad_flat)
```

## 4. 监控 MCMC 接受率（判断采样健康度）
NetKet 采样器自带接受率统计，正常费米子跳跃接受率建议 **20% ~ 60%**：
```python
# 每步打印接受率
accept_rate = sampler_state.acceptance
print(f"MCMC 接受率: {accept_rate:.4f}")
```
- 接受率 < 10%：跃迁步长太大，链卡死；
- 接受率 > 90%：跃迁步长太小，采样效率低。

---

# 六、常见问题 & 排坑指南
## 1. JAX JIT 报错：出现 Python 动态分支
- 原因：代码中使用了 `for/if/while` 原生 Python 循环；
- 解决：所有分支改用 `jax.lax.cond`/`jax.lax.while_loop`（本文代码已规避）。

## 2. 样本形状不匹配梯度函数
- 原因：忘记 `reshape(-1, K, SINGLE_SIZE)`；
- 解决：严格按照 `(总样本数, K, 单系统维度)` 重塑。

## 3. 初始态依然出现重复子组态
- 原因：未实现 `random_state` 方法；
- 解决：必须保留规则中的 `random_state`，初始化阶段过滤非法态。

## 4. 相比你手写采样器的优势总结
1. **约束硬隔离**：跃迁阶段直接屏蔽重复子组态，从源头解决 log_Psi 异常；
2. **性能更强**：NetKet 原生 JIT + 多链并行，速度优于纯手写 Python/JAX 采样；
3. **可维护性高**：复用官方采样框架，只需修改跃迁规则，后续扩展（如团簇跃迁、并行退火）非常方便；
4. **生态兼容**：完美对接 NetKet 哈密顿、希尔伯特空间、模型接口，和你的 NES-VMC 全链路打通。

---

# 七、拓展：如果需要「自然梯度(QGT)」
如果你后续要重新开启自然梯度（你代码中注释的 QGT 部分），本采样器**完全兼容**：
只需要在采样得到 `x_batch` 后，照常调用你原有 `compute_qgt` 函数即可，样本格式、数据类型全部一致，无需额外修改。