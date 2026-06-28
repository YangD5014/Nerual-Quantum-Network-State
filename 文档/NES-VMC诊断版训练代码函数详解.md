# NES-VMC 诊断版训练代码 函数详解

## 概述

本文档详细介绍 `h2-6-31G-K2-Stable版本-GPT copy.ipynb` 中的各个函数和代码块，帮助理解 NES-VMC（自然激发态变分蒙特卡洛）诊断版训练代码的结构和功能。

---

## 1. 导入模块 (Cell 1)

```python
import os, time, logging, numpy as np
import jax, jax.numpy as jnp
import flax.nnx as nnx
import netket as nk, netket.experimental as nkx
import optax
from jax.flatten_util import ravel_pytree
from pyscf import gto, scf, fci
```

**作用**：
- `jax` + `jax.numpy`：高性能自动微分和数值计算
- `flax.nnx`：Flax 新一代神经网络框架（用于定义 Ansatz）
- `netket`：量子蒙特卡洛采样
- `optax`：JAX 生态的优化器
- `pyscf`：Python 分子电子结构计算（生成 Hamiltonian）

---

## 2. 分子与基准设置 (Cell 1)

```python
bond_length = 1.8
geometry = [("H", (0.0, 0.0, 0.0)), ("H", (bond_length, 0.0, 0.0))]
mol = gto.M(atom=geometry, basis="6-31G", verbose=0)
mf = scf.RHF(mol).run(verbose=0)
hf_ground_energy = mf.e_tot

cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
```

**作用**：
- 定义 H₂ 分子（键长 1.8 Å）
- 使用 6-31G 基组
- RHF 自洽场计算获得 HF 能量
- FCI（全组态相互作用）计算获得精确的基态和激发态能量作为基准

---

## 3. Hilbert 空间定义 (Cell 1)

```python
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=4,
    s=1/2,
    n_fermions_per_spin=(1, 1),
)
K = 2
hi_ext = hi ** K
ha = nkx.operator.from_pyscf_molecule(mol)
```

**作用**：
- `hi`：单粒子希尔伯特空间
  - 4 个自旋轨道
  - 每个自旋通道 1 个电子（共 2 个电子）
- `hi_ext = hi ** K`：扩展到 K=2 个副本的直积空间
- `ha`：从 PySCF 分子生成二次量子化 Hamiltonian 算符

---

## 4. 采样边定义 (Cell 1)

```python
Hatree_Fock = hi.all_states()[0]
single_edges, singles, doubles = get_ccsd_excitations_and_sampler_edges_from_hf(Hatree_Fock)

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for i, j in single_edges:
        ext_edges.append((i + offset, j + offset))

ext_edges = jnp.asarray(ext_edges)
nes_rule = NESFermionHopRule(edges=ext_edges, K=K, single_size=SINGLE_SIZE)
```

**作用**：
- `single_edges`：费米子跃迁边（如 (3,0) 表示从轨道 3 跳到轨道 0）
- 构建 K=2 个副本的扩展跃迁边
- `NESFermionHopRule`：自定义的 Metropolis 跃迁规则，保证 NES 约束（各副本子组态互不相同）

---

## 5. 工具函数 (Cell 2)

### 5.1 `tree_l2_norm`

```python
def tree_l2_norm(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return jnp.array(0.0)
    return jnp.sqrt(sum([jnp.sum(jnp.abs(x) ** 2) for x in leaves]))
```

**作用**：计算 PyTree（嵌套参数结构）的全局 L2 范数

**用途**：监控梯度/参数的整体大小

---

### 5.2 `tree_all_finite`

```python
def tree_all_finite(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    if len(leaves) == 0:
        return True
    flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
    return bool(jnp.all(jnp.asarray(flags)))
```

**作用**：检查 PyTree 中所有元素是否都是有限值（非 inf/nan）

**用途**：训练异常检测

---

### 5.3 `unique_ratio_from_samples`

```python
def unique_ratio_from_samples(samples):
    arr = np.asarray(samples)
    arr = arr.reshape(arr.shape[0], -1)
    unique = len({tuple(row.tolist()) for row in arr})
    return unique / max(arr.shape[0], 1)
```

**作用**：计算采样样本的唯一比例

**用途**：
- 反映采样的多样性
- 如果 ratio 很低，说明采样器陷入重复

---

### 5.4 `tree_batch_std_norm`

```python
def tree_batch_std_norm(tree, batch_size):
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0
    for leaf in leaves:
        if leaf.ndim >= 1 and leaf.shape[0] == batch_size:
            centered = leaf - jnp.mean(leaf, axis=0, keepdims=True)
            total = total + jnp.sum(jnp.abs(centered) ** 2)
    return jnp.sqrt(total)
```

**作用**：计算 dlogΨ 在 batch 维度上的标准差范数

**用途**：诊断梯度消失问题。如果接近 0，说明不同样本对参数的响应几乎相同

---

### 5.5 `tree_batch_mean_norm`

```python
def tree_batch_mean_norm(tree, batch_size):
    leaves = jax.tree_util.tree_leaves(tree)
    total = 0.0
    for leaf in leaves:
        if leaf.ndim >= 1 and leaf.shape[0] == batch_size:
            mean_leaf = jnp.mean(leaf, axis=0)
            total = total + jnp.sum(jnp.abs(mean_leaf) ** 2)
    return jnp.sqrt(total)
```

**作用**：计算 dlogΨ 在 batch 维度上的均值范数

**用途**：与 std_norm 配合计算 `dlog_std_ratio`

---

### 5.6 `safe_real` / `safe_float`

```python
def safe_real(x):
    return float(jnp.real(x))

def safe_float(x):
    return float(jnp.asarray(x))
```

**作用**：将 JAX 数组安全转换为 Python float（用于打印/日志）

---

## 6. 诊断函数 (Cell 3)

### 6.1 `compute_diagnostics`

```python
def compute_diagnostics(total_params, x_batch, samples, E_L_mean):
    n_total = x_batch.shape[0]
    n_diag = min(DIAG_BATCH_SIZE, n_total)
    x_diag = x_batch[:n_diag]
    samples_diag = samples[:n_diag]
    # ... 计算各种诊断量 ...
    return {诊断字典}
```

**作用**：计算当前训练状态下的完整诊断量

**返回的诊断量分类**：

#### logΨ 统计
| 诊断量 | 说明 |
|--------|------|
| `log_real_mean/std/span` | logΨ 实部的均值/标准差/跨度 |
| `log_imag_mean/std/span` | logΨ 虚部的均值/标准差/跨度 |

#### 矩阵条件数
| 诊断量 | 说明 |
|--------|------|
| `psi_cond_first` | 第一个样本的 Ψ 矩阵条件数 |
| `psi_cond_mean/max` | 条件数的均值/最大值 |

#### Shift（稳定化偏移）
| 诊断量 | 说明 |
|--------|------|
| `shift_mean/std/span` | L_max 的统计量 |

#### 局域能量迹
| 诊断量 | 说明 |
|--------|------|
| `trace_real_mean/std/span` | tr(E_L) 实部的统计 |
| `trace_imag_mean/std/span` | tr(E_L) 虚部的统计 |
| `valid_ratio` | 有效计算比例 |

#### Hermiticity 检查
| 诊断量 | 说明 |
|--------|------|
| `herm_error` | E_L 非厄米程度 |
| `imag_norm` | E_L 虚部范数 |
| `eig_vals_herm` | 厄米化后的本征值 |
| `eig_vals_raw` | 原始本征值 |

#### 采样诊断
| 诊断量 | 说明 |
|--------|------|
| `unique_ratio` | 采样唯一比例 |

#### 梯度诊断
| 诊断量 | 说明 |
|--------|------|
| `dlog_std_norm` | dlogΨ 批次方差 |
| `dlog_mean_norm` | dlogΨ 批次均值 |
| `dlog_std_ratio` | 方差/均值比（关键！） |

---

### 6.2 `log_diagnostics`

```python
def log_diagnostics(step, loss_mean, grad_norm_raw, grad_norm_update,
                   grad_norm_clipped, diag):
    # 打印各种诊断信息
    # 包括能量、梯度、logΨ统计、shift统计等
```

**作用**：将诊断信息格式化输出到日志

**告警机制**：
```python
if log_real_span < 1e-6:
    logger.warning(">>> WARNING: logΨ.real span ≈ 0，疑似 amplitude collapse")

if trace_real_std < 1e-8:
    logger.warning(">>> WARNING: trace(E_L).real std 很小，covariance 梯度无信号")

if dlog_std_ratio < 1e-6:
    logger.warning(">>> WARNING: dlogΨ batch variation 很小")

if herm_error > 1.0:
    logger.warning(">>> WARNING: E_L_mean 非 Hermitian")
```

---

## 7. 日志配置 (Cell 4)

```python
logger = logging.getLogger(f"NES_VMC_K{K}_diagnostic")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

simple_formatter = logging.Formatter("%(message)s")

file_handler = logging.FileHandler(log_filename, mode="w", encoding="utf-8")
console_handler = logging.StreamHandler()

logger.addHandler(file_handler)
logger.addHandler(console_handler)
```

**作用**：
- 创建独立 logger（避免与其他模块冲突）
- 同时输出到文件和控制台
- 简洁格式（只输出消息，不带时间戳/级别）

---

## 8. 超参数 (Cell 5)

| 参数 | 值 | 说明 |
|------|-----|------|
| `N_CHAINS` | 16*K | Metropolis 采样链数 |
| `N_WARMUP` | 50 | 预热步数 |
| `N_SAMPLES_PER_CHAIN` | 200 | 每链采样数 |
| `SWEEP_SIZE` | 30 | 每步扫描次数 |
| `N_ITER` | 300 | 总训练步数 |
| `Natural_Grad` | False | 是否使用自然梯度 |
| `lr` | 0.01 | 学习率 |
| `clip_norm` | 1.0 | 梯度裁剪阈值 |
| `grad_skip_threshold` | 10.0 | 梯度跳过阈值 |
| `qgt_diag_shift` | 0.1 | QGT 正则化参数 |
| `DIAG_BATCH_SIZE` | 128 | 诊断用批次大小 |

---

## 9. 模型初始化 (Cell 6)

```python
total_ansatz = NESTotalAnsatz_stable(SINGLE_SIZE, K, hidden_dim, rngs=nnx.Rngs(rng_seed))
total_machine, total_graphdef, total_params = create_machine_stable(total_ansatz)
total_matrix_machine, _, _ = create_machine_matrix_stable(total_ansatz)
total_max_machine, _, _ = create_machine_max_stable(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, _, _ = create_single_machine(ansatz)
    single_machine_list.append(m)
```

**作用**：
- 创建 K=2 个副本的 NES Ansatz（带数值稳定化）
- `total_machine`：采样器用机器（还原原始 logΨ）
- `total_matrix_machine`：损失函数用（返回 L_stable）
- `total_max_machine`：损失函数用（返回 L_max）
- `single_machine_list`：K 个单态机器列表（用于 Ham_Psi 计算）

---

## 10. 优化器 (Cell 7)

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(clip_norm),
    optax.sgd(learning_rate=lr),
)
opt_state = optimizer.init(total_params)
```

**作用**：
- 链式优化器：先梯度裁剪，再 SGD 更新
- `clip_norm=1.0` 防止梯度爆炸

---

## 11. 采样器 (Cell 8)

```python
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=N_CHAINS,
    sweep_size=SWEEP_SIZE,
)

sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# warmup
for _ in range(N_WARMUP):
    _, sampler_state = nes_sampler.sample(...)
```

**作用**：
- 创建带 NES 约束的 Metropolis 采样器
- `nes_rule` 确保采样时 K 个副本的子组态互不相同
- 预热阶段让采样器达到平衡分布

---

## 12. 训练循环 (Cell 9)

### 12.1 采样阶段

```python
samples_raw, sampler_state = nes_sampler.sample(
    machine=total_machine,
    parameters=total_params,
    state=sampler_state,
    chain_length=N_SAMPLES_PER_CHAIN,
)
samples = samples_raw.reshape(-1, hi_ext.size)
x_batch = samples.reshape(-1, K, SINGLE_SIZE)
```

**作用**：从当前参数对应的波函数分布中采样

### 12.2 梯度计算

```python
grad_raw, loss_mean, E_L_mean = nes_vmc_gradient_stable(
    ha=ha,
    total_matrix_machine=total_matrix_machine,
    total_max_machine=total_max_machine,
    total_machine=total_machine,
    single_machine_list=single_machine_list,
    total_params=total_params,
    x_batch=x_batch,
)
```

**作用**：计算变分能量梯度

### 12.3 自然梯度（可选）

```python
if Natural_Grad:
    qgt_reg_mat, _ = compute_qgt(...)
    ng_flat = jnp.linalg.solve(qgt_reg_mat, grad_raw_flat)
    grad_update = unravel_fn(ng_flat)
else:
    grad_update = grad_raw
```

**作用**：使用量子几何张量预条件化梯度

### 12.4 异常检测

```python
grad_finite = tree_all_finite(grad_raw)
loss_finite = bool(jnp.isfinite(loss_mean))
grad_explode = bool(grad_norm_raw > grad_skip_threshold)

if (not grad_finite) or (not loss_finite) or grad_explode:
    n_skipped += 1
    continue  # 跳过这次更新
```

**作用**：检测梯度/损失异常，跳过危险更新

### 12.5 诊断与记录

```python
if need_print:
    diag = compute_diagnostics(...)
    log_diagnostics(...)
    # 记录历史
    history["step"].append(step)
    history["loss"].append(loss_mean)
    ...
```

**作用**：计算并记录详细诊断信息

### 12.6 参数更新

```python
updates, opt_state = optimizer.update(grad_update, opt_state, total_params)
total_params = optax.apply_updates(total_params, updates)
```

**作用**：应用梯度更新参数

---

## 13. 诊断指标解读

### 13.1 梯度异常诊断

| 指标 | 正常范围 | 异常含义 |
|------|----------|----------|
| `grad_norm_raw` | < 10 | > 10 表示梯度爆炸 |
| `grad_norm_update` | ~ grad_norm_raw | 远大于表示自然梯度放大 |
| `grad_norm_clipped` | ≤ clip_norm | = clip_norm 表示被裁剪 |

### 13.2 波函数诊断

| 指标 | 正常范围 | 异常含义 |
|------|----------|----------|
| `log_real_span` | > 1e-6 | ≈ 0 表示 amplitude collapse |
| `psi_cond_mean` | < 100 | > 1000 表示矩阵病态 |

### 13.3 能量诊断

| 指标 | 正常范围 | 异常含义 |
|------|----------|----------|
| `trace_real_std` | > 1e-8 | ≈ 0 表示协方差梯度无信号 |
| `herm_error` | < 0.1 | > 1 表示 E_L 非厄米，结果不可信 |

### 13.4 梯度信号诊断

| 指标 | 正常范围 | 异常含义 |
|------|----------|----------|
| `dlog_std_ratio` | > 1e-6 | ≈ 0 表示所有样本响应相同 |

---

## 14. 训练流程图

```
┌─────────────────────────────────────────────────────────┐
│                      开始训练                            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 1. 采样：从 Ψ_θ 分布采样 x_batch                        │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 梯度：nes_vmc_gradient_stable 计算 grad_raw          │
└─────────────────────────────────────────────────────────┘
                         ↓
         ┌────────────────────────────────┐
         │ Natural_Grad = True?          │
         └────────────────────────────────┘
            ↓ Yes           ↓ No
┌──────────────────┐  ┌──────────────────┐
│ 计算 QGT         │  │ grad_update =     │
│ grad = QGT⁻¹ grad│  │ grad_raw          │
└──────────────────┘  └──────────────────┘
            ↓                 ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 异常检测：grad_finite? loss_finite? grad_explode?  │
└─────────────────────────────────────────────────────────┘
         ↓ 异常              ↓ 正常
┌──────────────────┐  ┌──────────────────────────────┐
│ Skip 更新        │  │ 4. Optimizer 更新           │
│ n_skipped++     │  │ clip → SGD → params         │
└──────────────────┘  └──────────────────────────────┘
                              ↓
         ┌────────────────────────────────┐
         │ need_print = 每10步/最后一步/   │
         │           异常时                 │
         └────────────────────────────────┘
            ↓ Yes           ↓ No
┌──────────────────────────────┐
│ 5. 诊断：compute_diagnostics │
│    log_diagnostics            │
│    记录 history               │
└──────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │ 结束?           │
                    └─────────────────┘
                        ↓ Yes
┌─────────────────────────────────────────────────────────┐
│                      训练完成                           │
└─────────────────────────────────────────────────────────┘
```

---

## 15. 关键文件依赖

| 文件 | 导入内容 |
|------|----------|
| `NES_VMC.py` | `NESTotalAnsatz_stable`, `create_machine_stable`, `create_machine_matrix_stable`, `create_machine_max_stable`, `create_single_machine`, `nes_vmc_gradient_stable`, `NES_loss_energy_stable`, `compute_qgt`, `NESFermionHopRule` |
| `NES_VMC_H2_631G.py` | `get_ccsd_excitations_and_sampler_edges_from_hf` |

---

## 16. 总结

本 notebook 实现了一个**带完整诊断功能的 NES-VMC 训练框架**，核心特点：

1. **数值稳定化**：对数域改造防止 exp 溢出
2. **异常检测**：实时监控梯度、能量、采样等异常
3. **灵活超参**：支持自然梯度开关、梯度裁剪、跳过危险更新
4. **详细日志**：记录训练过程中的所有关键诊断量
