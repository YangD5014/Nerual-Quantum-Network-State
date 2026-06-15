# NES-VMC 数值稳定策略说明文档
**文档用途**：汇总基于对数域、最大值偏移、矩阵正则化等全套数值稳定方案，适配 NetKet+JAX+Flax NNX 实现的 NES-VMC 算法，对齐论文 `S8 Numerical Stability` 章节，明确原理、执行流程、代码实现规范与避坑要点。
**适用场景**：量子多体 NES-VMC 激发态计算（以 H₂ 分子费米子体系为例）

---

## 一、背景概述
NES-VMC 核心计算依赖**多波函数矩阵、矩阵求逆、行列式、哈密顿线性求和**，直接在实数域计算会出现典型浮点问题：
1. **数值下溢/上溢**：多粒子波函数 $\psi$ 幅值极小/极大，直接指数运算产生 `0`/`Inf`；
2. **矩阵奇异**：$\boldsymbol{\Psi}$ 矩阵行/列近似线性相关，求逆失败、出现 `NaN`；
3. 哈密顿求和 $\hat{H}\psi$ 多组态累加引发精度丢失；
4. 行列式 $\det(\boldsymbol{\Psi})$ 因大量乘积运算快速下溢。

本文档整合**论文原生方案 + 工程适配优化**，形成全链路数值稳定规范，覆盖波函数、哈密顿作用、矩阵运算、行列式、采样全环节。

---

## 二、核心术语统一（结合代码与论文）
### 1. 样本定义
- **单扩展样本**：$\boldsymbol{X} = [x^1, x^2, ..., x^K]$，形状 `(K, n_spin)`
  $K$：目标激发态数量；$x^i$：原系统单粒子组态；
- **批量样本**：多个扩展样本组成批次，形状 `(batch_size, K, n_spin)`；
- 规则：**所有稳定操作均在「单个扩展样本」内部独立执行，不跨批次共享参数**。

### 2. 核心矩阵定义
对单个扩展样本 $\boldsymbol{X}$，定义 $K\times K$ 复数矩阵：
1. 对数波函数矩阵：$\boldsymbol{L},\ L_{ij} = \log\big(\psi_j(x^i)\big)$
   行$i$：第$i$个粒子集；列$j$：第$j$个单态 Ansatz；
2. 哈密顿作用对数矩阵：$\boldsymbol{M},\ M_{ij} = \log\big(\hat{H}\psi_j(x^i)\big)$；
3. 局域能量矩阵：$\boldsymbol{E_L} = \boldsymbol{\Psi}^{-1}\hat{H}\boldsymbol{\Psi}$（NES 损失核心）。

### 3. 关键偏移量
$L_\mathrm{max}$：**单个扩展样本内 $\boldsymbol{L}$ 矩阵所有元素的最大值（复数全局最大）**，标量，逐样本计算，**禁止使用批次全局最大值**。

---

## 三、全套数值稳定策略（分模块）
## 模块1：对数域最大值偏移（核心策略，论文标准方案）
### 1. 原理
对数运算将**乘除转为加减**，规避乘积型下溢/上溢；
对矩阵统一减去单样本内部最大值，强制偏移后对数元素 $\widetilde{L}_{ij} \le 0$，指数还原后 $\exp(\widetilde{L}_{ij}) \in (0,1]$，彻底限制浮点范围。
同时：**同偏移量作用于 $\boldsymbol{L}$ 和 $\boldsymbol{M}$，比值类物理量完全不变**，无物理偏差。

### 2. 标准执行步骤（单扩展样本）
1. 得到原始对数矩阵：$\boldsymbol{L},\boldsymbol{M}$；
2. 计算单样本最大值：$L_\mathrm{max} = \max\limits_{i,j} L_{ij}$；
3. 统一偏移：
   $$\widetilde{L}_{ij} = L_{ij} - L_\mathrm{max},\quad \widetilde{M}_{ij} = M_{ij} - L_\mathrm{max}$$
4. 指数还原得到稳定实数矩阵：
   $$\widetilde{\Psi}_{ij} = \exp(\widetilde{L}_{ij}),\quad \widetilde{H\Psi}_{ij} = \exp(\widetilde{M}_{ij})$$

### 3. 物理不变性证明
局域能量 $O_L = \dfrac{\hat{H}\psi}{\psi}$：
$$
\dfrac{\exp(\widetilde{M}_{ij})}{\exp(\widetilde{L}_{ij})}
= \exp(M_{ij}-L_{ij}) = O_L
$$
偏移仅做整体缩放，**所有物理观测量保持真值**。

### 4. 批量处理规范
使用 `jax.vmap` 逐样本并行执行偏移，每个样本拥有独立 $L_\mathrm{max}$，代码范式：
```python
@jax.jit
def single_offset(log_mat):
    l_max = jnp.max(log_mat)
    return log_mat - l_max, l_max

# 批量矩阵 shape: (B, K, K)
batch_L_shift, batch_Lmax = jax.vmap(single_offset)(batch_L)
```

### 5. 禁忌
❌ 禁止使用整个 batch 的全局最大值做统一偏移；
❌ 禁止对 $\boldsymbol{M}$ 单独计算最大值偏移（必须共用 $\boldsymbol{L}$ 的 $L_\mathrm{max}$）。

---

## 模块2：哈密顿作用 $\hat{H}\psi$ 稳定计算
### 1. 问题
离散哈密顿：$\hat{H}\psi(x) = \sum\limits_{x'} h(x,x')\psi(x')$，多组态求和易下溢，**严禁先取对数再作用哈密顿**（$\ln(\hat{H}\psi) \ne \hat{H}(\ln\psi)$）。

### 2. 标准计算链路（不可颠倒）
1. 模型输出 $\log\psi(x')$；
2. 单组态内部最大值偏移 + 指数还原，得到稳定 $\psi(x')$；
3. 执行哈密顿线性求和：$\hat{H}\psi = \sum h(x,x')\psi(x')$；
4. 对求和结果取对数，得到 $M_{ij} = \log(\hat{H}\psi)$。

### 3. 代码约束
所有 `Ham_psi`/`Ham_Psi` 函数必须遵循上述链路，禁止 `Ĥ(logψ)` 写法。

---

## 模块3：矩阵求逆正则化（解决矩阵奇异）
### 1. 问题
当 $\boldsymbol{\Psi}$ 矩阵行/列近似线性相关（行列式趋近于0），直接求逆会产生巨大数值误差、出现 `NaN`。

### 2. 方案：极小对角正则化
在偏移后的稳定矩阵 $\widetilde{\Psi}$ 上添加**极小常数单位阵**，柔和破坏奇异性，不影响物理结果：
$$
\boldsymbol{\Psi}_\mathrm{reg} = \widetilde{\Psi} + \varepsilon \cdot \boldsymbol{I}
$$
- 推荐参数：$\varepsilon = 10^{-8} \sim 10^{-6}$（根据体系微调）；
- 使用工具：优先 `jnp.linalg.solve`（求解线性方程组），优于 `jnp.linalg.inv`（直接求逆，稳定性更差）。

### 3. 适用位置
计算局域能量矩阵 $\boldsymbol{E_L} = \boldsymbol{\Psi}^{-1}\hat{H}\boldsymbol{\Psi}$ 前，必须对 $\widetilde{\Psi}$ 做正则化。

---

## 模块4：行列式 $\det(\boldsymbol{\Psi})$ 稳定计算
### 1. 问题
总 Ansatz $\Psi_\mathrm{total}=\det(\boldsymbol{\Psi})$，大量乘积运算极易下溢，是采样分布 $\Psi_\mathrm{total}^2$ 的核心依赖。

### 2. 分层稳定方案
1. 前置：对单样本 $\boldsymbol{L}$ 执行**最大值偏移**，得到 $\widetilde{\Psi}$；
2. 底层工具：使用 JAX 原生 `jnp.linalg.slogdet`（对数行列式，官方优化），替代手动计算行列式；
3. 偏移补偿：矩阵整体缩放 $\widetilde{\Psi}=e^{-L_\mathrm{max}}\Psi$，满足：
   $$\det(\widetilde{\Psi}) = e^{-K\cdot L_\mathrm{max}} \cdot \det(\boldsymbol{\Psi})$$
   还原真实对数行列式：
   $$\log|\det(\boldsymbol{\Psi})| = \log|\det(\widetilde{\Psi})| + K\cdot L_\mathrm{max}$$

### 3. 代码规范
```python
# 单样本
sign, log_abs_det_tilde = jnp.linalg.slogdet(Psi_tilde)
log_det = log_abs_det_tilde + K * L_max
log_Psi_total = log_det + 1j * jnp.angle(sign)
```

---

## 模块5：采样器约束（物理+数值双重保障）
### 1. 物理约束（同时影响数值）
NES-VMC 要求扩展样本满足：$i\ne j \implies x^i \ne x^j$
若存在重复组态，$\boldsymbol{\Psi}$ 矩阵出现线性相关，行列式为 0，直接矩阵奇异。

### 2. 工程实现
自定义 Metropolis 采样规则，在采样/跃迁后**校验组态重复性**：
- 生成初始态：循环采样直到获得无重复组态的扩展样本；
- 跃迁候选态：若出现重复组态，舍弃候选、保留原组态。

### 3. 作用
从源头避免矩阵奇异，大幅降低正则化依赖。

---

## 模块6：异常值兜底策略（运行时防护）
针对极端浮点异常（`NaN/Inf`）的兜底规则，作为补充防护：
1. MCMC 重采样：若样本对应 $\boldsymbol{\Psi}$ 元素趋近于0，执行额外 MCMC 步重新采样；
2. 无效样本剔除：单样本计算出现 `NaN` 时，该样本不参与能量/梯度统计；
3. 迭代跳过：连续大量迭代出现数值异常，跳过当前优化步，防止训练崩溃。

> 说明：该策略为兜底方案，优先依靠前5种核心策略。

---

## 四、全链路执行顺序（标准流水线）
以**单个扩展样本**为单位，按顺序执行所有稳定策略：
1. 神经网络输出 $\boldsymbol{L} = \log\psi_j(x^i)$（对数波函数矩阵）；
2. 计算单样本 $L_\mathrm{max}$，执行全局最大值偏移，得到 $\widetilde{L}$；
3. 指数还原得到稳定矩阵 $\widetilde{\Psi}$；
4. 哈密顿作用计算 $\hat{H}\psi$，取对数得到 $\boldsymbol{M}$，共用 $L_\mathrm{max}$ 偏移得到 $\widetilde{M}$；
5. 指数还原得到 $\widetilde{H\Psi}$；
6. 对 $\widetilde{\Psi}$ 添加对角正则化 $\varepsilon \boldsymbol{I}$；
7. 求解 $\boldsymbol{E_L} = \boldsymbol{\Psi}_\mathrm{reg}^{-1} \cdot \widetilde{H\Psi}$，计算迹作为损失；
8. 行列式计算：偏移矩阵 + `slogdet` + 偏移补偿，得到 $\log\det(\boldsymbol{\Psi})$ 用于采样；
9. 采样器持续校验组态无重复，从源头规避矩阵奇异。

> 批量样本：使用 `vmap` 并行执行上述全流程，每个样本独立计算 $L_\mathrm{max}$。

---

## 五、参数配置建议（工程实测）
| 稳定项                | 推荐参数 | 适用场景                     |
| --------------------- | -------- | ---------------------------- |
| 矩阵正则化系数 $\varepsilon$ | $1\mathrm{e}{-8}$ | 常规分子体系（H₂、小分子） |
| 矩阵正则化系数 $\varepsilon$ | $1\mathrm{e}{-6}$ | 大分子/高激发态（矩阵易奇异） |
| 最大值偏移             | 逐样本独立 | 全场景强制使用               |
| 行列式计算工具         | `slogdet` | 禁止手动行列式计算           |

---

## 六、常见问题与排查方案
### 1. 训练出现 NaN
1. 优先级1：检查采样器，是否存在重复组态；
2. 优先级2：调大正则化系数（$1\mathrm{e}{-8} \rightarrow 1\mathrm{e}{-6}$）；
3. 优先级3：校验 $\boldsymbol{M}$ 是否使用同一 $L_\mathrm{max}$（禁止单独偏移）。

### 2. 能量剧烈震荡、结果失真
1. 检查链路：是否出现 $\hat{H}(\log\psi)$ 错误写法；
2. 检查：是否误用 batch 全局最大值偏移。

### 3. 行列式下溢、采样失效
1. 确认行列式执行「偏移 + slogdet + 偏移补偿」三步；
2. 检查 $K\cdot L_\mathrm{max}$ 补偿项是否遗漏。

### 4. 梯度消失/梯度爆炸
数值稳定策略不直接影响梯度，但若底层矩阵精度丢失，会间接引发；优先修复矩阵奇异、下溢问题。

---

## 七、关键总结与编码规范
### 1. 核心原则
1. **对数域+单样本最大值偏移**：全链路基础，解决上下溢；
2. **运算顺序铁律**：$\ln(\hat{O}\psi) \ne \hat{O}(\ln\psi)$，永远先算符作用、后取对数；
3. **单样本独立**：$L_\mathrm{max}$、偏移、正则化均作用于单个扩展样本，不跨批次；
4. **多重防护**：对数偏移为主，矩阵正则、采样约束为辅，兜底策略做最后保障。

### 2. 强制编码规范
1. 所有波函数、哈密顿相关计算，默认走对数域；
2. $\boldsymbol{L}$ 与 $\boldsymbol{M}$ 必须共用同一个 $L_\mathrm{max}$；
3. 矩阵求逆优先 `jnp.linalg.solve`，搭配极小正则项；
4. 行列式统一使用 `jnp.linalg.slogdet` 并补偿偏移量；
5. 采样器必须开启「无重复组态」校验。

### 3. 文档适配范围
本方案完全对齐论文 `S8 Numerical Stability`，同时适配 NetKet+JAX+Flax NNX 实现的 NES-VMC，可直接作为代码开发、调试、迭代的标准依据。