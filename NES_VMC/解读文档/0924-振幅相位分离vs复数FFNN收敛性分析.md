# 振幅-相位分离 FFNN 为何只收敛到 HF 能量，而复数 FFNN 能接近 FCI

> 日期：2026-09-24
> 对比对象：
>
> - `experiments/H2分子/量子-经典混合式 Ansatz/FFNN振幅相位分离版-VMC.ipynb`（实参数，logψ = logA + iφ）
> - `experiments/H2分子/量子-经典混合式 Ansatz/自然梯度VMC-H2.ipynb`（复参数，holomorphic FFNN）

***
## 代码对比
复数FFNN模型:
```python
class SingleStateAnsatz(nnx.Module):
    """单态 Ansatz：适配费米子系统的复数值 FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

```
相位-参数分离模型:
```python
import flax.nnx as nnx
class SingleStateAnsatz_Amplitude(nnx.Module):
    """单态 Ansatz：适配费米子系统的实数值 FFNN"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
    
class SingleStateAnsatz_Phase(nnx.Module):
    """单态 Ansatz：适配费米子系统的相位 FFNN"""
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=float)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=float)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=float)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

    
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.real = SingleStateAnsatz_Amplitude(n_spin_orbitals, hidden_dim, rngs=rngs)
        self.phase = SingleStateAnsatz_Phase(n_spin_orbitals, hidden_dim, rngs=rngs)
    

    def __call__(self, x: jax.Array) -> jax.Array:
        logA = self.real(x)    # log(A)
        phi = self.phase(x)    # phase φ
        logpsi = logA + 1j * phi
        return logpsi

```

## 1. 目前的情况总结

| <br />           | 振幅-相位分离版                  | 复数 FFNN 版                  |
| ---------------- | ------------------------- | -------------------------- |
| 参数               | 实数（两个实 FFNN：A、φ）          | 复数（param\_dtype=complex）   |
| logψ             | logA(x) + iφ(x)，两网络独立     | tanh(Wx+b) 全复数运算           |
| 梯度               | vjp 近似 Wirtinger（非全纯）     | jax.grad(holomorphic=True) |
| 最终能量             | **-0.99742 ± 0.00007 Ha** | **-1.04561 ± 0.00110 Ha**  |
| FCI 基准           | -1.05434745 Ha            | -1.05434745 Ha             |
| 绝对误差             | 0.05693 Ha（5.40%）         | 0.00874 Ha（0.83%）          |
| HF 能量            | -0.99749729 Ha            | -0.99749729 Ha             |
| 关联能 E\_HF−E\_FCI | **0.05685 Ha**            | **0.05685 Ha**             |

**关键观察**：振幅-相位版的最终误差 0.05693 与关联能 0.05685 几乎完全相等——它收敛到的不是"某个平庸的变分态"，而是**精确的 HF 行列式**。且能量标准差从 Step 50 的 3.8e-4 塌缩到 Step 100 之后的 0.000000，说明 MCMC 采样器已经**完全冻结在 HF 组态上**（任何单翻转的接受率都趋于 0）。

而复数 FFNN 版的能量标准差始终保持在 1e-3 量级，采样器一直"活着"，能量持续下降到 -1.0456（仍差 8.7 mHa，也还有采样墙的痕迹）。

***

## 2. 根本原因分析

### 2.1 不是表达能力问题

两个网络对 H2/6-31G 这个 16 维 Fock 空间（组态维度 8 bit）都足够大（hidden = 2×8+4）。振幅×相位的形式 logψ = logA + iφ 在数学上是**万能的**——任意复数都能写成幅度×相位。问题出在**优化动力学**，不在表达力。

### 2.2 核心原因一：相位参数初始时位于（近似）平稳流形

H2 的哈密顿量是实对称的，基态可以取成实波函数。FCI 基态的组态展开近似为：

```
|ψ₀⟩ ≈ c₀|HF⟩ + c_D|D⟩,   c₀ > 0, c_D < 0（双激发系数为负）
```

**符号结构（相对相位 π）就是关联能的唯一来源**。

能量期望对相位只通过相位差进入：

```
⟨H⟩ ⊃ Σ_ση A_σ A_η H_ση cos(φ_η − φ_σ)
```

对相位的梯度 ∝ sin(φ\_η − φ\_σ)。而小随机初始化下，相位网络的输出 φ(x) ≈ 0（所有组态相位几乎相同），于是：

- **sin(Δφ) ≈ 0 → 相位方向的一阶力恒为零**，φ ≡ const 是 ⟨H⟩ 的精确平稳流形；
- 网络被困在"正定波函数"扇区里，而正定波函数学不到 c\_D < 0 这个符号；
- 相位要靠二阶（或平坦方向上的随机游走）才能动起来，400 步迭代根本走不出去。

这就是 VMC 文献里经典的 **sign problem / 符号结构瓶颈**：幅度好学，符号难学。

### 2.3 核心原因二：采样自锁（与原因一互相强化）

符号学不到 → c\_D = ψ(D)/ψ(HF) 无法变负、始终很小 →

1. Metropolis 从 HF 出发经 single\_rule 逐比特翻转，单激发组态 ⟨single|H|HF⟩ = 0（Brillouin 定理），双激发组态要两步翻转且 ψ(D) ≈ 0，**D 几乎从未被访问**；
2. E\_loc 的涨落只来自 H\_HD·ψ(D)/ψ(HF) 这一项，∝ c\_D → 方差 → 0（日志里 std → 0.000000 正是这个信号）；
3. 想让 c\_D 增大的梯度信号本身就 ∝ c\_D，逃逸速率是指数型慢的；而大量 HF 样本产生的力只是在继续"磨尖" ψ 在 HF 处的幅度；
4. 最终自洽冻结点：ψ ≈ HF 行列式，E = E\_HF，方差 = 0，梯度 = 0。

日志证据链完全吻合：Step 50 起 error 就钉死在 0.0569（= 关联能），std 塌缩为 0。

### 2.4 复数 FFNN 为什么能逃出去

复数网络 logψ = tanh(W·tanh(W·tanh(...)))，W 为复数：

1. **初始化时各组态的相位是"一般位置"的**：复权重使 tanh 输出的辐角杂乱分布，Δφ 不是 0 → sin(Δφ) ≠ 0 → **相位旋转方向存在一阶下降力**。能量立刻通过"把相对相位转向 π"获得一阶下降，同时 c\_D 的幅度同步增长，采样器保持活跃，形成正反馈；
2. holomorphic=True 的 jax.grad 给出精确的全纯 Wirtinger 梯度，实部/虚部耦合一致，SR（QGT + solve）在复流形上是干净的自然梯度；
3. 实测：600 步内收敛到 -1.0456，误差 0.83%（剩余误差同样来自有限采样，但程度轻得多）。

**一句话**：实参数振幅-相位网络把"符号"交给了一个初始化在平稳流形上的独立相位网络，且采样自锁使它无法逃逸；复数网络把幅度和相位纠缠在同一个复参数化里，初始化即有非零相位力，符号结构可以被一阶动力学学到。

### 2.5 实现层面值得注意的问题（次要但真实）

1. **force 梯度缺少取实部**：`forces_expect_hermitian`（vjp 版）计算的是
   `⟨(E_loc − Ē)·conj(∇logψ)⟩`，这是一个**复数量**。NetKet 内部取的是 `2 Re⟨(E_loc−Ē)·conj(∇logψ)⟩`（对实参数这才是 ∇⟨H⟩）。这里没取 Re 就交给 optax.sgd——实参数 pytree 会被复数更新**静默变成 complex dtype**，之后 "实参数网络" 的前提已被破坏（虽然虚部更新恰好不大，没造成数值崩溃，但语义已错）。代码里的 `grad * 2` 说明意识到了因子 2，但漏了取实部。
2. 第 3 个 cell 里带 `holomorphic` 参数的 `forces_expect_hermitian` / `compute_qgt` 是**死代码**，被第 4 个 cell 的同名 vjp 版本完全遮蔽，容易误读。
3. 超参差异（400 vs 600 步、n\_chains 200/150、diag\_shift 0.01、grad×2）对结果有影响，但都不是主因——主因是 2.2/2.3 的结构性问题。

***

## 3. 结论

- 振幅-相位分离版收敛到 HF **不是 bug 导致的偶然**，而是"相位平稳流形 + 符号结构采样自锁"的必然：它只能表示/维持正定波函数，而 H2 的关联能恰恰全部藏在双激发系数的负号里。
- 复数 FFNN 依靠初始化时的非平凡相位结构和全纯梯度，一阶动力学就能旋转相位、长大双激发幅度，因此能接近 FCI。

## 4. 改进建议（若坚持实参数振幅-相位形式）

按性价比排序：

1. **修梯度**：force 改为 `grad = 2·Re⟨(E_loc − Ē)·conj(∇logψ)⟩`，并保证 params 始终是实 dtype（或显式 `jax.tree.map(lambda x: x.real, grad)`）。
2. **打破相位平稳初始化**：给相位网络更大的输出层初始化尺度，或对训练样本先验地施加非平凡相位（例如按组态奇偶性给 ±π/2 的固定偏置），使 sin(Δφ) ≠ 0 的一阶力从一开始就存在。
3. **消除采样自锁**：16 维全空间可用 `nk.vqs.FullSumState`（或直接对 2⁸ 组态全求和）做精确梯度，绕过 MCMC 冻结（项目记忆里已有先例：小系统用全空间精确求和代替联合 MCMC）。
4. 训练工程：提高迭代步数（≥600）、早期用更大 diag\_shift（0.01→0.1 再退火）、去掉 grad×2 的手调 hack。
5. 诊断实验（可选）：固定 φ ≡ 0 训练纯幅度网络，看最好能到多少——若只能到略低于 HF 的某个值，就直接验证了"正定扇区 + 采样自锁"的结论。

