# 纯 JAX 实现 VMC 梯度下降计算

本文档详细说明如何使用自定义的 JAX 函数完成变分量子蒙特卡洛（VMC）的梯度下降优化，而不依赖 NetKet 的高级优化接口。

## 实验设置

以下代码不允许更改：

```python
# ===================== 环境配置 =====================
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial
from jax import flatten_util

print(f"JAX version: {jax.__version__}")
print(f"NetKet version: {nk.__version__}")

# ===================== 1. H₂ 分子定义 & FCI 基准 =====================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")

# ===================== 2. NetKet 哈密顿量和采样器 =====================
ha = nkx.operator.from_pyscf_molecule(mol)

hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)

edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=16, sweep_size=32)

print(f"Hilbert space size: {hi.size}")
```

## 一、理论基础

### 1.1 VMC 能量期望值

在变分量子蒙特卡洛中，基态能量的期望值为：

$$
\langle E \rangle = \frac{\langle \psi(\theta) | \hat{H} | \psi(\theta) \rangle}{\langle \psi(\theta) | \psi(\theta) \rangle}
$$

通过采样可以写为：

$$
\langle E \rangle = \sum_{\sigma} E_{\text{loc}}(\sigma) \pi(\sigma)
$$

其中：
- $\sigma$ 是系统的一个组态（sample）
- $E_{\text{loc}}(\sigma) = \frac{\hat{H}\psi(\sigma)}{\psi(\sigma)}$ 是局部能量
- $\pi(\sigma) = \frac{|\psi(\sigma)|^2}{\sum_{\sigma'} |\psi(\sigma')|^2}$ 是采样概率

### 1.2 Force-based 梯度公式

能量对参数 $\theta$ 的梯度为：

$$
\nabla_{\theta} \langle E \rangle = \langle (E_{\text{loc}} - \langle E \rangle) \nabla_{\theta} \log \psi \rangle
$$

这个公式的物理意义是：
- $(E_{\text{loc}} - \langle E \rangle)$ 是中心化的局部能量（force）
- $\nabla_{\theta} \log \psi$ 是波函数对数的梯度
- 梯度是两者的协方差

### 1.3 参数更新规则

使用朴素梯度下降法：

$$
\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \langle E \rangle
$$

其中 $\eta = 0.01$ 是学习率。

## 二、神经网络 Ansatz

### 2.1 网络结构定义

```python
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
```

**关键点**：
- 使用 Flax NNX 框架定义神经网络
- 所有层的参数类型为 `complex`（复数）
- 激活函数使用 `tanh`
- 输入是费米子占据数向量，输出是复数波函数振幅

### 2.2 包装为 Machine 函数

```python
def create_machine(model: nnx.Module):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state
```

**说明**：
- `nnx.split()` 将模型分为静态结构（graphdef）和动态参数（state）
- `machine` 函数接受参数和样本，返回波函数值
- 使用 `@jax.jit` 进行 JIT 编译加速

## 三、核心梯度计算函数

### 3.1 局部能量计算

```python
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    
    这对应 NetKet 的 local_value_kernel
    """
    eta, H_eta = ha.get_conn_padded(sigma)
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
```

**计算步骤**：
1. `ha.get_conn_padded(sigma)` 获取哈密顿量作用在样本 `sigma` 上产生的所有连接态 `eta` 和对应的矩阵元 `H_eta`
2. 计算原始样本和连接态的波函数值
3. 利用公式 $E_{\text{loc}} = \sum_{\eta} H_{\sigma\eta} \frac{\psi(\eta)}{\psi(\sigma)}$ 计算局部能量

**形状变化**：
- `sigma`: `(n_samples, n_orbitals)`
- `eta`: `(n_samples, n_connected, n_orbitals)`
- `H_eta`: `(n_samples, n_connected)`
- 返回值：`(n_samples,)`

### 3.2 统计量计算

```python
def statistics(x):
    """计算样本统计量"""
    mean = jnp.mean(x)
    var = jnp.var(x)
    return mean, jnp.sqrt(var / x.shape[0])
```

返回能量的均值和标准差（考虑样本数的影响）。

### 3.3 Force-based 梯度计算（核心）

```python
@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    核心：复刻 NetKet 的 forces_expect_hermitian 函数
    
    使用 force-based 梯度计算：
    ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    
    关键：对于复数值网络，使用 holomorphic=True
    """
    # 1. 计算局部能量
    O_loc = compute_local_energies(machine, params, sigma)
    
    # 2. 统计能量均值
    O_mean, O_std = statistics(O_loc)
    
    # 3. 中心化局部能量
    O_centered = O_loc - O_mean
    
    # 4. 计算 ∇log ψ 对每个样本
    # 使用 jax.grad 计算复数梯度（holomorphic=True）
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    # 5. 计算 force-based 梯度
    # grad = ⟨(E_loc - E_mean) ∇log ψ⟩ = (1/N) Σ (E_loc[i] - E_mean) ∇log ψ(σ[i])
    # grad_matrix 已经是 PyTree 结构，每个元素的形状是 (n_samples, ...)
    # 关键修复：O_centered 形状为 (n_samples,)，需要正确广播到梯度张量的每个维度
    # 使用 reshape 将 O_centered 变为 (n_samples, 1, 1, ..., 1) 以匹配梯度张量
    def weight_and_mean(grad_component):
        # grad_component 形状：(n_samples, d1, d2, ...)
        # O_centered 形状：(n_samples,)
        # 需要广播相乘后沿 axis=0 求平均
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    
    return O_mean, O_std, grad
```

**详细步骤解析**：

#### 步骤 1-3：计算中心化的局部能量
```python
O_loc = compute_local_energies(machine, params, sigma)  # 形状：(n_samples,)
O_mean, O_std = statistics(O_loc)
O_centered = O_loc - O_mean  # 形状：(n_samples,)
```

#### 步骤 4：批量计算每个样本的梯度
```python
def compute_grad_for_sample(s):
    return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
```

**关键点**：
- `jax.grad(..., holomorphic=True)`：对复数函数求导，使用 holomorphic 梯度
- `jax.vmap`：向量化操作，批量处理所有样本
- `grad_matrix` 是一个 PyTree，包含网络中每个参数的梯度
  - 例如：`linear1.kernel` 的梯度形状为 `(n_samples, 4, 12)`
  - `linear1.bias` 的梯度形状为 `(n_samples, 12)`

#### 步骤 5：加权平均得到最终梯度（关键修复）

```python
def weight_and_mean(grad_component):
    # grad_component 形状：(n_samples, d1, d2, ...)
    # O_centered 形状：(n_samples,)
    weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
    return jnp.mean(weights * jnp.conj(grad_component), axis=0)

grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
```

**形状广播的关键**：
- `O_centered` 原始形状：`(1008,)`
- 对于 `linear1.kernel` 梯度 `(1008, 4, 12)`，需要 reshape 为 `(1008, 1, 1)`
- 对于 `linear1.bias` 梯度 `(1008, 12)`，需要 reshape 为 `(1008, 1)`
- 通过广播机制，实现逐样本加权

**为什么要用 `jnp.conj`**：
- 梯度公式中包含 $\nabla \log \psi^*$（复共轭）
- 对于全纯函数，$\nabla \log \psi^* = (\nabla \log \psi)^*$

## 四、训练循环

### 4.1 初始化

```python
rngs = nnx.Rngs(21)
model = SingleStateAnsatz(4, hidden_dim=12, rngs=rngs)
machine, graphdef, params = create_machine(model)

sampler_state = sampler.init_state(machine, params, seed=1)
optimizer = optax.sgd(learning_rate=0.01)  # 学习率 0.01
opt_state = optimizer.init(params)

# 训练参数
N_ITER = 800   # 迭代 800 次
N_SAMPLES = 1008  # 样本数
USE_SR = False  # 不使用 SR 自然梯度
```

### 4.2 训练主循环

```python
history = {
    'step': [],
    'energy': [],
    'energy_std': [],
    'error': []
}

for step in range(N_ITER):
    # 1. 采样
    samples, sampler_state = sampler.sample(
        machine, params, state=sampler_state, 
        chain_length=N_SAMPLES // sampler.n_chains
    )
    samples = samples.reshape(-1, hi.size)
    
    # 2. 计算 force-based 能量和梯度
    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
    
    # 3. 应用 SR 自然梯度（本例不使用）
    if USE_SR:
        grad = apply_sr(machine, params, samples, grad, diag_shift=0.1)
    
    # 4. 更新参数（朴素梯度下降）
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    # 5. 记录历史
    if step % 10 == 0 or step == N_ITER - 1:
        error = jnp.abs(energy.real - E_fcis[0])
        history['step'].append(step)
        history['energy'].append(float(energy.real))
        history['energy_std'].append(float(energy_std))
        history['error'].append(float(error))
        print(f"Step {step:3d} | E: {energy.real:.8f} ± {energy_std:.6f} | FCI: {E_fcis[0]:.8f} | Error: {error:.6f}")
```

**训练流程**：
1. **采样**：使用 Metropolis 采样生成新的样本
2. **计算梯度**：调用自定义的 `forces_expect_hermitian` 函数
3. **参数更新**：使用 Optax 的 SGD 优化器
4. **记录日志**：每 10 步记录一次能量和误差

## 五、关键技术点总结

### 5.1 复数梯度处理

```python
jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)
```

- `holomorphic=True` 告诉 JAX 函数是全纯的（复可微）
- 这样 JAX 会自动处理复数梯度的计算规则

### 5.2 PyTree 操作

```python
grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
```

- `grad_matrix` 是 PyTree 结构，包含所有参数的梯度
- `tree_map` 对 PyTree 中每个张量应用相同的操作
- 保持了参数的层次结构

### 5.3 形状广播修复

**问题**：
- `O_centered` 形状：`(n_samples,)`
- 梯度分量形状：`(n_samples, d1, d2, ...)`
- 直接相乘会导致广播错误

**解决**：
```python
weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
```

动态添加维度，确保正确的广播行为。

### 5.4 与 NetKet 的对比

| 功能 | 本实现 | NetKet 原版 |
|------|--------|-------------|
| 梯度计算 | `jax.grad(holomorphic=True)` | `nkjax.vjp(conjugate=True)` |
| 优化器 | Optax SGD | NetKet SGD + SR |
| 梯度公式 | Force-based | Force-based |
| 数学等价性 | ✅ 是 | ✅ 是 |

## 六、实验结果

### 6.1 典型训练输出

```
============================================================
开始纯 JAX VMC 训练 (朴素梯度下降法)
============================================================
Step   0 | E: -0.49394149 ± 0.007212 | FCI: -1.01546825 | Error: 0.521527
Step  50 | E: -0.82518315 ± 0.005686 | FCI: -1.01546825 | Error: 0.190285
Step 100 | E: -0.85778405 ± 0.003305 | FCI: -1.01546825 | Error: 0.157684
...
Step 799 | E: -0.87533746 ± 0.001391 | FCI: -1.01546825 | Error: 0.140131

============================================================
训练完成!
最终能量：-0.87546783 ± 0.001389 Ha
FCI 基准：-1.01546825 Ha
绝对误差：0.140000 Ha
相对误差：13.7868%
============================================================
```

### 6.2 结果分析

1. **收敛趋势**：
   - 初始能量约 -0.49 Ha，误差 0.52 Ha
   - 最终能量约 -0.875 Ha，误差 0.14 Ha
   - 能量单调下降，表明优化有效

2. **误差来源**：
   - 朴素梯度下降收敛较慢
   - 不使用 SR 预处理，收敛速度受限
   - 可以通过增加迭代次数或使用 SR 改进

3. **与 SR 对比**：
   - SR（随机重配置）是自然梯度方法
   - SR 通常收敛更快，但计算成本更高
   - 朴素梯度下降实现简单，适合理解原理

## 七、完整代码流程

```
初始化
  ↓
定义神经网络 Ansatz
  ↓
包装为 machine 函数
  ↓
初始化采样器和优化器
  ↓
训练循环 (N_ITER 次)
  ├─ 采样 (Metropolis)
  ├─ 计算局部能量 E_loc
  ├─ 计算能量均值 ⟨E⟩
  ├─ 中心化：E_loc - ⟨E⟩
  ├─ 计算 ∇log ψ (jax.grad + holomorphic=True)
  ├─ 加权平均：⟨(E_loc - ⟨E⟩) ∇log ψ⟩
  ├─ 参数更新：θ ← θ - η·grad
  └─ 记录日志
  ↓
输出最终结果
```

## 八、扩展与改进

### 8.1 添加 SR 自然梯度

可以参考 `VMC_jax_native.ipynb` 中的 SR 实现：

```python
@partial(jax.jit, static_argnames=("machine",))
def apply_sr(machine, params, sigma, grad, diag_shift=0.1):
    """应用 SR 预处理：求解 F⁻¹ ∇E"""
    qgt_reg, unravel_fn = compute_qgt_and_flat_grad(machine, params, sigma, diag_shift)
    grad_flat, _ = flatten_util.ravel_pytree(grad)
    nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
    nat_grad = unravel_fn(nat_grad_flat)
    return nat_grad
```

### 8.2 调整学习率

可以使用学习率调度：

```python
import optax

# 学习率衰减
scheduler = optax.exponential_decay(
    init_value=0.01,
    decay_steps=100,
    decay_rate=0.95
)
optimizer = optax.sgd(learning_rate=scheduler)
```

### 8.3 使用 Adam 优化器

```python
optimizer = optax.adam(learning_rate=0.001)
```

## 九、参考资料

- [VMC_jax_native.ipynb](../0504/VMC_jax_native.ipynb) - 包含 SR 的完整实现
- [VMC 的梯度计算.md](../0504/VMC 的梯度计算.md) - 梯度公式推导
- [VMC 梯度计算对比分析.ipynb](./VMC 梯度计算对比分析.ipynb) - 与 NetKet 的详细对比
- NetKet 官方文档：https://www.netket.org/

## 十、总结

本文档展示了如何：

1. ✅ 使用纯 JAX 实现 VMC 的 force-based 梯度计算
2. ✅ 正确处理复数神经网络的梯度（`holomorphic=True`）
3. ✅ 解决形状广播问题（动态 reshape）
4. ✅ 使用 Optax 进行参数更新
5. ✅ 完成完整的 VMC 优化循环

关键优势：
- **不依赖 NetKet 黑盒**：完全理解梯度计算过程
- **灵活可扩展**：可以轻松修改网络结构、优化器等
- **教学价值**：深入理解 VMC 的数学原理和数值实现
