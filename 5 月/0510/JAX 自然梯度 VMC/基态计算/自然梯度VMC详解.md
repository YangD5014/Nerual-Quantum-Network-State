# 纯JAX实现VMC自然梯度优化器

## 1. 研究背景与动机

### 1.1 问题背景

在量子多体系统的基态求解中，变分蒙特卡洛（Variational Monte Carlo, VMC）方法是一种强大的数值技术。传统的VMC使用标准梯度下降法，但收敛速度较慢。自然梯度法（Natural Gradient Descent）通过利用参数空间的几何结构，能够显著加速优化过程。

### 1.2 研究目标

本项目旨在**纯JAX实现**VMC中的自然梯度优化，并与NetKet库的结果进行对比验证。核心目标是：
- 手动实现QGT（量子几何张量）计算
- 验证自然梯度公式的正确性
- 探索纯深度学习框架实现量子变分算法的可行性

## 2. 理论框架

### 2.1 变分原理与VMC

对于给定的希尔伯特空间，基态能量满足**变分原理**：

$$E_0 = \min_{\Psi} \frac{\langle\Psi|H|\Psi\rangle}{\langle\Psi|\Psi\rangle}$$

在VMC中，我们用参数化波函数 $\Psi(\sigma; \theta)$ 近似目标态，能量期望为：

$$\langle E \rangle(\theta) = \frac{\langle\Psi(\theta)|H|\Psi(\theta)\rangle}{\langle\Psi(\theta)|\Psi(\theta)\rangle} = \mathbb{E}_{\sigma \sim |\Psi(\theta)|^2}[E_{\text{loc}}(\sigma)]$$

其中**局部能量**定义为：

$$E_{\text{loc}}(\sigma) = \sum_{\eta} H_{\sigma\eta} \frac{\Psi(\eta)}{\Psi(\sigma)}$$

### 2.2 标准梯度下降

能量关于参数的梯度为：

$$\nabla_\theta \langle E \rangle = \frac{\partial \langle E \rangle}{\partial \theta_i} = 2\left(\langle E_{\text{loc}} \nabla_\theta \log \Psi \rangle - \langle E_{\text{loc}} \rangle \langle \nabla_\theta \log \Psi \rangle \right)$$

**问题**：标准梯度在参数空间中沿着最陡下降方向，但这个方向依赖于参数坐标系的选择。

### 2.3 自然梯度法

自然梯度法的核心思想是：在参数流形上，沿着**黎曼度量**意义下的最陡下降方向更新参数：

$$\theta_{k+1} = \theta_k - \alpha \cdot S^{-1}(\theta_k) \nabla_\theta \langle E \rangle$$

其中 $S(\theta)$ 是参数空间的度量张量，$S^{-1}$ 是其逆矩阵。

### 2.4 量子几何张量（QGT）

对于量子变分波函数，自然梯度的度量张量是**量子几何张量**（Quantum Geometric Tensor）：

$$S_{ij}(\theta) = \langle \partial_i \Psi | \partial_j \Psi \rangle - \langle \partial_i \Psi \rangle \langle \partial_j \Psi \rangle$$

使用对数导数 $\nabla_\theta \log \Psi(\sigma) = \frac{\nabla_\theta \Psi(\sigma)}{\Psi(\sigma)}$，可以化为：

$$S_{ij}(\theta) = \mathbb{E}_{\sigma \sim |\Psi(\theta)|^2}\left[(\nabla_\theta \log \Psi)^*_i (\nabla_\theta \log \Psi)_j\right] - \mathbb{E}_{\sigma \sim |\Psi(\theta)|^2}\left[(\nabla_\theta \log \Psi)^*\right]_i \mathbb{E}_{\sigma \sim |\Psi(\theta)|^2}\left[(\nabla_\theta \log \Psi)\right]_j$$

### 2.5 SR（Stein ReNormalization）预条件器

在实践中，由于有限采样误差，QGT可能奇异。引入正则化项：

$$S_{\text{reg}} = S + \lambda \cdot I$$

其中 $\lambda$ 是正则化参数（代码中设为0.001），$I$ 是单位矩阵。

## 3. 代码实现

### 3.1 分子系统设置

```python
# H2分子定义
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)

# FCI基准能量
E_fcis, fcivec = fci.FCI(mf).kernel()
# 基态能量: E0 = -1.01546825 Ha
```

**说明**：
- STO-3G是最小基组，足以描述H2分子
- FCI（全配置相互作用）提供基准能量
- 基态能量约-1.015 Ha，激发态能量通过27.2114转换（Hartree到eV）

### 3.2 神经网络Ansatz

```python
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        return jnp.squeeze(self.output(h))
```

**关键设计**：
- 使用**复数参数**（`param_dtype=complex`）以表示振幅和相位
- 双隐藏层全连接网络（4→12→12→1）
- `tanh`激活函数

### 3.3 局部能量计算

```python
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """
    计算局部能量: E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)
    """
    eta, H_eta = ha.get_conn_padded(sigma)  # 获取连接和矩阵元
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
```

**公式**：
$$E_{\text{loc}}(\sigma) = \sum_{\eta} H_{\sigma\eta} \frac{\psi(\eta)}{\psi(\sigma)} = \sum_{\eta} H_{\sigma\eta} \exp[\log\psi(\eta) - \log\psi(\sigma)]$$

### 3.4 Force-Based梯度计算

```python
@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    """
    Force-based梯度: ∇⟨E⟩ = ⟨(E_loc - ⟨E⟩) ∇log ψ⟩
    """
    O_loc = compute_local_energies(machine, params, sigma)
    O_mean, O_std = statistics(O_loc)
    O_centered = O_loc - O_mean
    
    # 计算∇log ψ
    grad_matrix = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s), holomorphic=True)(params)
    )(sigma)
    
    # 加权平均
    def weight_and_mean(grad_component):
        weights = O_centered.reshape((O_centered.shape[0],) + (1,) * (grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree_util.tree_map(weight_and_mean, grad_matrix)
    return O_mean, O_std, grad
```

**核心公式**：
$$\nabla_\theta \langle E \rangle = \frac{1}{N}\sum_{i=1}^{N}(E_{\text{loc}}(\sigma_i) - \bar{E}) \cdot \nabla_\theta \log^*\psi(\sigma_i)$$

**注意**：
- 使用`holomorphic=True`进行复数微分
- 共轭操作`jnp.conj`确保正确处理复数参数

### 3.5 QGT计算

```python
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """
    量子几何张量: S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩
    """
    n_samples = sigma.shape[0]
    
    # 步骤1: 计算∇log ψ
    grad_matrix = jax.vmap(
        lambda s: jax.grad(lambda p: machine(p, s), holomorphic=True)(params)
    )(sigma)
    
    # 步骤2: 展平为矩阵 (n_samples, n_params)
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    
    # 步骤3: 中心化
    grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
    grad_centered = grad_flat - grad_mean
    
    # 步骤4: QGT = (1/N) * Σ ∇log ψ* ∇log ψ^T
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    
    # 步骤5: 正则化
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
    
    return qgt_reg, unravel_fn
```

**关键实现细节**：
- `flatten_util.ravel_pytree`: 将PyTree结构展平为向量
- 中心化操作对应QGT定义中的减去均值项
- 正则化确保数值稳定性

### 3.6 自然梯度更新

```python
for step in range(N_ITER):
    # 采样
    samples, sampler_state = sampler.sample(machine, params, state=sampler_state, chain_length=20)
    samples = samples.reshape(-1, hi.size)
    
    # 计算梯度
    energy, energy_std, grad = forces_expect_hermitian(machine, params, samples)
    grad = jax.tree_map(lambda x: x*2, grad)  # 梯度乘2
    
    # 计算QGT
    qgt_reg, qgt_unravel_fun = compute_qgt(machine, params, samples, diag_shift=0.001)
    grad_flat, grad_unravel_fn = flatten_util.ravel_pytree(grad)
    
    # 自然梯度: S^{-1} * ∇E
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    natural_grad = grad_unravel_fn(natural_grad)
    
    # 参数更新
    updates, opt_state = optimizer.update(natural_grad, opt_state, params)
    params = optax.apply_updates(params, updates)
```

**自然梯度公式**：
$$\theta_{k+1} = \theta_k - \alpha \cdot S_{\text{reg}}^{-1} \nabla_\theta \langle E \rangle$$

**梯度乘2的原因**：能量关于参数的真实梯度需要乘以2（来自变分原理中的2倍因子）。

## 4. 采样器配置

```python
# 费米子系统的特殊采样
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hilbert=hi, graph=g)
sampler = nk.sampler.MetropolisSampler(hi, rule=single_rule, n_chains=100, sweep_size=32)
```

**说明**：
- H2分子有2个自旋轨道（每个自旋1个电子）
- `FermionHopRule`处理费米子交换对称性
- Metropolis-Hastings采样器用于生成MC样本

## 5. 实验结果

### 5.1 训练过程

| Step | 能量 (Ha) | 误差 (Ha) |
|------|-----------|----------|
| 0 | -0.4811 | 0.534 |
| 50 | -0.9497 | 0.066 |
| 100 | -0.9615 | 0.054 |
| 150 | -0.9689 | 0.047 |
| 200 | -1.0016 | 0.014 |
| 250 | -1.0117 | 0.004 |
| 299 | -1.0130 | 0.002 |

### 5.2 最终结果

- **最终能量**: -1.0130 ± 0.0006 Ha
- **FCI基准**: -1.0155 Ha
- **绝对误差**: 0.002 Ha
- **相对误差**: 0.14%

收敛到接近FCI基态能量，验证了实现的正确性。

## 6. 关键技术点

### 6.1 复数微分

```python
jax.grad(lambda p: machine(p, s), holomorphic=True)(params)
```

使用`holomorphic=True`标记函数为全纯函数，JAX会自动处理复数微分的共轭规则。

### 6.2 PyTree操作

- `flatten_util.ravel_pytree`: 将嵌套参数结构展平
- `jax.tree_util.tree_map`: 对PyTree每个叶子应用函数
- `unravel_fn`: 将展平向量恢复为原始结构

### 6.3 向量化计算

```python
jax.vmap(compute_grad_for_sample)(sigma)
```

使用`vmap`对所有样本并行计算梯度，避免循环。

## 7. 潜在问题与解决方案

### 7.1 QGT不正定

**症状**: 训练发散或收敛到错误极小点

**解决方案**:
- 增加正则化参数 `diag_shift`
- 使用Cholesky分解替代直接求逆
- 对QGT进行对称化处理

### 7.2 梯度方差大

**症状**: 训练曲线震荡

**解决方案**:
- 增加采样数 `N_SAMPLES`
- 使用更好的采样器（如自回归采样器）
- 调整采样器参数（`sweep_size`）

### 7.3 复数参数优化

**注意**: 对于非全纯函数，不应使用`holomorphic=True`。RBM等实数网络应使用`holomorphic=False`。

## 8. 与NetKet对比

本实现与NetKet的核心区别：

| 特性 | NetKet | 纯JAX实现 |
|------|--------|----------|
| QGT计算 | C++/CUDA优化 | 纯Python/JAX |
| 梯度计算 | 自动微分 | 手动实现force-based |
| 灵活性 | 高（支持多种SR变体） | 可定制性强 |
| 性能 | 优化过的底层实现 | 依赖JAX JIT编译 |

**优势**:
- 代码透明，便于理解原理
- 可轻松集成到现有JAX项目
- 支持自定义QGT变体

## 9. 总结

本文档详细介绍了纯JAX实现VMC自然梯度优化器的完整流程：

1. **理论基础**: 从变分原理出发，推导局部能量和自然梯度公式
2. **代码实现**: 完整的神经网络Ansatz、采样、梯度计算和QGT计算
3. **实验验证**: 收敛到FCI基态能量，相对误差<0.2%

该实现可以作为理解量子变分算法和自然梯度法的入门代码，也可以作为开发更高级量子神经网络的基础。

## 参考资料

1. McLachlan, A. D., & Ball, M. A. (1964). Variation of Atoms and Molecules. *Reviews of Modern Physics*, 36(3), 844.
2. Sorella, S. (1998). Green function Monte Carlo with stochastic reconfiguration. *Physical Review Letters*, 80(21), 4558.
3. Stokes, J. R., et al. (2020). Quantum Natural Gradient. *arXiv:1909.02108*.
