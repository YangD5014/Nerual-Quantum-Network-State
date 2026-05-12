# 纯 JAX 实现 VMC 梯度计算 - 说明文档（修复版）

## 文件说明

### 创建的文件

1. **`VMC_jax_native.ipynb`** - 主要的 Jupyter Notebook（已修复）
2. **`test_vmc_jax_fixed.py`** - 测试脚本（修复版）
3. **`README.md`** - 本说明文档

### 参考文档

- `/Users/yangjianfei/mac_vscode/神经网络量子态/4 月/0424/目前的 API 调查进展.md` - API 调查笔记
- `/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0504/VMC_previous.py` - 原始有问题的代码

## 问题与修复

### 原始问题

`VMC_previous.py` 中的梯度计算使用了 `jax.value_and_grad`，这不是 QMC 标准的 force-based 梯度。

### 第一次尝试的问题

尝试使用 `jax.jacobian` 计算梯度时遇到复数值网络的问题：
```python
# 错误：会报错
grad_matrix = jax.jacobian(log_psi_fun, holomorphic=True)(params)
grad_flat, unflatten = flatten_util.ravel_pytree(grad_matrix)
grad_flat = grad_flat.reshape(sigma.shape[0], -1)
grad = unflatten(grad_flat)  # ValueError!
```

**问题原因**: `flatten_util.ravel_pytree` 期望输入是 1D 数组，但 jacobian 返回的是 2D 数组 (n_samples, params)。

### 最终修复方案

使用 `jax.grad` + `jax.vmap` 的组合：

```python
# 正确实现
def compute_grad_for_sample(s):
    return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)

# 直接使用 tree_map 处理 PyTree
grad = jax.tree_util.tree_map(
    lambda g: jnp.mean(jnp.expand_dims(O_centered, -1) * jnp.conj(g), axis=0),
    grad_matrix
)
```

## 核心实现

### Force-based 梯度公式

NetKet 使用 force-based 形式计算梯度：

$$
\nabla \langle E \rangle = \langle (E_{loc} - \langle E \rangle) \nabla \log \psi \rangle
$$

其中：
- $E_{loc}(\sigma) = \sum_\eta H_{\sigma\eta} \frac{\psi(\eta)}{\psi(\sigma)}$ 是局部能量
- $\nabla \log \psi$ 是波函数对数梯度

### 关键代码实现

```python
@partial(jax.jit, static_argnames=("machine",))
def forces_expect_hermitian(machine, params, sigma):
    # 1. 计算局部能量
    O_loc = compute_local_energies(machine, params, sigma)
    
    # 2. 统计能量均值
    O_mean, O_std = statistics(O_loc)
    
    # 3. 中心化局部能量
    O_centered = O_loc - O_mean
    
    # 4. 计算 ∇log ψ 对每个样本
    def log_psi_single(p, s):
        return machine(p, s)
    
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)
    
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    # 5. 计算 force-based 梯度
    grad = jax.tree_util.tree_map(
        lambda g: jnp.mean(jnp.expand_dims(O_centered, -1) * jnp.conj(g), axis=0),
        grad_matrix
    )
    
    return O_mean, O_std, grad
```

### 与 NetKet 源码的对应

| 本实现 | NetKet 源码位置 |
|--------|----------------|
| `compute_local_energies` | `vqs/mc/kernels.py:local_value_kernel` |
| `forces_expect_hermitian` | `vqs/mc/mc_state/expect_forces.py` |
| `compute_qgt_and_flat_grad` | SR 预处理部分 |
| `apply_sr` | QGT 应用部分 |

## 技术要点

### 1. 复数值网络的处理

对于复数值神经网络，必须使用 `holomorphic=True`：

```python
# 正确
grad = jax.grad(log_psi_single, holomorphic=True)(params)

# 错误（会报错）
grad = jax.grad(log_psi_single)(params)
# TypeError: jacrev requires real-valued outputs...
```

### 2. PyTree 处理

直接使用 `jax.tree_util.tree_map` 处理 PyTree 结构：

```python
# 正确：直接处理 PyTree
grad = jax.tree_util.tree_map(
    lambda g: jnp.mean(jnp.expand_dims(O_centered, -1) * jnp.conj(g), axis=0),
    grad_matrix
)

# 错误：手动展平会出错
grad_flat, unflatten = flatten_util.ravel_pytree(grad_matrix)
grad_flat = grad_flat.reshape(n_samples, -1)
grad = unflatten(grad_flat)  # ValueError!
```

### 3. 向量化

使用 `jax.vmap` 批量处理样本：

```python
# 对每个样本计算梯度
def compute_grad_for_sample(s):
    return jax.grad(lambda p: log_psi_single(p, s), holomorphic=True)(params)

# 批量处理所有样本
grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
```

### 4. SR 自然梯度

量子几何张量 (QGT) 的计算和应用：

```python
# QGT = ⟨∇log ψ ∇log ψ†⟩
grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
grad_flat = grad_flat.reshape(n_samples, -1)
grad_centered = grad_flat - jnp.mean(grad_flat, axis=0, keepdims=True)
qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered

# 正则化
qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])

# 求解自然梯度
grad_flat, _ = flatten_util.ravel_pytree(grad)
nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
nat_grad = unravel_fn(nat_grad_flat)
```

## 使用方法

### 运行 Notebook

```bash
cd "/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0504"
jupyter notebook VMC_jax_native.ipynb
```

### 运行测试脚本

```bash
cd "/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0504"
python test_vmc_jax_fixed.py
```

## 环境要求

```bash
pip install jax jaxlib flax optax netket pyscf
```

推荐版本：
- JAX: 0.4.x
- Flax: 0.8.x
- NetKet: 3.9.x

## 预期结果

训练完成后，能量应该收敛到接近 FCI 基准值：

```
H₂ FCI 基准能量
============================================================
E0 = -1.13730604 Ha  |  激发能：0.0000 eV
E1 = -0.16275316 Ha  |  激发能：26.4785 eV
E2 =  0.49505774 Ha  |  激发能：44.3709 eV

训练完成！
最终能量：-1.13xxxxxx ± 0.00xxxx Ha
FCI 基准：-1.13730604 Ha
绝对误差：0.00xxxx Ha
```

## 核心公式总结

1. **局部能量**
   $$E_{loc}(\sigma) = \sum_\eta H_{\sigma\eta} \frac{\psi(\eta)}{\psi(\sigma)}$$

2. **Force-based 梯度**
   $$\nabla \langle E \rangle = \langle (E_{loc} - \langle E \rangle) \nabla \log \psi \rangle$$

3. **QGT 矩阵**
   $$F = \langle \nabla \log \psi \nabla \log \psi^\dagger \rangle$$

4. **自然梯度**
   $$\nabla_{nat} = F^{-1} \nabla E$$

## 参考资料

1. [NetKet 文档](https://netket.readthedocs.io/)
2. [JAX 自动微分文档](https://jax.readthedocs.io/en/latest/jax.html#automatic-differentiation)
3. [Flax NNX 文档](https://flax.readthedocs.io/en/latest/nnx/index.html)
4. API 调查笔记：`/Users/yangjianfei/mac_vscode/神经网络量子态/4 月/0424/目前的 API 调查进展.md`

## 修复历史

### v2 (当前版本) - 2024-05-04
- ✅ 使用 `jax.grad` + `holomorphic=True` 处理复数值网络
- ✅ 使用 `jax.vmap` 批量处理样本
- ✅ 直接使用 `jax.tree_util.tree_map` 处理 PyTree
- ✅ 避免了 `flatten_util.ravel_pytree` 的 2D 数组问题

### v1 (初始版本) - 2024-05-04
- ❌ 使用 `jax.jacobian` 导致 PyTree 展平问题
- ❌ `flatten_util.ravel_pytree` 处理 2D 数组时报错
