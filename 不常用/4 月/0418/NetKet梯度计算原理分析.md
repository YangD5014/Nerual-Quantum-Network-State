# NetKet 梯度计算原理深度分析

## 目录
1. [背景与问题](#背景与问题)
2. [NetKet 梯度计算流程](#netket-梯度计算流程)
3. [源码分析](#源码分析)
4. [关键差异对比](#关键差异对比)
5. [数学推导](#数学推导)
6. [修复方案](#修复方案)

---

## 背景与问题

在使用 NetKet 进行变分蒙特卡洛（VMC）计算分子基态能量时，发现两种实现方式结果差异巨大：

- **NetKet 原生实现**：能量收敛到 FCI 精度（-1.01546826 Ha）
- **自定义实现**：能量无法收敛，效果很差

**核心问题**：在相同样本、相同模型参数的情况下，期望值一致，但梯度不一致！

---

## NetKet 梯度计算流程

### 整体架构

```
VMC Driver
    ↓
expect_and_grad (入口)
    ↓
expect_and_forces (对于 hermitian 算符)
    ↓
force_to_grad (转换 forces 为梯度)
```

### 关键文件路径

1. **VMC 驱动**: `/netket/driver/vmc.py`
2. **梯度计算入口**: `/netket/vqs/mc/mc_state/expect_grad.py`
3. **Forces 计算**: `/netket/vqs/mc/mc_state/expect_forces.py`
4. **局部能量核**: `/netket/vqs/mc/kernels.py`
5. **Force 转换**: `/netket/vqs/mc/common.py`

---

## 源码分析

### 1. 局部能量计算 (`kernels.py`)

```python
@batch_discrete_kernel
def local_value_kernel(logpsi: Callable, pars: PyTree, σ: Array, args: PyTree):
    """
    计算局部能量 E_loc(σ)
    
    数学公式：
    E_loc(σ) = Σ_η H_{σ,η} * exp(logψ(η) - logψ(σ))
    """
    σp, mel = args
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))
```

**关键点**：
- 使用 `logψ(η) - logψ(σ)` 避免数值溢出
- 批量处理所有连接态 η

### 2. 期望值计算 (`expect.py`)

```python
@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Stats:
    # 计算局部能量
    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)
    
    # 计算统计量（均值、方差等）
    Ō_stats = mpi_statistics(L_σ.reshape((n_chains, -1)))
    
    return Ō_stats
```

**关键点**：
- 返回的是统计对象，包含均值和方差
- 使用 MPI 进行分布式计算

### 3. 梯度计算核心 (`expect_forces.py`)

```python
@partial(jax.jit, static_argnums=(0, 1, 2))
def forces_expect_hermitian(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree]:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    n_samples = σ.shape[0] * mpi.n_nodes

    # 1. 计算局部能量
    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    # 2. 计算期望值
    Ō = statistics(O_loc.reshape((n_chains, -1)))

    # 3. 【关键】中心化处理
    O_loc -= Ō.mean

    # 4. 计算 VJP（向量-雅可比积）
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,  # 【关键】使用共轭
        has_aux=is_mutable,
    )
    
    # 5. 【关键】计算梯度
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    Ō_grad, _ = mpi.mpi_sum_jax(Ō_grad)

    return (Ō, Ō_grad, new_model_state)
```

**三大关键点**：
1. **中心化**: `O_loc -= Ō.mean` 
2. **共轭**: `jnp.conjugate(O_loc)`
3. **VJP**: 使用 `conjugate=True` 参数

### 4. Force 到梯度的转换 (`common.py`)

```python
@jax.jit
def force_to_grad(Ō_grad, parameters):
    """
    将 forces 转换为梯度
    
    对于复数参数：grad = 2 * force
    对于实数参数：grad = 2 * Re(force)
    """
    Ō_grad = jax.tree_util.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )
    # 【关键】乘以 2
    Ō_grad = jax.tree_util.tree_map(lambda x: 2 * x, Ō_grad)
    return Ō_grad
```

---

## 关键差异对比

### 用户实现 vs NetKet 实现

| 步骤 | 用户实现 | NetKet 实现 | 影响 |
|------|---------|------------|------|
| **局部能量** | ✓ 正确 | ✓ 正确 | 期望值一致 |
| **中心化** | ✗ 未使用 | ✓ `O_loc -= Ō.mean` | **关键差异** |
| **共轭** | ✗ 未使用 | ✓ `jnp.conjugate(O_loc)` | **关键差异** |
| **VJP** | ✓ 使用 `value_and_grad` | ✓ 使用 `vjp` | 方法不同 |
| **梯度缩放** | ✗ 未乘以 2 | ✓ `2 * grad` | **关键差异** |

### 用户代码中的错误

```python
# 用户的错误实现
def loss_fn(s):
    log_psi = model_forward((graphdef, s), samples)
    eta, H_sigmaeta = hamiltonian.get_conn_padded(samples)
    logpsi_eta = model_forward((graphdef, s), eta)
    
    log_psi = jnp.expand_dims(log_psi, axis=-1)
    Eloc = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - log_psi), axis=-1)
    energy = jnp.mean(Eloc)
    
    # ❌ 错误1：没有中心化
    # ❌ 错误2：直接用 Eloc 而不是 centered_Eloc
    # ❌ 错误3：没有使用共轭
    loss = jnp.mean(Eloc * log_psi.squeeze())
    
    return loss, energy
```

---

## 数学推导

### 正确的梯度公式

对于变分波函数 ψ(θ)，能量期望值为：

```
E(θ) = <ψ(θ)|H|ψ(θ)> / <ψ(θ)|ψ(θ)>
```

梯度为：

```
∂E/∂θ = 2 Re[<∂ψ/∂θ|H|ψ> - E<∂ψ/∂θ|ψ>]
      = 2 Re[cov(O_k, E_loc)]
```

其中 `O_k = ∂logψ/∂θ_k` 是对数导数算符。

### NetKet 的实现逻辑

1. **计算局部能量**：
   ```
   E_loc(σ) = Σ_η H_{σ,η} ψ(η)/ψ(σ)
   ```

2. **中心化**：
   ```
   E_loc^c = E_loc - <E_loc>
   ```

3. **计算 Forces**：
   ```
   F_k = <O_k^* E_loc^c> / N
   ```
   通过 VJP 实现：
   ```
   F = VJP[logψ](conj(E_loc^c) / N)
   ```

4. **转换为梯度**：
   ```
   grad_k = 2 * F_k  (复数参数)
   grad_k = 2 * Re(F_k)  (实数参数)
   ```

### 为什么需要中心化？

**数学原因**：

```
∂E/∂θ = 2 Re[<O_k^* E_loc> - <O_k^*> <E_loc>]
      = 2 Re[cov(O_k, E_loc)]
```

协方差公式本身就包含中心化项！

**物理意义**：
- 中心化消除了波函数归一化的影响
- 确保梯度只与能量涨落有关，而不是绝对值

---

## 修复方案

### 正确的实现

```python
@partial(jax.jit, static_argnames=("model_forward", "graphdef"))
def energy_and_grad_correct(graphdef, state, model_forward, hamiltonian, samples):
    """
    完全按照 NetKet 的方式计算梯度和能量
    """
    n_samples = samples.shape[0]
    
    def logpsi_fn(s):
        return model_forward((graphdef, s), samples)
    
    # 1. 计算局部能量
    def compute_eloc(s):
        log_psi = model_forward((graphdef, s), samples)
        eta, H_sigmaeta = hamiltonian.get_conn_padded(samples)
        logpsi_eta = model_forward((graphdef, s), eta)
        
        log_psi_expanded = jnp.expand_dims(log_psi, axis=-1)
        Eloc = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - log_psi_expanded), axis=-1)
        return Eloc
    
    Eloc = compute_eloc(state)
    energy = jnp.mean(Eloc)
    
    # 2. 【关键】中心化
    Eloc_centered = Eloc - energy
    
    # 3. 【关键】使用 VJP 计算梯度
    # 注意：这里需要使用 jax.vjp 而不是 jax.value_and_grad
    _, vjp_fn = jax.vjp(logpsi_fn, state)
    
    # 4. 【关键】使用共轭
    grad = vjp_fn(jnp.conjugate(Eloc_centered) / n_samples)[0]
    
    # 5. 【关键】乘以 2
    grad = jax.tree_map(lambda x: 2 * x, grad)
    
    return energy, grad
```

### 使用 `jax.value_and_grad` 的实现

如果坚持使用 `jax.value_and_grad`，需要构造正确的 loss 函数：

```python
@partial(jax.jit, static_argnames=("model_forward", "graphdef"))
def energy_and_grad_with_value_and_grad(graphdef, state, model_forward, hamiltonian, samples):
    """
    使用 value_and_grad 的正确实现
    """
    n_samples = samples.shape[0]
    
    def loss_fn(s):
        log_psi = model_forward((graphdef, s), samples)
        eta, H_sigmaeta = hamiltonian.get_conn_padded(samples)
        logpsi_eta = model_forward((graphdef, s), eta)
        
        log_psi_expanded = jnp.expand_dims(log_psi, axis=-1)
        Eloc = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - log_psi_expanded), axis=-1)
        energy = jnp.mean(Eloc)
        
        # 【关键】中心化
        Eloc_centered = Eloc - energy
        
        # 【关键】构造正确的 loss
        # 这等价于 cov(O_k, E_loc)
        loss = jnp.mean(Eloc_centered * log_psi)
        
        return loss, energy
    
    # 使用 holomorphic=True 处理复数参数
    (loss, energy), grad = jax.value_and_grad(loss_fn, has_aux=True, holomorphic=True)(state)
    
    # 【关键】乘以 2
    grad = jax.tree_map(lambda x: 2 * x, grad)
    
    return energy, grad
```

---

## 总结

### 核心要点

1. **中心化是必须的**：`E_loc_centered = E_loc - <E_loc>`
2. **共轭处理**：`jnp.conjugate(E_loc_centered)`
3. **梯度缩放**：最终梯度需要乘以 2
4. **VJP vs value_and_grad**：两种方法都可以，但需要正确构造 loss

### 为什么用户的实现失败？

用户代码的主要问题：
1. ❌ 没有中心化局部能量
2. ❌ 没有正确处理复数梯度
3. ❌ 没有乘以 2 的缩放因子

这些错误导致梯度方向和大小都不正确，优化过程无法收敛。

### NetKet 的设计哲学

NetKet 的实现遵循了严格的数学推导：
- 使用协方差公式计算梯度
- 通过 VJP 高效计算向量-雅可比积
- 正确处理复数参数的梯度

这种实现既高效又数学上严格，是 VMC 方法的标准实现方式。
