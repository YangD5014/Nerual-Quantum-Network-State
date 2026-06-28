# `expect_and_forces()` 函数调查报告

## 1. 函数定位

`vstate.expect_and_forces()` 方法是 NetKet 库中 `MCState` 类的核心方法，用于计算量子期望值和对应的力向量（force vector）。

### 主要定义位置

| 层级 | 文件路径 | 行号 |
|------|----------|------|
| **MCState 类方法** | `netket/vqs/mc/mc_state/state.py` | 第 632 行 |
| **基类抽象方法** | `netket/vqs/base.py` | 第 214 行 |
| **核心实现函数** | `netket/vqs/mc/mc_state/expect_forces.py` | 第 41 行 |
| **分块处理实现** | `netket/vqs/mc/mc_state/expect_forces_chunked.py` | 第 69 行 |

**NetKet 安装路径：**
```
/Users/yangjianfei/.local/lib/python3.9/site-packages/netket/
```

---

## 2. 调用链路

```
vstate.expect_and_forces(O)
    │
    ▼
MCState.expect_and_forces()  [state.py:632]
    │
    ▼
expect_and_forces(vstate, O, chunk_size, mutable)  [expect_forces.py:41]
    │
    ▼
forces_expect_hermitian()  [expect_forces.py:69]  ← 核心计算函数
```

---

## 3. 函数签名与参数

### MCState 类方法 (state.py:632-671)

```python
@timing.timed
def expect_and_forces(
    self,
    O: AbstractOperator,
    *,
    mutable: Optional[CollectionFilter] = None,
) -> tuple[Stats, PyTree]:
```

**参数说明：**
- `O`: 要计算期望值的算符 (AbstractOperator)
- `mutable`: 指定模型状态中哪些集合是可变的（用于 BatchNorm 等场景）

**返回值：**
- `Stats`: 量子期望值 `<O>` 的统计估计
- `PyTree`: 力向量 `F_j = Cov[∂_j log ψ, O_loc]`

---

## 4. 核心实现逻辑

### 4.1 入口函数 (expect_forces.py:41-65)

```python
@dispatch
def expect_and_forces(
    vstate: MCState,
    Ô: AbstractOperator,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:
    # 1. 获取局部核函数参数
    σ, args = get_local_kernel_arguments(vstate, Ô)
    
    # 2. 获取局部估计器函数
    local_estimator_fun = get_local_kernel(vstate, Ô)
    
    # 3. 调用核心计算函数
    Ō, Ō_grad, new_model_state = forces_expect_hermitian(
        local_estimator_fun,
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )
    
    # 4. 更新模型状态（如果需要）
    if mutable is not False:
        vstate.model_state = new_model_state
    
    return Ō, Ō_grad
```

### 4.2 核心计算函数 (expect_forces.py:68-115)

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

    # 1. 计算局部估计量 O_loc
    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    # 2. 计算期望值的统计量
    Ō = statistics(O_loc.reshape((n_chains, -1)))

    # 3. 中心化
    O_loc -= Ō.mean

    # 4. 使用 VJP (Vector-Jacobian Product) 计算力向量
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    # 5. MPI 求和（分布式计算支持）
    Ō_grad, _ = mpi.mpi_sum_jax(Ō_grad)

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, Ō_grad, new_model_state
```

---

## 5. 数学原理

### 5.1 力向量定义

力向量 `F_j` 定义为波函数对数导数与算符局部估计量的协方差：

$$F_j = \text{Cov}[\partial_j \log\psi, O_{\text{loc}}]$$

### 5.2 与梯度的关系

- **对于复全纯态 (complex holomorphic states)**：
  $$\frac{\partial\langle O\rangle}{\partial(\theta_j)^\star} = F_j$$

- **对于实参数态 (real-parameter states)**：
  $$\frac{\partial\langle O\rangle}{\partial\theta_j} = 2\text{Re}[F_j]$$

### 5.3 局部估计量

局部估计量定义为：
$$O_{\text{loc}}(\sigma) = \frac{\langle\sigma|O|\psi\rangle}{\langle\sigma|\psi\rangle}$$

---

## 6. 分块处理版本

当 `chunk_size` 被指定时，会调用分块处理版本 (expect_forces_chunked.py:69-94)，用于处理大规模样本时的内存优化。

### 分块处理核心函数

```python
@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def forces_expect_hermitian_chunked(
    chunk_size: int,
    local_value_kernel_chunked: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree]:
    # 使用分块 VJP 计算梯度
    vjp_fun_chunked = nkjax.vjp_chunked(
        lambda w, σ: model_apply_fun({"params": w, **model_state}, σ),
        parameters,
        σ,
        conjugate=True,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )
    # ...
```

---

## 7. 继承体系

```
VariationalState (base.py)
    │
    ├── expect_and_forces() [抽象方法, L214]
    │
    └── MCState (mc/mc_state/state.py)
            │
            └── expect_and_forces() [具体实现, L632]
                    │
                    ├── expect_forces.py [无分块版本]
                    └── expect_forces_chunked.py [分块版本]
```

---

## 8. 相关文件汇总

| 文件 | 作用 |
|------|------|
| `netket/vqs/base.py` | 定义 `VariationalState` 基类和抽象方法 |
| `netket/vqs/mc/mc_state/state.py` | `MCState` 类定义，包含 `expect_and_forces` 方法入口 |
| `netket/vqs/mc/mc_state/expect_forces.py` | 核心计算实现（无分块） |
| `netket/vqs/mc/mc_state/expect_forces_chunked.py` | 分块处理实现 |
| `netket/vqs/full_summ/expect.py` | `FullSumState` 的实现版本 |

---

## 9. 使用示例

```python
import netket as nk

# 创建 MCState
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)

# 计算期望值和力向量
expectation, forces = vstate.expect_and_forces(hamiltonian)

# expectation: Stats 对象，包含期望值的均值、方差等统计信息
# forces: PyTree 结构，包含力向量（梯度）
```

---

## 10. 关键技术点

1. **JAX JIT 编译**: 使用 `@partial(jax.jit, ...)` 进行即时编译优化
2. **VJP (Vector-Jacobian Product)**: 高效计算梯度
3. **MPI 支持**: 支持分布式计算
4. **分块处理**: 大规模样本时的内存优化
5. **多态分发**: 使用 `@dispatch` 装饰器支持不同参数类型的处理

---

## 11. 与 `expect_and_grad` 的区别

| 方法 | 用途 | 适用场景 |
|------|------|----------|
| `expect_and_forces` | 计算力向量（协方差形式） | 厄米算符，用于 VMC 优化 |
| `expect_and_grad` | 计算标准梯度 | 一般优化场景 |

力向量是 VMC (Variational Monte Carlo) 方法中的核心概念，直接对应于量子期望值对参数的梯度。
