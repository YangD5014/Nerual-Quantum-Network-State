# 如何查找 `vstate.expect_and_grad(ha)` 的定义位置

## 问题背景

在代码中看到：
```python
vstate.expect_and_grad(ha)
```

如何找到这个 `expect_and_grad` 方法到底定义在哪里？

## 完整的查找路径

### 第一步：确定 vstate 的类型

从代码中可以看到：
```python
vstate = nk.vqs.MCState(sampler, model, n_samples=1008)
```

所以 `vstate` 的类型是 `nk.vqs.MCState`。

### 第二步：查找 MCState 类中的定义

`MCState` 类定义在：
```
/opt/miniconda3/envs/Neural/lib/python3.11/site-packages/netket/vqs/mc/mc_state/state.py
```

在这个文件中，第 **661-699 行** 找到了 `expect_and_grad` 方法：

```python
@timing.timed
def expect_and_grad(
    self,
    O: AbstractOperator,
    *,
    mutable: CollectionFilter | None = None,
    **kwargs,
) -> tuple[Stats, PyTree]:
    # ... 文档字符串 ...
    if mutable is None:
        mutable = self.mutable

    return expect_and_grad(  # 注意：这里调用的是模块级别的函数！
        self,
        O,
        self.chunk_size,
        mutable=mutable,
        **kwargs,
    )
```

**关键点**：`MCState.expect_and_grad()` 方法本身不实现具体逻辑，而是调用模块级别的 `expect_and_grad` 函数！

### 第三步：查找模块级别的 expect_and_grad 函数

在 `netket/vqs/base.py` 第 **407 行** 找到了通用定义：

```python
def expect_and_grad(
    vstate: VariationalState,
    O: AbstractOperator,
    *args,
    mutable: CollectionFilter = False,
    **kwargs,
) -> tuple[Stats, PyTree]:
    """
    计算期望值和梯度的通用函数
    """
```

这个函数使用了 **Plum 库的 @dispatch 装饰器**（类似单分派/多分派的泛型函数）。

### 第四步：查找具体的分发实现

因为使用了 `@dispatch` 装饰器，实际调用哪个实现取决于参数类型。

对于 `MCState` + `AbstractOperator`，实际调用的是：

**文件**：`/opt/miniconda3/envs/Neural/lib/python3.11/site-packages/netket/vqs/mc/mc_state/expect_grad.py`

**第 42-66 行**：
```python
@expect_and_grad.dispatch
def expect_and_grad_default_formula(
    vstate: MCState,
    Ô: AbstractOperator,
    chunk_size: int | None,
    *args,
    mutable: CollectionFilter = False,
    use_covariance: bool | None = None,
) -> tuple[Stats, PyTree]:
    
    if use_covariance is None:
        use_covariance = Ô.is_hermitian

    if use_covariance:
        # 使用协方差公式
        Ō, Ō_grad = expect_and_forces(vstate, Ô, chunk_size, *args, mutable=mutable)
        Ō_grad = force_to_grad(Ō_grad, vstate.parameters)
        return Ō, Ō_grad
    else:
        # 使用非厄米公式
        return expect_and_grad_nonhermitian(
            vstate, Ô, chunk_size, *args, mutable=mutable
        )
```

### 第五步：查看运行时输出

代码中添加了调试输出，运行时可以看到：
```
调用了./vqs/nc_state/state.py 下的 expect_and_grad 函数
调用了./vqs/mc_state/expect_grad.py 下的 expect_and_grad_default_formula 函数
```

这验证了我们的查找路径！

## 完整的调用链

```
vstate.expect_and_grad(ha)
    ↓ (MCState 类的方法)
netket/vqs/mc/mc_state/state.py:661 - MCState.expect_and_grad()
    ↓ (调用模块级函数)
netket/vqs/base.py:407 - expect_and_grad() [通用分发函数]
    ↓ (根据参数类型分发)
netket/vqs/mc/mc_state/expect_grad.py:42 - expect_and_grad_default_formula()
    ↓ (如果 use_covariance=True)
netket/vqs/mc/mc_state/expect_grad.py:60 - expect_and_forces()
    ↓
force_to_grad()
```

## 查找技巧总结

### 1. **使用 Grep 搜索函数定义**
```bash
grep -r "def expect_and_grad" /path/to/netket/vqs/
```

### 2. **使用 Python 的 inspect 模块**
```python
import inspect
import netket as nk

vstate = nk.vqs.MCState(...)
print(inspect.getfile(vstate.expect_and_grad))  # 查看定义文件
print(inspect.signature(vstate.expect_and_grad))  # 查看函数签名
```

### 3. **使用 IDE 的"跳转到定义"功能**
- VS Code: F12 或 Ctrl+ 点击
- PyCharm: Ctrl+B 或 Ctrl+ 点击

### 4. **查看继承关系**
```python
print(type(vstate))              # 查看具体类型
print(type(vstate).__mro__)      # 查看方法解析顺序
```

### 5. **添加调试输出**
在关键位置添加 `print()` 语句，如本例所示。

## 关键文件位置

| 文件 | 作用 | 行号 |
|------|------|------|
| `netket/vqs/base.py` | 基类 VariationalState 和通用分发函数 | 215, 407 |
| `netket/vqs/mc/mc_state/state.py` | MCState 类及其 expect_and_grad 方法 | 661-699 |
| `netket/vqs/mc/mc_state/expect_grad.py` | 具体实现（默认公式） | 42-66 |
| `netket/vqs/mc/mc_state/expect_grad_chunked.py` | 分块版本实现 | 39+ |

## 为什么这么复杂？

NetKet 使用了**多分派（multiple dispatch）**模式：

1. **基类定义接口**：`VariationalState.expect_and_grad()` 提供统一接口
2. **子类重写**：`MCState.expect_and_grad()` 添加特定逻辑（如 chunk_size）
3. **分发函数**：根据参数类型自动选择具体实现
4. **多种实现**：
   - 厄米算符 vs 非厄米算符
   - 分块 vs 不分块
   - 不同状态类型（MCState, MCMixedState, FullSummState）

这种设计使得代码非常灵活，可以支持多种组合，但也增加了追踪难度。

## 快速定位流程图

```
看到：vstate.expect_and_grad(ha)
         ↓
1. 确定 vstate 类型 → nk.vqs.MCState
         ↓
2. 搜索 MCState.expect_and_grad 
   → netket/vqs/mc/mc_state/state.py:661
         ↓
3. 查看它调用了什么 
   → expect_and_grad(self, O, self.chunk_size, ...)
         ↓
4. 搜索这个 expect_and_grad 
   → netket/vqs/base.py:407 (带@dispatch)
         ↓
5. 查找具体分发实现
   → netket/vqs/mc/mc_state/expect_grad.py:42
         ↓
找到最终实现！
```

## 练习

尝试用同样的方法查找：
1. `vstate.expect(ha)` 的定义
2. `vstate.grad(ha)` 的定义
3. `gs.run()` 的定义（VMC driver）

答案提示：
- `expect`: `netket/vqs/mc/mc_state/state.py` → `netket/vqs/base.py` → `netket/vqs/mc/mc_state/expect.py`
- `grad`: 类似
- `run`: `netket/driver/abstract_variational_driver.py`
