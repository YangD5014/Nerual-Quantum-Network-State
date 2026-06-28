# 如何查找 `nk.vqs.get_local_kernel_arguments(vstate, ha)` 的定义位置

## 问题背景

在代码中看到：
```python
nk.vqs.get_local_kernel_arguments(vstate, ha)
```

如何找到这个函数到底定义在哪里？实际调用的是哪个实现？

## 快速答案

**抽象定义**：`netket/vqs/mc/common.py` 第 32 行  
**具体实现**：`netket/vqs/mc/mc_state/expect.py` 第 58 行（针对 DiscreteOperator）  
**调用机制**：使用 `@dispatch` 装饰器的多分派系统

## 完整的查找路径

### 第一步：确定函数的导出位置

`nk.vqs.get_local_kernel_arguments` 从 `netket.vqs` 模块导出。

查看 `netket/vqs/__init__.py` 第 23 行：
```python
from .mc import MCState, MCMixedState, get_local_kernel_arguments, get_local_kernel
```

发现它是从 `.mc` 模块导入的。

### 第二步：追踪到 mc 模块

查看 `netket/vqs/mc/__init__.py` 第 28 行：
```python
from .common import check_hilbert, get_local_kernel_arguments, get_local_kernel
```

发现它是从 `.common` 模块导入的！

### 第三步：找到抽象定义

查看 `netket/vqs/mc/common.py` 第 32 行：

```python
@dispatch.abstract
def get_local_kernel_arguments(vstate: Any, Ô: Any):
    """
    返回用于计算算符 O 期望值的样本，以及连接元素和矩阵元素。
    
    参数：
        vstate: 变分态
        Ô: 算符
    
    返回：
        包含 2 个元素的元组 (sigma, args)
        - sigma: 用于计算经典期望值的样本
        - args: 可以传递给 local_kernel 的任何内容
    """
```

**关键点**：这是一个**抽象分派函数**（`@dispatch.abstract`），本身没有具体实现！

### 第四步：查找具体实现

使用 `@dispatch` 装饰器的函数会根据参数类型自动选择实现。

搜索所有具体实现：
```bash
grep -r "@dispatch" netket/vqs/mc/mc_state/expect.py | grep "get_local_kernel_arguments"
```

找到 **4 个具体实现**在 `netket/vqs/mc/mc_state/expect.py`：

#### 实现 1：Squared 算符（第 44 行）
```python
@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: Squared):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, (σp, mels)
```

#### 实现 2：DiscreteOperator（第 58 行）⭐ **最常用**
```python
@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: DiscreteOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, (σp, mels)
```

#### 实现 3：DiscreteJaxOperator（第 72 行）
```python
@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: DiscreteJaxOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    return σ, Ô
```

#### 实现 4：ContinuousOperator（第 85 行）
```python
@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: ContinuousOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)
    σ = vstate.samples
    args = Ô._pack_arguments()
    return σ, args
```

### 第五步：确定实际调用哪个实现

在你的代码中：
```python
ha = nkx.operator.from_pyscf_molecule(mol)
```

`ha` 是从 PySCF 分子创建的算符，类型是 **DiscreteOperator**（离散算符）。

因此实际调用的是：
**`netket/vqs/mc/mc_state/expect.py` 第 58 行的 `DiscreteOperator` 实现**

### 第六步：理解返回值

对于 DiscreteOperator，函数返回：
```python
σ = vstate.samples                    # 当前的样本配置
σp, mels = Ô.get_conn_padded(σ)      # 连接的状态和矩阵元
return σ, (σp, mels)
```

- `σ`: 形状为 `(n_chains, n_samples_per_chain, n_sites)` 的样本
- `σp`: 每个样本的所有连接状态，形状 `(n_samples, n_conn_max, n_sites)`
- `mels`: 对应的矩阵元，形状 `(n_samples, n_conn_max)`

这些返回值会被传递给 `get_local_kernel` 返回的核函数来计算局部期望值。

## 完整的调用链

```
nk.vqs.get_local_kernel_arguments(vstate, ha)
    ↓ (从 netket.vqs 导出)
netket/vqs/mc/common.py:32 - 抽象定义
    ↓ (根据参数类型分发)
netket/vqs/mc/mc_state/expect.py:58 - DiscreteOperator 实现
    ↓
Ô.get_conn_padded(σ)  # 获取连接状态和矩阵元
    ↓
return σ, (σp, mels)
```

## 与其他函数的关系

`get_local_kernel_arguments` 通常和 `get_local_kernel` 一起使用：

```python
# 在 netket/vqs/mc/mc_state/expect.py:108
σ, args = get_local_kernel_arguments(vstate, Ô)
local_estimator_fun = get_local_kernel(vstate, Ô)

# 然后使用这两个来计算期望值
Ō = local_estimator_fun(
    apply_fun, parameters, model_state, 
    σ, args, machine_pow
)
```

## 查找技巧总结

### 1. **查看模块的 __init__.py**
追踪导出路径：
```python
# netket/vqs/__init__.py
from .mc import get_local_kernel_arguments

# netket/vqs/mc/__init__.py  
from .common import get_local_kernel_arguments
```

### 2. **搜索 @dispatch 装饰器**
```bash
grep -rn "@dispatch" netket/vqs/mc/mc_state/expect.py | grep get_local_kernel_arguments
```

### 3. **查看类型注解**
抽象定义中的类型提示告诉你需要找哪些具体实现：
```python
@dispatch.abstract
def get_local_kernel_arguments(vstate: Any, Ô: Any):
#                                      ↑ 需要根据实际类型查找
```

### 4. **使用 Python 运行时信息**
```python
import netket as nk

vstate = nk.vqs.MCState(...)
ha = nkx.operator.from_pyscf_molecule(mol)

print(type(vstate))  # <class 'netket.vqs.mc.mc_state.state.MCState'>
print(type(ha))      # 确定算符类型
```

### 5. **添加调试输出**
在 `common.py` 添加：
```python
@dispatch.abstract
def get_local_kernel_arguments(vstate: Any, Ô: Any):
    print(f"get_local_kernel_arguments 被调用")
    print(f"vstate 类型：{type(vstate)}")
    print(f"Ô 类型：{type(Ô)}")
```

### 6. **查看 Plum 的 dispatch 信息**
```python
from netket.vqs.mc import get_local_kernel_arguments
print(get_local_kernel_arguments)  # 显示所有注册的分发
```

## 关键文件位置

| 文件 | 作用 | 行号 |
|------|------|------|
| `netket/vqs/__init__.py` | 从 mc 模块导出 | 23 |
| `netket/vqs/mc/__init__.py` | 从 common 模块导出 | 28 |
| `netket/vqs/mc/common.py` | **抽象定义** | 32 |
| `netket/vqs/mc/mc_state/expect.py` | **MCState + DiscreteOperator 实现** | 58 |
| `netket/vqs/mc/mc_state/expect.py` | MCState + Squared 实现 | 44 |
| `netket/vqs/mc/mc_state/expect.py` | MCState + DiscreteJaxOperator 实现 | 72 |
| `netket/vqs/mc/mc_state/expect.py` | MCState + ContinuousOperator 实现 | 85 |
| `netket/vqs/mc/mc_mixed_state/expect.py` | MCMixedState 的实现 | 35+ |

## 为什么使用 @dispatch？

NetKet 使用 Plum 库的 `@dispatch` 实现**多分派（multiple dispatch）**：

1. **根据参数类型自动选择实现**
   - 不需要手动写 `if isinstance(Ô, DiscreteOperator): ...`
   - 代码更清晰、更易扩展

2. **支持多种组合**
   - 不同的状态类型（MCState, MCMixedState）
   - 不同的算符类型（DiscreteOperator, ContinuousOperator, Squared, etc.）
   - 不同的实现方式（普通、分块、JAX 优化）

3. **用户可扩展**
   - 用户可以为自己的算符类型添加新的分发规则
   - 无需修改 NetKet 核心代码

## 实际例子

### 例 1：离散算符（你的情况）
```python
from pyscf import gto, scf
import netket.experimental as nkx

mol = gto.M(atom='H 0 0 0; H 0 0 1.4', basis='STO-3G')
ha = nkx.operator.from_pyscf_molecule(mol)  # DiscreteOperator

# 调用路径：
# get_local_kernel_arguments(vstate, ha)
#   → mc_state/expect.py:58 (DiscreteOperator 版本)
#   → ha.get_conn_padded(σ)
```

### 例 2：连续算符
```python
import netket as nk

# 连续系统（如谐振子）
hilbert = nk.hilbert.Particle(1)
H = nk.operator.Kinetic(hilbert) + nk.operator.Potential(hilbert, lambda x: x**2)

# 调用路径：
# get_local_kernel_arguments(vstate, H)
#   → mc_state/expect.py:85 (ContinuousOperator 版本)
#   → H._pack_arguments()
```

### 例 3：平方算符
```python
H_squared = nk.operator.Squared(ha)

# 调用路径：
# get_local_kernel_arguments(vstate, H_squared)
#   → mc_state/expect.py:44 (Squared 版本)
#   → ha.parent.get_conn_padded(σ)
```

## 快速定位流程图

```
看到：nk.vqs.get_local_kernel_arguments(vstate, ha)
         ↓
1. 查看导出路径
   nk.vqs.__init__.py → nk.vqs.mc.__init__.py → mc/common.py
         ↓
2. 找到抽象定义
   netket/vqs/mc/common.py:32 (@dispatch.abstract)
         ↓
3. 确定参数类型
   vstate: MCState
   ha: DiscreteOperator (从 PySCF 创建)
         ↓
4. 搜索具体实现
   grep -r "@dispatch" netket/vqs/mc/mc_state/expect.py
         ↓
5. 匹配类型
   MCState + DiscreteOperator → expect.py:58
         ↓
找到最终实现！
```

## 与 expect_and_grad 的关系

`get_local_kernel_arguments` 是 `expect_and_grad` 内部调用的底层函数：

```
expect_and_grad(vstate, ha)
    ↓
expect_and_grad_default_formula()
    ↓
expect_and_forces(vstate, ha)
    ↓
get_local_kernel_arguments(vstate, ha)  ← 这里！
get_local_kernel(vstate, ha)
    ↓
计算局部期望值和梯度
```

## 练习

尝试用同样的方法查找：

1. `nk.vqs.get_local_kernel(vstate, ha)` 的定义
2. `nk.vqs.expect(vstate, ha)` 的定义
3. `nk.vqs.MCState.samples` 的定义

答案提示：
- `get_local_kernel`: `common.py:50` (抽象) → `mc_state/expect.py:68` (DiscreteOperator)
- `expect`: `base.py:165` → `mc_state/expect.py:105`
- `samples`: `mc_state/state.py:449` (property)

## 总结

查找 `@dispatch` 装饰的函数的步骤：

1. **找抽象定义**：通常在 `common.py` 或 `base.py`
2. **确定参数类型**：vstate 和 Ô 的具体类型
3. **搜索具体实现**：查找所有 `@dispatch` 装饰的同名函数
4. **匹配类型**：找到与你的参数类型匹配的实现
5. **验证**：运行时输出或调试确认

掌握这个方法，你就可以追踪 NetKet 中任何使用多分派的函数！
