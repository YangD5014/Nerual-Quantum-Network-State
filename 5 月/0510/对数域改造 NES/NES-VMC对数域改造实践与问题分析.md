# NES-VMC 对数域数值稳定性改造 实践与问题分析

## 1. 研究背景

**目标**：在标准 NES-VMC 算法中，当神经网络训练到后期时，波函数矩阵 $M$ 的元素值可能变得非常大或非常小。这会导致 `jnp.exp(L)` 数值溢出，最终训练崩溃（NaN 或 -inf）。

**解决方案**：将对数域数值稳定化技术引入 NES-VMC，通过减去每行的最大值来避免指数运算的溢出。

**实践结果**：改造**未能成功**，训练在第 80 步左右崩溃，出现了 `-inf` 和 `nan`。本文档记录改造的实现过程、代码分析、以及问题诊断。

---

## 2. 改造实现详解

### 2.1 NESTotalAnsatz_stable：Ansatz 的改造

改造后的 `NESTotalAnsatz_stable` 类在标准 `NESTotalAnsatz` 基础上，额外返回 `L_max`：

```python
class NESTotalAnsatz_stable(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)
            ansatz = SingleStateAnsatz(n_spin_orbitals, hidden_dim, rngs=sub_rngs)
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            # 核心改造：记录 L.max() 用于后续稳定化
            L_stable = L - L.max()
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            # 返回三个值（改造点）
            return log_Psi_stable, L_stable, L.max()

        # 批量处理支持
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin*self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'不支持的输入形状: {x.shape}')
```

**改造点**：

1. `L_stable = L - L.max()`：稳定化后的矩阵
2. `return log_Psi_stable, L_stable, L.max()`：**返回三个值**（原版本只返回两个）

### 2.2 L 矩阵稳定化的数学含义

原始 L 矩阵：
$$
L_{ij} = \ln \psi_j(x^i)
$$

稳定化：
$$
L_{\text{stable}} = L - L_{\max}, \quad L_{\max} = \max_{i,j} L_{ij}
$$

原始 $M$ 矩阵：
$$
M = \exp(L)
$$

稳定化后的 $M_{\text{stable}}$ 矩阵：
$$
M_{\text{stable}} = \exp(L_{\text{stable}}) = \exp(L - L_{\max}) = \exp(L) \cdot e^{-L_{\max}I} = M \cdot e^{-L_{\max}I}
$$

关键性质：
- $M$ 和 $M_{\text{stable}}$ 只差一个全局因子 $e^{-L_{\max}}$
- 在后续计算 $M^{-1} \hat{H}\Psi$ 时，这个因子会抵消

---

### 2.3 create_machine_matrix_max：提取 L.max

```python
def create_machine_matrix_max(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix= m(sigma)
        max = log_M_matrix.max(axis=(1,2))
        return max

    return machine, graphdef, state
```

**问题**：这里 `m(sigma)` 返回三个值 `log_Psi_stable, L_stable, L_max`，但代码只解包了两个。这会导致 `ValueError: too many values to unpack`。

---

### 2.4 NES_Loss_Stable：稳定化的损失函数

这是改造的核心函数，notebook 中的实现在 `NES_VMC.py` 中：

```python
def NES_Loss_Stable(ha, total_matrix_machine, total_max_machine,
                    single_machine_list, total_params, x):
    """
    对数域稳定化损失函数

    核心思想：
    1. M_stable = exp(L_stable) = exp(L - L_max) - 数值稳定
    2. HΨ_stable = exp(log(HΨ) - L_max) - 与 M_stable 对齐尺度
    3. E_L = M_stable⁻¹ · HΨ_stable = M⁻¹ · HΨ - 数学上等价
    """
    def log_Psi_Matrix_stable(total_matrix_machine, total_params, x):
        L_stable = total_matrix_machine(total_params, x)  # L_stable = L - L_max
        return L_stable

    def log_HPsi_stable(ha, max_machine, single_machine_list, total_params, x):
        HamPsi = Ham_Psi(ha=ha, single_machine_list=single_machine_list,
                         total_params=total_params, x=x)
        log_psi_max = max_machine(total_params, x)  # L_max per sample
        log_psi_max = log_psi_max.reshape(-1, 1, 1)
        log_HamPsi = jnp.log(HamPsi)  # 先取对数
        log_HamPsi_stable = log_HamPsi - log_psi_max  # 稳定化
        return log_HamPsi_stable

    # 计算稳定化后的矩阵
    Psi_stable = jnp.exp(log_Psi_Matrix_stable(total_matrix_machine, total_params, x))
    HPsi_stable = jnp.exp(log_HPsi_stable(ha, total_max_machine,
                                           single_machine_list, total_params, x))

    # 矩阵求逆
    Psi_stable_inv = jnp.linalg.solve(Psi_stable, HPsi_stable)

    # 损失 = 迹
    return jnp.real(jnp.trace(Psi_stable_inv, axis1=-2, axis2=-1)), Psi_stable_inv
```

**数学推导**：

原始损失函数（非稳定化）：
$$
\mathcal{L} = \mathrm{Tr}(M^{-1} \hat{H}\Psi)
$$

其中 $M = \exp(L)$，$M_{\text{stable}} = \exp(L - L_{\max}) = M \cdot e^{-L_{\max}I}$

稳定化后的损失函数：
$$
\mathcal{L}_{\text{stable}} = \mathrm{Tr}(M_{\text{stable}}^{-1} \cdot \hat{H}\Psi_{\text{stable}})
$$

其中：
- $M_{\text{stable}} = M \cdot e^{-L_{\max}I}$
- $\hat{H}\Psi_{\text{stable}} = \hat{H}\Psi \cdot e^{-L_{\max}I}$（通过 $\log(\hat{H}\Psi) - L_{\max}$ 实现）

因此：
$$
\mathcal{L}_{\text{stable}} = \mathrm{Tr}((e^{-L_{\max}}M)^{-1} \cdot (e^{-L_{\max}}\hat{H}\Psi))
= \mathrm{Tr}(M^{-1} \hat{H}\Psi) = \mathcal{L}
$$

数学上完全等价。

---

## 3. 崩溃过程分析

### 3.1 训练日志

训练在第 80 步左右出现数值崩溃：

```
Step  75 | Loss: -1.8607750450291942|0st能量=-1.02482400 Ha｜1st能量=-0.83595104 Ha｜2st能量=-0.83595104 Ha
#-----------------------------------------#
log_Psi: mean=-inf+nanj | min=-inf+0.000j | max=-inf+0.000j
grad norm = 24626.5783
Step  80 | Loss: 92.64876458641389|0st能量=-70470745347005565... Ha｜...
#-----------------------------------------#
log_Psi: mean=nan+nanj | min=nan+nanj | max=nan+nanj
grad norm = nan
Step  85 | Loss: nan|0st能量=nan Ha｜1st能量=nan Ha｜2st能量=nan Ha
```

### 3.2 崩溃节点定位

从 traceback 分析：

```
Cell In[77], line 25, in NES_Loss_Stable
    HPsi_stable = jnp.exp(log_HPsi_stable(...))

File ".../NES_VMC.py", line 255, in Ham_Psi
    return jax.vmap(_single_HamPsi)(x)
```

崩溃发生在 `Ham_Psi` 内部的 `jax.vmap` 中，具体是在 `_single_HamPsi` 函数的 `HPsi.at[i, j].set(val)` 行。

### 3.3 崩溃原因分析

**可能原因 1：Ham_Psi 中的数值下溢**

`Ham_Psi` 函数计算 $\hat{H}\Psi(\mathbf{x})$，其中包含：

```python
def _single_hpsi(x_single):
    x_primes, mels = ha.get_conn_padded(x_single)
    log_psi_vals = single_machine(params, x_primes)
    psi_vals = jnp.exp(log_psi_vals)  # 这里可能下溢！
    return jnp.sum(mels * psi_vals)
```

当 `single_machine` 返回的 `log_psi_vals` 是一个很大的负数时，`jnp.exp(log_psi_vals)` 会下溢到 0。然后后续的 `jnp.log(0) = -inf`。

**可能原因 2：create_machine_matrix_max 的解包错误**

`NESTotalAnsatz_stable` 返回三个值，但 `create_machine_matrix_max` 只解包两个：

```python
log_psi_total,log_M_matrix= m(sigma)  # 只解包2个，但 m(sigma) 返回3个值！
max = log_M_matrix.max(axis=(1,2))
```

这会导致 `ValueError: too many values to unpack`。但奇怪的是训练进行了 80 步才崩溃...

**可能原因 3：变量名不一致**

在 `NES_VMC_Stable.py` 中：
```python
total_matirx_max,_,_ = create_machine_matrix_max(total_ansatz)  # 变量名有拼写错误
```

但在训练循环中使用：
```python
grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                             total_machine_max=total_max,  # 但这里用的是 total_max！
                                             ...)
```

`total_max` 未定义！这会导致 `NameError`。

---

## 4. 代码问题汇总

### 4.1 Bug 1：create_machine_matrix_max 的返回值解包错误

**位置**：`NES_VMC_Stable.py` 第 205-217 行

**问题**：`NESTotalAnsatz_stable.__call__` 返回三个值 `(log_Psi_stable, L_stable, L_max)`，但 `create_machine_matrix_max` 只解包两个：

```python
def create_machine_matrix_max(model: NESTotalAnsatz):
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix= m(sigma)  # ❌ ValueError: too many values to unpack
        max = log_M_matrix.max(axis=(1,2))
        return max
```

**应该改为**：
```python
def create_machine_matrix_max(model: NESTotalAnsatz):
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, L_stable, L_max = m(sigma)  # ✅ 正确解包3个值
        return L_max
```

### 4.2 Bug 2：create_machine_matrix 的返回值解包错误

**位置**：`NES_VMC_Stable.py` 第 192-203 行

**问题**：同样的问题，`create_machine_matrix` 也只解包两个值：

```python
def create_machine_matrix(model: NESTotalAnsatz):
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total,log_M_matrix= m(sigma)  # ❌ ValueError
        return log_M_matrix
```

### 4.3 Bug 3：变量名不一致

**位置**：`NES_VMC_Stable.py` 第 487 行 vs 训练循环

```python
# 第 487 行：变量名有拼写错误
total_matirx_max,_,_ = create_machine_matrix_max(total_ansatz)  # 拼写错误：matirx → matrix

# 训练循环中：使用了不同的变量名
grad, loss_mean, E_L_mean = nes_vmc_gradient(...,
    total_machine_max=total_max,  # ❌ NameError: total_max 未定义
    ...)
```

### 4.4 Bug 4：Ham_Psi 在 vmap 中的索引问题

**位置**：`NES_VMC_Stable.py` 第 290-304 行

```python
elif x.ndim == 3:
    def _single_HamPsi(x_single):
        HPsi = jnp.zeros((K, K), dtype=complex)
        for i in range(K):
            xi = x_single[i]
            for j in range(K):
                machine_j = single_machine_list[j]
                params_j = total_params['single_ansatz_list'][j]
                val = Ham_psi(ha, machine_j, params_j, xi)
                HPsi = HPsi.at[i, j].set(val)  # ❌ JAX vmap 中可能出问题
        return HPsi
    return jax.vmap(_single_HamPsi)(x)
```

在 JAX 的 `vmap` 内部使用 `.at[i, j].set()` 进行原地索引赋值，在某些情况下会导致 shape 推断问题。

---

## 5. 对数域改造的理论正确性

尽管代码存在 bug，**对数域改造的理论是正确的**。

### 5.1 改造的数学推导

假设原始物理量：

- 波函数矩阵：$M = \exp(L)$
- 哈密顿量作用：$\hat{H}\Psi$

原始局域能量矩阵：
$$
E_L = M^{-1} \cdot \hat{H}\Psi
$$

改造后的局域能量矩阵：
$$
E_{L,\text{stable}} = M_{\text{stable}}^{-1} \cdot \hat{H}\Psi_{\text{stable}}
$$

其中：
- $M_{\text{stable}} = \exp(L - L_{\max}) = \exp(L) \cdot e^{-L_{\max}I} = M \cdot e^{-L_{\max}I}$
- $\hat{H}\Psi_{\text{stable}} = \exp(\log(\hat{H}\Psi) - L_{\max}) = \hat{H}\Psi \cdot e^{-L_{\max}I}$

代入：
$$
E_{L,\text{stable}} = (M \cdot e^{-L_{\max}I})^{-1} \cdot (\hat{H}\Psi \cdot e^{-L_{\max}I})
= (e^{-L_{\max}I})^{-1} \cdot M^{-1} \cdot \hat{H}\Psi \cdot e^{-L_{\max}I}
= M^{-1} \cdot \hat{H}\Psi
= E_L
$$

**结论**：改造后的损失函数在数学上与原始损失函数完全等价。

### 5.2 数值稳定性的改善

原始版本：
- 直接计算 $\exp(L)$，当 $L$ 很大时 → `inf`
- 直接计算 $\hat{H}\Psi$，数值范围不受控制

稳定化版本：
- $\exp(L - L_{\max})$ 的元素范围在 $[0, 1]$ 之间（因为最大值元素变成了 0）
- $\exp(\log(\hat{H}\Psi) - L_{\max})$ 与 $M_{\text{stable}}$ 对齐尺度

---

## 6. 修复建议

### 6.1 修复 create_machine_matrix_max

```python
def create_machine_matrix_max(model: NESTotalAnsatz):
    """返回 L_max（用于损失函数稳定化）"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_stable, L_stable, L_max = m(sigma)  # 正确解包3个值
        return L_max

    return machine, graphdef, state
```

### 6.2 修复 create_machine_matrix

```python
def create_machine_matrix(model: NESTotalAnsatz):
    """返回 L_stable（用于损失函数计算）"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_stable, L_stable, L_max = m(sigma)  # 正确解包3个值
        return L_stable

    return machine, graphdef, state
```

### 6.3 修复变量名不一致

```python
# 错误的写法
total_matirx_max,_,_ = create_machine_matrix_max(total_ansatz)

# 正确的写法
total_matrix_max, _, _ = create_machine_matrix_max(total_ansatz)

# 并且在训练循环中使用相同的变量名
grad, loss_mean, E_L_mean = nes_vmc_gradient(...,
    total_machine_max=total_matrix_max,  # ✅
    ...)
```

### 6.4 修复 Ham_Psi 的 vmap 索引

```python
elif x.ndim == 3:
    def _single_HamPsi(x_single):
        # 使用 jnp.zeros 初始化并填充，而不是原地 set
        results = []
        for i in range(K):
            xi = x_single[i]
            row = []
            for j in range(K):
                machine_j = single_machine_list[j]
                params_j = total_params['single_ansatz_list'][j]
                val = Ham_psi(ha, machine_j, params_j, xi)
                row.append(val)
            results.append(row)
        HPsi = jnp.array(results, dtype=complex)
        return HPsi
    return jax.vmap(_single_HamPsi)(x)
```

---

## 7. 完整数据流图（改造后）

```
┌─────────────────────────────────────────────────────────────────┐
│                    输入：扩展组态 x (batch, K, n_spin)          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│             total_matrix_machine → L_stable                     │
│             total_max_machine → L_max                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  L_stable = L - L_max（逐样本）                                  │
│  M_stable = exp(L_stable) → 数值稳定                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Ham_Psi(...) → HamΨ（原始尺度）                                 │
│  log(HamΨ) → 取对数                                              │
│  HamΨ_stable = exp(log(HamΨ) - L_max) → 与 M_stable 对齐        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  E_L_stable = M_stable⁻¹ · HamΨ_stable                         │
│  Loss = Tr(E_L_stable)                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  理论上：E_L_stable = E_L（数学等价）                             │
│  但数值上更稳定                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8. 总结

| 方面 | 评价 |
|------|------|
| **改造思路** | ✅ 正确：对数域稳定化是解决指数溢出问题的标准方法 |
| **数学推导** | ✅ 正确：改造前后数学等价，损失函数不变 |
| **代码实现** | ❌ 存在多个 bug，导致无法正常运行 |
| **Bug 1** | `create_machine_matrix_max` 返回值解包错误（2 vs 3 个值） |
| **Bug 2** | `create_machine_matrix` 返回值解包错误（2 vs 3 个值） |
| **Bug 3** | 变量名不一致：`total_matirx_max` vs `total_max` |
| **Bug 4** | `Ham_Psi` 中 vmap 索引可能导致 shape 问题 |

**下一步**：修复上述 bug 后重新测试。

---

## 9. 参考文件

- 改造实现：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC_Stable.py`
- 测试 Notebook：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC对数域改造 K2.ipynb`
- 标准版本：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/NES_new_vserison/NES_VMC.py`
