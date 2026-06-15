# NES-VMC 对数域改造 代码解读

## 一、研究背景

在 NES-VMC（Natural Excited State Variational Monte Carlo）算法中，当神经网络训练到后期时，波函数矩阵 $\Psi(\mathbf{x})$ 的元素值可能变得非常大或非常小。这会导致在计算损失函数时出现**数值溢出**问题。

本文件对比分析**改造前**（原始版本）与**改造后**（对数域改造版本）的核心差异。

---

## 二、核心改造点对比

### 1. NESTotalAnsatz 的 model 输出

#### 改造前（原始版本）

```python
# 文件：NES_new_vserison/NES_VMC.py
# 行号：91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
    log_Psi = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi, L  # 直接返回 L，未稳定化
```

#### 改造后（对数域改造版本）

```python
# 文件：对数域改造 NES/NES_VMC.py
# 行号：91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    L_stable = L - L.max()  # 核心改造：减去每行的最大值
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
    log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi_stable, L_stable, L.max()  # 返回稳定化后的 L_stable 和 L.max()
```

#### 关键区别

| 项目 | 改造前 | 改造后 |
|------|--------|--------|
| `L` 矩阵处理 | 未稳定化 | 稳定化：`L_stable = L - L.max()` |
| 返回值 | `(log_Psi, L)` | `(log_Psi_stable, L_stable, L.max())` |
| 数值稳定性 | 可能有溢出风险 | 数值稳定 |

---

### 2. 损失函数计算

#### 改造前（原始版本）

```python
# 文件：NES_new_vserison/NES_VMC.py
# 行号：248-255

def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)  # 获取原始 L 矩阵
    Psi_Matrix = jnp.exp(log_M)  # 直接取指数，可能数值溢出
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### 改造后（对数域改造版本）

```python
# 文件：对数域改造 NES/NES_VMC.py
# 行号：85-97

def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    L_stable = total_matrix_machine(total_params, x)  # 获取稳定化后的 L_stable
    Psi_Matrix_stable = jnp.exp(L_stable)  # 指数运算稳定化

    M = jnp.log(Ham_Psi(ha, single_machine_list, total_params, x))  # 对 H_psi 取对数
    M_stable = M - total_max_machine(total_params, x).reshape(-1,1,1)  # 稳定化
    HPsi_stable = jnp.exp(M_stable)  # 指数运算稳定化

    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### 关键区别

| 步骤 | 改造前 | 改造后 |
|------|--------|--------|
| `Psi_Matrix` | `exp(L)` 直接计算 | `exp(L_stable)` 稳定化后计算 |
| `H_psi_x` | 原始值直接使用 | 先取 `log(H_psi)` 再减 `L.max()` |
| `HPsi` | 直接 `H_psi_x` | `exp(M_stable)` 稳定化后计算 |

---

### 3. 新增 create_machine_max 函数

#### 改造前

仅有两个函数：
- `create_machine`：返回 `log_psi_total`
- `create_machine_matrix`：返回 `log_M_matrix`

#### 改造后

新增 `create_machine_max` 函数：

```python
# 文件：对数域改造 NES/NES_VMC.py

def create_machine_max(model: NESTotalAnsatz):
    """将 Flax NNX 模型包装为 NetKet 风格的 machine 函数"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, L_max = m(sigma)  # 返回 L.max()
        return L_max

    return machine, graphdef, state
```

---

## 三、数学稳定性分析

### 1. 数值溢出问题

当 `L` 矩阵的元素值较大时（如训练后期），`jnp.exp(L)` 会导致数值溢出：

```python
# 假设 L 的元素为 1000
jnp.exp(1000)  # → inf (数值溢出)

# 改造后：L_stable = L - L.max() = L - 1000 ≈ [0, -10, ...]
jnp.exp(L_stable)  # → [1, 4.5e-5, ...] (数值稳定)
```

### 2. 对数域改造的物理一致性

改造后的计算等价于：

$$
\Psi^{-1} \tilde{H} \Psi = \exp(L_{\text{stable}})^{-1} \cdot \exp(M_{\text{stable}})
$$

其中：
- $L_{\text{stable}} = \log(\Psi) - L_{\max}$
- $M_{\text{stable}} = \log(\tilde{H}\Psi) - L_{\max}$

这与论文 S8 章节的数值稳定化方案一致。

---

## 四、合理性分析

### ✅ 合理的部分

1. **稳定性提升**：对数域改造有效避免了 `exp()` 操作的数值溢出问题
2. **数学等价性**：在去除公共因子 $e^{L_{\max}}$ 后，矩阵求逆和行列式计算的相对关系保持不变
3. **兼容原有架构**：改造未改变 `NESTotalAnsatz` 的整体结构和采样器

### ⚠️ 需要注意的部分

1. **新引入 `create_machine_max`**：需要额外计算和传递 `L.max()`，增加了代码复杂度
2. **批量处理一致性**：`L.max()` 需要正确 reshape 为 `(batch, 1, 1)` 以匹配 `M` 的形状
3. **训练监控**：log_Psi 的值范围会发生变化（因为稳定化），监控时需注意

---

## 五、总结

| 改造项 | 改造内容 | 目的 |
|--------|----------|------|
| `L_stable = L - L.max()` | 矩阵稳定化 | 防止 exp() 溢出 |
| `M_stable = M - L.max()` | Hamiltonian 矩阵稳定化 | 保证与 Psi_Matrix 的数值一致性 |
| 新增 `create_machine_max` | 返回 `L.max()` | 提供稳定化所需的公共因子 |
| `NES_loss_energy_stable` | 新的损失函数 | 在对数域进行矩阵运算 |

改造提升了数值稳定性，对于深层网络或长期训练场景尤为重要，同时保持了与原算法数学上的等价性。

---

## 六、参考文件

- 原始版本：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/NES_new_vserison/NES_VMC.py`
- 改造版本：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC.py`
- 改造测试：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC对数域改造 K2.ipynb`
