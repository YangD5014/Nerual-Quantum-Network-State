# NES-VMC 代码效果差的原因分析

## 1. FCI 基准 vs 训练结果对比

| 状态 | FCI 能量 (Ha) | 训练结果 (Ha) | 误差 |
|------|--------------|--------------|------|
| E0 (基态) | -1.0155 | ~-1.22 | ~0.2 |
| E1 (第一激发态) | -0.8754 | ~-0.88 | ~0.0 |
| E2 (第二激发态) | -0.4294 | ~-0.10 | **~0.33 (严重!)** |
| E3 (第三激发态) | -0.2692 | - | - |

**关键观察**：
- 基态能量比 FCI 更负（过估计）
- 第二激发态完全错误（-0.10 vs -0.43），差了约 14 eV
- 训练 loss 在 -2.2 附近波动，不收敛到正确值

---

## 2. 核心问题分析

### 2.1 `Ham_Psi` 函数中的索引混淆问题（最关键）

查看 [NES_VMC.py:176-195](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L176-L195) 的 `Ham_Psi` 函数：

```python
def Ham_Psi(ha, total_ansatz, x):
    hilber_size = total_ansatz.n_spin  # 4
    k = total_ansatz.K                  # 2 or 3
    x = x.reshape(k, hilber_size)      # x.shape = (k, 4)
    H_psi_x_i = []
    for i in range(k):
        tmp = []
        for j in range(k):
            ele = Ham_psi(ha, model=total_ansatz.single_ansatz_list[j], x=x[i])
            tmp.append(ele)
        H_psi_x_i.append(tmp)
    HPsi = jnp.array(H_psi_x_i).reshape(k, k)
    return HPsi
```

**问题**：当计算 `H_psi_x[i,j]` 时：
- 代码计算：`Ham_psi(x[i])` 使用第 i 个样本，但取了 **j-th ansatz** 的模型输出
- 数学上应该是：`⟨x^i|H|ψ_j⟩`，即哈密顿量作用于 x^i 位置，但与第 j 个波函数耦合

**更严重的问题**：`Ham_psi(x[i])` 返回的是 **⟨x^i|H|x^i⟩**（对角元），而非完整的矩阵元！

### 2.2 `Ham_psi` 函数的物理含义问题

查看 [NES_VMC.py:165-175](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L165-L175)：

```python
def Ham_psi(ha, model, x):
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = jax.vmap(model)(x_primes)
    psi_vals = jnp.exp(log_psi_vals)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x
```

`Ham_psi` 计算的是 **H|ψ⟩ 在位置 x 处的投影**：
```
Hψ(x) = ⟨x|H|ψ⟩ = Σ_{x'} ⟨x|H|x'⟩ ψ(x')
```

这是正确的 scalar 值，但问题是：**`Ham_Psi` 构建矩阵的方式不正确**

### 2.3 损失函数与真实物理量的偏差

查看 [NES_VMC.py:198-210](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L198-L210)：

```python
def NES_loss_energy(ha, graphdef, params, x):
    total_model = nnx.merge(graphdef, params)
    log_psi_det, log_M = total_model(x)
    Psi_Matrix = jnp.exp(log_M)
    H_psi_x = Ham_Psi(ha, total_model, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv)), Psi_Matrix_inv
```

**损失函数计算**：`tr(Ψ^(-1) · H_ψ)`

但根据 NES-VMC 理论，正确的量应该是：
```
L = tr(S^(-1) · H) = tr(Ψ^(-1) · H_ext · Ψ)
```

其中 H 是哈密顿量矩阵（`H_mn = ⟨ψ_m|H|ψ_n⟩`），而 **不是** `H_ψ`。

### 2.4 扩展哈密顿量的结构理解

扩展哈密顿量定义为：
```
H_ext = H_1 ⊗ I_2 ⊗ ... ⊗ I_K + ... + I_1 ⊗ ... ⊗ H_K
```

当 `H_ext` 作用于乘积态 `Ψ(x) = ψ_i(x^1)...ψ_i(x^K)` 时：
```
H_ext Ψ = Σ_i (I⊗...⊗H⊗...⊗I) ψ_j(x^1)...ψ_j(x^K)
```

对于 i ≠ j 的项，由于不同子系统的态正交，这些矩阵元为 **零**。

**问题**：代码假设 H_ext 是块对角的（只在 i=j 时非零），这是 **错误的近似**。

---

## 3. 梯度计算的问题

查看 [NES_VMC.py:218-250](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L218-L250)：

```python
def nes_vmc_gradient(ha, graphdef, params, x_batch):
    E_L_batch = compute_local_energy_matrix_batch(ha, graphdef, params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)
    
    tr_E_loc_batch = jnp.real(jnp.trace(E_L_batch, axis1=1, axis2=2))
    tr_E_mean = jnp.real(jnp.trace(E_L_mean))
    
    tr_centered = tr_E_loc_batch - tr_E_mean
    dlogPsi_batch = vmap_grad_logPsi(params, x_batch)
    
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape((-1,) + (1,)*(grad_component.ndim - 1))
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)
    
    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
```

**问题 1**：梯度公式中的 **2 倍因子缺失**

标准 VMC 梯度是 `∇⟨E⟩ = 2⟨(E_L - E_0) ∇log ψ*⟩`，但代码使用的是：
```python
grad = jax.tree.map(weight_and_mean, dlogPsi_batch)
```
缺少了系数 2。

**问题 2**：对于复数参数，正确的梯度需要同时考虑 ∂/∂θ_re 和 ∂/∂θ_im，当前实现可能不完全正确。

---

## 4. QGT (量子几何张量) 计算的简化

查看 [NES_VMC.py:470-530](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L470-L530) 的 `compute_nes_qgt`：

```python
def compute_nes_qgt(machine, graphdef, params, samples, diag_shift=0.01):
    def forward_logM_flat(params, x):
        model = nnx.merge(graphdef, params)
        ln_det_M, ln_M = model(x)
        return jnp.ravel(ln_M)  # 只返回 log M 矩阵的展开
    
    def grad_single(x):
        def scalar_forward(params, x):
            f = forward_logM_flat(params, x)
            return jnp.real(f).sum() + 1j * jnp.imag(f).sum()
        grad = jax.grad(scalar_forward, holomorphic=True)(params, x)
        return grad_flat
```

**问题**：
1. 只对 `ln_M` 求导，**忽略了对 `ln_det_M` 的导数**
2. 但损失函数 `NES_loss_energy` 中使用的是 `log_psi_det`（行列式的对数）
3. QGT 应该反映 `log Ψ = log det(Ψ_matrix)` 的梯度结构，而非仅仅是 `log M`

---

## 5. 采样相关问题

### 5.1 初始状态生成

查看 [NES_VMC.py:292-325](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L292-L325) 的 `generate_random_initial_states`：

```python
def generate_random_initial_states(hi_ext, n_chains, seed=42):
    key = jr.PRNGKey(seed)
    n_spin = hi_ext.size // 2  # 每个子系统的自旋数
    for _ in range(n_chains):
        # ...
        x1 = s1[:n_spin]
        x2 = s2[n_spin:]  # 这里假设 x2 在后半部分
```

**问题**：对于 `hi_ext = hi ** K`（张量希尔伯特空间），`random_state` 返回的是展平的向量，**切分方式不正确**。

### 5.2 edges 定义与 K 的匹配

在 notebook 中使用：
```python
K = 3
tensor_edges = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,11)]
```

对于 K=3，总自旋轨道数 = 4×3 = 12，edges 覆盖 0-11。但代码中传入的 edges 可能不匹配。

---

## 6. 训练超参数问题

查看训练循环 [NES_VMC.py:580-640](file:///Users/yangjianfei/mac_vscode/神经网络量子态/5%20月/0510/NES/NES_VMC.py#L580-L640)：

```python
N_CHAINS = 16 
N_WARMUP = 32
N_SAMPLES_PER_CHAIN = 100
SWEEP_SIZE = 32
N_ITER = 100

optimizer = optax.sgd(learning_rate=0.01)
```

**问题**：
1. 学习率 0.01 可能过大，导致训练不稳定
2. 每个链只采样 100 个样本，统计误差可能较大
3. 没有使用更高级的优化器（如 Adam 或带有动量的 SGD）

---

## 7. 总结与建议

### 主要问题优先级

| 优先级 | 问题 | 严重程度 | 影响 |
|--------|------|--------|------|
| **P0** | `Ham_Psi` 索引混淆 | 致命 | 哈密顿量作用完全错误 |
| **P0** | `Ham_psi` 只返回对角元 | 致命 | 丢失所有 off-diagonal 贡献 |
| **P1** | 损失函数与理论偏差 | 高 | 训练目标不正确 |
| **P1** | QGT 忽略 `log det` 导数 | 高 | 自然梯度方向错误 |
| **P2** | 梯度公式缺 2 倍因子 | 中 | 收敛慢 |
| **P2** | 采样/初始状态问题 | 中 | 统计误差大 |
| **P3** | 超参数未调优 | 低 | 次要影响 |

### 建议的修复方向

1. **重写 `Ham_Psi`**：正确计算 `⟨x^i|H|x^j⟩` 矩阵元
2. **修正损失函数**：确保 `L = tr(S^(-1) · H_ext · Ψ)` 的实现
3. **完善 QGT**：包含 `log det` 的梯度贡献
4. **调整超参数**：降低学习率，增加样本数，考虑使用 Adam
5. **验证中间结果**：打印 `Ψ_matrix`、`H_psi_x`、`E_L` 的形状和值，确保维度匹配

### 验证方法

在修复前，建议：
1. 用 K=1（标准 VMC）验证基态能量是否正确
2. 检查 `Ham_psi` 是否返回正确的 local energy
3. 验证 `NES_loss_energy` 对于单位矩阵 ansatz 是否给出正确结果