# NES-VMC 对数域数值稳定性改造说明

## 1. 改造目标

在 NES-VMC 中，总 Ansatz 写成行列式形式：

$$
\Psi(x^1,\ldots,x^K)=\det M,
\qquad
M_{ij}=\psi_j(x^i).
$$

代码中通常不直接输出 $\psi_j(x^i)$，而是输出它的对数：

$$
L_{ij}=\log \psi_j(x^i),
\qquad
M_{ij}=\exp(L_{ij}).
$$

原始代码后期训练时容易出现：

- `jnp.exp(L)` 溢出为 `inf`；
- `jnp.exp(log_psi_vals)` 下溢为 `0`；
- 后续 `jnp.log(0)` 变成 `-inf`；
- `solve`、`trace`、梯度中继续传播出 `nan`。

因此，本次改造的目标是：

1. 在构造 $M$ 和 $H\Psi$ 时尽量保留在 log domain；
2. 按论文 S8 的思想，对 $\log\psi_i(x^j)$ 的整个矩阵取全局最大值并减去；
3. 保证稳定化前后的局域能量矩阵数学等价；
4. 修复原代码中由返回值、变量名、采样约束等导致的实现错误。

---

## 2. 论文 S8 对数域稳定化思想

论文 S8 的核心说法是：在计算矩阵 $\Psi$ 和 $\hat O\Psi$ 时，先在 log domain 中计算 $\psi_i(x^j)$ 和 local operator，然后从：

$$
\log\psi_i(x^j)
$$

以及对应的 operator 作用项中，减去 $\log\psi_i(x^j)$ 在所有 $i,j$ 上的最大值，再从 log domain 转回 real domain。

也就是说，对每一个 walker / sample，取：

$$
L_{\max}=\max_{i,j}\operatorname{Re}(L_{ij}).
$$

注意这里是整个 $K\times K$ 矩阵的全局最大值，不是每一行的最大值。

在代码中对应为：

```python
L_max = jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)
L_stable = L - L_max
M_stable = jnp.exp(L_stable)
```

由于 $L$ 是复数，对 `L.max()` 直接取最大值是不合适的。真正控制 $|\exp(L)|$ 大小的是 $\operatorname{Re}(L)$，所以必须使用：

```python
jnp.real(L)
```

来取最大值。

---

## 3. 数学等价性

原始局域能量矩阵为：

$$
E_L=M^{-1}H\Psi.
$$

令：

$$
c=L_{\max}=\max_{i,j}\operatorname{Re}(L_{ij}).
$$

稳定化后的矩阵为：

$$
M_s=\exp(L-c)=e^{-c}M.
$$

右侧的哈密顿量作用矩阵也用同一个 $c$ 缩放：

$$
(H\Psi)_s=e^{-c}H\Psi.
$$

于是：

$$
M_s^{-1}(H\Psi)_s
=
(e^{-c}M)^{-1}(e^{-c}H\Psi)
=
M^{-1}H\Psi.
$$

因此，只要 $M$ 和 $H\Psi$ 使用同一个 $L_{\max}$，稳定化前后的局域能量矩阵完全等价。

---

## 4. 本次代码改造内容

### 4.1 Ansatz 返回真实 `log_Psi`、稳定化矩阵和 `L_max`

原始问题代码大致为：

```python
L_stable = L - L.max()
sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
return log_Psi_stable, L_stable, L.max()
```

这里有三个问题：

1. `L.max()` 直接作用在 complex array 上，不合理；
2. `log_Psi_stable` 不是原始波函数的真实 `log_Psi`；
3. 如果把 `log_Psi_stable` 给 Metropolis sampler，会改变采样分布。

改造后使用：

```python
L_max = jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)
L_stable = L - L_max
M_stable = jnp.exp(L_stable)

sign, log_abs_det_stable = jnp.linalg.slogdet(M_stable)
log_Psi = log_abs_det_stable + self.K * jnp.squeeze(L_max) + 1j * jnp.angle(sign)

return log_Psi, L_stable, L_max
```

这里要特别注意：

$$
\det(e^{L-L_{\max}})=e^{-K L_{\max}}\det(e^L).
$$

所以真实的 `log_Psi` 必须加回：

$$
K L_{\max}.
$$

否则采样器会采样错误的分布。这个 bug 很隐蔽，因为它不一定马上报错，只是会认真地训练一个错误目标，堪称代码界的礼貌诈骗。

---

### 4.2 wrapper 函数统一三返回值

由于 `NESTotalAnsatz` 现在返回：

```python
log_Psi, L_stable, L_max
```

所以 wrapper 必须同步修改。

#### `create_machine`

用于采样器，只返回真实 `log_Psi`：

```python
def create_machine(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return log_Psi

    return machine, graphdef, state
```

#### `create_machine_matrix`

用于损失函数，返回稳定化后的 $L$：

```python
def create_machine_matrix(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return L_stable

    return machine, graphdef, state
```

#### `create_machine_matrix_max`

用于损失函数中稳定化 $H\Psi$：

```python
def create_machine_matrix_max(model):
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_Psi, L_stable, L_max = m(sigma)
        return L_max

    return machine, graphdef, state
```

---

### 4.3 将 `Ham_psi` 改为 log-domain 计算

原代码中：

```python
log_psi_vals = single_machine(params, x_primes)
psi_vals = jnp.exp(log_psi_vals)
return jnp.sum(mels * psi_vals)
```

问题是：如果 `log_psi_vals` 很小，`jnp.exp(log_psi_vals)` 会下溢到 0。之后再做：

```python
jnp.log(Ham_Psi(...))
```

就可能得到 `-inf`。

改造后新增 `log_Ham_psi`：

```python
def complex_logsumexp(log_terms):
    c = jnp.max(jnp.real(log_terms))
    s = jnp.sum(jnp.exp(log_terms - c))
    return c + jnp.log(s)
```

```python
def log_Ham_psi(ha, single_machine, params, x):
    x_primes, mels = ha.get_conn_padded(x)
    log_psi_vals = single_machine(params, x_primes)

    nonzero = mels != 0
    log_abs_mels = jnp.where(nonzero, jnp.log(jnp.abs(mels)), -jnp.inf)
    phase_mels = jnp.where(nonzero, jnp.angle(mels), 0.0)

    log_terms = log_abs_mels + log_psi_vals + 1j * phase_mels
    return complex_logsumexp(log_terms)
```

数学上对应：

$$
H\psi_j(x)=\sum_{x'} H_{x,x'}\psi_j(x')
=
\sum_{x'} H_{x,x'}\exp(\log\psi_j(x')).
$$

在 log-domain 中写成：

$$
\log H\psi_j(x)
=
\log\sum_{x'}\exp\left(\log |H_{x,x'}|+\log\psi_j(x')+i\arg H_{x,x'}\right).
$$

这样可以避免在求和前直接把很小的 `log_psi` 转成 0。

---

### 4.4 新增 `log_Ham_Psi`

原来的 `Ham_Psi` 返回普通域矩阵：

$$
(H\Psi)_{ij}=H\psi_j(x^i).
$$

稳定化版本需要返回：

$$
\log(H\Psi)_{ij}=\log(H\psi_j(x^i)).
$$

因此新增：

```python
def log_Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)

    def _single_LogHamPsi(x_single):
        rows = []
        for i in range(K):
            xi = x_single[i]
            row = []
            for j in range(K):
                machine_j = single_machine_list[j]
                params_j = total_params['single_ansatz_list'][j]
                val = log_Ham_psi(ha, machine_j, params_j, xi)
                row.append(val)
            rows.append(jnp.stack(row))
        return jnp.stack(rows)

    if x.ndim == 2:
        return _single_LogHamPsi(x)
    elif x.ndim == 3:
        return jax.vmap(_single_LogHamPsi)(x)
    else:
        raise ValueError(f"不支持的输入形状: {x.shape}")
```

这里顺便避免了在 `vmap` 内部大量使用 `.at[i, j].set()`。虽然 `.at` 在 JAX 中不是普通意义的原地修改，但用 list + `jnp.stack` 更直观，也更不容易写错 shape。

---

### 4.5 改造稳定化损失函数

原代码：

```python
M = jnp.log(Ham_Psi(ha, single_machine_list, total_params, x))
M_stable = M - total_max_machine(total_params, x).reshape(-1, 1, 1)
HPsi_stable = jnp.exp(M_stable)
```

问题是：`Ham_Psi` 已经在普通域中计算完成，数值下溢已经可能发生。之后再 `log`，已经太晚了。

改造后：

```python
def NES_loss_energy_stable(
    ha,
    total_matrix_machine,
    total_max_machine,
    single_machine_list,
    total_params,
    x,
):
    L_stable = total_matrix_machine(total_params, x)
    L_max = total_max_machine(total_params, x)

    Psi_Matrix_stable = jnp.exp(L_stable)

    log_HPsi = log_Ham_Psi(ha, single_machine_list, total_params, x)
    log_HPsi_stable = log_HPsi - L_max
    HPsi_stable = jnp.exp(log_HPsi_stable)

    E_L = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    loss = jnp.real(jnp.trace(E_L, axis1=-2, axis2=-1))

    return loss, E_L
```

数据流变为：

```text
L = log ψ
↓
L_max = max_{i,j} Re(L_ij)
↓
L_stable = L - L_max
↓
M_stable = exp(L_stable)

log_HPsi = log(H ψ)    # 直接在 log domain 中计算
↓
log_HPsi_stable = log_HPsi - L_max
↓
HPsi_stable = exp(log_HPsi_stable)

E_L = solve(M_stable, HPsi_stable)
```

---

## 5. 原代码的主要问题总结

### 5.1 `L.max()` 直接作用于复数矩阵

原代码：

```python
L_stable = L - L.max()
```

问题：

- `L` 是 complex；
- 复数没有自然的大小关系；
- 控制 `exp(L)` 模长的是 `Re(L)`；
- 应改为 `jnp.max(jnp.real(L), axis=(-2, -1), keepdims=True)`。

---

### 5.2 使用了稳定化后的 `log_Psi_stable` 做采样

原代码返回：

```python
return log_Psi_stable, L_stable, L.max()
```

问题：

- sampler 应该使用真实的 $\log\Psi$；
- `log_Psi_stable` 少了 $K L_{\max}$；
- 这会改变 Metropolis 接受率，从而改变采样分布。

正确做法：

```python
log_Psi = log_abs_det_stable + K * L_max + phase
```

---

### 5.3 `Ham_Psi` 仍然在普通域中计算

原代码：

```python
psi_vals = jnp.exp(log_psi_vals)
return jnp.sum(mels * psi_vals)
```

问题：

- `exp(log_psi_vals)` 可能下溢为 0；
- 后面 `jnp.log(Ham_Psi(...))` 会出现 `log(0) = -inf`；
- 这正是训练中出现 `-inf`、`nan` 的重要来源。

正确做法：

- 不要先算普通域 `Ham_Psi`；
- 直接计算 `log_Ham_Psi`；
- 再减去同一个 `L_max`。

---

### 5.4 wrapper 返回值解包错误

原代码中 `NESTotalAnsatz` 返回三个值，但 wrapper 仍然只解包两个：

```python
log_psi_total, log_M_matrix = m(sigma)
```

问题：

- 会导致 `ValueError: too many values to unpack`；
- 或者在不同版本混用时产生更隐蔽的错误。

正确做法：

```python
log_Psi, L_stable, L_max = m(sigma)
```

---

### 5.5 变量名和函数名错误

原代码中有：

```python
total_matirx_max, _, _ = create_machine_max(total_ansatz)
```

问题：

- `matirx` 拼写错误；
- `create_machine_max` 没有定义；
- 后面又使用 `total_max`，但它没有定义。

正确写法：

```python
total_matrix_max, _, _ = create_machine_matrix_max(total_ansatz)
```

训练循环中传入：

```python
total_machine_max=total_matrix_max
```

---

### 5.6 duplicate check 只检查了第 0 个子组态

原代码：

```python
return jnp.any(
    jnp.all(sub[..., 1:, :] == sub[..., 0:1, :], axis=-1),
    axis=-1
).squeeze()
```

问题：

它只检查：

```text
x1 == x0
x2 == x0
...
```

但没有检查：

```text
x1 == x2
x1 == x3
...
```

如果两个非第 0 个子组态重复，矩阵 $M$ 会出现两行相同，导致：

$$
\det M=0,
\qquad
\log\Psi=-\infty.
$$

正确做法是 pairwise 检查所有子组态：

```python
sub = sigma_ext.reshape((-1, self.K, self.single_size))
eq = jnp.all(sub[:, :, None, :] == sub[:, None, :, :], axis=-1)
eye = jnp.eye(self.K, dtype=bool)
dup = jnp.any(eq & ~eye[None, :, :], axis=(1, 2))
```

---

### 5.7 对 complex `log_Psi` 直接做 min/max 监控

原代码中可能出现：

```python
log_Psi_batch.min()
log_Psi_batch.max()
```

问题：

- `log_Psi_batch` 是 complex；
- 对 complex 做 min/max 不合适。

正确做法是监控实部：

```python
log_Psi_real = jnp.real(log_Psi_batch)
log_Psi_real.min()
log_Psi_real.max()
```

---

## 6. 改造后的完整数据流

```text
输入样本 x: (batch, K, n_spin)
↓
NESTotalAnsatz
  计算 L_ij = log ψ_j(x^i)
  计算 L_max = max_{i,j} Re(L_ij)
  计算 L_stable = L - L_max
  计算真实 log_Psi = logdet(exp(L_stable)) + K L_max
↓
采样器使用真实 log_Psi
↓
损失函数使用 L_stable 构造 M_stable
↓
log_Ham_Psi 直接在 log domain 中计算 log(Hψ_j(x^i))
↓
log_HPsi_stable = log_HPsi - L_max
↓
HPsi_stable = exp(log_HPsi_stable)
↓
E_L = solve(M_stable, HPsi_stable)
↓
loss = Re Tr(E_L)
↓
梯度计算与参数更新
```

---

## 7. 本次改造没有完全解决的问题

这次改造主要解决的是：

- `exp(L)` 的溢出；
- `exp(log_psi)` 的下溢；
- `log(0)` 导致的 `-inf`；
- wrapper 和变量名错误；
- 采样器重复组态检查不完整。

但它不保证完全解决所有数值问题。

仍然可能存在：

1. **矩阵病态问题**  
   即使 $M_s$ 的元素不会溢出，如果 $M_s$ 接近奇异，`jnp.linalg.solve` 仍然可能产生巨大数值。

2. **复数相消问题**  
   `complex_logsumexp` 可以减少溢出/下溢，但如果求和项发生严重相消，结果仍可能接近 0。

3. **节点附近 local energy 发散问题**  
   NES-VMC 中即使 $\det M\neq 0$，某些单个矩阵元素 $\psi_i(x^j)$ 仍可能接近 0，这会导致 local energy 不稳定。

4. **梯度爆炸问题**  
   如果 `grad norm` 很大，仍然需要梯度裁剪、减小学习率或加强自然梯度正则化。

---

## 8. 建议监控量

训练时建议加入以下监控：

```python
log_Psi_real = jnp.real(total_machine(total_params, x_batch))
L_stable = total_matrix_machine(total_params, x_batch)
L_max = total_matrix_max(total_params, x_batch)

print("log_Psi real mean/min/max", log_Psi_real.mean(), log_Psi_real.min(), log_Psi_real.max())
print("L_max mean/min/max", L_max.mean(), L_max.min(), L_max.max())
print("L_stable real min/max", jnp.real(L_stable).min(), jnp.real(L_stable).max())
print("grad norm", jnp.linalg.norm(grad_flat))
```

如果条件允许，也可以监控：

```python
cond_M = jnp.linalg.cond(jnp.exp(L_stable))
```

如果 `cond_M` 很大，说明问题不是指数溢出，而是矩阵接近奇异。那就不是 `L_max` 一个补丁能救的，毕竟数值线性代数从不讲人情。

---

## 9. 小结

本次改造的核心是：

$$
L_{\max}=\max_{i,j}\operatorname{Re}(L_{ij})
$$

$$
M_s=\exp(L-L_{\max})
$$

$$
(H\Psi)_s=\exp(\log(H\Psi)-L_{\max})
$$

$$
E_L=M_s^{-1}(H\Psi)_s=M^{-1}H\Psi.
$$

同时，采样器必须使用真实的：

$$
\log\Psi=\log\det(M_s)+K L_{\max}.
$$

如果只返回 `log_Psi_stable`，就会改变采样分布；如果只稳定化 $M$ 而不稳定化 $H\Psi$，仍然会在 `Ham_Psi` 里下溢；如果 `Ham_Psi` 先普通域计算再 `log`，就已经错过了对数域稳定化最关键的一步。

因此，本次改造不是简单地把：

```python
L_stable = L - L.max()
```

换成另一个写法，而是把 `Ansatz -> wrapper -> HΨ -> loss -> sampler` 这一整条链路统一到对数域稳定化框架中。
