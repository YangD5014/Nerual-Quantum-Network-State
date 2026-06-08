在自然梯度下降（Natural Gradient Descent）框架下，**QGT（量子几何张量，Quantum Geometric Tensor）矩阵 \(S\) 应该基于波函数的梯度（即 \(\partial_\theta \log \Psi\)），而不是损失函数的梯度**。这是自然梯度方法的核心定义：参数空间的黎曼度量由模型（波函数）本身的统计流形决定，与具体损失函数无关。具体说明如下：

---

### 1. 标准 VMC 中的自然梯度

对于基态 VMC，损失函数为能量期望 \(E(\theta) = \langle \psi_\theta | H | \psi_\theta \rangle / \langle \psi_\theta | \psi_\theta \rangle\)，自然梯度更新为：
\[
\theta \leftarrow \theta - \eta \, S^{-1} \, \nabla_\theta E,
\]
其中 \(S\) 就是 QGT（或其正则化形式），定义为：
\[
S_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | \partial_j \psi \rangle
= \langle \delta \partial_i \log \psi ,\, \delta \partial_j \log \psi \rangle.
\]
可见 \(S\) 仅依赖于波函数 \(\psi_\theta\) 对参数的导数（对数梯度），与能量 \(E\) 的梯度无关。**度量由参数化的概率/量子流形决定，而非当前的目标函数**。

---

### 2. NES-VMC 中的适用性

NES-VMC 的损失函数是：
\[
\mathcal{L}(\theta) = \operatorname{Tr}\big( \Psi_\theta^{-1} H \Psi_\theta \big),
\]
其中 \(\Psi_\theta\) 是一个 \(K \times K\) 矩阵值波函数（用于同时求解多个态）。尽管损失函数形式更复杂，但参数空间仍是波函数 \(\Psi_\theta\) 的参数。因此，**自然梯度下降仍应使用基于 \(\Psi_\theta\) 的 QGT**，即：
\[
S = \mathbb{E}\left[ \big( \delta \nabla_\theta \log \Psi \big)^\dagger \big( \delta \nabla_\theta \log \Psi \big) \right],
\]
其中 \(\log \Psi\) 是矩阵对数（或逐分量对数），且需要正确处理矩阵值输出的协方差结构。这里 **\(S\) 依然只依赖于波函数的对数梯度，与 \(\mathcal{L}\) 的具体形式无关**。

---

### 3. 为什么实验中自然梯度失效？

实验代码中的 `compute_nes_qgt` 计算了 `total_machine` 对单个样本 \(x\) 的梯度：
```python
def _single_grad(x):
    return jax.grad(lambda p: total_machine(p, x), holomorphic=True)(params)
```
这里 `total_machine(p, x)` 的输出是波函数 \(\Psi_\theta(x)\) 的某种表示（如 \(\log \Psi\) 或 \(\log M\)），**这个梯度确实是基于波函数的**，理论上符合上述定义。那么问题出在哪里？

#### ❌ 效率问题
- 使用 `for` 循环逐样本计算梯度，导致每次迭代需 \(N_{\text{samples}}\) 次反向传播（6400次），自然慢 ~13 倍。

#### ❌ 物理不匹配（更致命）
- NES-VMC 的波函数是 **矩阵值**，即 \(\Psi_\theta(x)\) 是一个 \(K \times K\) 矩阵。其对数梯度 \(\nabla_\theta \log \Psi\) 应当是一个 **张量**（参数 × 矩阵元素）。但 `total_machine` 的输出通常被展平成向量或直接返回两个量（`log_Psi, log_M`），导致 QGT 的计算**丢失了矩阵结构**。
- 损失函数 \(\mathcal{L} = \operatorname{Tr}(E_L)\) 的梯度中出现了 \(\Psi^{-1}\)、矩阵乘法等，这些操作会对不同通道的梯度产生混合。正确的 QGT 必须考虑 **矩阵值输出的协方差**，即：
  \[
  S = \mathbb{E}\left[ \operatorname{vec}(\delta \nabla_\theta \log \Psi)^\dagger \operatorname{vec}(\delta \nabla_\theta \log \Psi) \right],
  \]
  其中 \(\operatorname{vec}\) 将矩阵拉直。实验代码中直接对 `total_machine` 的标量输出（或单分量输出）求导，忽略了矩阵内部的关联，导致 QGT 与真实的黎曼度量不匹配。
- **因此，即使理论定义是“基于波函数的梯度”，当前实现也没有正确计算出矩阵值波函数的 QGT**，从而自然梯度方向错误（表现为 grad 正负号异常、收敛受阻）。

---

### 4. 结论与建议

| 问题 | 答案 |
|------|------|
| \(S\) 应基于哪种梯度？ | **波函数的梯度**（\(\partial_\theta \Psi\) 或 \(\partial_\theta \log \Psi\)），而不是损失函数的梯度。 |
| 实验为何失败？ | 1. 效率低：`for` 循环逐样本求梯度。<br>2. 物理错误：未正确处理矩阵值波函数的 QGT（丢失了矩阵结构的协方差）。 |

**改进建议**：
- 向量化梯度计算（`vmap`）。
- 正确构造矩阵值波函数的 QGT：对每个样本，计算 `grad_log_psi` 形状为 `(num_params, K, K)`，然后展平后计算协方差。
- 参考 NetKet 中对多态 VMC 的 QGT 实现（如 `nk.optimizer.SR` 的 `qgt` 参数）。

**一句话总结**：自然梯度下降的核心度量 **必须是基于模型（波函数）的 QGT**，而不是基于损失函数的梯度；但在 NES-VMC 中，必须正确处理矩阵值波函数的协方差结构，否则即便使用了波函数梯度，也无法加速收敛。