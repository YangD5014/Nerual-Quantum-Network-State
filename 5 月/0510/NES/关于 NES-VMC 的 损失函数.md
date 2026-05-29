# NES-VMC 算法在 NetKet 官方 API 中的复现：目的与进展

## 1. 研究目的

**目标**：完全基于 NetKet 框架的高层 API（`MCState`、`VMC` 驱动）复现 **NES-VMC（Natural Excited State Variational Monte Carlo）算法**，用于计算量子多体系统（如 H₂ 分子）的前 $K$ 个激发态能量。

**要求**：

- 使用 NetKet 内置的扩展希尔伯特空间 `hi ** K`
- 使用 `MCState` 管理变分态与采样器
- 使用 `VMC` 驱动自动执行训练循环
- 最终通过训练得到的模型，对角化平均局域能量矩阵，获得基态与激发态能量

## 2. NES-VMC 算法核心思想

### 2.1 问题背景

在量子力学中，我们通常需要求解哈密顿算符 $\hat{H}$ 的本征值问题，即找到最低的 $K$ 个本征函数。对于量子多体系统，直接对角化哈密顿矩阵通常是不可行的，因为希尔伯特空间的维度随粒子数指数增长。

NES-VMC 将原系统前 $K$ 个激发态的求解问题**等价转化为一个"扩展系统"的基态求解问题**。

### 2.2 扩展希尔伯特空间

设 $x = (x\_1, \dots, x\_N)$ 表示一组包含 $N$ 个粒子的粒子集（particle set），其中 $x\_i$ 表示第 $i$ 个粒子的状态。扩展希尔伯特空间由 $K$ 个原系统副本张量积构成，每个配置对应 $K$ 个组态 $\mathbf{x} = (x^1, \dots, x^K)$。

### 2.3 TotalAnsatz 的构成

设 $\psi_i$ 表示第 $i$ 个 $N$ 粒子波函数（可能未归一化），则 **TotalAnsatz** 定义为矩阵 $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$ 的行列式：

$$
\Psi(\mathbf{x}) \equiv \det\begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \cdots & \psi_K(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \cdots & \psi_K(x^2) \\
\vdots & \vdots & \ddots & \vdots \\
\psi_1(x^K) & \psi_2(x^K) & \cdots & \psi_K(x^K)
\end{pmatrix}

$$

其中：

- $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$：将所有电子集合与所有波函数结合的矩阵
- $\psi_i(x^j)$：第 $i$ 个单态 Ansatz 在第 $j$ 个粒子集上的值
- $\Psi(\mathbf{x}) = \det(\Psi(\mathbf{x}))$：总 Ansatz，可以看作是由 $N$ 粒子波函数组成的未归一化 Slater 行列式

**关键性质**：通过将总 Ansatz 表示为单态 Ansatz 的行列式，可以防止不同 Ansatz 坍缩到同一状态，而不需要显式要求它们正交。

### 2.4 扩展哈密顿量

定义扩展哈密顿量 $\tilde{H} = \hat{H}\_1 \oplus \hat{H}\_2 \oplus \cdots \oplus \hat{H}\_K$，其中 $\hat{H}\_i$ 是仅作用于第 $i$ 个粒子集的哈密顿量。$\tilde{H}$ 的基态能量等于原系统 $\hat{H}$ 最低 $K$ 个能量之和，其基态波函数正是上述行列式形式的 $\Psi^\star$。

## 3. 损失函数

### 3.1 目标函数（Rayleigh 商）

NES-VMC 的目标函数为扩展哈密顿量关于总 Ansatz 的 Rayleigh 商：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\langle\Psi|\Psi\rangle}

$$

利用矩阵行列式引理，可以将其重写为迹形式：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\det(S)} = \mathrm{Tr}\left(S^{-1}\hat{H}\right)

$$

其中 $S$ 为重叠矩阵：

$$
S = \begin{pmatrix}
\langle\psi\_1|\psi\_1\rangle & \cdots & \langle\psi\_1|\psi\_K\rangle \\
\vdots & \ddots & \vdots \\
\langle\psi\_K|\psi\_1\rangle & \cdots & \langle\psi\_K|\psi\_K\rangle
\end{pmatrix}

$$

### 3.2 局域能量矩阵

通过 Monte Carlo 采样，损失函数可以写成期望值形式：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\Psi^{-1}(\mathbf{x})\tilde{H}\Psi(\mathbf{x})\right)\right]

$$

定义**局域能量矩阵**为：

$$
E\_L(\mathbf{x}) \equiv \Psi^{-1}(\mathbf{x})\tilde{H}\Psi(\mathbf{x})

$$

这是一个 $K \times K$ 矩阵，其迹即为标量局域能量。当 $K = 1$ 时，这退化为标准 VMC 中的局域能量。

## 4. 梯度公式

### 4.1 标准 VMC 梯度回顾

对于基态 VMC，能量关于变分参数 $\theta$ 的梯度为：

$$
\nabla\_\theta \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} = 2\mathbb{E}_{x \sim \psi^2}\left[\left(E_L(x) - \mathbb{E}_{x' \sim \psi^2}[E_L(x')]\right)\nabla\_\theta \log|\psi(x)|\right]

$$

### 4.2 NES-VMC 梯度

对于总 Ansatz，梯度计算类似。损失函数是迹形式 $\mathcal{L} = \mathrm{Tr}(E_L(\mathbf{x}))$，定义对数幅度：

$$
\log|\Psi(\mathbf{x})| = \log\det(\Psi(\mathbf{x})) = \mathrm{Tr}\left(\log(\Psi(\mathbf{x}))\right)

$$

梯度公式为：

$$
\nabla_\theta \mathcal{L} = 2\mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\left(E_L(\mathbf{x}) - \bar{E}_L\right)\nabla_\theta \log\Psi(\mathbf{x})\right)\right]

$$

其中：
- $E_L(\mathbf{x}) = \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})$ 是局域能量矩阵
- $\bar{E}_L = \mathbb{E}_{\mathbf{x}' \sim \Psi^2}[E_L(\mathbf{x}')]$ 是局域能量矩阵的期望值
- $\nabla_\theta \log\Psi(\mathbf{x})$ 是波函数矩阵对参数 $\theta$ 的对数梯度

当 $K = 1$ 时，上式退化为标准 VMC 的梯度公式。

### 4.3 批量 walker 的梯度估计

与标准 VMC 类似，可以使用同一批次中独立的 walker 来获得无偏梯度估计：

$$
\nabla_\theta \mathcal{L} = \frac{N-1}{2N}\mathbb{E}_{x\_1,\dots,x\_N}\left[\frac{1}{N}\sum_{i=1}^N\left(E_L(x_i) - \frac{1}{N}\sum\_{j=1}^N E_L(x_j)\right)\nabla_\theta \log|\Psi(x_i)|\right]

$$

## 5. 激发态能量提取

### 5.1 能量矩阵的对角化

训练完成后，通过大量采样累积局域能量矩阵：

$$
\bar{E}_L = \mathbb{E}_{\mathbf{x} \sim \Psi^2}[E_L(\mathbf{x})]

$$

然后对 $\bar{E}\_L$ 进行对角化：

$$
\bar{E}_L = U\Lambda U^{-1}

$$

其中 $\Lambda = \mathrm{diag}(E\_1, E\_2, \dots, E\_K)$ 包含按能量排序的本征值。

### 5.2 物理解释

当单态 Ansatz 是本征函数的线性组合 $\psi\_i = \sum\_j a\_{ij}\psi\_j^\star$ 时，有：

$$
\Psi^{-1}\hat{H}\Psi = A^{-1}\Lambda A

$$

其中 $A$ 是系数矩阵。因此，通过对角化可以直接获得各激发态的能量 $E\_1, E\_2, \dots, E\_K$。
为了给 NES-VMC 设计 Sampler:
这是为了给 H2 分子设计满足 NES-VMC 的 Sampler
以下代码不要更改 需要强调的是 edges =[α1,α2,β1,β2] 这样的顺序

```python
# ===================== 环境配置 =====================
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import nnx
import optax
from functools import partial
from jax import flatten_util
import matplotlib.pyplot as plt
from tqdm import tqdm
from jax import vmap,jit,grad
from NES_VMC import create_machine,hi_ext,hi,sampler,SingleStateAnsatz,NESTotalAnsatz,ha,compute_loss_and_grad,E_fcis
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI 精确基准
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCI 基准能量")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  激发能：{exc:.4f} eV")
# ===================== NetKet 哈密顿量和采样器 =====================
ha = nkx.operator.from_pyscf_molecule(mol)

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K=2
hi_ext = hi**K
edges = [(0, 1), (2, 3)]
tensor_edges = [(0,1),(2,3),(4,5),(6,7)]
```

首先要计算 损失函数:

$$
\hat{H}\Psi(\mathbf{x}) \triangleq 
\begin{pmatrix}
\hat{H}\psi_1(\mathbf{x}^1) & \dots & \hat{H}\psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\hat{H}\psi_1(\mathbf{x}^K) & \dots & \hat{H}\psi_K(\mathbf{x}^K)
\end{pmatrix}

$$

我比较确定了的对应代码是:
```python
def Ham_psi(ha: nk.operator.DiscreteOperator, model, x):
    """计算 Hψ(x)，model 输出 log_psi 时完全正确"""
    x_primes, mels = ha.get_conn_padded(x)
    # 1. 计算所有 σ' 的 log_psi
    log_psi_vals = jax.vmap(model)(x_primes)
    # 2. 指数还原成 ψ(σ')
    psi_vals = jnp.exp(log_psi_vals)
    # 3. 求和得到 Hψ(x)
    H_psi_x = jnp.sum(mels * psi_vals)
    return H_psi_x

def Ham_Psi(ha, total_ansatz, x):
    """计算扩展哈密顿量作用在总 Ansatz 上的矩阵"""
    hilber_size = total_ansatz.n_spin
    k = total_ansatz.K
    x = x.reshape(k, hilber_size)
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

接下来损失函数:

$$
\mathcal{L} = \mathrm{Tr}\left(\Psi(\mathbf{x})^{-1} \hat{H}\Psi(\mathbf{x})\right)

$$

目前已经得到了 $\hat{H}\Psi(\mathbf{x})$, 只需要计算 $\Psi(\mathbf{x})^{-1}$ 即可。

```python
def NES_loss_energy(ha:nk.operator.DiscreteOperator,total_ansatz:NESTotalAnsatz,x:jnp.ndarray):
    """计算NES-VMC的损失函数  \mathcal{L} = \mathrm{Tr}\left(\Psi(\mathbf{x})^{-1} \hat{H}\Psi(\mathbf{x})\right)
    ha: 哈密顿量
    total_ansatz: 总Ansatz
    x: 输入状态
    返回:
    损失函数值, \mathbf{x})^{-1} @ \hat{H}\Psi(\mathbf{x}) 的矩阵
    """
    #首先计算一下\Psi(\mathbf{x})
    value, log_M = total_ansatz(x)
    Psi_Matrix = jnp.exp(log_M)
    # 接下来计算 \hat{H}\Psi(\mathbf{x})
    H_psi_x = Ham_Psi(ha, total_ansatz, x)
    # \mathbf{x})^{-1} @ \hat{H}\Psi(\mathbf{x})
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix,H_psi_x)
    return jnp.trace(Psi_Matrix_inv),Psi_Matrix_inv
```
为了得到梯度: $\nabla_\theta \mathcal{L} = 2\mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\left(E_L(\mathbf{x}) - \bar{E}_L\right)\nabla_\theta \log\Psi(\mathbf{x})\right)\right]$
