# 神经网络变分蒙特卡洛（VMC）求解激发态

## 📖 目录

1. [背景介绍](#背景介绍)
2. [理论基础](#理论基础)
3. [代码实现](#代码实现)
4. [核心算法](#核心算法)
5. [使用指南](#使用指南)
6. [结果分析](#结果分析)

***

## 背景介绍

### 什么是变分蒙特卡洛（VMC）？

变分蒙特卡洛（Variational Monte Carlo, VMC）是量子多体问题中一种强大的数值方法。它通过**蒙特卡洛采样**和**变分原理**来近似计算量子系统的基态和激发态能量。

**变分原理**：对于任意归一化的试探波函数 $|\Psi\rangle$，有
$$\langle E \rangle = \frac{\langle \Psi | H | \Psi \rangle}{\langle \Psi | \Psi \rangle} \geq E\_0$$
其中 $E\_0$ 是系统的真实基态能量。

### 为什么需要神经网络？

传统VMC使用人为设定的试探波函数形式（如Slater-Jastrow波函数），需要大量物理直觉来设计。**神经网络VMC**使用神经网络作为通用函数逼近器，可以自动学习复杂的量子关联效应。

### 什么是激发态？

在量子力学中，除基态外的所有量子态都称为**激发态**：

- **第一激发态** $E\_1$：能量最低的激发态
- **第二激发态** $E\_2$：能量次低的激发态
- 以此类推...

激发态在光谱学、化学反应动力学等领域有重要应用。

### 求解激发态的挑战

1. **正交约束**：激发态必须与所有低能态正交
2. **变分原理不直接适用**：变分原理只保证基态能量下界
3. **局部极小问题**：容易收敛到错误的态

***

## 理论基础

### 1. 惩罚项方法

对于第一激发态，构造带惩罚的变分泛函：
$$\mathcal{L}(\theta) = \langle E \rangle\_\theta + \lambda \cdot |\langle \Psi\_0 | \Psi\_\theta \rangle|^2$$

其中：

- $\langle E \rangle\_\theta = \frac{\langle \Psi\_\theta | H | \Psi\_\theta \rangle}{\langle \Psi\_\theta | \Psi\_\theta \rangle}$ 是能量期望
- $\langle \Psi\_0 | \Psi\_\theta \rangle$ 是与基态的重叠
- $\lambda$ 是惩罚系数（通常取 5\~20）

**物理意义**：

- 能量项驱动态向低能方向优化
- 惩罚项强制态与基态正交（$|\langle \Psi\_0 | \Psi\_\theta \rangle|^2 \to 0$）

### 2. 多激发态推广

对于第 $k$ 个激发态，惩罚项扩展为：
$$\mathcal{L}(\theta\_k) = \langle E\_k \rangle + \sum\_{n=0}^{k-1} \lambda\_n |\langle \Psi\_n | \Psi\_k \rangle|^2$$

必须同时与**所有更低能态**正交。

### 3. VMC采样

使用波函数模方作为采样分布：
$$\sigma \sim |\Psi(\sigma; \theta)|^2$$

局部能量定义为：
$$E\_{\text{loc}}(\sigma) = \sum\_\eta H\_{\sigma\eta} \frac{\Psi(\eta)}{\Psi(\sigma)}$$

能量期望通过采样估计：
$$\langle E \rangle \approx \frac{1}{N} \sum\_{i=1}^{N} E\_{\text{loc}}(\sigma\_i)$$

### 4. Force-Based梯度

能量梯度（force-based）：
$$\nabla\_\theta \langle E \rangle = 2 \langle (E\_{\text{loc}} - \langle E \rangle) \nabla\_\theta \log \Psi^\* \rangle$$

惩罚项梯度：
$$\nabla\_\theta |\langle \Psi\_0 | \Psi\_\theta \rangle|^2 = 2 \text{Re} \left\[ \langle \Psi\_0 | \Psi\_\theta \rangle \cdot \langle \frac{\Psi\_0^*}{\Psi\_\theta^*} \nabla\_\theta \Psi\_\theta \rangle \right]$$

### 5. 自然梯度（QGT）

量子几何张量（Quantum Geometric Tensor, QGT）：
$$S\_{ij} = \langle \partial\_i \log \Psi^\* \partial\_j \log \Psi \rangle - \langle \partial\_i \log \Psi^\* \rangle \langle \partial\_j \log \Psi \rangle$$

自然梯度更新：
$$\theta\_{k+1} = \theta\_k - \alpha \cdot S^{-1} \nabla\_\theta \mathcal{L}$$

自然梯度考虑了参数空间的黎曼几何结构，比普通梯度更稳定高效。

***

## 代码实现

### 文件结构

```
激发态计算/
├── Excited_VMC.py           # 核心VMC实现
├── 激发态VMC0514.ipynb      # 使用示例
├── MultiStateVMC.py        # 多激发态扩展（新增）
└── 多激发态VMC测试.ipynb    # 多态测试（新增）
```

### Excited\_VMC.py 代码结构

#### 1. 系统定义（H₂分子）

```python
# 分子几何结构
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI计算获取基准能量
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
```

**说明**：

- 使用PySCF进行电子结构计算
- STO-3G基组是最小的原子轨道基组
- FCI（全配置相互作用）给出精确的量子化学能量基准

#### 2. 哈密顿量和希尔伯特空间

```python
# NetKet格式的哈密顿量算符
ha = nkx.operator.from_pyscf_molecule(mol)

# 费米子希尔伯特空间（4个自旋轨道，2个电子）
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,      # 2个空间轨道
    s=1/2,             # 自旋1/2
    n_fermions_per_spin=(1,1)  # α和β电子各1个
)
```

**说明**：

- $n=2$ 个空间轨道 → $2n=4$ 个自旋轨道
- 总电子数 = 2（一个H原子贡献1个电子）
- 希尔伯特空间维度 = $C\_4^2 = 6$

#### 3. 神经网络Ansatz

```python
class SingleStateAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, hidden_dim=16, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)
```

**网络架构**：

```
输入(4) → Linear(4→12) → tanh → Linear(12→12) → tanh → Linear(12→1) → 输出(1)
```

**特点**：

- 使用复数参数（complex64）以保持波函数的相位信息
- 双隐藏层MLP
- 输出是 $\log \Psi(\sigma)$（对数振幅）

#### 4. 模型封装

```python
def create_machine(model: nnx.Module):
    """将Flax NNX模型包装为NetKet风格的machine函数"""
    graphdef, state = nnx.split(model)
    
    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        return m(sigma)
    
    return machine, graphdef, state
```

**说明**：

- 使用Flax NNX的函数式编程范式
- 将模型参数和结构分离
- JAX JIT编译加速计算

***

## 核心算法

### 1. 局部能量计算

```python
@partial(jax.jit, static_argnames=("machine",))
def compute_local_energies(machine, params, sigma):
    """计算局部能量 E_loc(σ) = Σ_η H(σ→η) ψ(η)/ψ(σ)"""
    eta, H_eta = ha.get_conn_padded(sigma)  # 获取非零矩阵元
    logpsi_sigma = machine(params, sigma)
    logpsi_eta = machine(params, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)
    return jnp.sum(H_eta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)
```

**核心公式**：
$$E\_{\text{loc}}(\sigma) = \sum\_\eta H\_{\sigma\eta} \frac{\Psi(\eta)}{\Psi(\sigma)}$$

**实现细节**：

- `ha.get_conn_padded()` 返回哈密顿量的连接和矩阵元
- 对数形式避免数值溢出：$\frac{\Psi(\eta)}{\Psi(\sigma)} = \exp\[\log\Psi(\eta) - \log\Psi(\sigma)]$

### 2. 带惩罚的梯度计算

```python
@partial(jax.jit, static_argnames=("machine_gs","machine_es",))
def forces_expect_with_penalty(machine_gs, machine_es, params_gs, params_es, sigma, lam):
    # 1. 能量项
    Eloc = compute_local_energies(machine_es, params_es, sigma)
    E, std = statistics(Eloc)
    E_cent = Eloc - E

    # 2. 重叠积分 ⟨Ψ₀|Ψₑₛ⟩
    log_psi_es = machine_es(params_es, sigma)
    log_psi_gs = machine_gs(params_gs, sigma)
    O = jnp.mean(jnp.exp(log_psi_gs - log_psi_es))
    overlap2 = jnp.abs(O) ** 2

    # 3. 梯度计算...
```

**关键点**：

- `E_cent = Eloc - E`：中心化局部能量
- `O`：重叠积分的采样估计
- `overlap2`：重叠模方（惩罚项的目标）

### 3. 量子几何张量（QGT）

```python
def compute_qgt(machine, params, sigma, diag_shift=0.1):
    """计算QGT: S_ij = ⟨∂_i log ψ* ∂_j log ψ⟩ - ⟨∂_i log ψ*⟩⟨∂_j log ψ⟩"""
    n_samples = sigma.shape[0]
    
    # 计算每个样本的∇log ψ
    def compute_grad_for_sample(s):
        return jax.grad(lambda p: machine(p, s), holomorphic=True)(params)
    grad_matrix = jax.vmap(compute_grad_for_sample)(sigma)
    
    # 展平为矩阵
    grad_flat, unravel_fn = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(n_samples, -1)
    
    # 中心化
    grad_centered = grad_flat - jnp.mean(grad_flat, axis=0)
    
    # QGT = (1/N) Σ ∇log ψ* ∇log ψ^T
    qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
    qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])  # 正则化
    
    return qgt_reg, unravel_fn
```

**说明**：

- 使用`holomorphic=True`计算复数梯度
- `vmap`并行计算所有样本的梯度
- 正则化项`diag_shift * I`保证数值稳定

### 4. 自然梯度更新

```python
# 计算自然梯度
qgt_reg, unravel_fn = compute_qgt(machine_es, params_es, samples, diag_shift=0.001)
grad_flat, unravel_grad_fn = flatten_util.ravel_pytree(total_grad)
natural_grad_flat = jnp.linalg.solve(qgt_reg.astype(complex), grad_flat)
natural_grad = unravel_grad_fn(natural_grad_flat)

# 参数更新
updates, opt_state = optimizer.update(natural_grad, opt_state, params_es)
params_es = optax.apply_updates(params_es, updates)
```

**数学形式**：
$$\theta\_{\text{natural}} = S^{-1} \nabla \mathcal{L}$$

***

## 使用指南

### 基本使用流程

```python
# 1. 导入模块
from Excited_VMC import (
    ha, hi, E_fcis,
    SingleStateAnsatz, create_machine,
    compute_local_energies, statistics, compute_qgt,
    forces_expect_with_penalty, exact_energy_efficient, sampler
)
import flax.nnx as nnx
import optax

# 2. 创建模型
excited_model = SingleStateAnsatz(4, 12, rngs=nnx.Rngs(22))
machine_es, graphdef, params_es = create_machine(excited_model)

# 3. 加载已收敛的基态
# （需要先训练基态并保存）
checkpointer = ocp.PyTreeCheckpointer()
graphdef_gs, params_gs = nnx.split(ground_model)
restore_state = checkpointer.restore("暂存态/ground_state", graphdef_gs)
ground_model = nnx.merge(graphdef_gs, restore_state)
machine_gs, _, _ = create_machine(ground_model)

# 4. 初始化优化器
optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(params_es)

# 5. 训练循环
for step in range(N_ITER):
    # 采样
    samples, sampler_state = sampler.sample(machine_es, params_es, chain_length=20)
    samples = samples.reshape(-1, 4)
    
    # 计算梯度
    total_loss, E_mean, E_std, total_grad, overlap_sq = forces_expect_with_penalty(
        machine_gs, machine_es,
        params_gs, params_es,
        samples, lam=1.0
    )
    
    # 自然梯度更新
    qgt_reg, _ = compute_qgt(machine_es, params_es, samples)
    grad_flat, _ = flatten_util.ravel_pytree(total_grad)
    natural_grad = jnp.linalg.solve(qgt_reg, grad_flat)
    
    updates, opt_state = optimizer.update(natural_grad, opt_state, params_es)
    params_es = optax.apply_updates(params_es, updates)
    
    # 打印日志
    if step % 50 == 0:
        error = abs(E_mean.real - E_fcis[1])
        print(f"Step {step:3d} | E: {E_mean:.6f} | FCI-E1: {E_fcis[1]:.6f} | Err: {error:.6f}")
```

### 关键参数说明

| 参数              | 含义      | 建议值        |
| --------------- | ------- | ---------- |
| `lam`           | 正交惩罚系数  | 1.0\~10.0  |
| `learning_rate` | 学习率     | 0.01\~0.1  |
| `chain_length`  | 马尔可夫链长度 | 20\~50     |
| `n_samples`     | 采样数     | 500\~2000  |
| `diag_shift`    | QGT正则化  | 0.001\~0.1 |

### 注意事项

1. **基态必须先收敛**：激发态优化依赖于正确的基态波函数
2. **惩罚系数选择**：太大会导致训练震荡，太小会导致态不正交
3. **采样充分性**：确保马尔可夫链充分混合
4. **复数参数**：必须使用`param_dtype=complex`

***

## 结果分析

### 典型训练结果

根据`激发态VMC0514.ipynb`的训练输出：

```
============================================================
H₂ FCI 基准能量
============================================================
E0 = -1.01546825 Ha  |  激发能: 0.0000 eV
E1 = -0.87542794 Ha  |  激发能: 3.8107 eV
E2 = -0.42938376 Ha  |  激发能: 15.9482 eV
E3 = -0.26922131 Ha  |  激发能: 20.3064 eV

============================================================
开始 激发态 VMC 训练 (能量惩罚 + 正交约束 + 自然梯度)
目标：第一激发态 E1
============================================================
Step   0 | E: -0.40857893 ± 0.005799 | Overlap²: 0.170846 | Err: 0.466849
Step  50 | E: -0.78763678 ± 0.004294 | Overlap²: 0.004143 | Err: 0.087791
Step 100 | E: -0.84074360 ± 0.002813 | Overlap²: 0.002730 | Err: 0.034684
Step 150 | E: -0.85952520 ± 0.002004 | Overlap²: 0.003194 | Err: 0.015903
...
Step 399 | E: -0.85383937 ± 0.002262 | Overlap²: 0.003225 | Err: 0.021589

============================================================
训练完成！
第一激发态能量：-0.85383937 Ha
FCI 激发态基准：-0.87542794 Ha
绝对误差：0.055235 Ha
============================================================
```

### 结果解读

1. **能量收敛**：从初始的-0.41 Ha逐步收敛到-0.85 Ha左右
2. **正交性**：重叠平方从0.17降到0.003，说明正交约束有效
3. **误差**：最终误差约0.021 Ha（VMC的固有误差）
4. **激发能**：VMC结果与FCI基准的激发能误差在合理范围内

### 误差来源

1. **变分误差**：神经网络表达能力有限
2. **采样误差**：蒙特卡洛采样的统计误差
3. **正交不完全**：惩罚系数有限导致的残余重叠
4. **局部极小**：优化可能陷入次优解

### 改进方向

1. **更大的网络**：增加隐藏层维度或深度
2. **更多采样**：增加样本数减少统计误差
3. **自适应惩罚**：动态调整惩罚系数
4. **更优优化器**：使用Adam或其他自适应学习率方法

***

## 附录：数学公式汇总

### 核心损失函数

$$\mathcal{L}(\theta) = \langle E \rangle + \lambda |\langle \Psi\_0 | \Psi\_\theta \rangle|^2$$

### 能量期望值

$$\langle E \rangle = \frac{\langle \Psi | H | \Psi \rangle}{\langle \Psi | \Psi \rangle}$$

### 局部能量

$$E\_{\text{loc}}(\sigma) = \sum\_\eta H\_{\sigma\eta} \frac{\Psi(\eta)}{\Psi(\sigma)}$$

### 总梯度

$$\nabla\_\theta \mathcal{L} = 2\langle (E\_{\text{loc}} - \langle E \rangle) \nabla\_\theta \log \Psi^\* \rangle + 2\lambda \text{Re}\[\langle \Psi\_0 | \Psi\_\theta \rangle \cdot \langle \frac{\Psi\_0^*}{\Psi\_\theta^*} \nabla\_\theta \Psi\_\theta \rangle]$$

### 自然梯度更新

$$\theta\_{k+1} = \theta\_k - \alpha \cdot S^{-1} \nabla\_\theta \mathcal{L}$$

***

## 参考资料

1. **NetKet文档**: <https://www.netket.org/>
2. **Flax NNX**: <https://flax.readthedocs.io/en/latest/nnx/index.html>
3. **PySCF**: <https://pyscf.org/>
4. **JAX**: <https://jax.readthedocs.io/>

***

*文档生成时间：2024年5月*
