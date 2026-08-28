# NES-VMC 算法流程

基于 `LiH_molecule_K4_STO-3G.ipynb` 的代码顺序与《NES-VMC 算法详解》背景文档，提炼出的核心算法流程。

---

## 1. NES-VMC 核心思想

- 用 **K 个独立副本** 的单态 Ansatz 构造一个 K×K 行列式：
  $$\Psi_\theta(\mathbf{x}_1,\ldots,\mathbf{x}_K) = \det\!\big[f_j(\mathbf{x}_i)\big]_{i,j=1}^{K}$$
- 子组态互不相同 ⇒ 重叠矩阵自动为单位阵 ⇒ 同时获得 K 个激发态。
- 损失 = 局域能量矩阵的迹：
  $$E_{\mathrm{NES}} = \mathrm{Re}\,\mathrm{Tr}\!\left(\boldsymbol{\Psi}^{-1}\mathbf{H}\boldsymbol{\Psi}\right)$$

---

## 2. 核心流程图

```mermaid
flowchart TD
    A([开始]) --> B[构建 Hilbert + 哈密顿量 H<br/>载入扩展边 ext_edges]
    B --> C[NES 采样规则<br/>NESFermionHopRule<br/>约束: K 副本子组态互异]
    C --> D[初始化 K 个独立 SingleStateAnsatz<br/>组装 NESTotalAnsatz / 行列式结构]
    D --> E[构造机器函数<br/>total_machine, total_matrix_machine]
    E --> F[FCI/eigsh 基准能量<br/>eigvals 0..K-1]
    F --> G[初始化 NetKet 采样器<br/>MetropolisSampler + NES 规则]
    G --> H[初始化优化器<br/>clip_by_global_norm + SGD]
    H --> I([训练主循环 step = 0..N_ITER])

    I --> J[Metropolis 采样<br/>samples ~ |Psi|^2]
    J --> K[构造扩展组态 x_batch<br/>shape: B x K x SINGLE_SIZE]
    K --> L[计算局域能量矩阵 E_L<br/>E_L_ij = sum_m mels * Psi x']
    L --> M[计算 NES 损失与梯度<br/>loss = Re Tr Psi^-1 E_L]
    M --> N{Natural_Grad?}

    N -- 否 --> P[原始梯度]
    N -- 是 --> O[QGT 预条件<br/>nat_grad = QGT^-1 grad]
    O --> Q[梯度全局裁剪<br/>clip_by_global_norm]
    P --> Q
    Q --> R[optax.sgd 更新参数]
    R --> S[对角化 E_L<br/>eigvals = K 个态能量]
    S --> T[记录历史<br/>log_Psi / grad_norm / cond Psi]
    T --> U{收敛 or step = N_ITER-1?}

    U -- 否 --> I
    U -- 是 --> V[保存 history / 可视化]
    V --> W([结束])
```

---

## 3. 关键映射（代码 ↔ 公式）

| 步骤 | 代码函数 | 公式/作用 |
|------|---------|----------|
| Ansatz 构造 | `NESTotalAnsatz` + `create_machine_matrix` | $L_{ij}=\log f_j(\mathbf{x}_i)$，$\log\Psi=\mathrm{slogdet}(\exp L)$ |
| 采样 | `MetropolisSampler` + `NESFermionHopRule` | 从 $|\Psi|^2$ 采样，并保证 K 副本互异 |
| 局域能量 | `Ham_psi` / `Ham_Psi` | $(E_L)_{ij}=\sum_{\mathbf{x}'} \langle\mathbf{x}_i|H|\mathbf{x}'\rangle\Psi(\mathbf{x}'_j)$ |
| 损失 | `NES_loss_energy` | $E_{\mathrm{NES}}=\mathrm{Re}\,\mathrm{Tr}(\boldsymbol{\Psi}^{-1}\mathbf{H}\boldsymbol{\Psi})$ |
| 梯度 | `nes_vmc_gradient` / `make_grad_fn` | $\nabla_\theta E_{\mathrm{NES}}$ |
| 自然梯度 | `compute_qgt` / `make_qgt_fn` | $\delta\theta = (S+\epsilon I)^{-1}\nabla E$ |
| 能量提取 | `jnp.linalg.eig(E_L)` | 一次前向同时得到 K 个本征能量 |

---

## 4. 训练监控指标

| 指标 | 含义 | 期望 |
|------|------|------|
| `eig_vals[k]` | 第 k 激发态能量 | 逐步逼近 FCI 基准 |
| `loss` | NES 变分损失 | 单调下降，趋向 $\sum_{k=0}^{K-1} E_k$ |
| `log_Psi` (mean/min/max) | 复数波函数稳定性 | 虚部漂移小，无 NaN |
| `grad_norm_raw / natural / clipped` | 梯度范数链路 | clipped ≤ `clip_norm` |
| `cond(Ψ)` | 行列式矩阵条件数 | 数值稳定 ⇒ 条件数适中 |

---

## 5. 依赖文件

| 文件 | 作用 |
|------|------|
| `NES_VMC.py` | Ansatz、采样器、损失、梯度、QGT |
| `LiH.py` | 分子系统构建：Hilbert、H、扩展边、HF 态 |
| `LiH_molecule_K4_STO-3G.ipynb` | K=4 LiH/STO-3G 训练主流程 |

---

*文档生成时间：2026-08-04*
