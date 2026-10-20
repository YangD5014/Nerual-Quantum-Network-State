# Quantum-Enhanced NES-VMC：方法方案总结

## 1. 研究目标

构建 **Quantum-Enhanced Natural Excited-State Variational Monte Carlo（QE-NES-VMC）**，将经典神经网络（NN）、参数化量子线路（PQC）和 NES-VMC 的多态联合波函数结构结合起来，并研究面向联合分布的自回归重要性采样。

目标不是简单地把 PQC 加到现有 NES-VMC 上，而是建立一套数学自洽的混合波函数、联合采样、加权能量估计和参数优化框架。

> **重要说明：** 以下推导以联合波函数确实定义为
> \[
> \mathcal D(X)=\det[\psi_i(\mathbf x_j)]_{i,j=0}^{K-1}
> \]
> 为前提。实现前必须核对现有 NES-VMC 代码中 \(\Psi=\det[\psi_0,\ldots,\psi_{K-1}]\) 的精确定义。如果实际算法的行列式构造不同，采样分布、局部能量和梯度都需要相应修改。

## 2. 与 Quantum-Enhanced VMC 的关系

相关 Quantum-Enhanced VMC 工作的核心思路包括：

1. 将波函数 Ansatz 分成经典神经网络部分和量子线路部分。
2. 使用经典自回归网络生成候选构型。
3. 使用重要性采样校正提议分布与目标波函数分布之间的差异，减少采样阶段对昂贵量子评估的依赖。

QE-NES-VMC 的潜在贡献应放在 **NES-VMC 的多态联合结构** 上，而不应只声称“把 PQC 用于激发态求解”。单纯组合已有组件未必足以构成强创新，需要通过严格推导和对照实验验证联合采样、梯度与优化设计的必要性。

## 3. 混合单态 Ansatz

对 \(K\) 个待求态，定义

\[
\psi_i(\mathbf x;\lambda_i,\theta_i)
=
\phi_i^{\mathrm{NN}}(\mathbf x;\lambda_i)
\phi_i^{\mathrm{QC}}(\mathbf x;\theta_i),
\qquad i=0,\ldots,K-1.
\]

- \(\phi_i^{\mathrm{NN}}\)：经典神经网络部分。
- \(\phi_i^{\mathrm{QC}}\)：PQC 对应的量子部分。
- \(\lambda_i,\theta_i\)：分别为经典和量子参数。
- \(\mathbf x\)：二次量子化 Fock 空间中的占据数比特串，可施加粒子数、自旋等物理约束。

### 两种可比较的参数化设计

**方案 A：每个态拥有独立 NN 和 PQC**

\[
\psi_i(\mathbf x)=f_{\lambda_i}(\mathbf x)q_{\theta_i}(\mathbf x).
\]

优点是灵活，适合作为首个基线；缺点是参数和量子线路评估开销可能较大。

**方案 B：共享部分参数**

例如共享 PQC：

\[
\psi_i(\mathbf x)=f_{\lambda_i}(\mathbf x)q_{\theta}(\mathbf x).
\]

共享结构可能减少参数或计算，但也可能限制不同激发态的表达能力，并引入更强的优化耦合。

**建议：** 先用方案 A 建立可复现基线，再测试共享参数设计。

二次量子化表示正确编码了费米子 Fock 空间及费米子算符的性质，但它不意味着不同变分态自动正交。态间正交性与本征态质量仍须通过 NES-VMC 的目标函数和数值诊断来处理。

## 4. 多态联合行列式

令一次联合样本为

\[
X=(\mathbf x_0,\mathbf x_1,\ldots,\mathbf x_{K-1}).
\]

定义振幅矩阵

\[
A_{ij}(X)=\psi_i(\mathbf x_j),
\]

并构造联合波函数

\[
\mathcal D(X)=\det A(X).
\]

代入混合 Ansatz 得到 QE-NES-VMC 的候选形式：

\[
\boxed{
\mathcal D_{\lambda,\theta}(X)
=
\det\left[
\phi_i^{\mathrm{NN}}(\mathbf x_j;\lambda_i)
\phi_i^{\mathrm{QC}}(\mathbf x_j;\theta_i)
\right]_{i,j=0}^{K-1}
}
\]

这里的行列式是对“态索引 × 联合构型索引”形成的振幅矩阵取行列式；它不应与电子 Slater 行列式混淆。

该式是否与现有 NES-VMC 实现完全一致，必须通过源码确认。

## 5. 目标联合分布

若联合采样目标为 \(|\mathcal D(X)|^2\)，则

\[
P_{\mathrm{NES}}(X)
=
\frac{|\mathcal D(X)|^2}{Z_{\mathcal D}},
\qquad
Z_{\mathcal D}=\sum_X|\mathcal D(X)|^2.
\]

这里的求和针对允许的联合构型空间。采样器必须覆盖目标分布的支持集；若某些目标概率非零的构型在提议分布下概率为零，则重要性采样无法正确恢复目标分布。

## 6. 自回归提议分布

使用经典自回归模型 \(q_\eta(X)\) 生成候选联合构型：

\[
q_\eta(X)=
\prod_{j=0}^{K-1}
q_\eta(\mathbf x_j\mid \mathbf x_0,\ldots,\mathbf x_{j-1}).
\]

也可以先测试更简单的独立提议分布：

\[
q_\eta(X)=\prod_{j=0}^{K-1}p_\eta(\mathbf x_j).
\]

- **联合自回归提议：** 能显式建模联合构型间的依赖关系，但模型更复杂。
- **独立提议：** 实现简单，但可能需要承担更大的权重方差。

自回归模型可在生成阶段直接约束粒子数、自旋等条件，以避免生成不合法的 Fock 构型。

## 7. 联合重要性权重

对从 \(q_\eta(X)\) 生成的样本，定义未归一化重要性权重

\[
\boxed{
w(X)=\frac{|\mathcal D(X)|^2}{q_\eta(X)}
}
\]

批次内归一化权重为

\[
\widetilde w_b=
\frac{w(X^{(b)})}{\sum_{c=1}^{B}w(X^{(c)})}.
\]

关键点：

- 权重必须由**完整联合行列式**计算，而不是将各态的单独权重简单相乘。
- 一般有
  \[
  |\det A(X)|^2\ne\prod_{i,j}|\psi_i(\mathbf x_j)|^2.
  \]
- 提议分布必须覆盖目标分布的支持集。
- 自归一化重要性采样在有限样本下通常不是严格无偏的，但在适当条件下具有一致性。
- 如果 \(q_\eta\) 与目标分布差异很大，权重会退化，估计方差可能非常高。

### 有效样本量

用 ESS 监控权重退化：

\[
\mathrm{ESS}=
\frac{(\sum_b w_b)^2}{\sum_b w_b^2}.
\]

可测试的改进包括：预训练提议网络、混合多个提议分布、增加合法构型覆盖、监控权重尾部与 ESS。

## 8. 联合局部能量与加权能量估计

假设 NES-VMC 的扩展哈密顿量为

\[
\mathcal H_{\mathrm{ext}}=\sum_{j=0}^{K-1}H^{(j)},
\]

其中 \(H^{(j)}\) 只作用于联合构型 \(X\) 的第 \(j\) 个坐标。则

\[
E_L(X)=
\frac{\mathcal H_{\mathrm{ext}}\mathcal D(X)}
{\mathcal D(X)}.
\]

对离散构型空间，可写成

\[
E_L(X)=
\sum_{j=0}^{K-1}\sum_{\mathbf x'_j}
H_{\mathbf x_j,\mathbf x'_j}
\frac{\mathcal D(X_{j\rightarrow\mathbf x'_j})}
{\mathcal D(X)}.
\]

其中 \(X_{j\rightarrow\mathbf x'_j}\) 表示只替换第 \(j\) 个坐标的联合构型。

使用重要性样本的加权估计：

\[
\boxed{
\widehat E_{\mathrm{ext}}=
\frac{\sum_{b=1}^{B}w(X^{(b)})E_L(X^{(b)})}
{\sum_{b=1}^{B}w(X^{(b)})}
}
\]

该估计是自归一化重要性采样估计，有限样本下通常有偏。

### PQC 振幅访问的硬件限制

量子线路通常自然产生量子态，而不是无成本输出任意复振幅。经典 statevector 模拟器可能允许直接读取振幅；真实量子硬件则需要适用的干涉测量、辅助比特或其他振幅估计方案。因此，重要性采样可能减少采样阶段对 PQC 的重复调用，但**不会自动消除计算行列式及局部能量所需的量子振幅评估成本**。论文必须明确振幅如何估计、每次估计需要多少 shots，以及总测量预算。

## 9. 联合波函数的参数梯度

对参数 \(\alpha\)，定义

\[
O_\alpha(X)=
\frac{\partial\log\mathcal D(X)}{\partial\alpha}.
\]

当矩阵 \(A(X)\) 可逆时，利用行列式微分恒等式：

\[
\boxed{
O_\alpha(X)=
\operatorname{Tr}\left[
A^{-1}(X)\frac{\partial A(X)}{\partial\alpha}
\right]
}
\]

如果 \(\alpha\) 是第 \(i\) 个态的经典 NN 参数：

\[
\frac{\partial A_{ij}}{\partial\alpha}
=
\frac{\partial\phi_i^{\mathrm{NN}}(\mathbf x_j)}
{\partial\alpha}
\phi_i^{\mathrm{QC}}(\mathbf x_j).
\]

如果 \(\alpha\) 是 PQC 参数：

\[
\frac{\partial A_{ij}}{\partial\alpha}
=
\phi_i^{\mathrm{NN}}(\mathbf x_j)
\frac{\partial\phi_i^{\mathrm{QC}}(\mathbf x_j)}
{\partial\alpha}.
\]

经典部分可以用反向传播；PQC 部分可根据线路结构使用参数移位规则或其他量子梯度估计方法。

**数值风险：** 当 \(A(X)\) 接近奇异或 \(\det A(X)\) 接近零时，\(A^{-1}\) 和对数导数可能不稳定。实现中应比较直接行列式求导与线性代数求解方案，并监控条件数、极端局部能量及梯度范数。

## 10. VMC 能量梯度

在目标分布

\[
P(X)=\frac{|\mathcal D(X)|^2}{Z_{\mathcal D}}
\]

下，标准复波函数 VMC 能量梯度为

\[
\boxed{
\frac{\partial E}{\partial\alpha}
=
2\operatorname{Re}\left[
\langle O_\alpha^*E_L\rangle_P
-\langle O_\alpha^*\rangle_P\langle E_L\rangle_P
\right]
}
\]

可使用重要性权重对这些期望值进行估计。

注意：上述式子描述的是对波函数参数的梯度，并假设期望值对应目标分布。若同时训练提议分布参数 \(\eta\)，必须另行设计提议网络的训练目标及梯度处理，不能把它直接视作普通波函数参数。

## 11. QGT / SR 优化

可定义联合波函数的几何矩阵

\[
S_{\alpha\beta}=
\langle O_\alpha^*O_\beta\rangle_P
-\langle O_\alpha^*\rangle_P
\langle O_\beta\rangle_P.
\]

采用带正则化的 SR 更新：

\[
(S+\epsilon I)\Delta\boldsymbol\alpha
=-\eta\nabla_{\boldsymbol\alpha}E.
\]

- \(S\)：联合波函数参数的 QGT/SR 矩阵。
- \(\epsilon\)：数值正则化参数。
- \(\eta\)：更新步长。
- \(\boldsymbol\alpha\)：可同时包含 NN 与 PQC 参数。

所有统计量都应使用与目标分布一致的样本或正确加权估计。QGT 的构建本身也有计算与测量成本，需要计入资源分析。

## 12. 建议的实验路线

先从已有验证结果较好的 H 原子 \(K=4\)、\(K=6\) 开始，再扩展到 Li 原子或 H₂。

### 基线与消融实验

1. **原始 NES-VMC：** 纯经典 NN。
2. **QE-NES-VMC：** NN–PQC 混合 Ansatz，保留原采样策略。
3. **QE-NES-VMC + 联合重要性采样：** 检验联合提议分布与加权采样的收益。
4. **QE-NES-VMC + 联合重要性采样 + 加权 SR：** 检验完整方案的优化效果。
5. **可选共享参数实验：** 比较独立 PQC 与共享 PQC 的资源和精度。

### 建议报告的指标

- 各态能量及相对于 FCI/高精度基准的误差。
- 是否达到化学精度，以及采用的误差定义。
- 能量方差与训练收敛曲线。
- 联合重要性采样 ESS、权重方差及极端权重比例。
- PQC 调用数、总 shots、振幅估计成本、经典计算时间和墙钟时间。
- 不同 \(K\) 下的参数量、内存开销与稳定性。
- 重复随机种子的均值和标准差。

## 13. 当前最重要的待确认事项

在将以上方案写成严格算法或论文方法章节之前，首先核对现有 NES-VMC 的实现：

1. `Psi` / `det` 的实际定义是什么？是否真的为
   \(\mathcal D(X)=\det[\psi_i(\mathbf x_j)]\)？
2. 原算法使用的扩展哈密顿量 \(\mathcal H_{\mathrm{ext}}\) 具体如何定义？
3. 原始联合采样器采样的是 \(|\mathcal D(X)|^2\)，还是其他等价或不同的分布？
4. NN–PQC 混合振幅 \(\psi_i(\mathbf x)\) 在模拟器和真实硬件上分别如何获取？
5. 原 NES-VMC 的梯度、QGT/SR 与规范自由度处理是否能直接沿用？

只有在这些定义与代码一致后，才能确认重要性权重、局部能量及梯度公式是否严格适用于当前实现。

## 14. 论文定位建议

可以将论文贡献暂时组织为：

- **混合多态 Ansatz：** 将经典 NN 与 PQC 结合到 NES-VMC 的多态联合结构中。
- **联合重要性采样：** 针对联合行列式目标分布设计自回归提议和重要性校正。
- **加权自然梯度优化：** 将联合分布下的加权梯度和 QGT/SR 纳入统一训练流程。
- **资源—精度评估：** 系统比较多激发态精度、量子测量预算和采样效率。

应避免在实验之前宣称该方法一定降低总测量次数或一定优于经典 NES-VMC。更稳妥的研究假设是：在表达能力、联合采样效率和量子资源之间，QE-NES-VMC 可能提供可量化的折衷。

---

## 参考文献与核查说明

- *Quantum-Enhanced Neural Networks for Quantum Many-Body Simulations*：作为 NN–PQC 混合波函数、自回归采样和重要性采样的相关工作。正式投稿前应核对论文的最终出版版本、公式与实现细节。
- NES-VMC 原始论文及当前实现：必须核实联合波函数、扩展哈密顿量、采样分布与梯度定义，避免把一般化推导误当成已经证明适用于所有 NES-VMC 变体的公式。
