# NES-VMC Progressive Freezing 方案总结

## 1. 研究动机

NES-VMC 同时优化多个低能态/激发态时，不同能级的收敛速度不同。实际训练中，某些能级已经达到目标精度，但继续进行高轮次随机优化后，可能出现能量漂移甚至突然劣化。

可能原因包括：
- MCMC sampling noise；
- 统计估计得到的 M、S 随采样变化；
- generalized eigenvectors V 随 M、S 波动；
- near-degenerate states 的 rotation / state ordering fluctuation；
- QGT / natural-gradient 数值不稳定；
- 高轮次随机更新破坏已经收敛的 state。

因此提出 **Progressive Freezing（分段冻结）**：

> 当某个物理 Ritz state φ_i 达到收敛标准后，在当前时刻将其固定下来，使后续随机优化不再破坏该已收敛解。

第一阶段的主要目标不是降低问题规模，而是验证：

\[
\boxed{\text{Freezing 能否提高 NES-VMC 长时间训练的稳定性}}
\]

---

## 2. NES-VMC 中两个不同的 Ψ

设有 K 个 neural states：

\[
\Psi_{col}=[\psi_0,\psi_1,\ldots,\psi_{K-1}].
\]

### 2.1 MCMC 中使用的 determinant

给定 K 个 configuration：

\[
X=(x_0,\ldots,x_{K-1}),
\]

构造：

\[
\mathcal D(X)=\det
\begin{pmatrix}
\psi_0(x_0)&\cdots&\psi_{K-1}(x_0)\\
\vdots&&\vdots\\
\psi_0(x_{K-1})&\cdots&\psi_{K-1}(x_{K-1})
\end{pmatrix}.
\]

它用于构造 MCMC 采样分布，例如：

\[
p(X)\propto|\mathcal D(X)|^2.
\]

### 2.2 Local energy 中的波函数矩阵

计算 local-energy matrix 时使用：

\[
\mathbf\Psi(X)=
\begin{pmatrix}
\psi_0(x_0)&\cdots&\psi_{K-1}(x_0)\\
\vdots&&\vdots\\
\psi_0(x_{K-1})&\cdots&\psi_{K-1}(x_{K-1})
\end{pmatrix},
\]

并计算：

\[
E_L(X)=\mathbf\Psi(X)^{-1}H\mathbf\Psi(X).
\]

因此：

\[
\boxed{\mathcal D(X)=\det(\mathbf\Psi(X))}
\]

和

\[
\boxed{E_L(X)=\mathbf\Psi^{-1}H\mathbf\Psi}
\]

是不同层次的对象。冻结 determinant 中的一列并不自动等价于冻结对应 physical state。

---

## 3. Generalized eigenvalue problem

每轮 MCMC 后得到：

\[
M^{(t)},\quad S^{(t)}
\]

并求：

\[
\boxed{M^{(t)}v_i^{(t)}=\lambda_i^{(t)}S^{(t)}v_i^{(t)}}.
\]

组成：

\[
V^{(t)}=[v_0^{(t)},v_1^{(t)},\ldots,v_{K-1}^{(t)}].
\]

对应 Ritz state：

\[
\boxed{
\phi_i^{(t)}(x)=
\sum_{j=0}^{K-1}
\psi_j(x;\theta_j^{(t)})V_{ji}^{(t)}
}
\]

其中 v_i 是 K 维 coefficient vector，而 φ_i 是 Hilbert space 中的 wavefunction。

---

## 4. 为什么不能简单冻结原始 ψ_j

一般：

\[
\phi_i=\sum_jV_{ji}\psi_j.
\]

所以：

\[
\boxed{\text{冻结 }\psi_j\neq\text{冻结 }\phi_i}
\]

例如：

\[
\phi_0=0.3\psi_0+0.5\psi_1+0.4\psi_2+0.6\psi_3.
\]

如果只冻结 ψ_0，而其他网络继续训练，则 φ_0 仍会变化。

因此冻结对象应该是完整的 physical Ritz state φ_f。

---

## 5. Freeze 时刻

假设在训练第 t_f 轮：

\[
\phi_f=\phi_i^{(t_f)}
\]

满足收敛标准，例如：

\[
|\lambda_i-E_i^{target}|<\epsilon
\]

或连续若干轮：

\[
|\lambda_i^{(t)}-\lambda_i^{(t-1)}|<\epsilon.
\]

实际项目中应使用明确的、多轮稳定的 stopping criterion，而不是单次采样结果。

---

## 6. Snapshot 内容

冻结时建议同时保存：

\[
\boxed{v_f=v_i^{(t_f)}}
\]

以及构成该 Ritz state 的原始网络参数：

\[
\boxed{
\theta_0^f,\theta_1^f,\ldots,\theta_{K-1}^f
}
\]

于是：

\[
\boxed{
\phi_f(x)=
\sum_{j=0}^{K-1}
V_{jf}^{(t_f)}
\psi_j(x;\theta_j^f)
}
\]

以后重构 frozen state 时始终使用 snapshot 参数和 snapshot coefficient。

---

## 7. 后续训练中的 Frozen State

对于 t > t_f：

\[
\phi_f(x)=
\sum_jV_{jf}^{(t_f)}
\psi_j(x;\theta_j^{(t_f)})
\]

保持不变。

即：

\[
\boxed{\phi_f^{(t)}\equiv\phi_f^{(t_f)}}.
\]

新 MCMC 得到的新 V[:,i] 不应覆盖 frozen coefficient。

---

## 8. M、S、V 是否缩小？

对于第一阶段的稳定性实验，推荐：

\[
\boxed{M,S\text{ 继续保持 }K\times K}
\]

例如 K=4：

\[
M,S\in\mathbb C^{4\times4}.
\]

原因是 physical NES-VMC subspace 没有改变。冻结首先改变的是 optimization freedom，而不是 physical subspace dimension。

因此不建议第一版直接：

\[
4\times4\rightarrow3\times3
\]

删除 M、S 的一行一列。

---

## 9. V 的冻结方式

原始：

\[
V=[v_0,v_1,v_2,v_3].
\]

若 φ_0 收敛：

\[
v_0^f=v_0^{(t_f)}.
\]

后续概念上：

\[
\boxed{
V^{(t)}=
[v_0^f,v_1^{(t)},v_2^{(t)},v_3^{(t)}]
}
\]

其中 v_0^f 固定。

严格来说，此时 V 不再是新 M^(t)、S^(t) 的完整 generalized eigenvector matrix，因为一般：

\[
M^{(t)}v_0^f\neq\lambda_0S^{(t)}v_0^f.
\]

这不是 bug，而是冻结机制的定义结果。冻结后的 v_f 应被理解为 frozen Ritz state 的 coefficient representation，而不是“当前 M、S 的 eigenvector”。

---

## 10. “3×3 active V”与“4×4 physical V”

冻结 φ_0 后，可以把状态空间概念上分成：

\[
\text{Frozen space}\oplus\text{Active space}
\]

其中：

\[
\dim(\text{Frozen})=1,\qquad
\dim(\text{Active})=3.
\]

因此可以定义 active-state rotation：

\[
V_A\in\mathbb C^{3\times3}.
\]

但整体 physical basis 仍有 4 个 state：

\[
V_{total}=[v_f,V_A].
\]

第一版实现建议仍保留：

\[
M,S,V\sim4\times4
\]

只冻结 V 的指定列。

---

## 11. 为什么第一阶段不建议删除 M、S 的一行一列

M、S 描述当前 K-dimensional NES-VMC subspace 的统计结构。

冻结机制首先改变：

\[
\boxed{\text{optimization freedom}}
\]

而不是：

\[
\boxed{\text{physical subspace dimension}}.
\]

所以第一阶段：

\[
M,S:4\times4
\]

保持不变。

---

## 12. 与 penalty method 的区别

该方案不需要修改 Hamiltonian：

\[
H_{new}=H.
\]

也不需要：

\[
H+\beta P
\]

形式的 penalty，也不需要在 loss 中增加：

\[
\beta|\langle\psi|\phi_f\rangle|^2.
\]

核心只是：

\[
\boxed{
\theta_f\leftarrow snapshot(\theta_f)
}
\]

和：

\[
\boxed{
v_f\leftarrow snapshot(v_f)
}
\]

并继续保留原 NES-VMC determinant structure。

---

## 13. 大体系下的意义

该方案不需要构造 dense projector：

\[
P_\perp=I-|\phi_f\rangle\langle\phi_f|
\]

也不需要：

\[
P_\perp HP_\perp.
\]

Hamiltonian H 保持原样，原有 operator-action / local-energy machinery 可以继续使用。

因此它比显式 projector / penalty deflation 更适合进一步研究大 Hilbert space。

---

## 14. 第一阶段真正可能降低的计算量

第一阶段不一定降低 M、S 的尺寸，也不一定降低 determinant 的 physical dimension。

真正可以降低的是：

\[
\boxed{\text{可训练参数空间}}
\]

设：

\[
\theta=(\theta_{frozen},\theta_{active})
\]

冻结后：

\[
\delta\theta_{frozen}=0.
\]

只针对 active parameters 做 optimization：

\[
\delta\theta_{active}\neq0.
\]

如果后续实现 active-only QGT，则 QGT 可以从完整参数空间缩小到 active parameter block。

---

## 15. 推荐的 K=4 数据流

```text
Original NES-VMC
│
├── ψ0(θ0)
├── ψ1(θ1)
├── ψ2(θ2)
└── ψ3(θ3)
       │
       ▼
   MCMC sampling
       │
       ▼
   M, S : 4 × 4
       │
       ▼
   M v = λ S v
       │
       ▼
   V : 4 × 4
       │
       ├── φ0
       ├── φ1
       ├── φ2
       └── φ3
```

当 φ0 收敛：

```text
Freeze φ0
│
├── snapshot V[:,0]
├── snapshot θ0
├── snapshot θ1
├── snapshot θ2
└── snapshot θ3
```

得到：

```text
frozen_state:
    v_f
    θ0_f
    θ1_f
    θ2_f
    θ3_f
```

以后：

```text
ψ0(θ0_f) ─┐
ψ1(θ1_f) ─┤
ψ2(θ2_f) ─┼──> φ_frozen
ψ3(θ3_f) ─┘
```

而 active states 继续优化。

---

## 16. 推荐的第一版实验

对于 H₂ / K=4：

\[
M,S\in\mathbb C^{4\times4}
\]

\[
V\in\mathbb C^{4\times4}.
\]

当某个 level 收敛：

1. 保存 V[:,i]；
2. 保存所有构成该 Ritz state 的 neural-network parameters；
3. 保存 freeze 时的 energy / overlap / convergence information；
4. 后续新 V[:,i] 不覆盖 frozen vector；
5. frozen state 始终使用 snapshot parameters + snapshot coefficients；
6. M、S 继续按原算法更新；
7. active states 继续训练。

首先比较：

### Baseline

\[
\text{NES-VMC}
\]

持续训练。

### Frozen

\[
\text{NES-VMC + Progressive Freezing}.
\]

重点比较：

- 已收敛能级的长期能量漂移；
- 高轮次突然劣化是否减少；
- generalized eigenvalue fluctuation；
- state overlap fluctuation；
- MCMC variance；
- QGT condition number；
- natural-gradient update stability；
- frozen state 是否保持 chemical accuracy。

---

## 17. 当前算法的核心定义

> **NES-VMC Progressive Freezing：**
>
> 在 NES-VMC 训练过程中，当 generalized eigenproblem 得到的某个 Ritz state φ_i 满足预定收敛标准时，在当前时刻同时 snapshot 其 coefficient vector v_i 和构成该 Ritz state 的 neural-network parameters {θ_j}。此后该 state 不再接受新的 stochastic updates，而由 snapshot 参数和 coefficient vector 固定重构为 φ_i^frozen。原 Hamiltonian、determinant-based sampling structure 以及 physical NES-VMC subspace 维度保持不变。第一阶段主要研究该机制对长期训练稳定性的影响，而不是立即缩小 M、S 的矩阵维度。

---

## 18. 当前最重要的数学注意事项

冻结后必须区分：

### Frozen representation

\[
\boxed{
\phi_f(x)=
\sum_jv_{jf}^{*}\psi_j(x;\theta_j^{*})
}
\]

这是固定的 physical state。

### Current generalized eigensystem

\[
\boxed{
M^{(t)}V^{(t)}
=
S^{(t)}V^{(t)}\Lambda^{(t)}
}
\]

这是每轮根据新 MCMC 数据得到的当前 generalized eigensystem。

二者不能混为一谈。

---

## 19. 研究路线

建议分三阶段：

### Phase I：稳定性

保持：

\[
M,S,V=4\times4
\]

只冻结已收敛 state。

目标：

\[
\boxed{\text{证明 freeze 可以抑制高轮次 degradation}}
\]

### Phase II：Active-space optimization

保持 physical dimension = 4，但只优化 active parameters。

目标：

\[
\boxed{\text{降低 QGT / optimization dimension}}
\]

### Phase III：Progressive dimensional reduction

进一步研究：

\[
K=4
\rightarrow
1+3
\rightarrow
2+2
\rightarrow\cdots
\]

如何在不破坏 NES-VMC determinant 核心结构的情况下，真正降低 active problem size。

---

## 20. 一句话总结

当前最稳妥的定义是：

\[
\boxed{
\text{不删除 physical state，不修改 H，不加 penalty；}
}
\]

\[
\boxed{
\text{在收敛时 snapshot }(V[:,i],\theta_0,\ldots,\theta_{K-1}),
\text{ 将对应 }\phi_i\text{ 作为 frozen state。}
}
\]

第一阶段保持：

\[
\boxed{
M,S,V\text{ 的物理维度仍为 }4\times4
}
\]

只让指定的 \(V\) 列和对应 frozen state 不再接受新的随机优化结果。
