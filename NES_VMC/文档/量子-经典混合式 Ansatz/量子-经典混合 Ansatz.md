你现在的条件非常特殊：
* 你做的是 **二次量子化表象**；
* VMC 的 configuration 是 occupation-number bitstring，例如
  $$
  x=(n_1,n_2,\ldots,n_{N_{\rm orb}}),\qquad n_i\in\{0,1\};
  $$
* 因此波函数就是
  $$
  \Psi_\theta(x)=\langle x|\Psi_\theta\rangle;
  $$
* 对固定 \(N_e\) 的电子体系，采样空间天然就是 Fock space；
* 你又正在做 **NES-VMC / 多激发态**，所以不仅要表达基态，还要同时表达一组相关的 \(\psi_0,\psi_1,\ldots\)；
* 最重要的是，你希望 PQC 不是“装饰”，而是承担 **经典 NN 难以表达的量子关联/相位结构**。

所以我建议不要把问题定义成：

> “怎样把 PQC + NN 拼起来？”

而应该定义成：

> **“在二次量子化 Fock-space VMC 中，如何让经典 NN 负责可采样性，而 PQC 负责量子关联结构？”**

这会形成一个非常清晰的研究方向。

---

# 1. 先把现在主流的 NQS Ansatz 梳理清楚

2024 年的 NQS review 对目前主要架构做了比较系统的分类，包括 FFNN、RBM、CNN、GNN、autoregressive、Transformer 和 fermionic architectures。近几年明显的趋势是从 RBM/FFNN 向 autoregressive、Transformer 和 fermionic neural networks 发展。([doi.org][1])

对于你的问题，我会把它们归成下面几类。

---

## 1.1 Complex FFNN

最简单：

$$
x\rightarrow {\rm FFNN}_\theta(x)
$$

输出：

$$
\Psi_\theta(x)=
\exp\left[
A_\theta(x)+iP_\theta(x)
\right].
$$

也可以写成：

$$
\Psi_\theta(x)
=
R_\theta(x)e^{i\phi_\theta(x)}.
$$

你现在的 **complex FFNN** 基本就在这个类别。

### 优点

非常灵活：

$$
\{0,1\}^{N_{\rm orb}}
\rightarrow \mathbb C.
$$

对于二次量子化尤其自然。

### 缺点

最大的问题不是表达能力本身，而是：

> NN 不知道 configuration 之间的量子结构。

例如：

$$
|1100\rangle,\quad |1010\rangle,\quad |1001\rangle
$$

在你的输入中只是三个 bitstring。

但 Hamiltonian 实际上知道：

$$
H_{ij}
=
\langle i|\hat H|j\rangle
$$

以及它们之间的 excitation relationship。

普通 FFNN 并没有显式利用这种结构。

---

# 2. RBM

经典 NQS 的代表：

$$
\Psi(x)
=
\sum_h
e^{-E(x,h)}.
$$

或者解析掉 hidden variables：

$$
\log\Psi(x)
=
\sum_i a_i x_i
+
\sum_j
\log
2\cosh
\left(
b_j+\sum_iW_{ij}x_i
\right).
$$

RBM 是 NQS 历史上非常重要的 Ansatz。

优点：

* 参数相对少；
* 对离散 configuration 很自然；
* 很容易构造 complex RBM；
* 很适合 lattice/Fock-space。

但对于你现在的小分子 NES-VMC，我不会把 RBM 作为最终主力，而会把它作为 **baseline**。
---


而二次量子化：

$$
|\Psi\rangle
=
\sum_x
\Psi(x)|x\rangle
$$

其中：

$$
|x\rangle
=
|n_1n_2\cdots n_M\rangle
=
a_1^{\dagger n_1}\cdots
a_M^{\dagger n_M}|0\rangle.
$$

**Fock basis 本身已经建立在 fermionic creation operators 的反对易关系之上。**

因此你的 NN 只需要学习：

$$
x\rightarrow \Psi(x)
$$

而不是再构造：

$$
\det[\phi_i(r_j)].
$$

这确实是你做 PQC+NN Ansatz 一个非常重要的优势。

但注意：

### 不是说“完全不需要任何物理结构”。

你仍然需要考虑：

$$
N_e=\sum_i n_i
$$

particle-number conservation，

以及：

* spin
* \(S_z\)
* point-group symmetry
* parity
* orbital symmetry

等等。

也就是说：

> **你不需要显式构造 antisymmetric determinant，但可以/应该利用 Fock-space symmetry。**

这反而给你留下了很大的 Ansatz 设计空间。

---

# 9. 现在来看你提到的那篇论文

你提到的：

**Quantum-enhanced neural networks for quantum many-body simulations**

已经在 2026 年正式发表在 **PRX Intelligence**。([APS Journals][2])

而且它实际上和你的想法非常接近。

它定义：

$$
\boxed{
\langle s|\Psi\rangle
=
\langle s|\phi(\theta)\rangle
\langle s|\varphi(\lambda)\rangle
}
$$

也就是：

$$
\boxed{
\Psi(x)
=
\Psi_{\rm quantum}(x)
\Psi_{\rm classical}(x)
}
$$

其中 classical 部分使用 autoregressive Transformer，而 quantum 部分使用 PQC。([ResearchGate][4])

他们进一步让 PQC 分别承担 amplitude / phase：

$$
\ln\langle s|\phi(\theta)\rangle
=
f[s;U_1(\theta_1)]
+
i f[s;U_2(\theta_2)].
$$

其中：

$$
f[s;U(\theta)]
=
\sum_i
c_i
\langle s|U^\dagger(\theta)Z_iU(\theta)|s\rangle.
$$

也就是说：

### Quantum branch

$$
x
\rightarrow
PQC
\rightarrow
f_Q(x)
$$

### Classical branch

$$
x
\rightarrow
Transformer
\rightarrow
f_{NN}(x)
$$

最后：

$$
\boxed{
\Psi(x)
=
\exp[f_{NN}(x)+f_Q(x)]
}
$$

或者从乘法角度：

$$
\Psi(x)=
\Psi_{NN}(x)\Psi_{PQC}(x).
$$

这是一个非常漂亮的结构。

---

# 10. 为什么这个结构有效？

我认为最重要的一点不是：

> “PQC 比 NN 更强。”

而是：

> **两种模型承担不同的 function space。**

经典 Transformer 擅长：

$$
\text{probability distribution}
$$

PQC 擅长：

$$
\text{quantum correlation}.
$$

因此：

$$
\Psi(x)
=
\underbrace{\Psi_{\rm NN}(x)}_{\text{classical global structure}}
\times
\underbrace{\Psi_{\rm PQC}(x)}_{\text{quantum correlation}}
$$

相当于：

$$
\log\Psi
=
\log\Psi_{\rm NN}
+
\log\Psi_{\rm PQC}.
$$

这实际上是一个 **learned additive decomposition of log-wavefunction**。

他们在 LiH 上报告了一个非常有意思的结果：一个约 435 参数的较小 classical model 加入 4-layer PQC 后，仅增加约 60 个参数，就达到了远低于 chemical accuracy 的能量误差，并且优于参数量大得多的纯 classical NQS。([alphaXiv][5])

这个结果对你的研究方向非常有启发。

---

# 11. 但我不建议你直接照搬这个结构

因为你的问题比论文更特殊：

$$
\boxed{\text{second quantization + NES-VMC}}
$$

而他们主要是在：

$$
\text{NQS + VMC}
$$

框架中做 hybrid ansatz。

所以我建议你进一步往下面几个方向发展。

---

# 12. 方案一：最直接的 PQC × Complex FFNN

这是我最建议你首先实现的 baseline。

你的当前：

$$
\Psi_{\rm NN}(x)
=
e^{A_\theta(x)+iP_\theta(x)}.
$$

增加：

$$
\Psi_{\rm PQC}(x)
=
e^{Q_\lambda(x)+iR_\lambda(x)}.
$$

最终：

$$
\boxed{
\Psi(x)
=
e^{
A_\theta(x)+Q_\lambda(x)
+
i[P_\theta(x)+R_\lambda(x)]
}
}
$$

也就是：

$$
\boxed{
\log\Psi(x)
=
\log\Psi_{\rm NN}(x)
+
\log\Psi_{\rm PQC}(x)
}
$$

### 结构

```text
                 ┌── Complex FFNN ──┐
occupation x ────┤                  ├── log Ψ
                 └────── PQC ───────┘
```

具体：

```text
x = |110010...>

       │
       ├──────────────► FFNN
       │                  │
       │                  ├── amplitude
       │                  └── phase
       │
       └──────────────► PQC
                          │
                          ├── quantum amplitude
                          └── quantum phase

                    ↓

              Ψ_NN(x) × Ψ_Q(x)
```

---

# 13. 但是这里有一个非常值得你做的改进

不要让 PQC 只学习：

$$
Q(x)
$$

而是让 NN **控制 PQC**。

即：

$$
\boxed{
\theta_{\rm PQC}
=
g_\phi(x)
}
$$

于是：

$$
U(\theta)
\rightarrow
U(g_\phi(x)).
$$

也就是说：

> **configuration-dependent PQC。**

例如：

$$
x
\rightarrow
NN_\phi(x)
\rightarrow
\theta_1(x),\ldots,\theta_L(x)
$$

然后：

$$
U(x)=
\prod_l U_l(\theta_l(x)).
$$

最终：

$$
\Psi(x)
=
\Psi_{\rm classical}(x)
\Psi_{\rm PQC}(x;\theta(x)).
$$

这和论文简单的：

$$
\Psi_{NN}(x)\Psi_{PQC}(x)
$$

相比，是一个明显不同的 Ansatz。

---

# 14. 方案二：NN-generated PQC

我非常推荐你研究这个。

叫它：

> **Neural-Generated Quantum Circuit Ansatz**

或者：

> **Configuration-Adaptive Quantum Neural Ansatz**

结构：

$$
x
\rightarrow
NN
\rightarrow
\{\theta_i(x)\}
\rightarrow
PQC
\rightarrow
Q(x)
$$

例如：

$$
\theta(x)=W h_\theta(x)+b.
$$

然后：

$$
Q(x)
=
\langle x|
U^\dagger(\theta(x))
Z
U(\theta(x))
|x\rangle.
$$

最终：

$$
\boxed{
\Psi(x)
=
\exp[
A_\theta(x)
+
Q_\phi(x)
+iP_\theta(x)
]
}
$$

---

## 为什么这个特别适合你的二次量子化？

因为：

$$
x=(n_1,n_2,\ldots,n_M)
$$

本身就是 occupation information。

例如：

$$
|110000\rangle
$$

和：

$$
|101000\rangle
$$

对应不同 electronic configurations。

NN 可以识别：

> “当前 configuration 是什么？”

然后动态改变 PQC。

这意味着：

$$
\boxed{
\text{different electronic configurations}
\rightarrow
\text{different quantum circuits}
}
$$

这是非常有意思的。

---

# 15. 方案三：PQC 只负责 phase

这个其实非常值得研究。

你现在已经观察过：

> complex FFNN 中 real-part / phase / gauge drift 等问题。

那么完全可以：

$$
\Psi(x)
=
R_\theta(x)
e^{iP_\theta(x)}
$$

改成：

$$
\boxed{
\Psi(x)
=
R_\theta(x)
e^{iP_\phi^{PQC}(x)}
}
$$

即：

### NN：

负责：

$$
\log|\Psi(x)|
$$

### PQC：

负责：

$$
\arg\Psi(x).
$$

---

为什么有意义？

因为对许多电子结构问题：

$$
|\Psi(x)|^2
$$

通常相对容易优化，

而真正困难的东西之一就是：

$$
\boxed{\text{sign / phase structure}}
$$

所以可以让：

$$
NN\rightarrow amplitude
$$

而：

$$
PQC\rightarrow phase.
$$

这会形成非常干净的科学问题：

> **Can a shallow PQC efficiently learn the fermionic sign/phase structure in second-quantized VMC?**

这个问题比单纯：

> PQC + NN 比 NN 好

更加有研究价值。

---

# 16. 方案四：PQC 负责“correlation correction”

这个我觉得尤其适合小分子。

可以先让 NN 学一个 baseline：

$$
\Psi_{\rm NN}(x)
$$

然后 PQC 学 correction：

$$
\Delta_{\rm Q}(x).
$$

定义：

$$
\boxed{
\Psi(x)
=
\Psi_{\rm NN}(x)
e^{\Delta_Q(x)}
}
$$

也就是：

$$
\log\Psi(x)
=
\log\Psi_{\rm NN}(x)
+
\Delta_Q(x).
$$

这里：

$$
\Delta_Q(x)
$$

可以被解释成：

> **quantum correlation correction**

这和 quantum chemistry 中：

$$
\Psi
=
\Psi_{\rm reference}
+
\text{correlation correction}
$$

的思想非常接近。

---

# 17. 甚至可以直接联系 Hartree-Fock

你的二次量子化非常适合这么做。

先定义：

$$
\Psi_{\rm HF}(x)
$$

然后：

$$
\boxed{
\Psi(x)
=
\Psi_{\rm HF}(x)
\exp[
F_{\rm NN}(x)+F_{\rm PQC}(x)
]
}
$$

即：

```text
HF reference
     │
     ▼
  baseline
     │
     ├──── NN correction
     │
     └──── PQC correction
              │
              ▼
          correlated Ψ
```

对于小分子：

$$
H_2,\ LiH,\ BeH_2,\ H_4,\ H_2O
$$

这个结构非常容易解释。

你可以问：

> PQC 到底是在学习什么？

答案可以变成：

$$
\boxed{\text{correlation correction beyond HF}}
$$

而不是模糊地说：

> PQC increases expressivity.

这在论文叙事上会强很多。

---

# 18. 方案五：Hamiltonian-aware PQC

这是我认为你最有潜力进一步做的方向之一。

你的二次量子化 Hamiltonian：

$$
\hat H
=
\sum_{pq}h_{pq}a_p^\dagger a_q
+
\frac14
\sum_{pqrs}
h_{pqrs}
a_p^\dagger a_q^\dagger a_sa_r.
$$

因此你可以利用：

$$
h_{pq}
$$

和：

$$
h_{pqrs}
$$

来决定 PQC 的 entangling structure。

而不是：

```text
PQC:
q0 ──Ry──■────
         │
q1 ──Ry──X────
```

固定拓扑。

而是：

$$
|h_{pq}|
$$

决定 orbital \(p,q\) 之间是否建立 entanglement。

例如：

$$
E_{pq}=f(|h_{pq}|)
$$

然后：

$$
E_{pq}>\tau
$$

才连接：

$$
q_p\leftrightarrow q_q.
$$

于是：

$$
\boxed{
Hamiltonian
\rightarrow
orbital\ graph
\rightarrow
PQC\ topology
}
$$

这就变成：

> **Hamiltonian-aware quantum neural Ansatz**

而且特别适合小分子。

---

# 19. 方案六：Graph NN + PQC

进一步可以：

$$
G_H
=
(V,E,H)
$$

其中：

$$
V=\{\text{orbitals}\}
$$

边：

$$
E_{pq}=h_{pq}
$$

或者二体：

$$
E_{pqrs}=h_{pqrs}.
$$

然后：

$$
G_H
\rightarrow
GNN
\rightarrow
\theta_{\rm PQC}.
$$

即：

$$
\boxed{
H
\rightarrow GNN
\rightarrow PQC
\rightarrow \Psi
}
$$

这会让你的 Ansatz 不再只是：

> NN + quantum circuit

而是：

> **electronic Hamiltonian → learned quantum architecture → VMC wavefunction**

这已经是一个相当完整的研究方向。

---

# 20. 对你的 NES-VMC，还有一个更有意思的方案

这是我认为你最应该考虑的。

你不是只有一个：

$$
\Psi(x)
$$

而是：

$$
\Psi_0(x),\Psi_1(x),\ldots,\Psi_{K-1}(x).
$$

那么不要给每一个 state 完全独立的 NN + PQC。

而是：

$$
\boxed{
\Psi_k(x)
=
\Psi_{\rm shared}(x)
\Psi_k^{\rm state}(x)
}
$$

例如：

```text
                  shared encoder
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
             ψ0       ψ1       ψ2
              │        │        │
            PQC0      PQC1     PQC2
```

更加具体：

$$
h(x)=Encoder(x)
$$

然后：

$$
\theta_k(x)
=
W_k h(x)+b_k.
$$

最终：

$$
\Psi_k(x)
=
\Psi_{\rm shared}(x)
\exp[
F_k^{NN}(x)
+
F_k^{PQC}(x)
].
$$

---

# 21. 这样非常适合你现在的“冻结”研究

因为你前面正在研究：

> NES-VMC 的分段冻结。

如果：

$$
\Psi_0
$$

已经达到 chemical accuracy，

你可以冻结：

$$
\theta_0^{NN}
$$

以及：

$$
\theta_0^{PQC}.
$$

然后继续优化：

$$
\Psi_1,\Psi_2,\Psi_3.
$$

如果 shared encoder 也冻结，则：

$$
K=4
$$

变成：

$$
K=3
$$

的时候，不仅仅是：

$$
V:4\times4\rightarrow3\times3
$$

的问题。

你甚至可以研究：

$$
\boxed{
\text{frozen quantum branches}
}
$$

对 NES-VMC 的 computational scaling 有什么影响。

这和普通 NQS + PQC 的工作相比，是一个明显不同的研究问题。

---

# 22. 我建议你把几种 Ansatz 放在一个统一公式下面

其实可以定义：

$$
\boxed{
\log\Psi_k(x)
=
F_{\rm NN}(x)
+
F_{\rm PQC}(x)
+
F_k(x)
}
$$

其中：

### Shared classical component

$$
F_{\rm NN}(x)
$$

学习所有 states 的共同结构。

### Quantum component

$$
F_{\rm PQC}(x)
$$

学习量子关联。

### State-specific component

$$
F_k(x)
$$

负责区分：

$$
\Psi_0,\Psi_1,\Psi_2,\ldots
$$

这会非常适合你的 NES-VMC。

---

# 23. 你还可以进一步让 PQC 变成 state-dependent

即：

$$
U_k(\theta_k)
$$

变成：

$$
U(\theta,z_k)
$$

其中：

$$
z_k
$$

是 state embedding。

例如：

$$
z_0,z_1,z_2,z_3.
$$

于是：

$$
\boxed{
\Psi_k(x)
=
\Psi_{\rm NN}(x)
\Psi_{\rm PQC}(x,z_k)
}
$$

这样就可以：

$$
\text{one shared NN}
+
\text{one shared PQC}
+
\text{state embedding}
$$

而不是：

$$
K
$$

个完全独立的网络。

这对于 \(K\) 较大的 NES-VMC 很有意义。

---

# 24. 我把这些方案按研究价值重新整理

不是排名，而是不同研究路线：

| Ansatz                       | NN                       | PQC                    | 主要作用                                        | 对你适合程度 |
| ---------------------------- | ------------------------ | ---------------------- | ------------------------------------------- | ------ |
| Complex FFNN                 | amplitude+phase          | 无                      | baseline                                    | ★★★★★  |
| NN × PQC                     | amplitude+phase          | amplitude+phase        | 增强表达能力                                      | ★★★★★  |
| NN + PQC-phase               | amplitude                | phase                  | 学 sign/phase                                | ★★★★★  |
| NN × PQC-correction          | baseline                 | correlation correction | 类似 correlation recovery                     | ★★★★★  |
| NN → PQC                     | generates PQC parameters | adaptive               | configuration-dependent quantum correlation | ★★★★★  |
| GNN → PQC                    | Hamiltonian graph        | adaptive               | Hamiltonian-aware                           | ★★★★★  |
| Transformer × PQC            | sampling                 | correlation            | 接近已有工作                                      | ★★★★☆  |
| Shared NN + state PQC        | shared                   | state-specific         | NES-VMC                                     | ★★★★★  |
| Shared PQC + state embedding | shared                   | conditional            | scalable NES                                | ★★★★★  |

---

# 25. 那篇 2026 论文之后，还有哪些相关方向？

目前相关工作已经不只是“PQC + NN wavefunction”。

例如 2026 年还有一篇 **Polynomially efficient quantum enabled variational Monte Carlo for training neural-network quantum states for physico-chemical applications**，讨论的是用 quantum-enabled 方法帮助训练 NQS，而不是简单把 PQC 当 wavefunction factor。([Nature][6])

还有 **Quantum-Assisted Variational Monte Carlo**，其思路是使用 quantum-enhanced Metropolis-Hastings 来帮助采样 NQS 的概率分布。这个方向和你的工作有一个重要区别：

$$
\boxed{
\text{PQC as wavefunction}
}
$$

vs.

$$
\boxed{
\text{PQC as sampler}
}
$$

前者改变 Ansatz，后者改变 sampling。

([PubMed Central (PMC)][7])

这其实给你提供了一个非常大的研究空间：

```text
                 Quantum Computer
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
      Wavefunction   Sampling     Optimization
         PQC           PQC           PQC
          │            │            │
          └────────────┼────────────┘
                       ▼
                      VMC
```

你的研究可以明确选择第一条。

---

# 26. 我最建议你不要做的事情

我反而建议你暂时不要做：

$$
NN \rightarrow PQC \rightarrow NN \rightarrow PQC
$$

这种非常深的 hybrid architecture。

因为这样很容易变成：

> “我们把两个模型堆起来，然后效果变好了。”

科学解释不够清楚。

你真正应该回答的是：

$$
\boxed{
\text{NN 和 PQC 分别解决什么问题？}
}
$$

---

# 27. 我认为你最值得做的三个实验

如果是我帮你设计这条研究路线，我会首先做：

### Experiment A：Amplitude–Phase decomposition

比较：

$$
\Psi_{\rm FFNN}
$$

vs.

$$
\Psi_{\rm FFNN+PQC}
$$

vs.

$$
\boxed{
|\Psi|_{\rm NN}
\times
e^{i\phi_{\rm PQC}}
}
$$

测试：

* H₂
* LiH
* BeH₂
* H₄

比较：

$$
E-E_{\rm FCI}
$$

以及：

$$
N_{\rm parameters}
$$

和：

$$
N_{\rm PQC\ layers}.
$$

这个实验非常容易讲清楚。

---

# 28. Experiment B：Correlation-correction Ansatz

定义：

$$
\boxed{
\Psi=
\Psi_{\rm NN}
e^{F_{\rm PQC}}
}
$$

然后研究：

$$
\Delta E
=
E_{\rm NN}-E_{\rm hybrid}
$$

随着：

$$
L_{\rm PQC}=1,2,3,4
$$

怎么变化。

最重要的是做：

$$
\frac{\Delta E}{N_{\rm PQC}}
$$

这种指标。

如果一个 PQC layer 能替代大量 NN 参数，这就是很漂亮的结果。

论文中 LiH 的结果已经给出了一个非常有启发性的现象：少量 PQC 参数可以显著改善较小 classical NQS 的能量精度。([alphaXiv][5])

---

# 29. Experiment C：你真正可能形成自己方法的地方

做：

$$
\boxed{
H
\rightarrow
GNN
\rightarrow
PQC
\rightarrow
\Psi
}
$$

也就是：

### Hamiltonian-aware PQC

对每个 molecule：

$$
h_{pq},h_{pqrs}
$$

生成 orbital graph。

然后：

$$
GNN(H)
\rightarrow
\theta_{\rm PQC}.
$$

比较：

```text
Random PQC
       ↓
Fixed hardware-efficient PQC
       ↓
Hamiltonian-aware PQC
```

注意这里不是评价谁“最好”，而是研究：

> **Hamiltonian information 是否可以降低达到目标精度所需的 PQC depth / parameter count？**

这个问题就非常明确了。

---

# 30. 最后，我认为你真正可以形成的核心 Ansatz

结合你现在的 **second-quantized NES-VMC + PQC + NN + small molecules**，我最建议你最终考虑：

$$
\boxed{
\Psi_k(x)
=
\Psi_{\rm NN}(x)
\exp
\left[
F_k^{\rm PQC}(x)
\right]
}
$$

进一步：

$$
\boxed{
F_k^{\rm PQC}(x)
=
F_{\rm PQC}(x;z_k,H)
}
$$

于是：

$$
\boxed{
\Psi_k(x)
=
\Psi_{\rm NN}(x)
\exp
\left[
F_{\rm PQC}(x;z_k,H)
\right]
}
$$

其中：

* \(x\)：Fock-space occupation configuration；
* \(\Psi_{\rm NN}\)：负责 global amplitude / probability；
* \(H\)：提供 molecular Hamiltonian information；
* \(z_k\)：第 \(k\) 个 excited state 的 embedding；
* PQC：负责 quantum correlation / phase / state-specific correction。

整个结构：

```text
                 Second-quantized Hamiltonian H
                              │
                              ▼
                       Orbital / Hamiltonian
                             Graph
                              │
                              ▼
                            GNN
                              │
                              ▼
                     PQC parameters θ(H,z_k)
                              │
                              ▼
                    ┌─────── PQC ───────┐
                    │                   │
occupation x ───────┤                   │
                    ▼                   ▼
                NN branch          Quantum branch
                    │                   │
              log |ΨNN(x)|        F_PQC(x;H,k)
                    │                   │
                    └─────────┬─────────┘
                              ▼
                   Ψ_k(x)=ΨNN(x)e^F_PQC
                              │
                              ▼
                         NES-VMC
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                   M,S                  V
                    │                   │
                    └───────► generalized
                              eigenproblem
```

这已经不是简单的 **PQC + NN** 了，而是一个：

> **Hamiltonian-aware, second-quantized, quantum-classical neural wavefunction for NES-VMC**

的完整框架。

而且它和你现在正在做的 **NES-VMC 分段冻结** 可以自然接起来：当 \(\Psi_k\) 收敛后，不只是冻结 NN 参数，还可以冻结对应的 \(z_k\)、PQC 参数乃至 quantum branch，从而形成 **state-wise quantum/classical parameter freezing**。

---

## 我尤其建议你注意一个理论上的“卖点”

你现在最大的优势并不是：

> “二次量子化不用 determinant，所以网络可以更自由。”

真正更有价值的表述是：

$$
\boxed{
\text{Fock-space representation removes the need for explicit antisymmetrization}
}
$$

因此你可以把有限的模型容量用来学习：

$$
\boxed{
\text{correlation + amplitude + phase + excited-state structure}
}
$$

而不是花网络结构去恢复已经由 fermionic Fock basis 编码进去的交换反对称性。

这就给 **PQC** 留出了一个非常自然的位置：

$$
\boxed{
\text{Classical NN}
\rightarrow
\text{efficient representation/sampling}
}
$$

$$
\boxed{
\text{PQC}
\rightarrow
\text{quantum correlation / phase structure}
}
$$

这比“把量子电路塞进神经网络”要有明确得多的物理解释。

另外，2024 NQS review 也明确指出，目前并不存在一种对所有物理问题都显然最优的 NQS architecture；不同架构各有优势，因此针对具体物理表示设计 hybrid Ansatz 本身就是开放问题。([doi.org][1])

**如果你下一步要真正开始写代码，我建议先不要上 GNN。第一版直接在你现在的 `complex FFNN + NES-VMC` 上加入一个 `PQC branch`，实现这三个 ablation：**

$$
\boxed{
\text{FFNN}
}
$$

$$
\boxed{
\text{FFNN}\times\text{PQC}
}
$$

$$
\boxed{
\text{Amplitude-NN}\times\text{Phase-PQC}
}
$$

先在 **H₂ STO-3G → LiH STO-3G → H₄** 上做参数量、energy error、PQC depth 和训练稳定性对比。这样最容易判断 PQC 到底是在改善 amplitude、phase，还是单纯增加了模型容量；之后再把它升级成 **Hamiltonian-aware / state-conditioned PQC**，这一步才真正有机会形成你自己的方法。

[1]: https://doi.org/10.1088/2058-9565/ad7168?urlappend=%3Futm_source%3Dresearchgate.net%26utm_medium%3Darticle&utm_source=chatgpt.com "From architectures to applications: a review of neural quantum states - IOPscience"
[2]: https://journals.aps.org/prxintelligence/abstract/10.1103/2jpn-jh3x?utm_source=chatgpt.com "Quantum-Enhanced Neural Networks for Quantum Many-Body Simulations | PRX Intelligence"
[3]: https://github.com/google-deepmind/ferminet?utm_source=chatgpt.com "GitHub - google-deepmind/ferminet: An implementation of the Fermionic Neural Network for ab-initio electronic structure calculations · GitHub"
[4]: https://www.researchgate.net/publication/410909368_Quantum-Enhanced_Neural_Networks_for_Quantum_Many-Body_Simulations?utm_source=chatgpt.com "(PDF) Quantum-Enhanced Neural Networks for Quantum Many-Body Simulations"
[5]: https://www.alphaxiv.org/overview/2501.12130?utm_source=chatgpt.com "Quantum-enhanced neural networks for quantum many-body simulations | alphaXiv"
[6]: https://www.nature.com/articles/s41534-026-01230-1?utm_source=chatgpt.com "Polynomially efficient quantum enabled variational Monte Carlo for training neural-network quantum states for physico-chemical applications | npj Quantum Information"
[7]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12458056/?utm_source=chatgpt.com "Quantum-Assisted Variational Monte Carlo - PMC"
