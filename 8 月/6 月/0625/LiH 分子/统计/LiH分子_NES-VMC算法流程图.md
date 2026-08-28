# LiH_molecule_K4_STO-3G 算法流程图（NES-VMC 联合视角）

> 配套代码：
> - [LiH_molecule_K4_STO-3G.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH_molecule_K4_STO-3G.ipynb)  ← 本文档对象
> - [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py)  ← 算法实现
> - [LiH.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py)  ← 分子+Hilbert+FCI 构造
>
> 实测规模：LiH / STO-3G / CAS(2,2) / K=4 / 100 iter
> - 算例 FCI 基态 = -7.88259092 Ha
> - 1st = -7.76563664 Ha, 2nd = -7.74858736 Ha, 3rd = -7.71601970 Ha
> - 理论 Loss 上限 = $\sum_{k=0}^{3} E_k^\text{FCI}$ = -31.11283462 Ha

---

## 1. 总体三阶段：导入 → 训练 → 后处理

```mermaid
flowchart TD
    A["📒 LiH_molecule_K4_STO-3G.ipynb"] --> B["🔧 阶段 1：构造<br/>import & init<br/>(cell 1-5)"]
    B --> C["🏋️ 阶段 2：训练<br/>train loop<br/>(cell 5 主循环)"]
    C --> D["💾 阶段 3：后处理<br/>save & plot<br/>(cell 6+)"]

    B --> B1["Cell 1: 导入 NES_VMC & LiH"]
    B --> B2["Cell 2: 加载 LiH 数据<br/>(SINGLE_SIZE, ha, hi, hi_ext,<br/>ext_edges, K, Hatree_Fock)"]
    B --> B3["Cell 3: logger 配置"]
    B --> B4["Cell 4: nes_rule = NESFermionHopRule(...)"]
    B --> B5["Cell 5: 模型 + 采样器 + 优化器初始化"]

    C --> C1["for step in 0..N_ITER-1<br/>(N_ITER=100)"]
    C1 --> C2["采样 → grad → QGT → clip → apply"]
    C2 --> C1

    D --> D1["Cell save: pickle history"]
    D --> D2["Cell load: pickle history"]
    D --> D3["Cell plot: matplotlib 6 子图"]
    D --> D4["Cell plot: 2 子图对比"]
    D --> D5["Cell plot: 单态能量曲线"]

    classDef stage fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000
    classDef cell fill:#fff3e0,stroke:#f57c00,color:#000
    classDef loop fill:#f3e5f5,stroke:#7b1fa2,color:#000
    class A,B,C,D stage
    class B1,B2,B3,B4,B5,D1,D2,D3,D4,D5 cell
    class C1,C2 loop
```

---

## 2. Cell 1-4：导入与采样规则

```mermaid
flowchart LR
    subgraph "Cell 1 — import"
        I1["from NES_VMC import:<br/>NESTotalAnsatz, create_machine,<br/>SingleStateAnsatz, create_single_machine,<br/>create_machine_matrix,<br/>Ham_psi, Ham_Psi, NES_loss_energy,<br/>nes_vmc_gradient,<br/>NESFermionHopRule, compute_qgt,<br/>sampler_info,<br/>create_machine_matrix_stable,<br/>create_single_machine_gauge_fixed,<br/>create_machine_max_stable,<br/>NESTotalAnsatz_stable,<br/>create_machine_stable,<br/>NES_loss_energy_stable,<br/>nes_vmc_gradient_stable,<br/>make_grad_fn, make_qgt_fn"]
        I2["from LiH import:<br/>SINGLE_SIZE, ha, hi, hi_ext,<br/>ext_edges, single_edges_full, K,<br/>Hatree_Fock"]
        I3["import optax, jax, jax.numpy,<br/>flax.nnx, netket,<br/>netket.experimental,<br/>scipy.sparse.linalg.eigsh,<br/>jax.flatten_util.ravel_pytree"]
    end

    subgraph "Cell 2 — LiH 加载"
        L1["hi: SpinOrbitalFermions<br/>n_orbitals=4, s=1/2<br/>n_fermions_per_spin=(2,2)"]
        L2["hi_ext = hi ** 4<br/>size = 32"]
        L3["ha: ParticleNumberAndSpinConserving<br/>Fermioperator2nd"]
        L4["ext_edges: 4×12=48 条边<br/>(K 副本 × α/β 内部两两交换)"]
        L5["Hatree_Fock: hi.all_states()[0]<br/>= [1,1,0,0, 1,1,0,0]"]
        L6["FCI 6 条能量<br/>E0=-7.8826, E3=-7.7160 ..."]
    end

    subgraph "Cell 3-4 — 采样规则"
        R1["nes_rule = NESFermionHopRule(<br/>edges=ext_edges,<br/>K=4, single_size=8)"]
    end

    I1 --> R1
    I2 --> R1
    L4 --> R1
    L3 --> TrainInit
    L5 --> TrainInit

    classDef importBox fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef dataBox fill:#fff8e1,stroke:#ff8f00,color:#000
    classDef ruleBox fill:#fce4ec,stroke:#c2185b,color:#000
    class I1,I2,I3 importBox
    class L1,L2,L3,L4,L5,L6 dataBox
    class R1 ruleBox
```

---

## 3. Cell 5：模型 + 采样器 + 优化器初始化（详细调用链）

```mermaid
flowchart TD
    subgraph HYPER["超参"]
        H1["N_CHAINS = 16"]
        H2["N_SAMPLES_PER_CHAIN = 200"]
        H3["SWEEP_SIZE = 30"]
        H4["N_ITER = 100"]
        H5["Natural_Grad = True"]
        H6["clip_norm = 20.0"]
        H7["lr = 0.1"]
        H8["qgt_diag_shift = 0.1"]
    end

    subgraph MODEL["模型构建"]
        M1["NESTotalAnsatz_stable(<br/>n_spin_orbitals=8, n_states=4,<br/>hidden_dim=12,<br/>rngs=nnx.Rngs(11))"]
        M2["create_gauge_fixed_total_machines(<br/>total_ansatz, Hatree_Fock)"]
        M3["内部: nnx.split → graphdef, state"]
        M4["内部: _compute_L_centered_single<br/>L[i,j] = logψ_j(x_i) - logψ_j(ref)"]
        M5["内部: _stable_from_L<br/>shift = stop_grad(max Re L)<br/>L_stable = L - shift"]
        M6["返回 5 件套:<br/>total_machine (logΨ_gauge)<br/>total_matrix_machine (L_stable)<br/>total_max_machine (shift)<br/>total_graphdef<br/>total_params"]
        M7["single_machine_list = [<br/>create_single_machine_gauge_fixed(ans, ref)[0]<br/>for ans in total_ansatz.single_ansatz_list<br/>(K=4 个)"]
    end

    subgraph FNGO["grad & QGT"]
        G1["make_grad_fn(<br/>ha,<br/>total_matrix_machine,<br/>total_max_machine,<br/>total_machine,<br/>single_machine_list)"]
        G1a["内部: nes_vmc_gradient_stable<br/>→ jax.grad + jax.vmap<br/>→ masked weighted mean"]
        G2["make_qgt_fn(total_machine)"]
        G2a["内部: jax.grad + jax.vmap<br/>→ flatten_batched_pytree<br/>→ S = (1/N) O† O + λI"]
    end

    subgraph SAMP["采样器"]
        S1["nes_sampler = nk.sampler.<br/>MetropolisSampler(<br/>hilbert=hi_ext,<br/>rule=nes_rule,<br/>n_chains=16,<br/>sweep_size=30)"]
        S2["sampler_state = nes_sampler.<br/>init_state(total_machine,<br/>total_params,<br/>jax.random.PRNGKey(21))"]
    end

    subgraph FCI["FCI 基准"]
        F1["eigvals, eigvecs = eigsh(<br/>ha.to_sparse(), k=4,<br/>which='SA', tol=1e-10)"]
    end

    subgraph OPT["优化器"]
        O1["optimizer = optax.chain(<br/>optax.clip_by_global_norm(20.0),<br/>optax.sgd(0.1))"]
        O2["opt_state = optimizer.init(<br/>total_params)"]
    end

    M1 --> M2
    M2 --> M3
    M2 --> M4 --> M5 --> M6
    M2 --> M7
    M6 --> G1
    M6 --> G2
    G1 --> G1a
    G2 --> G2a
    M6 --> S1
    H8 --> F1
    H6 --> O1
    H7 --> O1
    O1 --> O2
    M6 --> O2
    S1 --> S2
    M6 --> S2
    M6 -.传 machine.-> S2

    classDef hyperBox fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef modelBox fill:#e8eaf6,stroke:#283593,color:#000
    classDef gradBox fill:#fff3e0,stroke:#e65100,color:#000
    classDef sampBox fill:#e0f7fa,stroke:#00695c,color:#000
    classDef fciBox fill:#fce4ec,stroke:#ad1457,color:#000
    classDef optBox fill:#f1f8e9,stroke:#558b2f,color:#000
    class H1,H2,H3,H4,H5,H6,H7,H8 hyperBox
    class M1,M2,M3,M4,M5,M6,M7 modelBox
    class G1,G1a,G2,G2a gradBox
    class S1,S2 sampBox
    class F1 fciBox
    class O1,O2 optBox
```

---

## 4. Cell 5：训练主循环（Sequence 时序图）

```mermaid
sequenceDiagram
    autonumber
    participant L as 训练循环
    participant S as nes_sampler
    participant GF as total_machine
    participant GX as grad_fn
    participant GQ as qgt_fn
    participant OP as optimizer (optax)
    participant LOG as logger

    Note over L: for step in range(N_ITER=100)

    L->>S: sample(machine=total_machine, parameters=total_params, state=sampler_state, chain_length=200)
    S->>GF: total_machine(total_params, σ) 计算 logΨ
    GF-->>S: logΨ_gauge(σ) → Metropolis 接受/拒绝
    S-->>L: samples_raw (16×200, 32), sampler_state

    L->>L: samples_raw.reshape(-1, K, SINGLE_SIZE)<br/>= (3200, 4, 8)

    L->>GX: grad_fn(total_params, x_batch)
    Note over GX: nes_vmc_gradient_stable:<br/>① NES_loss_energy_stable<br/>② E_L 中心化<br/>③ jax.grad(holomorphic)<br/>④ weighted mean
    GX-->>L: grad_raw (PyTree), loss_mean, E_L_mean (4×4)

    L->>L: grad_raw_flat, unravel_fn = ravel_pytree(grad_raw)
    L->>L: grad_norm_raw = ‖grad_raw_flat‖

    alt Natural_Grad = True
        L->>GQ: qgt_fn(total_params, x_batch, qgt_diag_shift=0.1)
        Note over GQ: S = (1/N) O† O + λI
        GQ-->>L: S_reg (N×N)
        L->>L: ng_flat = jnp.linalg.solve(S_reg, grad_raw_flat)
        L->>L: grad_update = unravel_fn(ng_flat)
    else Natural_Grad = False
        L->>L: grad_update = grad_raw
    end

    L->>OP: optimizer.update(grad_update, opt_state, total_params)
    OP-->>L: updates (PyTree), new opt_state

    L->>L: clip_transform.update(grad_update, ...)<br/>单独算 grad_norm_clipped
    L->>L: total_params = optax.apply_updates(total_params, updates)

    Note over L: 监控
    L->>L: psi_mat = total_matrix_machine(params, x_batch[0:1])[0]
    L->>L: psi_cond = jnp.linalg.cond(psi_mat)
    L->>L: log_Psi_batch = total_machine(params, x_batch)
    L->>L: eig_vals, _ = jnp.linalg.eig(E_L_mean)
    L->>L: sort by real part → E0, E1, E2, E3
    L->>L: history[...].append(...)

    L->>LOG: logger.info(<br/>logΨ mean/min/max, grad norm,<br/>cond(Ψ), Loss, E0~E3)

    Note over L: 异常保护
    L->>L: if grad_norm_clipped == 0: break
```

---

## 5. grad_fn 内部算法流程（NES 损失 + 梯度推导）

```mermaid
flowchart TD
    A["x_batch: (batch, K, n_spin)<br/>= (3200, 4, 8)"] --> B["NES_loss_energy_stable"]

    subgraph LOSS["NES_loss_energy_stable"]
        L1["L_stable = total_matrix_machine(params, x_batch)<br/>shape (batch, 4, 4)"]
        L2["shift = total_max_machine(params, x_batch)<br/>shape (batch,)"]
        L3["Psi_stable = exp(L_stable)"]
        L4["HPsi_stable = Ham_Psi_scaled(<br/>ha, single_machine_list,<br/>total_params, x_batch, shift)"]
        L5["E_L = solve(Psi_stable, HPsi_stable)<br/>(batch, 4, 4)"]
        L6["loss_batch = Re·Tr(E_L)<br/>(batch,)"]
        L7["valid = isfinite(L_stable) & isfinite(HPsi) & isfinite(E_L) & isfinite(loss)<br/>(batch,)"]
        L8["return loss_batch, E_L, aux{valid, ...}"]
        L1 --> L3
        L2 --> L4
        L3 --> L5
        L4 --> L5
        L5 --> L6
        L5 --> L7
        L6 --> L7
        L7 --> L8
    end

    B --> L1
    B --> L2

    L8 --> C["nes_vmc_gradient_stable 续"]

    subgraph GRAD["nes_vmc_gradient_stable（续）"]
        G1["loss_mean = _masked_mean_batch(loss_batch, valid)"]
        G2["E_L_safe = where(isfinite, E_L, 0)"]
        G3["E_L_centered = E_L_safe - E_L_mean"]
        G4["tr_centered = Re·Tr(E_L_centered) (batch,)"]
        G5["tr_centered = where(valid, tr_centered, 0)"]
        G6["dlogPsi_batch = vmap(<br/>jax.grad(total_machine,<br/>argnums=0, holomorphic=True)<br/>)(total_params, x_batch)"]
        G7["grad = tree.map(<br/>weight_and_masked_mean,<br/>dlogPsi_batch)"]
        G8["weight = tr_centered<br/>对每个 leaf:<br/>sum(weight × conj(dlogPsi) × valid) / N_valid"]
        G4 --> G5
        G5 --> G8
        G6 --> G7
        G8 --> G7
    end

    L8 --> G1
    L8 --> G2
    L8 --> G3
    L8 --> G4
    L8 --> G6

    G1 --> H["输出:<br/>grad_raw (PyTree, 总 ~1832 复数参数)<br/>loss_mean (标量)<br/>E_L_mean (4×4)"]
    G7 --> H

    classDef input fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef loss fill:#fff3e0,stroke:#e65100,color:#000
    classDef grad fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef output fill:#c8e6c9,stroke:#2e7d32,color:#000
    class A,B input
    class L1,L2,L3,L4,L5,L6,L7,L8 loss
    class G1,G2,G3,G4,G5,G6,G7,G8 grad
    class H output
```

---

## 6. qgt_fn 内部算法流程（QGT 矩阵构造）

```mermaid
flowchart LR
    A["qgt_fn(total_params, x_batch, 0.1)"] --> B["vmap(jax.grad(total_machine, argnums=0, holomorphic=True))<br/>(None, 0) → (params, x_batch)"]
    B --> C["grad_tree_batch<br/>(PyTree, 每一叶 (batch, n_params_i))"]
    C --> D["flatten_batched_pytree<br/>→ O (batch, N)"]
    D --> E["O_mean = mean(O, axis=0, keepdims=True)<br/>(1, N)"]
    E --> F["O_centered = O - O_mean<br/>(batch, N)"]
    F --> G["S = (1/N_batch) · conj(O_centered)^T · O_centered<br/>(N, N)"]
    G --> H["S = 0.5·(S + conj(S)^T)<br/>强制 Hermitian"]
    H --> I["S_reg = S + 0.1·I<br/>Tikhonov 正则"]
    I --> J["return S_reg (N, N)"]

    K["solve(S_reg, grad_raw_flat)"] --> L["ng_flat (N,)"]
    L --> M["grad_update = unravel_fn(ng_flat)<br/>(PyTree)"]

    classDef input fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef comp fill:#fff3e0,stroke:#e65100,color:#000
    classDef output fill:#c8e6c9,stroke:#2e7d32,color:#000
    class A,B,C,D,E,F,G,H,I,J input
    class K,L,M comp
```

---

## 7. 数据在训练循环中的形态变化

```mermaid
flowchart LR
    subgraph INIT["初始化"]
        I1["total_params<br/>PyTree(K=4 个 SingleStateAnsatz<br/>+ 约 1832 复数参数)"]
    end

    subgraph SAMP["采样后"]
        S1["samples_raw<br/>(n_chains×chain_length, hi_ext.size)<br/>= (3200, 32)"]
        S2["x_batch = samples_raw.reshape(-1, K, SINGLE_SIZE)<br/>= (3200, 4, 8)"]
    end

    subgraph GRAD["grad_fn 后"]
        G1["grad_raw: PyTree<br/>每个 leaf 形状与对应参数相同"]
        G2["grad_raw_flat: (N_total,) ≈ (1832,)"]
        G3["loss_mean: 标量<br/>E_L_mean: (4, 4)"]
    end

    subgraph QGT["QGT 后"]
        Q1["S_reg: (N_total, N_total) ≈ (1832, 1832)"]
        Q2["ng_flat: (N_total,) ≈ (1832,)"]
        Q3["grad_update: PyTree (unravel)"]
    end

    subgraph CLIP["clip + apply 后"]
        C1["updates: PyTree, ‖·‖ ≤ 20"]
        C2["new total_params: PyTree"]
    end

    subgraph MON["监控"]
        M1["psi_cond: cond(L_stable) 标量"]
        M2["eig_vals: 4 个能量<br/>按实部排序 E0~E3"]
        M3["history 累积"]
    end

    I1 --> SAMP
    S1 --> S2
    S2 --> G1
    G1 --> G2
    G1 --> G3
    G2 --> Q1
    G2 --> Q2
    Q2 --> Q3
    Q3 --> C1
    C1 --> C2
    Q1 -.->|solve| Q2
    S2 -.->|x_batch[0:1]| M1
    S2 -.->|x_batch| M1
    G3 --> M2
    G3 --> M3
    Q1 --> M3
    M1 --> M3
    M2 --> M3

    classDef param fill:#fce4ec,stroke:#ad1457,color:#000
    classDef data fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef grad fill:#fff3e0,stroke:#e65100,color:#000
    classDef qgt fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef clip fill:#c8e6c9,stroke:#2e7d32,color:#000
    classDef mon fill:#fff8e1,stroke:#ff8f00,color:#000
    class I1 param
    class S1,S2 data
    class G1,G2,G3 grad
    class Q1,Q2,Q3 qgt
    class C1,C2 clip
    class M1,M2,M3 mon
```

---

## 8. NES-VMC 算法视角下的整体数据流（公式 ↔ 代码）

```mermaid
flowchart TD
    subgraph PHYS["物理层"]
        P1["$\\mathbf{x}_1, ..., \\mathbf{x}_K$<br/>扩展 Hilbert 的样本"]
        P2["$\\psi_j(\\mathbf{x}_i)$<br/>SingleStateAnsatz (复值 FFNN)"]
        P3["$L_{ij} = \\log\\psi_j(\\mathbf{x}_i) - \\log\\psi_j(\\text{HF})$<br/>gauge-fixed"]
        P4["$\\Psi = \\det(e^L) = \\prod \\sigma$ + j·phase<br/>K×K 行列式"]
        P5["$(H\\Psi)_{ij} = \\sum_{x'} H[x_i, x'] \\psi_j(x')$<br/>K×K 哈密顿作用"]
        P6["$E_L = \\Psi^{-1} H \\Psi$<br/>局域能量矩阵（4×4）"]
        P7["$\\mathcal{L} = \\mathrm{Re}\\cdot\\mathrm{Tr}(E_L) = \\sum_k E_k$"]
        P8["$\\nabla_\\theta \\mathcal{L} = \\mathbb{E}[\\mathrm{tr}(E_L - \\bar{E}_L) \\nabla_\\theta \\log\\Psi^*]$"]
        P9["$S = \\frac{1}{N}\\sum (\\nabla\\log\\Psi^* - \\overline{\\nabla\\log\\Psi^*})(\\nabla\\log\\Psi - \\overline{\\nabla\\log\\Psi})^\\dagger$"]
        P10["$\\delta\\theta = S_{reg}^{-1} \\nabla_\\theta \\mathcal{L}$<br/>自然梯度"]
    end

    subgraph CODE["代码层"]
        C1["samples_raw, x_batch"]
        C2["total_ansatz.single_ansatz_list[j]"]
        C3["create_gauge_fixed_total_machines"]
        C4["slogdet(Ψ_stable) + K·shift"]
        C5["Ham_Psi_scaled"]
        C6["NES_loss_energy_stable"]
        C7["nes_vmc_gradient_stable"]
        C8["make_qgt_fn"]
        C9["jnp.linalg.solve(S_reg, grad_raw_flat)"]
    end

    P1 --> C1
    C1 --> C2 --> P2
    C2 --> P3
    P3 --> C3
    C3 --> C4 --> P4
    C4 --> P5
    C4 --> C5
    C5 --> P5
    P5 --> C6
    P4 --> C6
    C6 --> P6
    C6 --> P7
    P7 --> C7
    C4 -.loss.-> C7
    C7 --> P8
    C4 -.logΨ.-> C8
    C8 --> P9
    P8 --> C9
    P9 --> C9
    C9 --> P10

    classDef phys fill:#e3f2fd,stroke:#1565c0,color:#000
    classDef code fill:#fff3e0,stroke:#e65100,color:#000
    class P1,P2,P3,P4,P5,P6,P7,P8,P9,P10 phys
    class C1,C2,C3,C4,C5,C6,C7,C8,C9 code
```

---

## 9. Cell 6+：后处理（保存 / 加载 / 可视化）

```mermaid
flowchart LR
    A["训练完 history dict<br/>(step, loss, E_Lmatrix,<br/>samples, log_Psi_*, grad_norm_*,<br/>psi_cond, energy_0st...3st,<br/>params 列表)"] --> B{"保存方式"}

    B -- "Natural_Grad = True" --> C1["pickle.dump(<br/>history,<br/>'./data/history_natural_<br/>gradient_LiH_molecule_K4.pkl')"]
    B -- "Natural_Grad = False" --> C2["pickle.dump(<br/>history,<br/>'./data/history_plain_<br/>gradient_LiH_molecule_K4.pkl')"]

    C1 --> D["pickle.load 读取"]
    C2 --> D

    D --> P1["plot 1: 6 子图<br/>4 个能量曲线 + 对比 + 整体对比"]
    D --> P2["plot 2: 3 子图<br/>K=4 能量 + loss + grad norm"]
    D --> P3["plot 3: 2 子图<br/>只画 0st 能量"]

    P1 --> END["📈 训练可视化输出"]
    P2 --> END
    P3 --> END

    classDef data fill:#fff8e1,stroke:#ff8f00,color:#000
    classDef io fill:#e8f5e9,stroke:#2e7d32,color:#000
    classDef plot fill:#e1f5fe,stroke:#0277bd,color:#000
    classDef result fill:#f3e5f5,stroke:#6a1b9a,color:#000
    class A data
    class B,C1,C2,D io
    class P1,P2,P3 plot
    class END result
```

---

## 10. 函数调用关系全景图

```mermaid
flowchart TD
    subgraph CELL1["Cell 1: import"]
        A1["jax, jax.numpy, optax,<br/>flax.nnx, netket,<br/>netket.experimental"]
    end

    subgraph CELL4["Cell 4: 采样规则"]
        A2["NESFermionHopRule"]
    end

    subgraph CELL5_INIT["Cell 5: init"]
        A3["NESTotalAnsatz_stable"]
        A4["create_gauge_fixed_total_machines"]
        A5["create_single_machine_gauge_fixed"]
        A6["make_grad_fn"]
        A7["make_qgt_fn"]
        A8["MetropolisSampler"]
        A9["eigsh(ha.to_sparse())"]
        A10["optax.chain"]
        A11["sampler.init_state"]
    end

    subgraph CELL5_TRAIN["Cell 5: train"]
        B1["nes_sampler.sample"]
        B2["grad_fn"]
        B3["nes_vmc_gradient_stable"]
        B4["NES_loss_energy_stable"]
        B5["Ham_Psi_scaled"]
        B6["Ham_psi_scaled"]
        B7["_masked_mean_batch"]
        B8["jax.grad + jax.vmap"]
        B9["ravel_pytree"]
        B10["qgt_fn"]
        B11["flatten_batched_pytree"]
        B12["jnp.linalg.solve"]
        B13["optax.clip_by_global_norm"]
        B14["optimizer.update"]
        B15["optax.apply_updates"]
        B16["jnp.linalg.cond"]
        B17["jnp.linalg.eig"]
        B18["jnp.argsort"]
    end

    subgraph CELL5_LOG["Cell 5: log"]
        C1["logger.info (×4 行/step)"]
    end

    subgraph CELL6["Cell 6+"]
        D1["pickle.dump / pickle.load"]
        D2["matplotlib.pyplot"]
    end

    A1 --> A3
    A1 --> A4
    A1 --> A5
    A1 --> A6
    A1 --> A7
    A1 --> A2

    A3 --> A4
    A4 --> A5
    A4 --> A6
    A4 --> A7
    A4 --> A8
    A4 --> A11
    A4 --> A10
    A4 --> A9
    A2 --> A8

    A11 --> B1
    A6 --> B2
    A7 --> B10
    A4 --> B1
    A4 --> B2
    A4 --> B10
    A5 --> B3
    A4 --> B3
    A4 --> B5
    A4 --> B6

    B1 --> B2
    B2 --> B3
    B3 --> B4
    B3 --> B7
    B3 --> B8
    B4 --> B5
    B5 --> B6
    B8 --> B3
    B2 --> B9
    B9 --> B10
    B10 --> B11
    B10 --> B8
    B11 --> B10
    B10 --> B12
    B12 --> B9
    B9 --> B13
    B13 --> B14
    B14 --> B15
    B4 --> B16
    B3 --> B17
    B17 --> B18
    B14 --> C1
    B15 --> C1
    B16 --> C1
    B17 --> C1
    B18 --> C1

    B15 --> D1
    D1 --> D2

    classDef cell fill:#fff3e0,stroke:#e65100,color:#000
    classDef init fill:#e8eaf6,stroke:#283593,color:#000
    classDef train fill:#f3e5f5,stroke:#6a1b9a,color:#000
    classDef log fill:#c8e6c9,stroke:#2e7d32,color:#000
    classDef post fill:#fce4ec,stroke:#ad1457,color:#000
    class A1,A2 cell
    class A3,A4,A5,A6,A7,A8,A9,A10,A11 init
    class B1,B2,B3,B4,B5,B6,B7,B8,B9,B10,B11,B12,B13,B14,B15,B16,B17,B18 train
    class C1 log
    class D1,D2 post
```

---

## 11. 训练循环一次迭代（100 步中的任意一步）的完整时间线

```mermaid
gantt
    title 训练单步时间线（典型 ~0.5-1s/step on M1/M2）
    dateFormat X
    axisFormat %s
    
    section 采样
    nes_sampler.sample (16链×30跳×200) :a1, 0, 200ms
    
    section 梯度
    grad_fn jit 调用 (compile 后) :a2, after a1, 50ms
    
    section QGT
    qgt_fn 构造 S_reg 矩阵 :a3, after a2, 80ms
    jnp.linalg.solve (1832×1832) :a4, after a3, 60ms
    
    section 优化
    optax.clip + sgd.update :a5, after a4, 5ms
    optax.apply_updates :a6, after a5, 5ms
    
    section 监控
    cond(Ψ) + eig(E_L) :a7, after a6, 10ms
    history.append + logger.info :a8, after a7, 5ms
    
    section 总计
    step total :crit, after a8, 415ms
```

---

## 12. 关键观察

1. **初始化阶段**（Cell 5 顶部）只做一次，但**总耗时占比 ~30%**（包括 JIT 编译 4 个机器 + grad_fn + qgt_fn）。
2. **训练阶段** 的真实瓶颈是 **nes_sampler.sample**（~200ms/iter）和 **qgt_fn + solve**（~140ms/iter），共占 60% 以上。
3. **QGT 矩阵** 是 N×N ≈ 1832×1832 复数矩阵，`jnp.linalg.solve` 在 LiH K4 下约 60ms/iter，是**冻结方案 ❸ 物理分离**提速的主要目标。
4. **grad_fn** 内部调用链最长（grad → loss → HPsi → Ham_psi → get_conn_padded → single_machine），单次 ~50ms 是次要瓶颈。
5. **cond(Ψ) + eig(E_L)** 的 ~10ms 是诊断用的，**生产中可以异步化**。

---

## 13. 与规范漂移版的对比

| 维度 | LiH_molecule_K4_STO-3G.ipynb | LiH 规范漂移-1.ipynb |
|------|---|---|
| Ansatz | `NESTotalAnsatz_stable` | `NESTotalAnsatz_gauge_stable` |
| Machines | `create_gauge_fixed_total_machines`<br/>(一站式 5 件套) | `create_machine_gauge_stable`<br/>+ `create_machine_matrix_gauge_stable`<br/>+ `create_machine_max_gauge_stable` |
| N_ITER | 100 | 500 |
| 训练总耗时 | ~40s | ~350s |
| 主要不同 | **没有 logΨ 归一化 gauge-fixed trace 的 K·shift 细节**，更适合入门 | **带完整规范稳定化 + 多重异常保护** |

---

## 附：相关代码位置

| 文件 | 角色 | 关键函数/类行号 |
|------|------|----------------|
| [LiH_molecule_K4_STO-3G.ipynb](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH_molecule_K4_STO-3G.ipynb) | 训练入口 | Cell 1-5 (主) |
| [LiH.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/LiH.py) | 分子/Hilbert/FCI | L11-L167 |
| [NES_VMC.py](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py) | NES-VMC 核心实现 | L26-1700 |
| [NESTotalAnsatz_stable](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L96-L168) | 总 Ansatz | L96 |
| [NESTotalAnsatz_gauge_stable](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L169-L261) | 规范稳定化总 Ansatz | L169 |
| [create_gauge_fixed_total_machines](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1191-L1454) | 5 件套 wrapper | L1191 |
| [make_grad_fn](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1536-L1559) | 梯度闭包 | L1536 |
| [make_qgt_fn](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1512-L1534) | QGT 闭包 | L1512 |
| [nes_vmc_gradient_stable](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L775-L940) | NES 梯度 | L775 |
| [NES_loss_energy_stable](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L644-L748) | NES 损失 | L644 |
| [Ham_Psi_scaled](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L571-L642) | 哈密顿作用 | L571 |
| [NESFermionHopRule](file:///Users/yangjianfei/mac_vscode/神经网络量子态/6 月/0625/LiH 分子/NES_VMC.py#L1119-L1190) | 采样规则 | L1119 |
