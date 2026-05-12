# Natural Excited State in Hybrid VQMC

37-247208 Yang Jianfei, D3 Ishikawa Sato Laboratory, April 21 2026

## Abstract
Variational Quantum Monte Carlo (VQMC) with Neural Quantum States (NQS) has become a powerful tool for quantum many-body simulations, but excited-state calculations remain limited by poor expressivity and state collapse issues. This work integrates Natural Excited States (NES) into quantum-classical hybrid VQMC, constructing a hybrid Ansatz that combines parameterized quantum circuits (PQCs) and neural networks. By reformulating excited-state search as ground-state optimization in an extended Slater determinant space, our method avoids explicit orthogonalization and achieves unbiased gradient estimation. Benchmarks on the LiH molecule and Fermi-Hubbard model show that the proposed Natural Excited State Hybrid VQMC (NES-HVQMC) achieves higher accuracy for excited states under similar parameter counts, with clear advantages over pure classical NQS and conventional hybrid VQMC. This framework provides a stable, scalable route to excited-state simulations in strongly correlated quantum systems.

**Keywords—** Variational Monte Carlo, Quantum Computation, Neural Quantum State, Natural Excited State, Hybrid Ansatz

## 1. INTRODUCTION
Quantum many-body systems exhibit rich phenomena governed by strongly correlated electrons, and accurate simulation of both ground and excited states is critical for chemistry, condensed matter physics, and materials science. Variational Quantum Monte Carlo (VQMC) combines variational principles with stochastic sampling and has emerged as a scalable method for high-dimensional systems. The core of VQMC lies in the design of trial wavefunctions (Ansatz), and neural network quantum states (NQS) have greatly improved expressivity compared to traditional Slater-Jastrow forms.

However, two major challenges remain. First, conventional NQS and hybrid quantum-classical Ansätze are mostly designed for ground states, and excited-state simulations suffer from state collapse, biased gradients, or complex orthogonalization procedures. Second, purely classical neural networks face bottlenecks in expressivity and scaling, while quantum circuits naturally encode quantum entanglement but are limited by hardware noise and optimization difficulty.

Natural Excited States (NES) provide an elegant solution by transforming K excited-state problems into a single ground-state problem in an extended system. By representing the total wavefunction as a determinant of individual state Ansätze, NES automatically prevents collapse without penalty terms and enables unbiased energy and gradient estimation. This work introduces NES into quantum-classical hybrid VQMC, building a unified framework for accurate excited-state simulations.

Our contributions are threefold: (1) We propose the Natural Excited State Hybrid VQMC (NES-HVQMC) framework, merging PQC–neural network hybrid Ansatz with the NES variational principle. (2) We design a sequential optimization strategy for classical and quantum parameters using SR and Adam. (3) We validate the method on small molecules and Fermi-Hubbard models, showing superior accuracy and stability for excited states.

This work extends hybrid VQMC from ground states to excited states and offers a practical path toward excited-state simulation in the NISQ era.

## 2. BACKGROUND: HYBRID VQMC AND NATURAL EXCITED STATES
2.1 Variational Quantum Monte Carlo
VQMC approximates the ground state by minimizing the Rayleigh quotient:
E(θ) = ⟨Ψ(θ)|Ĥ|Ψ(θ)⟩ / ⟨Ψ(θ)|Ψ(θ)⟩.
Using Monte Carlo sampling, the local energy EL(x) = Ψ(x)⁻¹ĤΨ(x) is estimated stochastically. For large systems, sampling from |Ψ|² provides unbiased gradients.

2.2 Quantum-Classical Hybrid Ansatz
Traditional NQS uses classical networks to model amplitude and phase. Hybrid Ansätze split the wavefunction into quantum and classical parts:
Ψ(s; θ, λ) = Ψ_PQC(s; θ) · Ψ_NN(s; λ),
where PQC captures quantum correlations and NN improves flexibility. Parameters are optimized via SR for NN and Adam for PQC.

2.3 Natural Excited States
To obtain K excited states, NES defines a total wavefunction as a determinant:
Ψ(x¹,…,xᴷ) = det[ψᵢ(xʲ)],
where ψᵢ are individual state Ansätze. Minimizing the trace of the local energy matrix yields orthogonal states automatically, with no free parameters and unbiased gradients. The energy matrix is diagonalized at convergence to recover individual excited-state energies.

## 3. METHOD: NATURAL EXCITED STATE IN HYBRID VQMC
3.1 Hybrid NES Wavefunction
We extend the hybrid Ansatz to the NES framework. Each single-state Ansatz is a quantum-classical hybrid form:
ψᵢ(x; θᵢ, λᵢ) = Ψ_PQC(x; θᵢ) × Ψ_NN(x; λᵢ).
The total NES wavefunction is:
Ψ_total(x¹,…,xᴷ) = det[ψᵢ(xʲ)].
This inherits both the expressivity of hybrid quantum-classical models and the stability of NES.

3.2 Local Energy Matrix and Variational Principle
The local energy becomes a matrix:
E_L(x¹,…,xᴷ) = Ψ⁻¹ · ĤΨ,
where Ĥ = Ĥ₁⊕…⊕Ĥᴷ. The variational objective is Tr[E⟨E_L⟩], which corresponds to the sum of the K lowest energies.

3.3 Sequential Optimization
We optimize NN parameters with Stochastic Reconfiguration (SR) and PQC parameters with Adam:
1. Fix PQC parameters, optimize NN via SR.
2. Fix NN parameters, optimize PQC via Adam.
3. Iterate until convergence.

This strategy balances training stability and efficiency for hybrid systems.

## 4. EXPERIMENTS
4.1 Molecular System: LiH
We simulate a 6-qubit LiH molecule at bond length 2.4 Å. We compare:
- Classical NQS (d=3, d=8)
- Hybrid VQMC (ground state)
- NES-HVQMC (excited states)
Under similar parameter counts, NES-HVQMC achieves chemical accuracy for both ground and excited states, outperforming classical NQS.

4.2 Strongly Correlated System: Fermi-Hubbard Model
We test 1D and 2D 4-site Fermi-Hubbard models (U=4, t=1). Results show:
- NES-HVQMC converges faster and more accurately than FNN, HEA, and hybrid VQMC.
- Excited-state energies match exact values closely.
- No state collapse is observed during training.

4.3 Performance Summary
In all benchmarks, NES-HVQMC achieves:
- Higher accuracy for excited states
- Better stability (no collapse)
- Comparable computational cost
This validates the effectiveness of integrating NES into hybrid VQMC.

## 5. CONCLUSION
This work presents Natural Excited State in Hybrid VQMC (NES-HVQMC), a unified framework for excited-state quantum many-body simulation. By combining quantum-classical hybrid Ansätze with the NES variational principle, we avoid state collapse and achieve unbiased, stable optimization. Experiments on LiH and Fermi-Hubbard models confirm that the method improves accuracy and scalability for both ground and excited states.

NES-HVQMC provides a promising approach for excited-state calculations in chemistry and condensed matter physics. Future work will extend this method to larger systems, explore deeper quantum circuits, and apply it to optical and transport properties of quantum materials.

## REFERENCES
[1] G. Carleo and M. Troyer, Science, 355, 602 (2017).
[2] D. Pfau et al., arXiv:2308.16848 (2024).
[3] Z. Zhang et al., arXiv:2501.12130 (2025).
[4] J. Hermann et al., Nat. Chem., 12, 891 (2020).
[5] W. M. C. Foulkes et al., Rev. Mod. Phys., 73, 33 (2001).