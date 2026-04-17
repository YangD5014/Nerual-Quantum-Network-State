# Introduction

Variational Quantum Monte Carlo (VQMC) has achieved outstanding applications in quantum many-body system simulations, especially in recent years, the introduction of neural network wave functions has greatly improved the accuracy of VQMC. The application of neural network wave functions in VMC simulations of molecular systems has been proven to be very successful in accurately describing states with multi-reference characteristics and establishing strong correlations. Recently, these methods have been successfully extended to simulate excited states through penalty-based approaches. Although results in some applications have been promising, neural network-based excited state simulation methods are still under exploration.

This research will focus on using hybrid quantum-classical Ansatz architectures to simulate excited states of small molecular systems. On one hand, relevant research has proven that hybrid architectures combining parameterized quantum circuits with classical neural networks can be more

# Demo

1.[H2 激发态计算_RBM](./Netket_excited_state/H2_RBM_1.ipynb)  
2.[H2 激发态计算_Hybrid](./Netket_excited_state/H2_Hybrid_1.ipynb)  
3.[H2 激发态计算_comparison](./Netket_excited_state/ComparisionH2.ipynb)



# Neural Network Wave Functions

## 1. Overview of Neural Network Wave Functions

Neural network wave functions are wave function representation methods based on artificial neural networks. They represent the wave function of a quantum system as a parameterized function, where parameters are learned through the weights and biases of neural networks. The advantage of neural network wave functions is that they can learn complex wave function representations and can be computed in parallel on hardware, thereby improving simulation efficiency.

For quantum systems, the wave function $\Psi(\mathbf{x})$ can be expressed as:

$$\Psi(\mathbf{x}; \boldsymbol{\theta}) = A(\theta)e^{f(\mathbf{x}; \boldsymbol{\theta})}$$

where $\mathbf{x}$ is the configuration of the system (such as spin configuration or particle positions), $\boldsymbol{\theta}$ are the parameters of the neural network, and $f(\mathbf{x}; \boldsymbol{\theta})$ is the output of the neural network.

## 2. Restricted Boltzmann Machine (RBM)

A Restricted Boltzmann Machine is a two-layer neural network consisting of visible and hidden layers. For real-valued wave functions, the energy function of an RBM is:

$$E(\mathbf{v}, \mathbf{h}) = -\sum_{i=1}^{N_v} a_i v_i - \sum_{j=1}^{N_h} b_j h_j - \sum_{i=1}^{N_v}\sum_{j=1}^{N_h} W_{ij} v_i h_j$$

where:
- $\mathbf{v} = (v_1, v_2, \ldots, v_{N_v})$ are the states of visible layer neurons
- $\mathbf{h} = (h_1, h_2, \ldots, h_{N_h})$ are the states of hidden layer neurons
- $a_i$ are the biases of the visible layer
- $b_j$ are the biases of the hidden layer
- $W_{ij}$ are the connection weights

The amplitude of the wave function can be expressed as:

$$|\Psi(\mathbf{v})| = \sqrt{p(\mathbf{v})} = \frac{1}{\sqrt{Z}} \sum_{\mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$

where $Z$ is the partition function:

$$Z = \sum_{\mathbf{v}, \mathbf{h}} e^{-E(\mathbf{v}, \mathbf{h})}$$

By summing over the hidden layer, we obtain:

$$|\Psi(\mathbf{v})| = \frac{1}{\sqrt{Z}} \prod_{j=1}^{N_h} 2\cosh\left(b_j + \sum_{i=1}^{N_v} W_{ij} v_i\right) \exp\left(\sum_{i=1}^{N_v} a_i v_i\right)$$

## 3. Complex-valued Restricted Boltzmann Machine (Complex-value RBM)

For quantum systems, wave functions are typically complex-valued. Complex-valued Restricted Boltzmann Machines represent complex wave functions by introducing complex weights and biases.

# Basic Introduction and Computational Methods of VMC

## 1. Basic Principles of Variational Monte Carlo (VMC)

### 1.1 Variational Principle

For quantum mechanical systems, the Schrödinger equation is:

$$\hat{H}\Psi(\mathbf{R}) = E\Psi(\mathbf{R})$$

where $\hat{H}$ is the Hamiltonian operator, $\Psi(\mathbf{R})$ is the wave function, and $E$ is the energy.

The variational principle states that for any trial wave function $\Psi_T(\mathbf{R})$, its expected energy is always greater than or equal to the ground state energy:

$$E_0 \leq \frac{\langle \Psi_T|\hat{H}|\Psi_T \rangle}{\langle \Psi_T|\Psi_T \rangle} = \int \frac{|\Psi_T(\mathbf{R})|^2 \frac{\hat{H}\Psi_T(\mathbf{R})}{\Psi_T(\mathbf{R})} d\mathbf{R}}{\int |\Psi_T(\mathbf{R})|^2 d\mathbf{R}}$$

Define the local energy:

$$E_L(\mathbf{R}) = \frac{\hat{H}\Psi_T(\mathbf{R})}{\Psi_T(\mathbf{R})}$$

Then the expected energy can be expressed as:

$$E = \int |\Psi_T(\mathbf{R})|^2 E_L(\mathbf{R}) d\mathbf{R}$$

### 1.2 Monte Carlo Integration

Using Monte Carlo methods, sample $N$ configurations $\{\mathbf{R}_i\}$ from the probability distribution $|\Psi_T(\mathbf{R})|^2$, then the expected energy can be approximated as:

$$E \approx \frac{1}{N}\sum_{i=1}^{N} E_L(\mathbf{R}_i)$$

## 2. Ground State VMC Method

### 2.1 Metropolis-Hastings Algorithm

To sample from the distribution $|\Psi_T(\mathbf{R})|^2$, the Metropolis-Hastings algorithm is used:

1. Propose a new configuration $\mathbf{R}'$ from the current configuration $\mathbf{R}$
2. Calculate the acceptance probability:

$$A(\mathbf{R} \rightarrow \mathbf{R}') = \min\left(1, \frac{|\Psi_T(\mathbf{R}')|^2}{|\Psi_T(\mathbf{R})|^2}\right)$$

3. Accept the new configuration with probability $A$

### 2.2 Energy Gradient

For a parameterized trial wave function $\Psi_T(\mathbf{R}; \boldsymbol{\alpha})$, where $\boldsymbol{\alpha} = \{\alpha_1, \alpha_2, \ldots, \alpha_M\}$ are variational parameters, the gradient of energy with respect to parameters is:

$$\frac{\partial E}{\partial \alpha_k} = 2\langle E_L(\mathbf{R}) \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} \rangle - 2\langle E_L(\mathbf{R}) \rangle \langle \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} \rangle$$

where the logarithmic derivative is:

$$O_k(\mathbf{R}) = \frac{\partial \ln |\Psi_T(\mathbf{R})|}{\partial \alpha_k} = \frac{1}{\Psi_T(\mathbf{R})} \frac{\partial \Psi_T(\mathbf{R})}{\partial \alpha_k}$$

## 3. Excited State VMC Method

### 3.1 Variational Principle for Excited States

For the $n$-th excited state, if the trial wave function $\Psi_T^{(n)}(\mathbf{R})$ is orthogonal to all lower energy states:

$$\langle \Psi_T^{(n)}|\Psi^{(m)} \rangle = 0, \quad m = 0, 1, \ldots, n-1$$

Then its expected energy satisfies:

$$E_n \leq \frac{\langle \Psi_T^{(n)}|\hat{H}|\Psi_T^{(n)} \rangle}{\langle \Psi_T^{(n)}|\Psi_T^{(n)} \rangle}$$

### 3.2 Orthogonalization Methods

#### Method One: Explicit Orthogonalization

Given the ground state wave function $\Psi_0(\mathbf{R})$, construct a trial wave function orthogonal to the ground state:

$$\Psi_T^{(1)}(\mathbf{R}) = \hat{P}_1 \Phi(\mathbf{R}) = \left[1 - |\Psi_0\rangle\langle\Psi_0|\right]\Phi(\mathbf{R})$$

where $\Phi(\mathbf{R})$ is the initial trial wave function.

#### Method Two: Energy Minimization

Define the Lagrangian with orthogonal constraints:

$$\mathcal{L} = \langle \Psi_T^{(1)}|\hat{H}|\Psi_T^{(1)} \rangle - \lambda \langle \Psi_T^{(1)}|\Psi_T^{(1)} \rangle - \mu \langle \Psi_T^{(1)}|\Psi_0\rangle$$

### 3.3 Energy Gradient Under Orthogonal Constraints

For the first excited state, considering the orthogonal constraint $\langle \Psi_T^{(1)}|\Psi_0\rangle = 0$, the energy gradient is corrected as:

$$\frac{\partial E^{(1)}}{\partial \alpha_k} = 2\langle E_L^{(1)}(\mathbf{R}) O_k^{(1)}(\mathbf{R}) \rangle - 2\langle E_L^{(1)}(\mathbf{R}) \rangle \langle O_k^{(1)}(\mathbf{R}) \rangle$$

where:

$$E_L^{(1)}(\mathbf{R}) = \frac{\hat{H}\Psi_T^{(1)}(\mathbf{R})}{\Psi_T^{(1)}(\mathbf{R})}$$

$$O_k^{(1)}(\mathbf{R}) = \frac{1}{\Psi_T^{(1)}(\mathbf{R})} \frac{\partial \Psi_T^{(1)}(\mathbf{R})}{\partial \alpha_k}$$

## 4. Specific Implementation Methods

### 4.1 Neural Network Quantum States (NQS)

Use neural networks to represent wave functions:

$$\Psi_T(\mathbf{R}; \boldsymbol{\theta}) = e^{\ln \Psi_{\text{Jastrow}}(\mathbf{R}) + f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})}$$

where $f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})$ is the neural network output.

The logarithmic derivative is:

$$\frac{\partial \ln \Psi_T(\mathbf{R})}{\partial \theta_k} = \frac{\partial f_{\text{NN}}(\mathbf{R}; \boldsymbol{\theta})}{\partial \theta_k}$$

### 4.2 Excited State Construction

#### Orthogonal Constraint Method

Define the orthogonalization projection operator:

$$\hat{P}_1 = 1 - |\Psi_0\rangle\langle\Psi_0|$$

Then the first excited state trial wave function is:

$$\Psi_T^{(1)}(\mathbf{R}) = \hat{P}_1 \Phi(\mathbf{R}) = \Phi(\mathbf{R}) - \Psi_0(\mathbf{R}) \int \Psi_0^*(\mathbf{R}')\Phi(\mathbf{R}') d\mathbf{R}'$$

In Monte Carlo sampling, the orthogonalization coefficient is:

$$c = \frac{\langle \Psi_0|\Phi \rangle}{\langle \Psi_0|\Psi_0 \rangle} = \int \Psi_0^*(\mathbf{R})\Phi(\mathbf{R}) d\mathbf{R}$$

### 4.3 Energy Calculation

For the orthogonalized wave function, the local energy is:

$$E_L^{(1)}(\mathbf{R}) = \frac{\hat{H}\Psi_T^{(1)}(\mathbf{R})}{\Psi_T^{(1)}(\mathbf{R})} = \frac{\hat{H}[\Phi(\mathbf{R}) - c\Psi_0(\mathbf{R})]}{\Phi(\mathbf{R}) - c\Psi_0(\mathbf{R})}$$

## 5. Optimization Algorithms

### 5.1 Stochastic Gradient Descent (SGD)

Parameter update rule:

$$\alpha_k^{(t+1)} = \alpha_k^{(t)} - \eta \frac{\partial E}{\partial \alpha_k}$$

where $\eta$ is the learning rate.

### 5.2 Natural Gradient

Using the Fisher information matrix:

$$F_{kl} = \langle O_k(\mathbf{R}) O_l(\mathbf{R}) \rangle - \langle O_k(\mathbf{R}) \rangle \langle O_l(\mathbf{R}) \rangle$$

Natural gradient update:

$$\boldsymbol{\alpha}^{(t+1)} = \boldsymbol{\alpha}^{(t)} - \eta \mathbf{F}^{-1} \nabla E$$

## 6. Numerical Stability Considerations

### 6.1 Renormalization

To avoid numerical instability, renormalize the wave function:

$$\Psi_T(\mathbf{R}) \rightarrow \frac{\Psi_T(\mathbf{R})}{\sqrt{\langle \Psi_T|\Psi_T \rangle}}$$

### 6.2 Numerical Implementation of Orthogonal Constraints

In Monte Carlo sampling, orthogonal constraints are implemented as follows:

$$\langle \Psi_T^{(1)}|\Psi_0 \rangle = \frac{1}{N}\sum_{i=1}^{N} \frac{\Psi_T^{(1)}(\mathbf{R}_i)\Psi_0(\mathbf{R}_i)}{|\Psi_T^{(1)}(\mathbf{R}_i)|^2} \approx 0$$