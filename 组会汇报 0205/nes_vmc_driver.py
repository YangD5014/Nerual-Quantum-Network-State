"""
NES-VMC 驱动器 - 改进版

策略：分步优化
1. 先优化基态到收敛
2. 固定基态，优化激发态（与基态正交）
3. 计算哈密顿量矩阵对角化得到最终能量
"""

import jax
import jax.numpy as jnp
import netket as nk
from textwrap import dedent
from netket.utils.types import PyTree
from netket.operator import AbstractOperator
from netket.stats import Stats
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from typing import List, Optional
import nes_vmc
import json


class NESVMC:
    """
    NES-VMC 驱动器
    
    支持两种模式：
    1. 同时优化所有态
    2. 分步优化：先优化基态，再优化激发态
    """
    
    def __init__(
        self,
        hamiltonian: AbstractOperator,
        optimizer,
        *,
        variational_states: List[MCState],
        preconditioner: PreconditionerT = None,
        n_states: int = 2,
        penalty_strength: float = 0.5,
        **kwargs,
    ):
        if len(variational_states) != n_states:
            raise ValueError(
                f"variational_states的长度({len(variational_states)})必须等于n_states({n_states})"
            )
        
        for i, vs in enumerate(variational_states):
            if vs.hilbert != hamiltonian.hilbert:
                raise TypeError(
                    dedent(
                        f"""变分态{i}的希尔伯特空间{vs.hilbert}
                        与哈密顿量的希尔伯特空间{hamiltonian.hilbert}不匹配。
                        """
                    )
                )
        
        if preconditioner is None:
            preconditioner = identity_preconditioner
        
        self._vstate_list = variational_states
        self._n_states = n_states
        self._ham = hamiltonian.collect()
        self._preconditioner = preconditioner
        self._penalty_strength = penalty_strength
        
        self._optimizer = optimizer
        self._optimizer_state_list = []
        for i in range(n_states):
            opt_state = optimizer.init(variational_states[i].parameters)
            self._optimizer_state_list.append(opt_state)
        
        self._step_count = 0
        self._H_matrix = None
        self._S_matrix = None
        self._energies = None
        self._coefficients = None
        self._loss_stats = None
        self._fixed_states = []  # 已固定的态索引
    
    def _forward_and_backward(self, fixed_indices: List[int] = None):
        """执行前向传播和反向传播"""
        energy_stats_list, grad_list, H_matrix, S_matrix = nes_vmc.expect_and_grad_nes(
            self._vstate_list,
            self._ham,
            use_covariance=True,
            mutable=self._vstate_list[0].mutable,
            penalty_strength=self._penalty_strength,
            fixed_state_indices=fixed_indices,
        )
        
        self._loss_stats = energy_stats_list[0]
        self._H_matrix = H_matrix
        self._S_matrix = S_matrix
        
        self._energies, self._coefficients = nes_vmc.diagonalize_generalized_eigenvalue_problem(
            H_matrix, S_matrix
        )
        
        dp_list = []
        for i in range(self._n_states):
            if fixed_indices is not None and i in fixed_indices:
                dp_list.append(None)
            else:
                dp_i = self._preconditioner(self._vstate_list[i], grad_list[i])
                dp_i = jax.tree_map(
                    lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
                    dp_i,
                    self._vstate_list[i].parameters,
                )
                dp_list.append(dp_i)
        
        return dp_list
    
    def _step(self, fixed_indices: List[int] = None):
        """执行一步优化"""
        dp_list = self._forward_and_backward(fixed_indices)
        
        for i in range(self._n_states):
            if fixed_indices is not None and i in fixed_indices:
                continue
            
            updates, new_opt_state = self._optimizer.update(
                dp_list[i], 
                self._optimizer_state_list[i],
                self._vstate_list[i].parameters
            )
            self._optimizer_state_list[i] = new_opt_state
            
            new_params = jax.tree_map(
                lambda p, u: p + u,
                self._vstate_list[i].parameters,
                updates
            )
            self._vstate_list[i].parameters = new_params
        
        self._step_count += 1
    
    def optimize_ground_state(self, n_iter: int, show_progress: bool = True):
        """先优化基态"""
        print(f"优化基态 ({n_iter} 迭代)...")
        
        for step in range(n_iter):
            self._step(fixed_indices=[])
            
            if show_progress and step % 20 == 0:
                E_gs = float(self._vstate_list[0].expect(self._ham).mean.real)
                print(f"  Step {step}: E_gs = {E_gs:.8f} Ha")
        
        E_gs = float(self._vstate_list[0].expect(self._ham).mean.real)
        print(f"基态优化完成: E_gs = {E_gs:.8f} Ha")
        return E_gs
    
    def optimize_excited_states(self, n_iter: int, show_progress: bool = True):
        """优化激发态（基态已固定）"""
        print(f"优化激发态 ({n_iter} 迭代)...")
        
        fixed_indices = [0]  # 固定基态
        
        for step in range(n_iter):
            self._step(fixed_indices=fixed_indices)
            
            if show_progress and step % 20 == 0:
                energies = self.get_state_energies()
                print(f"  Step {step}: " + ", ".join([f"E{i}={e:.6f}" for i, e in enumerate(energies)]))
        
        print("激发态优化完成")
    
    def run_sequential(
        self,
        n_iter_ground: int,
        n_iter_excited: int,
        out: Optional[str] = None,
        show_progress: bool = True,
    ):
        """
        分步运行：先优化基态，再优化激发态
        
        这是更稳定的方法
        """
        log_data = {
            "Energy": {"iters": [], "Mean": {"real": [], "imag": []}},
            "Energies": {"iters": [], "values": []},
            "H_matrix": {"iters": [], "values": []},
            "S_matrix": {"iters": [], "values": []},
        }
        
        # 阶段1：优化基态
        print("="*60)
        print("阶段1: 优化基态")
        print("="*60)
        
        for step in range(n_iter_ground):
            self._step(fixed_indices=[])
            
            log_data["Energy"]["iters"].append(step)
            log_data["Energy"]["Mean"]["real"].append(float(self._energies[0].real))
            log_data["Energy"]["Mean"]["imag"].append(float(self._energies[0].imag))
            
            log_data["Energies"]["iters"].append(step)
            log_data["Energies"]["values"].append([float(e.real) for e in self._energies])
            
            H_matrix, S_matrix = nes_vmc.compute_matrices(self._vstate_list, self._ham)
            log_data["H_matrix"]["iters"].append(step)
            log_data["H_matrix"]["values"].append([[float(h.real) for h in row] for row in H_matrix])
            log_data["S_matrix"]["iters"].append(step)
            log_data["S_matrix"]["values"].append([[float(s.real) for s in row] for row in S_matrix])
            
            if show_progress and step % 20 == 0:
                E_gs = float(self._vstate_list[0].expect(self._ham).mean.real)
                print(f"Step {step}: E_gs = {E_gs:.8f} Ha")
        
        # 阶段2：固定基态，优化激发态
        print("\n" + "="*60)
        print("阶段2: 优化激发态 (基态已固定)")
        print("="*60)
        
        for step in range(n_iter_excited):
            self._step(fixed_indices=[0])
            
            total_step = n_iter_ground + step
            log_data["Energy"]["iters"].append(total_step)
            log_data["Energy"]["Mean"]["real"].append(float(self._energies[0].real))
            log_data["Energy"]["Mean"]["imag"].append(float(self._energies[0].imag))
            
            log_data["Energies"]["iters"].append(total_step)
            log_data["Energies"]["values"].append([float(e.real) for e in self._energies])
            
            H_matrix, S_matrix = nes_vmc.compute_matrices(self._vstate_list, self._ham)
            log_data["H_matrix"]["iters"].append(total_step)
            log_data["H_matrix"]["values"].append([[float(h.real) for h in row] for row in H_matrix])
            log_data["S_matrix"]["iters"].append(total_step)
            log_data["S_matrix"]["values"].append([[float(s.real) for s in row] for row in S_matrix])
            
            if show_progress and step % 20 == 0:
                energies = self.get_state_energies()
                overlap = abs(S_matrix[0, 1])
                print(f"Step {step}: E0={energies[0]:.6f}, E1={energies[1]:.6f}, overlap={overlap:.4f}")
        
        if out is not None:
            with open(f"{out}.log", "w") as f:
                json.dump(log_data, f, indent=2)
        
        return log_data
    
    def run(
        self,
        n_iter: int,
        out: Optional[str] = None,
        show_progress: bool = True,
        **kwargs,
    ):
        """同时优化所有态"""
        log_data = {
            "Energy": {"iters": [], "Mean": {"real": [], "imag": []}},
            "Energies": {"iters": [], "values": []},
            "H_matrix": {"iters": [], "values": []},
            "S_matrix": {"iters": [], "values": []},
        }
        
        for step in range(n_iter):
            self._step(fixed_indices=[])
            
            log_data["Energy"]["iters"].append(step)
            log_data["Energy"]["Mean"]["real"].append(float(self._energies[0].real))
            log_data["Energy"]["Mean"]["imag"].append(float(self._energies[0].imag))
            
            log_data["Energies"]["iters"].append(step)
            log_data["Energies"]["values"].append([float(e.real) for e in self._energies])
            
            H_matrix, S_matrix = nes_vmc.compute_matrices(self._vstate_list, self._ham)
            log_data["H_matrix"]["iters"].append(step)
            log_data["H_matrix"]["values"].append([[float(h.real) for h in row] for row in H_matrix])
            log_data["S_matrix"]["iters"].append(step)
            log_data["S_matrix"]["values"].append([[float(s.real) for s in row] for row in S_matrix])
            
            if show_progress and step % 10 == 0:
                print(f"Step {step}: Energies = {[f'{e.real:.6f}' for e in self._energies]}")
        
        if out is not None:
            with open(f"{out}.log", "w") as f:
                json.dump(log_data, f, indent=2)
        
        return log_data
    
    @property
    def energies(self) -> jnp.ndarray:
        return self._energies
    
    @property
    def H_matrix(self) -> jnp.ndarray:
        return self._H_matrix
    
    @property
    def S_matrix(self) -> jnp.ndarray:
        return self._S_matrix
    
    @property
    def step_count(self) -> int:
        return self._step_count
    
    def get_state_energies(self) -> List[float]:
        if self._energies is None:
            return None
        return [float(e.real) for e in self._energies]
    
    def get_excitation_energies(self) -> List[float]:
        if self._energies is None:
            return None
        ground_energy = float(self._energies[0].real)
        return [float(e.real) - ground_energy for e in self._energies[1:]]
    
    def info(self, depth=0):
        lines = [
            f"Hamiltonian    : {self._ham}",
            f"Optimizer      : {self._optimizer}",
            f"Preconditioner : {self._preconditioner}",
            f"N States       : {self._n_states}",
            f"Penalty        : {self._penalty_strength}",
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
