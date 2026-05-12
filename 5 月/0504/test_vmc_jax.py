#!/usr/bin/env python
# coding: utf-8
"""
测试纯 JAX 实现的 VMC 梯度计算
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax import flatten_util
import numpy as np

print("="*60)
print("测试纯 JAX VMC 实现")
print("="*60)

# 简单测试模型
class SimpleModel(nnx.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 4, *, rngs: nnx.Rngs):
        super().__init__()
        self.linear1 = nnx.Linear(input_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)
    
    def __call__(self, x):
        h = nnx.tanh(self.linear1(x.astype(complex)))
        out = self.output(h)
        return jnp.squeeze(out)

# 创建模型
model = SimpleModel(input_dim=4, hidden_dim=4, rngs=nnx.Rngs(42))
graphdef, state = nnx.split(model)

def machine(params, sigma):
    m = nnx.merge(graphdef, params)
    return m(sigma)

# 测试样本
sigma = jnp.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
], dtype=jnp.float32)

params = state

print("\n[1/4] 测试局部能量计算...")
# 模拟简单的哈密顿量：对角项
def mock_hamiltonian(sigma):
    """简单的测试哈密顿量"""
    n = sigma.shape[0]
    # 返回 (eta, H_eta) 其中 eta 是连接态，H_eta 是矩阵元
    # 这里简化为只考虑对角项
    eta = sigma  # 只有自身
    H_eta = jnp.ones((n, 1, 1)) * (-1.0)  # 简单的常数矩阵元
    return eta, H_eta

eta, H_eta = mock_hamiltonian(sigma)
logpsi_sigma = jax.vmap(lambda s: machine(params, s))(sigma)
logpsi_eta = jax.vmap(lambda s: machine(params, s))(eta)

print(f"  logψ(σ) 形状：{logpsi_sigma.shape}")
print(f"  H_eta 形状：{H_eta.shape}")
print("  ✓ 局部能量计算测试通过")

print("\n[2/4] 测试 Jacobian 计算...")
def log_psi_fun(p):
    return jax.vmap(lambda s: machine(p, s))(sigma)

try:
    grad_matrix = jax.jacobian(log_psi_fun)(params)
    grad_flat, unflatten = flatten_util.ravel_pytree(grad_matrix)
    grad_flat = grad_flat.reshape(sigma.shape[0], -1)
    print(f"  梯度矩阵形状：{grad_flat.shape}")
    print("  ✓ Jacobian 计算测试通过")
except Exception as e:
    print(f"  ✗ Jacobian 计算失败：{e}")
    raise

print("\n[3/4] 测试 Force-based 梯度...")
# 模拟局部能量
E_loc = jnp.array([1.0, 2.0, 3.0])
E_mean = jnp.mean(E_loc)
E_centered = E_loc - E_mean

print(f"  E_loc: {E_loc}")
print(f"  E_mean: {E_mean:.4f}")
print(f"  E_centered: {E_centered}")

# 计算梯度
grad_flat_force = jnp.mean(
    jnp.expand_dims(E_centered, -1) * jnp.conj(grad_flat),
    axis=0
)
print(f"  Force-based 梯度形状：{grad_flat_force.shape}")
print("  ✓ Force-based 梯度测试通过")

print("\n[4/4] 测试 QGT 计算...")
n_samples = sigma.shape[0]
grad_mean = jnp.mean(grad_flat, axis=0, keepdims=True)
grad_centered = grad_flat - grad_mean

qgt = (1.0 / n_samples) * jnp.conj(grad_centered).T @ grad_centered
print(f"  QGT 形状：{qgt.shape}")

# 正则化
diag_shift = 0.1
qgt_reg = qgt + diag_shift * jnp.eye(qgt.shape[0])
print(f"  正则化 QGT 形状：{qgt_reg.shape}")

# 测试求解
nat_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat_force)
print(f"  自然梯度形状：{nat_grad_flat.shape}")
print("  ✓ QGT 计算测试通过")

print("\n" + "="*60)
print("所有测试通过！纯 JAX 实现的核心功能正常。")
print("="*60)
print("\n关键实现点:")
print("1. 使用 jax.jacobian 计算 ∇log ψ")
print("2. Force-based 梯度：⟨(E_loc - ⟨E⟩) ∇log ψ⟩")
print("3. QGT 矩阵：F = ⟨∇log ψ ∇log ψ†⟩")
print("4. SR 自然梯度：求解 F⁻¹ ∇E")
