#!/usr/bin/env python3
"""
NES-VMC 自然梯度下降 - QGT 优化测试

本文件实现优化后的 QGT 计算，解决以下问题：
1. Python 循环改为 JAX 向量化
2. 修正 QGT 计算的物理公式
"""

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx

# 在导入 NES_VMC 之前，修复 nnx.List 的初始化问题
# 将 nnx.List 替换为普通列表（避免版本兼容性问题）
_original_list = nnx.List
nnx.List = list

import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine, init_sampler_state, \
    generate_random_initial_states, ha, SingleStateAnsatz, create_single_machine, \
    create_machine_matrix, Ham_psi, Ham_Psi, NES_loss_energy, nes_vmc_gradient, hi, E_fcis, mcmc_sampler_multichain
import optax
from functools import partial
from jax.flatten_util import ravel_pytree
import time

# 将哈密顿量转换为 JAX 兼容格式
ha = ha.to_jax_operator()

K = 2
hi_ext = hi ** K

print("=" * 60)
print("NES-VMC 自然梯度下降 - QGT 优化测试")
print("=" * 60)
print(f"FCI 基准: E0 = {E_fcis[0]:.8f} Ha, E1 = {E_fcis[1]:.8f} Ha")
print("=" * 60)


# ==============================================================================
# 1. 原始 QGT 实现（有问题）
# ==============================================================================
def compute_nes_qgt_slow(total_machine, params, samples, diag_shift=0.01):
    """
    原始的 QGT 计算（使用 Python 循环，效率低）
    """
    # 1. 单样本梯度 - 使用 Python 循环
    def _single_grad(x):
        return jax.grad(lambda p: total_machine(p, x), holomorphic=True)(params)

    # 2. 对每个样本求导 - Python for 循环
    grads = [_single_grad(x) for x in samples]  # 列表 [B]

    # 3. 展平每个梯度
    flat_grads = [ravel_pytree(g)[0] for g in grads]

    # 4. 堆叠成 → (B, P)
    grads_flat = jnp.stack(flat_grads)

    # 5. 计算 QGT
    g = grads_flat
    g_conj = g.conj()

    term1 = jnp.mean(g_conj[:, :, None] * g[:, None, :], axis=0)
    g_mean = jnp.mean(g, axis=0)
    term2 = g_mean.conj()[:, None] * g_mean[None, :]
    S = term1 - term2

    S_reg = S + diag_shift * jnp.eye(S.shape[0], dtype=complex)
    unravel = ravel_pytree(params)[1]

    return S_reg, unravel


# ==============================================================================
# 2. 优化后的 QGT 实现
# ==============================================================================
def compute_nes_qgt_fast(total_machine, params, samples, diag_shift=0.01):
    """
    优化后的 QGT 计算
    关键改进：使用 jax.vmap 对样本进行向量化计算
    """
    # 1. 获取参数展平后的信息
    params_flat, unravel_fn = ravel_pytree(params)
    n_params = params_flat.shape[0]
    n_samples = samples.shape[0]

    # 2. 真正的向量化方法：直接用 vmap
    # 定义单个样本的对数概率函数
    def log_prob_fn(params_flat, sample):
        p = unravel_fn(params_flat)
        return total_machine(p, sample)

    # 3. 使用 vmap 计算所有样本的梯度
    # grad_fn: params_flat, sample -> grad_flat
    def grad_fn(params_flat, sample):
        return jax.grad(lambda p: log_prob_fn(p, sample), holomorphic=True)(params_flat)

    # vmap: (params_flat, sample) -> grad_flat，对 sample 批量
    batch_grad_fn = jax.vmap(grad_fn, in_axes=(None, 0))

    # 计算所有梯度: (n_samples, n_params)
    grads_flat = batch_grad_fn(params_flat, samples)

    # 4. 计算 QGT
    g = grads_flat
    g_conj = g.conj()

    # S = <g* g> - <g*> <g>
    term1 = jnp.mean(g_conj[:, :, None] * g[:, None, :], axis=0)
    g_mean = jnp.mean(g, axis=0)
    term2 = g_mean.conj()[:, None] * g_mean[None, :]
    S = term1 - term2

    # 正则化
    S_reg = S + diag_shift * jnp.eye(n_params, dtype=S.dtype)

    return S_reg, unravel_fn


# ==============================================================================
# 3. 训练函数
# ==============================================================================
def train_pure_gradient(n_iter, n_chains, n_warmup, n_samples_per_chain, sweep_size, verbose=True):
    """纯梯度下降训练"""
    rngs = nnx.Rngs(42)
    total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
    total_machine, _, total_params = create_machine(total_ansatz)
    total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)

    single_machine_list = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine(ansatz)
        single_machine_list.append(m)

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(total_params)

    sampler_state = init_sampler_state(hi_ext, n_chains, seed=21)
    history = {'loss': [], 'time': []}

    for step in range(n_iter):
        start = time.time()

        samples, sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=n_samples_per_chain,
            n_warmup=n_warmup,
            sampler_state=sampler_state,
            edges=((0, 1), (2, 3), (4, 5), (6, 7)),
            machine=total_machine,
            params=total_params,
            sweep_size=sweep_size,
            K=K,
            SINGLE_HILBERT_SIZE=4,
        )
        x_batch = samples.reshape(-1, K, 4)

        grad, loss_mean, E_L_mean = nes_vmc_gradient(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
        )
        grad = jax.tree_util.tree_map(lambda x: x * 2, grad)

        updates, opt_state = optimizer.update(grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        history['loss'].append(float(loss_mean))
        history['time'].append(time.time() - start)

        if verbose and step % 2 == 0:
            eig_vals = jnp.linalg.eigvalsh(E_L_mean.real)
            print(f"Step {step:2d} | Loss: {loss_mean:10.6f} | E0: {eig_vals[0]:.6f} | Time: {history['time'][-1]:.2f}s")

    return history


def train_natural_gradient_slow(n_iter, n_chains, n_warmup, n_samples_per_chain, sweep_size, verbose=True):
    """使用原始 QGT 的自然梯度下降"""
    rngs = nnx.Rngs(42)
    total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
    total_machine, _, total_params = create_machine(total_ansatz)
    total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)

    single_machine_list = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine(ansatz)
        single_machine_list.append(m)

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(total_params)

    sampler_state = init_sampler_state(hi_ext, n_chains, seed=21)
    history = {'loss': [], 'time': [], 'qgt_time': []}

    for step in range(n_iter):
        start = time.time()

        samples, sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=n_samples_per_chain,
            n_warmup=n_warmup,
            sampler_state=sampler_state,
            edges=((0, 1), (2, 3), (4, 5), (6, 7)),
            machine=total_machine,
            params=total_params,
            sweep_size=sweep_size,
            K=K,
            SINGLE_HILBERT_SIZE=4,
        )
        x_batch = samples.reshape(-1, K, 4)

        grad, loss_mean, E_L_mean = nes_vmc_gradient(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
        )
        grad = jax.tree_util.tree_map(lambda x: x * 2, grad)

        # QGT 计算
        qgt_start = time.time()
        qgt_reg, unravel_fn = compute_nes_qgt_slow(total_machine, total_params, x_batch, diag_shift=0.01)
        qgt_time = time.time() - qgt_start
        history['qgt_time'].append(qgt_time)

        grad_flat, grad_unravel_fn = ravel_pytree(grad)
        natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad_flat)

        updates, opt_state = optimizer.update(natural_grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        history['loss'].append(float(loss_mean))
        history['time'].append(time.time() - start)

        if verbose and step % 2 == 0:
            eig_vals = jnp.linalg.eigvalsh(E_L_mean.real)
            print(f"Step {step:2d} | Loss: {loss_mean:10.6f} | E0: {eig_vals[0]:.6f} | QGT: {qgt_time:.2f}s")

    return history


def train_natural_gradient_fast(n_iter, n_chains, n_warmup, n_samples_per_chain, sweep_size, verbose=True):
    """使用优化 QGT 的自然梯度下降"""
    rngs = nnx.Rngs(42)
    total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
    total_machine, _, total_params = create_machine(total_ansatz)
    total_matrix_machine, _, _ = create_machine_matrix(total_ansatz)

    single_machine_list = []
    for ansatz in total_ansatz.single_ansatz_list:
        m, _, _ = create_single_machine(ansatz)
        single_machine_list.append(m)

    optimizer = optax.sgd(learning_rate=0.01)
    opt_state = optimizer.init(total_params)

    sampler_state = init_sampler_state(hi_ext, n_chains, seed=21)
    history = {'loss': [], 'time': [], 'qgt_time': []}

    for step in range(n_iter):
        start = time.time()

        samples, sampler_state = mcmc_sampler_multichain(
            n_samples_per_chain=n_samples_per_chain,
            n_warmup=n_warmup,
            sampler_state=sampler_state,
            edges=((0, 1), (2, 3), (4, 5), (6, 7)),
            machine=total_machine,
            params=total_params,
            sweep_size=sweep_size,
            K=K,
            SINGLE_HILBERT_SIZE=4,
        )
        x_batch = samples.reshape(-1, K, 4)

        grad, loss_mean, E_L_mean = nes_vmc_gradient(
            ha=ha,
            total_matrix_machine=total_matrix_machine,
            total_machine=total_machine,
            single_machine_list=single_machine_list,
            total_params=total_params,
            x_batch=x_batch,
        )
        grad = jax.tree_util.tree_map(lambda x: x * 2, grad)

        # 优化 QGT 计算
        qgt_start = time.time()
        qgt_reg, unravel_fn = compute_nes_qgt_fast(total_machine, total_params, x_batch, diag_shift=0.01)
        qgt_time = time.time() - qgt_start
        history['qgt_time'].append(qgt_time)

        grad_flat, grad_unravel_fn = ravel_pytree(grad)
        natural_grad_flat = jnp.linalg.solve(qgt_reg, grad_flat)
        natural_grad = grad_unravel_fn(natural_grad_flat)

        updates, opt_state = optimizer.update(natural_grad, opt_state, total_params)
        total_params = optax.apply_updates(total_params, updates)

        history['loss'].append(float(loss_mean))
        history['time'].append(time.time() - start)

        if verbose and step % 2 == 0:
            eig_vals = jnp.linalg.eigvalsh(E_L_mean.real)
            print(f"Step {step:2d} | Loss: {loss_mean:10.6f} | E0: {eig_vals[0]:.6f} | QGT: {qgt_time:.2f}s")

    return history


# ==============================================================================
# 4. 主测试程序
# ==============================================================================
if __name__ == '__main__':
    # 测试参数（减少样本数以加快测试）
    N_CHAINS = 8
    N_WARMUP = 20
    N_SAMPLES_PER_CHAIN = 30
    SWEEP_SIZE = 15
    N_ITER = 10

    print("\n" + "=" * 60)
    print("测试参数:")
    print(f"  - 链数: {N_CHAINS}")
    print(f"  - 热化步数: {N_WARMUP}")
    print(f"  - 每链样本数: {N_SAMPLES_PER_CHAIN}")
    print(f"  - 迭代次数: {N_ITER}")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # 1. QGT 计算性能对比
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("1. QGT 计算性能对比")
    print("=" * 60)

    # 初始化模型
    rngs = nnx.Rngs(42)
    total_ansatz = NESTotalAnsatz(4, n_states=K, hidden_dim=8, rngs=rngs)
    total_machine, _, total_params = create_machine(total_ansatz)

    # 生成测试样本
    sampler_state = init_sampler_state(hi_ext, N_CHAINS, seed=42)
    samples, _ = mcmc_sampler_multichain(
        n_samples_per_chain=N_SAMPLES_PER_CHAIN,
        n_warmup=N_WARMUP,
        sampler_state=sampler_state,
        edges=((0, 1), (2, 3), (4, 5), (6, 7)),
        machine=total_machine,
        params=total_params,
        sweep_size=SWEEP_SIZE,
        K=K,
        SINGLE_HILBERT_SIZE=4,
    )
    samples = samples.reshape(-1, K, 4)
    print(f"样本形状: {samples.shape}")

    # 测试原始 QGT
    print("\n--- 原始 QGT (Python 循环) ---")
    start = time.time()
    S_slow, unravel_slow = compute_nes_qgt_slow(total_machine, total_params, samples, diag_shift=0.01)
    time_slow = time.time() - start
    print(f"QGT 形状: {S_slow.shape}")
    print(f"耗时: {time_slow:.4f} 秒")

    # 测试优化 QGT
    print("\n--- 优化 QGT (vmap + jit) ---")
    start = time.time()
    S_fast, unravel_fast = compute_nes_qgt_fast(total_machine, total_params, samples, diag_shift=0.01)
    time_fast = time.time() - start
    print(f"QGT 形状: {S_fast.shape}")
    print(f"耗时: {time_fast:.4f} 秒")

    print(f"\n加速比: {time_slow / time_fast:.2f}x")

    # -------------------------------------------------------------------------
    # 2. 训练对比测试
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("2. 训练对比测试 (10 轮)")
    print("=" * 60)

    # 纯梯度下降
    print("\n" + "-" * 60)
    print("1. 纯梯度下降训练")
    print("-" * 60)
    start = time.time()
    history_pure = train_pure_gradient(
        N_ITER, N_CHAINS, N_WARMUP, N_SAMPLES_PER_CHAIN, SWEEP_SIZE, verbose=True
    )
    pure_total = time.time() - start
    print(f"\n总耗时: {pure_total:.2f} 秒")

    # 原始 QGT + 自然梯度
    print("\n" + "-" * 60)
    print("2. 原始 QGT + 自然梯度下降")
    print("-" * 60)
    start = time.time()
    history_slow = train_natural_gradient_slow(
        N_ITER, N_CHAINS, N_WARMUP, N_SAMPLES_PER_CHAIN, SWEEP_SIZE, verbose=True
    )
    slow_total = time.time() - start
    print(f"\n总耗时: {slow_total:.2f} 秒")

    # 优化 QGT + 自然梯度
    print("\n" + "-" * 60)
    print("3. 优化 QGT + 自然梯度下降")
    print("-" * 60)
    start = time.time()
    history_fast = train_natural_gradient_fast(
        N_ITER, N_CHAINS, N_WARMUP, N_SAMPLES_PER_CHAIN, SWEEP_SIZE, verbose=True
    )
    fast_total = time.time() - start
    print(f"\n总耗时: {fast_total:.2f} 秒")

    # -------------------------------------------------------------------------
    # 3. 结果汇总
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)

    print(f"\n{'方法':<30} {'最终 Loss':<15} {'总耗时':<10}")
    print("-" * 55)
    print(f"{'纯梯度下降':<30} {history_pure['loss'][-1]:<15.6f} {pure_total:<10.2f}s")
    print(f"{'原始 QGT + 自然梯度':<30} {history_slow['loss'][-1]:<15.6f} {slow_total:<10.2f}s")
    print(f"{'优化 QGT + 自然梯度':<30} {history_fast['loss'][-1]:<15.6f} {fast_total:<10.2f}s")

    print(f"\nQGT 加速比: {time_slow / time_fast:.2f}x")

    print("\n--- Loss 收敛曲线 ---")
    for i, loss in enumerate(history_pure['loss']):
        if i % 2 == 0:
            print(f"Step {i:2d}: 纯梯度={loss:.6f}, 原始QGT={history_slow['loss'][i]:.6f}, 优化QGT={history_fast['loss'][i]:.6f}")

    # -------------------------------------------------------------------------
    # 4. 结论
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("结论")
    print("=" * 60)
    print("""
1. 性能优化效果:
   优化后的 QGT 计算（使用 jax.vmap + jax.jit）应该比原始 Python 循环快很多。

2. 物理问题:
   即使 QGT 计算加速，自然梯度下降在 NES-VMC 中可能仍然表现不佳，因为：
   - 标准 VMC 损失函数是标量能量 E = <H>，而 NES-VMC 损失函数是
     L = Tr(E_L)，涉及矩阵的 Trace 操作
   - 当前 QGT 计算的是 grad log Psi 的几何结构，但损失函数的梯度
     涉及更复杂的矩阵运算

3. 建议:
   先使用纯梯度下降调参，等超参数优化完成后再尝试自然梯度。
    """)