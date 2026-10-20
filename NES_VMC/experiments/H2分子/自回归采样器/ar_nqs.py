"""因果掩码自回归 NQS + 约束自回归精确采样器（H2/6-31G，8 自旋轨道）。

基于「FFNN振幅相位分离实数版-典藏.ipynb」的实参数振幅-相位分离设计，
将 FFNN 重参数化为**自回归（因果掩码）网络**：

    logψ(x) = Σ_i [ ½·log π̃_i(x_i | x_<i)  +  i·( φm_i(x_<i) + x_i·φd_i(x_<i) ) ]

其中 π̃_i 由网络逐位置输出的对数几率 ℓ_i(x_<i) 给出，并在粒子数约束的
可行取值集上 renormalize（H2/6-31G：site 0-3 上旋段恰 1 个电子，
site 4-7 下旋段恰 1 个电子）。由此 |ψ(x)|² = Π_i π̃_i(x_i|x_<i) 在物理
扇区上**严格归一化**，无需全局配分函数。

采样：逐位顺序抽样 x_1→x_N，每步只做一次前向（因果掩码保证输出 i 仅依赖
x_<i），得到 i.i.d. 精确样本，彻底取代 Metropolis MCMC。支持 machine_pow
参数：pow=2 直接从 |ψ|² 精确采样（w 恒为 1）；pow=1 从 q ∝ |ψ| 采样并
返回重要性权重 w = |ψ|²/q，配合加权估计器（沿用典藏 notebook 的修复方案，
防止塌缩冻结）。

主算法（训练/梯度/QGT）全程 Matrix-Free：只调用 ha.get_conn_padded，
不做任何 2^N 全空间枚举。全空间仅出现在本文件的归一化/采样正确性自检
与训练脚本的诊断中。
"""

import jax
import jax.numpy as jnp
from functools import partial
from flax import nnx
from jax.scipy.special import logsumexp


# ============================================================
# 粒子数约束的段布局（H2/6-31G：两段各 4 个自旋轨道、各 1 个电子）
# ============================================================
class SegmentLayout:
    """按自旋段描述固定粒子数约束。specs: [(start, end, n_elec), ...]"""

    def __init__(self, n_bits, specs):
        self.n = n_bits
        self.specs = specs
        start = jnp.zeros(n_bits, dtype=jnp.int32)
        end = jnp.zeros(n_bits, dtype=jnp.int32)
        target = jnp.zeros(n_bits, dtype=jnp.int32)
        for s, e, ne in specs:
            start = start.at[s:e].set(s)
            end = end.at[s:e].set(e)
            target = target.at[s:e].set(ne)
        self.start, self.end, self.target = start, end, target

    def prefix_counts(self, x):
        """c[...,i] = 段内位置 i 之前已放置的电子数（x 为 (...,n) 的 0/1 组态）。"""
        c = jnp.zeros_like(x, dtype=jnp.int32)
        for s, e, _ in self.specs:
            seg = x[..., s:e]
            c = c.at[..., s:e].set(jnp.cumsum(seg, axis=-1) - seg)  # exclusive 前缀和
        return c

    def feasible(self, x):
        """返回 (feas0, feas1)：位置 i 放 0 / 放 1 是否仍可完成约束。"""
        c = self.prefix_counts(x)
        rem = self.target[None, :] - c                              # 含 i 在内还需放置的电子数
        slots = self.end[None, :] - jnp.arange(self.n)[None, :]     # 含 i 在内剩余槽位数
        feas1 = rem >= 1
        feas0 = rem <= slots - 1
        return feas0, feas1


# ============================================================
# 因果掩码 MLP：输出每个位置的 (ℓ_i, φm_i, φd_i)
# ============================================================
class CausalMaskedARNQS(nnx.Module):
    """
    实参数、复输出的自回归 NQS。
    结构：掩码层1 (n → n·h)：单位 (i,k) 仅看 x_j, j<i；
          掩码层2 (n·h → n·h)：单位 (i,k) 仅看层1位置 ≤ i；
          掩码头 (n·h → 3·n)：输出 i 仅看层2位置 ≤ i。
    因此输出 i 严格只依赖 x_<i（因果性）。
    """

    def __init__(self, n_bits: int, hidden_per_site: int = 4, *, rngs: nnx.Rngs):
        self.n = n_bits
        self.h = hidden_per_site
        d = n_bits * hidden_per_site
        self.l1 = nnx.Linear(n_bits, d, rngs=rngs, param_dtype=float)
        self.l2 = nnx.Linear(d, d, rngs=rngs, param_dtype=float)
        self.head = nnx.Linear(d, 3 * n_bits, rngs=rngs, param_dtype=float)

    def _masks(self):
        n, h = self.n, self.h
        idx = jnp.arange(n)
        lt = idx[:, None] > idx[None, :]   # lt[i,j] = (j < i)
        le = idx[:, None] <= idx[None, :]  # le[i,j] = (j <= i)
        # nnx.Linear kernel 布局 (in, out)
        # kernel 布局 (in, out)：M1[j, i*h+k] = (j<i)；M2[(j,m),(i,k)] = (j<=i)
        M1 = jnp.broadcast_to(lt.T[:, :, None], (n, n, h)).reshape(n, n * h)
        M2 = jnp.broadcast_to(le[:, None, :, None], (n, h, n, h)).reshape(n * h, n * h)
        M3 = jnp.broadcast_to(le[:, None, :, None], (n, h, n, 3)).reshape(n * h, 3 * n)
        return M1, M2, M3

    def __call__(self, x):
        """x: (..., n) 0/1 → out: (..., n, 3)，通道为 (ℓ, φm, φd)。"""
        x = x.astype(float)
        M1, M2, M3 = self._masks()
        hh = jnp.tanh(x @ (self.l1.kernel * M1) + self.l1.bias)
        hh = jnp.tanh(hh @ (self.l2.kernel * M2) + self.l2.bias)
        out = hh @ (self.head.kernel * M3) + self.head.bias
        return out.reshape(*x.shape[:-1], self.n, 3)


# ============================================================
# 由网络输出拼装 logψ 与逐位置条件分布
# ============================================================
def logpsi_from_out(out, x, layout: SegmentLayout):
    """
    out: (B,n,3) 网络输出；x: (B,n) int01 物理组态。
    logπ̃_i(b) = (2b-1)·ℓ_i - logsumexp_{b' feasible}( (2b'-1)·ℓ_i )
    Re logψ = ½ Σ logπ̃_i(x_i)；Im logψ = Σ (φm_i + x_i·φd_i)
    """
    ell = out[..., 0]
    phm = out[..., 1]
    phd = out[..., 2]
    feas0, feas1 = layout.feasible(x)
    neg = jnp.where(feas0, -ell, -jnp.inf)
    pos = jnp.where(feas1, ell, -jnp.inf)
    logD = logsumexp(jnp.stack([neg, pos], axis=-1), axis=-1)
    logpi = (2.0 * x - 1.0) * ell - logD
    re = 0.5 * jnp.sum(logpi, axis=-1)
    im = jnp.sum(phm + x.astype(float) * phd, axis=-1)
    return re + 1j * im


def create_machine(model, layout: SegmentLayout):
    """返回 (machine, net_out, graphdef, state)。machine 支持 (n,) 或 (B,n) 输入。"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def net_out(params, sigma):
        sigma = jnp.atleast_2d(sigma)
        m = nnx.merge(graphdef, params)
        return m(sigma)

    @jax.jit
    def machine(params, sigma):
        single = sigma.ndim == 1
        lp = logpsi_from_out(net_out(params, jnp.atleast_2d(sigma)), jnp.atleast_2d(sigma), layout)
        return lp[0] if single else lp

    return machine, net_out, graphdef, state


# ============================================================
# 自回归精确采样器（无 MCMC、无全空间枚举）
# ============================================================
def make_ar_sampler(net_out, layout: SegmentLayout, n_bits: int):
    """
    逐位顺序抽样。machine_pow=2：精确服从 |ψ|²（logw 恒为 0）；
    machine_pow=1：服从 q ∝ |ψ|，返回 w = |ψ|²/q（未归一化）。
    返回 (sigma, logw)，sigma: (B, n) int01。
    """

    @partial(jax.jit, static_argnames=("n_walkers",))
    def sample(params, rng, n_walkers, machine_pow=1.0):
        x = jnp.zeros((n_walkers, n_bits), dtype=jnp.int32)
        logw = jnp.zeros(n_walkers, dtype=float)
        for i in range(n_bits):
            rng, key = jax.random.split(rng)
            out = net_out(params, x)
            ell = out[:, i, 0]
            feas0, feas1 = layout.feasible(x)
            f0 = feas0[:, i]
            f1 = feas1[:, i]
            # q_i(b) ∝ exp( (p/2)·(2b-1)·ℓ_i )，p = machine_pow
            u0 = jnp.where(f0, -0.5 * machine_pow * ell, -jnp.inf)
            u1 = jnp.where(f1, 0.5 * machine_pow * ell, -jnp.inf)
            logZq = logsumexp(jnp.stack([u0, u1], axis=-1), axis=-1)
            p1 = jnp.exp(u1 - logZq)
            p1 = jnp.where(f1 & ~f0, 1.0, jnp.where(f0 & ~f1, 0.0, p1))
            b = (jax.random.uniform(key, (n_walkers,)) < p1).astype(jnp.int32)
            x = x.at[:, i].set(b)
            # log q_i(b) = (p/2)(2b-1)ℓ - logZq
            logq = 0.5 * machine_pow * (2.0 * b - 1.0) * ell - logZq
            # log π̃_i(b)
            d0 = jnp.where(f0, -ell, -jnp.inf)
            d1 = jnp.where(f1, ell, -jnp.inf)
            logD = logsumexp(jnp.stack([d0, d1], axis=-1), axis=-1)
            logpi = (2.0 * b - 1.0) * ell - logD
            logw = logw + logpi - logq
        return x, logw

    return sample
