import jax
import jax.numpy as jnp
import random


def make_get_all_next_states(edges):
    edges = tuple(tuple(e) for e in edges)

    @jax.jit
    def get_all_next_states_jit(S: jnp.ndarray):
        s_arr = S
        next_states = []
        masks = []

        for (i, j) in edges:
            occ_i = s_arr[i]
            occ_j = s_arr[j]
            valid = ((occ_i == 1) & (occ_j == 0)) | ((occ_i == 0) & (occ_j == 1))
            new_state = s_arr.at[i].set(occ_j).at[j].set(occ_i)
            next_states.append(new_state)
            masks.append(valid)

        return jnp.stack(next_states, axis=0), jnp.stack(masks, axis=0)

    return get_all_next_states_jit


def make_metropolis_hastings_step(edges, machine, params):
    get_all_next = make_get_all_next_states(edges)

    @jax.jit
    def metropolis_hastings_step_jit(S: jnp.ndarray, key: jax.Array):
        candidates, valid_mask = get_all_next(S)
        key, subk = jax.random.split(key)
        idx = jax.random.choice(subk, candidates.shape[0])
        S_cand = candidates[idx]
        is_valid = valid_mask[idx]

        log_psi_curr = machine(params, S)
        log_psi_cand = machine(params, S_cand)
        log_accept_ratio = 2 * jnp.real(log_psi_cand - log_psi_curr)

        key, subk = jax.random.split(key)
        u = jax.random.uniform(subk)
        accept = is_valid & (log_accept_ratio > jnp.log(u))

        S_new = jnp.where(accept, S_cand, S)
        return S_new, accept, key

    return metropolis_hastings_step_jit


def _single_chain_mcmc(n_samples, n_warmup, initial_state, edges, machine, params, seed):
    mh_step = make_metropolis_hastings_step(edges, machine, params)
    key = jax.random.PRNGKey(seed)

    def warmup_body(carry, _):
        state, rng = carry
        state, _, rng = mh_step(state, rng)
        return (state, rng), None

    (current_state, key), _ = jax.lax.scan(warmup_body, (initial_state, key), length=n_warmup)

    def sample_body(carry, _):
        state, rng = carry
        state, _, rng = mh_step(state, rng)
        return (state, rng), state

    (_, _), samples = jax.lax.scan(sample_body, (current_state, key), length=n_samples)

    return samples


def mcmc_sampler_multi_chain(
    n_samples: int,
    n_warmup: int,
    initial_states: jnp.ndarray,
    edges: tuple[tuple[int, int]],
    machine: callable,
    params,
    n_chains: int = None,
    seed: int = 42
):
    """
    多链 MCMC 采样器 - 支持并行多链采样

    参数：
    - n_samples: 每个链的样本数
    - n_warmup: 预烧步数
    - initial_states: 初始状态 (n_chains, n_orbitals)
    - edges: 跃迁边
    - machine: 波函数机器 machine(params, sigma)
    - params: 网络参数
    - n_chains: 链的数量（自动从 initial_states.shape[0] 推断）
    - seed: 随机种子

    返回：
    - samples: 所有链的样本 (n_chains, n_samples, n_orbitals)
    """
    if n_chains is None:
        n_chains = initial_states.shape[0]

    rng = random.Random(seed)

    all_samples = []
    for i in range(n_chains):
        key_seed = rng.randint(0, 2**31 - 1)
        samples = _single_chain_mcmc(
            n_samples, n_warmup, initial_states[i], edges, machine, params, key_seed
        )
        all_samples.append(samples)

    return jnp.stack(all_samples, axis=0)


def mcmc_sampler_multi_chain_total(
    n_total_samples: int,
    n_warmup: int,
    initial_state: jnp.ndarray,
    edges: tuple[tuple[int, int]],
    machine: callable,
    params,
    n_chains: int = 32,
    seed: int = 42
):
    """
    多链 MCMC 采样器（总样本数模式）- 自动分配每个链的样本数

    参数：
    - n_total_samples: 总样本数
    - n_warmup: 预烧步数
    - initial_state: 单个初始状态，会被复制到 n_chains 条链
    - edges: 跃迁边
    - machine: 波函数机器 machine(params, sigma)
    - params: 网络参数
    - n_chains: 链的数量（默认 32）
    - seed: 随机种子

    返回：
    - samples: 所有样本 (n_total_samples, n_orbitals)
    """
    samples_per_chain = n_total_samples // n_chains
    remainder = n_total_samples % n_chains

    initial_states = jnp.stack([initial_state] * n_chains)

    samples = mcmc_sampler_multi_chain(
        n_samples=samples_per_chain,
        n_warmup=n_warmup,
        initial_states=initial_states,
        edges=edges,
        machine=machine,
        params=params,
        n_chains=n_chains,
        seed=seed
    )

    samples = samples.reshape(-1, initial_state.shape[0])

    if remainder > 0:
        extra_samples = mcmc_sampler_multi_chain(
            n_samples=1,
            n_warmup=0,
            initial_states=initial_states[:remainder],
            edges=edges,
            machine=machine,
            params=params,
            n_chains=remainder,
            seed=seed + 10000
        )
        samples = jnp.concatenate([samples, extra_samples[:, 0, :]], axis=0)

    return samples


def sample_to_flat_samples(samples):
    """
    将多链采样结果展平为 (total_samples, n_orbitals) 的二维数组

    参数：
    - samples: (n_chains, n_samples_per_chain, n_orbitals)

    返回：
    - flat_samples: (n_chains * n_samples_per_chain, n_orbitals)
    """
    n_chains, n_samples_per_chain, n_orbitals = samples.shape
    return samples.reshape(n_chains * n_samples_per_chain, n_orbitals)
