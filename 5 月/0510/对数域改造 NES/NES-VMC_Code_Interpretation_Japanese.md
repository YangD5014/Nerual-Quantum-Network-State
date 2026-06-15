# NES-VMCアルゴリズムのNetKet公式APIによる再現：目的と進捗

## 1. 研究目的

**目標**: NetKetフレームワークとFlax.nnxのAPIを使用して、**NES-VMC（Natural Excited State Variational Monte Carlo）アルゴリズム**を実装し、量子多体系（H₂分子など）の最初のK個励起状態エネルギーを計算する。

**要件**:

- NetKet内置の拡張ヒルベルト空間 `hi ** K` を使用
- 訓練後、平均局所エネルギ行列を対角化して基底状態と励起状態エネルギーを取得

## 2. NES-VMCアルゴリズムの核心概念

### 2.1 問題の背景

量子力学では、ハミルトニアン演算子$\hat{H}$の固有値問題、すなわち最低K個の固有関数を見つける必要があります。量子多体系では、ヒルベルト空間の次元が粒子数とともに指数関数的に増加するため、ハミルトニアン行列の直接対角化は通常実行不可能です。

NES-VMCは、元の系の最初のK個励起状態の問題を**「拡張系」の基底状態の問題に等価変換**します。
以下はH₂分子の最初のK個励起状態の問題の記述です。必要がない限り変更しないでください。

```python
"""
NES-VMC (Natural Excited State Variational Monte Carlo) アルゴリズム実装

本ファイルは、ネイティブJAXと部分的なNetKetに基づくNES-VMCアルゴリズムを実装し、
量子多体系の励起状態エネルギーを計算します。
"""
import jax
import jax.numpy as jnp
import netket as nk
import netket.experimental as nkx
import numpy as np
from pyscf import gto, scf, fci
from flax import linen as nn
import flax.nnx as nnx
import optax
from tqdm import tqdm
from functools import partial
from jax import flatten_util
import orbax.checkpoint as ocp
from pathlib import Path
from jax import jit, vmap, grad, value_and_grad
import jax.numpy as jnp
import jax
import time
from functools import partial

# ==============================================================================
# 1. グローバルパラメータとH₂分子定義
# ==============================================================================
# ===================== H₂分子定義とFCIベンチマーク =====================
bond_length = 1.4
geometry = [('H', (0., 0., 0.)), ('H', (bond_length, 0., 0.))]
mol = gto.M(atom=geometry, basis='STO-3G', verbose=0)
mf = scf.RHF(mol).run(verbose=0)

# FCI精密ベンチマーク
cisolver = fci.FCI(mf)
cisolver.nroots = 4
E_fcis, fcivec = cisolver.kernel()
print("="*60)
print("H₂ FCIベンチマークエネルギー")
print("="*60)
for i, e in enumerate(E_fcis):
    exc = (e - E_fcis[0]) * 27.2114
    print(f"E{i} = {e:.8f} Ha  |  励起エネルギー: {exc:.4f} eV")
# ===================== NetKetハミルトニアンおよびサンプラー =====================
ha = nkx.operator.from_pyscf_molecule(mol)

hi = nkx.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K=2
hi_ext = hi**K
edges = [(0, 1), (2, 3),(4, 5),(6,7)]

```

### 2.2 拡張ヒルベルト空間

$\mathbf{X} = (x_1, \ldots, x_N)$ をN個の粒子の集合（粒子セット）を表すとします。ここで$x_i$は$i$番目の粒子の状態です。拡張ヒルベルト空間はK個の元の系のコピーのテンソル積で構成され、各配置はK個の構成 $\mathbf{x} = (x^1, \ldots, x^K)$ に対応します。


### 2.4 SingleStateAnsatzの構成

$\psi(\mathbf{x})$は通常のVMCアルゴリズムのAnsatzに対応します。注目すべきは、この場合では$\mathbf{x}$は粒子数保存、スピン保存、STO-3G下でのH₂分子の4つの合法的な構成に対応し、スピン順序は$[\alpha_1, \alpha_2, \beta_1, \beta_2]$です。4つの合法的な構成は$[1,0,1,0], [0,1,0,1], [1,0,1,1], [1,0,0,1]$です。

SingleStateAnsatzのコードは次のとおりです：

```python
class SingleStateAnsatz(nnx.Module):
    """単一状態Ansatz：フェルミオン系向け複素数値FFNN"""

    def __init__(self, n_spin_orbitals: int, hidden_dim: int = 16, *, rngs: nnx.Rngs):
        super().__init__()
        self.n_spin_orbitals = n_spin_orbitals
        self.linear1 = nnx.Linear(n_spin_orbitals, hidden_dim, rngs=rngs, param_dtype=complex)
        self.linear2 = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs, param_dtype=complex)
        self.output = nnx.Linear(hidden_dim, 1, rngs=rngs, param_dtype=complex)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = nnx.tanh(self.linear1(x))
        h = nnx.tanh(self.linear2(h))
        out = self.output(h)
        return jnp.squeeze(out)

def create_single_machine(model: SingleStateAnsatz):
    """Flax NNXモデルをNetKetスタイルのmachine関数にラップ"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi = m(sigma)
        return log_psi

    return machine, graphdef, state
```

注目すべきは、この場合、デフォルトでモデルのパラメータは複素数値であり、出力は$\ln{\psi(x)}$です。

### 2.3 TotalAnsatzの構成

$\psi_i$を$i$番目のN粒子波動関数（正規化されていない可能性あり）とします。**TotalAnsatz**は行列$\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$の行列式として定義されます：

$$
\Psi(\mathbf{x}) \equiv \det\begin{pmatrix}
\psi_1(x^1) & \psi_2(x^1) & \cdots & \psi_K(x^1) \\
\psi_1(x^2) & \psi_2(x^2) & \cdots & \psi_K(x^2) \\
\vdots & \vdots & \ddots & \vdots \\
\psi_1(x^K) & \psi_2(x^K) & \cdots & \psi_K(x^K)
\end{pmatrix}

$$

ここで：

- $\Psi(\mathbf{x}) \in \mathbb{R}^{K \times K}$：すべての電子の集合とすべての波動関数を組み合わせた行列
- $\psi_i(x^j)$：$i$番目の単一状態Ansatzの$j$番目の粒子セットでの値
- $\Psi(\mathbf{x}) = \det(\Psi(\mathbf{x}))$：総Ansatz、N粒子波動関数组成的非正規化Slater行列式と見なすことができます

**重要な性質**：総Ansatzを単一状態Ansatzの行列式として表現することで、異なるAnsatzが明示的に直交することを要求せずに同じ状態に崩壊するのを防ぐことができます。

```python
class NESTotalAnsatz(nnx.Module):
    def __init__(self, n_spin_orbitals: int, n_states: int = 2, hidden_dim: int = 8, *, rngs: nnx.Rngs):
        super().__init__()
        self.K = n_states
        self.n_spin = n_spin_orbitals

        self.single_ansatz_list = nnx.List()
        key = rngs.params()
        for _ in range(n_states):
            key, sub_key = jax.random.split(key)
            sub_rngs = nnx.Rngs(params=sub_key)

            ansatz = SingleStateAnsatz(
                n_spin_orbitals,
                hidden_dim,
                rngs=sub_rngs
            )
            self.single_ansatz_list.append(ansatz)

    def __call__(self, x: jax.Array):
        def _forward_single(x_single):
            # 形状: [K, n_spin]
            x_single = x_single.reshape(self.K, self.n_spin)
            L = jnp.zeros((self.K, self.K), dtype=complex)
            for i in range(self.K):
                for j in range(self.K):
                    L = L.at[i, j].set(
                        self.single_ansatz_list[j](x_single[i])
                    )
            L_stable = L - L.max()
            sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
            log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
            return log_Psi_stable, L_stable, L.max()

        # 安全なバッチ処理
        if x.ndim == 2 and x.shape[-1] == self.n_spin:
            return _forward_single(x)
        elif x.ndim == 2 and x.shape[-1] == self.n_spin*self.K:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 3:
            x = x.reshape(-1, self.K, self.n_spin)
            return jax.vmap(_forward_single)(x)
        elif x.ndim == 1:
            x = x[None, :]
            x = x.reshape(self.K, self.n_spin)
            return _forward_single(x)
        else:
            raise ValueError(f'サポートされていない入力形状: {x.shape}')


def create_machine(model: NESTotalAnsatz):
    """Flax NNXモデルをNetKetスタイルのmachine関数にラップ"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, _ = m(sigma)
        return log_psi_total

    return machine, graphdef, state

def create_machine_matrix(model: NESTotalAnsatz):
    """Flax NNXモデルをNetKetスタイルのmachine関数にラップ"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, _ = m(sigma)
        return log_M_matrix

    return machine, graphdef, state

def create_machine_max(model: NESTotalAnsatz):
    """Flax NNXモデルをNetKetスタイルのmachine関数にラップ"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, L_max = m(sigma)
        return L_max

    return machine, graphdef, state
```

NESTotalAnsatzの出力は次のとおりです：

$$
\ln{\Psi(\mathbf{X})} = \ln{\det{\mathbf{M} \triangleq \ln{\det{     \begin{pmatrix}
\psi_1(\mathbf{x}^1) & \dots & \psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\psi_1(\mathbf{x}^K) & \dots & \psi_K(\mathbf{x}^K)
\end{pmatrix}}}}}
$$

安定した計算を可能にするために、論文のS8セクション所述のように、対数領域変換を行う必要があります。


### 2.4 拡張ハミルトニアン

拡張ハミルトニアンを$\tilde{H} = \hat{H}_1 \oplus \hat{H}_2 \oplus \cdots \oplus \hat{H}_K$と定義します。ここで$\hat{H}_i$は$i$番目の粒子セットのみに作用するハミルトニアンです。$\tilde{H}$の基底状態エネルギーは元の系$\hat{H}$の最低K個のエネルギーの合計に等しく、その基底状態波動関数は前述の行列式形式の$\Psi^*$です。
NetKetはこのようなハミルトニアンの直和形式をサポートしていないようですので、間接的なアプローチを使用します：

```python
hi.all_states()
>>Array([[0, 1, 0, 1],
       [0, 1, 1, 0],
       [1, 0, 0, 1],
       [1, 0, 1, 0]], dtype=int8)

K=2
hi_ext = hi**K
hi_ext.all_states()
>>Array([[0, 1, 0, 1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0, 1, 1, 0],
       [0, 1, 0, 1, 1, 0, 0, 1],
       [0, 1, 0, 1, 1, 0, 1, 0],
       [0, 1, 1, 0, 0, 1, 0, 1],
       [0, 1, 1, 0, 0, 1, 1, 0],
       [0, 1, 1, 0, 1, 0, 0, 1],
       [0, 1, 1, 0, 1, 0, 1, 0],
       [1, 0, 0, 1, 0, 1, 0, 1],
       [1, 0, 0, 1, 0, 1, 1, 0],
       [1, 0, 0, 1, 1, 0, 0, 1],
       [1, 0, 0, 1, 1, 0, 1, 0],
       [1, 0, 1, 0, 0, 1, 0, 1],
       [1, 0, 1, 0, 0, 1, 1, 0],
       [1, 0, 1, 0, 1, 0, 0, 1],
       [1, 0, 1, 0, 1, 0, 1, 0]], dtype=int8)
```

後ほど、損失関数の定義が$\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})$であること，这里的$\mathcal{H}$は拡張ハミルトニアン$\tilde{H}$を指します。

これは次の式と同等に見なすことができます：

$$ \begin{align*}
\Psi(\mathbf{x})^{-1}\hat{\mathcal{H}}\Psi(\mathbf{x})
&= \mathrm{Tr}\left[ \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x}) \right]
\end{align*} $$

其中的$\hat{H}\Psi(\mathbf{x})$は次の`Ham_psi`と`Ham_Psi`関数で与えられます。

### 2.5 Ham_psiとHam_Psi関数

NES-VMCの損失関数（元の論文のEq.29に対応）
TotalAnsatzの値は：$\Psi(\mathbf{X})$または$\ln{\Psi(\mathbf{x})}$
SingleStateAnsatzの出力は$\ln(\psi(\mathbf{x}))$

$$
\hat{H}\Psi(\mathbf{x}) \triangleq
\begin{pmatrix}
\hat{H}\psi_1(\mathbf{x}^1) & \dots & \hat{H}\psi_K(\mathbf{x}^1) \\
\vdots & & \vdots \\
\hat{H}\psi_1(\mathbf{x}^K) & \dots & \hat{H}\psi_K(\mathbf{x}^K)
\end{pmatrix}
$$

```python
def Ham_psi(ha: nk.operator.DiscreteOperator, single_machine, params, x):
    """
    🔥 両方をサポート:
    - 単一状態 x: (n_spin,)
    - バッチ状態 x: (batch_size, n_spin)
    """
    # ======================
    # コア: 単一サンプルに自動的にバッチ次元を追加
    # ======================
    is_single = (x.ndim == 1)
    if is_single:
        x = x[None, :]  # (n_spin,) → (1, n_spin)

    # ======================
    # ベクトル化計算（バッチ処理）
    # ======================
    def _single_hpsi(x_single):
        x_primes, mels = ha.get_conn_padded(x_single)
        log_psi_vals = single_machine(params, x_primes)
        psi_vals = jnp.exp(log_psi_vals)
        return jnp.sum(mels * psi_vals)

    # バッチ処理
    H_psi_batch = jax.vmap(_single_hpsi)(x)

    # ======================
    # 単一入力の場合、単一出力に圧縮
    # ======================
    if is_single:
        return H_psi_batch[0]
    else:
        return H_psi_batch

def Ham_Psi(ha, single_machine_list, total_params, x):
    K = len(single_machine_list)
    # ======================
    # コア: 単一サンプルとバッチ処理の自動互換性
    # ======================
    if x.ndim == 2:
        # 入力形状: (K, n_spin) → 単一拡張状態 → (K,K)を返す
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]  # 単一状態: (4,)
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi

        return _single_HamPsi(x)

    elif x.ndim == 3:
        # 入力形状: (batch, K, n_spin) → バッチ → (batch, K, K)を返す
        def _single_HamPsi(x_single):
            HPsi = jnp.zeros((K, K), dtype=complex)
            for i in range(K):
                xi = x_single[i]
                for j in range(K):
                    machine_j = single_machine_list[j]
                    params_j = total_params['single_ansatz_list'][j]
                    val = Ham_psi(ha, machine_j, params_j, xi)
                    HPsi = HPsi.at[i, j].set(val)
            return HPsi

        # 自動バッチ処理！
        return jax.vmap(_single_HamPsi)(x)

    else:
        raise ValueError(f'サポートされていない入力形状: {x.shape}')
```

`Ham_psi`は$\hat{H}\psi_1(\mathbf{x}^1)$の計算に使用されます
`Ham_Psi`は$\hat{H}\Psi(\mathbf{x})$の計算に使用されます

### 2.6 サンプラーの設定

NES-VMCアルゴリズムでは、サンプラーは**拡張ヒルベルト空間**$\mathbf{x} = (x^1, x^2, \ldots, x^K)$で動作し、一度にK個の構成をサンプリングします。各$x^k$は元の系のヒルベルト空間$\hat{H}$に属します。
NetKetのSampler + カスタムRuleに基づいてNES-VMCのサンプラーを構築しました。

```python
# 単一系ヒルベルト空間
hi = nk.hilbert.SpinOrbitalFermions(
    n_orbitals=2,
    s=1/2,
    n_fermions_per_spin=(1,1),
)
K = 2  # NES拡張コピー数
hi_ext = hi ** K  # 拡張ヒルベルト空間
SINGLE_SIZE = hi.size  # 単一サブシステム次元 = 4
single_edges = ((0, 1), (2, 3))  # フェルミオン遷移エッジ
g = nk.graph.Graph(edges=single_edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)

total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =50
SINGLE_SIZE = hi.size  # 単一サブシステム次元 = 4

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # JAX配列に変換（重要な修正）
print(ext_edges)

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)

sampler_state = nes_sampler.init_state(total_machine, total_params, seed=1)
samples_raw, sampler_state = nes_sampler.sample(
    total_machine, total_params, state=sampler_state, chain_length=40
)
samples_raw.shape

```

#### コア制約：重複構成の禁止

**重要な制約**：拡張状態は$x^i \neq x^j$（$i \neq j$の場合）を満たす必要があります。これは、総Ansatzの行列式構造により、各コピーの構成が互いに異なる必要があるためです。さもなければ、行列$\Psi(\mathbf{x})$が同一の行/列を持ち、行列式がゼロになります。

K=2の場合、拡張状態の有効な構成数は$N_s^2 - N_s = 4^2 - 4 = 12$です（H₂分子の単一系ヒルベルト空間次元$N_s=4$）。単純な$4^2 = 16$ではありません。

#### アプローチ1：NetKet内置サンプラー

コードはNetKetの`TensorRule`を使用して拡張ヒルベルト空間のサンプラーを構築します：

```python
edges = [(0, 1), (2, 3)]
g = nk.graph.Graph(edges=edges)
single_rule = nk.sampler.rules.FermionHopRule(hi, graph=g)
tensor_rule = nk.sampler.rules.TensorRule(hi_ext, [single_rule] * K)
sampler = nk.sampler.MetropolisSampler(hi_ext, rule=tensor_rule, n_chains=100, sweep_size=32)
```

ここで：
- `FermionHopRule`：単一系ヒルベルト空間でフェルミオン遷移を実行（粒子数保存を満たす）
- `TensorRule`：単一系サンプラールールをK回複製し、拡張ヒル伯特空間適用

完全なコード:

```python
@nk.utils.struct.dataclass
class NESFermionHopRule(nk.sampler.rules.MetropolisRule):
    # 【edgesのみ保持：JAXはJAX配列のみ保存を許可、hi_extを完全に削除！】
    edges: jnp.ndarray

    def _check_duplicate(self, sigma_ext):
        """NES制約：サブ構成が重複しない（グローバルKとSINGLE_SIZEを使用、完全安全）"""
        sub = sigma_ext.reshape((*sigma_ext.shape[:-1], K, SINGLE_SIZE))
        return jnp.any(jnp.all(sub[...,1:,:] == sub[...,0:1,:], axis=-1), axis=-1)

    def transition(self, sampler, machine, parameters, state, rng, sigma):
        """遷移規則（変更なし）"""
        batch_size = sigma.shape[0]
        key1, key2 = jax.random.split(rng)

        e_idx = jax.random.randint(key1, (batch_size,), 0, self.edges.shape[0])
        sel_e = self.edges[e_idx]
        i, j = sel_e[:,0], sel_e[:,1]

        sigma_cand = sigma.at[jnp.arange(batch_size),i].set(sigma[jnp.arange(batch_size),j])
        sigma_cand = sigma_cand.at[jnp.arange(batch_size),j].set(sigma[jnp.arange(batch_size),i])

        invalid = self._check_duplicate(sigma_cand)
        new_sigma = jnp.where(invalid[:, None], sigma, sigma_cand)

        return new_sigma, None

    def random_state(self, sampler, machine, parameters, state, rng):
        """【コア修正】カスタムhi_extの代わりにsampler.hilbertを使用（NetKet標準写法、エラーなし）"""
        sigma_shape = state.σ.shape
        # サンプラーから直接ヒルベルト空間を取得（公式標準用法、100%JAX互換）
        hilbert = sampler.hilbert

        def gen_single(key):
            max_tries = 100  # 無限ループ防止
            def cond(c):
                return (c[0] < max_tries) & c[2]

            def body(c):
                tries, k, _, _ = c
                k, k_new = jax.random.split(k)  # 毎回RNGを更新、無限ループ防止
                s = hilbert.random_state(k_new)
                is_dup = self._check_duplicate(s)
                return (tries + 1, k, is_dup, s)

            init_c = (0, key, True, hilbert.random_state(key))
            final_c = jax.lax.while_loop(cond, body, init_c)
            tries, _, is_dup, s = final_c
            return jax.lax.cond(is_dup, lambda: hilbert.random_state(key), lambda: s)

        keys = jax.random.split(rng, sigma_shape[0])
        return jax.vmap(gen_single)(keys)
```


## 3. 損失関数

### 3.1 目的関数（Rayleigh商）

NES-VMCの目的関数は総Ansatzに関する拡張ハミルトニアンのRayleigh商です：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\langle\Psi|\Psi\rangle}

$$

行列行列式補題を使用して、トレース形式に書き直せます：

$$
\mathcal{L} = \frac{\langle\Psi|\tilde{H}|\Psi\rangle}{\det(S)} = \mathrm{Tr}\left(S^{-1}\hat{H}\right) = \mathrm{Tr}\left(\Psi^{-1}\tilde{H}\Psi\right)

$$

ここでΨ⁻¹HΨは次の関数で計算されます：

```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 正則化項を追加、行列特異性を防止
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. バッチ局所エネルギ行列
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)

    E_L_centered = E_L_batch - E_L_mean

    tr_centered =  jnp.trace(E_L_centered, axis1=-2, axis2=-1)

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. ∇logΨsを計算
    dlogPsi_batch = vmap_grad_logPsi(total_params, x_batch)

    # 5. コア重み付き平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean

```

### 3.2 局所エネルギ行列

Monte Carloサンプリングにより、損失関数は期待値形式で書けます：

$$
\mathcal{L} = \mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})\right)\right]
$$

**局所エネルギ行列**を定義します：

$$
E_L(\mathbf{x}) \equiv \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})
$$

これは$K \times K$行列であり、そのトレースがスカラー局所エネルギーです。$K = 1$の場合、これは標準VMCの局所エネルギーに退化します。

## 4. 勾配公式

### 4.1 標準VMC勾配の復習

基底状態VMCについて、パラメータθに関するエネルギーの勾配は：

$$
\nabla_\theta \frac{\langle\psi|\hat{H}|\psi\rangle}{\langle\psi|\psi\rangle} = 2\mathbb{E}_{x \sim \psi^2}\left[\left(E_L(x) - \mathbb{E}_{x' \sim \psi^2}[E_L(x')]\right)\nabla_\theta \log|\psi(x)|\right]

$$

### 4.2 NES-VMC勾配

総Ansatzについて、勾配計算は類似しています。損失関数はトレース形式$\mathcal{L} = \mathrm{Tr}(E_L(\mathbf{x}))$で、対数振幅を定義します：

$$
\log|\Psi(\mathbf{x})| = \log\det(\Psi(\mathbf{x})) = \mathrm{Tr}\left(\log(\Psi(\mathbf{x}))\right)

$$

勾配公式は：

$$
\nabla_\theta \mathcal{L} = 2\mathbb{E}_{\mathbf{x} \sim \Psi^2}\left[\mathrm{Tr}\left(\left(E_L(\mathbf{x}) - \bar{E}_L\right)\nabla_\theta \log\Psi(\mathbf{x})\right)\right]

$$

ここで：
- $E_L(\mathbf{x}) = \Psi^{-1}(\mathbf{x})\hat{H}\Psi(\mathbf{x})$は局所エネルギ行列
- $\bar{E}_L = \mathbb{E}_{\mathbf{x}' \sim \Psi^2}[E_L(\mathbf{x}')]$は局所エネルギ行列の期待値
- $\nabla_\theta \log \Psi(\mathbf{x})$はパラメータ$\theta$に関する波動関数行列の対数勾配

K = 1の場合、以上は標準VMCの勾配公式に退化します。

```python
def NES_loss_energy(ha, total_matrix_machine,single_machine_list,total_params, x):
    log_M = total_matrix_machine(total_params,x)
    Psi_Matrix = jnp.exp(log_M)
    # 正則化項を追加、行列特異性を防止
    #Psi_Matrix += 1e-6 * jnp.eye(Psi_Matrix.shape[0])
    H_psi_x = Ham_Psi(ha,single_machine_list,total_params,x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv

def nes_vmc_gradient(ha: nk.operator.DiscreteOperator,total_matrix_machine,total_machine,single_machine_list,total_params, x_batch):
    # 1. バッチ局所エネルギ行列
    loss_batch,E_L_batch = NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x_batch)
    E_L_mean = jnp.mean(E_L_batch, axis=0)

    tr_batch = loss_batch
    tr_mean = tr_batch.mean()
    tr_centered = tr_batch - tr_mean  # ✅ 正しい重み

    grad_logPsi = jax.grad(total_machine, argnums=0, holomorphic=True)
    vmap_grad_logPsi = jax.vmap(grad_logPsi, in_axes=(None, 0))

    # 4. ∇logΨsを計算
    dlogPsi_batch = vmap_grad_logPsia(total_params, x_batch)

    # 5. コア重み付き平均
    def weight_and_mean(grad_component):
        weights = tr_centered.reshape( (-1,) + (1,)*(grad_component.ndim - 1) )
        return jnp.mean(weights * jnp.conj(grad_component), axis=0)

    grad = jax.tree.map(weight_and_mean, dlogPsi_batch)

    loss_mean = loss_batch.mean()
    return grad, loss_mean, E_L_mean
```

### 4.3 バッチウォーカーの勾配推定

標準VMCと同様に、同じバッチ内の独立したウォーカーを使用して不偏勾配推定を取得できます：

$$
\nabla_\theta \mathcal{L} = \frac{N-1}{2N}\mathbb{E}_{x_1,\dots,x_N}\left[\frac{1}{N}\sum_{i=1}^N\left(E_L(x_i) - \frac{1}{N}\sum_{j=1}^N E_L(x_j)\right)\nabla_\theta \log|\Psi(x_i)|\right]

$$

## 5. 励起状態エネルギーの抽出

### 5.1 エネルギ行列の対角化

訓練後、大量のサンプリングにより局所エネルギ行列を蓄積します：

$$
\bar{E}_L = \mathbb{E}_{\mathbf{x} \sim \Psi^2}[E_L(\mathbf{x})]

$$

次にĒ_Lを対角化します：

$$
\bar{E}_L = U\Lambda U^{-1}

$$

ここで$\Lambda = \mathrm{diag}(E_1, E_2, \ldots, E_K)$はエネルギー順にソートされた固有値を含みます。

### 5.2 物理的解釈

単一状態Ansatzが固有関数の線形結合$\psi_i = \sum_j a_{ij} \psi_j^*$の場合：

$$

\Psi^{-1}\hat{H}\Psi = A^{-1}\Lambda A

$$

ここで$A$は係数行列です。したがって、対角化により各励起状態のエネルギー$E_1, E_2, \ldots, E_K$を直接取得できます。
次のコードを変更しないでください。edges = $[\alpha_1, \alpha_2, \beta_1, \beta_2]$ この順序を強調する必要があります。


## 6. テストケース

```python

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import netket as nk
import netket.experimental as nkx
import sys
sys.path.append('..')
from NES_VMC import NESTotalAnsatz, create_machine,init_sampler_state,\
    generate_random_initial_states,ha,SingleStateAnsatz,create_single_machine,\
        create_machine_matrix,Ham_psi,Ham_Psi,NES_loss_energy,nes_vmc_gradient,hi,E_fcis,mcmc_sampler_multichain,\
            compute_qgt
import optax
from typing import Callable
from functools import partial
from jax.flatten_util import ravel_pytree
import time
from collections import Counter
import numpy as np
K=2
hi_ext = hi**K

N_CHAINS = 16
N_WARMUP = 100
N_SAMPLES_PER_CHAIN = 200
SWEEP_SIZE = 30
N_ITER =200
SINGLE_SIZE = hi.size  # 単一サブシステム次元 = 4


total_ansatz = NESTotalAnsatz(4,K,12,rngs=nnx.Rngs(11))
total_machine, total_graphdef,total_params = create_machine(total_ansatz)
total_matrix_machine, total_graphdef,total_params = create_machine_matrix(total_ansatz)

single_machine_list = []
for ansatz in total_ansatz.single_ansatz_list:
    m, g, p = create_single_machine(ansatz)
    single_machine_list.append(m)



optimizer = optax.sgd(learning_rate=0.01)
opt_state = optimizer.init(total_params)

ext_edges = []
for k in range(K):
    offset = k * SINGLE_SIZE
    for (i, j) in single_edges:
        ext_edges.append((i + offset, j + offset))
ext_edges = jnp.array(ext_edges)  # JAX配列に変換（重要な修正）

nes_rule = NESFermionHopRule(edges=ext_edges)
nes_sampler = nk.sampler.MetropolisSampler(
    hilbert=hi_ext,
    rule=nes_rule,
    n_chains=16,
    sweep_size=20
)


# サンプラー状態の初期化（原init_sampler_stateの代わり）
sampler_rng = jax.random.PRNGKey(21)
sampler_state = nes_sampler.init_state(total_machine, total_params, sampler_rng)

# ==================== 訓練ループ（サンプラー部分のみ置換） ====================
print("\n" + "="*60)
print("マルチチェーンNES-VMC訓練を開始（NetKetカスタムサンプラー + 単純勾配降下）")
print("="*60)
print(f"基底状態エネルギー={E_fcis[0]:.8f} Ha| 第1励起状態エネルギー={E_fcis[1]:.8f} Ha| 第2励起状態エネルギー={E_fcis[2]:.8f} Ha")

history = {
    'step': [],
    'energy_0st': [],
    'energy_1st': [],
    'energy_std': [],
    'loss': [],
    'params': [],
    'E_Lmatrix':[],
    'natural_grad':[],
    'grad_flat':[],
    'samples':[],
    'log_Psi':[],
    'log_M':[],
    'log_Psi_mean':[],
    'log_Psi_min':[],
    'log_Psi_max':[],
    'grad_norm':[],
}

start_time = time.time()
for step in range(N_ITER):
    # 2. 正式サンプリング
    samples_raw, sampler_state = nes_sampler.sample(
        machine=total_machine, parameters=total_params,
        state=sampler_state, chain_length=N_SAMPLES_PER_CHAIN
    )
        # 3. 次元reshape、勾配関数入力に適用
    samples = samples_raw.reshape(-1, hi_ext.size)
    x_batch = samples.reshape(-1, K, 4)
    # 3. エネルギーと自然勾配を計算（元のコードと同じ論理）
    grad, loss_mean, E_L_mean = nes_vmc_gradient(ha=ha,
                                                 total_matrix_machine=total_matrix_machine,
                                                 total_machine=total_machine,
                                                 single_machine_list=single_machine_list,
                                                 total_params=total_params,
                                                 x_batch=samples.reshape(-1,K,4))

    grad_flat , grad_unravel_fn = ravel_pytree(grad)

    # 4. パラメータを更新
    updates, opt_state = optimizer.update(grad, opt_state, total_params)
    total_params = optax.apply_updates(total_params, updates)


    log_Psi_batch = total_machine(total_params, samples.reshape(-1,K,4))
    eig_vals, eig_vecs = jnp.linalg.eigh(E_L_mean)
    grad_norm = jnp.linalg.norm(grad_flat)


    history['step'].append(step)
    history['E_Lmatrix'].append(E_L_mean)
    history['samples'].append(samples)
    history['loss'].append(loss_mean)
    history['log_Psi_mean'].append(log_Psi_batch.mean())
    history['log_Psi_min'].append(log_Psi_batch.min())
    history['log_Psi_max'].append(log_Psi_batch.max())
    history['grad_norm'].append(grad_norm)
    history['energy_0st'].append(eig_vals[0])
    history['energy_1st'].append(eig_vals[1])
    history['params'].append(total_params)
    # 5. 履歴を記録
    if step % 50 == 0 or step == N_ITER - 1:
        print(f"log_Psi: mean={log_Psi_batch.mean():.3f} | min={log_Psi_batch.min():.3f} | max={log_Psi_batch.max():.3f}")
        print(f"grad norm = {grad_norm:.4f}")
        print(f"Step {step:3d} | Loss: {loss_mean}|0stエネルギー={eig_vals[0]:.8f} Ha| 1stエネルギー={eig_vals[1]:.8f} Ha")
        print('#-----------------------------------------#')


end_time = time.time()
print(f"訓練時間: {end_time - start_time:.2f} 秒")
print("\n" + "="*60)
print(f"訓練完了!")
print("="*60)

```

出力:

```python

============================================================
マルチチェーンNES-VMC訓練を開始（NetKetカスタムサンプラー + 単純勾配降下）
============================================================
基底状態エネルギー=-1.01546825 Ha| 第1励起状態エネルギー=-0.87542794 Ha| 第2励起状態エネルギー=-0.42938376 Ha
log_Psi: mean=1.338+0.070j | min=0.024-3.083j | max=1.513+0.075j
grad norm = 0.8250
Step   0 | Loss: -1.2576295690685688|0stエネルギー=-0.99557002 Ha| 1stエネルギー=-0.26205955 Ha
#-----------------------------------------#
log_Psi: mean=11.451-0.414j | min=9.999-2.433j | max=11.630+1.492j
grad norm = 0.5640
Step  50 | Loss: -1.547645795833384|0stエネルギー=-56.85370946 Ha| 1stエネルギー=55.30606367 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 100 | Loss: -1.5937140959638543|0stエネルギー=-0.94145467 Ha| 1stエネルギー=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=16.215-0.251j | min=16.215-1.625j | max=16.215+1.516j
grad norm = 0.0000
Step 150 | Loss: -1.5937140959638543|0stエネルギー=-0.94145467 Ha| 1stエネルギー=-0.65225943 Ha
#-----------------------------------------#
log_Psi: mean=10.994+0.302j | min=9.058-2.065j | max=11.009+2.080j
grad norm = 0.1187
Step 199 | Loss: -1.5891390172196438|0stエネルギー=-1.60155252 Ha| 1stエネルギー=0.01241350 Ha
#-----------------------------------------#
...

============================================================
訓練完了!
============================================================
```
