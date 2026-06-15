# NES-VMC 対数領域変換 コード解説

## 1. 研究背景

NES-VMC（Natural Excited State Variational Monte Carlo）アルゴリズムにおいて、ニューラルネットワークのトレーニングが後期段階になると、波動関数行列 $\Psi(\mathbf{x})$ の要素が非常に大き거나非常に小さくなる可能性があります。これにより、損失関数の計算時に**数値オーバーフロー**問題が発生します。

本文書では、**改造前**（オリジナルバージョン）と**改造後**（対数領域変換バージョン）のコアな違いを比較分析します。

---

## 2. コア改造点の比較

### 2.1 NESTotalAnsatz のモデル出力

#### 改造前（オリジナルバージョン）

```python
# ファイル：NES_new_vserison/NES_VMC.py
# 行番号：91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L))
    log_Psi = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi, L  # L を直接返す、安定化なし
```

#### 改造後（対数領域変換バージョン）

```python
# ファイル：対数領域変換 NES/NES_VMC.py
# 行番号：91-104

def _forward_single(x_single):
    x_single = x_single.reshape(self.K, self.n_spin)
    L = jnp.zeros((self.K, self.K), dtype=complex)
    for i in range(self.K):
        for j in range(self.K):
            L = L.at[i, j].set(
                self.single_ansatz_list[j](x_single[i])
            )
    L_stable = L - L.max()  # コア改造：行の最大値を減算
    sign, log_abs_det = jnp.linalg.slogdet(jnp.exp(L_stable))
    log_Psi_stable = log_abs_det + 1j * jnp.angle(sign)
    return log_Psi_stable, L_stable, L.max()  # 安定化後の L_stable と L.max() を返す
```

#### 主な違い

| 項目 | 改造前 | 改造後 |
|------|--------|--------|
| `L` 行列の処理 | 安定化なし | 安定化：`L_stable = L - L.max()` |
| 戻り値 | `(log_Psi, L)` | `(log_Psi_stable, L_stable, L.max())` |
| 数値安定性 | オーバーフローのリスクあり | 数値的に安定 |

---

### 2.2 損失関数の計算

#### 改造前（オリジナルバージョン）

```python
# ファイル：NES_new_vserison/NES_VMC.py
# 行番号：248-255

def NES_loss_energy(ha, total_matrix_machine, single_machine_list, total_params, x):
    log_M = total_matrix_machine(total_params, x)  # 元の L 行列を取得
    Psi_Matrix = jnp.exp(log_M)  # 直接指数演算、オーバーフローの可能性
    H_psi_x = Ham_Psi(ha, single_machine_list, total_params, x)
    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix, H_psi_x)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### 改造後（対数領域変換バージョン）

```python
# ファイル：対数領域変換 NES/NES_VMC.py
# 行番号：85-97

def NES_loss_energy_stable(ha, total_matrix_machine, total_max_machine,
                          single_machine_list, total_params, x):
    L_stable = total_matrix_machine(total_params, x)  # 安定化後の L_stable を取得
    Psi_Matrix_stable = jnp.exp(L_stable)  # 指数演算を安定化

    M = jnp.log(Ham_Psi(ha, single_machine_list, total_params, x))  # H_psi の対数を取る
    M_stable = M - total_max_machine(total_params, x).reshape(-1,1,1)  # 安定化
    HPsi_stable = jnp.exp(M_stable)  # 指数演算を安定化

    Psi_Matrix_inv = jnp.linalg.solve(Psi_Matrix_stable, HPsi_stable)
    return jnp.real(jnp.trace(Psi_Matrix_inv, axis1=-2, axis2=-1)), Psi_Matrix_inv
```

#### 主な違い

| 手順 | 改造前 | 改造後 |
|------|--------|--------|
| `Psi_Matrix` | `exp(L)` を直接計算 | `exp(L_stable)` を安定化後に計算 |
| `H_psi_x` | 生値を直接使用 | 先に `log(H_psi)` を取り、次に `L.max()` を減算 |
| `HPsi` | `H_psi_x` を直接使用 | `exp(M_stable)` を安定化後に計算 |

---

### 2.3 新規追加 `create_machine_max` 関数

#### 改造前

以下の2つの関数のみ存在：
- `create_machine`：`log_psi_total` を返す
- `create_machine_matrix`：`log_M_matrix` を返す

#### 改造後

新規 `create_machine_max` 関数を追加：

```python
# ファイル：対数領域変換 NES/NES_VMC.py

def create_machine_max(model: NESTotalAnsatz):
    """Flax NNX モデルを NetKet スタイルの machine 関数にラップ"""
    graphdef, state = nnx.split(model)

    @jax.jit
    def machine(params, sigma):
        m = nnx.merge(graphdef, params)
        log_psi_total, log_M_matrix, L_max = m(sigma)  # L.max() を返す
        return L_max

    return machine, graphdef, state
```

---

## 3. 数学的安定性分析

### 3.1 数値オーバーフロー問題

`L` 行列の要素が大きい場合（トレーニング後期など）、`jnp.exp(L)` は数値オーバーフローを起こします：

```python
# L の要素が 1000 の場合
jnp.exp(1000)  # → inf (数値オーバーフロー)

# 改造後：L_stable = L - L.max() = L - 1000 ≈ [0, -10, ...]
jnp.exp(L_stable)  # → [1, 4.5e-5, ...] (数値的に安定)
```

### 3.2 対数領域変換の物理的一貫性

改造後の計算は以下の天一：

$$
\Psi^{-1} \tilde{H} \Psi = \exp\left(L_{\text{stable}}\right)^{-1} \cdot \exp\left(M_{\text{stable}}\right)
$$

ここで：
- $L_{\text{stable}} = \log(\Psi) - L_{\max}$
- $M_{\text{stable}} = \log(\tilde{H}\Psi) - L_{\max}$

これは論文 S8 章の数値安定化スキームと一貫しています。

---

## 4. 妥当性分析

### ✅ 妥当な点

1. **安定性の向上**：対数領域変換により `exp()` 操作の数値オーバーフローを効果的に回避
2. **数学的同値性**：共通因子 $e^{L_{\max}}$ 除去後、行列逆演算と行列式計算の相対関係は変化しない
3. **既存アーキテクチャとの互換性**：改造は `NESTotalAnsatz` の全体構造とサンプラーを変更しない

### ⚠️ 注意が必要な点

1. **新規 `create_machine_max` の導入**：`L.max()` の追加計算と受け渡しがいるため、コード複雑度が増加
2. **バッチ処理の一貫性**：`L.max()` は `M` の形状に一致するように `(batch, 1, 1)` に正しくリサイズが必要
3. **トレーニング監視**：log_Psi の値範囲が（安定化により）変化するため、監視時に注意が必要

---

## 5. まとめ

| 改造項目 | 改造内容 | 目的 |
|----------|----------|------|
| `L_stable = L - L.max()` | 行列安定化 | exp() オーバーフローを防止 |
| `M_stable = M - L.max()` | Hamiltonian 行列安定化 | Psi_Matrix との数値的一貫性を確保 |
| 新規 `create_machine_max` | `L.max()` を返す | 安定化に必要な共通因子を提供 |
| `NES_loss_energy_stable` | 新規損失関数 | 対数領域で行列演算を実行 |

改造は数値安定性を向上させ、深層ネットワークや長期トレーニングシナリオにおいて特に重要でありながら、元のアルゴリズムとの数学的同値性を維持しています。

---

## 6. 参照ファイル

- オリジナルバージョン：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/NES_new_vserison/NES_VMC.py`
- 改造バージョン：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC.py`
- 改造テスト：`/Users/yangjianfei/mac_vscode/神经网络量子态/5 月/0510/对数域改造 NES/NES_VMC对数域改造 K2.ipynb`
