# Matrix-Free Deflated NES-VMC サンプル集（H₂ / 6-31G / 8 qubit）

本ディレクトリは、**Matrix-Free Deflated NES-VMC（方案 A'）** の実行例をまとめたものです。
H₂ 分子（6-31G、CAS(2e,4o) → 8 スピン軌道 = 8 qubit、全 CI 次元 16）を対象に、
K=4 で学習済みの基底状態〜第 4 励起状態を **凍結（frozen）** し、
残りのエネルギー準位を少数の追加ネットワーク列で **逐次探索（deflation）** します。

- 学習・勾配・QGT・サンプリング・モニタリング・固有状態抽出のすべてが **逐点（matrix-free）** で完結
- 2^N 次元の全空間配列（密行列 H、全空間振幅）は **一切使用しない**
- PySCF の FCI 対角化も呼ばない（`H2_COMPUTE_FCI=0` を import 前に設定し、`E_fcis` は参照しない）

---

## 1. アルゴリズムの要点

### 1.1 凍結 + deflation

旧 K=4 ネットワークの 4 本の列を回転行列 `v_rot` で混合して凍結列
`φ_i = Σ_j ψ_j^old · v_rot[j,i]` を作り、新規の active 列 `ψ_m`（m = 1..K）と合わせて

```text
Ψ_F(x) = det[ Ψ̃(x) − Φ̃(x)·c ]
```

を試行波動関数とします。`c` は凍結空間への射影係数で、これが「すでに求めた準位を
再度求めてしまう」ことを防ぐ（deflation）役割を果たします。

### 1.2 Z 抵消補題（MC Gram 射影）

参照密度 `q(y) = Σ_i|φ̃_i(y)|² + ε·Σ_m|ψ̃_m(y)|²`（未規格化、分配関数 Z 不明）から
サンプル `y ~ q/Z` を生成し、同一サンプルで

```text
Ĝ    = ⟨Φ̃|Φ̃⟩/Z,   M̂    = ⟨Φ̃|Ψ̃⟩/Z,
Ĥ_GG = ⟨Φ̃|HΦ̃⟩/Z,  M̂₂   = ⟨Φ̃|HΨ̃⟩/Z
c  = Ĝ⁻¹ M̂,        c₂ = Ĝ⁻¹(M̂₂ − Ĥ_GG·c)
```

を推定します。すべての Gram 量が同じ `1/Z` を持つため、**c と c₂ では Z が厳密に相殺**します。
したがって規格化定数を知る必要がなく、全空間内積も不要です。

### 1.3 スケール管理

`φ̃ = φ·e^{−f}`、`ψ̃_m = ψ_m·e^{−g_m}` とし、`f, g_m` を各バッチの aux サンプルの
最大値の EMA（`SCALE_BETA`）で更新します。これにより `q` の 2 項がともに O(1) になり、
`c, c₂` も同じ変換 `c[:,m] *= e^{Δf − Δg_m}` で追随します。

### 1.4 正しいモニタリング量：Gram 一般化固有値

各ステップのエネルギー準位は **`M v = λ S v`** の一般化固有値で監視します。

```text
S = Σ_n w_n A_n†A_n / Σw,   M = Σ_n w_n A_n†Z_n / Σw,   w ∝ |Ψ_F|
```

ここで `A_n = Ψ̃ − Φ̃c`、`Z_n = (HΨ̃ − HΦ̃c) − Φ̃c₂` です。
**`eig(sym⟨E_L⟩) = eig(⟨A⁻¹Z⟩)` は使用しません。**
`E_L` は 1 サンプルごとの行列逆の比推定量であり、トレース（= Loss）は正しくても
固有値スペクトルが大きく拡散します（K=4 凍結拡張では ±137 mHa の偏りを確認）。
Gram 一般化固有値は収束後の抽出（`M_χ v = λ S_χ v`）と同じ推定器で、`A⁻¹` を含みません。

---

## 2. ディレクトリ構成

| ディレクトリ | active K | 探索する準位 | 内容 |
|---|---|---|---|
| `冻结K4_探索K2/` | 2 | φ̂₄, φ̂₅ | 最小構成。2 準位だけを追加探索 |
| `冻结K4_探索K4/` | 4 | φ̂₄ … φ̂₇ | 4 準位を同時探索（推奨の基本例） |
| `冻结K4_探索K6/` | 6 | φ̂₄ … φ̂₉ | 6 準位まで一括探索（高次準位を含む） |

各ディレクトリの内容は同一構成です。

```text
H2_631G_8Qbit_K{K}_frozen4_deflation_MatrixFree_Pure.py   # 学習スクリプト（K=2/4/6）
data/       # history pkl、抽出した近似固有状態 npz
日志/       # トレーニングログ
graph/      # 学習曲線 PNG
制图.ipynb                               # 旧モニタリング量で保存した pkl 用の作図ノート
绘图_K*_frozen4_MatrixFree_Pure.ipynb    # 新モニタリング量対応の作図ノート（推奨）
```

共通モジュールは 1 つ上の階層（`H2分子/`）にあります。

- `NES_VMC_V1.py`：単一列 machine、ゲージ固定、`Ham_psi_scaled`（Hψ の逐点計算）
- `NES_VMC_deflation_mf.py`：方案 A' ツールボックス（`make_mf_bundle`、MC Gram 推定器、`update_scales`）
- `NES_VMC_tool.py`：`make_MS_estimator_fn` / `compute_lam_v_from_samples`（サンプル版 MV=SVA）
- `H2_6_31G_8Qbit.py`：系の定義（単一組態 Hilbert 空間、ハミルトニアン、HF 参照）

---

## 3. 実行結果（500 ステップ、凍結 K=4、FCI 参照値との差）

凍結源：`../data/26-10-22-01-58_history_natural_gradient_H2_molecule_K4__realFFNN.pkl`

| 実験 | 抽出準位 λ（Ha） | FCI 参照との差（mHa） | 所要時間 |
|---|---|---|---|
| 探索K2 | −0.027988, −0.013689 | +0.045, +0.814 | 174 s |
| 探索K4 | −0.028043, −0.014506, −0.005173, +0.023974 | −0.010, −0.003, +0.010, +0.003 | 432 s |
| 探索K6 | −0.028047, −0.014609, −0.005196, +0.022955, +0.235440, +0.257236 | −0.014, −0.105, −0.014, −1.016, +1.116, +0.030 | 981 s |

（FCI 参照値は外部の既知値：E₄=−0.02803278, E₅=−0.01450343, E₆=−0.00518297,
E₇=+0.02397075, E₈=+0.23432390, E₉=+0.25720575 Ha。学習スクリプト内では計算しません。）

観察：

- K=4 では 4 準位すべてが化学精度（1.6 mHa）内、最大でも 0.010 mHa。
- K=2 でも φ̂₄ は 0.045 mHa、φ̂₅ は 0.81 mHa（化学精度内）。
- K=6 では φ̂₄〜φ̂₇ が 0.014〜1.0 mHa、高次準位 φ̂₈/φ̂₉ が約 1.1 mHa。
  高次準位は表現能力・サンプル数に対して敏感で、必要ならステップ数や `NES_AUX_M` を増やします。

### 実行結果の図（凍結 K=4 + 探索 K）

**凍結 K=4 + 探索 K=2**（φ̂₄, φ̂₅）

![凍結K4 + 探索K2 の学習曲線](<冻结K4_探索K2/graph/K=4 + K=2.png>)

**凍結 K=4 + 探索 K=4**（φ̂₄ … φ̂₇）

![凍結K4 + 探索K4 の学習曲線](<冻结K4_探索K4/graph/K=4+K=4.png>)

**凍結 K=4 + 探索 K=6**（φ̂₄ … φ̂₉）

![凍結K4 + 探索K6 の学習曲線](<冻结K4_探索K6/graph/K=4+K=6.png>)

---

## 4. 実行方法

```bash
cd NES_VMC/experiments/H2分子/冻结K4_探索K4
python H2_631G_8Qbit_K4_frozen4_deflation_MatrixFree_Pure.py           # 500 ステップ（既定）
NES_N_ITER=5 python H2_631G_8Qbit_K4_frozen4_deflation_MatrixFree_Pure.py   # スモークテスト
```

K=2 / K=6 も同様に、それぞれのディレクトリでスクリプトを実行します。

環境変数：

| 変数 | 既定値 | 意味 |
|---|---|---|
| `NES_N_ITER` | 500 | 学習ステップ数 |
| `NES_AUX_M` | 1024 | 1 ステップあたりの aux サンプル数（Gram 推定精度 ∝ M^(−1/2)） |
| `NES_SCALE_BETA` | 0.9 | スケール f/g の EMA 平滑係数（初回は 1 = fresh） |
| `NES_INIT_SEED` | 11 | active 列の初期化シード |
| `H2_COMPUTE_FCI` | スクリプトが `0` を設定 | `0` で PySCF の全空間 FCI 対角化をスキップ |

主要ハイパーパラメータ（スクリプト内）：`N_CHAINS=16`、`N_SAMPLES_PER_CHAIN=200`（計 3200 サンプル）、
`SWEEP_SIZE=20`、`AUX_CHAINS=8`、`AUX_SWEEP=20`、`EPS_MIX=0.25`、
`lr=0.1`、`qgt_diag_shift=0.1`、`clip_norm=20.0`。

---

## 5. 出力物

- `data/<timestamp>_history_K{K}_frozen4_deflation_MatrixFree_Pure.pkl`
  `steps / lam（毎ステップの Gram 一般化固有値）/ loss / diag / params / samples / lam_extract / elapsed_s`
- `data/近似本征态_K{K}扩展_MatrixFree_Pure_<timestamp>.npz`
  抽出した `v`（回転行列）と `lam`（準位）
- `日志/<timestamp>_nes_vmc_K{K}_frozen4_deflation_MatrixFree_Pure.log`
- `graph/*.png`：学習曲線

### 作図ノート

`绘图_K*_frozen4_MatrixFree_Pure.ipynb` は pkl を読み、5 つのサブプロットを描きます。

1. 毎ステップの λ（Gram 一般化固有値）+ 抽出値の破線
2. Loss = Re tr⟨E_L⟩（トレース推定は不偏）
3. サンプリング健全性（valid_ratio / 主サンプラー ESS / aux 重み ESS）
4. 勾配ノルム（raw / natural、log 軸）
5. モニタリング量の収束誤差 |λ̂ − λ_extract|（log 軸 + 化学精度帯）

ノートは FCI も全空間配列も使わず、比較基準は pkl 内の `lam_extract` のみです。
旧モニタリング量（`eig(sym⟨E_L⟩)`）で保存された pkl を読み込むと警告を出します。

---

## 6. 注意点

- **モニタリング量の混同に注意**：`trace(eig(E_L)) = Loss` は常に正しい一方、
  `eig(⟨E_L⟩)` の各固有値は正しくありません。「Loss は正常なのに準位だけおかしい」場合は
  このバグを疑ってください。
- `aux` の重み ESS が小さい（例：49/1024）と Gram 推定の分散が増えます。
  必要に応じて `NES_AUX_M=4096` などを試してください。
- active 列の数 K を増やすと 1 ステップあたりのコストはおよそ K に比例して増えます
  （K=2: 174 s、K=4: 432 s、K=6: 981 s / 500 ステップ）。
- 凍結源を変更する場合は各スクリプトの `PKL_PATH` を書き換えるだけで、
  外部 npz は不要です（`v_rot` はサンプル版 MV=SVA でスクリプト内で計算）。

## 7. 関連ドキュメント

- `NES_VMC/解读文档/1016-方案A-prime-MC-Gram投影真MatrixFree-实现解读.md`
- `NES_VMC/解读文档/1019-方案A-prime训练脚本逐函数解读-MatrixFree与build_full_stage.md`
- `NES_VMC/解读文档/1020-纯净版真MatrixFree训练脚本-逐段代码解读.md`
- `NES_VMC/诊断报告/H2_631G_K4冻结4_K4扩展-图表未进化学精度误判诊断-20261023.md`
