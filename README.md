# TODO noisy win-shift-lose-stay (NWSLS) の実装など
- これを Policy 側で実装しようとすると、 Policy 行動の履歴を持つ必要があるため、 select_action が副作用を持ち、一貫性を失う
    - NWSLS では、行動 select_action! にしないといけない
- だから、 Estimator 側で実装し、 Estimator は W (行動価値) としてそのまま行動選択確率を出力し、確率をそのまま受け取る Policy を作ってやる、というのが良いかもしれない
    - しかし、 Estimator の側では select_action() の結果は知らないので、通信が難しい？
    - 実際には、 select_action は pol, est の両方を引数に取っているから大丈夫なのではないかな

# MultiBandits.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://tatsujit.github.io/MultiBandits.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://tatsujit.github.io/MultiBandits.jl/dev/)
[![Build Status](https://github.com/tatsujit/MultiBandits.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/tatsujit/MultiBandits.jl/actions/workflows/CI.yml?query=branch%3Amain)

多腕バンディット問題のシミュレーションフレームワーク。

推定器 (`Estimator`)、方策 (`Policy`)、環境 (`Environment`) を組み合わせてバンディットアルゴリズムを構成・実行・評価できます。

## インストール

このパッケージは Julia の General レジストリには登録されていないため、GitHub の URL を直接指定してインストールします。

```julia
# Julia の REPL で ] を押してパッケージモードに入り、以下を実行
] add https://github.com/tatsujit/MultiBandits.jl
```

または、`Pkg` を使って：

```julia
using Pkg
Pkg.add(url="https://github.com/tatsujit/MultiBandits.jl")
```

## クイックスタート

```julia
using MultiBandits, Random

# 環境を作成（3本腕の Bernoulli バンディット、報酬確率 0.2, 0.5, 0.8）
env = Environment([0.2, 0.5, 0.8])

# エージェントを作成（Greedy 方策 + Thompson Sampling 推定器）
agent = Agent(Greedy(), ThompsonSampling(3))

# 履歴の記録先を作成
history = History(3, 100)

# システムを組み立てて実行
system = System(agent, env, history; rng=MersenneTwister(42))
run!(system, 100)

# 結果を確認
history.actions   # 各試行で選択された腕
history.rewards   # 各試行で得た報酬
```

## 主な構成要素

### 環境 (Environment)

| 型 | 説明 |
|---|---|
| `Environment(reward_probs)` | 定常バンディット環境（Bernoulli 分布） |
| `Environment(n_arms)` | 腕数を指定して等間隔の確率で自動生成 |
| `NonStationaryEnvironment` | 非定常環境（途中で報酬分布が切り替わる） |

```julia
# 定常環境
env = Environment([0.2, 0.5, 0.8])

# 腕数だけ指定（3本なら [0.25, 0.5, 0.75]）
env = Environment(3)

# 非定常環境（50試行目で確率が反転）
env = NonStationaryEnvironment([0.2, 0.8], [50], [[0.8, 0.2]])
```

### 推定器 (Estimator)

| 型 | 説明 |
|---|---|
| `EmpiricalReward(n_arms)` | 標本平均による行動価値推定 |
| `ThompsonSampling(n_arms)` | Beta 分布によるトンプソンサンプリング |
| `ThompsonSamplingN(n_arms)` | 正規分布によるトンプソンサンプリング |
| `ThompsonSamplingNU(n_arms)` | 正規分布 + 効用変換によるトンプソンサンプリング |
| `UCB1(n_arms)` | UCB1 アルゴリズム |
| `RS(n_arms)` | RS 推定器 |
| `STEP(n_arms)` | STEP 推定器 |
| `CognitiveEstimator(...)` | 認知モデルベースの推定器（効用変換・固執性を含む） |

### 方策 (Policy)

| 型 | 説明 |
|---|---|
| `Greedy()` | 貪欲方策（最大価値の腕を選択） |
| `SoftmaxPolicy(β)` | Softmax（Boltzmann）方策 |
| `RandomResponding()` | ランダム方策 |
| `SimpleSat(aspiration)` | 満足化方策（aspiration level 以上なら搾取） |

### 便利コンストラクタ

```julia
# SimpleSat + EmpiricalReward のエージェントを一発で作成
agent = SimpleSatAgent(3, 0.5)
```

## 評価・分析

```julia
# 累積リグレット
regret = cumulative_regret(history.actions, history.expectations)

# 報酬の移動平均
ma = moving_average_rewards(history.rewards, 10)

# 選択確率の計算
probs = selection_probabilities(Greedy(), agent.estimator)
```

## 拡張機能 (Extensions)

オプションのパッケージを追加すると、追加機能が使えます。

```julia
] add CairoMakie   # プロット機能
] add DataFrames CSV  # DataFrame への変換
] add YAML          # YAML 設定ファイルの読み込み
```

## ライセンス

MIT License
