```@meta
CurrentModule = MultiBandits
```

# MultiBandits.jl

多腕バンディット問題のシミュレーションフレームワーク。

推定器 (`AbstractEstimator`)、方策 (`AbstractPolicy`)、環境 (`AbstractEnvironment`) を
組み合わせてバンディットアルゴリズムを構成・実行・評価できる。

## クイックスタート

```julia
using MultiBandits, Random

# 環境: 3本腕の Bernoulli バンディット
env = Environment([0.2, 0.5, 0.8])

# エージェント: Greedy + Thompson Sampling
agent = Agent(Greedy(), ThompsonSampling(3))

# シミュレーション実行
history = History(3, 100)
system = System(agent, env, history; rng=MersenneTwister(42))
run!(system, 100)

# 評価
regret = cumulative_regret(history.actions, mean(env))
```

## 主なコンポーネント

| コンポーネント | 基底型 | 実装例 |
|:---|:---|:---|
| 推定器 | [`AbstractEstimator`](@ref) | [`EmpiricalReward`](@ref), [`ThompsonSampling`](@ref), [`UCB1`](@ref), [`STEP`](@ref), [`RS`](@ref), [`CognitiveEstimator`](@ref) |
| 方策 | [`AbstractPolicy`](@ref) | [`Greedy`](@ref), [`SoftmaxPolicy`](@ref), [`SimpleSat`](@ref), [`RandomResponding`](@ref) |
| 環境 | [`AbstractEnvironment`](@ref) | [`Environment`](@ref), [`NonStationaryEnvironment`](@ref) |
| システム | [`System`](@ref) | エージェント + 環境 + 履歴を束ねる |

## API リファレンス

```@index
```
