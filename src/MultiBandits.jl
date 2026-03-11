"""
    MultiBandits

多腕バンディット問題のシミュレーションフレームワーク。

推定器 (`AbstractEstimator`)、方策 (`AbstractPolicy`)、環境 (`AbstractEnvironment`) を
組み合わせてバンディットアルゴリズムを構成・実行・評価できる。

# 基本的な使い方

```julia
using MultiBandits, Random

env = Environment([0.2, 0.5, 0.8])
agent = Agent(Greedy(), ThompsonSampling(3))
history = History(3, 100)
system = System(agent, env, history; rng=MersenneTwister(42))
run!(system, 100)
```
"""
module MultiBandits

using Random
using Distributions
using StatsBase
using SpecialFunctions
using Statistics

# 抽象型
export AbstractEstimator, AbstractPolicy, AbstractEnvironment, AbstractHistory

# 構造体
export Agent, System, Environment, NonStationaryEnvironment
export History, EstimatorHistory, LAHistory

# 推定器
export EmptyEstimator, EmpiricalReward
export ThompsonSampling, ThompsonSamplingN, ThompsonSamplingNU
export UCB1, STEP, RS, CognitiveEstimator

# 方策
export Greedy, SoftmaxPolicy, RandomResponding, SimpleSat

# 主要関数
export run!, step!, select_action, update!, sample_reward
export selection_probabilities, cumulative_regret, moving_average_rewards
export average_rewards, action_moving_averages
export cumulative_rewards, find_optimal, average_reward

# 便利コンストラクタ
export SimpleSatAgent

# _utils.jl must be loaded first (defines @ic0 macro used in core files)
include("_utils.jl")

include("core/action_value_estimator.jl")
include("core/policy.jl")
include("core/agent.jl")
include("core/environment.jl")
include("core/history.jl")
include("core/system.jl")
include("core/evaluation.jl")

# TODO: organize the plot functions, and then include them here
# include("plot/plot.jl") # 

end
