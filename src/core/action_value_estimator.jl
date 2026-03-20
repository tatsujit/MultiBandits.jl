"""
    AbstractEstimator

全推定器の基底抽象型。サブタイプは以下のフィールドを持つ必要がある:
- `n_arms::Int` — 腕の数
- `W::Vector{Float64}` — 行動選択に使用する行動価値ベクトル
"""
abstract type AbstractEstimator end


# include("./sample_average.jl")
# include("./thompson_sampling.jl")
# include("./prospectValue.jl")
# include("./symlogSampleAverageValue.jl")
# include("./discontinuousValue.jl")
# include("./discontinuousSampleAverageValue.jl")
# include("./sigmoidSampleAverageValue.jl")
# include("./sigmoidValue.jl")
include("./estimator/empirical_reward.jl")
include("./estimator/empty_estimator.jl")
include("./estimator/recording_estimator.jl")
include("./estimator/ucb1.jl")
include("./estimator/cognitive_estimator.jl")
include("./estimator/thompson_sampling.jl")
include("./estimator/thompson_sampling_N.jl")
include("./estimator/thompson_sampling_N_U.jl")
include("./estimator/rs.jl")
include("./estimator/step.jl")
# include("./model2-simd.jl")
# include("./model-fq.jl")
# include("./utility_functions.jl")
