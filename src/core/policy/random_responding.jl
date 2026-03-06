"""
    RandomResponding <: AbstractPolicy

ランダム行動選択方策。均等確率またはカスタム確率ベクトルで行動を選択する。
`EmptyEstimator` と組み合わせて使用する。

# コンストラクタ
- `RandomResponding(n_arms::Int)` — 均等確率 (1/n_arms)
- `RandomResponding(probs::Vector{<:Real})` — カスタム確率ベクトル
"""
struct RandomResponding <: AbstractPolicy
    probs::Vector{Float64}
    function RandomResponding(n_arms::Int)
        return new(ones(Float64, n_arms) ./ n_arms)
    end
    function RandomResponding(probs::Vector{<:Real})
        return new(Float64.(probs))
    end
end

# function selection_probabilities(policy::RandomResponding, estimator::AbstractActionValueEstimator)
function selection_probabilities(policy::RandomResponding, estimator::EmptyEstimator)
    return policy.probs
end

# function select_action(policy::RandomResponding, estimator::AbstractActionValueEstimator; rng::AbstractRNG=Random.default_rng())
function select_action(policy::RandomResponding, estimator::EmptyEstimator; rng::AbstractRNG=Random.default_rng())
    n_arms = length(policy.probs)
    return sample(rng, 1:n_arms, Weights(policy.probs))
end

# rr1 = RandomResponding(5)
# selection_probabilities(rr1, Estimator(5))

# using StatsBase
# cs = [select_action(s1, RandomResponding(5)) for _ in 1:1000]
# countmap(cs)
