# noisy win-stay-lose-shift policy

struct NoisyWinStayLoseShift <: AbstractPolicy
    # TODO: 直近の選択の履歴を残さなければいけない
    # 一般的には選択の履歴を持つ必要あり
    probs::Vector{Float64}
    function RandomResponding(n_arms::Int) # 腕の数だけ与えられたら一様確率で選択
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

"""
    Noisy Win-Stay-Lose-Shift 方策では、直近の選択の履歴を持つ必要がある
"""
function select_action(policy::RandomResponding, estimator::EmptyEstimator; rng::AbstractRNG=Random.default_rng())
    n_arms = length(policy.probs)
    return sample(rng, 1:n_arms, Weights(policy.probs))
end

# rr1 = RandomResponding(5)
# selection_probabilities(rr1, Estimator(5))

# using StatsBase
# cs = [select_action(s1, RandomResponding(5)) for _ in 1:1000]
# countmap(cs)
