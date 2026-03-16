# noisy win-stay-lose-shift policy
# import Pkg; Pkg.add("Parameters")
using Parameters: @unpack

mutable struct NoisyWinStayLoseShift <: AbstractPolicy
    # TODO: 直近の選択の履歴を残さなければいけない
    # 一般的には選択の履歴を持つ必要あり
    n_arms::Int
    previous_action::Int64
    previous_reward::Float64
    ϵ::Float64
    function NoisyWinStayLoseShift(n_arms::Int, ε::Float64=0.0) # 腕の数だけ与えられたら一様確率で選択
        return new(n_arms, 0, NaN, ε)
    end
end

# function selection_probabilities(policy::RandomResponding, estimator::AbstractActionValueEstimator)
function selection_probabilities(policy::NoisyWinStayLoseShift, estimator::AbstractEstimator)
    @unpack n_arms, previous_action, previous_reward, ϵ = policy
    probs = zeros(n_arms)
    randomness = ϵ / n_arms
    if previous_reward == 1.0
        probs[previous_action] = 1 - randomness
    elseif previous_reward == 0.0
        probs[previous_action] = randomness
    end
    probs = [randomness for _ in 1:n_arms]
    probs[previous_action] += 1 - ϵ
    return probs
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
