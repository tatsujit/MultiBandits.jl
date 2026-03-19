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
    K = n_arms
    probs = zeros(n_arms)   
    randomness = ϵ / K 
    if previous_reward == 1.0
        for a in 1:K
            if a == previous_action
                probs[a] = 1 - (K-1)*randomness
            else
                probs[a] = randomness
            end
        end
    elseif previous_reward == 0.0
        for a in 1:K
            if a == previous_action
                probs[a] = randomness
            else
                probs[a] = (1 - randomness)/(K-1)
            end
        end
    end
    @assert sum(probs) == 1.0
    return probs
end

"""
    Noisy Win-Stay-Lose-Shift 方策では、直近の選択の履歴を持つ必要がある
"""
function select_action(policy::NoisyWinStayLoseShift, estimator::EmptyEstimator; rng::AbstractRNG=Random.default_rng())
    probs = selection_probabilities(policy, estimator)
    return sample(rng, 1:policy.n_arms, Weights(probs))
end

# rr1 = RandomResponding(5)
# selection_probabilities(rr1, Estimator(5))

# using StatsBase
# cs = [select_action(s1, RandomResponding(5)) for _ in 1:1000]
# countmap(cs)
