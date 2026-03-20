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
function selection_probabilities(policy::NoisyWinStayLoseShift, estimator::AbstractEstimator; verbose::Bool=false)
    @unpack n_arms, previous_action, ϵ = policy
    previous_reward = estimator.previous_reward # ここでズレる。こうしないほうが良いかもしれない。
    K = n_arms
    probs = zeros(n_arms)   
    randomness = ϵ / K 
    # at the beginning, all arms are selected with equal probability
    if previous_action == 0
        for a in 1:K
            probs[a] = 1/K
        end
    # when the previous action was a win, the probability of the previous action is
    # 1 - (K-1)*randomness, and the other arms are selected with probability randomness
    elseif previous_reward == 1.0
        for a in 1:K
            if a == previous_action
                probs[a] = 1 - (K-1)*randomness
            else
                probs[a] = randomness
            end
        end
    # when the previous action was a lose, the probability of the previous action is randomness, 
    # and the other arms are selected with probability (1 - randomness)/(K-1)
    elseif previous_reward == 0.0
        for a in 1:K
            if a == previous_action
                probs[a] = randomness
            else
                probs[a] = (1 - randomness)/(K-1)
            end
        end
    end
    if verbose
        @show probs, sum(probs), previous_action, previous_reward, ϵ
    end
    @assert abs(sum(probs) - 1.0) < 1e-8
    return probs
end


"""
    Noisy Win-Stay-Lose-Shift 方策では、直近の選択の履歴を持つ必要がある
"""
function select_action(policy::NoisyWinStayLoseShift, estimator::RecordingEstimator; rng::AbstractRNG=Random.default_rng())
    probs = selection_probabilities(policy, estimator)
    action = sample(rng, 1:policy.n_arms, Weights(probs))
    policy.previous_action = estimator.previpous_action
    policy.previous_reward = NaN # to be updated in update!(RecordingEstimator)
    return action    
end

# rr1 = RandomResponding(5)
# selection_probabilities(rr1, Estimator(5))

# using StatsBase
# cs = [select_action(s1, RandomResponding(5)) for _ in 1:1000]
# countmap(cs)




"""
    test_selection_probabilities(n_arms = 3, ϵ = 1.0)

Test the selection_probabilities function for the NoisyWinStayLoseShift model.
When n_arms == 2, the model coincides with the original noisy-WSLS model, and it's tested. 

# Arguments
- `n_arms::Int`: number of arms
- `ϵ::Float64`: noise level
"""
function test_selection_probabilities(n_arms = 3, ϵ = 1.0)
    policy = MultiBandits.NoisyWinStayLoseShift(n_arms, ϵ)
    estimator = EmptyEstimator()

    # テスト1: previous_reward=1.0 (Win時), 全てのアームに対して確率出るか
    policy.previous_action = 2
    policy.previous_reward = 1.0
    probs_win = selection_probabilities(policy, estimator)
    @assert length(probs_win) == n_arms
    @assert all(prob -> prob ≥ 0.0 && prob ≤ 1.0, probs_win)
    @assert abs(sum(probs_win) - 1.0) < 1e-8

    # テスト2: previous_reward=0.0 (Lose時), 全てのアームに対して確率出るか
    policy.previous_action = 1
    policy.previous_reward = 0.0
    probs_lose = selection_probabilities(policy, estimator)
    @assert length(probs_lose) == n_arms
    @assert all(prob -> prob ≥ 0.0 && prob ≤ 1.0, probs_lose)
    @assert abs(sum(probs_lose) - 1.0) < 1e-8

    # テスト3: previous_rewardが1.0/0.0でない場合、確率ベクトルがすべて0になる
    # policy.previous_action = 2
    # policy.previous_reward = NaN
    # probs_nan = selection_probabilities(policy, estimator)
    # @assert length(probs_nan) == n_arms
    # こちらのケースでは定義次第なので振る舞いを出力
    # println("previous_reward = NaN でのprobs: ", probs_nan)

    # test 4: when n_arms == 2, the model coincides with the original noisy-WSLS model
    if n_arms == 2
        @assert all(probs_win .== [ϵ/2, 1-ϵ/2])
        @assert all(probs_lose .== [ϵ/2, 1-ϵ/2])
    end
    println("n_arms = $n_arms, ϵ = $ϵ")
    println("2でWin時の確率: ", probs_win)
    println("1でLose時の確率: ", probs_lose)
end

# テスト関数の呼び出し
#= test_selection_probabilities()
test_selection_probabilities(5, 1.0)
test_selection_probabilities(2, 0.5)
test_selection_probabilities(2, 1.0)
test_selection_probabilities(4, 0.0)
 =#
