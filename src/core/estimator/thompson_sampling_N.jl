# using Distributions

"""
    ThompsonSamplingN <: AbstractEstimator

Beta 分布から `samplesize` 回サンプリングし、その平均を行動価値とするトンプソンサンプリング。
`samplesize` が大きいほど探索が減り、搾取寄りになる。

# キーワード引数
- `samplesize::Int=1` — Beta 分布からのサンプル数（1 なら通常の ThompsonSampling と同等）
"""
struct ThompsonSamplingN <: AbstractEstimator
    n_arms::Int
    αs::Vector{Int} # success counts for each arm
    βs::Vector{Int} # failure counts for each arm
    Q::Vector{Float64} # 価値（ここではサンプルの値）
    W::Vector{Float64} # 総合的な「行動価値」
    N::Vector{Int} # 行動選択の回数（頻度）
    samplesize::Int # sample size from Beta distribution

    function ThompsonSamplingN(n_arms::Int; samplesize::Int=1)
        αs = ones(Int, n_arms) # success counts initialized to 1
        βs = ones(Int, n_arms) # failure counts initialized to 1
        Q = zeros(n_arms) # 行動価値
        W = zeros(n_arms) # 総合的な「行動価値」
        N = zeros(Int, n_arms) # 行動の選択頻度
        new(n_arms, αs, βs, Q, W, N, samplesize)
    end
end

function toString(e::ThompsonSamplingN)
    return "ThompsonSamplingN(n_arms=$(e.n_arms), samplesize=$(e.samplesize), αs=$(e.αs), βs=$(e.βs))"
end

function sample_action_values!(e::ThompsonSamplingN)
    n_arms = e.n_arms
    for a in 1:n_arms
        samplevalue = 0.0
        for _ in 1:e.samplesize
            samplevalue += rand(Beta(e.αs[a], e.βs[a]))
        end            
        sampleaverage = samplevalue / e.samplesize
        e.W[a] = e.Q[a] = sampleaverage # W==Q
    end
    return e.Q
end

"""
    Update ThompsonSampling as the estimator, with the observed reward for the selected action.
    For Thompson Sampling, this involves updating the success and failure counts.
    And the action values are re-sampled at the last stage of this update.
"""
function update!(e::ThompsonSamplingN, action::Int, reward::Float64)
    n_arms = e.n_arms
    reward == 1.0 ? (e.αs[action] += 1) : (e.βs[action] += 1)
    e.N[action] += 1
    sample_action_values!(e)
end

