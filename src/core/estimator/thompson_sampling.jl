# using Distributions

"""
    ThompsonSampling <: AbstractEstimator

Beta 分布によるトンプソンサンプリング推定器。Greedy 方策と組み合わせて使用する。

成功/失敗カウント（α, β）を更新し、Beta(α, β) からサンプリングした値を行動価値 W とする。

# 例
```julia
est = ThompsonSampling(4)  # 4本腕、Beta(1,1) で初期化
```
"""
struct ThompsonSampling <: AbstractEstimator
    n_arms::Int
    αs::Vector{Int} # success counts for each arm
    βs::Vector{Int} # failure counts for each arm
    Q::Vector{Float64} # 価値（ここではサンプルの値）
    W::Vector{Float64} # 総合的な「行動価値」
    N::Vector{Int} # 行動選択の回数（頻度）

    function ThompsonSampling(n_arms::Int)
        αs = ones(Int, n_arms) # success counts initialized to 1
        βs = ones(Int, n_arms) # failure counts initialized to 1
        Q = zeros(n_arms) # 行動価値
        W = zeros(n_arms) # 総合的な「行動価値」
        N = zeros(Int, n_arms) # 行動の選択頻度
        new(n_arms, αs, βs, Q, W, N)
    end
end

function toString(e::ThompsonSampling)
    return "ThompsonSampling(n_arms=$(e.n_arms), αs=$(e.αs), βs=$(e.βs))"
end

function sample_action_values!(e::ThompsonSampling)
    n_arms = e.n_arms
    for a in 1:n_arms
        e.W[a] = e.Q[a] = rand(Beta(e.αs[a], e.βs[a])) # W==Q
    end
    return e.Q
end

"""
    Update ThompsonSampling as the estimator, with the observed reward for the selected action.
    For Thompson Sampling, this involves updating the success and failure counts.
    And the action values are re-sampled at the last stage of this update.
"""
function update!(e::ThompsonSampling, action::Int, reward::Float64)
    n_arms = e.n_arms
    reward == 1.0 ? (e.αs[action] += 1) : (e.βs[action] += 1)
    e.N[action] += 1
    sample_action_values!(e)
end

