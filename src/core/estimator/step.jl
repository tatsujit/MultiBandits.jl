"""
    STEP <: AbstractEstimator

閾値超過確率による満足化 (Satisficing with Threshold Exceeding Probability) 推定器。
Beta 分布の右裾確率 `P(X ≥ aleph)` を行動価値 W とする。Greedy 方策と組み合わせて使用する。

# 引数
- `n_arms::Int` — 腕の数
- `aleph::Float64` — 要求水準（閾値）

# 例
```julia
est = STEP(4, 0.5)  # 4本腕、閾値 0.5
```
"""
struct STEP <: AbstractEstimator
    n_arms::Int
    αs::Vector{Int} # success counts for each arm
    βs::Vector{Int} # failure counts for each arm
    Q::Vector{Float64} # 価値（報酬平均）、行動選択には用いない
    W::Vector{Float64} # 行動選択に用いる確率値（右裾確率値）
    N::Vector{Int} # 各行動の選択回数（頻度）
    aleph::Float64
    function STEP(n_arms::Int, aleph::Float64)
        αs = ones(Int, n_arms) # success counts initialized to 1
        βs = ones(Int, n_arms) # failure counts initialized to 1
        Q = zeros(n_arms)
        W = zeros(n_arms)            
        N = zeros(Int, n_arms) # `N == αs + βs .- 2` holds
        new(n_arms, αs, βs, Q, W, N, aleph)
    end
end

function toString(e::STEP)
    return "STEP(n_arms=$(e.n_arms), αs=$(e.αs), βs=$(e.βs), aleph=$(e.aleph))"
end

"""
    Calculate STEP action values (threshold exceedence probabilities) based on current αs and βs.
"""
function calculate_action_values!(e::STEP)
    n_arms = e.n_arms
    for a in 1:n_arms
        e.Q[a] = e.αs[a] / (e.αs[a] + e.βs[a]) # reward average
        # STEP value calculation: probability that the reward exceeds the aspiration level `aleph`
        e.W[a] = 1.0 - cdf(Beta(e.αs[a], e.βs[a]), e.aleph)
    end
end
"""
    Update STEP as the estimator, with the observed reward for the selected action.
"""
function update!(e::STEP, action::Int, reward::Float64)
    n_arms = e.n_arms
    reward == 1.0 ? (e.αs[action] += 1) : (e.βs[action] += 1)
    e.N[action] += 1
    calculate_action_values!(e)
end


