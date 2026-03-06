"""
    RS <: AbstractEstimator

リスク感応型満足化 (Risk-Sensitive Satisficing) 推定器。
行動価値は `W[a] = N[a] * (Q[a] - aleph)` で計算される。Greedy 方策と組み合わせて使用する。

# 引数
- `n_arms::Int` — 腕の数
- `aleph::Float64` — 要求水準（aspiration level）

# 例
```julia
est = RS(4, 0.5)  # 4本腕、要求水準 0.5
```
"""
struct RS <: AbstractEstimator
    n_arms::Int    
    Q::Vector{Float64} # 価値（報酬平均）
    W::Vector{Float64} # 行動選択に用いる RS 値
    N::Vector{Int} # 各行動の選択回数（頻度）
    aleph::Float64
    function RS(n_arms::Int, aleph::Float64)
        Q = zeros(n_arms)
        W = zeros(n_arms)
        N = zeros(Int, n_arms)
        new(n_arms, Q, W, N, aleph)
    end
end

function toString(e::RS)
    return "RS(n_arms=$(e.n_arms), aleph=$(e.aleph))"
end

"""
    Update RS as the estimator, with the observed reward for the selected action.
"""
function update!(e::RS, action::Int, reward::Float64)
    n_arms = e.n_arms
    e.N[action] += 1 
    e.Q[action] = e.Q[action] * (e.N[action]-1) / e.N[action] + reward / e.N[action] # reward average update 
    # value re-calculation
    for a in 1:n_arms
        e.W[a] = e.N[a] * (e.Q[a] - e.aleph)
    end
end

"""
    The RS values normalized to close to [0, 1] (not totally; see Kamiya & Takahashi)
"""
function normalized_rs_values(e::RS)
    totalN = sum(e.N)
    return e.W ./ totalN
end
