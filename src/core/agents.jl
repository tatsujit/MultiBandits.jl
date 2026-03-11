# preset agents (bandit algorithms) to be predefined here. 

"""
    SimpleSatAgent(n_arms::Int, aspiration::Float64) -> Agent

`SimpleSat(aspiration)` + `EmpiricalReward(n_arms)` のエージェントを生成する便利コンストラクタ。
"""
function SimpleSatAgent(n_arms::Int, aspiration::Float64)
    policy = SimpleSat(aspiration)
    estimator = EmpiricalReward(n_arms)
    Agent(policy, estimator)
end

"""
    randomResponding(n_arms::Int, selection_probabilities::Vector{Float64}) -> Agent

`RandomResponding` 方策と `EmptyEstimator` を組み合わせたエージェントを生成する。

# 引数
- `n_arms::Int` — 腕数
- `selection_probabilities::Vector{Float64}` — 行動選択確率

# 例
```julia
agent = randomResponding(3, [0.25, 0.5, 0.75])
```
"""
function randomRespondingAgent(n_arms::Int, selection_probabilities::Vector{Float64})
    @assert n_arms == length(selection_probabilities)
    policy = RandomResponding(selection_probabilities)
    estimator = EmptyEstimator()
    Agent(policy, estimator)
end

function wslsAgent(n_arms::Int)
    policy = WinStayLoseShift(n_arms)
    estimator = EmptyEstimator()
    Agent(policy, estimator)
end