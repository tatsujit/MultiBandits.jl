# SoftmaxPolicy <: AbstractPolicy
# - computes the selection probabilities

"""
    SoftmaxPolicy <: AbstractPolicy

Softmax（Boltzmann）方策。逆温度 `β` で行動価値から選択確率を計算する。

確率は `P(a) = exp(β * W[a]) / Σ exp(β * W[i])` で計算される。

# コンストラクタ
- `SoftmaxPolicy()` — β=1.0（デフォルト）
- `SoftmaxPolicy(β::Float64)` — 逆温度を指定
"""
mutable struct SoftmaxPolicy <: AbstractPolicy
    β::Float64
    function SoftmaxPolicy() new(1.0) end
    function SoftmaxPolicy(β::Float64) new(β) end
end

function selection_probabilities(policy::SoftmaxPolicy, values::Vector{Float64})
    n_arms = length(values)
    proto_probs = [exp(policy.β * v) for v in values]
    return proto_probs / sum(proto_probs)
end

function selection_probabilities(β::Float64, values::Vector{Float64})
    n_arms = length(values)
    proto_probs = [exp(β * v) for v in values]
    return proto_probs / sum(proto_probs)
end

"""
Policy >: Softmax の持っている β を無視する（実質 Softmax という型だけ参照している）
W というのは行動効用 V と固執性 C のそれぞれに、それぞれの逆温度 β, φ をかけた合計のベクトル。
Estimator has its own β, so we use estimator.β instead of policy.β (= 1.0).
"""
function selection_probabilities(estimator::CognitiveEstimator)
    n_arms = length(estimator.W)
    proto_probs = [exp(w) for w in estimator.W]
    return proto_probs / sum(proto_probs)
end
