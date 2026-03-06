"""
    Agent

方策 (`AbstractPolicy`) と推定器 (`AbstractEstimator`) を組み合わせたバンディットエージェント。

# フィールド
- `policy::AbstractPolicy` — 行動選択方策
- `estimator::AbstractEstimator` — 行動価値推定器
"""
struct Agent
    policy::AbstractPolicy
    estimator::AbstractEstimator
end

toString(agent::Agent) = "Agent($(toString(agent.policy))_$(toString(agent.estimator)))"

include("agents.jl") # predefined agents (= policy x estimator)




################################################################
# for "Model" Estimator
################################################################
"""
Estimator が Model の場合は Policy >: Softmax の持っている β を無視する（実質 Softmax という型だけ参照）
W というのは行動効用 V と固執性 C のそれぞれに、それぞれの逆温度 β, φ をかけた合計のベクトル。
φ か τ のどちらかが 0.0 ならば、C は zeros なので影響なし。
"""
function select_action(policy::SoftmaxPolicy, estimator::CognitiveEstimator; rng::AbstractRNG=Random.default_rng())
    n_arms = length(estimator.W)
    # @ic estimator
    # probs = selection_probabilities(estimator.W)
    probs = selection_probabilities(policy, estimator.W)
    return sample(rng, 1:n_arms, Weights(probs))
end
"""
LAEnvironmentでの select_action
"""
function select_action(policy::SoftmaxPolicy, estimator::CognitiveEstimator, available_arms::Vector{Int}; rng::AbstractRNG=Random.default_rng())
    n_arms = length(estimator.W)
    probs = selection_probabilities(estimator.W, available_arms)
    return sample(rng, 1:n_arms, Weights(probs))
end
