"""
    Greedy <: AbstractPolicy

最大行動価値を持つ腕を選択する貪欲方策。同値の腕が複数ある場合はランダムに選択する。

# キーワード引数
- `randomize::Bool=true` — 同値腕をランダムに選ぶか、最初のインデックスを返すか
"""
struct Greedy <: AbstractPolicy
    randomize::Bool
    function Greedy(; randomize::Bool=true)
        new(randomize)
    end
end

"""
    Compute selection probabilities for each action based on the estimator's action values.
    Note that for GreedyPolicy, this will return a vector where the action(s) with the highest 
    value have equal probability, and all others have zero probability.

    NOTE: probably a small probability ϵ should be added to zero probability actions to allow likelihood calculation?
"""
function selection_probabilities(policy::Greedy, estimator::AbstractEstimator)
    values = estimator.W
    n_arms = length(values)
    max_value = maximum(values)
    candidates = findall(x -> x == max_value, values)
    unif_prob = 1.0 / length(candidates)
    probs = [a ∈ candidates ? unif_prob : 0.0 for a in 1:n_arms]
    return probs
end

# selection_probabilities(GreedyPolicy(), ThompsonSampling(10))

# n_arms = 10; est1 = Estimator(n_arms); update!(est1, 3, 1.0); update!(est1, 9, 1.0)
# selection_probabilities(GreedyPolicy(), est1)'
#
# # julia> selection_probabilities(GreedyPolicy(), est1)'
# # 1×10 adjoint(::Vector{Float64}) with eltype Float64:
# #  0.0  0.0  0.5  0.0  0.0  0.0  0.0  0.0  0.5  0.0
# 
# [select_action(GreedyPolicy(), est1; randomize=true) for _ in 1:10]'
# #  3  3  9  3  3  9  9  3  9  3

"""
    Select action based on Greedy policy using argmax on the estimator's action values.
    If multiple actions have the same maximum value, one is selected at random if `randomize` is true.
"""
function select_action(p::Greedy, e::AbstractEstimator; 
                       rng::AbstractRNG=Random.default_rng())
    return argmax(e.W; rng=rng, randomize=p.randomize)
end

"""
    Base.argmax(vec::Vector{Float64}; rng::AbstractRNG=Random.default_rng(), randomize::Bool=true)
    argmax() for action selection. 
    Because there is Base.argmax(), we need to extend it here for our use case.
    The error message was below: 
    # ERROR: invalid method definition in Main: function Base.argmax must be explicitly imported to be extended
"""
function Base.argmax(vec::Vector{Float64}; rng::AbstractRNG=Random.default_rng(), randomize::Bool=true)
    max_value = maximum(vec)
    candidates = findall(x -> x == max_value, vec)
    if candidates == Int64[] @ic0; @show vec end
    return randomize ? rand(rng, candidates) : candidates[1] # randomize == false ならば最初のインデックスを返す
end
