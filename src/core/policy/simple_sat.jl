# Michel, T., Hajiabolhassan, H., & Ortner, R. (2023). Regret Bounds for Satisficing in Multi-Armed Bandit Problems. Transactions on Machine Learning Research. https://openreview.net/forum?id=QnT41ZGNh9
# p. 4 Algorithm 1

# `notebook/on_action_selection_by_UCB-family.ipynb`
# でも試したりしている

"""
    SimpleSat <: AbstractPolicy

単純満足化方策。要求水準 `aspiration` 以上の腕があれば最大価値の腕を搾取し、
なければ全腕からランダムに探索する。

# 引数
- `aspiration::Float64` — 要求水準

# 例
```julia
agent = Agent(SimpleSat(0.5), EmpiricalReward(4))
```
"""
mutable struct SimpleSat <: AbstractPolicy
    aspiration::Float64
    randomize::Bool
    # initial_choices::Vector{Int}
    function SimpleSat(aspiration::Float64; randomize::Bool=true)
        # needs n_arm
        new(aspiration, randomize) # zeros(1)
    end
end

"""
    Compute selection probabilities for each action based on the estimator's action values.
    Note that for GreedyPolicy, this will return a vector where the action(s) with the highest 
    value have equal probability, and all others have zero probability.

    NOTE: probably a small probability ϵ should be added to zero probability actions to allow likelihood calculation?
"""
function selection_probabilities(policy::SimpleSat, estimator::AbstractEstimator)
    values = estimator.W
    n_arms = length(values)
    max_value = maximum(values)
    candidates = findall(x -> x == max_value, values)
    unif_prob = 1.0 / length(candidates)
    probs = [a ∈ candidates ? unif_prob : 0.0 for a in 1:n_arms]
    return probs
end


"""
    Select action based on Greedy policy using argmax on the estimator's action values.
    If multiple actions have the same maximum value, one is selected at random if `randomize` is true.
"""
function select_action(p::SimpleSat, e::AbstractEstimator; 
                       rng::AbstractRNG=Random.default_rng())
    # if 'achievable', exploit
    if any(>=(p.aspiration), e.W) 
        # @ic "achievable"
        # @ic e.W, p.aspiration
        #Base.argmax in greedy.jl
        return argmax(e.W; rng=rng, randomize=p.randomize)
    # if 'unachievable", explore
    else
        return rand(rng, 1:e.n_arms)
    end
end

