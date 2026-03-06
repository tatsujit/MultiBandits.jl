"""
    AbstractPolicy

全方策の基底抽象型。サブタイプは `select_action` と `selection_probabilities` を実装する。
"""
abstract type AbstractPolicy end

# include("./epsilon_greedy.jl")
# include("./ucb.jl")
# include("./ucb_tuned.jl")
# include("./satisficing.jl")
include("policy/greedy.jl")
include("policy/random_responding.jl")
include("policy/softmax.jl")
include("policy/simple_sat.jl")
