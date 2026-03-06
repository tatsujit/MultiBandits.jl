# import Pkg; Pkg.add("FunctionWrappers")
# using FunctionWrappers: FunctionWrapper
# 型を指定するにはいいだろうけど、ライブラリ増やしたくないのでとりあえずやめた

# suppose that the utility function has the type of 
# f: [0, 1] → [0, 1]. 
abstract type UtilityFunction end

struct StepUtilityFunction <: UtilityFunction
    origin::Float64
    uf::Function
    function StepUtilityFunction(origin::Float64)
        @assert 0 ≤ origin ≤ 1 "origin must be in [0, 1]"
        uf = step_utility_function(x, origin=origin)
        @assert 0 ≤ uf(0) ≤ uf(1) ≤ 1 "uf must be in [0, 1]"
        return new(origin, uf)
    end
end

struct LinearToStep <: UtilityFunction
    origin::Float64
    weight::Float64 # 0.0: totally linear, 1.0: totally step
    uf::Function
    function LinearStepHybridUtilityFunction(origin::Float64, weight::Float64)
        @assert 0 ≤ origin ≤ 1 "origin must be in [0, 1]"
        uf = x -> (1-weight)x + step_utility_function(x, origin=origin)
        @assert 0 ≤ uf(0) ≤ uf(1) ≤ 1 "uf must be in [0, 1]"
        return new(origin, weight, uf)
    end
end

struct SigmoidUtilityFunction{F} <: UtilityFunction
    origin::Float64
    uf::F
end