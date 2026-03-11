"""
    EmptyEstimator <: AbstractEstimator

何も推定しないプレースホルダー推定器。`RandomResponding` 方策と組み合わせて使用する。
"""
struct EmptyEstimator <: AbstractEstimator end

"""
    update!(e::EmptyEstimator, action::Int, reward::Float64)

何もせず `nothing` を返す。
"""
function update!(e::EmptyEstimator, action::Int, reward::Float64)
    # EmptyEstimator doesn't maintain any state, so nothing to update
    return nothing
end
