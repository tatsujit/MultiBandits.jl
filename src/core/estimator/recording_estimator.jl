"""
    RecordingEstimator <: AbstractEstimator

何も推定しないが action, reward は覚えておく推定器。`NoisyWinStayLoseShift` 方策と組み合わせて使用する。
"""
mutable struct RecordingEstimator <: AbstractEstimator
    previous_action::Int
    previous_reward::Float64
    function RecordingEstimator()
        return new(0, NaN)
    end
end

"""
    update!(e::RecordingEstimator, action::Int, reward::Float64)
    previous_action と previous_reward を記録する。
"""
function update!(e::RecordingEstimator, action::Int, reward::Float64)
    e.previous_action = action
    e.previous_reward = reward
    return nothing
end
