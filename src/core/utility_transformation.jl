# Step utility function implementation
step(origin::Real) = x -> x >= origin ? 1.0 : 0.0
step_f(origin::Real) = step(origin)
"""
    step_linear(origin::Real, weight::Real)::Function

    origin is the discontinuous point. 
    weight is the `step-ness` parameter: 
        weight=0.0 means the identity function, and
        weight=1.0 means the step function. 
"""
step_linear(origin::Real, weight::Real) = x -> (1-weight)*x + weight*step_f(origin)(x)

logistic(origin::Real, steepness::Real=10.0) = 
    x -> 1.0 / (1.0 + exp(-steepness * (x - origin)))

"""
    normalized logistic function f that satisfies f(0) = 0.0 and f(1) = 1.0
"""
function normalized_logistic(origin::Float64, steepness::Float64=10.0)
    value0 = logistic(origin, steepness)(0.0)
    value1 = logistic(origin, steepness)(1.0)
    coeff = 1 / (value1 - value0)
    # x -> (sigmoid_utility_function(origin, steepness)(x) - value0)
    x -> coeff * (logistic(origin, steepness)(x) - value0)
end



# ###############################################################
# # testing by plotting
# ###############################################################

# using CairoMakie

# xs = range(0, 1, length=200)

# fig = Figure(size=(1200, 800))

# # (1) step
# ax1 = Axis(fig[1, 1], title="step(origin)", xlabel="x", ylabel="utility")
# for origin in [0.3, 0.5, 0.7]
#     lines!(ax1, xs, step(origin).(xs), label="origin=$origin")
# end
# axislegend(ax1; position=:lt)

# # (2) step_linear
# ax2 = Axis(fig[1, 2], title="step_linear(origin, weight)", xlabel="x", ylabel="utility")
# for (origin, weight) in [(0.1, 0.0), (0.3, 0.3), (0.5, 0.7), (0.7, 1.0)]
#     lines!(ax2, xs, step_linear(origin, weight).(xs), label="origin=$origin, w=$weight")
# end
# axislegend(ax2; position=:lt)

# # (3) logistic
# ax3 = Axis(fig[2, 1], title="logistic(origin, steepness)", xlabel="x", ylabel="utility")
# for (origin, steepness) in [(0.5, 5.0), (0.5, 10.0), (0.5, 30.0), (0.3, 10.0), (0.7, 10.0)]
#     lines!(ax3, xs, logistic(origin, steepness).(xs), label="o=$origin, k=$steepness")
# end
# axislegend(ax3; position=:lt)

# # (4) normalized_logistic
# ax4 = Axis(fig[2, 2], title="normalized_logistic(origin, steepness)", xlabel="x", ylabel="utility")
# for (origin, steepness) in [(0.5, 1.0), (0.5, 5.0), (0.5, 10.0), (0.5, 30.0), (0.3, 10.0), (0.7, 10.0), ]
#     lines!(ax4, xs, normalized_logistic(origin, steepness).(xs), label="o=$origin, k=$steepness")
# end
# axislegend(ax4; position=:lt)

# display(fig)

# safesave(plotsdir("utility_transformation.pdf"), fig)