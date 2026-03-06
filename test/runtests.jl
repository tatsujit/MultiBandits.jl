using MultiBandits
using Test

println("Starting tests")
ti = time()

@testset "MultiBandits.jl" begin
    # include("test_estimators.jl")
    # include("test_system.jl")
    # include("test_selection_probabilities.jl")
    # include("test_evaluation.jl")
end

ti = time() - ti
println("\nTest took total time of:")
println(round(ti / 60, digits=3), " minutes")
