@testset "selection_probabilities" begin
    n_arms = 5

    function is_probability_distribution(probs::Vector{Float64})
        all(probs .>= 0.0) && isapprox(sum(probs), 1.0; atol=1e-10)
    end

    @testset "Greedy + EmpiricalReward" begin
        e = EmpiricalReward(n_arms)
        update!(e, 3, 1.0)
        probs = selection_probabilities(Greedy(), e)
        @test is_probability_distribution(probs)
        @test probs[3] == 1.0
    end

    @testset "Greedy: tied arms" begin
        e = EmpiricalReward(n_arms)
        probs = selection_probabilities(Greedy(), e)
        @test is_probability_distribution(probs)
        @test all(probs .≈ 1.0 / n_arms)
    end

    @testset "SoftmaxPolicy" begin
        values = [0.1, 0.5, 0.3, 0.8, 0.2]
        probs = selection_probabilities(SoftmaxPolicy(1.0), values)
        @test is_probability_distribution(probs)

        # high β concentrates on best arm
        probs_hot = selection_probabilities(SoftmaxPolicy(100.0), values)
        @test is_probability_distribution(probs_hot)
        @test probs_hot[4] > 0.99

        # β=0 gives uniform
        probs_cold = selection_probabilities(SoftmaxPolicy(0.0), values)
        @test is_probability_distribution(probs_cold)
        @test all(probs_cold .≈ 1.0 / n_arms)
    end

    @testset "RandomResponding" begin
        probs = selection_probabilities(RandomResponding(n_arms), EmptyEstimator())
        @test is_probability_distribution(probs)
        @test all(probs .≈ 1.0 / n_arms)

        custom_probs = [0.1, 0.2, 0.3, 0.25, 0.15]
        probs2 = selection_probabilities(RandomResponding(custom_probs), EmptyEstimator())
        @test is_probability_distribution(probs2)
        @test probs2 == custom_probs
    end

    @testset "SimpleSat" begin
        e = EmpiricalReward(n_arms)
        update!(e, 2, 1.0)
        probs = selection_probabilities(SimpleSat(0.5), e)
        @test is_probability_distribution(probs)
    end

    @testset "CognitiveEstimator (W vector)" begin
        e = CognitiveEstimator(n_arms; β=2.0)
        update!(e, 1, 1.0)
        update!(e, 3, 0.5)
        probs = selection_probabilities(e.W)
        @test is_probability_distribution(probs)
    end

    @testset "available_arms" begin
        W = [0.1, 0.5, 0.3, 0.8, 0.2]
        available = [2, 4, 5]
        probs = selection_probabilities(W, available)
        @test is_probability_distribution(probs)
        @test probs[1] == 0.0
        @test probs[3] == 0.0
        @test all(probs[available] .> 0.0)
    end
end
