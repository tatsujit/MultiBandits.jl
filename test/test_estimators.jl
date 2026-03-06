@testset "Estimators" begin
    n_arms = 3

    @testset "EmpiricalReward" begin
        e = EmpiricalReward(n_arms)
        @test all(e.Q .== 0.0)
        @test all(e.N .== 0)

        update!(e, 1, 1.0)
        @test e.Q[1] == 1.0
        @test e.N[1] == 1
        @test e.W[1] == 1.0

        update!(e, 1, 0.0)
        @test e.Q[1] ≈ 0.5
        @test e.N[1] == 2

        # untouched arms
        @test e.Q[2] == 0.0
        @test e.N[2] == 0
    end

    @testset "ThompsonSampling" begin
        e = ThompsonSampling(n_arms)
        @test all(e.αs .== 1)
        @test all(e.βs .== 1)

        update!(e, 2, 1.0)
        @test e.αs[2] == 2
        @test e.βs[2] == 1
        @test e.N[2] == 1

        update!(e, 2, 0.0)
        @test e.αs[2] == 2
        @test e.βs[2] == 2
        @test e.N[2] == 2

        @test all(isfinite.(e.W))
    end

    @testset "UCB1" begin
        e = UCB1(n_arms)
        @test all(e.Q .== Inf)

        update!(e, 1, 1.0)
        update!(e, 2, 0.0)
        update!(e, 3, 1.0)
        @test e.Q[1] == 1.0
        @test e.Q[2] == 0.0
        @test e.Q[3] == 1.0
        @test all(e.N .== 1)
        @test all(isfinite.(e.B))
        @test all(e.B .> 0.0)
    end

    @testset "STEP" begin
        aleph = 0.5
        e = STEP(n_arms, aleph)

        update!(e, 1, 1.0)
        update!(e, 1, 1.0)
        update!(e, 1, 1.0)
        @test e.W[1] > 0.5
        @test e.W[2] ≈ 0.5
        @test e.W[3] ≈ 0.5
    end

    @testset "RS" begin
        aleph = 0.5
        e = RS(n_arms, aleph)

        for _ in 1:3
            update!(e, 1, 1.0)
        end
        # RS value = N[a] * (Q[a] - aleph) = 3 * (1.0 - 0.5) = 1.5
        @test e.W[1] ≈ 1.5
        @test e.W[2] == 0.0
    end

    @testset "CognitiveEstimator (default = sample average)" begin
        e = CognitiveEstimator(n_arms)
        update!(e, 1, 1.0)
        update!(e, 1, 0.0)
        @test e.Q[1] ≈ 0.5
        @test e.N[1] == 2
    end

    @testset "CognitiveEstimator (fixed learning rate)" begin
        α = 0.3
        e = CognitiveEstimator(n_arms; α=α)
        update!(e, 1, 1.0)
        @test e.Q[1] ≈ 0.3
        update!(e, 1, 0.0)
        @test e.Q[1] ≈ 0.21
    end

    @testset "CognitiveEstimator (forgetting)" begin
        e = CognitiveEstimator(n_arms; αf=0.1, μ=0.5)
        update!(e, 1, 1.0)
        update!(e, 1, 1.0)
        # arm 2 is unselected, should drift toward μ=0.5
        @test e.Q[2] > 0.0
    end

    @testset "CognitiveEstimator (stickiness)" begin
        e = CognitiveEstimator(n_arms; τ=0.5, φ=1.0)
        update!(e, 1, 1.0)
        @test e.C[1] > 0.0
        @test e.C[2] == 0.0
        @test e.W[1] > e.V[1]
    end

    @testset "CognitiveEstimator (BetaCDF utility)" begin
        # Use fixed learning rate so Q stays in (0,1) after update
        e = CognitiveEstimator(n_arms; η=0.3, ν=5.0, α=0.5)
        update!(e, 1, 1.0)
        # Q[1] = 0.0 + 0.5*(1.0-0.0) = 0.5, V = betaCDF(0.5) != 0.5
        @test e.V[1] != e.Q[1]
        @test 0.0 <= e.V[1] <= 1.0
    end

    @testset "EmptyEstimator" begin
        e = EmptyEstimator()
        @test update!(e, 1, 1.0) === nothing
    end
end
