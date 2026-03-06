@testset "System run!" begin
    n_arms = 4
    trials = 100
    env = Environment([0.2, 0.4, 0.6, 0.8])

    @testset "Greedy + EmpiricalReward" begin
        agent = Agent(Greedy(), EmpiricalReward(n_arms))
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
        @test all(sys.history.actions .!= 0)
        @test all(r -> r == 0.0 || r == 1.0, sys.history.rewards)
    end

    @testset "Greedy + ThompsonSampling" begin
        agent = Agent(Greedy(), ThompsonSampling(n_arms))
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
        @test all(1 .<= sys.history.actions .<= n_arms)
    end

    @testset "Greedy + UCB1" begin
        agent = Agent(Greedy(), UCB1(n_arms))
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "Greedy + STEP" begin
        agent = Agent(Greedy(), STEP(n_arms, 0.5))
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "Greedy + RS" begin
        agent = Agent(Greedy(), RS(n_arms, 0.5))
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "RandomResponding + EmptyEstimator" begin
        agent = Agent(RandomResponding(n_arms), EmptyEstimator())
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "SimpleSat + EmpiricalReward" begin
        agent = SimpleSatAgent(n_arms, 0.5)
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "SoftmaxPolicy + CognitiveEstimator" begin
        est = CognitiveEstimator(n_arms; β=3.0, α=0.1)
        agent = Agent(SoftmaxPolicy(1.0), est)
        history = History(n_arms, trials)
        sys = System(agent, env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end

    @testset "NonStationaryEnvironment (before change point)" begin
        ns_env = NonStationaryEnvironment(
            [0.8, 0.2],
            [200],          # change point beyond trial count
            [[0.2, 0.8]]
        )
        agent = Agent(Greedy(), ThompsonSampling(2))
        history = History(2, trials)
        sys = System(agent, ns_env, history; rng=MersenneTwister(42))
        @test_nowarn run!(sys, trials)
    end
end
