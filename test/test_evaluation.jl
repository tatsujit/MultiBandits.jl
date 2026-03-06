@testset "Evaluation" begin
    @testset "cumulative_regret (stationary)" begin
        actions = [1, 1, 2, 2, 2]
        expectations = [0.3, 0.7]
        regret = cumulative_regret(actions, expectations)
        @test regret[2] ≈ 0.8
        @test regret[5] ≈ 0.8
        @test all(diff(regret) .>= 0.0)
    end

    @testset "cumulative_regret (non-stationary)" begin
        actions = [1, 2, 1]
        expectations = [[0.3, 0.7], [0.7, 0.3], [0.5, 0.5]]
        regret = cumulative_regret(actions, expectations)
        # trial 1: 0.7-0.3=0.4, trial 2: 0.7-0.3=0.4, trial 3: 0.5-0.5=0.0
        @test regret[1] ≈ 0.4
        @test regret[2] ≈ 0.8
        @test regret[3] ≈ 0.8
    end

    @testset "moving_average_rewards" begin
        rewards = [1.0, 0.0, 1.0, 1.0, 0.0]
        ma = moving_average_rewards(rewards, 3)
        @test length(ma) == 5
        @test ma[3] ≈ 2.0 / 3.0
        @test ma[5] ≈ 2.0 / 3.0
    end

    @testset "average_rewards" begin
        rewards = [1.0, 0.0, 1.0, 0.0]
        avg = average_rewards(rewards)
        @test avg[1] == 1.0
        @test avg[4] ≈ 0.5
    end

    @testset "action_moving_averages" begin
        actions = [1, 1, 2, 2, 1]
        ma = action_moving_averages(actions, 2, 3)
        @test size(ma) == (5, 2)
        # each row sums to 1
        for t in 1:5
            @test sum(ma[t, :]) ≈ 1.0
        end
    end
end
