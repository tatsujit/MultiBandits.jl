abstract type AbstractSystem end

"""
    System <: AbstractSystem

エージェント・環境・履歴を束ねたシミュレーションシステム。

# フィールド
- `agent::Agent` — バンディットエージェント
- `env::AbstractEnvironment` — バンディット環境
- `history::AbstractHistory` — 試行履歴の記録先
- `rng::AbstractRNG` — 乱数生成器

# 例
```julia
system = System(agent, env, History(n_arms, 100); rng=MersenneTwister(42))
run!(system, 100)
```
"""
struct System <: AbstractSystem
    agent::Agent
    env::AbstractEnvironment
    history::AbstractHistory
    rng::AbstractRNG
    function System(agent::Agent, env::AbstractEnvironment, history::AbstractHistory; rng=Random.GLOBAL_RNG)
        new(agent, env, history, rng)
    end
end

"""
    run!(system::System, trials::Int, verbose::Bool=false)

`trials` 回の試行を実行する。各試行で `step!` を呼び出す。
"""
function run!(system::System, trials::Int, verbose::Bool=false)
    for trial in 1:trials
        step!(system, trial, verbose)
    end
end

"""
    step!(system::System, trial::Int, verbose::Bool=false)

1ステップ (t → t+1) を実行する。行動選択 → 報酬サンプリング → 履歴記録 → 推定器更新 の順。
"""
function step!(system::System, trial::Int, verbose::Bool=false)
    expectations = mean(system.env) # calculate the expectations for the current environment
    # @ic expectations
    action = select_action(system.agent.policy, system.agent.estimator; rng=system.rng)
    reward = sample_reward(system.env, action; rng=system.rng)
    # @ic expectations
    record!(system.history, trial, action, expectations, reward, system.agent.estimator)
    update!(system.agent.estimator, action, reward)

    ################################################################
    # *for sleeping environments*
    ################################################################
    # if typeof(env) != LAEnvironment && typeof(env) != NLAEnvironment
    #     1
    # else # for regular (all actions always available) environments
    #     sample_available_arms!(system) # limited available arms をここで決める
    #     # shuffle_available_arms!(system)
    #     available_arms = env.available_arms
    #     action = select_action(policy, estimator, available_arms; rng)
    #     reward = sample_reward(env, action, rng=rng)
    #     record!(history, trial, action, available_arms, expectations, reward)
    #     update!(estimator, action, reward) # ここでも utility_function を入れるか
    # end
end

