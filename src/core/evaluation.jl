"""
    action_moving_averages(actions, n_arms, window_size)

    Calculate moving averages of action selections.

    Args:
        actions: Vector of selected actions (1-indexed)
        n_arms: Number of arms/actions
        window_size: Size of moving average window

    Returns:
        Matrix (n_trials × n_arms) of moving average selection frequencies
"""
function action_moving_averages(actions::Vector{Int}, n_arms::Int, window_size::Int)
    n_trials = length(actions)
    moving_averages = zeros(n_trials, n_arms)

    for t in 1:n_trials
        start_idx = max(1, t - window_size + 1)
        end_idx = t

        window_actions = actions[start_idx:end_idx]
        window_length = length(window_actions)

        for arm in 1:n_arms
            count = sum(window_actions .== arm)
            moving_averages[t, arm] = count / window_length
        end
    end

    return moving_averages
end

"""
    cumulative_reward(rewards)

    Calculate cumulative rewards over trials.

    Args:
        rewards: Vector of rewards

    Returns:
        Vector of cumulative rewards at each trial
"""
function cumulative_reward(rewards::Vector{Float64})
    cumsum(rewards)
end

"""
    average_rewards(rewards)

    Calculate running average reward at each trial.

    Args:
        rewards: Vector of rewards

    Returns:
        Vector of average rewards at each trial
"""
function average_rewards(rewards::Vector{Float64})
    n = length(rewards)
    cumsum(rewards) ./ (1:n)
end

"""
    moving_average_rewards(rewards, window_size)

    Calculate moving average of rewards with a sliding window.

    Args:
        rewards: Vector of rewards
        window_size: Size of moving average window

    Returns:
        Vector of moving average rewards at each trial
"""
function moving_average_rewards(rewards::Vector{Float64}, window_size::Int)
    n = length(rewards)
    moving_avg = zeros(n)
    for t in 1:n
        start_idx = max(1, t - window_size + 1)
        moving_avg[t] = Statistics.mean(rewards[start_idx:t])
    end
    moving_avg
end

"""
    find_optimal(expectations)

Find the optimal arm and its expected reward.

Returns:
    NamedTuple (arm=optimal_arm, reward=optimal_reward)
"""
function find_optimal(expectations::Vector{Float64})
    optimal_arm = argmax(expectations)
    optimal_reward = expectations[optimal_arm]
    (arm=optimal_arm, reward=optimal_reward)
end

"""
    cumulative_regret(actions, expectations::Vector{Float64})

    Calculate cumulative regret for stationary environment (constant expectations).
    Regret(t) = Σ_{τ=1}^{t} [max(expectations) - expectations[action_τ]]

    Args:
        actions: Vector of selected actions (1-indexed)
        expectations: Vector of expectations for each trial

    Returns:
        Vector of cumulative regret at each trial
"""
function cumulative_regret(actions::Vector{Int}, expectations::Vector{Float64})
    optimal_reward = maximum(expectations)
    n = length(actions)
    regret = zeros(n)
    for t in 1:n
        action = actions[t]
        regret_increment = optimal_reward - expectations[action]
        regret[t] = (t == 1) ? regret_increment : regret[t-1] + regret_increment
    end
    regret
end

"""
    cumulative_regret(actions, expectations::Vector{Vector{Float64}})

    Calculate cumulative regret for per-trial expectations 
        (such as when the environment is non-stationary)

    Args:
        actions: Vector of selected actions (1-indexed)
        expectations: Vector of Vector of expectations for each trial

    Returns:
        Vector of cumulative regret at each trial
"""
function cumulative_regret(actions::Vector{Int}, expectations::Vector{Vector{Float64}})
    n = length(actions)
    regret = zeros(n)
    for t in 1:n
        action = actions[t]
        optimal_reward = maximum(expectations[t])
        regret_increment = optimal_reward - expectations[t][action]
        regret[t] = (t == 1) ? regret_increment : regret[t-1] + regret_increment
    end
    regret
end

"""
    average_reward(systems, trials)

    Calculate average reward across multiple System runs at each trial.

    Args:
        systems: Vector of System
        trials: Number of trials

    Returns:
        Vector of average reward at each trial
"""
function average_reward(systems::Vector{System}, trials::Int)::Vector{Float64}
    all_rewards = [sys.history.rewards for sys in systems]
    n_systems = length(systems)
    average_reward = zeros(trials)
    for t in 1:trials
        total_reward = 0.0
        for i in 1:n_systems
            total_reward += all_rewards[i][t]
        end
        average_reward[t] = total_reward / n_systems
    end
    average_reward
end
