# preset agents (bandit algorithms) to be predefined here. 

"""
    SimpleSatAgent(n_arms::Int, aspiration::Float64) -> Agent

`SimpleSat(aspiration)` + `EmpiricalReward(n_arms)` のエージェントを生成する便利コンストラクタ。
"""
function SimpleSatAgent(n_arms::Int, aspiration::Float64)
    policy = SimpleSat(aspiration)
    estimator = EmpiricalReward(n_arms)
    return Agent(policy, estimator)
end