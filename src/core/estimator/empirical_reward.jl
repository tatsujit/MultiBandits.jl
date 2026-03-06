# the most basic estimator: empirical reward estimator
# this is also used for Simple-Sat and so on. 

"""
    EmpiricalReward <: AbstractEstimator

標本平均による行動価値推定器。各腕の報酬平均を Q 値として維持する。

# キーワード引数
- `Q0::Float64=0.0` — Q 値の初期値

# 例
```julia
est = EmpiricalReward(4)       # 4本腕、Q0=0.0
est = EmpiricalReward(4; Q0=Inf)  # 楽観的初期値
```
"""
struct EmpiricalReward <: AbstractEstimator
    n_arms::Int
    Q::Vector{Float64} # 価値（通常のQ値や平均報酬など）
    W::Vector{Float64} # 総合的な「行動価値」 W = βV + φC でありこれを Policy で行動選択に使う
    N::Vector{Int64} # 行動選択の回数（頻度）
    Q0::Float64
    function EmpiricalReward(n_arms::Int; Q0::Float64=0.0) # Q0::Float=Inf
        Q = fill(Q0, n_arms) # 行動価値
        N = zeros(n_arms) # 行動の選択頻度
        W = fill(Q0, n_arms)
        new(n_arms, Q, W, N, Q0)
    end
end

"""
    TODO: display the values in Q and N too?
"""
function toString(e::EmpiricalReward)
    return "EmpiricalReward(n_arms=$(e.n_arms))"
end

"""
    display() したときの表示
"""
function Base.show(io::IO, ::MIME"text/plain", obj::EmpiricalReward)
    toString(obj)
end

"""
    update!(e::EmpiricalReward, action::Int, reward::Float64)

選択した腕の Q 値を標本平均で更新し、W = Q とする。
"""
function update!(e::EmpiricalReward, action::Int, reward::Float64)
    n_arms = length(e.Q)
    e.N[action] += 1
    # 行動価値Q (報酬推定値) の更新
    # 報酬平均を価値Qにする
    # 報酬平均の初期値を 0/0 から Inf としているので、通常の更新では Inf のままになってしまう
    # 更新式によっては NaN になっちゃうかも
    if e.Q[action] == Inf
        e.Q[action] = reward
    else    
        n = e.N[action]
        e.Q[action] = e.Q[action] * (n-1)/n + (reward / n)
    end        
    e.W .= e.Q
    return nothing
end
