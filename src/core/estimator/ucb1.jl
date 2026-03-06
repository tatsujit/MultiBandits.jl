# UCB as estimator: empirical reward estimator
# this is to be used with Greedy, SatUCB, SatUCBplus, and so on. 

"""
    UCB1 <: AbstractEstimator

UCB1 アルゴリズムによる推定器。W = Q + c√(ln t / N) をGreedy方策で使用する。

# キーワード引数
- `Q0::Float64=Inf` — 未選択腕を優先探索するための初期値
- `c::Float64=√2` — 探索ボーナスの係数

# 例
```julia
est = UCB1(4)          # 4本腕、デフォルト設定
est = UCB1(4; c=1.0)   # 探索係数を変更
```
"""
struct UCB1 <: AbstractEstimator
    n_arms::Int
    Q::Vector{Float64} # 価値（通常のQ値や平均報酬など）
    B::Vector{Float64} # ボーナス項
    W::Vector{Float64} # 総合的な「行動価値」 W = Q + bonus を greedy に運用
    N::Vector{Int64} # 行動選択の回数（頻度）
    Q0::Float64 # Q値の初期値
    c::Float64 # ボーナス項の係数、デフォルト sqrt(2.0)
    function UCB1(n_arms::Int; Q0::Float64=Inf, c::Float64=sqrt(2.0)) # Q0::Float=0.0
        Q = fill(Q0, n_arms) # 行動価値
        B = fill(0.0, n_arms) # 初期値は 0.0 としておく。どうせ W = Q + B = Inf + B = Inf となるので
        W = fill(Q0, n_arms)
        N = zeros(n_arms) # 行動の選択頻度
        new(n_arms, Q, B, W, N, Q0, c)
    end
end

"""
    TODO: display the values in Q and N too?
"""
function toString(e::UCB1)
    return "UCB(n_arms=$(e.n_arms))"
end

"""
    display() したときの表示
"""
function Base.show(io::IO, ::MIME"text/plain", obj::UCB1)
    toString(obj)
end

"""
    update!(e::UCB1, action::Int, reward::Float64)

Q 値を標本平均で更新し、全腕のボーナス項 B と W = Q + B を再計算する。
"""
function update!(e::UCB1, action::Int, reward::Float64)
    e.N[action] += 1
    t = sum(e.N)
    if e.Q[action] == Inf # assuming that it is the first to choose the action
        e.Q[action] = reward
    else    
        n = e.N[action]
        e.Q[action] = e.Q[action] * (n-1)/n + (reward / n)
    end        
    for a in 1:e.n_arms
        if e.N[a] > 0
            e.B[a] = e.c * sqrt(log(t) / e.N[a])
            e.W[a] = e.Q[a] + e.B[a]
        end
        # N[a]==0 の腕は W[a]=Q0 (初期値, デフォルト Inf) のまま → Greedy で優先的に選ばれる
    end
    return nothing
end
