# include("utils/sampling.jl")
# include("utils/sleeping_arms.jl")

"""
    @ic0
デバッグ用マクロ。現在位置（ファイル名と行番号）を表示する。
"""
macro ic0()
    :(println("ic| ", $(string(__source__.file)), ":", $(__source__.line)))
end