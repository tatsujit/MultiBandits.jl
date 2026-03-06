using MultiBandits
using Documenter

DocMeta.setdocmeta!(MultiBandits, :DocTestSetup, :(using MultiBandits); recursive=true)

makedocs(;
    modules=[MultiBandits],
    authors="Tatsuji Takahashi <tatsujit.takahashi@gmail.com> and contributors",
    sitename="MultiBandits.jl",
    format=Documenter.HTML(;
        canonical="https://tatsujit.github.io/MultiBandits.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API リファレンス" => [
            "推定器 (Estimators)" => "api/estimators.md",
            "方策 (Policies)" => "api/policies.md",
            "環境 (Environments)" => "api/environments.md",
            "システム (System)" => "api/system.md",
            "評価 (Evaluation)" => "api/evaluation.md",
            "内部ユーティリティ" => "api/internals.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/tatsujit/MultiBandits.jl",
    devbranch="main",
)
