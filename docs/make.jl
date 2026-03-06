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
    ],
)

deploydocs(;
    repo="github.com/tatsujit/MultiBandits.jl",
    devbranch="main",
)
