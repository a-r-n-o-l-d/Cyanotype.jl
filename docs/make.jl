using Cyanotype
using Documenter

DocMeta.setdocmeta!(Cyanotype, :DocTestSetup, :(using Cyanotype); recursive=true)

makedocs(;
    modules=[Cyanotype],
    authors="Arnold",
    repo="https://github.com/a-r-n-o-l-d/Cyanotype.jl/blob/{commit}{path}#{line}",
    sitename="Cyanotype.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://a-r-n-o-l-d.github.io/Cyanotype.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/a-r-n-o-l-d/Cyanotype.jl",
    devbranch="main",
)
