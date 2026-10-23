using Documenter
using CutPotentials

DocMeta.setdocmeta!(CutPotentials, :DocTestSetup, :(using CutPotentials); recursive = true)

makedocs(;
    modules = [CutPotentials],
    repo = Remotes.GitHub("xuanzhaogao", "CutPotentials.jl"),
    authors = "Xuanzhao Gao <xgao@flatironinstitute.org>",
    sitename = "CutPotentials.jl",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true",
                             canonical = "https://xuanzhaogao.github.io/CutPotentials.jl", edit_link = "main"),
    pages = ["Home" => "index.md", "Method" => "method.md", "NUFFT derivatives" => "nufft.md", "Usage" => "usage.md", "Examples" => "examples.md",
             "Accuracy and cost" => "accuracy.md",
             "API" => "api.md"],
)

deploydocs(; repo = "github.com/xuanzhaogao/CutPotentials.jl.git", devbranch = "main")
