using Documenter
using CTCLoss

makedocs(;
    modules = [CTCLoss],
    authors = "Mateusz Kaduk <mateusz.kaduk@gmail.com>",
    repo = "https://github.com/mashu/CTCLoss.jl/blob/{commit}{path}{line}",
    sitename = "CTCLoss.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://mashu.github.io/CTCLoss.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(;
    repo = "github.com/mashu/CTCLoss.jl.git",
    devbranch = "main",
    push_preview = true,
)
