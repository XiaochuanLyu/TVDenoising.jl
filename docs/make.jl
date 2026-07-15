ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")

using Documenter
using Literate

const DOCS = @__DIR__
const REPO = dirname(DOCS)
const LIT = joinpath(DOCS, "lit", "examples")
const GENERATED = joinpath(DOCS, "src", "examples")
const CI = get(ENV, "CI", "false") == "true"

mkpath(GENERATED)
mkpath(joinpath(DOCS, "src", "assets"))

for file in ("1d_examples.jl", "2d_examples.jl")
    Literate.markdown(
        joinpath(LIT, file),
        GENERATED;
        documenter=true,
        execute=true,
        credit=false,
    )
end

makedocs(
    sitename="TV Denoising",
    format=Documenter.HTML(
        prettyurls=CI,
        edit_link=CI ? "main" : nothing,
        inventory_version="dev",
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "1D TV denoising" => "examples/1d_examples.md",
            "2D Kamilov denoising" => "examples/2d_examples.md",
        ],
    ],
)

deploydocs(
    repo="github.com/XiaochuanLyu/TVDenoising.jl.git",
    devbranch="main",
)
