# TV Denoising

This documentation is generated from Julia scripts in `docs/lit/examples`.

The examples are intentionally plain Julia plus Markdown comments, so they can be:

- run as normal Julia scripts while debugging;
- converted to documentation pages with Literate.jl;
- built into a static site with Documenter.jl.

## Build locally

From the repository root:

```julia
import Pkg
Pkg.activate("docs")
Pkg.instantiate()
include("docs/make.jl")
```

The generated site will be written to `docs/build`.

The 1D example compares Condat, ProxTV.jl, TVDenoise.jl, and the current Kamilov 1D method. TVDenoise.jl is used as a local path dependency in the docs environment during local development.

## GitHub Pages

`docs/make.jl` includes a `deploydocs` call for:

```julia
deploydocs(
    repo="github.com/XiaochuanLyu/TVDenoising.jl.git",
    devbranch="main",
)
```

The GitHub Actions workflow in `.github/workflows/Documentation.yml` does this by:

1. checking out this repository;
2. checking out `nikopj/TVDenoise.jl` into `TVDenoise.jl`;
3. running `Pkg.develop(path="TVDenoise.jl")` in the docs environment;
4. building the Literate examples and deploying the Documenter site.

On pushes to `main`, Documenter deploys the generated HTML to the `gh-pages` branch. On pull requests, it builds the docs but does not publish them.

## Examples

- [1D TV denoising](examples/1d_examples.md)
- [2D Kamilov denoising](examples/2d_examples.md)
