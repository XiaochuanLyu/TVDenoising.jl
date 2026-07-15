# TVDenoising.jl

`TVDenoising.jl` is a research codebase for studying total variation (TV)
denoising algorithms in Julia. The current focus is to compare exact 1D TV
solvers with transform-domain Kamilov-style methods, and to keep the
experiments reproducible through notebooks and Literate/Documenter examples.

## What This Project Does

The repository currently includes:

- exact 1D TV denoising with Condat direct solvers;
- reference comparisons against ProxTV.jl and TVDenoise.jl;
- weighted 1D Kamilov transform-shrink iterations, written as ISTA and FISTA;
- a one-pass 1D Kamilov experiment that does not match exact non-circular TV;
- compact 2D TV denoising experiments using one-pass, ISTA, and FISTA transform-shrink pipelines.

The 1D examples use Condat as the main exact reference. The Kamilov one-pass
method is kept as a diagnostic because it helped identify that the algorithm in
Kamilov's anisotropic TV paper should not be treated as an exact direct solver
for this 1D TV problem. The iterative form is therefore labeled as ISTA, with a
FISTA variant added for accelerated comparison.

## Current Results

The 1D example shows that Condat, ProxTV.jl, and TVDenoise.jl agree up to small
numerical tolerance. It also compares Kamilov one-pass, ISTA, and FISTA against
that exact reference, including timing and memory measurements for the iterative
methods.

The 2D example compares one-pass, ISTA, and FISTA versions of the current
transform-shrink pipeline on a synthetic image. These methods are useful for
checking implementation and visual behavior, but they are not presented as final
exact 2D TV solvers.

## Examples

- [1D TV denoising](examples/1d_examples.md)
- [2D TV denoising](examples/2d_examples.md)
