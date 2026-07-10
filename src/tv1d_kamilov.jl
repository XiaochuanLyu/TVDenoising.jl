using LinearAlgebra

"""
    W_kamilov_1d!(a, d, x)

Compute the 1D Kamilov analysis transform in place.

Important:
- This function uses periodic boundary conditions.
- This function overwrites `a` and `d`.
- This function does not allocate coefficient arrays.
- Without shrinkage, `WT_kamilov_1d!(x̂, a, d)` should reconstruct `x` up to floating-point error.

Arguments:
- `a`: output vector for local average coefficients
- `d`: output vector for local detail coefficients
- `x`: input signal vector

Returns:
- `(a, d)`: average and detail coefficient vectors
"""
function W_kamilov_1d!(a::AbstractVector, d::AbstractVector, x::AbstractVector)
    N = length(x)
    length(a) == N || throw(ArgumentError("length(a) must match length(x)."))
    length(d) == N || throw(ArgumentError("length(d) must match length(x)."))

    c = 1 / 2
    @inbounds for i in eachindex(x)
        iprev = ifelse(i == firstindex(x), lastindex(x), i - 1)
        a[i] = c * (x[i] + x[iprev])
        d[i] = c * (x[iprev] - x[i])
    end
    return a, d
end

"""
    WT_kamilov_1d!(x, a, d)

Apply the 1D Kamilov synthesis transform in place.

Important:
- This function overwrites `x`.
- This function assumes `a` and `d` came from the same periodic boundary convention as `W_kamilov_1d!`.
- This is the adjoint/reconstruction step used after detail shrinkage.

Arguments:
- `x`: output vector for reconstructed signal
- `a`: local average coefficient vector
- `d`: local detail coefficient vector

Returns:
- `x`: reconstructed signal vector
"""
function WT_kamilov_1d!(x::AbstractVector, a::AbstractVector, d::AbstractVector)
    N = length(x)
    length(a) == N || throw(ArgumentError("length(a) must match length(x)."))
    length(d) == N || throw(ArgumentError("length(d) must match length(x)."))

    c = 1 / 2
    @inbounds for i in eachindex(x)
        inext = ifelse(i == lastindex(x), firstindex(x), i + 1)
        x[i] = c * (a[i] - d[i]) + c * (a[inext] + d[inext])
    end
    return x
end

"""
    shrink_l2_1d!(d, τ)

Apply scalar soft-thresholding to 1D detail coefficients in place.

Important:
- This function overwrites `d`.
- For 1D, l2 shrinkage reduces to scalar soft-thresholding.
- `τ` must be nonnegative.

Arguments:
- `d`: detail coefficient vector to shrink
- `τ`: nonnegative shrinkage threshold

Returns:
- `d`: thresholded detail coefficient vector
"""
function shrink_l2_1d!(d::AbstractVector, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be nonnegative."))
    @. d = max(0, (abs(d) - τ)) * sign(d)
    return d
end

"""
    shrink_weighted_l1_1d!(d, τ)

Apply weighted scalar soft-thresholding to 1D detail coefficients in place.

Important:
- This function overwrites `d`.
- The first coefficient is the periodic wrap-around difference and is not thresholded.
- All non-wrap detail coefficients are thresholded by `τ`.
- This matches a weighted 1-norm with weights `[0, 1, 1, ..., 1]`.

Arguments:
- `d`: detail coefficient vector to shrink
- `τ`: nonnegative shrinkage threshold for non-wrap differences

Returns:
- `d`: thresholded detail coefficient vector
"""
function shrink_weighted_l1_1d!(d::AbstractVector, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be nonnegative."))

    @inbounds for i in eachindex(d)
        if i == firstindex(d)
            continue
        end
        d[i] = max(0, abs(d[i]) - τ) * sign(d[i])
    end
    return d
end

"""
    tv1d_kamilov_fppa!(x, y, λ, a, d, z, xprev; γ=0.01, maxiter=10000, tol=1e-8)

Run the weighted 1D Kamilov fast parallel proximal iteration.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `a`, `d`, `z`, and `xprev`.
- This uses the weighted 1-norm regularizer with wrap-around weight 0.
- Smaller `γ` usually gives a closer match to exact Condat TV, but needs more iterations.
- This is an iterative approximation to the Condat solution, not Condat's direct scan algorithm.

Arguments:
- `x`: output vector for the current denoised signal iterate
- `y`: noisy input signal vector
- `λ`: nonnegative TV regularization strength
- `a`: workspace vector for average coefficients
- `d`: workspace vector for detail coefficients
- `z`: workspace vector for the gradient step input
- `xprev`: workspace vector for the previous iterate
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FPPA iterations

Returns:
- `x`: final denoised signal iterate
"""
function tv1d_kamilov_fppa!(x::AbstractVector, y::AbstractVector, λ::Real,
                            a::AbstractVector, d::AbstractVector,
                            z::AbstractVector, xprev::AbstractVector;
                            γ::Real=0.01, maxiter::Integer=10000)
    length(x) == length(y) || throw(ArgumentError("length(x) must match length(y)."))
    length(a) == length(y) || throw(ArgumentError("length(a) must match length(y)."))
    length(d) == length(y) || throw(ArgumentError("length(d) must match length(y)."))
    length(z) == length(y) || throw(ArgumentError("length(z) must match length(y)."))
    length(xprev) == length(y) || throw(ArgumentError("length(xprev) must match length(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))
    γ > 0 || throw(ArgumentError("γ must be positive."))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive."))

    copyto!(x, y)
    threshold = 2 * γ * λ

    for _ in 1:maxiter
        copyto!(xprev, x)
        @inbounds for i in eachindex(y)
            z[i] = xprev[i] - γ * (xprev[i] - y[i])
        end

        W_kamilov_1d!(a, d, z)
        shrink_weighted_l1_1d!(d, threshold)
        WT_kamilov_1d!(x, a, d)

    end
    return x
end

"""
    tv1d_kamilov_fppa!(x, y, λ; γ=0.01, maxiter=10000)

Run the weighted 1D Kamilov FPPA with internally allocated workspace.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function allocates temporary workspace vectors.
- Use the full workspace form for benchmarks.

Arguments:
- `x`: output vector for the final denoised signal iterate
- `y`: noisy input signal vector
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FPPA iterations

Returns:
- `x`: final denoised signal iterate
"""
function tv1d_kamilov_fppa!(x::AbstractVector{Tx}, y::AbstractVector{Ty}, λ::Real;
                            γ::Real=0.01, maxiter::Integer=10000) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), typeof(γ), Float64)
    a = similar(y, T)
    d = similar(y, T)
    z = similar(y, T)
    xprev = similar(y, T)
    return tv1d_kamilov_fppa!(x, y, λ, a, d, z, xprev; γ=γ, maxiter=maxiter)
end

"""
    tv1d_kamilov_fppa(y, λ; γ=0.01, maxiter=10000)

Denoise a 1D signal with weighted Kamilov FPPA and return a new vector.

Important:
- This function reads `y` but does not modify it.
- This function allocates the output vector and all workspace vectors.
- Smaller `γ` generally improves agreement with exact 1D TV at higher iteration cost.

Arguments:
- `y`: noisy input signal vector
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FPPA iterations

Returns:
- `x`: final denoised signal iterate
"""
function tv1d_kamilov_fppa(y::AbstractVector, λ::Real;
                           γ::Real=0.01, maxiter::Integer=10000)
    x = similar(y, promote_type(eltype(y), typeof(λ), typeof(γ), Float64))
    return tv1d_kamilov_fppa!(x, y, λ; γ=γ, maxiter=maxiter)
end

"""
    tv1d_kamilov_denoise!(x, y, λ, a, d)

Denoise a 1D signal with one Kamilov transform-shrink-synthesis pass.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `a`, and `d`.
- This uses a weighted 1-norm on detail coefficients.
- The wrap-around detail coefficient has weight 0, matching non-circular TV.
- The threshold is `2λ` because `d = (x_prev - x) / 2`.
- This workspace form is the preferred version for benchmarks and parameter sweeps.

Arguments:
- `x`: output vector for denoised signal
- `y`: noisy input signal vector
- `λ`: nonnegative denoising strength
- `a`: workspace vector for average coefficients
- `d`: workspace vector for detail coefficients

Returns:
- `x`: denoised signal vector
"""
function tv1d_kamilov_denoise!(x::AbstractVector, y::AbstractVector, λ::Real,
                               a::AbstractVector, d::AbstractVector)
    length(x) == length(y) || throw(ArgumentError("length(x) must match length(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))
    W_kamilov_1d!(a, d, y)
    shrink_weighted_l1_1d!(d, 2λ)
    WT_kamilov_1d!(x, a, d)
    return x
end

"""
    tv1d_kamilov_denoise!(x, y, λ)

Denoise a 1D signal with internally allocated Kamilov workspace.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function allocates temporary `a` and `d` work vectors.
- Use `tv1d_kamilov_denoise!(x, y, λ, a, d)` to avoid repeated allocations.

Arguments:
- `x`: output vector for denoised signal
- `y`: noisy input signal vector
- `λ`: nonnegative denoising strength

Returns:
- `x`: denoised signal vector
"""
function tv1d_kamilov_denoise!(x::AbstractVector{Tx}, y::AbstractVector{Ty}, λ::Real) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), Float64)
    a = similar(y, T)
    d = similar(y, T)
    return tv1d_kamilov_denoise!(x, y, λ, a, d)
end

"""
    tv1d_kamilov_denoise(y, λ)

Denoise a 1D signal and return a newly allocated output vector.

Important:
- This function reads `y` but does not modify it.
- This function allocates the output vector and all workspace vectors.
- Use an in-place method for repeated calls or benchmarks.

Arguments:
- `y`: noisy input signal vector
- `λ`: nonnegative denoising strength

Returns:
- `x`: denoised signal vector
"""
function tv1d_kamilov_denoise(y::AbstractVector, λ::Real)
    x = similar(y, promote_type(eltype(y), typeof(λ), Float64))
    return tv1d_kamilov_denoise!(x, y, λ)
end
