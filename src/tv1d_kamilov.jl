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
    tv1d_kamilov_denoise!(x, y, λ, a, d)

Denoise a 1D signal with one Kamilov transform-shrink-synthesis pass.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `a`, and `d`.
- This is not the same exact 1D TV prox as Condat or ProxTV.
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
    shrink_l2_1d!(d, sqrt(2) * λ)
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
