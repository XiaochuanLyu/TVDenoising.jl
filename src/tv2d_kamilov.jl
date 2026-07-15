using LinearAlgebra

"""
    circshift_down(x)

Shift an array down by one row with periodic boundary conditions.

Important:
- This function allocates a shifted copy.
- This function is used by the readable allocating Kamilov transform.
- The in-place transform avoids this allocation.

Arguments:
- `x`: input array

Returns:
- shifted array with row index moved by `+1`
"""
function circshift_down(x)
    return circshift(x, (1, 0))
end

"""
    circshift_up(x)

Shift an array up by one row with periodic boundary conditions.

Important:
- This function allocates a shifted copy.
- This function is the inverse row shift of `circshift_down`.

Arguments:
- `x`: input array

Returns:
- shifted array with row index moved by `-1`
"""
function circshift_up(x)
    return circshift(x, (-1, 0))
end

"""
    circshift_right(x)

Shift an array right by one column with periodic boundary conditions.

Important:
- This function allocates a shifted copy.
- This function is used by the readable allocating Kamilov transform.

Arguments:
- `x`: input array

Returns:
- shifted array with column index moved by `+1`
"""
function circshift_right(x)
    return circshift(x, (0, 1))
end

"""
    circshift_left(x)

Shift an array left by one column with periodic boundary conditions.

Important:
- This function allocates a shifted copy.
- This function is the inverse column shift of `circshift_right`.

Arguments:
- `x`: input array

Returns:
- shifted array with column index moved by `-1`
"""
function circshift_left(x)
    return circshift(x, (0, -1))
end

"""
    W_kamilov_2d(x)

Compute the allocating 2D Kamilov analysis transform.

Important:
- This function reads `x` but does not modify it.
- This function allocates output coefficient matrices.
- This function also allocates shifted intermediate arrays.
- Use `W_kamilov_2d!` for benchmarks or repeated calls.

Arguments:
- `x`: input image matrix

Returns:
- `av`: vertical average coefficients
- `dv`: vertical detail coefficients
- `ah`: horizontal average coefficients
- `dh`: horizontal detail coefficients
"""
function W_kamilov_2d(x::AbstractMatrix)
    T = promote_type(eltype(x), Float64)
    xx = T.(x)

    D = 2
    c = 1 / (2 * sqrt(D))

    x_down = circshift_down(xx)
    av = c * (xx + x_down)
    dv = c * (x_down - xx)

    x_right = circshift_right(xx)
    ah = c * (xx + x_right)
    dh = c * (x_right - xx)

    return av, dv, ah, dh
end

"""
    WT_kamilov_2d(av, dv, ah, dh)

Apply the allocating 2D Kamilov synthesis transform.

Important:
- This function reads coefficient matrices but does not modify them.
- This function allocates the reconstructed output matrix.
- With no shrinkage, `WT_kamilov_2d(W_kamilov_2d(x)...)` should reconstruct `x` up to floating-point error.
- Use `WT_kamilov_2d!` to write into an existing output matrix.

Arguments:
- `av`: vertical average coefficients
- `dv`: vertical detail coefficients
- `ah`: horizontal average coefficients
- `dh`: horizontal detail coefficients

Returns:
- `x`: reconstructed image matrix
"""
function WT_kamilov_2d(av, dv, ah, dh)
    T = promote_type(eltype(av), eltype(dv), eltype(ah), eltype(dh), Float64)

    D = 2
    c = 1 / (2 * sqrt(D))

    xv = c * (av .- dv) .+ c * circshift_up(av .+ dv)
    xh = c .* (ah .- dh) .+ c .* circshift_left(ah .+ dh)

    return xv + xh
end

"""
    W_kamilov_2d!(av, dv, ah, dh, x)

Compute the 2D Kamilov analysis transform in place.

Important:
- This function reads `x` but does not modify it.
- This function overwrites `av`, `dv`, `ah`, and `dh`.
- This function uses periodic boundary conditions.
- This function avoids the shifted-array allocations used by `W_kamilov_2d`.

Arguments:
- `av`: output matrix for vertical average coefficients
- `dv`: output matrix for vertical detail coefficients
- `ah`: output matrix for horizontal average coefficients
- `dh`: output matrix for horizontal detail coefficients
- `x`: input image matrix

Returns:
- `(av, dv, ah, dh)`: coefficient matrices
"""
function W_kamilov_2d!(av::AbstractMatrix, dv::AbstractMatrix,
                       ah::AbstractMatrix, dh::AbstractMatrix,
                       x::AbstractMatrix)
    axes(av) == axes(x) || throw(ArgumentError("axes(av) must match axes(x)."))
    axes(dv) == axes(x) || throw(ArgumentError("axes(dv) must match axes(x)."))
    axes(ah) == axes(x) || throw(ArgumentError("axes(ah) must match axes(x)."))
    axes(dh) == axes(x) || throw(ArgumentError("axes(dh) must match axes(x)."))

    D = 2
    c = 1 / (2 * sqrt(D))
    rows = axes(x, 1)
    cols = axes(x, 2)

    @inbounds for j in cols, i in rows
        iprev = ifelse(i == first(rows), last(rows), i - 1)
        jprev = ifelse(j == first(cols), last(cols), j - 1)

        av[i, j] = c * (x[i, j] + x[iprev, j])
        dv[i, j] = c * (x[iprev, j] - x[i, j])
        ah[i, j] = c * (x[i, j] + x[i, jprev])
        dh[i, j] = c * (x[i, jprev] - x[i, j])
    end
    return av, dv, ah, dh
end

"""
    WT_kamilov_2d!(x, av, dv, ah, dh)

Apply the 2D Kamilov synthesis transform in place.

Important:
- This function overwrites `x`.
- This function reads `av`, `dv`, `ah`, and `dh` but does not modify them.
- This is the adjoint/reconstruction step used after detail shrinkage.
- With no shrinkage after `W_kamilov_2d!`, this should reconstruct the original image up to floating-point error.

Arguments:
- `x`: output matrix for reconstructed image
- `av`: vertical average coefficients
- `dv`: vertical detail coefficients
- `ah`: horizontal average coefficients
- `dh`: horizontal detail coefficients

Returns:
- `x`: reconstructed image matrix
"""
function WT_kamilov_2d!(x::AbstractMatrix,
                        av::AbstractMatrix, dv::AbstractMatrix,
                        ah::AbstractMatrix, dh::AbstractMatrix)
    axes(av) == axes(x) || throw(ArgumentError("axes(av) must match axes(x)."))
    axes(dv) == axes(x) || throw(ArgumentError("axes(dv) must match axes(x)."))
    axes(ah) == axes(x) || throw(ArgumentError("axes(ah) must match axes(x)."))
    axes(dh) == axes(x) || throw(ArgumentError("axes(dh) must match axes(x)."))

    D = 2
    c = 1 / (2 * sqrt(D))
    rows = axes(x, 1)
    cols = axes(x, 2)

    @inbounds for j in cols, i in rows
        inext = ifelse(i == last(rows), first(rows), i + 1)
        jnext = ifelse(j == last(cols), first(cols), j + 1)

        x[i, j] = c * (av[i, j] - dv[i, j]) +
                  c * (av[inext, j] + dv[inext, j]) +
                  c * (ah[i, j] - dh[i, j]) +
                  c * (ah[i, jnext] + dh[i, jnext])
    end
    return x
end

"""
    shrink_l2_2d!(dv, dh, τ)

Apply l2 vector shrinkage to 2D detail coefficients in place.

Important:
- This function overwrites `dv` and `dh`.
- Shrinkage is applied independently at each pixel to the vector `(dv[i,j], dh[i,j])`.
- This implements `T(z; τ) = max(||z||₂ - τ, 0) * z / ||z||₂`.
- `τ` must be nonnegative.

Arguments:
- `dv`: vertical detail coefficient matrix
- `dh`: horizontal detail coefficient matrix
- `τ`: nonnegative l2 shrinkage threshold

Returns:
- `(dv, dh)`: thresholded detail coefficient matrices
"""
function shrink_l2_2d!(dv::AbstractMatrix, dh::AbstractMatrix, τ::Real)
    axes(dv) == axes(dh) || throw(ArgumentError("axes(dv) must match axes(dh)."))
    τ >= 0 || throw(ArgumentError("τ must be nonnegative."))

    @inbounds for idx in eachindex(dv, dh)
        nrm = hypot(dv[idx], dh[idx])
        if nrm <= τ || nrm == 0
            dv[idx] = zero(eltype(dv))
            dh[idx] = zero(eltype(dh))
        else
            scale = (nrm - τ) / nrm
            dv[idx] *= scale
            dh[idx] *= scale
        end
    end
    return dv, dh
end

"""
    tv2d_kamilov_onepass!(x, y, λ, av, dv, ah, dh)

Denoise a 2D image with one Kamilov transform-shrink-synthesis pass.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `av`, `dv`, `ah`, and `dh`.
- This is a one-pass transform shrinkage denoiser, not an iterative exact 2D TV solver.
- For 2D, the shrinkage threshold is `2λ√2`, matching `2λ√D` with `D = 2`.
- This workspace form is the preferred version for benchmarks and parameter sweeps.

Arguments:
- `x`: output matrix for denoised image
- `y`: noisy input image matrix
- `λ`: nonnegative denoising strength
- `av`: workspace matrix for vertical average coefficients
- `dv`: workspace matrix for vertical detail coefficients
- `ah`: workspace matrix for horizontal average coefficients
- `dh`: workspace matrix for horizontal detail coefficients

Returns:
- `x`: denoised image matrix
"""
function tv2d_kamilov_onepass!(x::AbstractMatrix, y::AbstractMatrix, λ::Real,
                               av::AbstractMatrix, dv::AbstractMatrix,
                               ah::AbstractMatrix, dh::AbstractMatrix)
    axes(x) == axes(y) || throw(ArgumentError("axes(x) must match axes(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))

    D = 2
    W_kamilov_2d!(av, dv, ah, dh, y)
    shrink_l2_2d!(dv, dh, 2λ * sqrt(D))
    WT_kamilov_2d!(x, av, dv, ah, dh)
    return x
end

"""
    tv2d_kamilov_onepass!(x, y, λ)

Denoise a 2D image with internally allocated Kamilov workspace.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function allocates four coefficient workspace matrices.
- Use `tv2d_kamilov_onepass!(x, y, λ, av, dv, ah, dh)` to avoid repeated allocations.

Arguments:
- `x`: output matrix for denoised image
- `y`: noisy input image matrix
- `λ`: nonnegative denoising strength

Returns:
- `x`: denoised image matrix
"""
function tv2d_kamilov_onepass!(x::AbstractMatrix{Tx}, y::AbstractMatrix{Ty}, λ::Real) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), Float64)
    av = similar(y, T)
    dv = similar(y, T)
    ah = similar(y, T)
    dh = similar(y, T)
    return tv2d_kamilov_onepass!(x, y, λ, av, dv, ah, dh)
end

"""
    tv2d_kamilov_onepass(y, λ)

Denoise a 2D image and return a newly allocated output matrix.

Important:
- This function reads `y` but does not modify it.
- This function allocates the output matrix and all workspace matrices.
- Use an in-place method for repeated calls or benchmarks.

Arguments:
- `y`: noisy input image matrix
- `λ`: nonnegative denoising strength

Returns:
- `x`: denoised image matrix
"""
function tv2d_kamilov_onepass(y::AbstractMatrix, λ::Real)
    x = similar(y, promote_type(eltype(y), typeof(λ), Float64))
    return tv2d_kamilov_onepass!(x, y, λ)
end

"""
    tv2d_kamilov_ista!(x, y, λ, av, dv, ah, dh, z, xprev; γ=0.01, maxiter=1000)

Run weighted 2D Kamilov ISTA.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `av`, `dv`, `ah`, `dh`, `z`, and `xprev`.
- This applies the 2D transform-shrink step inside a proximal-gradient iteration.
- This is an iterative transform-domain TV experiment, not an exact standard 2D TV solver.

Arguments:
- `x`: output matrix for the current denoised iterate
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `av`: workspace matrix for vertical average coefficients
- `dv`: workspace matrix for vertical detail coefficients
- `ah`: workspace matrix for horizontal average coefficients
- `dh`: workspace matrix for horizontal detail coefficients
- `z`: workspace matrix for the gradient step input
- `xprev`: workspace matrix for the previous iterate
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of ISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_ista!(x::AbstractMatrix, y::AbstractMatrix, λ::Real,
                            av::AbstractMatrix, dv::AbstractMatrix,
                            ah::AbstractMatrix, dh::AbstractMatrix,
                            z::AbstractMatrix, xprev::AbstractMatrix;
                            γ::Real=0.01, maxiter::Integer=1000)
    axes(x) == axes(y) || throw(ArgumentError("axes(x) must match axes(y)."))
    axes(z) == axes(y) || throw(ArgumentError("axes(z) must match axes(y)."))
    axes(xprev) == axes(y) || throw(ArgumentError("axes(xprev) must match axes(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))
    γ > 0 || throw(ArgumentError("γ must be positive."))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive."))

    D = 2
    threshold = 2 * γ * λ * sqrt(D)
    copyto!(x, y)

    for _ in 1:maxiter
        copyto!(xprev, x)
        @inbounds for idx in eachindex(y)
            z[idx] = xprev[idx] - γ * (xprev[idx] - y[idx])
        end

        W_kamilov_2d!(av, dv, ah, dh, z)
        shrink_l2_2d!(dv, dh, threshold)
        WT_kamilov_2d!(x, av, dv, ah, dh)
    end
    return x
end

"""
    tv2d_kamilov_ista!(x, y, λ; γ=0.01, maxiter=1000)

Run weighted 2D Kamilov ISTA with internally allocated workspace.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function allocates temporary workspace matrices.
- Use the full workspace form for benchmarks.

Arguments:
- `x`: output matrix for the final denoised image iterate
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of ISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_ista!(x::AbstractMatrix{Tx}, y::AbstractMatrix{Ty}, λ::Real;
                            γ::Real=0.01, maxiter::Integer=1000) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), typeof(γ), Float64)
    av = similar(y, T)
    dv = similar(y, T)
    ah = similar(y, T)
    dh = similar(y, T)
    z = similar(y, T)
    xprev = similar(y, T)
    return tv2d_kamilov_ista!(x, y, λ, av, dv, ah, dh, z, xprev; γ=γ, maxiter=maxiter)
end

"""
    tv2d_kamilov_ista(y, λ; γ=0.01, maxiter=1000)

Denoise a 2D image with weighted Kamilov ISTA and return a new matrix.

Important:
- This function reads `y` but does not modify it.
- This function allocates the output matrix and all workspace matrices.

Arguments:
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of ISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_ista(y::AbstractMatrix, λ::Real;
                           γ::Real=0.01, maxiter::Integer=1000)
    x = similar(y, promote_type(eltype(y), typeof(λ), typeof(γ), Float64))
    return tv2d_kamilov_ista!(x, y, λ; γ=γ, maxiter=maxiter)
end

"""
    tv2d_kamilov_fista!(x, y, λ, av, dv, ah, dh, z, xprev, q, qprev; γ=0.01, maxiter=1000)

Run weighted 2D Kamilov FISTA.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`, `av`, `dv`, `ah`, `dh`, `z`, `xprev`, `q`, and `qprev`.
- This adds standard FISTA momentum to the 2D Kamilov ISTA iteration.
- This is an iterative transform-domain TV experiment, not an exact standard 2D TV solver.

Arguments:
- `x`: output matrix for the current denoised iterate
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `av`: workspace matrix for vertical average coefficients
- `dv`: workspace matrix for vertical detail coefficients
- `ah`: workspace matrix for horizontal average coefficients
- `dh`: workspace matrix for horizontal detail coefficients
- `z`: workspace matrix for the gradient step input
- `xprev`: workspace matrix for the previous denoised iterate
- `q`: workspace matrix for the current momentum point
- `qprev`: workspace matrix for the previous momentum point
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_fista!(x::AbstractMatrix, y::AbstractMatrix, λ::Real,
                             av::AbstractMatrix, dv::AbstractMatrix,
                             ah::AbstractMatrix, dh::AbstractMatrix,
                             z::AbstractMatrix, xprev::AbstractMatrix,
                             q::AbstractMatrix, qprev::AbstractMatrix;
                             γ::Real=0.01, maxiter::Integer=1000)
    axes(x) == axes(y) || throw(ArgumentError("axes(x) must match axes(y)."))
    axes(z) == axes(y) || throw(ArgumentError("axes(z) must match axes(y)."))
    axes(xprev) == axes(y) || throw(ArgumentError("axes(xprev) must match axes(y)."))
    axes(q) == axes(y) || throw(ArgumentError("axes(q) must match axes(y)."))
    axes(qprev) == axes(y) || throw(ArgumentError("axes(qprev) must match axes(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))
    γ > 0 || throw(ArgumentError("γ must be positive."))
    maxiter >= 1 || throw(ArgumentError("maxiter must be positive."))

    D = 2
    threshold = 2 * γ * λ * sqrt(D)
    copyto!(x, y)
    copyto!(q, y)
    t = one(float(γ))

    for _ in 1:maxiter
        copyto!(xprev, x)
        copyto!(qprev, q)

        @inbounds for idx in eachindex(y)
            z[idx] = qprev[idx] - γ * (qprev[idx] - y[idx])
        end

        W_kamilov_2d!(av, dv, ah, dh, z)
        shrink_l2_2d!(dv, dh, threshold)
        WT_kamilov_2d!(x, av, dv, ah, dh)

        tnext = (1 + sqrt(1 + 4 * t^2)) / 2
        momentum = (t - 1) / tnext
        @inbounds for idx in eachindex(y)
            q[idx] = x[idx] + momentum * (x[idx] - xprev[idx])
        end
        t = tnext
    end
    return x
end

"""
    tv2d_kamilov_fista!(x, y, λ; γ=0.01, maxiter=1000)

Run weighted 2D Kamilov FISTA with internally allocated workspace.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function allocates temporary workspace matrices.
- Use the full workspace form for benchmarks.

Arguments:
- `x`: output matrix for the final denoised image iterate
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_fista!(x::AbstractMatrix{Tx}, y::AbstractMatrix{Ty}, λ::Real;
                             γ::Real=0.01, maxiter::Integer=1000) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), typeof(γ), Float64)
    av = similar(y, T)
    dv = similar(y, T)
    ah = similar(y, T)
    dh = similar(y, T)
    z = similar(y, T)
    xprev = similar(y, T)
    q = similar(y, T)
    qprev = similar(y, T)
    return tv2d_kamilov_fista!(x, y, λ, av, dv, ah, dh, z, xprev, q, qprev; γ=γ, maxiter=maxiter)
end

"""
    tv2d_kamilov_fista(y, λ; γ=0.01, maxiter=1000)

Denoise a 2D image with weighted Kamilov FISTA and return a new matrix.

Important:
- This function reads `y` but does not modify it.
- This function allocates the output matrix and all workspace matrices.

Arguments:
- `y`: noisy input image matrix
- `λ`: nonnegative TV regularization strength
- `γ`: positive gradient/proximal step size
- `maxiter`: maximum number of FISTA iterations

Returns:
- `x`: final denoised image iterate
"""
function tv2d_kamilov_fista(y::AbstractMatrix, λ::Real;
                            γ::Real=0.01, maxiter::Integer=1000)
    x = similar(y, promote_type(eltype(y), typeof(λ), typeof(γ), Float64))
    return tv2d_kamilov_fista!(x, y, λ; γ=γ, maxiter=maxiter)
end

