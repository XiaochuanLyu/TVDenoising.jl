using LinearAlgebra

# include("1D_denoising.jl")

# function tv2d_rowcol(y::AbstractMatrix, λ::Real; ncycles::Int = 5)
#     T = promote_type(eltype(y), typeof(λ), Float64)
#     x = T.(y)

#     for cycle in 1:ncycles
#         # denoise rows
#         for i in axes(x, 1)
#             x[i, :] .= tv1d_condat_v1!(vec(x[i, :]), λ)
#         end

#         # denoise columns
#         for j in axes(x, 2)
#             x[:, j] .= tv1d_condat_v1!(vec(x[:, j]), λ)
#         end
#     end
#     return x
# end



function circshift_down(x)
    return circshift(x, (1, 0))
end

function circshift_up(x)
    return circshift(x, (-1, 0))
end

function circshift_right(x)
    return circshift(x, (0, 1))
end

function circshift_left(x)
    return circshift(x, (0, -1))
end

function W_kamilov_2d(x::AbstractMatrix)
    T = promote_type(eltype(x), Float64)
    xx = T.(x)

    D = 2
    c = 1 / (2 * sqrt(D))

    # vertical pair: current pixel and pixel below
    x_down = circshift_down(xx)
    av = c * (xx + x_down)
    dv = c * (x_down - xx)

    # horizontal pair: current pixel and pixel right
    x_right = circshift_right(xx)
    ah = c * (xx + x_right)
    dh = c * (x_right - xx)

    return av, dv, ah, dh
end

function WT_kamilov_2d(av, dv, ah, dh)
    T = promote_type(eltype(av), eltype(dv), eltype(ah), eltype(dh), Float64)

    D = 2
    c = 1 / (2 * sqrt(D))

    # vertical adjoint
    # av = c * (x + shift_down(x))
    # dv = c * (shift_down(x) - x)
    xv = c * (av .- dv) .+ c * circshift_up(av .+ dv)

    # horizontal adjoint
    # ah = c * (x + shift_right(x))
    # dh = c * (shift_right(x) - x)
    xh = c .* (ah .- dh) .+ c .* circshift_left(ah .+ dh)

    return xv + xh
end



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

function tv2d_kamilov_denoise!(x::AbstractMatrix, y::AbstractMatrix, λ::Real,
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

function tv2d_kamilov_denoise!(x::AbstractMatrix{Tx}, y::AbstractMatrix{Ty}, λ::Real) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), Float64)
    av = similar(y, T)
    dv = similar(y, T)
    ah = similar(y, T)
    dh = similar(y, T)
    return tv2d_kamilov_denoise!(x, y, λ, av, dv, ah, dh)
end

function tv2d_kamilov_denoise(y::AbstractMatrix, λ::Real)
    x = similar(y, promote_type(eltype(y), typeof(λ), Float64))
    return tv2d_kamilov_denoise!(x, y, λ)
end
