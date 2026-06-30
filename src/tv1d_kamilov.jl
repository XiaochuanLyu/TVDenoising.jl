using LinearAlgebra

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

function shrink_l2_1d!(d::AbstractVector, τ::Real)
    τ >= 0 || throw(ArgumentError("τ must be nonnegative."))
    @inbounds for i in eachindex(d)
        nrm = abs(d[i])
        if nrm <= τ || nrm == 0
            d[i] = zero(eltype(d))
        else
            d[i] *= (nrm - τ) / nrm
        end
    end
    return d
end

function tv1d_kamilov_denoise!(x::AbstractVector, y::AbstractVector, λ::Real,
                               a::AbstractVector, d::AbstractVector)
    length(x) == length(y) || throw(ArgumentError("length(x) must match length(y)."))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))
    W_kamilov_1d!(a, d, y)
    shrink_l2_1d!(d, 2λ)
    WT_kamilov_1d!(x, a, d)
    return x
end

function tv1d_kamilov_denoise!(x::AbstractVector{Tx}, y::AbstractVector{Ty}, λ::Real) where {Tx, Ty}
    T = promote_type(Tx, Ty, typeof(λ), Float64)
    a = similar(y, T)
    d = similar(y, T)
    return tv1d_kamilov_denoise!(x, y, λ, a, d)
end

function tv1d_kamilov_denoise(y::AbstractVector, λ::Real)
    x = similar(y, promote_type(eltype(y), typeof(λ), Float64))
    return tv1d_kamilov_denoise!(x, y, λ)
end