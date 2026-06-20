using LinearAlgebra

include("1D_denoising.jl")

function tv2d_rowcol(y::AbstractMatrix, λ::Real; ncycles::Int = 5)
    T = promote_type(eltype(y), typeof(λ), Float64)
    x = T.(y)

    for cycle in 1:ncycles
        # denoise rows
        for i in axes(x, 1)
            x[i, :] .= tv1d_condat(vec(x[i, :]), λ)
        end

        # denoise columns
        for j in axes(x, 2)
            x[:, j] .= tv1d_condat(vec(x[:, j]), λ)
        end
    end
    return x
end



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
    c = one(T) / (2 * sqrt(T(D)))

    # vertical pair: current pixel and pixel below
    x_down = circshift_down(xx)
    av = c .* (xx .+ x_down)
    dv = c .* (x_down .- xx)

    # horizontal pair: current pixel and pixel right
    x_right = circshift_right(xx)
    ah = c .* (xx .+ x_right)
    dh = c .* (x_right .- xx)

    return av, dv, ah, dh
end

function WT_kamilov_2d(av, dv, ah, dh)
    T = promote_type(eltype(av), eltype(dv), eltype(ah), eltype(dh), Float64)

    D = 2
    c = one(T) / (2 * sqrt(T(D)))

    # vertical adjoint
    # av = c * (x + shift_down(x))
    # dv = c * (shift_down(x) - x)
    xv = c .* (av .- dv) .+ c .* circshift_up(av .+ dv)

    # horizontal adjoint
    # ah = c * (x + shift_right(x))
    # dh = c * (shift_right(x) - x)
    xh = c .* (ah .- dh) .+ c .* circshift_left(ah .+ dh)

    return xv .+ xh
end


function tv2d_kamilov_denoise(y, λ)

end