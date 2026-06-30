using LinearAlgebra

function tv1d_condat_v2!(x::AbstractVector{Tx}, y::AbstractVector{Ty}, λ::U) where {Tx <: Number, Ty <: Number, U <: Real}
    T = promote_type(Tx, Ty, U, Float64)
    N = length(y)
    N == length(x) || throw(ArgumentError("length"))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))

    if λ == 0
        copyto!(x, y)
        return x
    end
    λ = T(λ)

    if N == 1
        x[1] = y[1]
        return x
    end

    indstart_low = Vector{Int}(undef, N)
    indstart_up = Vector{Int}(undef, N)
    j_low = 1
    j_up = 1
    jseg = 1
    indjseg = 1
    output_low_first = y[1] - λ
    output_low_curr = output_low_first
    output_up_first = y[1] + λ
    output_up_curr = output_up_first
    twoλ = 2 * λ
    indstart_low[1] = 1
    indstart_up[1] = 1

    @inbounds for i in 2:(N - 1)
        if y[i] >= output_low_curr
            if y[i] <= output_up_curr
                output_up_curr += (y[i] - output_up_curr) / (i - indstart_up[j_up] + 1)
                x[indjseg] = output_up_first
                while j_up > jseg
                    ind = indstart_up[j_up - 1]
                    if output_up_curr > x[ind]
                        break
                    end
                    output_up_curr += (x[ind] - output_up_curr) *
                                      ((indstart_up[j_up] - ind) / (i - ind + 1))
                    j_up -= 1
                end
                if j_up == jseg
                    while output_up_curr <= output_low_first && jseg < j_low
                        jseg += 1
                        indjseg2 = indstart_low[jseg]
                        output_up_curr += (output_up_curr - output_low_first) *
                                          ((indjseg2 - indjseg) / (i - indjseg2 + 1))
                        while indjseg < indjseg2
                            x[indjseg] = output_low_first
                            indjseg += 1
                        end
                        output_low_first = x[indjseg]
                    end
                    output_up_first = output_up_curr
                    j_up = jseg
                    indstart_up[j_up] = indjseg
                else
                    x[indstart_up[j_up]] = output_up_curr
                end
            else
                j_up += 1
                indstart_up[j_up] = i
                x[i] = y[i]
                output_up_curr = y[i]
            end
            output_low_curr += (y[i] - output_low_curr) / (i - indstart_low[j_low] + 1)
            x[indjseg] = output_low_first
            while j_low > jseg
                ind = indstart_low[j_low - 1]
                if output_low_curr < x[ind]
                    break
                end
                output_low_curr += (x[ind] - output_low_curr) *
                                   ((indstart_low[j_low] - ind) / (i - ind + 1))
                j_low -= 1
            end
            if j_low == jseg
                while output_low_curr >= output_up_first && jseg < j_up
                    jseg += 1
                    indjseg2 = indstart_up[jseg]
                    output_low_curr += (output_low_curr - output_up_first) *
                                       ((indjseg2 - indjseg) / (i - indjseg2 + 1))
                    while indjseg < indjseg2
                        x[indjseg] = output_up_first
                        indjseg += 1
                    end
                    output_up_first = x[indjseg]
                end
                j_low = jseg
                indstart_low[j_low] = indjseg
                if indjseg == i
                    output_low_first = output_up_first - twoλ
                else
                    output_low_first = output_low_curr
                end
            else
                x[indstart_low[j_low]] = output_low_curr
            end
        else
            j_low += 1
            indstart_low[j_low] = i
            x[i] = y[i]
            output_low_curr = y[i]
            output_up_curr += (output_low_curr - output_up_curr) / (i - indstart_up[j_up] + 1)
            x[indjseg] = output_up_first
            while j_up > jseg
                ind = indstart_up[j_up - 1]
                if output_up_curr > x[ind]
                    break
                end
                output_up_curr += (x[ind] - output_up_curr) *
                                  ((indstart_up[j_up] - ind) / (i - ind + 1))
                j_up -= 1
            end
            if j_up == jseg
                while output_up_curr <= output_low_first && jseg < j_low
                    jseg += 1
                    indjseg2 = indstart_low[jseg]
                    output_up_curr += (output_up_curr - output_low_first) *
                                      ((indjseg2 - indjseg) / (i - indjseg2 + 1))
                    while indjseg < indjseg2
                        x[indjseg] = output_low_first
                        indjseg += 1
                    end
                    output_low_first = x[indjseg]
                end
                j_up = jseg
                indstart_up[j_up] = indjseg
                if indjseg == i
                    output_up_first = output_low_first + twoλ
                else
                    output_up_first = output_up_curr
                end
            else
                x[indstart_up[j_up]] = output_up_curr
            end
        end
    end

    @inbounds begin
        i = N
        if y[i] + λ <= output_low_curr
            while jseg < j_low
                jseg += 1
                indjseg2 = indstart_low[jseg]
                while indjseg < indjseg2
                    x[indjseg] = output_low_first
                    indjseg += 1
                end
                output_low_first = x[indjseg]
            end
            while indjseg < i
                x[indjseg] = output_low_first
                indjseg += 1
            end
            x[indjseg] = y[i] + λ
        elseif y[i] - λ >= output_up_curr
            while jseg < j_up
                jseg += 1
                indjseg2 = indstart_up[jseg]
                while indjseg < indjseg2
                    x[indjseg] = output_up_first
                    indjseg += 1
                end
                output_up_first = x[indjseg]
            end
            while indjseg < i
                x[indjseg] = output_up_first
                indjseg += 1
            end
            x[indjseg] = y[i] - λ
        else
            output_low_curr += (y[i] + λ - output_low_curr) / (i - indstart_low[j_low] + 1)
            x[indjseg] = output_low_first
            while j_low > jseg
                ind = indstart_low[j_low - 1]
                if output_low_curr < x[ind]
                    break
                end
                output_low_curr += (x[ind] - output_low_curr) *
                                   ((indstart_low[j_low] - ind) / (i - ind + 1))
                j_low -= 1
            end
            if j_low == jseg
                if output_up_first >= output_low_curr
                    while indjseg <= i
                        x[indjseg] = output_low_curr
                        indjseg += 1
                    end
                else
                    output_up_curr += (y[i] - λ - output_up_curr) / (i - indstart_up[j_up] + 1)
                    x[indjseg] = output_up_first
                    while j_up > jseg
                        ind = indstart_up[j_up - 1]
                        if output_up_curr > x[ind]
                            break
                        end
                        output_up_curr += (x[ind] - output_up_curr) *
                                          ((indstart_up[j_up] - ind) / (i - ind + 1))
                        j_up -= 1
                    end
                    while jseg < j_up
                        jseg += 1
                        indjseg2 = indstart_up[jseg]
                        while indjseg < indjseg2
                            x[indjseg] = output_up_first
                            indjseg += 1
                        end
                        output_up_first = x[indjseg]
                    end
                    indjseg = indstart_up[j_up]
                    while indjseg <= i
                        x[indjseg] = output_up_curr
                        indjseg += 1
                    end
                end
            else
                while jseg < j_low
                    jseg += 1
                    indjseg2 = indstart_low[jseg]
                    while indjseg < indjseg2
                        x[indjseg] = output_low_first
                        indjseg += 1
                    end
                    output_low_first = x[indjseg]
                end
                indjseg = indstart_low[j_low]
                while indjseg <= i
                    x[indjseg] = output_low_curr
                    indjseg += 1
                end
            end
        end
    end
    return x
end
