using LinearAlgebra

function tv1d_condat_v1(y::AbstractVector{Ty}, λ::U) where {Ty <: Number, U <: Real}
    T = promote_type(Ty, U, Float64)
    N = length(y)
    if N == 0
        return T[]
    end
    if λ < 0
        throw(ArgumentError("λ must be nonnegative."))
    end
    if λ == 0
        return T.(y)
    end
    λ = T(λ)
    x = Vector{T}(undef, N)
    # Step 1
    k = 1
    k_0 = 1
    k_neg = 1
    k_pos = 1
    v_min = y[1] - λ
    v_max = y[1] + λ
    u_min = λ
    u_max = -λ
    while true
        # Step 2
        if k == N
            x[N] = v_min + u_min
            return x
        end
        # Steps 3 to 7: scan forward while k < N
        while k < N
            # Step 3: negative jump is necessary
            if (y[k + 1] + u_min) < (v_min - λ)
                x[k_0 : k_neg] .= v_min
                # for i in k_0 : k_neg
                #     x[i] = v_min
                # end
                k = k_neg + 1
                k_0 = k
                k_neg = k
                k_pos = k
                v_min = y[k]
                v_max = y[k] + 2λ
                u_min = λ
                u_max = -λ
                break
            # Step 4: positive jump is necessary
            elseif (y[k + 1] + u_max) > (v_max + λ)
                x[k_0 : k_pos] .= v_max
                # for i in k_0 : k_pos
                #     x[i] = v_max
                # end
                k = k_pos + 1
                k_0 = k
                k_neg = k
                k_pos = k
                v_min = y[k] - 2λ
                v_max = y[k]
                u_min = λ
                u_max = -λ
                break
            # Step 5: no jump yet, extend current segment
            else
                k += 1
                u_min += y[k] - v_min
                u_max += y[k] - v_max
                # Step 6: update lower bound
                if u_min >= λ
                    v_min += (u_min - λ) / (k - k_0 + 1)
                    u_min = λ
                    k_neg = k
                end
                # Step 6: update upper bound
                if u_max <= -λ
                    v_max += (u_max + λ) / (k - k_0 + 1)
                    u_max = -λ
                    k_pos = k
                end
            end
        end
        # If we broke out of the inner loop because of Step 3 or 4,
        # continue from Step 2.
        if k < N
            continue
        end
        # Now k == N. Steps 8 to 10 handle the final segment.
        # Step 8: final negative jump
        if u_min < 0
            x[k_0 : k_neg] .= v_min
            # for i in k_0 : k_neg
            #     x[i] = v_min
            # end
            k = k_neg + 1
            k_0 = k
            k_neg = k
            v_min = y[k]
            u_min = λ
            u_max = y[k] + λ - v_max
            continue
        # Step 9: final positive jump
        elseif u_max > 0
            x[k_0 : k_pos] .= v_max
            # for i in k_0 : k_pos
            #     x[i] = v_max
            # end
            k = k_pos + 1
            k_0 = k
            k_pos = k
            v_max = y[k]
            u_max = -λ
            u_min = y[k] - λ - v_min
            continue
        # Step 10: final valid segment
        else
            value = v_min + u_min / (k - k_0 + 1)
            x[k_0 : N] .= value
            # for i in k_0 : N
            #     x[i] = value
            # end
            return x
        end
    end
end

function tv1d_condat_v2(y::AbstractVector{Ty}, λ::U) where {Ty <: Number, U <: Real}
    T = promote_type(Ty, U, Float64)
    N = length(y)
    if N == 0
        return T[]
    end
    if λ < 0
        throw(ArgumentError("λ must be nonnegative."))
    end
    if λ == 0
        return T.(y)
    end
    λ = T(λ)
    x = Vector{T}(undef, N)
    if N == 1
        x[1] = T(y[1])
        return x
    end
    indstart_low = Vector{Int}(undef, N)
    indstart_up = Vector{Int}(undef, N)
    j_low = 1
    j_up = 1
    jseg = 1
    indjseg = 1
    output_low_first = T(y[1]) - λ
    output_low_curr = output_low_first
    output_up_first = T(y[1]) + λ
    output_up_curr = output_up_first
    twoλ = 2 * λ
    indstart_low[1] = 1
    indstart_up[1] = 1

    @inbounds for i in 2:(N - 1)
        yi = T(y[i])
        if yi >= output_low_curr
            if yi <= output_up_curr
                output_up_curr += (yi - output_up_curr) / (i - indstart_up[j_up] + 1)
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
                x[i] = yi
                output_up_curr = yi
            end
            output_low_curr += (yi - output_low_curr) / (i - indstart_low[j_low] + 1)
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
            x[i] = yi
            output_low_curr = yi
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
        yi = T(y[i])
        if yi + λ <= output_low_curr
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
            x[indjseg] = yi + λ
        elseif yi - λ >= output_up_curr
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
            x[indjseg] = yi - λ
        else
            output_low_curr += (yi + λ - output_low_curr) / (i - indstart_low[j_low] + 1)
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
                    output_up_curr += (yi - λ - output_up_curr) / (i - indstart_up[j_up] + 1)
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
