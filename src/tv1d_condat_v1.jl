using LinearAlgebra

function tv1d_condat_v1!(x::AbstractVector{Tx}, y::AbstractVector{Ty}, λ::U) where {Tx <: Number, Ty <: Number, U <: Real}
    T = promote_type(Tx, Ty, U, Float32)
    N = length(y)
    N == length(x) || throw(ArgumentError("length"))
    λ >= 0 || throw(ArgumentError("λ must be nonnegative."))

    if λ == 0
        copyto!(x, y)
        return x
    end
    λ = T(λ)

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
