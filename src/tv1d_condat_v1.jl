using LinearAlgebra

"""
    tv1d_condat_v1!(x, y, λ)

Compute the exact 1D TV denoising proximal operator with Condat's original direct algorithm.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function solves `min_x 0.5 * ||x - y||_2^2 + λ * sum(abs.(diff(x)))`.
- This is the original v1 direct scan algorithm from Condat's 1D TV paper.
- `λ` must be nonnegative.

Arguments:
- `x`: output vector for the denoised signal
- `y`: noisy input signal vector
- `λ`: nonnegative TV regularization strength

Returns:
- `x`: exact 1D TV-denoised signal
"""
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

    # Segment state from Condat's paper algorithm.
    # k_0 is the start of the current unresolved segment; k_neg/k_pos are the
    # last positions where the lower/upper bounds were tightened.
    k = 1
    k_0 = 1
    k_neg = 1
    k_pos = 1

    # v_min/v_max are lower and upper feasible segment levels. u_min/u_max are
    # accumulated residuals used to decide whether a negative or positive jump
    # must be emitted.
    v_min = y[1] - λ
    v_max = y[1] + λ
    u_min = λ
    u_max = -λ
    while true
        # If the unresolved segment is a single final sample, its value is
        # already determined by the lower candidate plus accumulated residual.
        if k == N
            x[N] = v_min + u_min
            return x
        end
        # Scan forward while the current segment can still absorb new samples.
        while k < N
            # The next point lies below the feasible tube, so a negative jump is
            # necessary and the lower segment value is committed.
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
            # The next point lies above the feasible tube, so a positive jump is
            # necessary and the upper segment value is committed.
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
            # No jump is forced yet; extend the current segment and update both
            # feasibility residuals.
            else
                k += 1
                u_min += y[k] - v_min
                u_max += y[k] - v_max
                # Tighten the lower candidate when its residual reaches λ.
                if u_min >= λ
                    v_min += (u_min - λ) / (k - k_0 + 1)
                    u_min = λ
                    k_neg = k
                end
                # Tighten the upper candidate when its residual reaches -λ.
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
        # Finish the tail: if lower residual is negative, commit the lower
        # segment and continue from the next unresolved sample.
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
        # Symmetric final case for an upper segment.
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
        # Otherwise the entire remaining segment is feasible and can be filled
        # with its averaged level.
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
