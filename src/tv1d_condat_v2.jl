using LinearAlgebra

"""
    tv1d_condat_v2!(x, y, λ)

Compute the exact 1D TV denoising proximal operator with Condat's later v2 algorithm.

Important:
- This function reads `y` but does not modify it.
- This function overwrites `x`.
- This function solves `min_x 0.5 * ||x - y||_2^2 + λ * sum(abs.(diff(x)))`.
- This is Condat's later v2 algorithm, using PAVA-like lower/upper stacks.
- The algorithm allocates two index work arrays internally.
- `λ` must be nonnegative.

Arguments:
- `x`: output vector for the denoised signal
- `y`: noisy input signal vector
- `λ`: nonnegative TV regularization strength

Returns:
- `x`: exact 1D TV-denoised signal
"""
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

    # Stack start indices for lower and upper approximations. These arrays
    # store segment boundaries; j_low and j_up are the current stack tops.
    indstart_low = Vector{Int}(undef, N)
    indstart_up = Vector{Int}(undef, N)
    j_low = 1
    j_up = 1

    # jseg/indjseg mark the earliest unresolved segment and sample position
    # that have not yet been written into the output.
    jseg = 1
    indjseg = 1

    # First and current levels for the active lower/upper approximations.
    # The "current" values are updated by running-mean formulas as samples are
    # scanned, which avoids recomputing sums over long segments.
    output_low_first = y[1] - λ
    output_low_curr = output_low_first
    output_up_first = y[1] + λ
    output_up_curr = output_up_first
    twoλ = 2 * λ
    indstart_low[1] = 1
    indstart_up[1] = 1

    # Process all interior samples. The final sample is handled separately so
    # the unresolved tail can be flushed into x exactly once.
    @inbounds for i in 2:(N - 1)
        # If y[i] is inside the feasible upper tube, update the upper running
        # mean; otherwise start a new upper segment.
        if y[i] >= output_low_curr
            if y[i] <= output_up_curr
                output_up_curr += (y[i] - output_up_curr) / (i - indstart_up[j_up] + 1)
                x[indjseg] = output_up_first
                # Merge adjacent upper segments when monotonicity is violated.
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
                    # If lower and upper approximations cross, commit resolved
                    # lower segments before continuing.
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

            # Update the lower approximation for this same sample.
            output_low_curr += (y[i] - output_low_curr) / (i - indstart_low[j_low] + 1)
            x[indjseg] = output_low_first
            # Merge adjacent lower segments when monotonicity is violated.
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
                # If lower and upper approximations cross, commit resolved
                # upper segments before continuing.
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
            # y[i] falls below the current lower approximation. Start a new
            # lower segment, then update/merge the upper approximation.
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

    # Finalization step: include the last sample and write every remaining
    # unresolved segment into x.
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
