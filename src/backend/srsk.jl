function nonzero_tensor_entries(a::Array{T,4}) where {T}
    vals = T[]
    idxs = Matrix{Int}(undef, 4, 0)
    for i4 in axes(a, 4), i3 in axes(a, 3), i2 in axes(a, 2), i1 in axes(a, 1)
        v = a[i1, i2, i3, i4]
        if v != 0
            push!(vals, v)
            idxs = hcat(idxs, [i1, i2, i3, i4])
        end
    end
    return vals, idxs
end

function unique_last_pairs(indices::Matrix{Int})
    pairs = unique(eachcol(indices[3:4, :]))
    out = Matrix{Int}(undef, 2, length(pairs))
    for (i, p) in enumerate(pairs)
        out[:, i] = collect(p)
    end
    return out
end

function calc_srsk(fiber::Fiber{T}, sim::Simulation{T}, num_spatial_modes::Integer) where {T}
    sra_vals, sra_idx = nonzero_tensor_entries(fiber.sr)
    if sim.scalar
        sk_factor = sim.ellipticity == 0 ? one(T) :
                    sim.ellipticity == 1 ? T(2) / T(3) :
                    error("Scalar mode supports linear or circular polarization only.")
        sk_vals = sk_factor .* sra_vals
        sk_idx = copy(sra_idx)
        srb_vals = T[]
        srb_idx = Matrix{Int}(undef, 4, 0)
    else
        sk_vals, sk_idx, sra_vals, sra_idx, srb_vals, srb_idx = polarized_srsk(sra_vals, sra_idx, T(sim.ellipticity))
    end
    if sim.include_Raman
        sk_vals = (one(T) - fiber.fr) .* sk_vals
    end
    return SRSKInfo(; sk=collect(T, sk_vals), sk_indices=sk_idx,
                    sra=collect(T, sra_vals), sra_indices=sra_idx,
                    sra_indices34=unique_last_pairs(sra_idx),
                    srb=collect(T, srb_vals), srb_indices=srb_idx,
                    srb_indices34=size(srb_idx, 2) == 0 ? Matrix{Int}(undef, 2, 0) : unique_last_pairs(srb_idx))
end

function polarized_srsk(sra_vals::Vector{T}, sra_idx::Matrix{Int}, ellipticity::T) where {T}
    odd(x) = 2x - 1
    even(x) = 2x
    sra_out_vals = T[]
    sra_out_idx = Matrix{Int}(undef, 4, 0)
    sk_out_vals = T[]
    sk_out_idx = Matrix{Int}(undef, 4, 0)
    srb_out_vals = T[]
    srb_out_idx = Matrix{Int}(undef, 4, 0)
    for col in axes(sra_idx, 2)
        i1, i2, i3, i4 = sra_idx[:, col]
        v = sra_vals[col]
        sra_terms = ([odd(i1), odd(i2), odd(i3), odd(i4)],
                     [odd(i1), odd(i2), even(i3), even(i4)],
                     [even(i1), even(i2), odd(i3), odd(i4)],
                     [even(i1), even(i2), even(i3), even(i4)])
        for term in sra_terms
            push!(sra_out_vals, v)
            sra_out_idx = hcat(sra_out_idx, term)
        end
        if ellipticity == 0
            sk_terms = (([odd(i1), odd(i2), odd(i3), odd(i4)], one(T)),
                        ([even(i1), even(i2), even(i3), even(i4)], one(T)),
                        ([odd(i1), odd(i2), even(i3), even(i4)], T(2) / T(3)),
                        ([even(i1), even(i2), odd(i3), odd(i4)], T(2) / T(3)),
                        ([odd(i1), even(i2), even(i3), odd(i4)], T(1) / T(3)),
                        ([even(i1), odd(i2), odd(i3), even(i4)], T(1) / T(3)))
            srb_terms = (([odd(i1), odd(i2), odd(i3), odd(i4)], one(T)),
                         ([even(i1), even(i2), even(i3), even(i4)], one(T)),
                         ([odd(i1), even(i2), odd(i3), even(i4)], T(1) / T(2)),
                         ([even(i1), odd(i2), odd(i3), even(i4)], T(1) / T(2)),
                         ([odd(i1), even(i2), even(i3), odd(i4)], T(1) / T(2)),
                         ([even(i1), odd(i2), even(i3), odd(i4)], T(1) / T(2)))
        else
            error("Circular/elliptical polarized SRSK is not implemented yet.")
        end
        for (term, coeff) in sk_terms
            push!(sk_out_vals, coeff * v)
            sk_out_idx = hcat(sk_out_idx, term)
        end
        for (term, coeff) in srb_terms
            push!(srb_out_vals, coeff * v)
            srb_out_idx = hcat(srb_out_idx, term)
        end
    end
    sk_vals, sk_idx = combine_duplicate_terms(sk_out_vals, sk_out_idx)
    sra_vals2, sra_idx2 = combine_duplicate_terms(sra_out_vals, sra_out_idx)
    srb_vals2, srb_idx2 = combine_duplicate_terms(srb_out_vals, srb_out_idx)
    return sk_vals, sk_idx, sra_vals2, sra_idx2, srb_vals2, srb_idx2
end

function combine_duplicate_terms(vals::Vector{T}, idx::Matrix{Int}) where {T}
    acc = Dict{NTuple{4,Int},T}()
    for j in axes(idx, 2)
        key = Tuple(idx[:, j])
        acc[key] = get(acc, key, zero(T)) + vals[j]
    end
    keys_sorted = sort(collect(keys(acc)))
    out_idx = Matrix{Int}(undef, 4, length(keys_sorted))
    out_vals = Vector{T}(undef, length(keys_sorted))
    for (j, key) in enumerate(keys_sorted)
        out_idx[:, j] = collect(key)
        out_vals[j] = acc[key]
    end
    return out_vals, out_idx
end
