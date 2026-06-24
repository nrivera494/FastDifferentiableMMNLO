struct CPFastRHSCache{T,PFM,PIM,PFR,PIR,WT}
    rank::Int
    at::Matrix{Complex{T}}
    conj_at::Matrix{Complex{T}}
    b2::Matrix{Complex{T}}
    b3::Matrix{Complex{T}}
    b4::Matrix{Complex{T}}
    p::Matrix{Complex{T}}
    k::Matrix{Complex{T}}
    conv::Matrix{Complex{T}}
    nonlinear::Matrix{Complex{T}}
    u2::Matrix{Complex{T}}
    u3::Matrix{Complex{T}}
    u4::Matrix{Complex{T}}
    wu1t::WT
    fft_modes!::PFM
    ifft_modes!::PIM
    fft_rank!::PFR
    ifft_rank!::PIR
end

struct CPRamanLowRankRHSCache{T,C,SM,PFR,PIR}
    base::C
    raman_rank::Int
    effective_rank::Base.RefValue{Int}
    qbasis::Matrix{Complex{T}}
    basis_conv::Matrix{Complex{T}}
    qcoeff::Matrix{Complex{T}}
    raman_weights::Matrix{Complex{T}}
    qnorms::Vector{T}
    selected::Vector{Int}
    selected_mask::SM
    fft_basis!::PFR
    ifft_basis!::PIR
end

function _rank_product2!(dest::AbstractMatrix{Complex{T}},
                         a::AbstractMatrix{Complex{T}},
                         b::AbstractMatrix{Complex{T}}) where {T}
    nt, nr = size(dest)
    @inbounds for r in 1:nr
        @simd for t in 1:nt
            dest[t, r] = a[t, r] * b[t, r]
        end
    end
    return dest
end

function _rank_product3!(dest::AbstractMatrix{Complex{T}},
                         a::AbstractMatrix{Complex{T}},
                         b::AbstractMatrix{Complex{T}},
                         c::AbstractMatrix{Complex{T}}) where {T}
    nt, nr = size(dest)
    @inbounds for r in 1:nr
        @simd for t in 1:nt
            dest[t, r] = a[t, r] * b[t, r] * c[t, r]
        end
    end
    return dest
end

function _rank_filter_time!(dest::AbstractMatrix{Complex{T}},
                            h::AbstractVector{Complex{T}}) where {T}
    nt, nr = size(dest)
    @inbounds for r in 1:nr
        @simd for t in 1:nt
            dest[t, r] *= h[t]
        end
    end
    return dest
end

function CPFastRHSCache(cp, nt::Integer; T=Float64, flags=FFTW.MEASURE)
    cpd = _as_cp(cp)
    nm = size(cpd.U[1], 1)
    rank = length(cpd.λ)
    u1 = Complex{T}.(cpd.U[1])
    u2 = Complex{T}.(cpd.U[2])
    u3 = Complex{T}.(cpd.U[3])
    u4 = Complex{T}.(cpd.U[4])
    λ = Complex{T}.(cpd.λ)
    wu1 = u1 .* reshape(λ, 1, :)

    at = zeros(Complex{T}, nt, nm)
    conj_at = similar(at)
    b2 = zeros(Complex{T}, nt, rank)
    b3 = similar(b2)
    b4 = similar(b2)
    p = similar(b2)
    k = similar(at)
    conv = similar(b2)
    nonlinear = similar(at)

    fft_modes! = plan_fft!(at, 1; flags)
    ifft_modes! = plan_ifft!(nonlinear, 1; flags)
    fft_rank! = plan_fft!(conv, 1; flags)
    ifft_rank! = plan_ifft!(conv, 1; flags)

    return CPFastRHSCache(rank, at, conj_at, b2, b3, b4, p, k, conv,
                          nonlinear, u2, u3, u4, transpose(wu1),
                          fft_modes!, ifft_modes!, fft_rank!, ifft_rank!)
end

function CPRamanLowRankRHSCache(cp, nt::Integer, raman_rank::Integer;
                                T=Float64, flags=FFTW.MEASURE)
    base = CPFastRHSCache(cp, nt; T, flags)
    rank = base.rank
    k = min(Int(raman_rank), rank)
    k > 0 || error("raman_rank must be positive.")

    qbasis = zeros(Complex{T}, nt, k)
    basis_conv = similar(qbasis)
    qcoeff = zeros(Complex{T}, k, rank)
    raman_weights = zeros(Complex{T}, rank, size(base.wu1t, 2))
    qnorms = zeros(T, rank)
    selected = zeros(Int, k)
    selected_mask = falses(rank)
    fft_basis! = plan_fft!(basis_conv, 1; flags)
    ifft_basis! = plan_ifft!(basis_conv, 1; flags)

    return CPRamanLowRankRHSCache(base, k, Ref(0), qbasis, basis_conv,
                                  qcoeff, raman_weights, qnorms, selected, selected_mask,
                                  fft_basis!, ifft_basis!)
end

raman_effective_rank(cache::CPRamanLowRankRHSCache) = cache.effective_rank[]

function _select_top_norm_columns!(selected::Vector{Int},
                                   selected_mask::AbstractVector{Bool},
                                   qnorms::Vector{T},
                                   q::AbstractMatrix{Complex{T}}) where {T}
    fill!(selected_mask, false)
    rank = size(q, 2)
    @inbounds for r in 1:rank
        s = zero(T)
        for t in axes(q, 1)
            s += abs2(q[t, r])
        end
        qnorms[r] = s
    end

    @inbounds for j in eachindex(selected)
        best = 1
        best_norm = T(-1)
        for r in 1:rank
            if !selected_mask[r] && qnorms[r] > best_norm
                best = r
                best_norm = qnorms[r]
            end
        end
        selected[j] = best
        selected_mask[best] = true
    end
    return selected
end

function _orthonormalize_selected_qbasis!(qbasis::AbstractMatrix{Complex{T}},
                                          q::AbstractMatrix{Complex{T}},
                                          selected::Vector{Int};
                                          tol::T=sqrt(eps(T))) where {T}
    nt, kmax = size(qbasis)
    effective = 0
    @inbounds for j in 1:kmax
        src = selected[j]
        effective += 1
        srcnrm2 = zero(T)
        for t in 1:nt
            qbasis[t, effective] = q[t, src]
            srcnrm2 += abs2(q[t, src])
        end

        for _ in 1:2
            for p in 1:(effective - 1)
                α = zero(Complex{T})
                for t in 1:nt
                    α += conj(qbasis[t, p]) * qbasis[t, effective]
                end
                for t in 1:nt
                    qbasis[t, effective] -= α * qbasis[t, p]
                end
            end
        end

        nrm2 = zero(T)
        for t in 1:nt
            nrm2 += abs2(qbasis[t, effective])
        end
        nrm = sqrt(nrm2)
        if nrm <= tol * max(sqrt(srcnrm2), one(T))
            for t in 1:nt
                qbasis[t, effective] = zero(Complex{T})
            end
            effective -= 1
        else
            invnrm = inv(nrm)
            for t in 1:nt
                qbasis[t, effective] *= invnrm
            end
        end
    end
    @inbounds for j in (effective + 1):kmax
        for t in 1:nt
            qbasis[t, j] = zero(Complex{T})
        end
    end
    return effective
end

function n_op_rk4ip_cp_cached!(out::AbstractMatrix{Complex{T}},
                               aw::AbstractMatrix{Complex{T}},
                               sim::Simulation{T},
                               n2_prefactor::AbstractVector{Complex{T}},
                               cache::CPFastRHSCache{T},
                               f_raman::T,
                               haw::AbstractVector{Complex{T}},
                               hbw::AbstractVector{Complex{T}},
                               at_noise::Union{Nothing,AbstractMatrix{Complex{T}}}=nothing) where {T}
    isempty(hbw) || error("CPFastRHSCache currently supports scalar CP RHS with no separate hbw path.")
    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")

    cache.at .= aw
    cache.fft_modes! * cache.at
    at_noise === nothing || (cache.at .+= at_noise)

    mul!(cache.b2, cache.at, cache.u2)
    mul!(cache.b3, cache.at, cache.u3)
    cache.conj_at .= conj.(cache.at)
    mul!(cache.b4, cache.conj_at, cache.u4)

    _rank_product3!(cache.p, cache.b2, cache.b3, cache.b4)
    mul!(cache.k, cache.p, cache.wu1t)
    cache.nonlinear .= (one(T) - f_raman) .* cache.k

    if sim.include_Raman && !isempty(haw)
        _rank_product2!(cache.conv, cache.b3, cache.b4)
        cache.ifft_rank! * cache.conv
        _rank_filter_time!(cache.conv, haw)
        cache.fft_rank! * cache.conv
        _rank_product2!(cache.p, cache.b2, cache.conv)
        mul!(cache.k, cache.p, cache.wu1t)
        cache.nonlinear .+= cache.k
    end

    cache.ifft_modes! * cache.nonlinear
    @inbounds for m in 1:nm, t in 1:nt
        out[t, m] = cache.nonlinear[t, m] * n2_prefactor[t]
    end
    return out
end

function n_op_rk4ip_cp_cached(aw::AbstractMatrix{Complex{T}},
                              sim::Simulation{T},
                              n2_prefactor::AbstractVector{Complex{T}},
                              cache::CPFastRHSCache{T},
                              f_raman::T,
                              haw::AbstractVector{Complex{T}},
                              hbw::AbstractVector{Complex{T}},
                              at_noise::Union{Nothing,AbstractMatrix{Complex{T}}}=nothing) where {T}
    out = similar(aw)
    return n_op_rk4ip_cp_cached!(out, aw, sim, n2_prefactor, cache,
                                 f_raman, haw, hbw, at_noise)
end

function n_op_rk4ip_cp_lowrank_raman_cached!(out::AbstractMatrix{Complex{T}},
                                             aw::AbstractMatrix{Complex{T}},
                                             sim::Simulation{T},
                                             n2_prefactor::AbstractVector{Complex{T}},
                                             cache::CPRamanLowRankRHSCache{T},
                                             f_raman::T,
                                             haw::AbstractVector{Complex{T}},
                                             hbw::AbstractVector{Complex{T}},
                                             at_noise::AbstractMatrix{Complex{T}}) where {T}
    isempty(hbw) || error("CPRamanLowRankRHSCache currently supports scalar CP RHS with no separate hbw path.")
    base = cache.base
    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")

    base.at .= aw
    base.fft_modes! * base.at
    base.at .+= at_noise

    mul!(base.b2, base.at, base.u2)
    mul!(base.b3, base.at, base.u3)
    base.conj_at .= conj.(base.at)
    mul!(base.b4, base.conj_at, base.u4)

    _rank_product3!(base.p, base.b2, base.b3, base.b4)
    mul!(base.k, base.p, base.wu1t)
    base.nonlinear .= (one(T) - f_raman) .* base.k

    if sim.include_Raman && !isempty(haw)
        _rank_product2!(base.conv, base.b3, base.b4)

        _select_top_norm_columns!(cache.selected, cache.selected_mask,
                                  cache.qnorms, base.conv)
        keff = _orthonormalize_selected_qbasis!(cache.qbasis, base.conv,
                                                cache.selected)
        cache.effective_rank[] = keff
        if keff > 0
            @views begin
                coeff = cache.qcoeff[1:keff, :]
                basis = cache.qbasis[:, 1:keff]
                basis_conv = cache.basis_conv[:, 1:keff]
                mul!(coeff, adjoint(basis), base.conv)
                basis_conv .= basis
            end
            cache.ifft_basis! * cache.basis_conv
            @views cache.basis_conv[:, 1:keff] .*= reshape(haw, :, 1)
            cache.fft_basis! * cache.basis_conv

            @inbounds for l in 1:keff
                for m in 1:nm, r in 1:base.rank
                    cache.raman_weights[r, m] = cache.qcoeff[l, r] * base.wu1t[r, m]
                end
                mul!(base.k, base.b2, cache.raman_weights)
                for m in 1:nm
                    @views base.nonlinear[:, m] .+= cache.basis_conv[:, l] .* base.k[:, m]
                end
            end
        end
    else
        cache.effective_rank[] = 0
    end

    base.ifft_modes! * base.nonlinear
    out .= base.nonlinear
    @inbounds for m in 1:nm
        @views out[:, m] .*= n2_prefactor
    end
    return out
end

function n_op_rk4ip_cp_lowrank_raman_cached(aw::AbstractMatrix{Complex{T}},
                                            sim::Simulation{T},
                                            n2_prefactor::AbstractVector{Complex{T}},
                                            cache::CPRamanLowRankRHSCache{T},
                                            f_raman::T,
                                            haw::AbstractVector{Complex{T}},
                                            hbw::AbstractVector{Complex{T}},
                                            at_noise::AbstractMatrix{Complex{T}}) where {T}
    out = similar(aw)
    return n_op_rk4ip_cp_lowrank_raman_cached!(out, aw, sim, n2_prefactor,
                                               cache, f_raman, haw, hbw,
                                               at_noise)
end
