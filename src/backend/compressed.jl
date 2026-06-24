Base.@kwdef struct CPDecomposition{T}
    λ::Vector{T}
    U::NTuple{4,Matrix{T}}
end

Base.@kwdef struct CompressedSRSK{T}
    sk::CPDecomposition{T}
    sra::Union{Nothing,CPDecomposition{T}} = nothing
    srb::Union{Nothing,CPDecomposition{T}} = nothing
    fr::T = zero(T)
end

_cp_lambda(cp::CPDecomposition) = cp.λ
_cp_factors(cp::CPDecomposition) = cp.U
_cp_lambda(cp) = cp.λ
_cp_factors(cp) = cp.U

function _as_cp(cp::CPDecomposition{T}) where {T}
    return cp
end

function _as_cp(cp)
    λ = collect(cp.λ)
    U = ntuple(i -> Matrix(cp.U[i]), 4)
    return CPDecomposition(; λ=λ, U=U)
end

function _scaled_cp(cp, scale)
    cpd = _as_cp(cp)
    return CPDecomposition(; λ=scale .* cpd.λ, U=cpd.U)
end

function cp_reconstruct(cp, dims::NTuple{4,Int})
    cpd = _as_cp(cp)
    T = eltype(cpd.λ)
    out = zeros(T, dims)
    U1, U2, U3, U4 = cpd.U
    @inbounds for r in eachindex(cpd.λ)
        λr = cpd.λ[r]
        for i4 in 1:dims[4], i3 in 1:dims[3], i2 in 1:dims[2], i1 in 1:dims[1]
            out[i1, i2, i3, i4] += λr * U1[i1, r] * U2[i2, r] * U3[i3, r] * U4[i4, r]
        end
    end
    return out
end

cp_reconstruct(cp, template::AbstractArray{<:Any,4}) = cp_reconstruct(cp, size(template))

function _mode_unfold(X::AbstractArray, n::Integer)
    order = (n, (k for k in 1:ndims(X) if k != n)...)
    return reshape(permutedims(X, order), size(X, n), :)
end

function _khatri_rao_rows(factors::Vector{<:AbstractMatrix})
    kr = Array(factors[1])
    R = size(kr, 2)
    for F in factors[2:end]
        old_rows = size(kr, 1)
        new_rows = size(F, 1)
        out = similar(kr, old_rows * new_rows, R)
        @inbounds for j in 1:new_rows, i in 1:old_rows
            out[(j - 1) * old_rows + i, :] .= kr[i, :] .* F[j, :]
        end
        kr = out
    end
    return kr
end

function _normalize_columns!(A; floor_norm=eps(real(eltype(A))))
    R = size(A, 2)
    norms = zeros(real(eltype(A)), R)
    @inbounds for r in 1:R
        nr = norm(view(A, :, r))
        if nr <= floor_norm
            nr = one(real(eltype(A)))
        else
            A[:, r] ./= nr
        end
        norms[r] = nr
    end
    return norms
end

function _warm_started_factors(X::AbstractArray{T,4}, rank::Integer;
                               init=nothing, rng=Random.default_rng()) where {T}
    rank > 0 || error("rank must be positive.")
    U = ntuple(n -> 0.01 .* randn(rng, T, size(X, n), rank), 4)
    λ = ones(T, rank)
    if init !== nothing
        init_cp = _as_cp(init)
        r0 = min(length(init_cp.λ), rank)
        @inbounds for n in 1:4
            U[n][:, 1:r0] .= init_cp.U[n][:, 1:r0]
        end
        λ[1:r0] .= init_cp.λ[1:r0]
        if r0 < rank
            scale = (norm(X) / sqrt(rank))^(1 / 4)
            @inbounds for n in 1:4
                U[n][:, r0+1:rank] .*= scale / T(0.01)
                _normalize_columns!(view(U[n], :, r0+1:rank))
            end
        end
    else
        scale = (norm(X) / sqrt(rank))^(1 / 4)
        @inbounds for n in 1:4
            U[n] .*= scale / T(0.01)
            _normalize_columns!(U[n])
        end
    end
    return λ, U
end

"""
    cp_als_warm(X, rank; init=nothing, maxiter=75, tol=1e-5, ...)

Fit a rank-`rank` CP decomposition of a real fourth-order tensor using
alternating least squares. If `init` is supplied, its existing columns are
copied into the new rank and only the added columns are freshly initialized.
The return value is `(cp, history)`, where `cp.λ` and `cp.U` match the layout
used by the earlier notebook-local GCP objects.
"""
function cp_als_warm(X::AbstractArray{T,4}, rank::Integer;
                     init=nothing, maxiter::Integer=75, tol=1e-5,
                     ridge=T(1e-8), check_every::Integer=5,
                     rng=Random.default_rng(), verbose::Bool=false) where {T}
    Xnorm = norm(X)
    Xnorm > zero(T) || error("Cannot decompose a zero tensor.")
    λ, U = _warm_started_factors(X, rank; init=init, rng=rng)
    history = Float64[]
    for it in 1:maxiter
        for n in 1:4
            rest = [k for k in 1:4 if k != n]
            KR = _khatri_rao_rows([U[k] for k in rest])
            gram = ones(T, rank, rank)
            for k in rest
                gram .*= U[k]' * U[k]
            end
            @inbounds for r in 1:rank
                gram[r, r] += ridge
            end
            U[n] .= (_mode_unfold(X, n) * KR) / gram
            λ .= _normalize_columns!(U[n])
        end
        if it == 1 || it % check_every == 0 || it == maxiter
            cp = CPDecomposition(; λ=copy(λ), U=ntuple(n -> copy(U[n]), 4))
            err = norm(cp_reconstruct(cp, X) .- X) / Xnorm
            push!(history, Float64(err))
            verbose && @info "CP-ALS" iter=it rank=rank relerr=err
            if length(history) >= 2 &&
               abs(history[end-1] - history[end]) <= tol * max(history[end-1], eps())
                break
            end
        end
    end
    cp = CPDecomposition(; λ=copy(λ), U=ntuple(n -> copy(U[n]), 4))
    return cp, history
end

function _dense_tensor(vals::AbstractVector{T}, idx::AbstractMatrix{Int},
                       n::Integer) where {T}
    out = zeros(T, n, n, n, n)
    @inbounds for q in eachindex(vals)
        i1, i2, i3, i4 = idx[:, q]
        out[i1, i2, i3, i4] += vals[q]
    end
    return out
end

"""
    compressed_srsk_tensors(fiber, sim, rank; init=nothing, ...)

Build CP decompositions for the unscaled tensors that enter the forward and
adjoint nonlinearities. `sk` is built with Raman disabled, so the forward
compressed contraction applies `(1 - fr)` at evaluation time. Raman tensors
`sra` and `srb` are unscaled; the Raman kernels carry the Raman fraction.
"""
function compressed_srsk_tensors(fiber::Fiber{T}, sim::Simulation{T}, rank::Integer;
                                 init=nothing, maxiter::Integer=75, tol=1e-5,
                                 ridge=T(1e-8), check_every::Integer=5,
                                 rng=Random.default_rng(),
                                 verbose::Bool=false) where {T}
    n = sim.scalar ? size(fiber.sr, 1) : 2 * size(fiber.sr, 1)
    sim_kerr = Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0, dz=sim.dz,
                             save_period=sim.save_period, midx=sim.midx,
                             scalar=sim.scalar, ellipticity=sim.ellipticity,
                             include_Raman=false, gain_model=sim.gain_model,
                             pulse_centering=sim.pulse_centering,
                             progress_bar=sim.progress_bar,
                             step_method=sim.step_method,
                             cs=sim.cs, cs_model=sim.cs_model,
                             betas=sim.betas, source=sim.source)
    srsk = calc_srsk(fiber, sim_kerr, size(fiber.sr, 1))
    sk_tensor = _dense_tensor(srsk.sk, srsk.sk_indices, n)
    sk_cp, sk_hist = cp_als_warm(sk_tensor, rank; init=init, maxiter=maxiter,
                                 tol=tol, ridge=ridge, check_every=check_every,
                                 rng=rng, verbose=verbose)
    sra_cp = nothing
    srb_cp = nothing
    histories = Dict(:sk => sk_hist)
    if sim.include_Raman
        sra_tensor = _dense_tensor(srsk.sra, srsk.sra_indices, n)
        sra_cp, sra_hist = cp_als_warm(sra_tensor, rank; init=sk_cp,
                                       maxiter=maxiter, tol=tol, ridge=ridge,
                                       check_every=check_every,
                                       rng=rng, verbose=verbose)
        histories[:sra] = sra_hist
        if !sim.scalar && !isempty(srsk.srb)
            srb_tensor = _dense_tensor(srsk.srb, srsk.srb_indices, n)
            srb_cp, srb_hist = cp_als_warm(srb_tensor, rank; init=sra_cp,
                                           maxiter=maxiter, tol=tol, ridge=ridge,
                                           check_every=check_every,
                                           rng=rng, verbose=verbose)
            histories[:srb] = srb_hist
        end
    end
    return CompressedSRSK(; sk=sk_cp, sra=sra_cp, srb=srb_cp, fr=fiber.fr), histories
end

_compressed_bundle(cp::CompressedSRSK{T}, fr::T) where {T} =
    CompressedSRSK(; sk=_as_cp(cp.sk),
                   sra=cp.sra === nothing ? _as_cp(cp.sk) : _as_cp(cp.sra),
                   srb=cp.srb === nothing ? nothing : _as_cp(cp.srb),
                   fr=fr)

function _compressed_bundle(cp, fr)
    cpd = _as_cp(cp)
    return CompressedSRSK(; sk=cpd, sra=cpd, srb=nothing, fr=fr)
end

function _compressed_cubic(A::AbstractMatrix{Complex{T}}, cp) where {T}
    cpd = _as_cp(cp)
    λ = Complex{T}.(cpd.λ)
    U1, U2, U3, U4 = cpd.U
    B2 = A * Complex{T}.(U2)
    B3 = A * Complex{T}.(U3)
    B4 = conj.(A) * Complex{T}.(U4)
    WU1 = Complex{T}.(U1) .* reshape(λ, 1, :)
    return (B2 .* B3 .* B4) * transpose(WU1)
end

function _compressed_raman(A::AbstractMatrix{Complex{T}}, cp,
                           kernel::AbstractVector{Complex{T}}) where {T}
    cpd = _as_cp(cp)
    λ = Complex{T}.(cpd.λ)
    U1, U2, U3, U4 = cpd.U
    B2 = A * Complex{T}.(U2)
    B3 = A * Complex{T}.(U3)
    B4 = conj.(A) * Complex{T}.(U4)
    conv = forward_fft(kernel .* inverse_fft(B3 .* B4, dims=1), dims=1)
    WU1 = Complex{T}.(U1) .* reshape(λ, 1, :)
    return (B2 .* conv) * transpose(WU1)
end

function nonlinear_term_compressed(at::Array{Complex{T},3},
                                   n2_prefactor::AbstractVector{Complex{T}},
                                   compressed, fr::T,
                                   haw::AbstractVector{Complex{T}},
                                   hbw::AbstractVector{Complex{T}},
                                   include_raman::Bool) where {T}
    nt, planes, nm = size(at)
    cs = _compressed_bundle(compressed, fr)
    nonlinear = zeros(Complex{T}, nt, planes, nm)
    for p in 1:planes
        A = @view at[:, p, :]
        nonlinear[:, p, :] .= (one(T) - fr) .* _compressed_cubic(A, cs.sk)
        if include_raman && !isempty(haw)
            nonlinear[:, p, :] .+= _compressed_raman(A, cs.sra, haw)
            if cs.srb !== nothing && !isempty(hbw)
                nonlinear[:, p, :] .+= _compressed_raman(A, cs.srb, hbw)
            end
        end
    end
    nw = inverse_fft(nonlinear, dims=1)
    @inbounds for m in 1:nm
        nw[:, :, m] .*= n2_prefactor
    end
    return nw
end

function n_op_rk4ip_compressed(aw::Matrix{Complex{T}}, sim::Simulation{T},
                               n2_prefactor::Vector{Complex{T}}, compressed,
                               fr::T, haw, hbw,
                               at_noise::Matrix{Complex{T}}) where {T}
    at = forward_fft(aw, dims=1) .+ at_noise
    nl = nonlinear_term_compressed(reshape(at, size(at, 1), 1, size(at, 2)),
                                   n2_prefactor, compressed, fr, haw, hbw,
                                   sim.include_Raman)
    return reshape(nl, size(aw, 1), size(aw, 2))
end

function _adjoint_delta_tensors_cp(ut::AbstractMatrix{Complex{T}}, cp) where {T}
    cpd = _as_cp(cp)
    λ = cpd.λ
    U1, U2, U3, U4 = cpd.U
    nt, nm = size(ut)
    delta1 = zeros(T, nt, nm, nm)
    delta2 = zeros(Complex{T}, nt, nm, nm)
    C1 = (ut * Complex{T}.(U2)) .* (conj.(ut) * Complex{T}.(U1))
    C2 = (ut * Complex{T}.(U2)) .* (ut * Complex{T}.(U1))
    @inbounds for r in eachindex(λ), j in 1:nm, i in 1:nm, t in 1:nt
        w = λ[r] * U3[i, r] * U4[j, r]
        delta1[t, i, j] += w * real(C1[t, r])
        delta2[t, i, j] += w * C2[t, r]
    end
    return delta1, delta2
end

function _adjoint_gamma_pair_cp(a::AbstractMatrix{Complex{T}},
                                b::AbstractMatrix{Complex{T}}, cp) where {T}
    cpd = _as_cp(cp)
    λ = cpd.λ
    U1, U2, U3, U4 = cpd.U
    nt, nm = size(a)
    out = zeros(Complex{T}, nt, nm, nm)
    C = (a * Complex{T}.(U1)) .* (b * Complex{T}.(U2))
    @inbounds for r in eachindex(λ), j in 1:nm, i in 1:nm, t in 1:nt
        out[t, i, j] += λ[r] * U3[i, r] * U4[j, r] * C[t, r]
    end
    return out
end

function _adjoint_rhs_kerr_compressed(lambda_tilde::Matrix{Complex{T}},
                                      u_tilde::Matrix{Complex{T}},
                                      z::T,
                                      d_op::Matrix{Complex{T}},
                                      tau_window::Vector{T},
                                      gamma_cp,
                                      one_m_fR::T,
                                      hRω::AbstractVector{Complex{T}}=Complex{T}[];
                                      raman_gammas::Tuple=(),
                                      raman_kernels::Tuple=()) where {T}
    nt, nm = size(lambda_tilde)
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdaw = exp_p .* lambda_tilde
    lambdat = forward_fft(lambdaw .* reshape(tau_window, :, 1), dims=1)
    lambdawc_t = conj.(lambdat)

    uw = exp_p .* u_tilde
    ut = forward_fft(uw, dims=1)
    delta1, delta2 = _adjoint_delta_tensors_cp(ut, gamma_cp)
    active_raman_gammas = isempty(raman_gammas) && !isempty(hRω) ? (gamma_cp,) : raman_gammas
    active_raman_kernels = isempty(raman_kernels) && !isempty(hRω) ? (hRω,) : raman_kernels
    deltaR1 = zeros(T, nt, nm, nm)
    for (gamma_r, kernel_r) in zip(active_raman_gammas, active_raman_kernels)
        delta1_r, _ = gamma_r === gamma_cp ? (delta1, delta2) : _adjoint_delta_tensors_cp(ut, gamma_r)
        conv = forward_fft(Complex{T}.(delta1_r), dims=1)
        conv .*= reshape(kernel_r, :, 1, 1)
        conv = fftshift(inverse_fft(conv, dims=1), 1)
        deltaR1 .+= real.(conv)
    end

    term1_t = zeros(Complex{T}, nt, nm)
    term2_t = zeros(Complex{T}, nt, nm)
    @inbounds for t in 1:nt, j in 1:nm
        acc1 = zero(Complex{T})
        acc2 = zero(Complex{T})
        for i in 1:nm
            acc1 += lambdat[t, i] * (2 * one_m_fR * delta1[t, i, j] + deltaR1[t, i, j])
            acc2 += lambdawc_t[t, i] * delta2[t, i, j]
        end
        term1_t[t, j] = acc1
        term2_t[t, j] = acc2
    end

    term1 = 1im .* exp_m .* inverse_fft(term1_t, dims=1)
    term2 = -1im .* exp_m .* inverse_fft(term2_t, dims=1)
    rhs = term1 .+ one_m_fR .* term2

    if !isempty(active_raman_kernels)
        sigma = Complex{T}.(exp.(1im * T(π) .* repeat([zero(T), one(T)], div(nt, 2))))
        for (gamma_r, kernel_r) in zip(active_raman_gammas, active_raman_kernels)
            pair3 = _adjoint_gamma_pair_cp(lambdat, conj.(ut), gamma_r)
            tmp3 = inverse_fft(pair3, dims=1)
            tmp3 .*= reshape(kernel_r .* sigma, :, 1, 1)
            tmp3 = forward_fft(tmp3, dims=1)
            term3_t = _adjoint_sum_pair(tmp3, ut)

            pair4 = _adjoint_gamma_pair_cp(lambdawc_t, ut, gamma_r)
            tmp4 = inverse_fft(pair4, dims=1)
            tmp4 .*= reshape(kernel_r .* sigma, :, 1, 1)
            tmp4 = forward_fft(tmp4, dims=1)
            term4_t = _adjoint_sum_pair(tmp4, ut)

            rhs .+= 1im .* exp_m .* inverse_fft(term3_t, dims=1)
            rhs .+= -1im .* exp_m .* inverse_fft(term4_t, dims=1)
        end
    end

    return rhs
end

function _adjoint_rank_channels(ut::AbstractMatrix{Complex{T}}, cp) where {T}
    cpd = _as_cp(cp)
    U1, U2, _, _ = cpd.U
    B1 = ut * Complex{T}.(U1)
    B2 = ut * Complex{T}.(U2)
    C1 = real.(B2 .* conj.(B1))
    C2 = B2 .* B1
    return C1, C2, B1, B2
end

function _adjoint_rhs_kerr_compressed_rankchannels(lambda_tilde::Matrix{Complex{T}},
                                                   u_tilde::Matrix{Complex{T}},
                                                   z::T,
                                                   d_op::Matrix{Complex{T}},
                                                   tau_window::Vector{T},
                                                   gamma_cp,
                                                   one_m_fR::T,
                                                   hRω::AbstractVector{Complex{T}}=Complex{T}[];
                                                   raman_gammas::Tuple=(),
                                                   raman_kernels::Tuple=()) where {T}
    nt, _ = size(lambda_tilde)
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdaw = exp_p .* lambda_tilde
    lambdat = forward_fft(lambdaw .* reshape(tau_window, :, 1), dims=1)
    lambdawc_t = conj.(lambdat)

    uw = exp_p .* u_tilde
    ut = forward_fft(uw, dims=1)

    gamma0 = _as_cp(gamma_cp)
    λ0 = Complex{T}.(gamma0.λ)
    _, _, U3_0, U4_0 = gamma0.U
    U3c_0 = Complex{T}.(U3_0)
    U4c_0 = Complex{T}.(U4_0)
    WU4_0 = U4c_0 .* reshape(λ0, 1, :)
    C1_0, C2_0, _, _ = _adjoint_rank_channels(ut, gamma0)

    L3_0 = lambdat * U3c_0
    LC3_0 = lambdawc_t * U3c_0
    term1_t = (2 * one_m_fR) .* ((L3_0 .* Complex{T}.(C1_0)) * transpose(WU4_0))
    term2_t = (LC3_0 .* C2_0) * transpose(WU4_0)
    term4_raman_t = zeros(Complex{T}, size(term2_t))

    active_raman_gammas = isempty(raman_gammas) && !isempty(hRω) ? (gamma0,) : raman_gammas
    active_raman_kernels = isempty(raman_kernels) && !isempty(hRω) ? (hRω,) : raman_kernels

    if !isempty(active_raman_kernels)
        sigma = Complex{T}.(exp.(1im * T(π) .* repeat([zero(T), one(T)], div(nt, 2))))
        for (gamma_r_raw, kernel_r) in zip(active_raman_gammas, active_raman_kernels)
            gamma_r = _as_cp(gamma_r_raw)
            λr = Complex{T}.(gamma_r.λ)
            U1, U2, U3, U4 = gamma_r.U
            U1c = Complex{T}.(U1)
            U2c = Complex{T}.(U2)
            U3c = Complex{T}.(U3)
            U4c = Complex{T}.(U4)
            WU3 = U3c .* reshape(λr, 1, :)
            WU4 = U4c .* reshape(λr, 1, :)

            C1, _, _, _ = gamma_r === gamma0 ? (C1_0, C2_0, nothing, nothing) :
                           _adjoint_rank_channels(ut, gamma_r)
            convC1 = forward_fft(Complex{T}.(C1), dims=1)
            convC1 .*= reshape(kernel_r, :, 1)
            convC1 = fftshift(inverse_fft(convC1, dims=1), 1)
            L3 = gamma_r === gamma0 ? L3_0 : lambdat * U3c
            term1_t .+= (L3 .* Complex{T}.(real.(convC1))) * transpose(WU4)

            L1 = lambdat * U1c
            U2bar = conj.(ut) * U2c
            pair3_channels = L1 .* U2bar
            conv3 = inverse_fft(pair3_channels, dims=1)
            conv3 .*= reshape(kernel_r .* sigma, :, 1)
            conv3 = forward_fft(conv3, dims=1)
            U4u = ut * U4c
            term3_t = (conv3 .* U4u) * transpose(WU3)

            LC1 = lambdawc_t * U1c
            U2u = ut * U2c
            pair4_channels = LC1 .* U2u
            conv4 = inverse_fft(pair4_channels, dims=1)
            conv4 .*= reshape(kernel_r .* sigma, :, 1)
            conv4 = forward_fft(conv4, dims=1)
            term4_t = (conv4 .* U4u) * transpose(WU3)

            term1_t .+= term3_t
            term4_raman_t .+= term4_t
        end
    end

    term1 = 1im .* exp_m .* inverse_fft(term1_t, dims=1)
    term2 = -1im .* exp_m .* inverse_fft(term2_t, dims=1)
    term4 = -1im .* exp_m .* inverse_fft(term4_raman_t, dims=1)
    return term1 .+ one_m_fR .* term2 .+ term4
end
