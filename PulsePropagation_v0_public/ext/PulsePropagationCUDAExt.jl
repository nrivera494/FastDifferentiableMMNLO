module PulsePropagationCUDAExt

using CUDA
using FFTW: fft, ifft, plan_fft!, plan_ifft!
using LinearAlgebra: I, mul!, norm
using Random
import PulsePropagation

struct CUDAFullRHSCache{T,A2,A3,PFM,PIM,PFR,PIR}
    nm::Int
    at::A2
    nonlinear::A2
    raman_pairs::A3
    sk::CUDA.CuArray{T,4}
    sra::CUDA.CuArray{T,4}
    fft_modes!::PFM
    ifft_modes!::PIM
    fft_pairs!::PFR
    ifft_pairs!::PIR
end

struct CUDACPFastRHSCache{T,A,PFM,PIM,PFR,PIR}
    rank::Int
    at::A
    conj_at::A
    b2::A
    b3::A
    b4::A
    p::A
    k::A
    conv::A
    nonlinear::A
    u2::A
    u3::A
    u4::A
    wu1t::A
    fft_modes!::PFM
    ifft_modes!::PIM
    fft_rank!::PFR
    ifft_rank!::PIR
end

function PulsePropagation.cuda_full_rhs_cache(srsk::PulsePropagation.SRSKInfo{S}, nt::Integer,
                                     nm::Integer; T=S) where {S}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    sk = CUDA.CuArray(PulsePropagation._dense_tensor(T.(srsk.sk), srsk.sk_indices, nm))
    sra = CUDA.CuArray(PulsePropagation._dense_tensor(T.(srsk.sra), srsk.sra_indices, nm))

    at = CUDA.zeros(Complex{T}, nt, nm)
    nonlinear = similar(at)
    raman_pairs = CUDA.zeros(Complex{T}, nt, nm, nm)

    fft_modes! = plan_fft!(at, 1)
    ifft_modes! = plan_ifft!(nonlinear, 1)
    fft_pairs! = plan_fft!(raman_pairs, 1)
    ifft_pairs! = plan_ifft!(raman_pairs, 1)

    return CUDAFullRHSCache{T,typeof(at),typeof(raman_pairs),
                            typeof(fft_modes!),typeof(ifft_modes!),
                            typeof(fft_pairs!),typeof(ifft_pairs!)}(
        nm, at, nonlinear, raman_pairs, sk, sra,
        fft_modes!, ifft_modes!, fft_pairs!, ifft_pairs!)
end

function PulsePropagation.cuda_cp_fast_rhs_cache(cp, nt::Integer; T=Float64)
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    cpd = PulsePropagation._as_cp(cp)
    nm = size(cpd.U[1], 1)
    rank = length(cpd.λ)

    u1 = Complex{T}.(cpd.U[1])
    u2 = CUDA.CuArray(Complex{T}.(cpd.U[2]))
    u3 = CUDA.CuArray(Complex{T}.(cpd.U[3]))
    u4 = CUDA.CuArray(Complex{T}.(cpd.U[4]))
    λ = Complex{T}.(cpd.λ)
    wu1t = CUDA.CuArray(Matrix(transpose(u1 .* reshape(λ, 1, :))))

    at = CUDA.zeros(Complex{T}, nt, nm)
    conj_at = similar(at)
    b2 = CUDA.zeros(Complex{T}, nt, rank)
    b3 = similar(b2)
    b4 = similar(b2)
    p = similar(b2)
    k = similar(at)
    conv = similar(b2)
    nonlinear = similar(at)

    fft_modes! = plan_fft!(at, 1)
    ifft_modes! = plan_ifft!(nonlinear, 1)
    fft_rank! = plan_fft!(conv, 1)
    ifft_rank! = plan_ifft!(conv, 1)

    return CUDACPFastRHSCache{T,typeof(at),typeof(fft_modes!),typeof(ifft_modes!),
                              typeof(fft_rank!),typeof(ifft_rank!)}(
        rank, at, conj_at, b2, b3, b4, p, k, conv, nonlinear,
        u2, u3, u4, wu1t, fft_modes!, ifft_modes!, fft_rank!, ifft_rank!)
end

function _mode_unfold_cuda(X, n::Integer)
    order = (n, (k for k in 1:ndims(X) if k != n)...)
    return reshape(permutedims(X, order), size(X, n), :)
end

function _khatri_rao_rows_cuda(factors)
    kr = factors[1]
    R = size(kr, 2)
    for F in factors[2:end]
        old_rows = size(kr, 1)
        new_rows = size(F, 1)
        kr = reshape(reshape(kr, old_rows, 1, R) .* reshape(F, 1, new_rows, R),
                     old_rows * new_rows, R)
    end
    return kr
end

function _normalize_columns_cuda!(A)
    norms = vec(sqrt.(sum(abs2, A; dims=1)))
    safe = ifelse.(norms .<= eps(eltype(real(norms))), one(eltype(real(norms))), norms)
    A ./= reshape(safe, 1, :)
    return Array(safe)
end

function _cp_reconstruct_cuda(λ, U, dims)
    T = eltype(λ)
    out = CUDA.zeros(T, dims)
    for r in eachindex(λ)
        out .+= λ[r] .* reshape(U[1][:, r], :, 1, 1, 1) .*
                reshape(U[2][:, r], 1, :, 1, 1) .*
                reshape(U[3][:, r], 1, 1, :, 1) .*
                reshape(U[4][:, r], 1, 1, 1, :)
    end
    return out
end

function PulsePropagation.cp_als_warm_cuda(X::AbstractArray{T,4}, rank::Integer;
                                           init=nothing, maxiter::Integer=75,
                                           tol=1e-5, ridge=T(1e-8),
                                           check_every::Integer=5,
                                           rng=Random.default_rng(),
                                           verbose::Bool=false,
                                           device=nothing,
                                           synchronize::Bool=true) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    rank > 0 || error("rank must be positive.")
    Xd = CUDA.CuArray(T.(X))
    Xnorm = norm(Xd)
    Xnorm > zero(T) || error("Cannot decompose a zero tensor.")

    λ_cpu, U_cpu = PulsePropagation._warm_started_factors(T.(X), rank; init=init, rng=rng)
    λ = CUDA.CuArray(T.(λ_cpu))
    U = [CUDA.CuArray(T.(U_cpu[n])) for n in 1:4]
    history = Float64[]
    for it in 1:maxiter
        for n in 1:4
            rest = [k for k in 1:4 if k != n]
            KR = _khatri_rao_rows_cuda([U[k] for k in rest])
            gram = CUDA.ones(T, rank, rank)
            for k in rest
                gram .*= U[k]' * U[k]
            end
            gram .+= CUDA.CuArray(ridge .* Matrix{T}(I, rank, rank))
            U[n] = (_mode_unfold_cuda(Xd, n) * KR) / gram
            λ .= CUDA.CuArray(_normalize_columns_cuda!(U[n]))
        end
        if it == 1 || it % check_every == 0 || it == maxiter
            err = norm(_cp_reconstruct_cuda(λ, U, size(X)) .- Xd) / Xnorm
            synchronize && CUDA.synchronize()
            push!(history, Float64(err))
            verbose && @info "CUDA CP-ALS" iter=it rank=rank relerr=err
            if length(history) >= 2 &&
               abs(history[end - 1] - history[end]) <= tol * max(history[end - 1], eps())
                break
            end
        end
    end
    cp = PulsePropagation.CPDecomposition(;
        λ=Array(λ),
        U=ntuple(n -> Array(U[n]), 4),
    )
    return cp, history
end

function _assert_cuda_array(x, name)
    x isa CUDA.CuArray || error("$name must be a CuArray for the in-place CUDA RHS.")
    return x
end

function _full_kerr_kernel!(nonlinear, at, sk, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        i1 = (idx - 1) ÷ nt + 1
        acc = zero(eltype(nonlinear))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
            acc += sk[i1, i2, i3, i4] * at[t, i2] * at[t, i3] * conj(at[t, i4])
        end
        @inbounds nonlinear[t, i1] = acc
    end
    return nothing
end

function _raman_pairs_kernel!(pairs, at, sra, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        rem = (idx - 1) ÷ nt
        i1 = rem % nm + 1
        i2 = rem ÷ nm + 1
        acc = zero(eltype(pairs))
        @inbounds for i4 in 1:nm, i3 in 1:nm
            acc += sra[i1, i2, i3, i4] * at[t, i3] * conj(at[t, i4])
        end
        @inbounds pairs[t, i1, i2] = acc
    end
    return nothing
end

function _raman_combine_kernel!(nonlinear, pairs, at, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        i1 = (idx - 1) ÷ nt + 1
        acc = zero(eltype(nonlinear))
        @inbounds for i2 in 1:nm
            acc += pairs[t, i1, i2] * at[t, i2]
        end
        @inbounds nonlinear[t, i1] += acc
    end
    return nothing
end

function _launch_1d(kernel, total::Int, args...)
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks kernel(args..., total)
    return nothing
end

function _adjoint_delta_kernel!(delta1, delta2, ut, gamma, nt::Int, nm::Int,
                                total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        rem = (idx - 1) ÷ nt
        i = rem % nm + 1
        j = rem ÷ nm + 1
        acc1 = zero(eltype(delta1))
        acc2 = zero(eltype(delta2))
        @inbounds for k in 1:nm, l in 1:nm
            g = gamma[l, k, i, j]
            uk = ut[t, k]
            ul = ut[t, l]
            acc1 += g * real(uk * conj(ul))
            acc2 += g * uk * ul
        end
        @inbounds begin
            delta1[t, i, j] = acc1
            delta2[t, i, j] = acc2
        end
    end
    return nothing
end

function _adjoint_terms_kernel!(term1_t, term2_t, lambdat, lambdawc_t,
                                delta1, delta2, one_m_fR, nt::Int, nm::Int,
                                total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        j = (idx - 1) ÷ nt + 1
        acc1 = zero(eltype(term1_t))
        acc2 = zero(eltype(term2_t))
        @inbounds for i in 1:nm
            acc1 += lambdat[t, i] * (2 * one_m_fR * delta1[t, i, j])
            acc2 += lambdawc_t[t, i] * delta2[t, i, j]
        end
        @inbounds begin
            term1_t[t, j] = acc1
            term2_t[t, j] = acc2
        end
    end
    return nothing
end

function _cuda_adjoint_full_rhs(lambda_tilde::CUDA.CuMatrix{Complex{T}},
                                u_tilde::CUDA.CuMatrix{Complex{T}},
                                z::T,
                                d_op::CUDA.CuMatrix{Complex{T}},
                                tau::CUDA.CuVector{T},
                                gamma::CUDA.CuArray{T,4},
                                one_m_fR::T) where {T}
    nt, nm = size(lambda_tilde)
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdat = fft(exp_p .* lambda_tilde .* reshape(tau, :, 1), 1)
    lambdawc_t = conj.(lambdat)
    ut = fft(exp_p .* u_tilde, 1)

    delta1 = CUDA.zeros(T, nt, nm, nm)
    delta2 = CUDA.zeros(Complex{T}, nt, nm, nm)
    _launch_1d(_adjoint_delta_kernel!, nt * nm * nm,
               delta1, delta2, ut, gamma, nt, nm)

    term1_t = CUDA.zeros(Complex{T}, nt, nm)
    term2_t = similar(term1_t)
    _launch_1d(_adjoint_terms_kernel!, nt * nm,
               term1_t, term2_t, lambdat, lambdawc_t, delta1, delta2,
               one_m_fR, nt, nm)

    term1 = 1im .* exp_m .* ifft(term1_t, 1)
    term2 = -1im .* exp_m .* ifft(term2_t, 1)
    return term1 .+ one_m_fR .* term2
end

function _cuda_cp_adjoint_factors(cp, ::Type{T}) where {T}
    cpd = PulsePropagation._as_cp(cp)
    λ_cpu = Complex{T}.(cpd.λ)
    U4_cpu = Complex{T}.(cpd.U[4])
    λ = CUDA.CuArray(λ_cpu)
    U1 = CUDA.CuArray(Complex{T}.(cpd.U[1]))
    U2 = CUDA.CuArray(Complex{T}.(cpd.U[2]))
    U3 = CUDA.CuArray(Complex{T}.(cpd.U[3]))
    U4 = CUDA.CuArray(U4_cpu)
    WU4t = CUDA.CuArray(Matrix(transpose(U4_cpu .* reshape(λ_cpu, 1, :))))
    return (; λ, U1, U2, U3, U4, WU4t)
end

function _cuda_adjoint_cp_rhs(lambda_tilde::CUDA.CuMatrix{Complex{T}},
                              u_tilde::CUDA.CuMatrix{Complex{T}},
                              z::T,
                              d_op::CUDA.CuMatrix{Complex{T}},
                              tau::CUDA.CuVector{T},
                              cp,
                              one_m_fR::T) where {T}
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdat = fft(exp_p .* lambda_tilde .* reshape(tau, :, 1), 1)
    lambdawc_t = conj.(lambdat)
    ut = fft(exp_p .* u_tilde, 1)

    B1 = ut * cp.U1
    B2 = ut * cp.U2
    C1 = real.(B2 .* conj.(B1))
    C2 = B2 .* B1
    L3 = lambdat * cp.U3
    LC3 = lambdawc_t * cp.U3

    term1_t = (2 * one_m_fR) .* ((L3 .* Complex{T}.(C1)) * cp.WU4t)
    term2_t = (LC3 .* C2) * cp.WU4t
    term1 = 1im .* exp_m .* ifft(term1_t, 1)
    term2 = -1im .* exp_m .* ifft(term2_t, 1)
    return term1 .+ one_m_fR .* term2
end

function _cuda_interpolate_forward(u_tilde::CUDA.CuArray{Complex{T},3},
                                   zgrid::AbstractVector{T}, z::T) where {T}
    nsave = length(zgrid)
    z <= zgrid[1] && return u_tilde[:, :, 1]
    z >= zgrid[end] && return u_tilde[:, :, end]
    q = searchsortedlast(zgrid, z)
    q = clamp(q, 1, nsave - 1)
    z0 = zgrid[q]
    z1 = zgrid[q + 1]
    θ = (z - z0) / (z1 - z0)
    return (one(T) - θ) .* u_tilde[:, :, q] .+ θ .* u_tilde[:, :, q + 1]
end

function _cuda_adjoint_rk4(rhs, y::CUDA.CuMatrix{Complex{T}}, z::T,
                           h::T) where {T}
    k1 = rhs(y, z)
    k2 = rhs(y .+ (h / 2) .* k1, z + h / 2)
    k3 = rhs(y .+ (h / 2) .* k2, z + h / 2)
    k4 = rhs(y .+ h .* k3, z + h)
    return y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function _cuda_adjoint_save_points(zgrid::AbstractVector{T}, zsave_adj,
                                   return_lambdaw_zsave::Bool) where {T}
    if return_lambdaw_zsave
        return zsave_adj === nothing ? reverse(T.(zgrid)) :
               reverse(sort(T.(collect(zsave_adj))))
    end
    return T[zgrid[1]]
end

function _cuda_integrate_adjoint(lambda_terminal::AbstractMatrix{Complex{T}},
                                 forward::PulsePropagation.PropagationOutput{T},
                                 fiber::PulsePropagation.Fiber{T},
                                 sim::PulsePropagation.Simulation{T},
                                 rhs_builder;
                                 dz_adj::T=sim.dz,
                                 zsave_adj=nothing,
                                 return_lambdaw_zsave::Bool=false,
                                 synchronize::Bool=true) where {T}
    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    size(lambda_terminal) == (nt, nm) ||
        error("lambda_terminal must have size $(nt) x $(nm).")
    length(forward.z) == size(forward.fields, 3) ||
        error("forward.z and forward.fields have inconsistent save counts.")
    all(diff(forward.z) .> zero(T)) ||
        error("forward.z must be strictly increasing.")
    dz_adj > zero(T) || error("dz_adj must be positive.")

    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    omega = PulsePropagation.ifft_frequency_grid(nt, forward.dt)
    d_op, _ = PulsePropagation.calc_d_op(fiber2, sim, nt, forward.dt, omega, fields0)
    zgrid = T.(forward.z)
    tau = (omega .+ T(2π) * sim.f0) ./ (T(2π) * sim.f0)
    tau .*= T.(PulsePropagation.damped_freq_window(nt))

    d_op_d = CUDA.CuArray(d_op)
    tau_d = CUDA.CuArray(tau)
    u_tilde = Array{Complex{T},3}(undef, nt, nm, length(zgrid))
    @inbounds for q in eachindex(zgrid)
        aw = PulsePropagation.inverse_fft(forward.fields[:, :, q], dims=1)
        u_tilde[:, :, q] = exp.(-d_op .* zgrid[q]) .* aw
    end
    u_tilde_d = CUDA.CuArray(u_tilde)
    lambda_d = CUDA.CuArray(exp.(-d_op .* zgrid[end]) .*
                            Matrix{Complex{T}}(lambda_terminal))

    rhs = rhs_builder(d_op_d, tau_d, u_tilde_d, zgrid)
    save_desc = _cuda_adjoint_save_points(zgrid, zsave_adj, return_lambdaw_zsave)
    all((zgrid[1] .<= save_desc) .& (save_desc .<= zgrid[end])) ||
        error("zsave_adj points must lie within the forward z interval.")

    saved_z = T[]
    saved_states = Vector{Array{Complex{T},2}}()
    current_z = zgrid[end]
    for target_z in save_desc
        target_z <= current_z + sqrt(eps(T)) ||
            error("zsave_adj points must be sorted within the forward z interval.")
        while current_z - target_z > sqrt(eps(T)) * max(one(T), abs(current_z))
            step = min(T(dz_adj), current_z - target_z)
            h = -step
            lambda_d = _cuda_adjoint_rk4(rhs, lambda_d, current_z, h)
            current_z += h
        end
        current_z = target_z
        synchronize && CUDA.synchronize()
        push!(saved_z, current_z)
        if return_lambdaw_zsave
            push!(saved_states, Array(exp.(d_op_d .* current_z) .* lambda_d))
        else
            push!(saved_states, Array(lambda_d))
        end
    end

    if !return_lambdaw_zsave
        out = Array{Complex{T},3}(undef, nt, nm, 1)
        out[:, :, 1] = saved_states[end]
        return PulsePropagation.AdjointOutput{T}(; z=[zgrid[1]], lambdaw=out,
                                                dt=forward.dt,
                                                frame=:interaction)
    end

    z_asc = reverse(saved_z)
    states_asc = reverse(saved_states)
    out = Array{Complex{T},3}(undef, nt, nm, length(states_asc))
    @inbounds for q in eachindex(states_asc)
        out[:, :, q] = states_asc[q]
    end
    return PulsePropagation.AdjointOutput{T}(; z=collect(z_asc), lambdaw=out,
                                            dt=forward.dt, frame=:lab)
end

function PulsePropagation.n_op_rk4ip_full_cuda!(
    out::CUDA.CuMatrix{Complex{T}},
    aw::CUDA.CuMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDAFullRHSCache{T},
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    isempty(hbw) || error("CUDA full RHS currently supports scalar propagation with no separate hbw path.")
    n2_prefactor = _assert_cuda_array(n2_prefactor, "n2_prefactor")
    haw = _assert_cuda_array(haw, "haw")
    at_noise === nothing || at_noise isa CUDA.CuArray ||
        error("at_noise must be nothing or a CuArray for the in-place CUDA RHS.")

    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")
    cache.nm == nm || error("CUDA full RHS cache was built for $(cache.nm) modes, got $nm.")

    cache.at .= aw
    cache.fft_modes! * cache.at
    at_noise === nothing || (cache.at .+= at_noise)

    total_modes = nt * nm
    _launch_1d(_full_kerr_kernel!, total_modes, cache.nonlinear,
               cache.at, cache.sk, nt, nm)

    if sim.include_Raman && !isempty(haw)
        _launch_1d(_raman_pairs_kernel!, nt * nm * nm, cache.raman_pairs,
                   cache.at, cache.sra, nt, nm)
        cache.ifft_pairs! * cache.raman_pairs
        cache.raman_pairs .*= reshape(haw, :, 1, 1)
        cache.fft_pairs! * cache.raman_pairs
        _launch_1d(_raman_combine_kernel!, total_modes, cache.nonlinear,
                   cache.raman_pairs, cache.at, nt, nm)
    end

    cache.ifft_modes! * cache.nonlinear
    out .= cache.nonlinear .* reshape(n2_prefactor, :, 1)
    return out
end

function PulsePropagation.n_op_rk4ip_full_cuda(
    aw::AbstractMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDAFullRHSCache{T},
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    awd = aw isa CUDA.CuArray ? aw : CUDA.CuArray(aw)
    out = similar(awd)
    n2d = n2_prefactor isa CUDA.CuArray ? n2_prefactor : CUDA.CuArray(n2_prefactor)
    hawd = haw isa CUDA.CuArray ? haw : CUDA.CuArray(haw)
    hbwd = hbw isa CUDA.CuArray ? hbw : CUDA.CuArray(hbw)
    noise_d = at_noise === nothing || at_noise isa CUDA.CuArray ? at_noise : CUDA.CuArray(at_noise)
    return PulsePropagation.n_op_rk4ip_full_cuda!(out, awd, sim, n2d, cache,
                                         hawd, hbwd, noise_d)
end

function PulsePropagation.n_op_rk4ip_cp_cuda!(
    out::CUDA.CuMatrix{Complex{T}},
    aw::CUDA.CuMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDACPFastRHSCache{T},
    f_raman::T,
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    isempty(hbw) || error("CUDA CP RHS currently supports scalar CP propagation with no separate hbw path.")
    n2_prefactor = _assert_cuda_array(n2_prefactor, "n2_prefactor")
    haw = _assert_cuda_array(haw, "haw")
    at_noise === nothing || at_noise isa CUDA.CuArray ||
        error("at_noise must be nothing or a CuArray for the in-place CUDA RHS.")

    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")

    cache.at .= aw
    cache.fft_modes! * cache.at
    at_noise === nothing || (cache.at .+= at_noise)

    mul!(cache.b2, cache.at, cache.u2)
    mul!(cache.b3, cache.at, cache.u3)
    cache.conj_at .= conj.(cache.at)
    mul!(cache.b4, cache.conj_at, cache.u4)

    cache.p .= cache.b2 .* cache.b3 .* cache.b4
    mul!(cache.k, cache.p, cache.wu1t)
    cache.nonlinear .= (one(T) - f_raman) .* cache.k

    if sim.include_Raman && !isempty(haw)
        cache.conv .= cache.b3 .* cache.b4
        cache.ifft_rank! * cache.conv
        cache.conv .*= reshape(haw, :, 1)
        cache.fft_rank! * cache.conv
        cache.p .= cache.b2 .* cache.conv
        mul!(cache.k, cache.p, cache.wu1t)
        cache.nonlinear .+= cache.k
    end

    cache.ifft_modes! * cache.nonlinear
    out .= cache.nonlinear .* reshape(n2_prefactor, :, 1)
    return out
end

function PulsePropagation.n_op_rk4ip_cp_cuda(
    aw::AbstractMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDACPFastRHSCache{T},
    f_raman::T,
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    awd = aw isa CUDA.CuArray ? aw : CUDA.CuArray(aw)
    out = similar(awd)
    n2d = n2_prefactor isa CUDA.CuArray ? n2_prefactor : CUDA.CuArray(n2_prefactor)
    hawd = haw isa CUDA.CuArray ? haw : CUDA.CuArray(haw)
    hbwd = hbw isa CUDA.CuArray ? hbw : CUDA.CuArray(hbw)
    noise_d = at_noise === nothing || at_noise isa CUDA.CuArray ? at_noise : CUDA.CuArray(at_noise)
    return PulsePropagation.n_op_rk4ip_cp_cuda!(out, awd, sim, n2d, cache, f_raman,
                                       hawd, hbwd, noise_d)
end

function _rk4ip_full_cuda_step!(a1w::CUDA.CuMatrix{Complex{T}},
                                a0w::CUDA.CuMatrix{Complex{T}},
                                dz::T,
                                half::CUDA.CuMatrix{Complex{T}},
                                a_ip::CUDA.CuMatrix{Complex{T}},
                                stage::CUDA.CuMatrix{Complex{T}},
                                k1::CUDA.CuMatrix{Complex{T}},
                                k2::CUDA.CuMatrix{Complex{T}},
                                k3::CUDA.CuMatrix{Complex{T}},
                                k4::CUDA.CuMatrix{Complex{T}},
                                sim::PulsePropagation.Simulation{T},
                                n2_prefactor::CUDA.CuVector{Complex{T}},
                                cache::CUDAFullRHSCache{T},
                                haw::CUDA.CuVector{Complex{T}},
                                hbw::CUDA.CuVector{Complex{T}},
                                at_noise=nothing) where {T}
    a_ip .= half .* a0w

    PulsePropagation.n_op_rk4ip_full_cuda!(k1, a0w, sim, n2_prefactor, cache, haw, hbw, at_noise)
    k1 .= half .* k1

    stage .= a_ip .+ k1 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_full_cuda!(k2, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    stage .= a_ip .+ k2 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_full_cuda!(k3, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    stage .= half .* (a_ip .+ k3 .* dz)
    PulsePropagation.n_op_rk4ip_full_cuda!(k4, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    a1w .= half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (dz / 6)) .+ k4 .* (dz / 6)
    return a1w
end

function _rk4ip_cp_cuda_step!(a1w::CUDA.CuMatrix{Complex{T}},
                              a0w::CUDA.CuMatrix{Complex{T}},
                              dz::T,
                              half::CUDA.CuMatrix{Complex{T}},
                              a_ip::CUDA.CuMatrix{Complex{T}},
                              stage::CUDA.CuMatrix{Complex{T}},
                              k1::CUDA.CuMatrix{Complex{T}},
                              k2::CUDA.CuMatrix{Complex{T}},
                              k3::CUDA.CuMatrix{Complex{T}},
                              k4::CUDA.CuMatrix{Complex{T}},
                              sim::PulsePropagation.Simulation{T},
                              n2_prefactor::CUDA.CuVector{Complex{T}},
                              cache::CUDACPFastRHSCache{T},
                              f_raman::T,
                              haw::CUDA.CuVector{Complex{T}},
                              hbw::CUDA.CuVector{Complex{T}},
                              at_noise=nothing) where {T}
    a_ip .= half .* a0w

    PulsePropagation.n_op_rk4ip_cp_cuda!(k1, a0w, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)
    k1 .= half .* k1

    stage .= a_ip .+ k1 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k2, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    stage .= a_ip .+ k2 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k3, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    stage .= half .* (a_ip .+ k3 .* dz)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k4, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    a1w .= half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (dz / 6)) .+ k4 .* (dz / 6)
    return a1w
end

function PulsePropagation.propagate_rk4ip_full_cuda(fiber::PulsePropagation.Fiber{T},
                                           initial_condition::PulsePropagation.InitialCondition{T},
                                           sim::PulsePropagation.Simulation{T};
                                           zsave=nothing,
                                           dz=nothing,
                                           device=nothing,
                                           synchronize::Bool=true) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.gain_model == 0 ||
        error("propagate_rk4ip_full_cuda currently supports passive propagation only.")

    fields0 = PulsePropagation.last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    step_dz = dz === nothing ? sim.dz : T(dz)
    step_dz > zero(T) || error("dz must be positive.")
    omega = PulsePropagation.ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = PulsePropagation.calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(PulsePropagation.damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = PulsePropagation.raman_model(fiber2, sim, nt, initial_condition.dt)
    isempty(hbw) || error("propagate_rk4ip_full_cuda currently supports scalar propagation with no separate hbw path.")
    srsk = PulsePropagation.calc_srsk(fiber3, sim, size(fiber3.sr, 1))

    save_z = if zsave === nothing
        save_period = sim.save_period == 0 ? fiber2.L0 : sim.save_period
        collect(T, 0:round(Int, fiber2.L0 / save_period)) .* save_period
    else
        T.(collect(zsave))
    end
    isempty(save_z) && error("zsave must contain at least one point.")
    first(save_z) == zero(T) || error("zsave must start at 0.")
    last(save_z) == fiber2.L0 || error("zsave must end at fiber.L0.")
    all(diff(save_z) .> zero(T)) || error("zsave must be strictly increasing.")

    half = CUDA.CuArray(exp.(d_op .* (step_dz / 2)))
    aw = CUDA.CuArray(PulsePropagation.inverse_fft(fields0, dims=1))
    next_aw = similar(aw)
    a_ip = similar(aw)
    stage = similar(aw)
    k1 = similar(aw)
    k2 = similar(aw)
    k3 = similar(aw)
    k4 = similar(aw)
    n2_prefactor_d = CUDA.CuArray(n2_prefactor)
    haw_d = CUDA.CuArray(haw)
    hbw_d = CUDA.CuArray(hbw)
    cache = PulsePropagation.cuda_full_rhs_cache(srsk, nt, nm; T=T)

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    out[:, :, 1] = fields0

    zprev = zero(T)
    @inbounds for q in 2:length(save_z)
        znext = save_z[q]
        nsteps = PulsePropagation._fixed_step_count(znext - zprev, step_dz; label="save interval")
        for _ in 1:nsteps
            _rk4ip_full_cuda_step!(next_aw, aw, step_dz, half, a_ip, stage,
                                   k1, k2, k3, k4, sim, n2_prefactor_d, cache,
                                   haw_d, hbw_d, nothing)
            aw, next_aw = next_aw, aw
        end
        synchronize && CUDA.synchronize()
        out[:, :, q] = Array(fft(aw, 1))
        zprev = znext
    end

    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PulsePropagation.PropagationOutput{T}(; z=save_z, fields=out,
                                        dt=initial_condition.dt,
                                        betas=sim_betas, dz=save_dz,
                                        t_delay=delays)
end

function PulsePropagation.propagate_rk4ip_cp_cuda(fiber::PulsePropagation.Fiber{T},
                                         initial_condition::PulsePropagation.InitialCondition{T},
                                         sim::PulsePropagation.Simulation{T},
                                         srsk_cp;
                                         zsave=nothing,
                                         dz=nothing,
                                         device=nothing,
                                         synchronize::Bool=true) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.gain_model == 0 ||
        error("propagate_rk4ip_cp_cuda currently supports passive propagation only.")

    fields0 = PulsePropagation.last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    step_dz = dz === nothing ? sim.dz : T(dz)
    step_dz > zero(T) || error("dz must be positive.")
    omega = PulsePropagation.ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = PulsePropagation.calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(PulsePropagation.damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = PulsePropagation.raman_model(fiber2, sim, nt, initial_condition.dt)
    isempty(hbw) || error("propagate_rk4ip_cp_cuda currently supports scalar CP propagation with no separate hbw path.")

    save_z = if zsave === nothing
        save_period = sim.save_period == 0 ? fiber2.L0 : sim.save_period
        collect(T, 0:round(Int, fiber2.L0 / save_period)) .* save_period
    else
        T.(collect(zsave))
    end
    isempty(save_z) && error("zsave must contain at least one point.")
    first(save_z) == zero(T) || error("zsave must start at 0.")
    last(save_z) == fiber2.L0 || error("zsave must end at fiber.L0.")
    all(diff(save_z) .> zero(T)) || error("zsave must be strictly increasing.")

    half = CUDA.CuArray(exp.(d_op .* (step_dz / 2)))
    aw = CUDA.CuArray(PulsePropagation.inverse_fft(fields0, dims=1))
    next_aw = similar(aw)
    a_ip = similar(aw)
    stage = similar(aw)
    k1 = similar(aw)
    k2 = similar(aw)
    k3 = similar(aw)
    k4 = similar(aw)
    n2_prefactor_d = CUDA.CuArray(n2_prefactor)
    haw_d = CUDA.CuArray(haw)
    hbw_d = CUDA.CuArray(hbw)
    cache = PulsePropagation.cuda_cp_fast_rhs_cache(srsk_cp, nt; T=T)

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    out[:, :, 1] = fields0

    zprev = zero(T)
    @inbounds for q in 2:length(save_z)
        znext = save_z[q]
        nsteps = PulsePropagation._fixed_step_count(znext - zprev, step_dz; label="save interval")
        for _ in 1:nsteps
            _rk4ip_cp_cuda_step!(next_aw, aw, step_dz, half, a_ip, stage,
                                 k1, k2, k3, k4, sim, n2_prefactor_d, cache,
                                 fiber3.fr, haw_d, hbw_d, nothing)
            aw, next_aw = next_aw, aw
        end
        synchronize && CUDA.synchronize()
        out[:, :, q] = Array(fft(aw, 1))
        zprev = znext
    end

    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PulsePropagation.PropagationOutput{T}(; z=save_z, fields=out,
                                        dt=initial_condition.dt,
                                        betas=sim_betas, dz=save_dz,
                                        t_delay=delays)
end

function PulsePropagation.solve_adjoint_cuda(lambda_terminal::AbstractMatrix{Complex{T}},
                                             forward::PulsePropagation.PropagationOutput{T},
                                             fiber::PulsePropagation.Fiber{T},
                                             sim::PulsePropagation.Simulation{T};
                                             device=nothing,
                                             synchronize::Bool=true,
                                             dz_adj::T=sim.dz,
                                             zsave_adj=nothing,
                                             return_lambdaw_zsave::Bool=false,
                                             linear_gain=nothing,
                                             raman::Symbol=:none,
                                             raman_fraction::T=T(0.18),
                                             adaptive::Bool=false,
                                             abstol::T=T(1e-9),
                                             reltol::T=T(1e-9)) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.scalar || error("solve_adjoint_cuda currently supports scalar modal propagation only.")
    sim.gain_model == 0 || error("solve_adjoint_cuda currently supports passive propagation only.")
    linear_gain === nothing || error("solve_adjoint_cuda does not yet support linear_gain.")
    forward.linear_gain === nothing || error("solve_adjoint_cuda does not yet support linear_gain.")
    raman in (:none, :off) ||
        error("solve_adjoint_cuda currently supports passive Kerr adjoints only; Raman CUDA adjoints are not implemented yet.")
    adaptive == false ||
        error("solve_adjoint_cuda currently uses fixed-step RK4 integration; adaptive=true is not implemented.")
    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    sim_kerr = PulsePropagation.Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0,
        dz=sim.dz, save_period=sim.save_period, midx=sim.midx,
        scalar=sim.scalar, ellipticity=sim.ellipticity,
        include_Raman=false, gain_model=sim.gain_model,
        pulse_centering=sim.pulse_centering,
        progress_bar=sim.progress_bar,
        step_method=sim.step_method, cs=sim.cs, cs_model=sim.cs_model,
        betas=sim.betas, source=sim.source)
    gamma = CUDA.CuArray(PulsePropagation._adjoint_gamma_tensor(fiber2, sim_kerr,
                                                                nm; tensor=:sk))
    rhs_builder = function(d_op_d, tau_d, u_tilde_d, zgrid)
        return (λ, z) -> begin
            u_z = _cuda_interpolate_forward(u_tilde_d, zgrid, T(z))
            _cuda_adjoint_full_rhs(λ, u_z, T(z), d_op_d, tau_d, gamma,
                                   one(T))
        end
    end
    return _cuda_integrate_adjoint(lambda_terminal, forward, fiber, sim,
        rhs_builder; dz_adj=dz_adj, zsave_adj=zsave_adj,
        return_lambdaw_zsave=return_lambdaw_zsave,
        synchronize=synchronize)
end

function PulsePropagation.solve_adjoint_compressed_rankchannels_cuda(
    lambda_terminal::AbstractMatrix{Complex{T}},
    forward::PulsePropagation.PropagationOutput{T},
    fiber::PulsePropagation.Fiber{T},
    sim::PulsePropagation.Simulation{T},
    compressed;
    device=nothing,
    synchronize::Bool=true,
    dz_adj::T=sim.dz,
    zsave_adj=nothing,
    return_lambdaw_zsave::Bool=false,
    raman::Symbol=:none,
    raman_fraction::T=T(0.18),
    adaptive::Bool=false,
    abstol::T=T(1e-9),
    reltol::T=T(1e-9)) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.scalar || error("solve_adjoint_compressed_rankchannels_cuda currently supports scalar modal propagation only.")
    sim.gain_model == 0 ||
        error("solve_adjoint_compressed_rankchannels_cuda currently supports passive propagation only.")
    forward.linear_gain === nothing ||
        error("solve_adjoint_compressed_rankchannels_cuda does not support linear_gain.")
    raman in (:none, :off) ||
        error("solve_adjoint_compressed_rankchannels_cuda currently supports passive Kerr adjoints only; Raman CUDA adjoints are not implemented yet.")
    adaptive == false ||
        error("solve_adjoint_compressed_rankchannels_cuda currently uses fixed-step RK4 integration; adaptive=true is not implemented.")
    fields0 = forward.fields[:, :, 1]
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim,
                                                            size(fields0, 2))
    c = T(2.99792458e-4)
    omega0 = T(2π) * sim.f0
    scale = fiber2.n2 * omega0 / c
    cs = PulsePropagation._compressed_bundle(compressed, zero(T))
    gamma_cp = _cuda_cp_adjoint_factors(PulsePropagation._scaled_cp(cs.sk, scale), T)
    rhs_builder = function(d_op_d, tau_d, u_tilde_d, zgrid)
        return (lambda_state, z) -> begin
            u_z = _cuda_interpolate_forward(u_tilde_d, zgrid, T(z))
            _cuda_adjoint_cp_rhs(lambda_state, u_z, T(z), d_op_d, tau_d,
                                 gamma_cp, one(T))
        end
    end
    return _cuda_integrate_adjoint(lambda_terminal, forward, fiber, sim,
        rhs_builder; dz_adj=dz_adj, zsave_adj=zsave_adj,
        return_lambdaw_zsave=return_lambdaw_zsave,
        synchronize=synchronize)
end

end
