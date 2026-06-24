function _adjoint_gamma_tensor(fiber::Fiber{T}, sim::Simulation{T},
                               nm::Integer; tensor::Symbol=:sk) where {T}
    c = T(2.99792458e-4)
    omega0 = T(2π) * sim.f0
    srsk = calc_srsk(fiber, sim, size(fiber.sr, 1))
    gamma = zeros(T, nm, nm, nm, nm)
    scale = fiber.n2 * omega0 / c
    vals, idx = if tensor === :sk
        srsk.sk, srsk.sk_indices
    elseif tensor === :sra
        srsk.sra, srsk.sra_indices
    elseif tensor === :srb
        srsk.srb, srsk.srb_indices
    else
        error("Unknown adjoint tensor $(tensor).")
    end
    @inbounds for q in eachindex(vals)
        i1, i2, i3, i4 = idx[:, q]
        gamma[i1, i2, i3, i4] += scale * vals[q]
    end
    return gamma
end

function _adjoint_delta_tensors(ut::AbstractMatrix{Complex{T}},
                                gamma::Array{T,4}) where {T}
    nt, nm = size(ut)
    delta1 = zeros(T, nt, nm, nm)
    delta2 = zeros(Complex{T}, nt, nm, nm)
    @inbounds for t in 1:nt, i in 1:nm, j in 1:nm
        acc1 = zero(T)
        acc2 = zero(Complex{T})
        for k in 1:nm, l in 1:nm
            g = gamma[l, k, i, j]
            uk = ut[t, k]
            ul = ut[t, l]
            acc1 += g * real(uk * conj(ul))
            acc2 += g * uk * ul
        end
        delta1[t, i, j] = acc1
        delta2[t, i, j] = acc2
    end
    return delta1, delta2
end

function _adjoint_gamma_pair(a::AbstractMatrix{Complex{T}},
                             b::AbstractMatrix{Complex{T}},
                             gamma::Array{T,4}) where {T}
    nt, nm = size(a)
    out = zeros(Complex{T}, nt, nm, nm)
    @inbounds for t in 1:nt, i in 1:nm, j in 1:nm
        acc = zero(Complex{T})
        for l in 1:nm, k in 1:nm
            acc += gamma[l, k, i, j] * a[t, l] * b[t, k]
        end
        out[t, i, j] = acc
    end
    return out
end

function _adjoint_sum_pair(pair::Array{Complex{T},3},
                           u::AbstractMatrix{Complex{T}}) where {T}
    nt, nm, _ = size(pair)
    out = zeros(Complex{T}, nt, nm)
    @inbounds for t in 1:nt, i in 1:nm
        acc = zero(Complex{T})
        for j in 1:nm
            acc += pair[t, i, j] * u[t, j]
        end
        out[t, i] = acc
    end
    return out
end

function agarwal_raman_response(nt::Integer, dt::T,
                                fR::T=T(0.18),
                                tau1::T=T(12.2),
                                tau2::T=T(32.0)) where {T}
    t = (collect(T, 0:nt-1) .- floor(T, nt / 2)) .* dt .* T(1e-12)
    hrt = zeros(T, nt)
    idx = t .>= zero(T)
    tpos = t[idx]
    hrt[idx] .= fR .* dt .* T(1e3) .* (tau1^2 + tau2^2) ./ (tau1 * tau2^2) .*
                exp.(-tpos .* T(1e15) ./ tau2) .* sin.(tpos .* T(1e15) ./ tau1)
    return forward_fft(hrt, dims=1)
end

function _adjoint_rhs_kerr(lambda_tilde::Matrix{Complex{T}},
                           u_tilde::Matrix{Complex{T}},
                           z::T,
                           d_op::Matrix{Complex{T}},
                           linear_gain::Union{Nothing,LinearGain{T}},
                           tau_window::Vector{T},
                           gamma::Array{T,4},
                           one_m_fR::T,
                           hRω::AbstractVector{Complex{T}}=Complex{T}[];
                           raman_gammas::Tuple=(),
                           raman_kernels::Tuple=()) where {T}
    nt, nm = size(lambda_tilde)
    adj_lin = _adjoint_linear_exponent(d_op, z, linear_gain)
    adj_exp_p = exp.(adj_lin)
    adj_exp_m = exp.(-adj_lin)
    fwd_exp_p = exp.(_linear_exponent(d_op, z, linear_gain))

    lambdaw = adj_exp_p .* lambda_tilde
    lambdat = forward_fft(lambdaw .* reshape(tau_window, :, 1), dims=1)
    lambdawc_t = conj.(lambdat)

    uw = fwd_exp_p .* u_tilde
    ut = forward_fft(uw, dims=1)
    delta1, delta2 = _adjoint_delta_tensors(ut, gamma)
    active_raman_gammas = isempty(raman_gammas) && !isempty(hRω) ? (gamma,) : raman_gammas
    active_raman_kernels = isempty(raman_kernels) && !isempty(hRω) ? (hRω,) : raman_kernels
    deltaR1 = zeros(T, nt, nm, nm)
    for (gamma_r, kernel_r) in zip(active_raman_gammas, active_raman_kernels)
        delta1_r, _ = gamma_r === gamma ? (delta1, delta2) : _adjoint_delta_tensors(ut, gamma_r)
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

    term1 = 1im .* adj_exp_m .* inverse_fft(term1_t, dims=1)
    term2 = -1im .* adj_exp_m .* inverse_fft(term2_t, dims=1)
    rhs = term1 .+ one_m_fR .* term2

    if !isempty(active_raman_kernels)
        sigma = Complex{T}.(exp.(1im * T(π) .* repeat([zero(T), one(T)], div(nt, 2))))
        for (gamma_r, kernel_r) in zip(active_raman_gammas, active_raman_kernels)
            pair3 = _adjoint_gamma_pair(lambdat, conj.(ut), gamma_r)
            tmp3 = inverse_fft(pair3, dims=1)
            tmp3 .*= reshape(kernel_r .* sigma, :, 1, 1)
            tmp3 = forward_fft(tmp3, dims=1)
            term3_t = _adjoint_sum_pair(tmp3, ut)

            pair4 = _adjoint_gamma_pair(lambdawc_t, ut, gamma_r)
            tmp4 = inverse_fft(pair4, dims=1)
            tmp4 .*= reshape(kernel_r .* sigma, :, 1, 1)
            tmp4 = forward_fft(tmp4, dims=1)
            term4_t = _adjoint_sum_pair(tmp4, ut)

            rhs .+= 1im .* adj_exp_m .* inverse_fft(term3_t, dims=1)
            rhs .+= -1im .* adj_exp_m .* inverse_fft(term4_t, dims=1)
        end
    end

    return rhs
end

function _adjoint_rhs_kerr(lambda_tilde::Matrix{Complex{T}},
                           u_tilde::Matrix{Complex{T}},
                           z::T,
                           d_op::Matrix{Complex{T}},
                           tau_window::Vector{T},
                           gamma::Array{T,4},
                           one_m_fR::T,
                           hRω::AbstractVector{Complex{T}}=Complex{T}[];
                           raman_gammas::Tuple=(),
                           raman_kernels::Tuple=()) where {T}
    return _adjoint_rhs_kerr(lambda_tilde, u_tilde, z, d_op, nothing,
                             tau_window, gamma, one_m_fR, hRω;
                             raman_gammas=raman_gammas,
                             raman_kernels=raman_kernels)
end

function _adjoint_linear_exponent(d_op::AbstractMatrix{Complex{T}}, z::T,
                                  gain::Union{Nothing,LinearGain{T}}) where {T}
    nt, nm = size(d_op)
    return d_op .* z .- Complex{T}.(_linear_gain_power_integral(gain, z, nt, nm) ./ 2)
end

function _linear_interpolate(a::Matrix{Complex{T}}, b::Matrix{Complex{T}},
                             θ::T) where {T}
    return (one(T) - θ) .* a .+ θ .* b
end

function _adjoint_interpolate_forward(u_tilde::Array{Complex{T},3},
                                      zgrid::AbstractVector{T},
                                      z::T,
                                      ode_sol=nothing) where {T}
    ode_sol !== nothing && return Matrix{Complex{T}}(ode_sol(z))
    nsave = length(zgrid)
    z <= zgrid[1] && return u_tilde[:, :, 1]
    z >= zgrid[end] && return u_tilde[:, :, end]
    q = searchsortedlast(zgrid, z)
    q = clamp(q, 1, nsave - 1)
    z0 = zgrid[q]
    z1 = zgrid[q + 1]
    θ = (z - z0) / (z1 - z0)
    return _linear_interpolate(u_tilde[:, :, q], u_tilde[:, :, q + 1], θ)
end

"""
    solve_adjoint(lambda_terminal, forward, fiber, sim; kwargs...)

Integrate the continuous adjoint equation backward along a saved forward
trajectory. `lambda_terminal` is a lab-frame Fourier-domain terminal adjoint
with size `Nt x Nm`. The current implementation supports passive scalar
multimode propagation and the Kerr part of the Julia-adjoint-solver adjoint.

The forward trajectory must be saved densely enough for the requested adjoint
step; the adjoint is integrated with DifferentialEquations.jl `Vern9()` over
the saved forward trajectory using linear interpolation in z.

With terminal condition `lambda_terminal = 2uω(L)` for a carrier-normalized
photon number `Nt*ε*sum(abs2, uω(L))`, the input photon-normalized Fourier
gradient is `0.5 * sqrt(Nt*ε) * conj(adjoint.lambdaw[:, :, 1])`.
"""
function solve_adjoint(lambda_terminal::AbstractMatrix{Complex{T}},
                       forward::PropagationOutput{T},
                       fiber::Fiber{T},
                       sim::Simulation{T};
                       dz_adj::T=sim.dz,
                       zsave_adj=nothing,
                       return_lambdaw_zsave::Bool=false,
                       linear_gain=nothing,
                       raman::Symbol=:none,
                       raman_fraction::T=T(0.18),
                       adaptive::Bool=false,
                       abstol::T=T(1e-9),
                       reltol::T=T(1e-9)) where {T}
    sim.gain_model == 0 || error("solve_adjoint gain adjoint is not implemented yet.")
    raman in (:none, :off, :agarwal, :model, :auto) ||
        error("raman must be :none, :off, :agarwal, :model, or :auto.")

    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    size(lambda_terminal) == (nt, nm) ||
        error("lambda_terminal must have size $(nt) x $(nm).")
    length(forward.z) == size(forward.fields, 3) ||
        error("forward.z and forward.fields have inconsistent save counts.")
    all(diff(forward.z) .> zero(T)) ||
        error("forward.z must be strictly increasing.")
    dz_adj > zero(T) || error("dz_adj must be positive.")

    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, forward.dt)
    d_op, _ = calc_d_op(fiber2, sim, nt, forward.dt, omega, fields0)
    zgrid = forward.z
    if forward.linear_gain !== nothing && linear_gain === nothing
        error("Forward trajectory used linear_gain; pass the same linear_gain to solve_adjoint.")
    end
    if forward.linear_gain === nothing && linear_gain !== nothing
        error("solve_adjoint received linear_gain, but the forward trajectory was passive.")
    end
    gain = _canonical_linear_gain(linear_gain, nt, nm, fiber2.L0; zsave=zgrid)
    if !_linear_gain_matches(forward.linear_gain, gain, nt, nm, fiber2.L0)
        error("solve_adjoint linear_gain does not match the forward trajectory.")
    end
    sim_kerr = Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0, dz=sim.dz,
                             save_period=sim.save_period, midx=sim.midx,
                             scalar=sim.scalar, ellipticity=sim.ellipticity,
                             include_Raman=false, gain_model=sim.gain_model,
                             pulse_centering=sim.pulse_centering,
                             progress_bar=sim.progress_bar,
                             step_method=sim.step_method,
                             cs=sim.cs, cs_model=sim.cs_model,
                             betas=sim.betas, source=sim.source)
    gamma = _adjoint_gamma_tensor(fiber2, sim_kerr, nm; tensor=:sk)
    tau = (omega .+ T(2π) * sim.f0) ./ (T(2π) * sim.f0)
    tau .*= T.(damped_freq_window(nt))
    one_m_fR = one(T)
    hRω = Complex{T}[]
    raman_gammas = ()
    raman_kernels = ()
    if raman === :agarwal
        one_m_fR = one(T) - raman_fraction
        hRω = agarwal_raman_response(nt, forward.dt, raman_fraction)
    elseif raman in (:model, :auto) && sim.include_Raman
        # fiber_raman, haw, hbw = raman_model(fiber2, sim, nt, forward.dt)
        fiber_raman, haw, hbw = raman_adjoint_model_kernels(fiber2, sim, nt, forward.dt)
        one_m_fR = one(T) - fiber_raman.fr
        gamma_a = _adjoint_gamma_tensor(fiber_raman, sim_kerr, nm; tensor=:sra)
        if !isempty(hbw)
            gamma_b = _adjoint_gamma_tensor(fiber_raman, sim_kerr, nm; tensor=:srb)
            raman_gammas = (gamma_a, gamma_b)
            raman_kernels = (haw, hbw)
        else
            raman_gammas = (gamma_a,)
            raman_kernels = (haw,)
        end
    end

    nsave = length(zgrid)
    u_tilde = Array{Complex{T},3}(undef, nt, nm, nsave)
    @inbounds for q in 1:nsave
        aw = inverse_fft(forward.fields[:, :, q], dims=1)
        u_tilde[:, :, q] = exp.(-_linear_exponent(d_op, zgrid[q], gain)) .* aw
    end

    lambda_tilde = exp.(-_adjoint_linear_exponent(d_op, zgrid[end], gain)) .* Matrix{Complex{T}}(lambda_terminal)

    function rhs!(dlambda, lambda, _, z)
        u_z = _adjoint_interpolate_forward(u_tilde, zgrid, T(z), forward.ode_sol)
        dlambda .= _adjoint_rhs_kerr(lambda, u_z, T(z), d_op, gain, tau, gamma,
                                     one_m_fR, hRω;
                                     raman_gammas=raman_gammas,
                                     raman_kernels=raman_kernels)
        return nothing
    end

    saveat = if return_lambdaw_zsave
        if zsave_adj === nothing
            reverse(zgrid)
        else
            reverse(sort(T.(collect(zsave_adj))))
        end
    else
        [zgrid[1]]
    end
    prob = ODEProblem(rhs!, lambda_tilde, (zgrid[end], zgrid[1]))
    sol = DifferentialEquations.solve(prob, Vern9(); dt=dz_adj, adaptive=adaptive,
                abstol=abstol, reltol=reltol, saveat=saveat)
    sol.retcode == ReturnCode.Success ||
        error("Vern9 adjoint solve failed with retcode $(sol.retcode).")

    if !return_lambdaw_zsave
        lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, 1)
        lambdaw_saved[:, :, 1] = sol.u[end]
        return AdjointOutput{T}(; z=[zgrid[1]], lambdaw=lambdaw_saved,
                                dt=forward.dt, frame=:interaction)
    end

    z_saved_ascending = reverse(T.(sol.t))
    lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, length(sol.u))
    @inbounds for (q, (z, state)) in enumerate(zip(z_saved_ascending, reverse(sol.u)))
        lambdaw_saved[:, :, q] = exp.(_adjoint_linear_exponent(d_op, z, gain)) .* state
    end
    return AdjointOutput{T}(; z=collect(z_saved_ascending), lambdaw=lambdaw_saved,
                            dt=forward.dt, frame=:lab)
end

function solve_adjoint_compressed(lambda_terminal::AbstractMatrix{Complex{T}},
                                  forward::PropagationOutput{T},
                                  fiber::Fiber{T},
                                  sim::Simulation{T},
                                  compressed;
                                  dz_adj::T=sim.dz,
                                  zsave_adj=nothing,
                                  return_lambdaw_zsave::Bool=false,
                                  raman::Symbol=:none,
                                  raman_fraction::T=T(0.18),
                                  adaptive::Bool=false,
                                  abstol::T=T(1e-9),
                                  reltol::T=T(1e-9)) where {T}
    sim.gain_model == 0 || error("solve_adjoint_compressed gain adjoint is not implemented yet.")
    forward.linear_gain === nothing ||
        error("solve_adjoint_compressed does not currently support linear_gain.")
    raman in (:none, :off, :agarwal, :model, :auto) ||
        error("raman must be :none, :off, :agarwal, :model, or :auto.")

    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    size(lambda_terminal) == (nt, nm) ||
        error("lambda_terminal must have size $(nt) x $(nm).")
    length(forward.z) == size(forward.fields, 3) ||
        error("forward.z and forward.fields have inconsistent save counts.")
    all(diff(forward.z) .> zero(T)) ||
        error("forward.z must be strictly increasing.")
    dz_adj > zero(T) || error("dz_adj must be positive.")

    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, forward.dt)
    d_op, _ = calc_d_op(fiber2, sim, nt, forward.dt, omega, fields0)
    c = T(2.99792458e-4)
    omega0 = T(2π) * sim.f0
    scale = fiber2.n2 * omega0 / c
    tau = (omega .+ omega0) ./ omega0
    tau .*= T.(damped_freq_window(nt))

    cs = _compressed_bundle(compressed, zero(T))
    gamma_cp = _scaled_cp(cs.sk, scale)
    one_m_fR = one(T)
    hRω = Complex{T}[]
    raman_gammas = ()
    raman_kernels = ()
    if raman === :agarwal
        one_m_fR = one(T) - raman_fraction
        hRω = agarwal_raman_response(nt, forward.dt, raman_fraction)
    elseif raman in (:model, :auto) && sim.include_Raman
        # fiber_raman, haw, hbw = raman_model(fiber2, sim, nt, forward.dt)
        fiber_raman, haw, hbw = raman_adjoint_model_kernels(fiber2, sim, nt, forward.dt)
        one_m_fR = one(T) - fiber_raman.fr
        cs_model = _compressed_bundle(compressed, fiber_raman.fr)
        gamma_a = _scaled_cp(cs_model.sra, scale)
        if cs_model.srb !== nothing && !isempty(hbw)
            gamma_b = _scaled_cp(cs_model.srb, scale)
            raman_gammas = (gamma_a, gamma_b)
            raman_kernels = (haw, hbw)
        else
            raman_gammas = (gamma_a,)
            raman_kernels = (haw,)
        end
    end

    zgrid = forward.z
    nsave = length(zgrid)
    u_tilde = Array{Complex{T},3}(undef, nt, nm, nsave)
    @inbounds for q in 1:nsave
        aw = inverse_fft(forward.fields[:, :, q], dims=1)
        u_tilde[:, :, q] = exp.(-d_op .* zgrid[q]) .* aw
    end

    lambda_tilde = exp.(-d_op .* zgrid[end]) .* Matrix{Complex{T}}(lambda_terminal)

    function rhs!(dlambda, lambda, _, z)
        u_z = _adjoint_interpolate_forward(u_tilde, zgrid, T(z), forward.ode_sol)
        dlambda .= _adjoint_rhs_kerr_compressed(lambda, u_z, T(z), d_op, tau,
                                                gamma_cp, one_m_fR, hRω;
                                                raman_gammas=raman_gammas,
                                                raman_kernels=raman_kernels)
        return nothing
    end

    saveat = if return_lambdaw_zsave
        if zsave_adj === nothing
            reverse(zgrid)
        else
            reverse(sort(T.(collect(zsave_adj))))
        end
    else
        [zgrid[1]]
    end
    prob = ODEProblem(rhs!, lambda_tilde, (zgrid[end], zgrid[1]))
    sol = DifferentialEquations.solve(prob, Vern9(); dt=dz_adj, adaptive=adaptive,
                abstol=abstol, reltol=reltol, saveat=saveat)
    sol.retcode == ReturnCode.Success ||
        error("Vern9 compressed adjoint solve failed with retcode $(sol.retcode).")

    if !return_lambdaw_zsave
        lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, 1)
        lambdaw_saved[:, :, 1] = sol.u[end]
        return AdjointOutput{T}(; z=[zgrid[1]], lambdaw=lambdaw_saved,
                                dt=forward.dt, frame=:interaction)
    end

    z_saved_ascending = reverse(T.(sol.t))
    lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, length(sol.u))
    @inbounds for (q, (z, state)) in enumerate(zip(z_saved_ascending, reverse(sol.u)))
        lambdaw_saved[:, :, q] = exp.(d_op .* z) .* state
    end
    return AdjointOutput{T}(; z=collect(z_saved_ascending), lambdaw=lambdaw_saved,
                            dt=forward.dt, frame=:lab)
end

function solve_adjoint_compressed_rankchannels(lambda_terminal::AbstractMatrix{Complex{T}},
                                               forward::PropagationOutput{T},
                                               fiber::Fiber{T},
                                               sim::Simulation{T},
                                               compressed;
                                               dz_adj::T=sim.dz,
                                               zsave_adj=nothing,
                                               return_lambdaw_zsave::Bool=false,
                                               raman::Symbol=:none,
                                               raman_fraction::T=T(0.18),
                                               adaptive::Bool=false,
                                               abstol::T=T(1e-9),
                                               reltol::T=T(1e-9)) where {T}
    sim.gain_model == 0 || error("solve_adjoint_compressed_rankchannels gain adjoint is not implemented yet.")
    forward.linear_gain === nothing ||
        error("solve_adjoint_compressed_rankchannels does not currently support linear_gain.")
    raman in (:none, :off, :agarwal, :model, :auto) ||
        error("raman must be :none, :off, :agarwal, :model, or :auto.")

    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    size(lambda_terminal) == (nt, nm) ||
        error("lambda_terminal must have size $(nt) x $(nm).")
    length(forward.z) == size(forward.fields, 3) ||
        error("forward.z and forward.fields have inconsistent save counts.")
    all(diff(forward.z) .> zero(T)) ||
        error("forward.z must be strictly increasing.")
    dz_adj > zero(T) || error("dz_adj must be positive.")

    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, forward.dt)
    d_op, _ = calc_d_op(fiber2, sim, nt, forward.dt, omega, fields0)
    c = T(2.99792458e-4)
    omega0 = T(2π) * sim.f0
    scale = fiber2.n2 * omega0 / c
    tau = (omega .+ omega0) ./ omega0
    tau .*= T.(damped_freq_window(nt))

    cs = _compressed_bundle(compressed, zero(T))
    gamma_cp = _scaled_cp(cs.sk, scale)
    one_m_fR = one(T)
    hRω = Complex{T}[]
    raman_gammas = ()
    raman_kernels = ()
    if raman === :agarwal
        one_m_fR = one(T) - raman_fraction
        hRω = agarwal_raman_response(nt, forward.dt, raman_fraction)
    elseif raman in (:model, :auto) && sim.include_Raman
        # fiber_raman, haw, hbw = raman_model(fiber2, sim, nt, forward.dt)
        fiber_raman, haw, hbw = raman_adjoint_model_kernels(fiber2, sim, nt, forward.dt)
        one_m_fR = one(T) - fiber_raman.fr
        cs_model = _compressed_bundle(compressed, fiber_raman.fr)
        gamma_a = _scaled_cp(cs_model.sra, scale)
        if cs_model.srb !== nothing && !isempty(hbw)
            gamma_b = _scaled_cp(cs_model.srb, scale)
            raman_gammas = (gamma_a, gamma_b)
            raman_kernels = (haw, hbw)
        else
            raman_gammas = (gamma_a,)
            raman_kernels = (haw,)
        end
    end

    zgrid = forward.z
    nsave = length(zgrid)
    u_tilde = Array{Complex{T},3}(undef, nt, nm, nsave)
    @inbounds for q in 1:nsave
        aw = inverse_fft(forward.fields[:, :, q], dims=1)
        u_tilde[:, :, q] = exp.(-d_op .* zgrid[q]) .* aw
    end

    lambda_tilde = exp.(-d_op .* zgrid[end]) .* Matrix{Complex{T}}(lambda_terminal)

    function rhs!(dlambda, lambda, _, z)
        u_z = _adjoint_interpolate_forward(u_tilde, zgrid, T(z), forward.ode_sol)
        dlambda .= _adjoint_rhs_kerr_compressed_rankchannels(lambda, u_z, T(z),
                                                             d_op, tau, gamma_cp,
                                                             one_m_fR, hRω;
                                                             raman_gammas=raman_gammas,
                                                             raman_kernels=raman_kernels)
        return nothing
    end

    saveat = if return_lambdaw_zsave
        if zsave_adj === nothing
            reverse(zgrid)
        else
            reverse(sort(T.(collect(zsave_adj))))
        end
    else
        [zgrid[1]]
    end
    prob = ODEProblem(rhs!, lambda_tilde, (zgrid[end], zgrid[1]))
    sol = DifferentialEquations.solve(prob, Vern9(); dt=dz_adj, adaptive=adaptive,
                abstol=abstol, reltol=reltol, saveat=saveat)
    sol.retcode == ReturnCode.Success ||
        error("Vern9 rank-channel compressed adjoint solve failed with retcode $(sol.retcode).")

    if !return_lambdaw_zsave
        lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, 1)
        lambdaw_saved[:, :, 1] = sol.u[end]
        return AdjointOutput{T}(; z=[zgrid[1]], lambdaw=lambdaw_saved,
                                dt=forward.dt, frame=:interaction)
    end

    z_saved_ascending = reverse(T.(sol.t))
    lambdaw_saved = Array{Complex{T},3}(undef, nt, nm, length(sol.u))
    @inbounds for (q, (z, state)) in enumerate(zip(z_saved_ascending, reverse(sol.u)))
        lambdaw_saved[:, :, q] = exp.(d_op .* z) .* state
    end
    return AdjointOutput{T}(; z=collect(z_saved_ascending), lambdaw=lambdaw_saved,
                            dt=forward.dt, frame=:lab)
end
