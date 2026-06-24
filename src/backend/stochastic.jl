"""
    photon_vacuum_ensemble(mean_uω, dt, f0, ntraj; kwargs...)

Construct an ensemble of unshifted Fourier-domain initial conditions
`uω = mean_uω + δuω` from complex Gaussian noise in photon-bin coordinates.

`mean_uω` must have size `Nt x Nm` and use the same unshifted Fourier
convention as `inverse_fft(time_domain_field, dims=1)`.  The sampled
photon-coordinate noise is

```
δBω = noise_scale * (randn() + im * randn())
```

with `noise_scale = 1/sqrt(2)` by default, so `E|δBω|^2 = 1`.
It is mapped back to solver units by `δuω = δBω / sqrt(weight[k])`.
Bins with nonpositive lab frequency have zero photon weight and receive no
noise.
"""
function photon_vacuum_ensemble(mean_uω::AbstractMatrix{Complex{T}},
                                dt::T,
                                f0::T,
                                ntraj::Integer;
                                rng=nothing,
                                shifted::Bool=false,
                                noise_scale::T=inv(sqrt(T(2)))) where {T}
    ntraj > 0 || error("ntraj must be positive.")
    nt, nm = size(mean_uω)
    weights = T.(photon_bin_weights(nt, dt, f0; shifted=shifted))
    out = Array{Complex{T},3}(undef, nt, nm, ntraj)
    @inbounds for q in 1:ntraj
        out[:, :, q] = mean_uω
        for m in 1:nm, k in 1:nt
            w = weights[k]
            if w > zero(T)
                δB = noise_scale * Complex{T}(_stochastic_randn(rng),
                                              _stochastic_randn(rng))
                out[k, m, q] += δB / sqrt(w)
            end
        end
    end
    return out
end

_stochastic_randn(rng) = rng === nothing ? randn() : randn(rng)

"""
    raman_langevin_spectrum(nt, dt, f0, hRω, gamma_eff; temperature_K=300)

Return the unshifted finite-grid Raman Langevin spectrum `Cω` used for
spontaneous Raman driving.  The normalization matches the adjoint Raman
variance contraction

```
gamma_eff^2 * dz * df_Hz * sum(Cω .* abs2.(fft(Bt)))
```

for a time-domain test function `Bt`.  `hRω` must use the same unshifted
Fourier ordering as `ifft_frequency_grid(nt, dt)` and should already include
the fractional Raman strength.
"""
function raman_langevin_spectrum(nt::Integer,
                                 dt::T,
                                 f0::T,
                                 hRω::AbstractVector{Complex{T}},
                                 gamma_eff::T;
                                 temperature_K::T=T(300.0)) where {T}
    length(hRω) == nt || error("hRω must have length nt.")
    gamma_eff > zero(T) || error("gamma_eff must be positive.")
    omega = ifft_frequency_grid(nt, dt)
    hbar = T(1.054571817e-34)
    kb = T(1.380649e-23)
    cω = zeros(T, nt)
    @inbounds for k in 1:nt
        kabs = k <= (nt ÷ 2 + 1) ? k : nt - k + 2
        omega_abs_si = abs(omega[kabs]) * T(1e12)
        nω = if omega_abs_si == zero(T)
            zero(T)
        else
            x = hbar * omega_abs_si / (kb * temperature_K)
            x > T(500) ? zero(T) : inv(exp(x) - one(T))
        end
        theta = omega[k] >= T(-1e-10) ? one(T) : zero(T)
        cω[k] = (hbar / gamma_eff) * (T(2π) * f0 * T(1e12)) *
                abs(imag(hRω[kabs])) * (T(2) * nω + theta)
    end
    return cω
end

function raman_langevin_spectrum(nt::Integer,
                                 dt::T,
                                 f0::T,
                                 gamma_eff::T;
                                 raman_fraction::T=T(0.18),
                                 temperature_K::T=T(300.0)) where {T}
    hRω = agarwal_raman_response(nt, dt, raman_fraction)
    return raman_langevin_spectrum(nt, dt, f0, hRω, gamma_eff;
                                   temperature_K=temperature_K)
end

"""
    sample_raman_langevin_noise(Cω, dz, dt; rng=nothing)

Sample one finite-step Raman Langevin realization on the unshifted time grid.
If `ηω = fft(ηt)`, then

```
E abs2(ηω[k]) = Nt^2 * dz * df_Hz * Cω[k]
```

with `df_Hz = 1 / (Nt * dt_ps) * 1e12`.  This is the discrete version of a
`δ(z-z')` Langevin correlation: the regularized per-unit-length noise contains
the usual `1/sqrt(dz)`, while the finite field increment uses `sqrt(dz)`.
The Raman spectrum is generally asymmetric, so this sampler intentionally uses
complex circular Fourier noise rather than enforcing a real time-domain process.
"""
function sample_raman_langevin_noise(Cω::AbstractVector{T},
                                     dz::T,
                                     dt::T;
                                     rng=nothing,
                                     noise_scale::T=one(T)) where {T}
    nt = length(Cω)
    dz >= zero(T) || error("dz must be nonnegative.")
    df_hz = inv(T(nt) * dt) * T(1e12)
    ηω = zeros(Complex{T}, nt)
    @inbounds for k in 1:nt
        c = Cω[k]
        c > zero(T) || continue
        ξ = Complex{T}(_stochastic_randn(rng), _stochastic_randn(rng)) / sqrt(T(2))
        ηω[k] = noise_scale * T(nt) * sqrt(dz * df_hz * c) * ξ
    end
    return inverse_fft(ηω, dims=1)
end

"""
    sample_real_raman_langevin_noise(Cω, dz, dt; rng=nothing, noise_scale=1)

Sample a real time-domain Raman Langevin increment.  Frequency bins are drawn
in Hermitian pairs using the symmetrized pair spectrum `(Cω[k] + Cω[-k]) / 2`.
For real test functions this reproduces the same discrete variance
`dz * df_Hz * sum(Cω .* abs2.(fft(Bt)))`, while avoiding the unphysical
terminal amplitude kick produced by circular complex time-domain noise in
`δA = im * gamma * A * η`.
"""
function sample_real_raman_langevin_noise(Cω::AbstractVector{T},
                                          dz::T,
                                          dt::T;
                                          rng=nothing,
                                          noise_scale::T=one(T)) where {T}
    nt = length(Cω)
    dz >= zero(T) || error("dz must be nonnegative.")
    df_hz = inv(T(nt) * dt) * T(1e12)
    ηω = zeros(Complex{T}, nt)
    used = falses(nt)
    @inbounds for k in 1:nt
        used[k] && continue
        kp = k == 1 ? 1 : nt - k + 2
        if kp == k
            c = max(real(Cω[k]), zero(T))
            ηω[k] = noise_scale * T(nt) * sqrt(dz * df_hz * c) *
                    Complex{T}(_stochastic_randn(rng), 0)
        else
            c = max((real(Cω[k]) + real(Cω[kp])) / T(2), zero(T))
            ξ = Complex{T}(_stochastic_randn(rng), _stochastic_randn(rng)) / sqrt(T(2))
            ηω[k] = noise_scale * T(nt) * sqrt(dz * df_hz * c) * ξ
            ηω[kp] = conj(ηω[k])
            used[kp] = true
        end
        used[k] = true
    end
    return real.(inverse_fft(ηω, dims=1))
end

function _stochastic_zsave(fiber::Fiber{T}, sim::Simulation{T}, zsave) where {T}
    save_z = if zsave === nothing
        save_period = sim.save_period == 0 ? fiber.L0 : sim.save_period
        collect(T, 0:round(Int, fiber.L0 / save_period)) .* save_period
    else
        T.(collect(zsave))
    end
    isempty(save_z) && error("zsave must contain at least one point.")
    first(save_z) == zero(T) || error("zsave must start at 0.")
    last(save_z) == fiber.L0 || error("zsave must end at fiber.L0.")
    all(diff(save_z) .> zero(T)) || error("zsave must be strictly increasing.")
    return save_z
end

function _sample_diagonal_kick(correlation, dz, rng)
    T = typeof(real(one(eltype(correlation))))
    kick = zeros(Complex{T}, size(correlation))
    @inbounds for i in eachindex(correlation)
        c = real(correlation[i])
        c > zero(T) || continue
        ξ = Complex{T}(_stochastic_randn(rng), _stochastic_randn(rng)) / sqrt(T(2))
        # The continuum convention is <N(z)N*(z')> = C δ(z-z').
        # Over a finite step, ∫N dz has variance C*dz. Equivalently,
        # the regularized per-step noise strength contains the familiar
        # 1/sqrt(dz) from δ(0), and the actual field increment is sqrt(C*dz)*ξ.
        kick[i] = sqrt(c * dz) * ξ
    end
    return kick
end

function _as_kick_array(kicks, nt::Int, nm::Int, nsteps::Int, ntraj::Int)
    kicks === nothing && return nothing
    size(kicks) == (nt, nm, nsteps, ntraj) ||
        error("kicks must have size Nt x Nm x Nsteps x Ntraj.")
    return kicks
end

function _driven_correlations_to_kicks(correlations, nt::Int, nm::Int,
                                       nsteps::Int, ntraj::Int, dz, rng)
    correlations === nothing && return nothing
    size(correlations) == (nt, nm) ||
        error("diagonal drive correlations must have size Nt x Nm.")
    T = typeof(real(one(eltype(correlations))))
    kicks = Array{Complex{T},4}(undef, nt, nm, nsteps, ntraj)
    @inbounds for q in 1:ntraj, s in 1:nsteps
        kicks[:, :, s, q] = _sample_diagonal_kick(correlations, dz, rng)
    end
    return kicks
end

function _propagate_ode_driven(fiber::Fiber{T},
                               uω0::AbstractMatrix{Complex{T}},
                               dt::T,
                               sim::Simulation{T},
                               save_z::AbstractVector{T},
                               kicks::Union{Nothing,Array{Complex{T},4}},
                               traj_index::Integer;
                               linear_gain=nothing,
                               linear_gain_noise::Bool=false,
                               gain_noise_scale::T=T(0.5),
                               rng=nothing,
                               reltol::T=T(1e-10),
                               abstol::T=T(1e-10),
                               alg=Tsit5(),
                               deterministic_stepper::Symbol=:ode) where {T}
    sim.gain_model == 0 ||
        error("stochastic driven propagation currently supports passive propagation only.")
    nt, nm = size(uω0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, dt, omega, zeros(Complex{T}, nt, nm))
    gain = _canonical_linear_gain(linear_gain, nt, nm, fiber2.L0; zsave=save_z)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, dt)
    srsk = calc_srsk(fiber3, sim, size(fiber3.sr, 1))
    at_noise = zeros(Complex{T}, nt, nm)

    function rhs!(du_tilde, u_tilde, _, z)
        lin = _linear_exponent(d_op, T(z), gain)
        aw = exp.(lin) .* u_tilde
        nl = n_op_rk4ip(aw, sim, n2_prefactor, srsk, haw, hbw, at_noise)
        du_tilde .= exp.(-lin) .* nl
        return nothing
    end

    nsteps = length(save_z) - 1
    fields = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    fields[:, :, 1] = forward_fft(uω0, dims=1)
    use_rk4ip = deterministic_stepper === :rk4ip
    use_ode_integrator = deterministic_stepper === :ode_integrator
    deterministic_stepper in (:ode, :ode_integrator, :rk4ip) ||
        error("deterministic_stepper must be :ode, :ode_integrator, or :rk4ip.")
    if use_rk4ip
        dzs = diff(save_z)
        all(isapprox.(dzs, sim.dz; rtol=sqrt(eps(T)), atol=sqrt(eps(T)))) ||
            error("RK4IP stochastic stepping requires zsave spacing equal to sim.dz.")
    end
    u_tilde = copy(uω0)
    aw_fixed = copy(uω0)
    a5 = nothing
    integrator = if use_ode_integrator
        prob = ODEProblem(rhs!, u_tilde, (save_z[1], save_z[end]))
        DifferentialEquations.init(prob, alg; reltol=reltol, abstol=abstol, save_everystep=false)
    else
        nothing
    end
    @inbounds for s in 1:nsteps
        z0 = save_z[s]
        z1 = save_z[s + 1]
        if use_rk4ip
            d_step = _linear_gain_step_operator(d_op, gain, z0, z1, nt, nm)
            aw_fixed, a5, ok = step_rk4ip_nogain(aw_fixed, sim, n2_prefactor,
                                                 srsk, d_step, haw, hbw,
                                                 at_noise, a5)
            ok || error("Driven RK4IP step failed at step $s.")
            if kicks !== nothing
                aw_fixed .+= kicks[:, :, s, traj_index]
                a5 = nothing
            end
            if linear_gain_noise && gain !== nothing
                aw_fixed .+= _sample_linear_gain_kick(gain, z0, z1, nt, nm, dt,
                                                      sim.f0, rng=rng,
                                                      noise_scale=gain_noise_scale)
                a5 = nothing
            end
            fields[:, :, s + 1] = forward_fft(aw_fixed, dims=1)
        elseif use_ode_integrator
            DifferentialEquations.add_tstop!(integrator, z1)
            while integrator.t < z1
                DifferentialEquations.step!(integrator)
            end
            u_tilde = integrator.u
            if kicks !== nothing
                u_tilde .+= exp.(-_linear_exponent(d_op, z1, gain)) .* kicks[:, :, s, traj_index]
                DifferentialEquations.u_modified!(integrator, true)
            end
            if linear_gain_noise && gain !== nothing
                δaw_gain = _sample_linear_gain_kick(gain, z0, z1, nt, nm, dt,
                                                    sim.f0, rng=rng,
                                                    noise_scale=gain_noise_scale)
                u_tilde .+= exp.(-_linear_exponent(d_op, z1, gain)) .* δaw_gain
                DifferentialEquations.u_modified!(integrator, true)
            end
            fields[:, :, s + 1] = forward_fft(exp.(_linear_exponent(d_op, z1, gain)) .* u_tilde, dims=1)
        else
            prob = ODEProblem(rhs!, u_tilde, (z0, z1))
            sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, save_everystep=false)
            sol.retcode == ReturnCode.Success ||
                error("Driven ODE step failed with retcode $(sol.retcode).")
            u_tilde = Matrix{Complex{T}}(sol.u[end])
            if kicks !== nothing
                # Kicks are specified as lab-frame Fourier increments at z1.
                # The ODE state is interaction-picture, so map the lab kick into
                # the interaction picture before adding it.
                u_tilde .+= exp.(-_linear_exponent(d_op, z1, gain)) .* kicks[:, :, s, traj_index]
            end
            if linear_gain_noise && gain !== nothing
                δaw_gain = _sample_linear_gain_kick(gain, z0, z1, nt, nm, dt,
                                                    sim.f0, rng=rng,
                                                    noise_scale=gain_noise_scale)
                u_tilde .+= exp.(-_linear_exponent(d_op, z1, gain)) .* δaw_gain
            end
            fields[:, :, s + 1] = forward_fft(exp.(_linear_exponent(d_op, z1, gain)) .* u_tilde, dims=1)
        end
    end
    save_dz = zeros(T, length(save_z))
    save_dz[2:end] .= diff(save_z)
    return PropagationOutput{T}(; z=collect(save_z), fields=fields, dt=dt,
                                betas=sim_betas, dz=save_dz,
                                t_delay=zeros(T, length(save_z)),
                                linear_gain=gain)
end

function _linear_gain_step_operator(d_op::AbstractMatrix{Complex{T}},
                                    gain::Union{Nothing,LinearGain{T}},
                                    z0::T, z1::T,
                                    nt::Integer, nm::Integer) where {T}
    gain === nothing && return d_op
    dz = z1 - z0
    dz > zero(T) || error("RK4IP stochastic stepping requires positive dz.")
    dg = _linear_gain_power_integral(gain, z1, nt, nm) .-
         _linear_gain_power_integral(gain, z0, nt, nm)
    return d_op .+ Complex{T}.((dg ./ dz) ./ 2)
end

function _sample_linear_gain_kick(gain::LinearGain{T}, z0::T, z1::T,
                                  nt::Integer, nm::Integer, dt::T, f0::T;
                                  rng=nothing,
                                  noise_scale::T=T(0.5)) where {T}
    noise_scale == zero(T) && return zeros(Complex{T}, nt, nm)
    weights = T.(photon_bin_weights(nt, dt, f0; shifted=false))
    G = exp.(_linear_gain_power_integral(gain, z1, nt, nm) .-
            _linear_gain_power_integral(gain, z0, nt, nm))
    kick = zeros(Complex{T}, nt, nm)
    @inbounds for m in 1:nm, k in 1:nt
        weights[k] > zero(T) || continue
        added = max(G[k, m] - one(T), zero(T))
        added > zero(T) || continue
        δB = noise_scale * sqrt(added) *
             Complex{T}(_stochastic_randn(rng), _stochastic_randn(rng))
        kick[k, m] = δB / sqrt(weights[k])
    end
    return kick
end

function _default_gamma_eff(fiber::Fiber{T}, sim::Simulation{T}, nm::Integer) where {T}
    nm == 1 ||
        error("A scalar gamma_eff must be supplied for multimode Raman stochastic driving.")
    c = T(2.99792458e-4)
    return fiber.n2 * (T(2π) * sim.f0) / c * fiber.sr[1, 1, 1, 1]
end

function _propagate_ode_raman_driven(fiber::Fiber{T},
                                     uω0::AbstractMatrix{Complex{T}},
                                     dt::T,
                                     sim::Simulation{T},
                                     save_z::AbstractVector{T},
                                     Cω::AbstractVector{T},
                                     gamma_eff::T,
                                     traj_index::Integer;
                                     rng=nothing,
                                     reltol::T=T(1e-10),
                                     abstol::T=T(1e-10),
                                     alg=Tsit5(),
                                     include_self_steepening::Bool=false,
                                     keep_noise::Bool=false,
                                     noise_scale::T=one(T),
                                     deterministic_stepper::Symbol=:ode) where {T}
    sim.gain_model == 0 ||
        error("Raman stochastic propagation currently supports passive propagation only.")
    nt, nm = size(uω0)
    nm == 1 || error("Raman stochastic propagation currently supports scalar single-mode fields only.")
    length(Cω) == nt || error("Cω must have length Nt.")

    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, dt, omega, zeros(Complex{T}, nt, nm))
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, dt)
    srsk = calc_srsk(fiber3, sim, size(fiber3.sr, 1))
    at_noise = zeros(Complex{T}, nt, nm)
    shock = include_self_steepening ? Complex{T}.((omega .+ T(2π) * sim.f0) ./ (T(2π) * sim.f0)) :
            ones(Complex{T}, nt)

    function rhs!(du_tilde, u_tilde, _, z)
        aw = exp.(d_op .* T(z)) .* u_tilde
        nl = n_op_rk4ip(aw, sim, n2_prefactor, srsk, haw, hbw, at_noise)
        du_tilde .= exp.(-d_op .* T(z)) .* nl
        return nothing
    end

    nsteps = length(save_z) - 1
    fields = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    noises = keep_noise ? Array{Complex{T},3}(undef, nt, nm, nsteps) : nothing
    fields[:, :, 1] = forward_fft(uω0, dims=1)
    use_rk4ip = deterministic_stepper === :rk4ip
    deterministic_stepper in (:ode, :rk4ip) ||
        error("deterministic_stepper must be :ode or :rk4ip.")
    u_tilde = copy(uω0)
    aw_fixed = copy(uω0)
    a5 = nothing
    @inbounds for s in 1:nsteps
        z0 = save_z[s]
        z1 = save_z[s + 1]
        dz = z1 - z0
        aw_lab = if use_rk4ip
            isapprox(dz, sim.dz; rtol=sqrt(eps(T)), atol=sqrt(eps(T))) ||
                error("RK4IP Raman stochastic stepping requires zsave spacing equal to sim.dz.")
            aw_fixed, a5, ok = step_rk4ip_nogain(aw_fixed, sim, n2_prefactor,
                                                 srsk, d_op, haw, hbw,
                                                 at_noise, a5)
            ok || error("RK4IP Raman stochastic step failed at step $s.")
            aw_fixed
        else
            prob = ODEProblem(rhs!, u_tilde, (z0, z1))
            sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, save_everystep=false)
            sol.retcode == ReturnCode.Success ||
                error("Raman driven ODE step failed with retcode $(sol.retcode).")
            u_tilde = Matrix{Complex{T}}(sol.u[end])
            exp.(d_op .* z1) .* u_tilde
        end
        at_lab = forward_fft(aw_lab, dims=1)
        ηt = sample_real_raman_langevin_noise(Cω, dz, dt; rng=rng,
                                              noise_scale=noise_scale)
        noises !== nothing && (noises[:, 1, s] = ηt)

        δat = Complex{T}(1im) .* gamma_eff .* at_lab[:, 1] .* ηt
        δaw = inverse_fft(reshape(δat, nt, 1), dims=1)
        δaw .*= reshape(shock, nt, 1)
        if use_rk4ip
            aw_fixed .+= δaw
            a5 = nothing
            fields[:, :, s + 1] = forward_fft(aw_fixed, dims=1)
        else
            u_tilde .+= exp.(-d_op .* z1) .* δaw
            fields[:, :, s + 1] = forward_fft(exp.(d_op .* z1) .* u_tilde, dims=1)
        end
    end
    save_dz = zeros(T, length(save_z))
    save_dz[2:end] .= diff(save_z)
    out = PropagationOutput{T}(; z=collect(save_z), fields=fields, dt=dt,
                               betas=sim_betas, dz=save_dz,
                               t_delay=zeros(T, length(save_z)))
    return out, noises
end

"""
    stochastic_solve(fiber, uω_ensemble, dt, sim; kwargs...)

Propagate an ensemble of unshifted Fourier-domain initial fields.  The input
has size `Nt x Nm x Ntraj`, and the returned `fields` array has size
`Nt x Nm x Nz x Ntraj`, where each `fields[:, :, :, q]` is the same
time-domain output convention as `propagate_ode(...).fields`.

When neither `kicks` nor `drive_correlations` is supplied, each trajectory is
propagated by `propagate_ode`.  If stochastic driving is supplied, the routine
uses fixed steps on the requested `zsave` grid.  `kicks`, when supplied, are
actual lab-frame Fourier increments with size `Nt x Nm x (Nz-1) x Ntraj`.
`drive_correlations`, when supplied, is a diagonal per-unit-length correlation
array with size `Nt x Nm`, sampled as kicks with variance `C*dz`.
"""
function stochastic_solve(fiber::Fiber{T},
                          uω_ensemble::Array{Complex{T},3},
                          dt::T,
                          sim::Simulation{T};
                          zsave=nothing,
                          linear_gain=nothing,
                          linear_gain_noise::Bool=false,
                          gain_noise_scale::T=T(0.5),
                          kicks=nothing,
                          drive_correlations=nothing,
                          rng=nothing,
                          reltol::T=T(1e-10),
                          abstol::T=T(1e-10),
                          alg=Tsit5(),
                          deterministic_stepper::Symbol=:ode) where {T}
    nt, nm, ntraj = size(uω_ensemble)
    save_z = _stochastic_zsave(fiber, sim, zsave)
    nsteps = length(save_z) - 1
    actual_kicks = _as_kick_array(kicks, nt, nm, nsteps, ntraj)
    if drive_correlations !== nothing
        actual_kicks === nothing || error("Provide either kicks or drive_correlations, not both.")
        dzs = diff(save_z)
        all(isapprox.(dzs, dzs[1]; rtol=sqrt(eps(T)), atol=sqrt(eps(T)))) ||
            error("drive_correlations currently requires a uniform zsave grid.")
        actual_kicks = _driven_correlations_to_kicks(T.(drive_correlations), nt, nm,
                                                     nsteps, ntraj, dzs[1], rng)
    end

    out_fields = Array{Complex{T},4}(undef, nt, nm, length(save_z), ntraj)
    if actual_kicks === nothing
        @inbounds for q in 1:ntraj
            fields0 = forward_fft(uω_ensemble[:, :, q], dims=1)
            ic = InitialCondition{T}(; dt=dt, fields=reshape(fields0, nt, nm, 1))
            if linear_gain_noise && linear_gain !== nothing
                out = _propagate_ode_driven(fiber, uω_ensemble[:, :, q], dt, sim,
                                            save_z, nothing, q;
                                            linear_gain=linear_gain,
                                            linear_gain_noise=true,
                                            gain_noise_scale=gain_noise_scale,
                                            rng=rng,
                                            reltol=reltol, abstol=abstol, alg=alg,
                                            deterministic_stepper=deterministic_stepper)
            else
                out = propagate_ode(fiber, ic, sim; zsave=save_z,
                                    linear_gain=linear_gain, reltol=reltol,
                                    abstol=abstol, alg=alg)
            end
            out_fields[:, :, :, q] = out.fields
        end
    else
        @inbounds for q in 1:ntraj
            out = _propagate_ode_driven(fiber, uω_ensemble[:, :, q], dt, sim,
                                        save_z, actual_kicks, q;
                                        linear_gain=linear_gain,
                                        linear_gain_noise=linear_gain_noise,
                                        gain_noise_scale=gain_noise_scale,
                                        rng=rng,
                                        reltol=reltol, abstol=abstol, alg=alg,
                                        deterministic_stepper=deterministic_stepper)
            out_fields[:, :, :, q] = out.fields
        end
    end
    return (; z=save_z, fields=out_fields, kicks=actual_kicks)
end

"""
    stochastic_solve_raman(fiber, uω_ensemble, dt, sim; kwargs...)

Propagate an ensemble with multiplicative spontaneous Raman Langevin kicks.
This first implementation is scalar single-mode and passive.  `uω_ensemble`
has size `Nt x 1 x Ntraj` in the same unshifted lab-frame Fourier convention
as `inverse_fft(initial_time_field, dims=1)`.

The deterministic evolution over each `zsave` interval is solved with
`propagate_ode`'s interaction-picture RHS.  At the end of each interval a
finite-step Raman kick is applied in the lab frame:

```
δA(t) = im * gamma_eff * A(t) * η(t)
```

where `η` is a real time-domain Langevin process with Hermitian Fourier
coefficients sampled from the symmetrized positive/negative frequency pair
spectrum.  This reproduces the adjoint Raman variance contraction for real
observables and avoids a direct terminal amplitude kick from an imaginary
time-domain noise quadrature.
"""
function stochastic_solve_raman(fiber::Fiber{T},
                                uω_ensemble::Array{Complex{T},3},
                                dt::T,
                                sim::Simulation{T};
                                zsave=nothing,
                                gamma_eff=nothing,
                                Cω=nothing,
                                hRω=nothing,
                                raman_fraction=nothing,
                                temperature_K::T=T(300.0),
                                rng=nothing,
                                reltol::T=T(1e-10),
                                abstol::T=T(1e-10),
                                alg=Tsit5(),
                                include_self_steepening::Bool=false,
                                keep_noise::Bool=false,
                                noise_scale::T=one(T),
                                deterministic_stepper::Symbol=:ode) where {T}
    nt, nm, ntraj = size(uω_ensemble)
    nm == 1 || error("stochastic_solve_raman currently supports scalar single-mode fields only.")
    save_z = _stochastic_zsave(fiber, sim, zsave)
    γ = gamma_eff === nothing ? _default_gamma_eff(fiber, sim, nm) : T(gamma_eff)
    fr = raman_fraction === nothing ? (fiber.fr == zero(T) ? T(0.18) : fiber.fr) : T(raman_fraction)
    spectrum = if Cω !== nothing
        T.(Cω)
    elseif hRω !== nothing
        raman_langevin_spectrum(nt, dt, sim.f0, Complex{T}.(hRω), γ;
                                temperature_K=temperature_K)
    else
        raman_langevin_spectrum(nt, dt, sim.f0, γ; raman_fraction=fr,
                                temperature_K=temperature_K)
    end

    out_fields = Array{Complex{T},4}(undef, nt, nm, length(save_z), ntraj)
    noise_out = keep_noise ? Array{Complex{T},4}(undef, nt, nm, length(save_z) - 1, ntraj) : nothing
    @inbounds for q in 1:ntraj
        out, noises = _propagate_ode_raman_driven(fiber, uω_ensemble[:, :, q],
                                                  dt, sim, save_z, spectrum, γ, q;
                                                  rng=rng, reltol=reltol,
                                                  abstol=abstol, alg=alg,
                                                  include_self_steepening=include_self_steepening,
                                                  keep_noise=keep_noise,
                                                  noise_scale=noise_scale,
                                                  deterministic_stepper=deterministic_stepper)
        out_fields[:, :, :, q] = out.fields
        noise_out !== nothing && (noise_out[:, :, :, q] = noises)
    end
    return (; z=save_z, fields=out_fields, Cω=spectrum, gamma_eff=γ,
            raman_noise=noise_out)
end
