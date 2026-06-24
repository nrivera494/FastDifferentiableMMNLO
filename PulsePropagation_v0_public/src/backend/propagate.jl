"""
    propagate_ode(fiber, initial_condition, sim; zsave=nothing, reltol=1e-10,
                  abstol=1e-10, alg=Tsit5())

Integrate the passive multimode generalized NLSE as an interaction-picture ODE
using DifferentialEquations.jl. This mirrors the forward trajectory structure
used by `julia-adjoint-solver`: the ODE state is the lab-frame Fourier field
with the linear propagator factored out, while the returned `PropagationOutput`
stores lab-frame time-domain fields at the requested `zsave` locations.

This diagnostic/adjoint-oriented path currently supports passive propagation
only (`sim.gain_model == 0`).
"""
function propagate_ode(fiber::Fiber{T}, initial_condition::InitialCondition{T},
                       sim::Simulation{T};
                       zsave=nothing,
                       linear_gain=nothing,
                       reltol::T=T(1e-10),
                       abstol::T=T(1e-10),
                       alg=Tsit5()) where {T}
    sim.gain_model == 0 ||
        error("propagate_ode currently supports passive propagation only.")

    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, initial_condition.dt)
    srsk = calc_srsk(fiber3, sim, size(fiber3.sr, 1))
    at_noise = nothing

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
    gain = _canonical_linear_gain(linear_gain, nt, nm, fiber2.L0; zsave=save_z)

    u0_tilde = inverse_fft(fields0, dims=1)
    function rhs!(du_tilde, u_tilde, _, z)
        lin = _linear_exponent(d_op, T(z), gain)
        aw = exp.(lin) .* u_tilde
        nl = n_op_rk4ip(aw, sim, n2_prefactor, srsk, haw, hbw, at_noise)
        du_tilde .= exp.(-lin) .* nl
        return nothing
    end

    prob = ODEProblem(rhs!, u0_tilde, (zero(T), fiber2.L0))
    sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, dense=true)
    sol.retcode == ReturnCode.Success ||
        error("ODE forward solve failed with retcode $(sol.retcode).")

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    @inbounds for (q, z) in enumerate(save_z)
        u_tilde = Matrix{Complex{T}}(sol(z))
        aw = exp.(_linear_exponent(d_op, z, gain)) .* u_tilde
        out[:, :, q] = forward_fft(aw, dims=1)
    end
    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PropagationOutput{T}(; z=save_z, fields=out,
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=save_dz, t_delay=delays, ode_sol=sol,
                                linear_gain=gain)
end


function propagate_ode_cp(fiber::Fiber{T}, initial_condition::InitialCondition{T},
                       sim::Simulation{T}, srsk_cp;
                       zsave=nothing,
                       reltol::T=T(1e-10),
                       abstol::T=T(1e-10),
                       alg=Tsit5(),
                       dtmax=nothing,
                       fixed_dt=nothing) where {T}
    sim.gain_model == 0 ||
        error("propagate_ode currently supports passive propagation only.")

    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, initial_condition.dt)
    at_noise = zeros(Complex{T}, nt, nm)

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

    u0_tilde = inverse_fft(fields0, dims=1)
    function rhs!(du_tilde, u_tilde, _, z)
        aw = exp.(d_op .* T(z)) .* u_tilde
        nl = n_op_rk4ip_cp(aw, sim, n2_prefactor, srsk_cp, fiber3.fr, haw, hbw, at_noise)
        du_tilde .= exp.(-d_op .* T(z)) .* nl
        return nothing
    end

    prob = ODEProblem(rhs!, u0_tilde, (zero(T), fiber2.L0))
    solve_kwargs = if fixed_dt !== nothing
        (; reltol=reltol, abstol=abstol, dense=true,
         dt=T(fixed_dt), adaptive=false)
    elseif dtmax !== nothing
        (; reltol=reltol, abstol=abstol, dense=true, dtmax=T(dtmax))
    else
        (; reltol=reltol, abstol=abstol, dense=true)
    end
    sol = DifferentialEquations.solve(prob, alg; solve_kwargs...)
    sol.retcode == ReturnCode.Success ||
        error("ODE forward solve failed with retcode $(sol.retcode).")

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    @inbounds for (q, z) in enumerate(save_z)
        u_tilde = Matrix{Complex{T}}(sol(z))
        aw = exp.(d_op .* z) .* u_tilde
        out[:, :, q] = forward_fft(aw, dims=1)
    end
    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PropagationOutput{T}(; z=save_z, fields=out,
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=save_dz, t_delay=delays, ode_sol=sol)
end

function propagate_ode_cp_cached(fiber::Fiber{T}, initial_condition::InitialCondition{T},
                       sim::Simulation{T}, srsk_cp;
                       zsave=nothing,
                       reltol::T=T(1e-10),
                       abstol::T=T(1e-10),
                       alg=Tsit5(),
                       dtmax=nothing,
                       fixed_dt=nothing,
                       fftw_threads::Integer=1,
                       fftw_flags=FFTW.MEASURE,
                       blas_threads=nothing) where {T}
    sim.gain_model == 0 ||
        error("propagate_ode_cp_cached currently supports passive propagation only.")

    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, initial_condition.dt)
    at_noise = zeros(Complex{T}, nt, nm)

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

    old_fftw_threads = FFTW.get_num_threads()
    cache = try
        FFTW.set_num_threads(Int(fftw_threads))
        CPFastRHSCache(srsk_cp, nt; T=T, flags=fftw_flags)
    finally
        FFTW.set_num_threads(old_fftw_threads)
    end

    u0_tilde = inverse_fft(fields0, dims=1)
    aw = similar(u0_tilde)
    nl = similar(u0_tilde)
    function rhs!(du_tilde, u_tilde, _, z)
        zT = T(z)
        @. aw = exp(d_op * zT) * u_tilde
        n_op_rk4ip_cp_cached!(nl, aw, sim, n2_prefactor, cache,
                              fiber3.fr, haw, hbw, at_noise)
        @. du_tilde = exp(-d_op * zT) * nl
        return nothing
    end

    prob = ODEProblem(rhs!, u0_tilde, (zero(T), fiber2.L0))
    save_kwargs = (; saveat=save_z, save_everystep=false, dense=false)
    solve_kwargs = if fixed_dt !== nothing
        (; reltol=reltol, abstol=abstol, save_kwargs...,
         dt=T(fixed_dt), adaptive=false)
    elseif dtmax !== nothing
        (; reltol=reltol, abstol=abstol, save_kwargs..., dtmax=T(dtmax))
    else
        (; reltol=reltol, abstol=abstol, save_kwargs...)
    end
    old_blas_threads = blas_threads === nothing ? nothing : BLAS.get_num_threads()
    sol = try
        blas_threads === nothing || BLAS.set_num_threads(Int(blas_threads))
        DifferentialEquations.solve(prob, alg; solve_kwargs...)
    finally
        old_blas_threads === nothing || BLAS.set_num_threads(old_blas_threads)
    end
    sol.retcode == ReturnCode.Success ||
        error("ODE forward solve failed with retcode $(sol.retcode).")

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    length(sol.u) == length(save_z) ||
        error("ODE forward solve saved $(length(sol.u)) states, expected $(length(save_z)).")
    @inbounds for q in eachindex(save_z)
        z = save_z[q]
        u_tilde = sol.u[q]
        aw .= exp.(d_op .* z) .* u_tilde
        out[:, :, q] = forward_fft(aw, dims=1)
    end
    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PropagationOutput{T}(; z=save_z, fields=out,
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=save_dz, t_delay=delays, ode_sol=sol)
end


function _rk4ip_cp_cached_step!(a1w::AbstractMatrix{Complex{T}},
                                a0w::AbstractMatrix{Complex{T}},
                                dz::T,
                                half::AbstractMatrix{Complex{T}},
                                a_ip::AbstractMatrix{Complex{T}},
                                stage::AbstractMatrix{Complex{T}},
                                k1::AbstractMatrix{Complex{T}},
                                k2::AbstractMatrix{Complex{T}},
                                k3::AbstractMatrix{Complex{T}},
                                k4::AbstractMatrix{Complex{T}},
                                sim::Simulation{T},
                                n2_prefactor::AbstractVector{Complex{T}},
                                cache::CPFastRHSCache{T},
                                f_raman::T,
                                haw::AbstractVector{Complex{T}},
                                hbw::AbstractVector{Complex{T}},
                                at_noise::Union{Nothing,AbstractMatrix{Complex{T}}}=nothing) where {T}
    @. a_ip = half * a0w

    n_op_rk4ip_cp_cached!(k1, a0w, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)
    @. k1 = half * k1

    @. stage = a_ip + k1 * (dz / 2)
    n_op_rk4ip_cp_cached!(k2, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    @. stage = a_ip + k2 * (dz / 2)
    n_op_rk4ip_cp_cached!(k3, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    @. stage = half * (a_ip + k3 * dz)
    n_op_rk4ip_cp_cached!(k4, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    @. a1w = half * (a_ip + (k1 + 2 * k2 + 2 * k3) * (dz / 6)) + k4 * (dz / 6)
    return a1w
end

function _fixed_step_count(distance::T, dz::T; label="save interval") where {T}
    n = round(Int, distance / dz)
    isapprox(T(n) * dz, distance; rtol=T(1e-10), atol=T(1e-12)) ||
        error("$label $(distance) is not an integer multiple of dz=$(dz).")
    return n
end

function propagate_rk4ip_cp_cached(fiber::Fiber{T}, initial_condition::InitialCondition{T},
                                   sim::Simulation{T}, srsk_cp;
                                   zsave=nothing,
                                   dz=nothing,
                                   fftw_threads::Integer=1,
                                   fftw_flags=FFTW.MEASURE,
                                   blas_threads=nothing) where {T}
    sim.gain_model == 0 ||
        error("propagate_rk4ip_cp_cached currently supports passive propagation only.")

    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    step_dz = dz === nothing ? sim.dz : T(dz)
    step_dz > zero(T) || error("dz must be positive.")
    omega = ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, initial_condition.dt)
    at_noise = nothing

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

    half = exp.(d_op .* (step_dz / 2))
    aw = inverse_fft(fields0, dims=1)
    next_aw = similar(aw)
    a_ip = similar(aw)
    stage = similar(aw)
    k1 = similar(aw)
    k2 = similar(aw)
    k3 = similar(aw)
    k4 = similar(aw)

    old_fftw_threads = FFTW.get_num_threads()
    cache = try
        FFTW.set_num_threads(Int(fftw_threads))
        CPFastRHSCache(srsk_cp, nt; T=T, flags=fftw_flags)
    finally
        FFTW.set_num_threads(old_fftw_threads)
    end

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    out[:, :, 1] = fields0

    old_blas_threads = BLAS.get_num_threads()
    try
        blas_threads === nothing || BLAS.set_num_threads(Int(blas_threads))
        zprev = zero(T)
        @inbounds for q in 2:length(save_z)
            znext = save_z[q]
            nsteps = _fixed_step_count(znext - zprev, step_dz; label="save interval")
            for _ in 1:nsteps
                _rk4ip_cp_cached_step!(next_aw, aw, step_dz, half, a_ip, stage,
                                       k1, k2, k3, k4, sim, n2_prefactor, cache,
                                       fiber3.fr, haw, hbw, at_noise)
                aw, next_aw = next_aw, aw
            end
            out[:, :, q] = forward_fft(aw, dims=1)
            zprev = znext
        end
    finally
        blas_threads === nothing || BLAS.set_num_threads(old_blas_threads)
    end

    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PropagationOutput{T}(; z=save_z, fields=out,
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=save_dz, t_delay=delays)
end


function propagate_ode_compressed(fiber::Fiber{T}, initial_condition::InitialCondition{T},
                                  sim::Simulation{T}, compressed;
                                  zsave=nothing,
                                  reltol::T=T(1e-10),
                                  abstol::T=T(1e-10),
                                  alg=Tsit5()) where {T}
    sim.gain_model == 0 ||
        error("propagate_ode_compressed currently supports passive propagation only.")

    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = expand_betas_for_polarization(fiber, sim, nm)
    omega = ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = raman_model(fiber2, sim, nt, initial_condition.dt)
    at_noise = zeros(Complex{T}, nt, nm)

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

    u0_tilde = inverse_fft(fields0, dims=1)
    function rhs!(du_tilde, u_tilde, _, z)
        aw = exp.(d_op .* T(z)) .* u_tilde
        nl = n_op_rk4ip_compressed(aw, sim, n2_prefactor, compressed,
                                    fiber3.fr, haw, hbw, at_noise)
        du_tilde .= exp.(-d_op .* T(z)) .* nl
        return nothing
    end

    prob = ODEProblem(rhs!, u0_tilde, (zero(T), fiber2.L0))
    sol = DifferentialEquations.solve(prob, alg; reltol=reltol, abstol=abstol, dense=true)
    sol.retcode == ReturnCode.Success ||
        error("ODE forward solve failed with retcode $(sol.retcode).")

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    @inbounds for (q, z) in enumerate(save_z)
        u_tilde = Matrix{Complex{T}}(sol(z))
        aw = exp.(d_op .* z) .* u_tilde
        out[:, :, q] = forward_fft(aw, dims=1)
    end
    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PropagationOutput{T}(; z=save_z, fields=out,
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=save_dz, t_delay=delays, ode_sol=sol)
end
