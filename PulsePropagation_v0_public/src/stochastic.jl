abstract type AbstractInitialNoise end
abstract type AbstractStochasticDrive end

Base.@kwdef struct ShotNoise{T} <: AbstractInitialNoise
    noise_scale::T = T(0.5)
end

ShotNoise(; noise_scale=0.5) =
    ShotNoise{typeof(float(noise_scale))}(; noise_scale=float(noise_scale))

Base.@kwdef struct RamanLangevin{T} <: AbstractStochasticDrive
    model::Symbol = :system
    fraction::Union{Nothing,T} = nothing
    temperature_K::T = T(300.0)
    gamma_eff::Union{Nothing,T} = nothing
    noise_scale::T = one(T)
end

function RamanLangevin(model::Symbol=:system; fraction=nothing,
                       temperature_K=300.0, gamma_eff=nothing,
                       noise_scale=1.0)
    T = promote_type(typeof(float(temperature_K)), typeof(float(noise_scale)),
                     fraction === nothing ? Float64 : typeof(float(fraction)),
                     gamma_eff === nothing ? Float64 : typeof(float(gamma_eff)))
    return RamanLangevin{T}(; model=model,
                            fraction=fraction === nothing ? nothing : T(fraction),
                            temperature_K=T(temperature_K),
                            gamma_eff=gamma_eff === nothing ? nothing : T(gamma_eff),
                            noise_scale=T(noise_scale))
end

Base.@kwdef struct GainLangevin{T} <: AbstractStochasticDrive
    noise_scale::T = T(0.5)
end

GainLangevin(; noise_scale=0.5) =
    GainLangevin{typeof(float(noise_scale))}(; noise_scale=float(noise_scale))

Base.@kwdef struct StochasticEnsemble{T,N<:AbstractInitialNoise}
    uω::Array{Complex{T},3}
    noise::N
    problem::Any
end

Base.@kwdef struct StochasticTrajectory{T}
    z::Vector{T}
    fields::Array{Complex{T},4}
    problem::Any
    ensemble::Any
    drives::Vector{AbstractStochasticDrive}
    metadata::NamedTuple = (;)
end

"""
    stochastic_ensemble(problem; noise=ShotNoise(), ntraj, rng=nothing)

Construct stochastic initial conditions in unshifted Fourier coordinates.
`ShotNoise()` samples independent circular Gaussian photon-bin noise for every
frequency bin, spatial mode, and polarization channel.
"""
function stochastic_ensemble(problem::PulsePropagationProblem;
                             noise::AbstractInitialNoise=ShotNoise(),
                             ntraj::Integer,
                             rng=nothing)
    ntraj > 0 || error("ntraj must be positive.")
    noise isa ShotNoise ||
        error("Only ShotNoise initial ensembles are implemented.")
    objs = backend_objects(problem)
    nt, nm, _ = size(objs.ic.fields)
    uω0 = inverse_fft(objs.ic.fields[:, :, end], dims=1)
    ens = photon_vacuum_ensemble(uω0, objs.ic.dt, objs.sim.f0, ntraj;
                                 rng=rng,
                                 noise_scale=Float64(noise.noise_scale))
    return StochasticEnsemble{Float64,typeof(noise)}(; uω=ens, noise=noise,
                                                     problem=problem)
end

function _stochastic_zsave(problem::PulsePropagationProblem, L)
    return _zsave(problem.solver.saveat, L)
end

function _stochastic_drives(drives)
    raman = nothing
    gain = nothing
    for drive in drives
        if drive isa RamanLangevin
            raman === nothing || error("Only one RamanLangevin drive is supported.")
            raman = drive
        elseif drive isa GainLangevin
            gain === nothing || error("Only one GainLangevin drive is supported.")
            gain = drive
        else
            error("Unsupported stochastic drive $(typeof(drive)).")
        end
    end
    raman !== nothing && gain !== nothing &&
        error("Combining RamanLangevin and GainLangevin is not implemented yet.")
    return (; raman, gain)
end

function _raman_fraction(drive::RamanLangevin, fiber)
    drive.fraction !== nothing && return Float64(drive.fraction)
    fiber.fr != 0 && return Float64(fiber.fr)
    return 0.18
end

function _raman_material(drive::RamanLangevin, fiber)
    drive.model === :system && return fiber.material
    drive.model === :agarwal && return "agarwal"
    drive.model === :silica && return "silica"
    drive.model === :chalcogenide && return "chalcogenide"
    error("Unsupported RamanLangevin model $(drive.model).")
end

function _stochastic_sim_with_raman(sim::Simulation{T}, include::Bool) where {T}
    return Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0,
                         dz=sim.dz, save_period=sim.save_period,
                         midx=sim.midx, scalar=sim.scalar,
                         ellipticity=sim.ellipticity,
                         include_Raman=include,
                         gain_model=sim.gain_model,
                         pulse_centering=sim.pulse_centering,
                         progress_bar=sim.progress_bar,
                         step_method=sim.step_method,
                         cs=sim.cs, cs_model=sim.cs_model,
                         betas=sim.betas,
                         source=sim.source)
end

function _stochastic_sim_fixed_rk4ip(sim::Simulation{T}, dz::T,
                                     include_raman::Bool) where {T}
    return Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0,
                         dz=dz, save_period=dz,
                         midx=sim.midx, scalar=sim.scalar,
                         ellipticity=sim.ellipticity,
                         include_Raman=include_raman,
                         gain_model=sim.gain_model,
                         pulse_centering=false,
                         progress_bar=false,
                         step_method="RK4IP",
                         cs=sim.cs, cs_model=sim.cs_model,
                         betas=sim.betas,
                         source=sim.source)
end

function _stochastic_fiber_for_raman(fiber::Fiber{T},
                                     drive::RamanLangevin) where {T}
    return Fiber{T}(; betas=fiber.betas, sr=fiber.sr, L0=fiber.L0,
                            n2=fiber.n2,
                            material=_raman_material(drive, fiber),
                            mfd=fiber.mfd, mm_folder=fiber.mm_folder,
                            fr=T(_raman_fraction(drive, fiber)))
end

"""
    stochastic_solve(problem, ensemble; drives=[], rng=nothing)

Propagate a stochastic ensemble. Multimode scalar and vector runs support
shot-noise initial ensembles. Raman and gain Langevin drives are currently
single-mode only.
"""
function stochastic_solve(problem::PulsePropagationProblem,
                          ensemble::StochasticEnsemble;
                          drives::AbstractVector=AbstractStochasticDrive[],
                          rng=nothing,
                          reltol=nothing,
                          abstol=nothing)
    objs = backend_objects(problem)
    nt, nm, ntraj = size(ensemble.uω)
    nt == size(objs.ic.fields, 1) || error("Ensemble grid does not match problem grid.")
    save_z = _stochastic_zsave(problem, objs.fiber.L0)
    rtol, atol = _solve_tolerances(problem.solver)
    reltol === nothing || (rtol = reltol)
    abstol === nothing || (atol = abstol)

    parsed = _stochastic_drives(drives)
    if nm > 1 && (parsed.raman !== nothing || parsed.gain !== nothing)
        error("Multimode stochastic propagation currently supports initial-condition noise only.")
    end
    st = if parsed.raman === nothing
        stochastic_solve(objs.fiber, ensemble.uω, objs.ic.dt, objs.sim;
                                 zsave=save_z, rng=rng,
                                 linear_gain=problem.solver.linear_gain,
                                 linear_gain_noise=parsed.gain !== nothing,
                                 gain_noise_scale=parsed.gain === nothing ? 0.5 : Float64(parsed.gain.noise_scale),
                                 reltol=Float64(rtol), abstol=Float64(atol),
                                 deterministic_stepper=parsed.gain === nothing ? :ode : :rk4ip)
    else
        problem.solver.linear_gain === nothing ||
            error("RamanLangevin stochastic propagation is not implemented with linear_gain.")
        drive = parsed.raman
        fiber_r = _stochastic_fiber_for_raman(objs.fiber, drive)
        sim_r = _stochastic_sim_with_raman(objs.sim, true)
        stochastic_solve_raman(
            fiber_r, ensemble.uω, objs.ic.dt, sim_r;
            zsave=save_z,
            gamma_eff=drive.gamma_eff === nothing ? nothing : Float64(drive.gamma_eff),
            raman_fraction=_raman_fraction(drive, fiber_r),
            temperature_K=Float64(drive.temperature_K),
            rng=rng,
            reltol=Float64(rtol),
            abstol=Float64(atol),
            noise_scale=Float64(drive.noise_scale),
            deterministic_stepper=:rk4ip)
    end
    return StochasticTrajectory{Float64}(; z=Float64.(st.z), fields=st.fields,
                                         problem=problem, ensemble=ensemble,
                                         drives=AbstractStochasticDrive[drives...],
                                         metadata=(; backend=st))
end

solve_stochastic(args...; kwargs...) = stochastic_solve(args...; kwargs...)

function _stochastic_field(st::StochasticTrajectory; z=:final, trajectory::Integer,
                           domain::Symbol=:time)
    idx = _z_index(st, z)
    fields_t = st.fields[:, :, idx, Int(trajectory)]
    if domain === :time
        return fields_t
    elseif domain === :frequency
        return inverse_fft(fields_t, dims=1)
    else
        error("domain must be :time or :frequency")
    end
end

function _z_index(st::StochasticTrajectory, z)
    z === :initial && return 1
    z === :final && return size(st.fields, 3)
    if z isa Integer
        return Int(z)
    end
    _, idx = findmin(abs.(st.z .- z))
    return idx
end

function stochastic_values(obs::SpectralPhotonNumber, st::StochasticTrajectory; z=:final)
    nt, nm, _, ntraj = size(st.fields)
    objs = backend_objects(st.problem)
    out = zeros(Float64, ntraj)
    for q in 1:ntraj
        fields_t = _stochastic_field(st; z=z, trajectory=q, domain=:time)
        filt = _filter_for_modes(obs.filter, fields_t, obs.modes)
        out[q] = filtered_photon_number(fields_t, objs.ic.dt,
                                                objs.sim.f0, filt;
                                                shifted=obs.shifted)
    end
    return out
end

function stochastic_values(obs::FilterEnergy, st::StochasticTrajectory; z=:final)
    ntraj = size(st.fields, 4)
    out = zeros(Float64, ntraj)
    for q in 1:ntraj
        u = _stochastic_field(st; z=z, trajectory=q, domain=obs.domain)
        obs.shifted && (u = fftshift(u, 1))
        out[q] = value(obs, u)
    end
    return out
end

function fano_factor(samples)
    x = Float64.(collect(samples))
    length(x) > 1 || error("At least two samples are required.")
    μ = mean(x)
    μ > 0 || error("Mean photon number must be positive.")
    return var(x; corrected=true) / μ
end

function _fixed_stochastic_zsave(L::T, dz) where {T}
    dz === nothing && error("dz_stochastic must be supplied for Raman Langevin variance estimates.")
    dz_t = T(dz)
    dz_t > zero(T) || error("dz_stochastic must be positive.")
    nsteps = max(1, ceil(Int, L / dz_t))
    z = collect(range(zero(T), L; length=nsteps + 1))
    return z, z[2] - z[1]
end

function _final_observable_value(obs, fields_t::AbstractMatrix, dt, problem, z)
    nt, nm = size(fields_t)
    out = PropagationOutput{Float64}(;
        z=[Float64(z)],
        fields=reshape(ComplexF64.(fields_t), nt, nm, 1),
        dt=Float64(dt),
        betas=Float64[],
        dz=[Float64(z)],
        t_delay=[0.0],
    )
    return value(obs, PulseTrajectory(; output=out, problem=problem))
end

"""
    stochastic_raman_added_variance(problem, obs; ntraj, dz_stochastic, kwargs...)

Estimate the Raman Langevin contribution to an observable variance with paired
scalar single-mode trajectories. Each trajectory is propagated twice from the
same shot-noise initial condition: once with the Raman Langevin amplitude set
to zero, and once with the requested `RamanLangevin` drive. The returned
`var_paired_increment` is `var(n_total - n_initial)` and is the lowest-noise
Monte Carlo estimator for the Raman-added variance.
"""
function stochastic_raman_added_variance(problem::PulsePropagationProblem,
                                         obs;
                                         ntraj::Integer,
                                         dz_stochastic,
                                         drive::RamanLangevin=RamanLangevin(),
                                         initial_noise::ShotNoise=ShotNoise(),
                                         seed_initial::Integer=1234,
                                         seed_raman::Integer=5678,
                                         reltol=nothing,
                                         abstol=nothing,
                                         progress::Bool=false)
    ntraj > 1 || error("ntraj must be greater than one.")
    objs = backend_objects(problem)
    nt, nm, _ = size(objs.ic.fields)
    nm == 1 || error("stochastic_raman_added_variance currently supports scalar single-mode fields only.")

    uω0 = inverse_fft(objs.ic.fields[:, :, end], dims=1)
    uω_ensemble = photon_vacuum_ensemble(
        uω0, objs.ic.dt, objs.sim.f0, ntraj;
        rng=MersenneTwister(seed_initial),
        noise_scale=Float64(initial_noise.noise_scale))
    return _stochastic_raman_added_variance_from_ensemble(
        problem, obs, uω_ensemble;
        dz_stochastic=dz_stochastic,
        drive=drive,
        seed_initial=seed_initial,
        seed_raman=seed_raman,
        initial_noise=initial_noise,
        reltol=reltol,
        abstol=abstol,
        progress=progress)
end

function _stochastic_raman_added_variance_from_ensemble(problem::PulsePropagationProblem,
                                                        obs,
                                                        uω_ensemble;
                                                        dz_stochastic,
                                                        drive::RamanLangevin,
                                                        seed_initial=nothing,
                                                        seed_raman::Integer=5678,
                                                        initial_noise=nothing,
                                                        reltol=nothing,
                                                        abstol=nothing,
                                                        progress::Bool=false)
    objs = backend_objects(problem)
    nt, nm, ntraj = size(uω_ensemble)
    nm == 1 || error("stochastic_raman_added_variance currently supports scalar single-mode fields only.")
    nt == size(objs.ic.fields, 1) || error("Ensemble grid does not match problem grid.")
    objs.sim.gain_model == 0 || error("Raman stochastic variance supports passive propagation only.")

    zsave, dz_actual = _fixed_stochastic_zsave(objs.fiber.L0, dz_stochastic)
    fiber_r = _stochastic_fiber_for_raman(objs.fiber, drive)
    sim_r = _stochastic_sim_fixed_rk4ip(objs.sim, dz_actual, true)
    rtol, atol = _solve_tolerances(problem.solver)
    reltol === nothing || (rtol = reltol)
    abstol === nothing || (atol = abstol)
    n_initial = zeros(Float64, ntraj)
    n_total = zeros(Float64, ntraj)
    zero_drive = RamanLangevin(drive.model; fraction=drive.fraction,
                               temperature_K=drive.temperature_K,
                               gamma_eff=drive.gamma_eff,
                               noise_scale=0.0)
    rng_total = MersenneTwister(seed_raman)
    report_every = max(1, ntraj ÷ 10)

    elapsed_initial = @elapsed begin
        for q in 1:ntraj
            st = stochastic_solve_raman(
                fiber_r, uω_ensemble[:, :, q:q], objs.ic.dt, sim_r;
                zsave=zsave,
                gamma_eff=zero_drive.gamma_eff === nothing ? nothing : Float64(zero_drive.gamma_eff),
                raman_fraction=_raman_fraction(zero_drive, fiber_r),
                temperature_K=Float64(zero_drive.temperature_K),
                rng=MersenneTwister(seed_raman),
                reltol=Float64(rtol),
                abstol=Float64(atol),
                noise_scale=0.0,
                deterministic_stepper=:rk4ip)
            n_initial[q] = _final_observable_value(obs, st.fields[:, :, end, 1],
                                                   objs.ic.dt, problem, objs.fiber.L0)
            progress && (q == 1 || q % report_every == 0 || q == ntraj) &&
                println("initial-only trajectory $q/$ntraj")
        end
    end

    elapsed_total = @elapsed begin
        for q in 1:ntraj
            st = stochastic_solve_raman(
                fiber_r, uω_ensemble[:, :, q:q], objs.ic.dt, sim_r;
                zsave=zsave,
                gamma_eff=drive.gamma_eff === nothing ? nothing : Float64(drive.gamma_eff),
                raman_fraction=_raman_fraction(drive, fiber_r),
                temperature_K=Float64(drive.temperature_K),
                rng=rng_total,
                reltol=Float64(rtol),
                abstol=Float64(atol),
                noise_scale=Float64(drive.noise_scale),
                deterministic_stepper=:rk4ip)
            n_total[q] = _final_observable_value(obs, st.fields[:, :, end, 1],
                                                 objs.ic.dt, problem, objs.fiber.L0)
            progress && (q == 1 || q % report_every == 0 || q == ntraj) &&
                println("initial + Raman trajectory $q/$ntraj")
        end
    end

    var_initial = var(n_initial; corrected=true)
    var_total = var(n_total; corrected=true)
    var_added_difference = var_total - var_initial
    var_paired_increment = var(n_total .- n_initial; corrected=true)
    mean_initial = mean(n_initial)
    mean_total = mean(n_total)

    return (; ntraj,
            zsave,
            dz_stochastic=dz_actual,
            mean_initial,
            mean_total,
            var_initial,
            var_total,
            var_added_difference,
            var_paired_increment,
            fano_initial=var_initial / mean_initial,
            fano_total=var_total / mean_total,
            fano_added_difference=var_added_difference / mean_total,
            fano_paired_increment=var_paired_increment / mean_total,
            n_initial,
            n_total,
            drive,
            initial_noise,
            seed_initial,
            seed_raman,
            elapsed_initial,
            elapsed_total)
end
