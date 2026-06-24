function import_backend(fiber::Fiber, sim::Simulation)
    basis =
        sim.scalar && length(sim.midx) == 1 ? SingleMode() :
        sim.scalar ? ModalBasis(sim.midx) :
        PolarizedModalBasis(sim.midx)
    system = PassiveFiber(;
        length=fiber.L0,
        lambda0=sim.lambda0,
        f0=sim.f0,
        material=material_from_string(fiber.material),
        dispersion=dispersion_from_betas(fiber.betas),
        geometry=ModeOverlapTensor(; sr=fiber.sr),
        n2=fiber.n2,
        raman_fraction=fiber.fr,
        source_path=fiber.mm_folder,
    )
    return system, basis
end

function backend_problem(fiber::Fiber, sim::Simulation, ic::InitialCondition;
                         solver=nothing, terms=nothing)
    system, basis = import_backend(fiber, sim)
    grid = TimeGrid{Float64}(; nt=size(ic.fields, 1), dt=ic.dt)
    state = PulseState(grid, ic.fields; basis=basis)
    save_spec = sim.save_period == 0 ? SaveAt([0.0, fiber.L0]) : SaveEvery(sim.save_period)
    step = FixedStep(; dz=sim.dz)
    solver2 = solver === nothing ? RK4IPSolver(; stepping=step, saveat=save_spec) : solver
    terms2 = terms === nothing ? _terms_from_sim(sim) : terms
    model = PropagationModel(; system=system, basis=basis, terms=terms2)
    return PulsePropagationProblem(; initial_state=state, model=model, solver=solver2)
end

function backend_objects(problem::PulsePropagationProblem)
    system = problem.model.system
    system isa GainFiber && error("GainFiber/rate-equation media are not part of the public v0 API. Use RK4IPSolver(linear_gain=LinearGain(...)) for linear gain.")
    material = _backend_material_name(system, problem.model)
    fiber = Fiber{Float64}(;
        betas=Float64.(betas(system.dispersion)),
        sr=Float64.(sr(system.geometry)),
        L0=Float64(system.length),
        n2=Float64(system.n2),
        material=material,
        mfd=nothing,
        mm_folder=system.source_path,
        fr=Float64(system.raman_fraction),
    )
    basis = problem.model.basis
    midx = basis isa SingleMode ? [1] :
           basis isa ModalBasis ? basis.modes :
           basis isa PolarizedModalBasis ? basis.spatial_modes : [1]
    scalar = !(basis isa PolarizedModalBasis)
    include_raman = has_term(problem.model, RamanTerm) && system.raman_fraction > 0
    save_period = _save_period(problem.solver.saveat, fiber.L0)
    dz = _step_dz(problem.solver.stepping)
    sim = Simulation{Float64}(;
        lambda0=system.lambda0,
        f0=system.f0,
        dz=dz,
        save_period=save_period,
        midx=midx,
        scalar=scalar,
        ellipticity=0.0,
        include_Raman=include_raman,
        gain_model=0,
        pulse_centering=false,
        progress_bar=false,
        step_method="RK4IP",
    )
    ic = InitialCondition{Float64}(;
        dt=problem.initial_state.grid.dt,
        fields=problem.initial_state.fields,
    )
    return (; fiber, sim, ic, gain=nothing)
end

function _backend_material_name(system, model)
    raman = _raman_term(model)
    raman === nothing && return material_name(system.material)
    raman.model === :agarwal && return "agarwal"
    raman.model === :silica && return "silica"
    raman.model === :chalcogenide && return "chalcogenide"
    return material_name(system.material)
end

_uses_agarwal_custom(::PropagationModel) = false

function solve(problem::PulsePropagationProblem)
    objs = backend_objects(problem)
    zsave = _zsave(problem.solver.saveat, objs.fiber.L0)
    reltol, abstol = _solve_tolerances(problem.solver)
    solver_cache = nothing
    out = if problem.solver.compression isa CPCompression
        problem.solver.linear_gain === nothing ||
            error("CPCompression is currently available for passive propagation only.")
        cpinfo = _resolve_cp_compression(problem.solver.compression, objs.fiber, objs.sim,
                                         problem.solver.backend)
        solver_cache = (; compression=cpinfo)
        if problem.solver.cp_rhs isa FixedRK4IPCPRHS
            rhs = problem.solver.cp_rhs
            fixed_dt = _fixed_dt(problem.solver.stepping)
            fixed_dt === nothing &&
                error("FixedRK4IPCPRHS requires FixedStep stepping.")
            if problem.solver.backend isa CUDABackend
                propagate_rk4ip_cp_cuda(objs.fiber, objs.ic, objs.sim, cpinfo.tensor;
                                        zsave=zsave,
                                        dz=fixed_dt,
                                        device=problem.solver.backend.device,
                                        synchronize=problem.solver.backend.synchronize)
            else
                propagate_rk4ip_cp_cached(objs.fiber, objs.ic, objs.sim, cpinfo.tensor;
                                          zsave=zsave,
                                          dz=fixed_dt,
                                          fftw_threads=rhs.fftw_threads,
                                          fftw_flags=_fftw_flags(rhs.fftw_flags),
                                          blas_threads=rhs.blas_threads)
            end
        elseif problem.solver.backend isa CUDABackend
            fixed_dt = _fixed_dt(problem.solver.stepping)
            fixed_dt === nothing &&
                error("CUDABackend with CPCompression requires FixedStep stepping.")
            propagate_rk4ip_cp_cuda(objs.fiber, objs.ic, objs.sim, cpinfo.tensor;
                                    zsave=zsave,
                                    dz=fixed_dt,
                                    device=problem.solver.backend.device,
                                    synchronize=problem.solver.backend.synchronize)
        elseif problem.solver.cp_rhs isa CachedCPRHS
            rhs = problem.solver.cp_rhs
            propagate_ode_cp_cached(objs.fiber, objs.ic, objs.sim, cpinfo.tensor;
                                    zsave=zsave,
                                    reltol=reltol,
                                    abstol=abstol,
                                    dtmax=_dtmax(problem.solver.stepping),
                                    fixed_dt=_fixed_dt(problem.solver.stepping),
                                    fftw_threads=rhs.fftw_threads,
                                    fftw_flags=_fftw_flags(rhs.fftw_flags),
                                    blas_threads=rhs.blas_threads)
        else
            propagate_ode_cp(objs.fiber, objs.ic, objs.sim, cpinfo.tensor;
                             zsave=zsave,
                             reltol=reltol,
                             abstol=abstol,
                             dtmax=_dtmax(problem.solver.stepping),
                             fixed_dt=_fixed_dt(problem.solver.stepping))
        end
    else
        if problem.solver.backend isa CUDABackend
            problem.solver.linear_gain === nothing ||
                error("CUDABackend forward propagation currently supports passive scalar propagation only.")
            fixed_dt = _fixed_dt(problem.solver.stepping)
            fixed_dt === nothing &&
                error("CUDABackend forward propagation requires FixedStep stepping.")
            propagate_rk4ip_full_cuda(objs.fiber, objs.ic, objs.sim;
                                      zsave=zsave,
                                      dz=fixed_dt,
                                      device=problem.solver.backend.device,
                                      synchronize=problem.solver.backend.synchronize)
        else
            propagate_ode(objs.fiber, objs.ic, objs.sim;
                          zsave=zsave,
                          linear_gain=problem.solver.linear_gain,
                          reltol=reltol,
                          abstol=abstol)
        end
    end
    return PulseTrajectory{Float64,typeof(out)}(; output=out, problem=problem,
                                                solver_cache=solver_cache)
end

function _resolve_cp_compression(compression::CPCompression, fiber, sim,
                                 backend::AbstractExecutionBackend=CPUBackend())
    sim.scalar ||
        error("CPCompression currently supports scalar modal propagation. Polarized CP support is not part of v0.")
    compression.initial_rank > 0 || error("CPCompression.initial_rank must be positive.")
    compression.max_rank >= compression.initial_rank ||
        error("CPCompression.max_rank must be at least initial_rank.")
    compression.rank_growth > 1 || error("CPCompression.rank_growth must be greater than 1.")
    compression.target_error > 0 || error("CPCompression.target_error must be positive.")
    if compression.tensor !== nothing
        cp = _as_cp(compression.tensor)
        err = _cp_relative_error(cp, fiber.sr)
        return (; tensor=cp, rank=length(cp.λ), error=err,
                target_error=compression.target_error, attempts=NamedTuple[],
                fitted=false, elapsed=0.0)
    end

    tensor = Float64.(fiber.sr)
    rng = Random.MersenneTwister(compression.seed)
    rank = compression.initial_rank
    init = nothing
    attempts = NamedTuple[]
    last_cp = nothing
    while rank <= compression.max_rank
        cp = nothing
        history = Float64[]
        elapsed = @elapsed begin
            cp, history = if backend isa CUDABackend
                cp_als_warm_cuda(
                    tensor, rank;
                    init=init,
                    maxiter=compression.maxiter,
                    tol=compression.tol,
                    ridge=compression.ridge,
                    check_every=compression.check_every,
                    rng=rng,
                    verbose=compression.verbose,
                    device=backend.device,
                    synchronize=backend.synchronize,
                )
            else
                cp_als_warm(
                    tensor, rank;
                    init=init,
                    maxiter=compression.maxiter,
                    tol=compression.tol,
                    ridge=compression.ridge,
                    check_every=compression.check_every,
                    rng=rng,
                    verbose=compression.verbose,
                )
            end
        end
        err = _cp_relative_error(cp, tensor)
        push!(attempts, (; rank=rank, error=err, history=history, elapsed=elapsed))
        err <= compression.target_error &&
            return (; tensor=cp, rank=rank, error=err,
                    target_error=compression.target_error, attempts=attempts,
                    fitted=true, elapsed=sum(a.elapsed for a in attempts))
        init = cp
        last_cp = cp
        rank *= compression.rank_growth
    end
    last_err = isempty(attempts) ? Inf : attempts[end].error
    error("CPCompression did not meet target_error=$(compression.target_error) by max_rank=$(compression.max_rank). Last error was $last_err.")
end

_cp_relative_error(cp, tensor) = norm(cp_reconstruct(cp, tensor) .- tensor) / norm(tensor)

function _backend_tuple(problem)
    objs = backend_objects(problem)
    return objs.fiber, objs.ic, objs.sim, objs.gain
end

function _terms_from_sim(sim)
    terms = AbstractModelTerm[DispersionTerm(), KerrTerm()]
    sim.include_Raman && push!(terms, RamanTerm())
    return terms
end

_save_period(saveat::SaveEvery, L) = saveat.dz
_save_period(saveat::SaveAt, L) = length(saveat.z) <= 2 ? 0.0 : minimum(diff(saveat.z))

_zsave(saveat::SaveAt, L) = saveat.z
_zsave(saveat::SaveEvery, L) = collect(range(0.0, L; step=saveat.dz))

_step_dz(step::FixedStep) = step.dz
_step_dz(step::AdaptiveStep) = step.initial_dz
_dtmax(step::FixedStep) = step.dz
_dtmax(step::AdaptiveStep) = step.max_dz
_fixed_dt(step::FixedStep) = step.dz
_fixed_dt(step::AdaptiveStep) = nothing

function _fftw_flags(flag::Symbol)
    flag === :ESTIMATE && return FFTW.ESTIMATE
    flag === :MEASURE && return FFTW.MEASURE
    flag === :PATIENT && return FFTW.PATIENT
    error("Unknown FFTW flag $(flag). Use :ESTIMATE, :MEASURE, or :PATIENT.")
end

_solve_tolerances(solver::RK4IPSolver) =
    solver.stepping isa AdaptiveStep ? (solver.stepping.rtol, solver.stepping.atol) :
                                       (solver.reltol, solver.abstol)
