Base.@kwdef struct VarianceObjective{O,M,N}
    observable::O
    method::M = Adjoint()
    normalization::N = PhotonNormalized()
    gradient_kwargs::NamedTuple = NamedTuple()
end

function VarianceObjective(observable;
                           method=Adjoint(),
                           normalization=PhotonNormalized(),
                           kwargs...)
    return VarianceObjective(; observable=observable, method=method,
                             normalization=normalization,
                             gradient_kwargs=NamedTuple(kwargs))
end

function variance_result(obj::VarianceObjective, problem::PulsePropagationProblem,
                         sol::PulseTrajectory=solve(problem))
    grad = gradient(problem, obj.observable;
                    trajectory=sol,
                    method=obj.method,
                    normalization=obj.normalization,
                    obj.gradient_kwargs...)
    initial_variance = sum(abs2, grad.initial_field_gradient)
    gain_added_variance = _linear_gain_added_variance(problem, sol, grad,
                                                      obj.normalization)
    variance = initial_variance + gain_added_variance
    mean_value = grad.value
    fano = mean_value != 0 ? variance / mean_value : NaN
    return (; value=mean_value,
            variance,
            initial_variance,
            gain_added_variance,
            fano,
            gradient=grad,
            observable=obj.observable,
            method=obj.method,
            normalization=obj.normalization)
end

function _linear_gain_added_variance(problem::PulsePropagationProblem,
                                     sol::PulseTrajectory,
                                     grad::GradientResult,
                                     normalization)
    gain = sol.output.linear_gain
    gain === nothing && return 0.0
    normalization isa PhotonNormalized ||
        error("linear_gain added variance currently requires PhotonNormalized variance gradients.")
    grad.adjoint_trajectory !== nothing ||
        error("linear_gain added variance requires a saved adjoint trajectory.")
    adj = photon_normalized_adjoint_trajectory(grad, problem)
    z = Float64.(adj.z)
    length(z) == size(adj.lambdaw, 3) ||
        error("Adjoint trajectory z grid and fields are inconsistent.")
    length(z) >= 2 ||
        error("linear_gain added variance requires return_lambdaw_zsave=true.")
    nt, nm, _ = size(adj.lambdaw)
    total = 0.0
    @inbounds for q in 1:length(z)-1
        dz = z[q + 1] - z[q]
        g0 = _linear_gain_rate(gain, z[q], nt, nm)
        g1 = _linear_gain_rate(gain, z[q + 1], nt, nm)
        total += 0.5 * dz * (
            sum(g0 .* abs2.(adj.lambdaw[:, :, q])) +
            sum(g1 .* abs2.(adj.lambdaw[:, :, q + 1]))
        )
    end
    return total
end

value(obj::VarianceObjective, problem::PulsePropagationProblem,
      sol::PulseTrajectory=solve(problem)) =
    variance_result(obj, problem, sol).variance

function fano(obj::VarianceObjective, problem::PulsePropagationProblem,
              sol::PulseTrajectory=solve(problem))
    return variance_result(obj, problem, sol).fano
end

function variance_directional_derivative(obj::VarianceObjective,
                                         problem::PulsePropagationProblem,
                                         direction::AbstractMatrix;
                                         trajectory=nothing,
                                         kwargs...)
    obj.method isa Adjoint ||
        error("Variance directional derivatives currently require method=PulsePropagation.Adjoint().")
    sol = trajectory === nothing ? solve(problem) : trajectory
    fiber, ic, sim, gain = _backend_tuple(problem)
    gain === nothing ||
        error("Variance directional derivatives are not wired for gain systems yet.")
    problem.solver.linear_gain === nothing ||
        error("Variance directional derivatives are not wired for linear_gain systems yet.")
    uω0 = inverse_fft(ic.fields[:, :, 1], dims=1)
    size(direction) == size(uω0) ||
        error("direction must have size $(size(uω0)); got $(size(direction)).")

    opts = merge(obj.gradient_kwargs, NamedTuple(kwargs))
    reltol = Float64(get(opts, :reltol, 1e-6))
    abstol = Float64(get(opts, :abstol, 1e-6))
    dz_adj = Float64(get(opts, :dz_adj, sim.dz))
    jvp_eps = Float64(get(opts, :jvp_eps, 1e-5))
    relative_step = Bool(get(opts, :relative_step, true))
    raman = get(opts, :raman, :auto)
    zsave = get(opts, :zsave, _zsave(problem.solver.saveat, fiber.L0))
    compressed = get(opts, :compressed, compressed_tensor(sol))

    terminal = _variance_terminal_function(obj.observable, problem)
    g0, forward0, adj0 = _variance_adjoint_gradient_initial(
        uω0, ic.dt, fiber, sim, terminal, problem;
        compressed=compressed, zsave=zsave, raman=raman,
        reltol=reltol, abstol=abstol, dz_adj=dz_adj)

    direction_norm = norm(direction)
    direction_norm > 0 || error("direction is zero; directional derivative is zero.")
    step = relative_step ? jvp_eps * max(norm(uω0), 1.0) / direction_norm : jvp_eps
    gp, _, _ = _variance_adjoint_gradient_initial(
        uω0 .+ step .* direction, ic.dt, fiber, sim, terminal, problem;
        compressed=compressed, zsave=zsave, raman=raman,
        reltol=reltol, abstol=abstol, dz_adj=dz_adj)
    gm, _, _ = _variance_adjoint_gradient_initial(
        uω0 .- step .* direction, ic.dt, fiber, sim, terminal, problem;
        compressed=compressed, zsave=zsave, raman=raman,
        reltol=reltol, abstol=abstol, dz_adj=dz_adj)
    Dg_direction = (gp .- gm) ./ (2 * step)

    denom = _variance_normalization_denominator(obj.normalization, size(g0, 1),
                                                ic.dt, sim.f0)
    variance = real(sum(abs2.(g0) ./ denom))
    dV = 2real(sum(conj.(g0) .* Dg_direction ./ denom))
    mean_value = value(obj.observable, PulseTrajectory(; output=forward0,
                                                       problem=problem))
    return (; value=mean_value,
            variance,
            fano=mean_value != 0 ? variance / mean_value : NaN,
            directional_derivative=dV,
            Dg_direction,
            adjoint_gradient=g0,
            direction,
            jvp_step=step,
            forward=forward0,
            adjoint=adj0,
            normalization=obj.normalization)
end

function _variance_terminal_function(obs, problem)
    return (uωL, forward) -> begin
        sol = PulseTrajectory(; output=forward, problem=problem)
        return _variance_terminal_for_adjoint(obs, sol)
    end
end

function _variance_terminal_for_adjoint(obs::SpectralPhotonNumber, sol)
    g = terminal_condition(obs, sol)
    return obs.shifted ? ifftshift(g, 1) : g
end

_variance_terminal_for_adjoint(obs::AbstractObservable, sol) =
    _terminal_condition_for_adjoint(obs, sol)

function _variance_adjoint_gradient_initial(uω0::AbstractMatrix{Complex{T}}, dt,
                                            fiber::Fiber{T},
                                            sim::Simulation{T},
                                            terminal_condition,
                                            problem::PulsePropagationProblem;
                                            compressed=nothing,
                                            zsave=nothing,
                                            raman::Symbol=:auto,
                                            reltol::Real=1e-6,
                                            abstol::Real=1e-6,
                                            dz_adj::Real=sim.dz) where {T}
    ic = InitialCondition{T}(; dt=T(dt),
        fields=reshape(forward_fft(uω0, dims=1),
                       size(uω0, 1), size(uω0, 2), 1))
    saveat = zsave === nothing ? [zero(T), fiber.L0] : T.(collect(zsave))
    forward = if compressed === nothing
        propagate_ode(fiber, ic, sim; zsave=saveat,
                              reltol=T(reltol), abstol=T(abstol))
    else
        propagate_ode_compressed(fiber, ic, sim, compressed;
                                         zsave=saveat,
                                         reltol=T(reltol), abstol=T(abstol))
    end
    uωL = inverse_fft(forward.fields[:, :, end], dims=1)
    λL = terminal_condition(uωL, forward)
    adj = if compressed === nothing
        solve_adjoint(λL, forward, fiber, sim; raman=raman,
                              raman_fraction=fiber.fr,
                              dz_adj=T(dz_adj),
                              reltol=T(reltol),
                              abstol=T(abstol))
    else
        solve_adjoint_compressed_rankchannels(
            λL, forward, fiber, sim, compressed; raman=raman,
            raman_fraction=fiber.fr, dz_adj=T(dz_adj),
            reltol=T(reltol), abstol=T(abstol))
    end
    return adj.lambdaw[:, :, 1], forward, adj
end

_variance_normalization_denominator(::PowerNormalized, nt, dt, f0) =
    ones(Float64, nt, 1)

_variance_normalization_denominator(::PhotonNormalized, nt, dt, f0) =
    reshape(photon_bin_weights(nt, dt, f0; shifted=false), :, 1)
