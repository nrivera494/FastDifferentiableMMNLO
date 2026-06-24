Base.@kwdef struct GradientResult{T}
    value::T
    terminal_condition::Array{Complex{T},2}
    initial_field_gradient::Any = nothing
    adjoint_trajectory::Any = nothing
    method::Any = Adjoint()
    normalization::Any = PowerNormalized()
end

function gradient(problem::PulsePropagationProblem, obs::AbstractObservable;
                  method=Adjoint(), wrt=InitialField(),
                  normalization=PowerNormalized(),
                  trajectory=nothing,
                  raman=:auto,
                  raman_fraction=nothing,
                  dz_adj=nothing,
                  reltol=1e-10,
                  abstol=1e-10,
                  return_lambdaw_zsave=true)
    wrt isa InitialField || error("Only wrt=InitialField() is supported so far.")
    method isa AutomaticDifferentiation && return _gradient_ad(problem, obs;
        method=method, wrt=wrt, normalization=normalization, trajectory=trajectory)
    method isa Adjoint || error("method must be Adjoint() or AutomaticDifferentiation().")
    sol = trajectory === nothing ? solve(problem) : trajectory
    terminal = _terminal_condition_for_adjoint(obs, sol)
    fiber, ic, sim, _ = _backend_tuple(problem)
    rf = raman_fraction === nothing ? fiber.fr : raman_fraction
    dz = dz_adj === nothing ? _adjoint_dz(problem, sim) : dz_adj
    adj = _solve_adjoint_for_gradient(problem, sol, terminal, fiber, sim;
                                      raman=raman, raman_fraction=rf,
                                      return_lambdaw_zsave=return_lambdaw_zsave,
                                      dz_adj=dz, reltol=reltol, abstol=abstol)
    g = adj.lambdaw[:, :, 1]
    if normalization isa PhotonNormalized
        weights = photon_bin_weights(size(g, 1), ic.dt, sim.f0; shifted=false)
        g = _photon_normalize_fourier(g, weights)
    end
    return GradientResult{Float64}(; value=value(obs, sol),
                                   terminal_condition=terminal,
                                   initial_field_gradient=g,
                                   adjoint_trajectory=adj,
                                   method=method,
                                   normalization=normalization)
end

function gradient(problem::PulsePropagationProblem, obs::SpectralPhotonNumber;
                  method=Adjoint(), wrt=InitialField(),
                  normalization=PhotonNormalized(),
                  trajectory=nothing,
                  raman=:auto,
                  raman_fraction=nothing,
                  dz_adj=nothing,
                  reltol=1e-10,
                  abstol=1e-10,
                  return_lambdaw_zsave=true)
    wrt isa InitialField || error("Only wrt=InitialField() is supported so far.")
    method isa AutomaticDifferentiation && return _gradient_ad(problem, obs;
        method=method, wrt=wrt, normalization=normalization, trajectory=trajectory)
    method isa Adjoint || error("method must be Adjoint() or AutomaticDifferentiation().")
    sol = trajectory === nothing ? solve(problem) : trajectory
    terminal = terminal_condition(obs, sol)
    fiber, ic, sim, _ = _backend_tuple(problem)
    rf = raman_fraction === nothing ? fiber.fr : raman_fraction
    dz = dz_adj === nothing ? _adjoint_dz(problem, sim) : dz_adj
    adj = _solve_adjoint_for_gradient(problem, sol, terminal, fiber, sim;
                                      raman=raman, raman_fraction=rf,
                                      return_lambdaw_zsave=return_lambdaw_zsave,
                                      dz_adj=dz, reltol=reltol, abstol=abstol)
    g = adj.lambdaw[:, :, 1]
    if normalization isa PhotonNormalized
        weights = photon_bin_weights(size(g, 1), ic.dt, sim.f0; shifted=false)
        g = _photon_normalize_fourier(g, weights)
    end
    return GradientResult{Float64}(; value=value(obs, sol),
                                   terminal_condition=terminal,
                                   initial_field_gradient=g,
                                   adjoint_trajectory=adj,
                                   method=method,
                                   normalization=normalization)
end

function photon_normalized_adjoint_trajectory(result::GradientResult,
                                              problem::PulsePropagationProblem)
    adj = result.adjoint_trajectory
    adj !== nothing || error("GradientResult has no adjoint_trajectory.")
    hasproperty(adj, :lambdaw) ||
        error("GradientResult adjoint_trajectory does not contain saved lambdaw data.")
    _, ic, sim, _ = _backend_tuple(problem)
    weights = photon_bin_weights(size(adj.lambdaw, 1), ic.dt, sim.f0;
                                         shifted=false)
    return (; z=adj.z,
            lambdaw=_photon_normalize_fourier(adj.lambdaw, weights),
            weights)
end

function _photon_normalize_fourier(a, weights)
    out = similar(a)
    for q in axes(a, 3), j in axes(a, 2), k in axes(a, 1)
        out[k, j, q] = weights[k] > 0 ? a[k, j, q] / sqrt(weights[k]) :
                         zero(eltype(a))
    end
    return out
end

function _photon_normalize_fourier(a::AbstractMatrix, weights)
    out = similar(a)
    for j in axes(a, 2), k in axes(a, 1)
        out[k, j] = weights[k] > 0 ? a[k, j] / sqrt(weights[k]) :
                      zero(eltype(a))
    end
    return out
end

function _solve_adjoint_for_gradient(problem::PulsePropagationProblem,
                                     sol::PulseTrajectory,
                                     terminal,
                                     fiber,
                                     sim;
                                     raman=:auto,
                                     raman_fraction=fiber.fr,
                                     return_lambdaw_zsave=true,
                                     dz_adj=sim.dz,
                                     reltol=1e-10,
                                     abstol=1e-10)
    compressed = compressed_tensor(sol)
    if compressed === nothing && problem.solver.compression isa CPCompression
        compressed = _resolve_cp_compression(problem.solver.compression, fiber, sim,
                                             problem.solver.backend).tensor
    end
    if compressed !== nothing
        problem.solver.linear_gain === nothing ||
            error("CPCompression adjoints do not currently support linear_gain.")
        problem.solver.backend isa CUDABackend && return solve_adjoint_compressed_rankchannels_cuda(
            terminal, sol.output, fiber, sim, compressed;
            raman=raman, raman_fraction=raman_fraction,
            return_lambdaw_zsave=return_lambdaw_zsave,
            dz_adj=dz_adj, reltol=reltol, abstol=abstol,
            device=problem.solver.backend.device,
            synchronize=problem.solver.backend.synchronize)
        return solve_adjoint_compressed_rankchannels(
            terminal, sol.output, fiber, sim, compressed;
            raman=raman, raman_fraction=raman_fraction,
            return_lambdaw_zsave=return_lambdaw_zsave,
            dz_adj=dz_adj, reltol=reltol, abstol=abstol)
    else
        problem.solver.backend isa CUDABackend && return solve_adjoint_cuda(
            terminal, sol.output, fiber, sim;
            linear_gain=problem.solver.linear_gain,
            raman=raman, raman_fraction=raman_fraction,
            return_lambdaw_zsave=return_lambdaw_zsave,
            dz_adj=dz_adj, reltol=reltol, abstol=abstol,
            device=problem.solver.backend.device,
            synchronize=problem.solver.backend.synchronize)
        return solve_adjoint(
            terminal, sol.output, fiber, sim;
            linear_gain=problem.solver.linear_gain,
            raman=raman, raman_fraction=raman_fraction,
            return_lambdaw_zsave=return_lambdaw_zsave,
            dz_adj=dz_adj, reltol=reltol, abstol=abstol)
    end
end

function _terminal_condition_for_adjoint(obs::FilterEnergy, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, obs.domain, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::BinEnergy, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, obs.domain, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::ProjectedEnergy, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, obs.domain, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::ProjectedPhotonNumber, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, obs.domain, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::QuadratureObservable, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, obs.domain, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::TemporalMoment, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, :time, false)
end

function _terminal_condition_for_adjoint(obs::SpectralMoment, sol)
    g = terminal_condition(obs, sol)
    return _terminal_to_unshifted_fourier(g, :frequency, obs.shifted)
end

function _terminal_condition_for_adjoint(obs::RatioObservable, sol)
    num = value(obs.numerator, sol)
    den = value(obs.denominator, sol)
    den == 0 && error("RatioObservable denominator is zero.")
    gnum = _terminal_condition_for_adjoint(obs.numerator, sol)
    gden = _terminal_condition_for_adjoint(obs.denominator, sol)
    size(gnum) == size(gden) ||
        error("RatioObservable adjoint terminal conditions must have matching sizes.")
    return (den .* gnum .- num .* gden) ./ den^2
end

_terminal_to_unshifted_fourier(g, ::Val{:time}, shifted::Bool=false) =
    shifted ? error("shifted=true is only meaningful for frequency-domain terminal conditions.") :
              size(g, 1) .* ifft(g, 1)

_terminal_to_unshifted_fourier(g, ::Val{:frequency}, shifted::Bool=false) =
    shifted ? ifftshift(g, 1) : g

_terminal_to_unshifted_fourier(g, domain::Symbol, shifted::Bool=false) =
    _terminal_to_unshifted_fourier(g, Val(domain), shifted)

function _gradient_ad(problem::PulsePropagationProblem, obs::AbstractObservable;
                      method::AutomaticDifferentiation,
                      wrt=InitialField(),
                      normalization=PowerNormalized(),
                      trajectory=nothing)
    result = _observable_gradient_ad(problem, obs; method=method,
                                     normalization=normalization)
    g = if normalization isa PhotonNormalized
        result.photon_fourier.dB_conj
    elseif normalization isa PowerNormalized
        result.dA_conj
    else
        error("Unsupported normalization $(typeof(normalization)).")
    end
    sol = trajectory === nothing ? solve(problem) : trajectory
    return GradientResult{Float64}(; value=Float64(result.value),
                                   terminal_condition=terminal_condition(obs, sol),
                                   initial_field_gradient=g,
                                   adjoint_trajectory=result,
                                   method=method,
                                   normalization=normalization)
end

function _gradient_ad(problem::PulsePropagationProblem, obs::SpectralPhotonNumber;
                      method::AutomaticDifferentiation,
                      wrt=InitialField(),
                      normalization=PhotonNormalized(),
                      trajectory=nothing)
    method.backend === :zygote ||
        error("Only AutomaticDifferentiation(:zygote) is supported so far.")
    wrt isa InitialField || error("Only wrt=InitialField() is supported so far.")
    problem.solver.linear_gain === nothing ||
        error("AutomaticDifferentiation() is not wired for linear_gain systems yet.")
    fiber, ic, sim, gain = _backend_tuple(problem)
    gain === nothing || error("AutomaticDifferentiation() is not wired for gain systems yet.")
    filt = _filter_for_modes(obs.filter, last_fields(ic), obs.modes)
    result = output_photon_number_gradient(
        fiber, last_fields(ic), ic.dt, sim, filt;
        shifted=obs.shifted,
        return_fourier=normalization isa PowerNormalized,
        return_photon_fourier=normalization isa PhotonNormalized)
    g = if normalization isa PhotonNormalized
        result.photon_fourier.dB_conj
    elseif normalization isa PowerNormalized
        result.dA_conj
    else
        error("Unsupported normalization $(typeof(normalization)).")
    end
    terminal = nothing
    return GradientResult{Float64}(; value=result.photon_number,
                                   terminal_condition=zeros(ComplexF64, size(filt)),
                                   initial_field_gradient=g,
                                   adjoint_trajectory=result,
                                   method=method,
                                   normalization=normalization)
end

function _observable_gradient_ad(problem::PulsePropagationProblem, obs::AbstractObservable;
                                 method::AutomaticDifferentiation,
                                 normalization=PowerNormalized())
    method.backend === :zygote ||
        error("Only AutomaticDifferentiation(:zygote) is supported so far.")
    problem.solver.compression === nothing ||
        error("AutomaticDifferentiation() is not wired for CPCompression yet.")
    problem.solver.linear_gain === nothing ||
        error("AutomaticDifferentiation() is not wired for linear_gain systems yet.")
    fiber, ic, sim, gain = _backend_tuple(problem)
    gain === nothing || error("AutomaticDifferentiation() is not wired for gain systems yet.")
    fields0 = _field_matrix(last_fields(ic))
    objective(a) = begin
        T = ad_real_type(eltype(a))
        ic2 = InitialCondition{T}(; dt=T(ic.dt),
            fields=reshape(a, size(a, 1), size(a, 2), 1))
        out = propagate_ad(fiber, ic2, sim)
        val = _ad_observable_value(obs, out, problem)
        val isa Real || error("AutomaticDifferentiation() requires a scalar real observable value.")
        return val
    end
    J = objective(fields0)
    gzyg = Zygote.gradient(objective, fields0)[1]
    dA = 0.5 .* conj.(gzyg)
    dA_conj = 0.5 .* gzyg
    dAw = nothing
    dAw_conj = nothing
    photon_fourier = nothing
    if normalization isa PhotonNormalized
        dAw, dAw_conj = _fourier_wirtinger_from_time(dA, dA_conj; shifted=false)
        dB, dB_conj = _photon_fourier_wirtinger_from_fourier(
            dAw, dAw_conj, ic.dt, sim.f0; shifted=false)
        photon_fourier = (; dB=dB, dB_conj=dB_conj)
    end
    fourier = dAw === nothing ? nothing : (; dAw=dAw, dAw_conj=dAw_conj)
    return (; value=J, dA=dA, dA_conj=dA_conj,
            zygote_gradient=gzyg, fourier=fourier,
            photon_fourier=photon_fourier)
end

function _ad_observable_field(out, domain::Symbol, shifted::Bool)
    fields_t = out.fields[:, :, end]
    if domain === :time
        shifted && error("shifted=true is only meaningful for frequency-domain observables.")
        return fields_t
    elseif domain === :frequency
        uω = dense_transform_dim1(fields_t, true)
        return shifted ? fftshift(uω, 1) : uω
    else
        error("domain must be :time or :frequency.")
    end
end

function _ad_observable_value(obs::FilterEnergy, out, problem)
    u = _ad_observable_field(out, obs.domain, obs.shifted)
    f = ChainRulesCore.ignore_derivatives() do
        _observable_filter(obs.filter, size(u), obs.modes)
    end
    return real(sum(f .* abs2.(u)))
end

_ad_observable_value(obs::BinEnergy, out, problem) =
    value(obs, _ad_observable_field(out, obs.domain, obs.shifted))

_ad_observable_value(obs::QuadratureObservable, out, problem) =
    value(obs, _ad_observable_field(out, obs.domain, obs.shifted))

function _ad_observable_value(obs::ProjectedEnergy, out, problem)
    u = _ad_observable_field(out, obs.domain, obs.shifted)
    up = _project_field(obs.U, u)
    f = ChainRulesCore.ignore_derivatives() do
        obs.filter === nothing ? ones(real(eltype(up)), size(up)) :
                                  _observable_filter(obs.filter, size(up), :all)
    end
    return real(sum(f .* abs2.(up)))
end

function _ad_observable_value(obs::ProjectedPhotonNumber, out, problem)
    u = _ad_observable_field(out, obs.domain, obs.shifted)
    up = _project_field(obs.U, u)
    f = ChainRulesCore.ignore_derivatives() do
        _projected_photon_filter(obs.weights, obs.filter, size(up))
    end
    return real(sum(f .* abs2.(up)))
end

function _ad_observable_value(obs::RatioObservable, out, problem)
    den = _ad_observable_value(obs.denominator, out, problem)
    den == 0 && error("RatioObservable denominator is zero.")
    return _ad_observable_value(obs.numerator, out, problem) / den
end

function _ad_observable_value(obs::TemporalMoment, out, problem)
    axis, selected = ChainRulesCore.ignore_derivatives() do
        time_axis(problem.initial_state.grid), Tuple(_selected_modes(obs.modes, size(out.fields, 2)))
    end
    return _moment_value_selected(_ad_observable_field(out, :time, false),
                                  axis, obs.order, obs.center,
                                  obs.normalized, selected)
end

function _ad_observable_value(obs::SpectralMoment, out, problem)
    axis, selected = ChainRulesCore.ignore_derivatives() do
        frequency_axis(problem.initial_state.grid; shifted=obs.shifted),
        Tuple(_selected_modes(obs.modes, size(out.fields, 2)))
    end
    return _moment_value_selected(_ad_observable_field(out, :frequency, obs.shifted),
                                  axis, obs.order, obs.center,
                                  obs.normalized, selected)
end

function _adjoint_dz(problem, sim)
    step = problem.solver.stepping
    step isa FixedStep && return step.dz
    return sim.dz
end
