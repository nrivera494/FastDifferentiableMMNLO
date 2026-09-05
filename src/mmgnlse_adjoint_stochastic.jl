"""
    _mmgnlse_vjp_finite_difference(field, lambda, parameters, z)

Reference vector--Jacobian product for the public MMGNLSE right-hand side.
The defining real pairing is

```
2real(sum(conj(lambda) .* δrhs)) =
2real(sum(conj(vjp) .* δfield)).
```

This implementation is deliberately a correctness fallback: it uses centered
finite differences in the real and imaginary directions of every state
coordinate. Solver-specific analytic VJPs may replace `_mmgnlse_vjp` without
changing the adjoint facade.
"""
function _mmgnlse_vjp_finite_difference(field::AbstractArray{<:Complex,3},
                                         lambda::AbstractArray{<:Complex,3},
                                         parameters::MMGNLSEParameters,
                                         z::Real;
                                         relative_step=nothing)
    size(lambda) == size(field) || throw(DimensionMismatch(
        "The VJP state and cotangent must have matching shapes."))
    T = promote_type(typeof(float(real(z))),
                     typeof(float(real(zero(eltype(field))))), Float64)
    step_scale = relative_step === nothing ? eps(T)^(T(1) / T(3)) : T(relative_step)
    isfinite(step_scale) && step_scale > zero(T) || throw(ArgumentError(
        "The finite-difference VJP relative step must be finite and positive."))

    state = Complex{T}.(Array(field))
    cotangent = Complex{T}.(Array(lambda))
    result = similar(state)
    plus = copy(state)
    minus = copy(state)

    pairing(rhs) = real(sum(conj.(cotangent) .* rhs))
    @inbounds for index in eachindex(state)
        value = state[index]
        h = step_scale * max(one(T), abs(value))

        plus[index] = value + h
        minus[index] = value - h
        real_derivative = (pairing(_mmgnlse_rhs(plus, parameters, z)) -
                           pairing(_mmgnlse_rhs(minus, parameters, z))) / (2h)

        plus[index] = value + im * h
        minus[index] = value - im * h
        imaginary_derivative = (pairing(_mmgnlse_rhs(plus, parameters, z)) -
                                pairing(_mmgnlse_rhs(minus, parameters, z))) / (2h)

        result[index] = real_derivative + im * imaginary_derivative
        plus[index] = value
        minus[index] = value
    end
    return result
end

# Production adjoints use the analytic dense/CP VJP. The coordinate finite-
# difference implementation above remains an independent regression oracle.
_mmgnlse_vjp(field, lambda, parameters::MMGNLSEParameters, z) =
    _mmgnlse_vjp_analytic(field, lambda, parameters, z)
_mmgnlse_vjp(field, lambda, parameters::MMGNLSEParameters, z, cache) =
    _mmgnlse_vjp_analytic(field, lambda, parameters, z, cache)

function _mmgnlse_cache_integration_z(solution::MMGNLSESolution)
    cache = solution.cache
    cache === nothing && return nothing
    hasproperty(cache, :integration_z) || return nothing
    return getproperty(cache, :integration_z)
end

function _mmgnlse_forward_is_step_saved(solution::MMGNLSESolution)
    integration_z = _mmgnlse_cache_integration_z(solution)
    integration_z === nothing && return false
    length(integration_z) == length(solution.z) || return false
    return all(solution.z .== integration_z)
end

function _mmgnlse_dense_forward_for_adjoint(solution::MMGNLSESolution,
                                             parameters::MMGNLSEParameters)
    solution.parameters === parameters || throw(ArgumentError(
        "forward_solution was constructed with a different MMGNLSEParameters object."))
    _mmgnlse_forward_is_step_saved(solution) && return solution
    return solve_mmgnlse(solution.initial_field, parameters, solution.dz;
                         method=solution.method, saveat=:steps)
end

function _mmgnlse_forward_interaction_fields(solution::MMGNLSESolution,
                                              parameters::MMGNLSEParameters,
                                              cache)
    interaction = ifft(solution.fields, 1)
    @inbounds for index in eachindex(solution.z)
        propagator = _mmgnlse_linear_propagator(
            parameters, zero(solution.z[index]), solution.z[index], cache)
        interaction[:, :, :, index] ./= propagator
    end
    return interaction
end

function _mmgnlse_interpolate_interaction_field(fields::AbstractArray{<:Complex,4},
                                                 zgrid::AbstractVector,
                                                 z::Real)
    tolerance = 64eps(float(eltype(zgrid))) *
                max(one(float(eltype(zgrid))), abs(first(zgrid)), abs(last(zgrid)))
    z < first(zgrid) - tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z before the forward trajectory."))
    z > last(zgrid) + tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z after the forward trajectory."))
    z <= first(zgrid) && return @view fields[:, :, :, 1]
    z >= last(zgrid) && return @view fields[:, :, :, end]

    right = searchsortedfirst(zgrid, z)
    zgrid[right] == z && return @view fields[:, :, :, right]
    left = right - 1
    weight = (z - zgrid[left]) / (zgrid[right] - zgrid[left])
    return (one(weight) - weight) .* @view(fields[:, :, :, left]) .+
           weight .* @view(fields[:, :, :, right])
end

mutable struct MMGNLSEScalarCPAdjointWorkspace{C,A3,A2,A1,P3F,P3I}
    cp::C
    interpolated::A3
    propagator::A3
    field_t::A3
    lambda_t::A3
    cubic_cotangent::A3
    vjp_t::A3
    spectral_scratch::A3
    rank_cotangent::A2
    cwu1::A2
    raman_combined::A1
    raman_combined_conj::A1
    fft_field!::P3F
    fft_lambda!::P3F
    fft_cubic!::P3F
    ifft_cubic!::P3I
    ifft_vjp!::P3I
end

function _mmgnlse_scalar_cp_adjoint_workspace(
    prototype::Array{ComplexF64,3},
    cp::MMGNLSECPDecomposition,
    raman,
)
    nt, nm, _ = size(prototype)
    forward_workspace = _mmgnlse_scalar_cp_workspace(cp, nt, nm)
    arrays = ntuple(_ -> similar(prototype), 7)
    interpolated, propagator, field_t, lambda_t,
        cubic_cotangent, vjp_t, spectral_scratch = arrays
    rank_cotangent = zeros(ComplexF64, nt, cp_rank(cp))
    weighted_u1 = ComplexF64.(cp.U[1]) .*
                  reshape(ComplexF64.(cp.λ), 1, :)
    cwu1 = conj.(weighted_u1)
    raman_combined = ComplexF64.(raman.ha .+ raman.hb)
    raman_combined_conj = conj.(raman_combined)
    return MMGNLSEScalarCPAdjointWorkspace(
        forward_workspace.cp, interpolated, propagator, field_t,
        lambda_t, cubic_cotangent, vjp_t, spectral_scratch,
        rank_cotangent, cwu1, raman_combined, raman_combined_conj,
        plan_fft!(field_t, 1; flags=FFTW.MEASURE),
        plan_fft!(lambda_t, 1; flags=FFTW.MEASURE),
        plan_fft!(cubic_cotangent, 1; flags=FFTW.MEASURE),
        plan_ifft!(cubic_cotangent, 1; flags=FFTW.MEASURE),
        plan_ifft!(spectral_scratch, 1; flags=FFTW.MEASURE))
end

function _mmgnlse_interpolate_interaction_field!(
    out, fields, zgrid, z,
)
    z <= first(zgrid) && (out .= @view(fields[:, :, :, 1]); return out)
    z >= last(zgrid) && (out .= @view(fields[:, :, :, end]); return out)
    right = searchsortedfirst(zgrid, z)
    if zgrid[right] == z
        out .= @view fields[:, :, :, right]
        return out
    end
    left = right - 1
    weight = (z - zgrid[left]) / (zgrid[right] - zgrid[left])
    out .= (1 - weight) .* @view(fields[:, :, :, left]) .+
           weight .* @view(fields[:, :, :, right])
    return out
end

function _mmgnlse_scalar_cp_backproject!(
    out, rank_values, factor, cache,
)
    mul!(cache.k, rank_values, adjoint(factor))
    out .+= cache.k
    return out
end

function _mmgnlse_scalar_cp_vjp_nonlinear!(
    out,
    field_t,
    cotangent_t,
    solver_cache,
    workspace::MMGNLSEScalarCPAdjointWorkspace,
)
    fill!(out, zero(eltype(out)))
    !solver_cache.nonlinear_active && return out
    workspace.cubic_cotangent .= cotangent_t
    workspace.ifft_cubic! * workspace.cubic_cotangent
    workspace.cubic_cotangent .*= conj.(solver_cache.nonlinear_prefactor)
    workspace.fft_cubic! * workspace.cubic_cotangent

    cache = workspace.cp
    nt, nm, _ = size(field_t)
    field = reshape(field_t, nt, nm)
    cotangent = reshape(workspace.cubic_cotangent, nt, nm)
    mul!(cache.b2, field, cache.u2)
    mul!(cache.b3, field, cache.u3)
    cache.conj_at .= conj.(field)
    mul!(cache.b4, cache.conj_at, cache.u4)
    mul!(workspace.rank_cotangent, cotangent, workspace.cwu1)
    result = reshape(out, nt, nm)
    fraction = solver_cache.raman.fraction

    if fraction != 1
        cache.p .= (1 - fraction) .* workspace.rank_cotangent .*
                   conj.(cache.b3 .* cache.b4)
        _mmgnlse_scalar_cp_backproject!(result, cache.p, cache.u2, cache)
        cache.p .= (1 - fraction) .* workspace.rank_cotangent .*
                   conj.(cache.b2 .* cache.b4)
        _mmgnlse_scalar_cp_backproject!(result, cache.p, cache.u3, cache)
        cache.p .= (1 - fraction) .* workspace.rank_cotangent .*
                   conj.(cache.b2 .* cache.b3)
        mul!(cache.k, cache.p, adjoint(cache.u4))
        result .+= conj.(cache.k)
    end

    if !iszero(fraction)
        cache.conv .= cache.b3 .* cache.b4
        cache.ifft_rank! * cache.conv
        _rank_filter_time!(cache.conv, workspace.raman_combined)
        cache.fft_rank! * cache.conv
        cache.p .= fraction .* workspace.rank_cotangent .* conj.(cache.conv)
        _mmgnlse_scalar_cp_backproject!(result, cache.p, cache.u2, cache)

        cache.conv .= fraction .* workspace.rank_cotangent .* conj.(cache.b2)
        cache.ifft_rank! * cache.conv
        _rank_filter_time!(cache.conv, workspace.raman_combined_conj)
        cache.fft_rank! * cache.conv
        cache.p .= cache.conv .* conj.(cache.b4)
        _mmgnlse_scalar_cp_backproject!(result, cache.p, cache.u3, cache)
        cache.p .= cache.conv .* conj.(cache.b3)
        mul!(cache.k, cache.p, adjoint(cache.u4))
        result .+= conj.(cache.k)
    end
    return out
end

function _mmgnlse_validate_adjoint_terminal(lambda_terminal,
                                             parameters::MMGNLSEParameters)
    ndims(lambda_terminal) == 3 || throw(DimensionMismatch(
        "lambda must retain public axes (frequency, mode, polarization)."))
    expected = parameter_field_shape(parameters)
    size(lambda_terminal) == expected || throw(DimensionMismatch(
        "lambda must have public shape $expected; got $(size(lambda_terminal))."))
    all(_mmgnlse_isfinite, lambda_terminal) || throw(ArgumentError(
        "lambda contains a non-finite value."))
    return ComplexF64.(Array(lambda_terminal))
end

function _mmgnlse_validate_adjoint_method(method)
    nameof(typeof(method)) === :Vern9 || throw(ArgumentError(
        "The CPU MMGNLSE adjoint currently requires method=Vern9()."))
    return method
end

function _mmgnlse_validate_adjoint_mode(mode)
    selected = mode isa Symbol ? mode : Symbol(mode)
    selected in (:continuous, :discrete_rk4ip) || throw(ArgumentError(
        "adjoint_mode must be :continuous or :discrete_rk4ip; got " *
        "$(repr(selected))."))
    return selected
end

function _mmgnlse_photon_weights(parameters::MMGNLSEParameters)
    has_dof(parameters.domain.dofs, :time) || throw(ArgumentError(
        "Photon normalization requires the :time degree of freedom."))
    grid = parameters.domain.time_grid
    carrier_frequency_thz = parameters.omega0 / (2π)
    return Float64.(photon_bin_weights(grid.nt, grid.dt, carrier_frequency_thz;
                                      shifted=true))
end

function _mmgnlse_convert_adjoint_units(field_t, parameters, units::Symbol)
    units === :power && return Array(field_t)
    units === :photon || throw(ArgumentError("units must be :power or :photon."))
    weights = _mmgnlse_photon_weights(parameters)
    lambda_w = fftshift(ifft(field_t, 1), 1)
    @inbounds for polarization in axes(lambda_w, 3), mode in axes(lambda_w, 2),
                  frequency in axes(lambda_w, 1)
        weight = weights[frequency]
        lambda_w[frequency, mode, polarization] = weight > 0 ?
            lambda_w[frequency, mode, polarization] / sqrt(weight) :
            zero(eltype(lambda_w))
    end
    return fft(ifftshift(lambda_w, 1), 1)
end

function _mmgnlse_resolve_checkpoint_stride(checkpoint_stride,
                                             nsteps::Integer)
    nsteps > 0 || throw(ArgumentError(
        "A checkpointed adjoint requires at least one forward step."))
    stride = if checkpoint_stride === :auto
        min(Int(nsteps), 256)
    elseif checkpoint_stride isa Integer
        Int(checkpoint_stride)
    else
        throw(ArgumentError(
            "checkpoint_stride must be :auto or a positive integer."))
    end
    stride > 0 || throw(ArgumentError(
        "checkpoint_stride must be positive."))
    return min(stride, Int(nsteps))
end

function _mmgnlse_checkpoint_indices(npoints::Integer,
                                      checkpoint_stride)
    npoints >= 2 || throw(ArgumentError(
        "A checkpointed trajectory requires at least two grid points."))
    stride = _mmgnlse_resolve_checkpoint_stride(
        checkpoint_stride, npoints - 1)
    indices = collect(1:stride:Int(npoints))
    last(indices) == npoints || push!(indices, Int(npoints))
    return indices, stride
end

function _mmgnlse_discrete_save_indices(integration_z::AbstractVector,
                                         saveat)
    if saveat === :steps
        indices = collect(eachindex(integration_z))
        return indices, Float64.(integration_z)
    end
    targets, _ = _mmgnlse_save_targets(
        Float64(last(integration_z)), saveat)
    scale = max(1.0, abs(first(integration_z)), abs(last(integration_z)))
    tolerance = 128eps(Float64) * scale
    indices = Vector{Int}(undef, length(targets))
    for (target_index, target) in enumerate(targets)
        right = searchsortedfirst(integration_z, target)
        candidates = if right <= firstindex(integration_z)
            (firstindex(integration_z),)
        elseif right > lastindex(integration_z)
            (lastindex(integration_z),)
        else
            (right - 1, right)
        end
        distances = map(index -> abs(integration_z[index] - target),
                        candidates)
        selected = candidates[argmin(distances)]
        distances[argmin(distances)] <= tolerance || throw(ArgumentError(
            "The discrete RK4IP adjoint can save only at forward step " *
            "coordinates. Requested z=$target is not on the forward grid."))
        indices[target_index] = selected
    end
    return indices, Float64.(integration_z[indices])
end

function _mmgnlse_discrete_forward_spec(
    parameters::MMGNLSEParameters,
    dz_adj::Real;
    forward_solution,
    initial_field,
    dz_forward,
)
    (forward_solution === nothing) ⊻ (initial_field === nothing) ||
        throw(ArgumentError(
            "Supply exactly one of forward_solution or initial_field."))
    isfinite(dz_adj) && dz_adj > 0 || throw(ArgumentError(
        "dz_adj must be finite and positive."))

    source = forward_solution
    if forward_solution !== nothing
        forward_solution isa MMGNLSESolution || throw(ArgumentError(
            "forward_solution must be an MMGNLSESolution."))
        forward_solution.parameters === parameters || throw(ArgumentError(
            "forward_solution was constructed with a different " *
            "MMGNLSEParameters object."))
        dz_forward === nothing || throw(ArgumentError(
            "dz_forward is only valid when initial_field is supplied."))
        forward_solution.method isa RK4IP || throw(ArgumentError(
            "adjoint_mode=:discrete_rk4ip requires an RK4IP forward solution."))
        if forward_solution.cache !== nothing &&
           hasproperty(forward_solution.cache, :adaptive) &&
           getproperty(forward_solution.cache, :adaptive)
            throw(ArgumentError(
                "adjoint_mode=:discrete_rk4ip currently requires a " *
                "fixed-step forward solution."))
        end
        if forward_solution.cache !== nothing &&
           hasproperty(forward_solution.cache, :precision) &&
           getproperty(forward_solution.cache, :precision) !== :float64
            throw(ArgumentError(
                "adjoint_mode=:discrete_rk4ip replays in Float64; " *
                "rerun the forward solve with precision=:float64."))
        end
        step = Float64(forward_solution.dz)
        initial = _mmgnlse_validate_initial_field(
            forward_solution.initial_field, parameters)
        cached_grid = _mmgnlse_cache_integration_z(forward_solution)
        if cached_grid === nothing
            targets, every_step = _mmgnlse_save_targets(
                Float64(parameters.length), nothing)
            integration_z, _ = _mmgnlse_step_grid(
                Float64(parameters.length), step, targets, every_step)
        else
            integration_z = Float64.(cached_grid)
        end
    else
        dz_forward === nothing && throw(ArgumentError(
            "dz_forward is required when initial_field is supplied."))
        isfinite(dz_forward) && dz_forward > 0 || throw(ArgumentError(
            "dz_forward must be finite and positive."))
        step = Float64(dz_forward)
        initial = _mmgnlse_validate_initial_field(initial_field, parameters)
        targets, every_step = _mmgnlse_save_targets(
            Float64(parameters.length), nothing)
        integration_z, _ = _mmgnlse_step_grid(
            Float64(parameters.length), step, targets, every_step)
    end

    tolerance = 128eps(Float64) * max(1.0, abs(step), abs(float(dz_adj)))
    abs(Float64(dz_adj) - step) <= tolerance || throw(ArgumentError(
        "For adjoint_mode=:discrete_rk4ip, dz_adj must equal the nominal " *
        "forward step ($step); got $(Float64(dz_adj))."))
    first(integration_z) == 0 || throw(ArgumentError(
        "The forward integration grid must begin at z=0."))
    last(integration_z) == parameters.length || throw(ArgumentError(
        "The forward integration grid must end at parameters.length."))
    all(diff(integration_z) .> 0) || throw(ArgumentError(
        "The forward integration grid must be strictly increasing."))
    return initial, integration_z, step, source
end

"""Pull a cotangent through the nonlinear spectral RHS alone."""
function _mmgnlse_nonlinear_spectral_vjp(
    field_w,
    cotangent_w,
    parameters::MMGNLSEParameters,
    cache,
)
    !cache.nonlinear_active && return zeros(ComplexF64, size(field_w))
    field_t = fft(field_w, 1)
    cotangent_t = fft(cotangent_w, 1)
    return ifft(
        _mmgnlse_vjp_nonlinear(
            field_t, cotangent_t, parameters, cache),
        1,
    )
end

"""
Pull a cotangent through one fixed-step RK4IP update.

The stage nonlinearities delegate to `_mmgnlse_vjp_nonlinear`, so the Kerr,
isotropic Raman, and anisotropic Raman conventions are identical to the
continuous adjoint. Only the Runge--Kutta composition is differentiated here.
"""
function _mmgnlse_rk4ip_step_vjp(
    field_w,
    lambda_next,
    parameters::MMGNLSEParameters,
    z0::Real,
    z1::Real,
    cache,
)
    to_midpoint = _mmgnlse_linear_propagator(
        parameters, z0, (z0 + z1) / 2, cache)
    from_midpoint = _mmgnlse_linear_propagator(
        parameters, (z0 + z1) / 2, z1, cache)
    if !cache.nonlinear_active
        return conj.(from_midpoint .* to_midpoint) .* lambda_next
    end

    step = z1 - z0
    midpoint_base = to_midpoint .* field_w
    k1 = to_midpoint .* _mmgnlse_nonlinear_spectral(
        field_w, parameters, cache)
    stage2 = midpoint_base .+ (step / 2) .* k1
    k2 = _mmgnlse_nonlinear_spectral(stage2, parameters, cache)
    stage3 = midpoint_base .+ (step / 2) .* k2
    k3 = _mmgnlse_nonlinear_spectral(stage3, parameters, cache)
    stage4 = from_midpoint .* (midpoint_base .+ step .* k3)

    propagated = conj.(from_midpoint) .* lambda_next
    bar_midpoint = copy(propagated)
    bar_k1 = (step / 6) .* propagated
    bar_k2 = (step / 3) .* propagated
    bar_k3 = (step / 3) .* propagated

    bar_stage = _mmgnlse_nonlinear_spectral_vjp(
        stage4, (step / 6) .* lambda_next, parameters, cache)
    bar_midpoint .+= conj.(from_midpoint) .* bar_stage
    bar_k3 .+= step .* conj.(from_midpoint) .* bar_stage

    bar_stage = _mmgnlse_nonlinear_spectral_vjp(
        stage3, bar_k3, parameters, cache)
    bar_midpoint .+= bar_stage
    bar_k2 .+= (step / 2) .* bar_stage

    bar_stage = _mmgnlse_nonlinear_spectral_vjp(
        stage2, bar_k2, parameters, cache)
    bar_midpoint .+= bar_stage
    bar_k1 .+= (step / 2) .* bar_stage

    lambda_previous = _mmgnlse_nonlinear_spectral_vjp(
        field_w, conj.(to_midpoint) .* bar_k1, parameters, cache)
    lambda_previous .+= conj.(to_midpoint) .* bar_midpoint
    return lambda_previous
end

function _mmgnlse_store_discrete_adjoint!(
    fields,
    output_index::Integer,
    lambda_w,
    parameters,
    units,
)
    lambda_t = fft(lambda_w, 1)
    @views fields[:, :, :, output_index] .=
        _mmgnlse_convert_adjoint_units(lambda_t, parameters, units)
    return fields
end

function _mmgnlse_solve_adjoint_discrete_cpu(
    lambda_terminal,
    parameters::MMGNLSEParameters,
    dz_adj::Real;
    forward_solution,
    initial_field,
    dz_forward,
    units::Symbol,
    saveat,
    checkpoint_stride,
)
    initial, integration_z, forward_step, source =
        _mmgnlse_discrete_forward_spec(
            parameters, dz_adj;
            forward_solution, initial_field, dz_forward)
    checkpoint_indices, resolved_stride = _mmgnlse_checkpoint_indices(
        length(integration_z), checkpoint_stride)
    save_indices, saved_z = _mmgnlse_discrete_save_indices(
        integration_z, saveat)
    save_lookup = Dict(index => output for
                       (output, index) in enumerate(save_indices))

    cache = _mmgnlse_solver_cache(parameters)
    nt, nm, np = size(initial)
    checkpoints = Array{ComplexF64,4}(
        undef, nt, nm, np, length(checkpoint_indices))
    field_w = ifft(initial, 1)
    @views checkpoints[:, :, :, 1] .= field_w
    checkpoint_cursor = 2
    for step_index in 1:length(integration_z)-1
        field_w = _mmgnlse_rk4ip_step(
            field_w, parameters,
            integration_z[step_index], integration_z[step_index + 1], cache)
        if checkpoint_cursor <= length(checkpoint_indices) &&
           step_index + 1 == checkpoint_indices[checkpoint_cursor]
            @views checkpoints[:, :, :, checkpoint_cursor] .= field_w
            checkpoint_cursor += 1
        end
    end
    checkpoint_cursor == length(checkpoint_indices) + 1 || error(
        "Not every forward checkpoint was stored.")

    maximum_segment_steps = maximum(diff(checkpoint_indices))
    segment_states = Array{ComplexF64,4}(
        undef, nt, nm, np, maximum_segment_steps + 1)
    lambda_w = ifftshift(
        _mmgnlse_validate_adjoint_terminal(lambda_terminal, parameters), 1)
    fields = Array{ComplexF64,4}(
        undef, nt, nm, np, length(saved_z))
    if haskey(save_lookup, length(integration_z))
        _mmgnlse_store_discrete_adjoint!(
            fields, save_lookup[length(integration_z)], lambda_w,
            parameters, units)
    end

    for segment_index in length(checkpoint_indices)-1:-1:1
        first_state = checkpoint_indices[segment_index]
        last_state = checkpoint_indices[segment_index + 1]
        segment_steps = last_state - first_state
        @views segment_states[:, :, :, 1] .=
            checkpoints[:, :, :, segment_index]
        replay = copy(@view segment_states[:, :, :, 1])
        for local_step in 1:segment_steps
            global_step = first_state + local_step - 1
            replay = _mmgnlse_rk4ip_step(
                replay, parameters,
                integration_z[global_step],
                integration_z[global_step + 1], cache)
            @views segment_states[:, :, :, local_step + 1] .= replay
        end
        for local_step in segment_steps:-1:1
            global_step = first_state + local_step - 1
            lambda_w = _mmgnlse_rk4ip_step_vjp(
                @view(segment_states[:, :, :, local_step]),
                lambda_w, parameters,
                integration_z[global_step],
                integration_z[global_step + 1], cache)
            if haskey(save_lookup, global_step)
                _mmgnlse_store_discrete_adjoint!(
                    fields, save_lookup[global_step], lambda_w,
                    parameters, units)
            end
        end
    end

    return MMGNLSEAdjointSolution(
        z=saved_z,
        fields=fields,
        parameters=parameters,
        units=units,
        cache=(
            backend=:cpu,
            adjoint_mode=:discrete_rk4ip,
            forward_solution=source,
            integration_z=integration_z,
            forward_step,
            checkpoint_stride=resolved_stride,
            checkpoint_z=integration_z[checkpoint_indices],
            checkpoint_count=length(checkpoint_indices),
            replay_peak_planes=maximum_segment_steps + 1,
            frame=:spectral,
            core=cache,
        ),
    )
end

"""
    solve_adjoint(lambda, parameters, dz_adj;
                  forward_solution=nothing, initial_field=nothing,
                  dz_forward=nothing, method=Vern9(), units=:power,
                  saveat=nothing, backend=:cpu, device=nothing,
                  synchronize=true, adjoint_mode=:continuous,
                  checkpoint_stride=:auto)

Integrate the continuous MMGNLSE adjoint backward in the interaction picture
with fixed-step `Vern9`. Dispersion and longitudinally varying gain/loss are
applied through their exact diagonal propagators; `Vern9` integrates only the
nonlinear VJP.
`lambda` is the centered, negative-to-positive spectral Wirtinger derivative
`dJ/dconj(Aomega(L))`, with public shape `(frequency, mode, polarization)`.

Exactly one forward source is required. Supply an existing `forward_solution`,
or supply `initial_field` together with `dz_forward`. Sparse forward solutions
are deterministically replayed with `saveat=:steps` for interpolation. The
returned trajectory is always ordered in ascending z. With `units=:photon`,
saved gradients are `dJ/dconj(B) = dJ/dconj(A)/sqrt(weight)`; bins at
nonpositive absolute frequency are zero. CUDA variants return the same
host-array solution representation as the CPU backend and use the same CP
policy as `solve_mmgnlse`: `:cuda` is the baseline CP implementation,
`:cuda_cp_optimized` enables rank-agnostic CP optimizations only, and
`:cuda_optimized` additionally enables rank-tuned choices.

Set `adjoint_mode=:discrete_rk4ip` to differentiate the exact fixed-step
RK4IP forward map in Float64. Mixed-precision forward solutions are rejected
because their replay would use a different numerical map. This mode requires
`dz_adj == dz_forward` (or the nominal
step of `forward_solution`) and replays bounded segments between uniformly
spaced checkpoints. `checkpoint_stride=:auto` currently uses at most 256
forward steps per replay segment. The discrete nonlinear pullback uses the
same analytic Kerr and Raman VJP as the continuous adjoint.
"""
function solve_adjoint(lambda_terminal::AbstractArray{<:Number,3},
                       parameters::MMGNLSEParameters,
                       dz_adj::Real;
                       forward_solution=nothing,
                       initial_field=nothing,
                       dz_forward=nothing,
                       method=Vern9(),
                       units::Symbol=:power,
                       saveat=nothing,
                       backend=:cpu,
                       device=nothing,
                       synchronize::Bool=true,
                       adjoint_mode::Symbol=:continuous,
                       checkpoint_stride=:auto)
    selected_adjoint_mode = _mmgnlse_validate_adjoint_mode(adjoint_mode)
    selected_backend = _mmgnlse_validate_solver_backend(backend, parameters)
    selected_backend === :cuda_cp_symmetric_experimental &&
        throw(ArgumentError(
            "backend=:cuda_cp_symmetric_experimental is forward-only and " *
            "does not alter the CUDA adjoint pathways."))
    if selected_backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
        return _mmgnlse_solve_adjoint_cuda(
            lambda_terminal, parameters, dz_adj;
            forward_solution, initial_field, dz_forward, method, units,
            saveat, device, synchronize, backend=selected_backend,
            adjoint_mode=selected_adjoint_mode, checkpoint_stride)
    end
    (forward_solution === nothing) ⊻ (initial_field === nothing) ||
        throw(ArgumentError(
            "Supply exactly one of forward_solution or initial_field."))
    isfinite(dz_adj) && dz_adj > 0 || throw(ArgumentError(
        "dz_adj must be finite and positive."))
    units in (:power, :photon) || throw(ArgumentError(
        "units must be :power or :photon."))
    if selected_adjoint_mode === :discrete_rk4ip
        return _mmgnlse_solve_adjoint_discrete_cpu(
            lambda_terminal, parameters, dz_adj;
            forward_solution, initial_field, dz_forward, units, saveat,
            checkpoint_stride)
    end
    _mmgnlse_validate_adjoint_method(method)

    forward = if forward_solution !== nothing
        forward_solution isa MMGNLSESolution || throw(ArgumentError(
            "forward_solution must be an MMGNLSESolution."))
        dz_forward === nothing || throw(ArgumentError(
            "dz_forward is only valid when initial_field is supplied."))
        _mmgnlse_dense_forward_for_adjoint(forward_solution, parameters)
    else
        dz_forward === nothing && throw(ArgumentError(
            "dz_forward is required when initial_field is supplied."))
        isfinite(dz_forward) && dz_forward > 0 || throw(ArgumentError(
            "dz_forward must be finite and positive."))
        solve_mmgnlse(initial_field, parameters, dz_forward; saveat=:steps)
    end

    first(forward.z) == 0 || throw(ArgumentError(
        "The forward trajectory must begin at z=0."))
    last(forward.z) == parameters.length || throw(ArgumentError(
        "The forward trajectory must end at parameters.length."))
    all(diff(forward.z) .> 0) || throw(ArgumentError(
        "The forward trajectory z values must be strictly increasing."))

    lambda_w = _mmgnlse_validate_adjoint_terminal(lambda_terminal, parameters)
    lambda_raw = ifftshift(lambda_w, 1)

    T = promote_type(eltype(forward.z), typeof(float(dz_adj)))
    targets, every_step = _mmgnlse_save_targets(T(parameters.length), saveat)
    integration_z, saved_z = _mmgnlse_step_grid(
        T(parameters.length), T(dz_adj), targets, every_step)
    descending_saved = reverse(saved_z)
    descending_stops = reverse(integration_z[1:end-1])

    adjoint_core_cache = _mmgnlse_solver_cache(parameters)
    forward_interaction = _mmgnlse_forward_interaction_fields(
        forward, parameters, adjoint_core_cache)
    terminal_propagator = _mmgnlse_linear_propagator(
        parameters, zero(T), T(parameters.length), adjoint_core_cache)
    # For dλ/dz = -L†λ - N′†λ, the homogeneous adjoint propagator from
    # zero to z is inv(conj(P(0,z))). Thus λ̃ = conj(P(0,z)) .* λ.
    lambda_interaction = conj.(terminal_propagator) .* lambda_raw
    cp_workspace = _mmgnlse_use_scalar_cp_fast_path(
        parameters, forward.initial_field) ?
        _mmgnlse_scalar_cp_adjoint_workspace(
            similar(lambda_interaction), parameters.S,
            adjoint_core_cache.raman) :
        nothing

    function rhs!(dlambda, lambda_tilde, _, z)
        if !adjoint_core_cache.nonlinear_active
            fill!(dlambda, zero(eltype(dlambda)))
            return nothing
        end
        coordinate = T(z)
        if cp_workspace === nothing
            forward_tilde = _mmgnlse_interpolate_interaction_field(
                forward_interaction, forward.z, coordinate)
            forward_propagator = _mmgnlse_linear_propagator(
                parameters, zero(T), coordinate, adjoint_core_cache)
            field_t = fft(forward_propagator .* forward_tilde, 1)
            lambda_raw_z = lambda_tilde ./ conj.(forward_propagator)
            lambda_t = fft(lambda_raw_z, 1)
            nonlinear_vjp_t = _mmgnlse_vjp_nonlinear(
                field_t, lambda_t, parameters, adjoint_core_cache)
            dlambda .=
                -conj.(forward_propagator) .* ifft(nonlinear_vjp_t, 1)
        else
            _mmgnlse_interpolate_interaction_field!(
                cp_workspace.interpolated, forward_interaction,
                forward.z, coordinate)
            cp_workspace.propagator .= exp.(
                adjoint_core_cache.beta_operator .* coordinate)
            cp_workspace.field_t .=
                cp_workspace.propagator .* cp_workspace.interpolated
            cp_workspace.fft_field! * cp_workspace.field_t
            cp_workspace.lambda_t .=
                lambda_tilde ./ conj.(cp_workspace.propagator)
            cp_workspace.fft_lambda! * cp_workspace.lambda_t
            _mmgnlse_scalar_cp_vjp_nonlinear!(
                cp_workspace.vjp_t, cp_workspace.field_t,
                cp_workspace.lambda_t, adjoint_core_cache, cp_workspace)
            cp_workspace.spectral_scratch .= cp_workspace.vjp_t
            cp_workspace.ifft_vjp! * cp_workspace.spectral_scratch
            dlambda .= -conj.(cp_workspace.propagator) .*
                       cp_workspace.spectral_scratch
        end
        return nothing
    end

    problem = ODEProblem(rhs!, lambda_interaction,
                         (T(parameters.length), zero(T)))
    ode_solution = DifferentialEquations.solve(
        problem, method; dt=T(dz_adj), adaptive=false,
        saveat=descending_saved, tstops=descending_stops,
        save_everystep=false, dense=false)
    ode_solution.retcode == ReturnCode.Success || error(
        "Vern9 adjoint solve failed with retcode $(ode_solution.retcode).")
    length(ode_solution.u) == length(saved_z) || error(
        "Adjoint solver saved $(length(ode_solution.u)) states; expected $(length(saved_z)).")

    nt, nm, np = parameter_field_shape(parameters)
    fields = Array{ComplexF64,4}(undef, nt, nm, np, length(saved_z))
    @inbounds for (index, (coordinate, state)) in enumerate(
            zip(saved_z, reverse(ode_solution.u)))
        forward_propagator = _mmgnlse_linear_propagator(
            parameters, zero(T), coordinate, adjoint_core_cache)
        lambda_raw_z = state ./ conj.(forward_propagator)
        lambda_t = fft(lambda_raw_z, 1)
        fields[:, :, :, index] =
            _mmgnlse_convert_adjoint_units(lambda_t, parameters, units)
    end
    return MMGNLSEAdjointSolution(
        z=collect(T, saved_z), fields=fields, parameters=parameters,
        units=units, cache=(forward_solution=forward,
                            integration_z=integration_z,
                            frame=:interaction,
                            core=adjoint_core_cache,
                            ode_solution=ode_solution))
end

function _mmgnlse_noise_specification(noise,
                                      target::NTuple{3,<:Integer})
    specification = if noise isa Real
        reshape([float(noise)], 1, 1, 1)
    elseif noise isa AbstractArray{<:Real}
        1 <= ndims(noise) <= 3 || throw(ArgumentError(
            "noise must have at most three positional axes (frequency, mode, polarization)."))
        isempty(noise) && throw(ArgumentError("noise cannot be empty."))
        reshape(float.(Array(noise)), size(noise)...,
                ntuple(_ -> 1, 3 - ndims(noise))...)
    else
        throw(ArgumentError(
            "noise must be a real scalar or positional real array."))
    end
    all(isfinite, specification) || throw(ArgumentError(
        "noise Fano factors must be finite."))
    all(>=(0), specification) || throw(ArgumentError(
        "noise Fano factors must be nonnegative."))
    for axis in 1:3
        size(specification, axis) in (1, target[axis]) ||
            throw(DimensionMismatch(
                "noise axis $axis has size $(size(specification, axis)); " *
                "expected 1 or $(target[axis])."))
    end
    result = similar(specification, target)
    result .= specification
    return result
end

function _mmgnlse_sample_initial_noise(parameters::MMGNLSEParameters,
                                        noise,
                                        ntraj::Integer,
                                        rng::AbstractRNG)
    ntraj > 0 || throw(ArgumentError("ntraj must be positive."))
    target = parameter_field_shape(parameters)
    fano = _mmgnlse_noise_specification(noise, target)
    weights = _mmgnlse_photon_weights(parameters)
    delta_b = zeros(ComplexF64, target..., ntraj)
    @inbounds for trajectory in 1:ntraj,
                  polarization in 1:target[3], mode in 1:target[2],
                  frequency in 1:target[1]
        weights[frequency] > 0 || continue
        scale = sqrt(fano[frequency, mode, polarization]) / 2
        delta_b[frequency, mode, polarization, trajectory] =
            scale * (randn(rng) + im * randn(rng))
    end
    return fano, weights, delta_b
end

function _mmgnlse_add_photon_noise(initial_field, delta_b, weights)
    spectrum = fftshift(ifft(initial_field, 1), 1)
    noisy_spectrum = copy(spectrum)
    @inbounds for polarization in axes(spectrum, 3), mode in axes(spectrum, 2),
                  frequency in axes(spectrum, 1)
        weight = weights[frequency]
        weight > 0 || continue
        noisy_spectrum[frequency, mode, polarization] +=
            delta_b[frequency, mode, polarization] / sqrt(weight)
    end
    return fft(ifftshift(noisy_spectrum, 1), 1)
end

"""
    solve_mmgnlse_stochastic(initial_field, parameters, dz, ntraj;
                             noise=1, seed=nothing, rng=nothing, kwargs...)

Propagate an ensemble with initial complex Gaussian noise in photon-bin
coordinates. `noise` is the Fano factor: a scalar is uniform, a vector is
frequency-dependent, and positional singleton arrays may additionally select
mode and polarization. Each independent component is sampled as
`delta_B = sqrt(F)/2 * (xi1 + i*xi2)`. No distributed Langevin source is added.
"""
function solve_mmgnlse_stochastic(initial_field,
                                   parameters::MMGNLSEParameters,
                                   dz::Real,
                                   ntraj::Integer;
                                   noise=1,
                                   seed=nothing,
                                   rng=nothing,
                                   kwargs...)
    ntraj > 0 || throw(ArgumentError("ntraj must be positive."))
    seed === nothing || seed isa Integer || throw(ArgumentError(
        "seed must be an integer or nothing."))
    seed === nothing || rng === nothing || throw(ArgumentError(
        "Supply at most one of seed and rng."))
    active_rng = seed === nothing ?
        (rng === nothing ? Random.default_rng() : rng) : MersenneTwister(seed)
    active_rng isa AbstractRNG || throw(ArgumentError(
        "rng must be an AbstractRNG."))

    mean_field = _mmgnlse_validate_initial_field(initial_field, parameters)
    fano, weights, delta_b = _mmgnlse_sample_initial_noise(
        parameters, noise, ntraj, active_rng)

    first_solution = nothing
    fields = nothing
    for trajectory in 1:ntraj
        noisy_field = _mmgnlse_add_photon_noise(
            mean_field, @view(delta_b[:, :, :, trajectory]), weights)
        solution = solve_mmgnlse(noisy_field, parameters, dz; kwargs...)
        if trajectory == 1
            first_solution = solution
            nt, nm, np, nz = size(solution.fields)
            fields = Array{ComplexF64,5}(undef, nt, nm, np, nz, ntraj)
        else
            solution.z == first_solution.z || error(
                "Stochastic trajectories produced inconsistent saved z grids.")
        end
        fields[:, :, :, :, trajectory] = solution.fields
    end

    return MMGNLSEStochasticSolution(
        z=copy(first_solution.z), fields=fields, parameters=parameters,
        noise=(fano=fano, photon_noise=delta_b, weights=weights, seed=seed))
end

function _mmgnlse_stochastic_saved_fields(solution::MMGNLSEStochasticSolution,
                                          z, trajectory)
    if z === :all
        trajectory === :all && return solution.fields
        return solution.fields[:, :, :, :, trajectory]
    end
    index = _mmgnlse_z_index(solution.z, z)
    trajectory === :all && return solution.fields[:, :, :, index, :]
    return solution.fields[:, :, :, index, trajectory]
end

function get_temporal_field(solution::MMGNLSEStochasticSolution;
                            z=:final, trajectory=:all,
                            units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    field_t = _mmgnlse_stochastic_saved_fields(solution, z, trajectory)
    units === :power && return field_t
    field_w = fftshift(ifft(field_t, 1), 1)
    photon_w = _mmgnlse_convert_spectral_units(
        field_w, solution.parameters;
        from=:power, to=:photon, role=:forward)
    return fft(ifftshift(photon_w, 1), 1)
end

function get_spectral_field(solution::MMGNLSEStochasticSolution;
                            z=:final, trajectory=:all,
                            units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    field_t = _mmgnlse_stochastic_saved_fields(solution, z, trajectory)
    field_w = fftshift(ifft(field_t, 1), 1)
    return _mmgnlse_convert_spectral_units(
        field_w, solution.parameters;
        from=:power, to=units, role=:forward)
end
