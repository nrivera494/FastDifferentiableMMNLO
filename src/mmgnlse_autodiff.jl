function _mmgnlse_ad_polarization_stack(x, y)
    return cat(reshape(x, size(x)..., 1),
               reshape(y, size(y)..., 1);
               dims=3)
end

function _mmgnlse_ad_dense_spatial_instantaneous(field_t, S)
    nt, nm, np = size(field_t)
    Ax = field_t[:, :, 1]
    x_modes = map(1:nm) do i1
        terms = map(CartesianIndices((nm, nm, nm))) do index
            i2, i3, i4 = Tuple(index)
            s = S[i1, i2, i3, i4]
            x = s .* Ax[:, i2] .* Ax[:, i3] .* conj.(Ax[:, i4])
            if np == 1
                x
            else
                Ay = field_t[:, :, 2]
                x .+ s .* (
                    (2 / 3) .* Ax[:, i2] .* Ay[:, i3] .* conj.(Ay[:, i4]) .+
                    (1 / 3) .* Ay[:, i2] .* Ay[:, i3] .* conj.(Ax[:, i4]))
            end
        end
        reduce(+, terms)
    end
    x_field = hcat(x_modes...)
    np == 1 && return reshape(x_field, nt, nm, 1)

    Ay = field_t[:, :, 2]
    y_modes = map(1:nm) do i1
        terms = map(CartesianIndices((nm, nm, nm))) do index
            i2, i3, i4 = Tuple(index)
            s = S[i1, i2, i3, i4]
            s .* (
                Ay[:, i2] .* Ay[:, i3] .* conj.(Ay[:, i4]) .+
                (2 / 3) .* Ay[:, i2] .* Ax[:, i3] .* conj.(Ax[:, i4]) .+
                (1 / 3) .* Ax[:, i2] .* Ax[:, i3] .* conj.(Ay[:, i4]))
        end
        reduce(+, terms)
    end
    return _mmgnlse_ad_polarization_stack(x_field, hcat(y_modes...))
end

function _mmgnlse_ad_cp_spatial_instantaneous(field_t, cp)
    nt, nm, np = size(field_t)
    Ax = field_t[:, :, 1]
    B2x, B3x, B4x, WU1 = _cp_projected_fields(Ax, cp)
    Px = (B2x .* B3x .* B4x) * transpose(WU1)
    np == 1 && return reshape(Px, nt, nm, 1)

    Ay = field_t[:, :, 2]
    _, U2, U3, U4 = cp.U
    B2y = Ay * U2
    B3y = Ay * U3
    B4y = conj.(Ay) * U4
    Px = (B2x .* B3x .* B4x .+
          (2 / 3) .* B2x .* B3y .* B4y .+
          (1 / 3) .* B2y .* B3y .* B4x) * transpose(WU1)
    Py = (B2y .* B3y .* B4y .+
          (2 / 3) .* B2y .* B3x .* B4x .+
          (1 / 3) .* B2x .* B3x .* B4y) * transpose(WU1)
    return _mmgnlse_ad_polarization_stack(Px, Py)
end

function _mmgnlse_ad_dense_component_instantaneous(field_t, tensor)
    nt, nm, np = size(field_t)
    field = reshape(field_t, nt, nm * np)
    S = flatten_component_tensor(tensor)
    nc = size(field, 2)
    modes = map(1:nc) do i1
        terms = map(CartesianIndices((nc, nc, nc))) do index
            i2, i3, i4 = Tuple(index)
            S[i1, i2, i3, i4] .* field[:, i2] .* field[:, i3] .*
            conj.(field[:, i4])
        end
        reduce(+, terms)
    end
    return reshape(hcat(modes...), nt, nm, np)
end

function _mmgnlse_ad_cp_component_instantaneous(field_t, cp)
    nt, nm, np = size(field_t)
    field = reshape(field_t, nt, nm * np)
    B2, B3, B4, WU1 = _cp_projected_fields(field, cp)
    return reshape((B2 .* B3 .* B4) * transpose(WU1), nt, nm, np)
end

function _mmgnlse_ad_instantaneous(field_t, overlap::SpatialOverlap)
    return _mmgnlse_ad_dense_spatial_instantaneous(field_t, overlap.values)
end

function _mmgnlse_ad_instantaneous(field_t, overlap::ComponentOverlap)
    return _mmgnlse_ad_dense_component_instantaneous(field_t, overlap.values)
end

function _mmgnlse_ad_instantaneous(field_t, overlap::MMGNLSECPDecomposition)
    overlap.metadata.layout === :spatial &&
        return _mmgnlse_ad_cp_spatial_instantaneous(field_t, overlap)
    overlap.metadata.layout === :component &&
        return _mmgnlse_ad_cp_component_instantaneous(field_t, overlap)
    throw(ArgumentError(
        "Unsupported CP overlap layout $(overlap.metadata.layout)."))
end

function _mmgnlse_ad_dense_spatial_delayed(field_t, S, kernels)
    nt, nm, np = size(field_t)
    Ax = field_t[:, :, 1]
    pairs = CartesianIndices((nm, nm))
    if np == 1
        q = reshape(hcat(map(pairs) do index
            i3, i4 = Tuple(index)
            _mmgnlse_filter_raman(
                Ax[:, i3] .* conj.(Ax[:, i4]), kernels.ha .+ kernels.hb)
        end...), nt, nm, nm)
        modes = map(1:nm) do i1
            terms = map(CartesianIndices((nm, nm, nm))) do index
                i2, i3, i4 = Tuple(index)
                S[i1, i2, i3, i4] .* Ax[:, i2] .* q[:, i3, i4]
            end
            reduce(+, terms)
        end
        return reshape(hcat(modes...), nt, nm, 1)
    end

    Ay = field_t[:, :, 2]
    qa = reshape(hcat(map(pairs) do index
        i3, i4 = Tuple(index)
        xx = Ax[:, i3] .* conj.(Ax[:, i4])
        yy = Ay[:, i3] .* conj.(Ay[:, i4])
        _mmgnlse_filter_raman(xx .+ yy, kernels.ha)
    end...), nt, nm, nm)
    qxx = reshape(hcat(map(pairs) do index
        i3, i4 = Tuple(index)
        _mmgnlse_filter_raman(
            Ax[:, i3] .* conj.(Ax[:, i4]), kernels.hb)
    end...), nt, nm, nm)
    qyy = reshape(hcat(map(pairs) do index
        i3, i4 = Tuple(index)
        _mmgnlse_filter_raman(
            Ay[:, i3] .* conj.(Ay[:, i4]), kernels.hb)
    end...), nt, nm, nm)
    qxy = reshape(hcat(map(pairs) do index
        i3, i4 = Tuple(index)
        _mmgnlse_filter_raman(
            Ax[:, i3] .* conj.(Ay[:, i4]) .+
            Ay[:, i3] .* conj.(Ax[:, i4]), kernels.hb)
    end...), nt, nm, nm)

    x_modes = map(1:nm) do i1
        terms = map(CartesianIndices((nm, nm, nm))) do index
            i2, i3, i4 = Tuple(index)
            s = S[i1, i2, i3, i4]
            s .* (Ax[:, i2] .* (qa[:, i3, i4] .+ qxx[:, i3, i4]) .+
                   0.5 .* Ay[:, i2] .* qxy[:, i3, i4])
        end
        reduce(+, terms)
    end
    y_modes = map(1:nm) do i1
        terms = map(CartesianIndices((nm, nm, nm))) do index
            i2, i3, i4 = Tuple(index)
            s = S[i1, i2, i3, i4]
            s .* (Ay[:, i2] .* (qa[:, i3, i4] .+ qyy[:, i3, i4]) .+
                   0.5 .* Ax[:, i2] .* qxy[:, i3, i4])
        end
        reduce(+, terms)
    end
    return _mmgnlse_ad_polarization_stack(
        hcat(x_modes...), hcat(y_modes...))
end

function _mmgnlse_ad_cp_spatial_delayed_part(field_t, cp, kernel, part)
    nt, nm, np = size(field_t)
    Ax = field_t[:, :, 1]
    B2x, B3x, B4x, WU1 = _cp_projected_fields(Ax, cp)
    if np == 1
        q = _mmgnlse_filter_raman(B3x .* B4x, kernel)
        return reshape((B2x .* q) * transpose(WU1), nt, nm, 1)
    end

    Ay = field_t[:, :, 2]
    _, U2, U3, U4 = cp.U
    B2y = Ay * U2
    B3y = Ay * U3
    B4y = conj.(Ay) * U4
    if part === :isotropic
        q = _mmgnlse_filter_raman(B3x .* B4x .+ B3y .* B4y, kernel)
        return _mmgnlse_ad_polarization_stack(
            (B2x .* q) * transpose(WU1),
            (B2y .* q) * transpose(WU1))
    end

    qxx = _mmgnlse_filter_raman(B3x .* B4x, kernel)
    qyy = _mmgnlse_filter_raman(B3y .* B4y, kernel)
    qxy = _mmgnlse_filter_raman(B3x .* B4y .+ B3y .* B4x, kernel)
    return _mmgnlse_ad_polarization_stack(
        (B2x .* qxx .+ 0.5 .* B2y .* qxy) * transpose(WU1),
        (B2y .* qyy .+ 0.5 .* B2x .* qxy) * transpose(WU1))
end

function _mmgnlse_ad_cp_spatial_delayed(field_t, cp, kernels)
    isotropic = _mmgnlse_ad_cp_spatial_delayed_part(
        field_t, cp, kernels.ha, :isotropic)
    anisotropic = _mmgnlse_ad_cp_spatial_delayed_part(
        field_t, cp, kernels.hb, :anisotropic)
    return isotropic .+ anisotropic
end

function _mmgnlse_ad_dense_component_delayed(field_t, tensor, kernels)
    nt, nm, np = size(field_t)
    field = reshape(field_t, nt, nm * np)
    S = flatten_component_tensor(tensor)
    nc = size(field, 2)
    kernel = kernels.ha .+ kernels.hb
    pairs = CartesianIndices((nc, nc))
    q = reshape(hcat(map(pairs) do index
        i3, i4 = Tuple(index)
        _mmgnlse_filter_raman(
            field[:, i3] .* conj.(field[:, i4]), kernel)
    end...), nt, nc, nc)
    modes = map(1:nc) do i1
        terms = map(CartesianIndices((nc, nc, nc))) do index
            i2, i3, i4 = Tuple(index)
            S[i1, i2, i3, i4] .* field[:, i2] .* q[:, i3, i4]
        end
        reduce(+, terms)
    end
    return reshape(hcat(modes...), nt, nm, np)
end

function _mmgnlse_ad_cp_component_delayed(field_t, cp, kernels)
    nt, nm, np = size(field_t)
    field = reshape(field_t, nt, nm * np)
    B2, B3, B4, WU1 = _cp_projected_fields(field, cp)
    q = _mmgnlse_filter_raman(B3 .* B4, kernels.ha .+ kernels.hb)
    return reshape((B2 .* q) * transpose(WU1), nt, nm, np)
end

function _mmgnlse_ad_delayed(field_t, overlap::SpatialOverlap, kernels)
    return _mmgnlse_ad_dense_spatial_delayed(
        field_t, overlap.values, kernels)
end

function _mmgnlse_ad_delayed(
    field_t, overlap::ComponentOverlap, kernels)
    return _mmgnlse_ad_dense_component_delayed(
        field_t, overlap.values, kernels)
end

function _mmgnlse_ad_delayed(
    field_t, overlap::MMGNLSECPDecomposition, kernels)
    overlap.metadata.layout === :spatial &&
        return _mmgnlse_ad_cp_spatial_delayed(field_t, overlap, kernels)
    overlap.metadata.layout === :component &&
        return _mmgnlse_ad_cp_component_delayed(field_t, overlap, kernels)
    throw(ArgumentError(
        "Unsupported CP overlap layout $(overlap.metadata.layout)."))
end

function _mmgnlse_ad_nonlinear_spectral(field_w, parameters, cache)
    field_t = fft(field_w, 1)
    instantaneous = _mmgnlse_ad_instantaneous(field_t, parameters.S)
    fraction = cache.raman.fraction
    cubic = if iszero(fraction)
        instantaneous
    else
        delayed = _mmgnlse_ad_delayed(field_t, parameters.S, cache.raman)
        (1 - fraction) .* instantaneous .+ fraction .* delayed
    end
    return cache.nonlinear_prefactor .* ifft(cubic, 1)
end

function _mmgnlse_ad_step(field_w, data, parameters, cache)
    data.nonlinear || return data.full .* field_w
    step = data.z1 - data.z0
    midpoint_base = data.to_midpoint .* field_w
    k1 = data.to_midpoint .* _mmgnlse_ad_nonlinear_spectral(
        field_w, parameters, cache)
    k2 = _mmgnlse_ad_nonlinear_spectral(
        midpoint_base .+ step / 2 .* k1, parameters, cache)
    k3 = _mmgnlse_ad_nonlinear_spectral(
        midpoint_base .+ step / 2 .* k2, parameters, cache)
    endpoint = data.from_midpoint .* (midpoint_base .+ step .* k3)
    k4 = _mmgnlse_ad_nonlinear_spectral(endpoint, parameters, cache)
    return data.from_midpoint .* (
        midpoint_base .+ step / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3)) .+
        step / 6 .* k4
end

function _mmgnlse_ad_step_data(parameters, z0, z1, cache)
    midpoint = (z0 + z1) / 2
    return (; z0, z1, nonlinear=cache.nonlinear_active,
             full=_mmgnlse_linear_propagator(parameters, z0, z1, cache),
             to_midpoint=_mmgnlse_linear_propagator(
                 parameters, z0, midpoint, cache),
             from_midpoint=_mmgnlse_linear_propagator(
                 parameters, midpoint, z1, cache))
end

function _mmgnlse_ad_step_grid(parameters, dz)
    steps, _ = _mmgnlse_step_grid(parameters.length, dz, Float64[], false)
    return steps
end

function _mmgnlse_ad_propagate_range(
    initial_w,
    steps,
    first_step::Int,
    last_step::Int,
    parameters,
    cache)
    field_w = initial_w
    for index in first_step:last_step
        data = ChainRulesCore.ignore_derivatives() do
            _mmgnlse_ad_step_data(
                parameters, steps[index], steps[index + 1], cache)
        end
        field_w = _mmgnlse_ad_step(field_w, data, parameters, cache)
    end
    return field_w
end

function _mmgnlse_ad_checkpoint_blocks(steps, checkpoint_steps::Int,
                                        parameters, cache)
    number_of_steps = length(steps) - 1
    return map(1:checkpoint_steps:number_of_steps) do first_step
        last_step = min(first_step + checkpoint_steps - 1, number_of_steps)
        constants = (steps, first_step, last_step, parameters, cache)
        field_w -> begin
            block_steps, block_first, block_last, block_parameters, block_cache =
                ChainRulesCore.ignore_derivatives() do
                    constants
                end
            _mmgnlse_ad_propagate_range(
                field_w, block_steps, block_first, block_last,
                block_parameters, block_cache)
        end
    end
end

function _mmgnlse_ad_propagate(
    initial_w,
    steps,
    parameters,
    cache,
    checkpoint_blocks)
    if checkpoint_blocks === nothing
        return _mmgnlse_ad_propagate_range(
            initial_w, steps, 1, length(steps) - 1, parameters, cache)
    end

    field_w = initial_w
    for block in checkpoint_blocks
        field_w = Zygote.checkpointed(block, field_w)
    end
    return field_w
end

const _MMGNLSE_AUTODIFF_AUTO_CHECKPOINT_STEPS = 10

function _mmgnlse_ad_checkpoint_steps(checkpointing)
    checkpointing === :auto &&
        return _MMGNLSE_AUTODIFF_AUTO_CHECKPOINT_STEPS
    checkpointing === :none && return nothing
    checkpointing isa Bool && throw(ArgumentError(
        "checkpointing must be :auto, :none, or a positive integer block size."))
    checkpointing isa Integer || throw(ArgumentError(
        "checkpointing must be :auto, :none, or a positive integer block size."))
    checkpointing > 0 || throw(ArgumentError(
        "The checkpointing block size must be positive."))
    return Int(checkpointing)
end

"""
    get_autodiff_adjoint(observable, initial_field, parameters, dz;
                         method=RK4IP(), units=:power,
                         checkpointing=:auto)

Return the centered spectral cotangent at `z=0` computed by reverse-mode
automatic differentiation through the fixed-step MMGNLSE propagation. The
result has public axes `(frequency, mode, polarization)` and matches
`get_spectral_field(solve_adjoint(...); z=:initial, units=units)`.

Zygote is the only differentiation backend. The implementation is CPU-only;
`method` currently supports `RK4IP()` and `units` may be `:power` or
`:photon`.

`checkpointing=:auto` uses blocks of 10 RK4IP steps so that Zygote retains
only one block's reverse-mode tape at a time. Pass a positive integer to set
the number of steps per block, or `checkpointing=:none` to retain the complete
uncheckpointed tape. Checkpointing recomputes each block once during the
reverse pass, exchanging one additional forward propagation for substantially
lower peak memory.
"""
function get_autodiff_adjoint(
    observable::AbstractObservable,
    initial_field,
    parameters::MMGNLSEParameters,
    dz;
    method=RK4IP(),
    units::Symbol=:power,
    checkpointing=:auto)
    method isa RK4IP || throw(ArgumentError(
        "get_autodiff_adjoint currently supports method=RK4IP() only."))
    _mmgnlse_validate_field_units(units)
    field = _mmgnlse_validate_initial_field(initial_field, parameters)
    step = Float64(dz)
    isfinite(step) && step > 0 || throw(ArgumentError(
        "dz must be finite and positive."))

    forward = solve_mmgnlse(field, parameters, step; method, saveat=nothing)
    terminal = terminal_condition(observable, forward)
    cache = _mmgnlse_solver_cache(parameters)
    steps = _mmgnlse_ad_step_grid(parameters, step)
    checkpoint_steps = _mmgnlse_ad_checkpoint_steps(checkpointing)
    checkpoint_blocks = checkpoint_steps === nothing ? nothing :
        _mmgnlse_ad_checkpoint_blocks(
            steps, checkpoint_steps, parameters, cache)
    initial_w = ifft(field, 1)

    objective = raw -> begin
        final_raw = _mmgnlse_ad_propagate(
            raw, steps, parameters, cache, checkpoint_blocks)
        final_centered = fftshift(final_raw, 1)
        2real(sum(conj.(terminal) .* final_centered))
    end
    gradient_raw = 0.5 .* Zygote.gradient(objective, initial_w)[1]
    gradient_centered = fftshift(gradient_raw, 1)
    return _mmgnlse_convert_spectral_units(
        gradient_centered, parameters;
        from=:power, to=units, role=:adjoint)
end
