"""Fixed-step fourth-order Runge--Kutta interaction-picture method."""
struct RK4IP end

"""
Public forward solution for the MMGNLSE facade.

`fields` always has axes `(time, mode, polarization, z)`.  `z` contains the
exact coordinates of those saved planes, including shortened endpoint steps.
"""
Base.@kwdef struct MMGNLSESolution{T,A,P,I,M,C}
    z::Vector{T}
    fields::A
    parameters::P
    initial_field::I
    dz::T
    method::M
    cache::C = nothing
end

Base.@kwdef struct MMGNLSEAdjointSolution{T,A,P,C}
    z::Vector{T}
    fields::A
    parameters::P
    units::Symbol = :power
    cache::C = nothing
end

Base.@kwdef struct MMGNLSEStochasticSolution{T,A,P,N}
    z::Vector{T}
    fields::A
    parameters::P
    noise::N
end

function _mmgnlse_z_index(zgrid, z)
    z === :initial && return 1
    z === :final && return length(zgrid)
    z isa Integer && return Int(z)
    _, index = findmin(abs.(zgrid .- z))
    return index
end

function _mmgnlse_saved_fields(solution, z)
    z === :all && return solution.fields
    return solution.fields[:, :, :, _mmgnlse_z_index(solution.z, z)]
end

function _mmgnlse_validate_field_units(units::Symbol)
    units in (:power, :photon) || throw(ArgumentError(
        "units must be :power or :photon; got $(repr(units))."))
    return units
end

function _mmgnlse_frequency_scale(field, parameters, numerator::Symbol)
    weights = _mmgnlse_photon_weights(parameters)
    size(field, 1) == length(weights) || throw(DimensionMismatch(
        "The field frequency axis does not match the parameter time grid."))
    roots = sqrt.(weights)
    factors = if numerator === :sqrt_weight
        roots
    elseif numerator === :inverse_sqrt_weight
        map(root -> root > 0 ? inv(root) : zero(root), roots)
    else
        throw(ArgumentError("Unknown photon-normalization scale $numerator."))
    end
    return field .* reshape(factors, length(factors),
                            ntuple(_ -> 1, ndims(field) - 1)...)
end

function _mmgnlse_convert_spectral_units(field, parameters;
                                         from::Symbol,
                                         to::Symbol,
                                         role::Symbol)
    _mmgnlse_validate_field_units(from)
    _mmgnlse_validate_field_units(to)
    from === to && return field
    role in (:forward, :adjoint) || throw(ArgumentError(
        "role must be :forward or :adjoint; got $(repr(role))."))

    power_to_photon = from === :power && to === :photon
    numerator = if role === :forward
        power_to_photon ? :sqrt_weight : :inverse_sqrt_weight
    else
        power_to_photon ? :inverse_sqrt_weight : :sqrt_weight
    end
    return _mmgnlse_frequency_scale(field, parameters, numerator)
end

function _mmgnlse_spectral_from_stored(solution, z;
                                       stored_units::Symbol,
                                       units::Symbol,
                                       role::Symbol)
    field_t = _mmgnlse_saved_fields(solution, z)
    field_w = fftshift(ifft(field_t, 1), 1)
    return _mmgnlse_convert_spectral_units(
        field_w, solution.parameters; from=stored_units, to=units, role=role)
end

"""
    get_temporal_field(solution::MMGNLSESolution; z=:final, units=:power)
    get_temporal_field(solution::MMGNLSEAdjointSolution; z=:initial, units=:power)

Return a saved temporal field in power or photon units. `z=:all` returns every
saved plane. Photon-unit temporal fields are defined by converting the
centered spectral field and transforming it back to time.
"""
function get_temporal_field(solution::MMGNLSESolution;
                            z=:final, units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    units === :power && return _mmgnlse_saved_fields(solution, z)
    field_w = _mmgnlse_spectral_from_stored(
        solution, z; stored_units=:power, units, role=:forward)
    return fft(ifftshift(field_w, 1), 1)
end

function get_temporal_field(solution::MMGNLSEAdjointSolution;
                            z=:initial, units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    units === solution.units && return _mmgnlse_saved_fields(solution, z)
    field_w = _mmgnlse_spectral_from_stored(
        solution, z; stored_units=solution.units, units, role=:adjoint)
    return fft(ifftshift(field_w, 1), 1)
end

"""
    get_spectral_field(solution::MMGNLSESolution; z=:final, units=:power)
    get_spectral_field(solution::MMGNLSEAdjointSolution; z=:initial, units=:power)

Return the public centered, negative-to-positive frequency representation in
power or photon units. `z=:all` returns every saved plane. The package
convention is `Aomega = fftshift(ifft(At, 1), 1)`.
"""
function get_spectral_field(solution::MMGNLSESolution;
                            z=:final, units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    return _mmgnlse_spectral_from_stored(
        solution, z; stored_units=:power, units, role=:forward)
end

function get_spectral_field(solution::MMGNLSEAdjointSolution;
                            z=:initial, units::Symbol=:power)
    _mmgnlse_validate_field_units(units)
    return _mmgnlse_spectral_from_stored(
        solution, z; stored_units=solution.units, units, role=:adjoint)
end

_mmgnlse_has_dof(domain, name::Symbol) = name in domain.dofs.names

function _mmgnlse_public_shape(parameters)
    return parameter_field_shape(parameters)
end

function _mmgnlse_validate_initial_field(initial_field, parameters)
    expected = _mmgnlse_public_shape(parameters)
    actual = size(initial_field)
    actual == expected || throw(DimensionMismatch(
        "initial_field must have public shape (time, mode, polarization) = " *
        "$expected; received $actual."))
    all(isfinite, initial_field) || error("initial_field contains a non-finite value.")
    return ComplexF64.(Array(initial_field))
end

function _mmgnlse_validate_backend(backend)
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    return backend
end

# Public component tensors use mode-fast order: all modes for x followed by
# all modes for y. The facade otherwise keeps the axes separate.
function _mmgnlse_flatten_components(field::AbstractArray{T,3}) where {T}
    nt, nm, np = size(field)
    return reshape(field, nt, nm * np)
end

function _mmgnlse_unflatten_components(field::AbstractMatrix, nm::Integer,
                                       np::Integer)
    nt = size(field, 1)
    return reshape(field, nt, nm, np)
end

function _mmgnlse_save_targets(length_m::T, saveat) where {T}
    length_m > zero(T) || error("MMGNLSE length must be positive.")
    if saveat === nothing
        return T[zero(T), length_m], false
    elseif saveat === :steps
        return T[zero(T), length_m], true
    elseif saveat isa Real
        spacing = T(saveat)
        spacing > zero(T) || error("saveat spacing must be positive.")
        values = collect(T, zero(T):spacing:length_m)
        isempty(values) && push!(values, zero(T))
        values[end] == length_m || push!(values, length_m)
        return values, false
    else
        values = sort!(unique!(T.(collect(saveat))))
        all(z -> zero(T) <= z <= length_m, values) ||
            error("saveat positions must lie in [0, length].")
        isempty(values) || values[1] == zero(T) || pushfirst!(values, zero(T))
        isempty(values) && push!(values, zero(T))
        values[end] == length_m || push!(values, length_m)
        return values, false
    end
end

function _mmgnlse_step_grid(length_m::T, dz::T, save_targets,
                            save_every_step::Bool) where {T}
    dz > zero(T) || error("dz must be positive.")
    z = zero(T)
    target_index = 2
    steps = T[zero(T)]
    saved = T[zero(T)]
    atol = eps(T) * max(one(T), length_m) * T(16)
    while z < length_m
        next_target = target_index <= length(save_targets) ?
                      save_targets[target_index] : length_m
        znext = min(z + dz, next_target, length_m)
        abs(znext - next_target) <= atol && (znext = next_target)
        abs(znext - length_m) <= atol && (znext = length_m)
        znext > z || error("Could not construct a strictly increasing step grid.")
        z = znext
        push!(steps, z)
        if save_every_step || abs(z - next_target) <= atol || z == length_m
            push!(saved, z)
        end
        if target_index <= length(save_targets) && abs(z - next_target) <= atol
            target_index += 1
        end
    end
    saved[end] == length_m || push!(saved, length_m)
    return steps, unique!(saved)
end

function _mmgnlse_raw_omega(parameters)
    nt, _, _ = _mmgnlse_public_shape(parameters)
    nt == 1 && return zeros(Float64, 1)
    dt = parameters.domain.time_grid.dt
    return ifft_frequency_grid(nt, dt)
end

function _mmgnlse_beta_array(beta, nt::Int, nm::Int, np::Int,
                             omega_raw::AbstractVector)
    if beta isa TaylorBeta
        coefficients = _mmgnlse_expand_beta_coefficients(beta.coefficients, nm, np)
        order_count = size(coefficients, 1)
        reference0 = coefficients[1, 1, 1]
        reference1 = order_count >= 2 ? coefficients[2, 1, 1] : zero(reference0)
        out = zeros(ComplexF64, nt, nm, np)
        for order in 0:order_count-1
            coefficient = coefficients[order + 1, :, :]
            order == 0 && (coefficient = coefficient .- reference0)
            order == 1 && (coefficient = coefficient .- reference1)
            out .+= (1im / factorial(order)) .* reshape(omega_raw .^ order, nt, 1, 1) .*
                   reshape(coefficient, 1, nm, np)
        end
        return out
    elseif beta isa SampledBeta
        values = _mmgnlse_expand_sampled_beta(beta.values, nt, nm, np)
        centered_omega = fftshift(omega_raw)
        sampled = _mmgnlse_interpolate_sampled_beta(beta.omega, values,
                                                     centered_omega)
        beta0, beta1 = _mmgnlse_reference_line(beta.omega,
                                               @view(values[:, 1, 1]))
        values_raw = ifftshift(sampled, 1)
        return 1im .* (values_raw .- beta0 .-
                        reshape(omega_raw, nt, 1, 1) .* beta1)
    elseif beta isa Number
        iszero(beta) || error("A nonzero scalar beta is ambiguous; use TaylorBeta or SampledBeta.")
        return zeros(ComplexF64, nt, nm, np)
    else
        error("Unsupported beta representation $(typeof(beta)).")
    end
end

function _mmgnlse_linear_interpolate(x::AbstractVector, y::AbstractVector, xq)
    length(x) == length(y) || throw(DimensionMismatch("Interpolation axes do not match."))
    length(x) == 1 && return y[1]
    tolerance = 64eps(float(eltype(x))) * max(one(float(eltype(x))), maximum(abs, x))
    xq < first(x) - tolerance && throw(ArgumentError(
        "The sampled beta grid does not cover requested detuning $xq rad/ps."))
    xq > last(x) + tolerance && throw(ArgumentError(
        "The sampled beta grid does not cover requested detuning $xq rad/ps."))
    xq <= first(x) && return y[1]
    xq >= last(x) && return y[end]
    right = searchsortedfirst(x, xq)
    left = right - 1
    weight = (xq - x[left]) / (x[right] - x[left])
    return muladd(weight, y[right] - y[left], y[left])
end

function _mmgnlse_interpolate_sampled_beta(omega, values, target_omega)
    nt, nm, np = size(values)
    length(omega) == nt || throw(DimensionMismatch("Sampled beta frequency count mismatch."))
    out = similar(values, length(target_omega), nm, np)
    @inbounds for p in 1:np, m in 1:nm, i in eachindex(target_omega)
        out[i, m, p] = _mmgnlse_linear_interpolate(
            omega, @view(values[:, m, p]), target_omega[i])
    end
    return out
end

function _mmgnlse_reference_line(omega, values)
    beta0 = _mmgnlse_linear_interpolate(omega, values, zero(eltype(omega)))
    length(omega) == 1 && return beta0, zero(beta0)
    right = searchsortedfirst(omega, zero(eltype(omega)))
    if right <= 1
        left, right = 1, 2
    elseif right > length(omega)
        left, right = length(omega) - 1, length(omega)
    elseif omega[right] == 0 && 1 < right < length(omega)
        left, right = right - 1, right + 1
    else
        left = right - 1
    end
    beta1 = (values[right] - values[left]) / (omega[right] - omega[left])
    return beta0, beta1
end

function _mmgnlse_expand_beta_coefficients(values, nm::Int, np::Int)
    array = Array(values)
    ndims(array) <= 3 || error("Taylor beta must have axes (order, mode, polarization).")
    shaped = reshape(array, size(array)..., ntuple(_ -> 1, 3 - ndims(array))...)
    size(shaped, 2) in (1, nm) || error("Taylor beta mode axis must have length 1 or $nm.")
    size(shaped, 3) in (1, np) || error("Taylor beta polarization axis must have length 1 or $np.")
    return repeat(shaped, 1, nm ÷ size(shaped, 2), np ÷ size(shaped, 3))
end

function _mmgnlse_expand_sampled_beta(values, nt::Int, nm::Int, np::Int)
    array = Array(values)
    ndims(array) <= 3 || error("Sampled beta must have axes (frequency, mode, polarization).")
    shaped = reshape(array, size(array)..., ntuple(_ -> 1, 3 - ndims(array))...)
    size(shaped, 1) == nt || error("Sampled beta frequency axis must have length Nt=$nt.")
    size(shaped, 2) in (1, nm) || error("Sampled beta mode axis must have length 1 or $nm.")
    size(shaped, 3) in (1, np) || error("Sampled beta polarization axis must have length 1 or $np.")
    return repeat(shaped, 1, nm ÷ size(shaped, 2), np ÷ size(shaped, 3))
end

function _mmgnlse_profile_slice(profile::AbstractArray{<:Real,4}, parameters,
                                 z::Real; raw_frequency::Bool=true)
    nt, nm, np = _mmgnlse_public_shape(parameters)
    nf, nmp, npp, nz = size(profile)
    nf in (1, nt) || throw(DimensionMismatch(
        "Profile frequency axis must have length 1 or $nt; got $nf."))
    nmp in (1, nm) || throw(DimensionMismatch(
        "Profile mode axis must have length 1 or $nm; got $nmp."))
    npp in (1, np) || throw(DimensionMismatch(
        "Profile polarization axis must have length 1 or $np; got $npp."))
    zero(parameters.length) <= z <= parameters.length || throw(ArgumentError(
        "Longitudinal coordinate $z lies outside [0, $(parameters.length)]."))

    slice = if nz == 1
        Array(@view profile[:, :, :, 1])
    else
        scaled = clamp(z / parameters.length, zero(z), one(z)) * (nz - 1)
        left = min(floor(Int, scaled) + 1, nz)
        right = min(left + 1, nz)
        weight = scaled - (left - 1)
        (one(weight) - weight) .* Array(@view(profile[:, :, :, left])) .+
            weight .* Array(@view(profile[:, :, :, right]))
    end
    expanded = repeat(slice, nt ÷ nf, nm ÷ nmp, np ÷ npp)
    return raw_frequency && nf == nt ? ifftshift(expanded, 1) : expanded
end

function _mmgnlse_profile_integral(profile::AbstractArray{<:Real,4}, parameters,
                                    z0::Real, z1::Real;
                                    raw_frequency::Bool=true)
    z1 >= z0 || throw(ArgumentError("Profile integration requires z1 >= z0."))
    nt, nm, np = _mmgnlse_public_shape(parameters)
    z1 == z0 && return zeros(promote_type(eltype(profile), typeof(float(z0))),
                             nt, nm, np)
    nz = size(profile, 4)
    if nz == 1
        return (z1 - z0) .* _mmgnlse_profile_slice(
            profile, parameters, z0; raw_frequency)
    end

    knot_spacing = parameters.length / (nz - 1)
    result = zeros(promote_type(eltype(profile), typeof(float(z0))), nt, nm, np)
    position = z0
    tolerance = 32eps(typeof(float(parameters.length))) *
                max(one(parameters.length), parameters.length)
    while position < z1 - tolerance
        next_knot_index = floor(Int, position / knot_spacing + tolerance) + 1
        next_knot = min(next_knot_index * knot_spacing, parameters.length)
        endpoint = min(z1, next_knot > position + tolerance ? next_knot : position + knot_spacing)
        left = _mmgnlse_profile_slice(profile, parameters, position; raw_frequency)
        right = _mmgnlse_profile_slice(profile, parameters, endpoint; raw_frequency)
        result .+= (endpoint - position) / 2 .* (left .+ right)
        position = endpoint
    end
    return result
end

function _mmgnlse_raman_kernels(parameters)
    nt, _, _ = _mmgnlse_public_shape(parameters)
    parameters.raman isa NoRaman &&
        return (ha=ComplexF64[], hb=ComplexF64[], fraction=0.0)
    nt > 1 || throw(ArgumentError("Raman response requires a temporal grid."))
    dt = parameters.domain.time_grid.dt
    lag = dt .* collect(0:nt-1)
    response = raman_response(parameters.raman, lag)
    ha = ComplexF64.(ifft(response.ha, 1) .* (nt * dt))
    hb = ComplexF64.(ifft(response.hb, 1) .* (nt * dt))
    return (ha=ha, hb=hb, fraction=Float64(raman_fraction(parameters.raman)))
end

_mmgnlse_overlap_iszero(overlap::SpatialOverlap) = all(iszero, overlap.values)
_mmgnlse_overlap_iszero(overlap::ComponentOverlap) = all(iszero, overlap.values)
_mmgnlse_overlap_iszero(overlap::MMGNLSECPDecomposition) = cp_iszero(overlap)

function _mmgnlse_solver_cache(parameters)
    nt, nm, np = _mmgnlse_public_shape(parameters)
    omega = _mmgnlse_raw_omega(parameters)
    beta_operator = _mmgnlse_beta_array(parameters.beta, nt, nm, np, omega)
    c_m_per_ps = 2.99792458e-4
    damp = Float64.(damped_freq_window(nt))
    nonlinear_prefactor = reshape(
        ComplexF64.(1im .* parameters.n2 .* (omega .+ parameters.omega0) ./
                   c_m_per_ps .* damp), nt, 1, 1)
    nonlinear_active = any(!iszero, nonlinear_prefactor) &&
                       !_mmgnlse_overlap_iszero(parameters.S)
    return (; omega, beta_operator, nonlinear_prefactor, nonlinear_active,
            raman=_mmgnlse_raman_kernels(parameters))
end

function _mmgnlse_linear_operator(parameters, z, cache)
    alpha = _mmgnlse_profile_slice(parameters.alpha, parameters, z)
    gain = _mmgnlse_profile_slice(parameters.gain, parameters, z)
    return cache.beta_operator .+ (gain .- alpha) ./ 2
end

function _mmgnlse_linear_propagator(parameters, z0, z1, cache)
    alpha_integral = _mmgnlse_profile_integral(parameters.alpha, parameters, z0, z1)
    gain_integral = _mmgnlse_profile_integral(parameters.gain, parameters, z0, z1)
    exponent = cache.beta_operator .* (z1 - z0) .+
               (gain_integral .- alpha_integral) ./ 2
    return exp.(exponent)
end

function _mmgnlse_filter_raman(channels, kernel)
    isempty(kernel) && return zeros(ComplexF64, size(channels))
    all(iszero, kernel) && return zeros(ComplexF64, size(channels))
    return fft(reshape(kernel, :, ntuple(_ -> 1, ndims(channels) - 1)...) .*
               ifft(channels, 1), 1)
end

function _mmgnlse_dense_spatial_raman(field_t, S, ha, hb)
    nt, nm, np = size(field_t)
    T = promote_type(eltype(field_t), eltype(S), ComplexF64)
    out = zeros(T, nt, nm, np)
    if np == 1
        kernel = ha .+ hb
        q = Array{T}(undef, nt, nm, nm)
        @inbounds for i4 in 1:nm, i3 in 1:nm
            q[:, i3, i4] .= _mmgnlse_filter_raman(
                field_t[:, i3, 1] .* conj.(field_t[:, i4, 1]), kernel)
        end
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm, i1 in 1:nm
            out[:, i1, 1] .+= S[i1, i2, i3, i4] .* field_t[:, i2, 1] .* q[:, i3, i4]
        end
        return out
    end

    q_a = Array{T}(undef, nt, nm, nm)
    q_xx = Array{T}(undef, nt, nm, nm)
    q_yy = similar(q_xx)
    q_xy = similar(q_xx)
    @inbounds for i4 in 1:nm, i3 in 1:nm
        xx = field_t[:, i3, 1] .* conj.(field_t[:, i4, 1])
        yy = field_t[:, i3, 2] .* conj.(field_t[:, i4, 2])
        xy = field_t[:, i3, 1] .* conj.(field_t[:, i4, 2]) .+
             field_t[:, i3, 2] .* conj.(field_t[:, i4, 1])
        q_a[:, i3, i4] .= _mmgnlse_filter_raman(xx .+ yy, ha)
        q_xx[:, i3, i4] .= _mmgnlse_filter_raman(xx, hb)
        q_yy[:, i3, i4] .= _mmgnlse_filter_raman(yy, hb)
        q_xy[:, i3, i4] .= _mmgnlse_filter_raman(xy, hb)
    end
    half = T(1) / T(2)
    @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm, i1 in 1:nm
        s = S[i1, i2, i3, i4]
        ax = field_t[:, i2, 1]
        ay = field_t[:, i2, 2]
        out[:, i1, 1] .+= s .* (ax .* (q_a[:, i3, i4] .+ q_xx[:, i3, i4]) .+
                                     half .* ay .* q_xy[:, i3, i4])
        out[:, i1, 2] .+= s .* (ay .* (q_a[:, i3, i4] .+ q_yy[:, i3, i4]) .+
                                     half .* ax .* q_xy[:, i3, i4])
    end
    return out
end

function _mmgnlse_instantaneous_cubic(field_t, overlap::SpatialOverlap)
    return dense_spatial_isotropic_cubic(field_t, overlap.values)
end

function _mmgnlse_instantaneous_cubic(field_t, overlap::ComponentOverlap)
    flattened = flatten_component_tensor(overlap.values)
    return dense_component_cubic(field_t, flattened)
end

function _mmgnlse_instantaneous_cubic(field_t, overlap::MMGNLSECPDecomposition)
    overlap.metadata.layout == :spatial && return cp_spatial_isotropic_cubic(field_t, overlap)
    overlap.metadata.layout == :component && return cp_component_cubic(field_t, overlap)
    throw(ArgumentError("Unsupported CP overlap layout $(overlap.metadata.layout)."))
end

function _mmgnlse_delayed_cubic(field_t, overlap::SpatialOverlap, kernels)
    return _mmgnlse_dense_spatial_raman(field_t, overlap.values,
                                        kernels.ha, kernels.hb)
end

function _mmgnlse_delayed_cubic(field_t, overlap::MMGNLSECPDecomposition, kernels)
    if overlap.metadata.layout == :spatial
        filter_ha = channels -> _mmgnlse_filter_raman(channels, kernels.ha)
        filter_hb = channels -> _mmgnlse_filter_raman(channels, kernels.hb)
        isotropic = cp_spatial_raman_cubic(
            field_t, overlap, filter_ha; part=:isotropic)
        anisotropic = cp_spatial_raman_cubic(
            field_t, overlap, filter_hb; part=:anisotropic)
        return isotropic .+ anisotropic
    elseif overlap.metadata.layout == :component
        nt, nm, npol = size(field_t)
        (nm == overlap.metadata.nmodes &&
         npol == overlap.metadata.npolarizations) || throw(DimensionMismatch(
            "Field shape does not match component CP metadata."))
        cp_iszero(overlap) && return zeros(
            promote_type(eltype(field_t), eltype(overlap.λ),
                         map(eltype, overlap.U)...), size(field_t))
        flat_field = flatten_field_components(field_t)
        B2, B3, B4, WU1 = _cp_projected_fields(flat_field, overlap)
        filtered_pair = _mmgnlse_filter_raman(
            B3 .* B4, kernels.ha .+ kernels.hb)
        flat_out = (B2 .* filtered_pair) * transpose(WU1)
        return reshape(flat_out, nt, nm, npol)
    end
    throw(ArgumentError("Unsupported CP overlap layout $(overlap.metadata.layout)."))
end

function _mmgnlse_delayed_cubic(field_t, overlap::ComponentOverlap, kernels)
    nt, nm, npol = size(field_t)
    flat_field = flatten_field_components(field_t)
    S = flatten_component_tensor(overlap.values)
    nc = nm * npol
    size(S) == (nc, nc, nc, nc) || throw(DimensionMismatch(
        "The component tensor and flattened field dimensions do not match."))
    T = promote_type(eltype(field_t), eltype(S), ComplexF64)
    pair = Array{T}(undef, nt, nc, nc)
    kernel = kernels.ha .+ kernels.hb
    @inbounds for i4 in 1:nc, i3 in 1:nc
        pair[:, i3, i4] .= _mmgnlse_filter_raman(
            flat_field[:, i3] .* conj.(flat_field[:, i4]), kernel)
    end
    flat_out = zeros(T, nt, nc)
    @inbounds for i4 in 1:nc, i3 in 1:nc, i2 in 1:nc, i1 in 1:nc
        flat_out[:, i1] .+= S[i1, i2, i3, i4] .*
                            flat_field[:, i2] .* pair[:, i3, i4]
    end
    return reshape(flat_out, nt, nm, npol)
end

function _mmgnlse_nonlinear_spectral(field_w, parameters, cache)
    !cache.nonlinear_active && return zeros(ComplexF64, size(field_w))
    field_t = fft(field_w, 1)
    instantaneous = _mmgnlse_instantaneous_cubic(field_t, parameters.S)
    fraction = cache.raman.fraction
    cubic = if iszero(fraction)
        instantaneous
    else
        delayed = _mmgnlse_delayed_cubic(field_t, parameters.S, cache.raman)
        (1 - fraction) .* instantaneous .+ fraction .* delayed
    end
    return cache.nonlinear_prefactor .* ifft(cubic, 1)
end

function _mmgnlse_rhs_spectral(field_w, parameters, z,
                               cache=_mmgnlse_solver_cache(parameters))
    validate_public_field(field_w, parameters.domain, parameter_mode_count(parameters);
                          name=:spectral_field)
    linear = _mmgnlse_linear_operator(parameters, z, cache) .* field_w
    return linear .+ _mmgnlse_nonlinear_spectral(field_w, parameters, cache)
end

"""Internal full moving-frame ODE RHS in public time-domain axes."""
function _mmgnlse_rhs(field_t, parameters, z)
    validate_public_field(field_t, parameters.domain, parameter_mode_count(parameters))
    cache = _mmgnlse_solver_cache(parameters)
    return fft(_mmgnlse_rhs_spectral(ifft(field_t, 1), parameters, z, cache), 1)
end

function _mmgnlse_rk4ip_step(field_w, parameters, z0, z1, cache)
    !cache.nonlinear_active && return (
        _mmgnlse_linear_propagator(parameters, z0, z1, cache) .* field_w)
    step = z1 - z0
    midpoint = (z0 + z1) / 2
    to_midpoint = _mmgnlse_linear_propagator(parameters, z0, midpoint, cache)
    from_midpoint = _mmgnlse_linear_propagator(parameters, midpoint, z1, cache)
    midpoint_base = to_midpoint .* field_w
    k1 = to_midpoint .* _mmgnlse_nonlinear_spectral(field_w, parameters, cache)
    k2 = _mmgnlse_nonlinear_spectral(midpoint_base .+ step / 2 .* k1,
                                     parameters, cache)
    k3 = _mmgnlse_nonlinear_spectral(midpoint_base .+ step / 2 .* k2,
                                     parameters, cache)
    endpoint_stage = from_midpoint .* (midpoint_base .+ step .* k3)
    k4 = _mmgnlse_nonlinear_spectral(endpoint_stage, parameters, cache)
    return from_midpoint .* (midpoint_base .+
           step / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3)) .+ step / 6 .* k4
end

function _mmgnlse_solve_rk4ip(initial_field, parameters, dz, saveat)
    cache = _mmgnlse_solver_cache(parameters)
    targets, save_every_step = _mmgnlse_save_targets(parameters.length, saveat)
    steps, saved = _mmgnlse_step_grid(parameters.length, dz, targets, save_every_step)
    nt, nm, np = size(initial_field)
    fields = Array{ComplexF64}(undef, nt, nm, np, length(saved))
    fields[:, :, :, 1] .= initial_field
    field_w = ifft(initial_field, 1)
    save_index = 2
    tolerance = 32eps(Float64) * max(1.0, Float64(parameters.length))
    for index in 1:length(steps)-1
        z0, z1 = steps[index], steps[index + 1]
        field_w = _mmgnlse_rk4ip_step(field_w, parameters, z0, z1, cache)
        if save_index <= length(saved) && abs(z1 - saved[save_index]) <= tolerance
            fields[:, :, :, save_index] .= fft(field_w, 1)
            save_index += 1
        end
    end
    save_index == length(saved) + 1 || error("Not every requested plane was saved.")
    return MMGNLSESolution(; z=Float64.(saved), fields,
        parameters, initial_field=copy(initial_field), dz=Float64(dz),
        method=RK4IP(), cache=(; integration_z=Float64.(steps), core=cache))
end

function _mmgnlse_solve_tsit5(initial_field, parameters, dz, saveat, method)
    cache = _mmgnlse_solver_cache(parameters)
    targets, save_every_step = _mmgnlse_save_targets(parameters.length, saveat)
    steps, saved = _mmgnlse_step_grid(parameters.length, dz, targets, save_every_step)
    field_w0 = ifft(initial_field, 1)
    function rhs!(derivative, field_w, _, z)
        derivative .= _mmgnlse_rhs_spectral(field_w, parameters, z, cache)
        return nothing
    end
    problem = ODEProblem(rhs!, field_w0, (0.0, Float64(parameters.length)))
    ode_solution = DifferentialEquations.solve(problem, method;
        adaptive=false, dt=Float64(dz), tstops=Float64.(saved),
        saveat=Float64.(saved), save_start=true, save_end=true,
        save_everystep=false)
    length(ode_solution.u) == length(saved) || error(
        "Tsit5 returned $(length(ode_solution.u)) planes for $(length(saved)) requested saves.")
    nt, nm, np = size(initial_field)
    fields = Array{ComplexF64}(undef, nt, nm, np, length(saved))
    @inbounds for index in eachindex(saved)
        fields[:, :, :, index] .= fft(ode_solution.u[index], 1)
    end
    return MMGNLSESolution(; z=Float64.(saved), fields,
        parameters, initial_field=copy(initial_field), dz=Float64(dz), method,
        cache=(; integration_z=Float64.(steps), core=cache, ode_solution))
end

"""
    solve_mmgnlse(initial_field, parameters, dz;
                  method=RK4IP(), saveat=nothing, backend=:cpu,
                  device=nothing, synchronize=true)

Solve the MMGNLSE on the selected execution backend. Public fields retain axes
`time × mode × polarization`; saved output adds a final `z` axis. RK4IP is the
default fixed-step method. `Tsit5()` is accepted only in fixed-step mode.
Every requested save coordinate and the exact fiber endpoint are landed on by
shortening a step when necessary. `backend=:cuda` requires CUDA.jl and returns
the same host-array solution representation as `backend=:cpu`.
"""
function solve_mmgnlse(initial_field, parameters::MMGNLSEParameters, dz;
                       method=RK4IP(), saveat=nothing, backend=:cpu,
                       device=nothing, synchronize::Bool=true)
    field = _mmgnlse_validate_initial_field(initial_field, parameters)
    step = Float64(dz)
    isfinite(step) && step > 0 || throw(ArgumentError("dz must be finite and positive."))
    selected_backend = _mmgnlse_validate_backend(backend)
    if selected_backend === :cuda
        return _mmgnlse_solve_cuda(
            field, parameters, step;
            method, saveat, device, synchronize)
    end
    if method isa RK4IP
        return _mmgnlse_solve_rk4ip(field, parameters, step, saveat)
    elseif nameof(typeof(method)) == :Tsit5
        return _mmgnlse_solve_tsit5(field, parameters, step, saveat, method)
    end
    throw(ArgumentError("Forward method must be RK4IP() or fixed-step Tsit5()."))
end
