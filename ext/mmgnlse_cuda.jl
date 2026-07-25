# CUDA execution core for the public MMGNLSE interface. This file is included
# inside `PulsePropagationCUDAExt` and intentionally does not add CUDA as a hard
# dependency of PulsePropagation.

struct CUDAMMGNLSEProfile{A,T}
    values::A
    length::T
end

abstract type AbstractCUDAMMGNLSEOverlap end

struct CUDAMMGNLSEDenseSpatial{A} <: AbstractCUDAMMGNLSEOverlap
    values::A
end

struct CUDAMMGNLSEDenseComponent{A} <: AbstractCUDAMMGNLSEOverlap
    values::A
    nmodes::Int
    npolarizations::Int
end

struct CUDAMMGNLSECP{U,A,C} <: AbstractCUDAMMGNLSEOverlap
    U::U
    wu1t::A
    cwu1::C
    layout::Symbol
    nmodes::Int
    npolarizations::Int
    rank::Int
end

struct CUDAMMGNLSEForwardCache{B,P,N,R,O}
    beta_operator::B
    alpha::P
    gain::P
    nonlinear_prefactor::N
    raman::R
    overlap::O
    nonlinear_active::Bool
end

function _cuda_mmgnlse_profile(profile, parameters)
    nt, nm, np = PulsePropagation.parameter_field_shape(parameters)
    nz = size(profile, 4)
    expanded = Array{Float64}(undef, nt, nm, np, nz)
    if nz == 1
        expanded[:, :, :, 1] .= PulsePropagation._mmgnlse_profile_slice(
            profile, parameters, 0.0)
    else
        for index in 1:nz
            z = parameters.length * (index - 1) / (nz - 1)
            expanded[:, :, :, index] .=
                PulsePropagation._mmgnlse_profile_slice(profile, parameters, z)
        end
    end
    return CUDAMMGNLSEProfile(CUDA.CuArray(expanded),
                              Float64(parameters.length))
end

function _cuda_mmgnlse_profile_slice(profile::CUDAMMGNLSEProfile, z::Real)
    nz = size(profile.values, 4)
    nz == 1 && return @view(profile.values[:, :, :, 1])
    scaled = clamp(Float64(z) / profile.length, 0.0, 1.0) * (nz - 1)
    left = min(floor(Int, scaled) + 1, nz)
    right = min(left + 1, nz)
    weight = scaled - (left - 1)
    iszero(weight) && return @view(profile.values[:, :, :, left])
    return (1 - weight) .* @view(profile.values[:, :, :, left]) .+
           weight .* @view(profile.values[:, :, :, right])
end

function _cuda_mmgnlse_profile_integral(profile::CUDAMMGNLSEProfile,
                                        z0::Real, z1::Real)
    z1 >= z0 || throw(ArgumentError(
        "Profile integration requires z1 >= z0."))
    nt, nm, np, nz = size(profile.values)
    if z1 == z0
        return CUDA.zeros(Float64, nt, nm, np)
    elseif nz == 1
        return (z1 - z0) .* @view(profile.values[:, :, :, 1])
    end

    knot_spacing = profile.length / (nz - 1)
    result = CUDA.zeros(Float64, nt, nm, np)
    position = Float64(z0)
    endpoint_z = Float64(z1)
    tolerance = 32eps(Float64) * max(1.0, profile.length)
    while position < endpoint_z - tolerance
        next_knot_index =
            floor(Int, position / knot_spacing + tolerance) + 1
        next_knot = min(next_knot_index * knot_spacing, profile.length)
        endpoint = min(endpoint_z,
            next_knot > position + tolerance ?
            next_knot : position + knot_spacing)
        left = _cuda_mmgnlse_profile_slice(profile, position)
        right = _cuda_mmgnlse_profile_slice(profile, endpoint)
        result .+= (endpoint - position) / 2 .* (left .+ right)
        position = endpoint
    end
    return result
end

function _cuda_mmgnlse_overlap(overlap::PulsePropagation.SpatialOverlap)
    return CUDAMMGNLSEDenseSpatial(
        CUDA.CuArray(ComplexF64.(overlap.values)))
end

function _cuda_mmgnlse_overlap(overlap::PulsePropagation.ComponentOverlap)
    flattened = PulsePropagation.flatten_component_tensor(overlap.values)
    return CUDAMMGNLSEDenseComponent(
        CUDA.CuArray(ComplexF64.(flattened)),
        size(overlap.values, 1), size(overlap.values, 2))
end

function _cuda_mmgnlse_overlap(
    overlap::PulsePropagation.MMGNLSECPDecomposition,
)
    factors_cpu = ntuple(index -> ComplexF64.(overlap.U[index]), 4)
    weights = ComplexF64.(overlap.λ)
    weighted_u1 = factors_cpu[1] .* reshape(weights, 1, :)
    factors = ntuple(index -> CUDA.CuArray(factors_cpu[index]), 4)
    wu1t = CUDA.CuArray(Matrix(transpose(weighted_u1)))
    cwu1 = CUDA.CuArray(conj.(weighted_u1))
    return CUDAMMGNLSECP(
        factors, wu1t, cwu1, overlap.metadata.layout, overlap.metadata.nmodes,
        overlap.metadata.npolarizations, length(weights))
end

function _cuda_mmgnlse_forward_cache(parameters)
    cpu_cache = PulsePropagation._mmgnlse_solver_cache(parameters)
    raman = (
        ha=CUDA.CuArray(ComplexF64.(cpu_cache.raman.ha)),
        hb=CUDA.CuArray(ComplexF64.(cpu_cache.raman.hb)),
        fraction=Float64(cpu_cache.raman.fraction),
        ha_active=any(value -> !iszero(value), cpu_cache.raman.ha),
        hb_active=any(value -> !iszero(value), cpu_cache.raman.hb),
    )
    return CUDAMMGNLSEForwardCache(
        CUDA.CuArray(ComplexF64.(cpu_cache.beta_operator)),
        _cuda_mmgnlse_profile(parameters.alpha, parameters),
        _cuda_mmgnlse_profile(parameters.gain, parameters),
        CUDA.CuArray(ComplexF64.(cpu_cache.nonlinear_prefactor)),
        raman,
        _cuda_mmgnlse_overlap(parameters.S),
        cpu_cache.nonlinear_active,
    )
end

function _cuda_mmgnlse_linear_operator(cache::CUDAMMGNLSEForwardCache,
                                       z::Real)
    alpha = _cuda_mmgnlse_profile_slice(cache.alpha, z)
    gain = _cuda_mmgnlse_profile_slice(cache.gain, z)
    return cache.beta_operator .+ (gain .- alpha) ./ 2
end

function _cuda_mmgnlse_linear_propagator(cache::CUDAMMGNLSEForwardCache,
                                         z0::Real, z1::Real)
    alpha = _cuda_mmgnlse_profile_integral(cache.alpha, z0, z1)
    gain = _cuda_mmgnlse_profile_integral(cache.gain, z0, z1)
    return exp.(cache.beta_operator .* (z1 - z0) .+
                (gain .- alpha) ./ 2)
end

function _cuda_mmgnlse_dense_spatial_kernel!(
    out, field, S, nt::Int, nm::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
            s = S[i1, i2, i3, i4]
            if polarization == 1
                accumulator += s * (
                    field[t, i2, 1] * field[t, i3, 1] *
                    conj(field[t, i4, 1]))
                if np == 2
                    accumulator += s * (
                        (2 / 3) * field[t, i2, 1] * field[t, i3, 2] *
                        conj(field[t, i4, 2]) +
                        (1 / 3) * field[t, i2, 2] * field[t, i3, 2] *
                        conj(field[t, i4, 1]))
                end
            else
                accumulator += s * (
                    field[t, i2, 2] * field[t, i3, 2] *
                    conj(field[t, i4, 2]) +
                    (2 / 3) * field[t, i2, 2] * field[t, i3, 1] *
                    conj(field[t, i4, 1]) +
                    (1 / 3) * field[t, i2, 1] * field[t, i3, 1] *
                    conj(field[t, i4, 2]))
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_kernel!(
    out, field, S, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        i1 = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nc, i3 in 1:nc, i2 in 1:nc
            accumulator += S[i1, i2, i3, i4] *
                           field[t, i2] * field[t, i3] *
                           conj(field[t, i4])
        end
        @inbounds out[t, i1] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_delayed_kernel!(
    out, field, filtered_pair, S, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        i1 = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nc, i3 in 1:nc, i2 in 1:nc
            accumulator += S[i1, i2, i3, i4] *
                           field[t, i2] *
                           filtered_pair[t, i3, i4]
        end
        @inbounds out[t, i1] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_scalar_delayed_kernel!(
    out, field, filtered_pair, S, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        i1 = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
            accumulator += S[i1, i2, i3, i4] *
                           field[t, i2, 1] *
                           filtered_pair[t, i3, i4]
        end
        @inbounds out[t, i1, 1] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_polarized_delayed_kernel!(
    out, field, qa, qxx, qyy, qxy, S, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
            s = S[i1, i2, i3, i4]
            if polarization == 1
                accumulator += s * (
                    field[t, i2, 1] *
                    (qa[t, i3, i4] + qxx[t, i3, i4]) +
                    0.5 * field[t, i2, 2] * qxy[t, i3, i4])
            else
                accumulator += s * (
                    field[t, i2, 2] *
                    (qa[t, i3, i4] + qyy[t, i3, i4]) +
                    0.5 * field[t, i2, 1] * qxy[t, i3, i4])
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_filter_raman(channels, kernel)
    dimensions = (length(kernel),
                  ntuple(_ -> 1, ndims(channels) - 1)...)
    return fft(reshape(kernel, dimensions) .* ifft(channels, 1), 1)
end

function _cuda_mmgnlse_instantaneous(
    field, overlap::CUDAMMGNLSEDenseSpatial,
)
    nt, nm, np = size(field)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    _launch_1d(_cuda_mmgnlse_dense_spatial_kernel!, length(out),
               out, field, overlap.values, nt, nm, np)
    return out
end

function _cuda_mmgnlse_instantaneous(
    field, overlap::CUDAMMGNLSEDenseComponent,
)
    nt, nm, np = size(field)
    nm == overlap.nmodes && np == overlap.npolarizations ||
        throw(DimensionMismatch(
            "Field shape does not match the component overlap."))
    nc = nm * np
    flattened = reshape(field, nt, nc)
    out = CUDA.zeros(ComplexF64, nt, nc)
    _launch_1d(_cuda_mmgnlse_dense_component_kernel!, length(out),
               out, flattened, overlap.values, nt, nc)
    return reshape(out, nt, nm, np)
end

function _cuda_mmgnlse_cp_projections(field, overlap::CUDAMMGNLSECP)
    U1, U2, U3, U4 = overlap.U
    return field * U2, field * U3, conj.(field) * U4
end

function _cuda_mmgnlse_cp_spatial_instantaneous(
    field, overlap::CUDAMMGNLSECP,
)
    nt, nm, np = size(field)
    nm == overlap.nmodes || throw(DimensionMismatch(
        "Field mode count does not match the spatial CP decomposition."))
    ax = copy(@view field[:, :, 1])
    b2x, b3x, b4x = _cuda_mmgnlse_cp_projections(ax, overlap)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        @views out[:, :, 1] .=
            (b2x .* b3x .* b4x) * overlap.wu1t
        return out
    end
    ay = copy(@view field[:, :, 2])
    b2y, b3y, b4y = _cuda_mmgnlse_cp_projections(ay, overlap)
    @views out[:, :, 1] .= (
        b2x .* b3x .* b4x .+
        (2 / 3) .* b2x .* b3y .* b4y .+
        (1 / 3) .* b2y .* b3y .* b4x) * overlap.wu1t
    @views out[:, :, 2] .= (
        b2y .* b3y .* b4y .+
        (2 / 3) .* b2y .* b3x .* b4x .+
        (1 / 3) .* b2x .* b3x .* b4y) * overlap.wu1t
    return out
end

function _cuda_mmgnlse_instantaneous(field, overlap::CUDAMMGNLSECP)
    if overlap.layout == :spatial
        return _cuda_mmgnlse_cp_spatial_instantaneous(field, overlap)
    elseif overlap.layout == :component
        nt, nm, np = size(field)
        nm == overlap.nmodes && np == overlap.npolarizations ||
            throw(DimensionMismatch(
                "Field shape does not match the component CP decomposition."))
        flattened = reshape(field, nt, nm * np)
        b2, b3, b4 = _cuda_mmgnlse_cp_projections(flattened, overlap)
        return reshape(
            (b2 .* b3 .* b4) * overlap.wu1t, nt, nm, np)
    end
    throw(ArgumentError("Unsupported CP layout $(overlap.layout)."))
end

function _cuda_mmgnlse_delayed(
    field, overlap::CUDAMMGNLSEDenseSpatial, raman,
)
    nt, nm, np = size(field)
    ax = copy(@view field[:, :, 1])
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        pair = reshape(ax, nt, nm, 1) .*
               conj.(reshape(ax, nt, 1, nm))
        filtered = _cuda_mmgnlse_filter_raman(
            pair, raman.ha .+ raman.hb)
        _launch_1d(
            _cuda_mmgnlse_dense_spatial_scalar_delayed_kernel!,
            nt * nm, out, field, filtered, overlap.values, nt, nm)
        return out
    end

    ay = copy(@view field[:, :, 2])
    xx = reshape(ax, nt, nm, 1) .*
         conj.(reshape(ax, nt, 1, nm))
    yy = reshape(ay, nt, nm, 1) .*
         conj.(reshape(ay, nt, 1, nm))
    xy = reshape(ax, nt, nm, 1) .*
         conj.(reshape(ay, nt, 1, nm)) .+
         reshape(ay, nt, nm, 1) .*
         conj.(reshape(ax, nt, 1, nm))
    qa = _cuda_mmgnlse_filter_raman(xx .+ yy, raman.ha)
    qxx = _cuda_mmgnlse_filter_raman(xx, raman.hb)
    qyy = _cuda_mmgnlse_filter_raman(yy, raman.hb)
    qxy = _cuda_mmgnlse_filter_raman(xy, raman.hb)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_polarized_delayed_kernel!,
        length(out), out, field, qa, qxx, qyy, qxy,
        overlap.values, nt, nm)
    return out
end

function _cuda_mmgnlse_delayed(
    field, overlap::CUDAMMGNLSEDenseComponent, raman,
)
    nt, nm, np = size(field)
    nc = nm * np
    flattened = reshape(field, nt, nc)
    pair = reshape(flattened, nt, nc, 1) .*
           conj.(reshape(flattened, nt, 1, nc))
    filtered = _cuda_mmgnlse_filter_raman(
        pair, raman.ha .+ raman.hb)
    out = CUDA.zeros(ComplexF64, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_component_delayed_kernel!, length(out),
        out, flattened, filtered, overlap.values, nt, nc)
    return reshape(out, nt, nm, np)
end

function _cuda_mmgnlse_cp_spatial_delayed(
    field, overlap::CUDAMMGNLSECP, raman,
)
    nt, nm, np = size(field)
    ax = copy(@view field[:, :, 1])
    b2x, b3x, b4x = _cuda_mmgnlse_cp_projections(ax, overlap)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        filtered = _cuda_mmgnlse_filter_raman(
            b3x .* b4x, raman.ha .+ raman.hb)
        @views out[:, :, 1] .=
            (b2x .* filtered) * overlap.wu1t
        return out
    end

    ay = copy(@view field[:, :, 2])
    b2y, b3y, b4y = _cuda_mmgnlse_cp_projections(ay, overlap)
    qa = _cuda_mmgnlse_filter_raman(
        b3x .* b4x .+ b3y .* b4y, raman.ha)
    qxx = _cuda_mmgnlse_filter_raman(b3x .* b4x, raman.hb)
    qyy = _cuda_mmgnlse_filter_raman(b3y .* b4y, raman.hb)
    qxy = _cuda_mmgnlse_filter_raman(
        b3x .* b4y .+ b3y .* b4x, raman.hb)
    @views out[:, :, 1] .= (
        b2x .* (qa .+ qxx) .+ 0.5 .* b2y .* qxy) *
        overlap.wu1t
    @views out[:, :, 2] .= (
        b2y .* (qa .+ qyy) .+ 0.5 .* b2x .* qxy) *
        overlap.wu1t
    return out
end

function _cuda_mmgnlse_delayed(field, overlap::CUDAMMGNLSECP, raman)
    if overlap.layout == :spatial
        return _cuda_mmgnlse_cp_spatial_delayed(field, overlap, raman)
    elseif overlap.layout == :component
        nt, nm, np = size(field)
        flattened = reshape(field, nt, nm * np)
        b2, b3, b4 = _cuda_mmgnlse_cp_projections(flattened, overlap)
        filtered = _cuda_mmgnlse_filter_raman(
            b3 .* b4, raman.ha .+ raman.hb)
        return reshape(
            (b2 .* filtered) * overlap.wu1t, nt, nm, np)
    end
    throw(ArgumentError("Unsupported CP layout $(overlap.layout)."))
end

function _cuda_mmgnlse_nonlinear_spectral(
    field_w, cache::CUDAMMGNLSEForwardCache,
)
    if !cache.nonlinear_active
        return CUDA.zeros(ComplexF64, size(field_w)...)
    end
    field_t = fft(field_w, 1)
    instantaneous = _cuda_mmgnlse_instantaneous(
        field_t, cache.overlap)
    fraction = cache.raman.fraction
    cubic = if iszero(fraction)
        instantaneous
    else
        delayed = _cuda_mmgnlse_delayed(
            field_t, cache.overlap, cache.raman)
        (1 - fraction) .* instantaneous .+ fraction .* delayed
    end
    return cache.nonlinear_prefactor .* ifft(cubic, 1)
end

function _cuda_mmgnlse_rhs_spectral(
    field_w, cache::CUDAMMGNLSEForwardCache, z::Real,
)
    linear = _cuda_mmgnlse_linear_operator(cache, z) .* field_w
    return linear .+ _cuda_mmgnlse_nonlinear_spectral(field_w, cache)
end

# Fixed-step GPU solves do not consume an error norm. Supplying this inert
# callback prevents SciML from compiling its generic CuArray reduction path.
_cuda_mmgnlse_fixed_internalnorm(_, time) = zero(time)

function _cuda_mmgnlse_rk4ip_step(
    field_w, cache::CUDAMMGNLSEForwardCache, z0::Real, z1::Real,
)
    if !cache.nonlinear_active
        return _cuda_mmgnlse_linear_propagator(cache, z0, z1) .* field_w
    end
    step = z1 - z0
    midpoint = (z0 + z1) / 2
    to_midpoint = _cuda_mmgnlse_linear_propagator(
        cache, z0, midpoint)
    from_midpoint = _cuda_mmgnlse_linear_propagator(
        cache, midpoint, z1)
    midpoint_base = to_midpoint .* field_w
    k1 = to_midpoint .* _cuda_mmgnlse_nonlinear_spectral(
        field_w, cache)
    k2 = _cuda_mmgnlse_nonlinear_spectral(
        midpoint_base .+ step / 2 .* k1, cache)
    k3 = _cuda_mmgnlse_nonlinear_spectral(
        midpoint_base .+ step / 2 .* k2, cache)
    endpoint_stage = from_midpoint .* (midpoint_base .+ step .* k3)
    k4 = _cuda_mmgnlse_nonlinear_spectral(endpoint_stage, cache)
    return from_midpoint .* (
        midpoint_base .+ step / 6 .* (k1 .+ 2 .* k2 .+ 2 .* k3)) .+
        step / 6 .* k4
end

function _cuda_mmgnlse_validate_initial(initial_field, parameters)
    expected = PulsePropagation.parameter_field_shape(parameters)
    size(initial_field) == expected || throw(DimensionMismatch(
        "initial_field must have public shape " *
        "(time, mode, polarization) = $expected; received " *
        "$(size(initial_field))."))
    all(isfinite, initial_field) || throw(ArgumentError(
        "initial_field contains a non-finite value."))
    return ComplexF64.(initial_field)
end

function _cuda_mmgnlse_solve_rk4ip(
    initial_field, parameters, dz::Float64, saveat,
)
    gpu_cache = _cuda_mmgnlse_forward_cache(parameters)
    targets, every_step = PulsePropagation._mmgnlse_save_targets(
        Float64(parameters.length), saveat)
    steps, saved = PulsePropagation._mmgnlse_step_grid(
        Float64(parameters.length), dz, targets, every_step)
    nt, nm, np = size(initial_field)
    fields = CUDA.zeros(ComplexF64, nt, nm, np, length(saved))
    @views fields[:, :, :, 1] .= initial_field
    field_w = ifft(initial_field, 1)
    save_index = 2
    tolerance = 32eps(Float64) * max(1.0, Float64(parameters.length))
    for index in 1:length(steps)-1
        z0, z1 = steps[index], steps[index + 1]
        field_w = _cuda_mmgnlse_rk4ip_step(
            field_w, gpu_cache, z0, z1)
        if save_index <= length(saved) &&
           abs(z1 - saved[save_index]) <= tolerance
            @views fields[:, :, :, save_index] .= fft(field_w, 1)
            save_index += 1
        end
    end
    save_index == length(saved) + 1 ||
        error("Not every requested plane was saved.")
    cpu_core = PulsePropagation._mmgnlse_solver_cache(parameters)
    return PulsePropagation.MMGNLSESolution(
        z=Float64.(saved), fields=fields, parameters=parameters,
        initial_field=copy(initial_field), dz=dz,
        method=PulsePropagation.RK4IP(),
        cache=(integration_z=Float64.(steps), core=cpu_core,
               cuda=gpu_cache))
end

function _cuda_mmgnlse_solve_tsit5(
    initial_field, parameters, dz::Float64, saveat, method,
)
    gpu_cache = _cuda_mmgnlse_forward_cache(parameters)
    targets, every_step = PulsePropagation._mmgnlse_save_targets(
        Float64(parameters.length), saveat)
    steps, saved = PulsePropagation._mmgnlse_step_grid(
        Float64(parameters.length), dz, targets, every_step)
    field_w0 = ifft(initial_field, 1)
    function rhs!(derivative, field_w, _, z)
        derivative .= _cuda_mmgnlse_rhs_spectral(
            field_w, gpu_cache, z)
        return nothing
    end
    problem = PulsePropagation.ODEProblem(
        rhs!, field_w0, (0.0, Float64(parameters.length)))
    ode_solution = PulsePropagation.DifferentialEquations.solve(
        problem, method; adaptive=false, dt=dz,
        tstops=Float64.(saved), saveat=Float64.(saved),
        save_start=true, save_end=true, save_everystep=false,
        internalnorm=_cuda_mmgnlse_fixed_internalnorm)
    length(ode_solution.u) == length(saved) || error(
        "Tsit5 returned $(length(ode_solution.u)) planes for " *
        "$(length(saved)) requested saves.")
    nt, nm, np = size(initial_field)
    fields = CUDA.zeros(ComplexF64, nt, nm, np, length(saved))
    for index in eachindex(saved)
        @views fields[:, :, :, index] .=
            fft(ode_solution.u[index], 1)
    end
    cpu_core = PulsePropagation._mmgnlse_solver_cache(parameters)
    return PulsePropagation.MMGNLSESolution(
        z=Float64.(saved), fields=fields, parameters=parameters,
        initial_field=copy(initial_field), dz=dz, method=method,
        cache=(integration_z=Float64.(steps), core=cpu_core,
               cuda=gpu_cache, ode_solution=ode_solution))
end

function _cuda_mmgnlse_solve_device(
    initial_field::CUDA.CuArray{T,3},
    parameters::PulsePropagation.MMGNLSEParameters,
    dz::Float64;
    method=PulsePropagation.RK4IP(),
    saveat=nothing,
) where {T<:Number}
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    field = _cuda_mmgnlse_validate_initial(initial_field, parameters)
    if method isa PulsePropagation.RK4IP
        return _cuda_mmgnlse_solve_rk4ip(
            field, parameters, dz, saveat)
    elseif nameof(typeof(method)) == :Tsit5
        return _cuda_mmgnlse_solve_tsit5(
            field, parameters, dz, saveat, method)
    end
    throw(ArgumentError(
        "Forward method must be RK4IP() or fixed-step Tsit5()."))
end

"""
    solve_mmgnlse(initial_field, parameters, dz; backend=:cuda, ...)

CUDA execution of the public MMGNLSE forward solve. Computation stays on the
selected CUDA device, while the returned solution uses the same host-array
representation, public axes, and exact-save semantics as the CPU backend.
"""
function PulsePropagation._mmgnlse_solve_cuda(
    initial_field::AbstractArray{<:Number,3},
    parameters::PulsePropagation.MMGNLSEParameters,
    dz::Real;
    method=PulsePropagation.RK4IP(),
    saveat=nothing,
    device=nothing,
    synchronize::Bool=true,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    step = Float64(dz)
    isfinite(step) && step > 0 || throw(ArgumentError(
        "dz must be finite and positive."))
    device_solution = _cuda_mmgnlse_solve_device(
        CUDA.CuArray(ComplexF64.(initial_field)),
        parameters, step;
        method, saveat)
    synchronize && CUDA.synchronize()
    device_cache = device_solution.cache
    host_cache = (
        backend=:cuda,
        integration_z=device_cache.integration_z,
        core=device_cache.core,
        device=string(CUDA.device()),
    )
    return PulsePropagation.MMGNLSESolution(
        z=device_solution.z,
        fields=Array(device_solution.fields),
        parameters=parameters,
        initial_field=Array(device_solution.initial_field),
        dz=device_solution.dz,
        method=device_solution.method,
        cache=host_cache,
    )
end

include("mmgnlse_cuda_adjoint.jl")
