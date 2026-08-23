# CUDA execution core for the public MMGNLSE interface. This file is included
# inside `PulsePropagationCUDAExt` and intentionally does not add CUDA as a hard
# dependency of PulsePropagation.

struct CUDAMMGNLSEProfile{A,T}
    values::A
    length::T
end

abstract type AbstractCUDAMMGNLSEOverlap end

const _CUDA_MMGNLSE_CP_STACKED_PROJECTION_MAX_RANK = 224
const _CUDA_MMGNLSE_CP_STACKED_BACKPROJECTION_MIN_RANK = 40
const _CUDA_MMGNLSE_CP_STACKED_BACKPROJECTION_MAX_RANK = 184

function _cuda_mmgnlse_cp_optimization(backend::Symbol)
    backend === :cuda && return :baseline
    backend === :cuda_cp_optimized && return :rank_agnostic
    backend === :cuda_optimized && return :rank_tuned
    backend === :cuda_cp_symmetric_experimental &&
        return :symmetric_experimental
    throw(ArgumentError(
        "Unsupported CUDA MMGNLSE backend $(repr(backend))."))
end

@inline _cuda_mmgnlse_cp_is_optimized(overlap) =
    overlap.optimization !== :baseline
@inline _cuda_mmgnlse_cp_is_rank_tuned(overlap) =
    overlap.optimization in (:rank_tuned, :symmetric_experimental)

@inline _cuda_mmgnlse_cp_use_symmetric(overlap) =
    overlap.optimization === :symmetric_experimental &&
    overlap.symmetric_factors

@inline function _cuda_mmgnlse_cp_rank_uses_stacked_backprojection(rank::Int)
    return _CUDA_MMGNLSE_CP_STACKED_BACKPROJECTION_MIN_RANK <= rank <=
           _CUDA_MMGNLSE_CP_STACKED_BACKPROJECTION_MAX_RANK
end

@inline function _cuda_mmgnlse_cp_use_stacked_backprojection(overlap)
    return _cuda_mmgnlse_cp_is_rank_tuned(overlap) &&
           _cuda_mmgnlse_cp_rank_uses_stacked_backprojection(overlap.rank)
end

struct CUDAMMGNLSEDenseSpatial{A,G,R} <: AbstractCUDAMMGNLSEOverlap
    values::A
    gemm_values::G
    symmetric_values::R
    physical_symmetric::Bool
    algorithm::Symbol
end

struct CUDAMMGNLSEDenseComponent{A,G} <: AbstractCUDAMMGNLSEOverlap
    values::A
    gemm_values::G
    nmodes::Int
    npolarizations::Int
    algorithm::Symbol
end

struct CUDAMMGNLSECP{U,A,C,R} <: AbstractCUDAMMGNLSEOverlap
    U::U
    wu1t::A
    cwu1::C
    projection_factors::A
    backprojection_factors::A
    real_projection_factors::R
    real_wu1t::R
    layout::Symbol
    nmodes::Int
    npolarizations::Int
    rank::Int
    optimization::Symbol
    real_factors::Bool
    symmetric_factors::Bool
end

struct CUDAMMGNLSEForwardCache{B,P,N,R,O}
    beta_operator::B
    alpha::P
    gain::P
    nonlinear_prefactor::N
    raman::R
    overlap::O
    nonlinear_active::Bool
    constant_passive::Bool
end

"""
Reusable CUDA storage for the public CP forward solver.

The public API remains host-oriented, but every RK stage, CP projection,
Raman channel, and FFT scratch array is retained on the device for the whole
solve.  This restores the allocation discipline of the legacy CUDA solver
without reintroducing its scalar-only API restrictions.
"""
mutable struct CUDAMMGNLSECPForwardWorkspace{
    A3,V3,A2,R2,H2,P3F,P3I,P2F,P2I,P23F,P23I,P24F,P24I,PRF,PRI,
}
    field_t::A3
    nonlinear::A3
    instantaneous::A3
    delayed::A3
    next_field::A3
    midpoint_base::A3
    stage::A3
    k1::A3
    k2::A3
    k3::A3
    k4::A3
    to_midpoint::A3
    from_midpoint::A3
    passive_integral::A3
    projections::A3
    b2::V3
    b3::V3
    b4::V3
    conjugated::A2
    stacked_field::A2
    stacked_projections::A2
    stacked_rank::A2
    raman_channels::A2
    rank1::A2
    rank2::A2
    rank3::A2
    rank4::A2
    real_rank::R2
    half_rank::H2
    fft_field!::P3F
    ifft_field!::P3I
    fft_rank!::P2F
    ifft_rank!::P2I
    fft_rank3!::P23F
    ifft_rank3!::P23I
    fft_rank4!::P24F
    ifft_rank4!::P24I
    rfft_rank!::PRF
    irfft_rank!::PRI
    propagator_step::Float64
    propagator_valid::Bool
end

"""
Forward-only mixed-precision CP workspace. `state` owns the ComplexF64
interaction-picture state and RK accumulators; `rhs` owns the ComplexF32
nonlinear contraction, Raman, and FFT storage.
"""
struct CUDAMMGNLSEMixedCPForwardWorkspace{S,R}
    state::S
    rhs::R
end

"""FP64 RK state retained by the mixed CP solver, without FP64 CP RHS data."""
mutable struct CUDAMMGNLSECPStateWorkspace{A3,P3F}
    field_t::A3
    next_field::A3
    midpoint_base::A3
    stage::A3
    k1::A3
    k2::A3
    k3::A3
    k4::A3
    to_midpoint::A3
    from_midpoint::A3
    passive_integral::A3
    fft_field!::P3F
    propagator_step::Float64
    propagator_valid::Bool
end

"""
Forward-only mixed-precision dense workspace. `state` owns the ComplexF64
interaction-picture state and RK accumulators; `rhs` owns the ComplexF32
nonlinear contraction, Raman, and FFT storage.
"""
struct CUDAMMGNLSEMixedDenseForwardWorkspace{S,R}
    state::S
    rhs::R
end

"""Extra storage for the embedded adaptive RK4IP controller."""
mutable struct CUDAMMGNLSEAdaptiveWorkspace{W,A}
    integrator::W
    fsal::A
    endpoint_rhs::A
    error::A
end

mutable struct CUDAMMGNLSEDenseForwardWorkspace{A3,P3F,P3I}
    field_t::A3
    nonlinear::A3
    next_field::A3
    midpoint_base::A3
    stage::A3
    k1::A3
    k2::A3
    k3::A3
    k4::A3
    to_midpoint::A3
    from_midpoint::A3
    passive_integral::A3
    fft_field!::P3F
    ifft_field!::P3I
    propagator_step::Float64
    propagator_valid::Bool
end

struct CUDAMMGNLSEDenseRamanForwardWorkspace{B,A3,A4,P,C,P4F,P4I}
    base::B
    delayed::A3
    pairs::A4
    products::P
    contracted::C
    algorithm::Symbol
    tile_time::Int
    fft_pairs!::P4F
    ifft_pairs!::P4I
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

function _cuda_mmgnlse_profile_integral_add!(
    result,
    profile::CUDAMMGNLSEProfile,
    z0::Real,
    z1::Real,
    coefficient,
)
    z1 >= z0 || throw(ArgumentError(
        "Profile integration requires z1 >= z0."))
    z1 == z0 && return result
    nz = size(profile.values, 4)
    if nz == 1
        @views result .+= coefficient * (z1 - z0) .* profile.values[:, :, :, 1]
        return result
    end

    knot_spacing = profile.length / (nz - 1)
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

        left_scaled = clamp(position / profile.length, 0.0, 1.0) * (nz - 1)
        left_index = min(floor(Int, left_scaled) + 1, nz)
        left_right = min(left_index + 1, nz)
        left_weight = left_scaled - (left_index - 1)
        right_scaled = clamp(endpoint / profile.length, 0.0, 1.0) * (nz - 1)
        right_index = min(floor(Int, right_scaled) + 1, nz)
        right_right = min(right_index + 1, nz)
        right_weight = right_scaled - (right_index - 1)
        scale = coefficient * (endpoint - position) / 2
        @views result .+= scale .* (
            (1 - left_weight) .* profile.values[:, :, :, left_index] .+
            left_weight .* profile.values[:, :, :, left_right] .+
            (1 - right_weight) .* profile.values[:, :, :, right_index] .+
            right_weight .* profile.values[:, :, :, right_right])
        position = endpoint
    end
    return result
end

function _cuda_mmgnlse_passive_integral!(
    result, cache::CUDAMMGNLSEForwardCache, z0::Real, z1::Real,
)
    fill!(result, zero(eltype(result)))
    _cuda_mmgnlse_profile_integral_add!(result, cache.gain, z0, z1, 0.5)
    _cuda_mmgnlse_profile_integral_add!(result, cache.alpha, z0, z1, -0.5)
    return result
end

function _cuda_mmgnlse_physical_spatial_tensor(values)
    eltype(values) <: Real || return false
    n1, n2, n3, n4 = size(values)
    n1 == n2 == n3 == n4 || return false
    scale = max(maximum(abs, values), 1.0)
    tolerance = 64eps(Float64) * scale
    @inbounds for i4 in 1:n4, i3 in 1:n3, i2 in 1:n2, i1 in 1:n1
        value = values[i1, i2, i3, i4]
        abs(value - values[i2, i1, i3, i4]) <= tolerance || return false
        abs(value - values[i1, i3, i2, i4]) <= tolerance || return false
        abs(value - values[i1, i2, i4, i3]) <= tolerance || return false
    end
    return true
end

function _cuda_mmgnlse_symmetric_spatial_matrix(
    values, ::Type{T}, physical_symmetric::Bool,
) where {T<:AbstractFloat}
    nm = size(values, 1)
    physical_symmetric || return CUDA.zeros(Complex{T}, 0, 0)
    npairs = nm * (nm + 1) ÷ 2
    matrix = Matrix{Complex{T}}(undef, npairs, nm * nm)
    @inbounds for i4 in 1:nm, i1 in 1:nm, i3 in 1:nm, i2 in 1:i3
        pair = i2 + (i3 - 1) * i3 ÷ 2
        output = i1 + (i4 - 1) * nm
        matrix[pair, output] = Complex{T}(values[i1, i2, i3, i4])
    end
    return CUDA.CuArray(matrix)
end

const _CUDA_MMGNLSE_DENSE_GEMM_MIN_SCALAR_WIDTH = 8
const _CUDA_MMGNLSE_DENSE_GEMM_MIN_POLARIZED_WIDTH = 6
const _CUDA_MMGNLSE_DENSE_GEMM_MIN_SCALAR_WORK = 30_000
const _CUDA_MMGNLSE_DENSE_GEMM_MIN_POLARIZED_WORK = 16_000
const _CUDA_MMGNLSE_DENSE_GEMM_DEFAULT_TILE_TIME = 2048
const _CUDA_MMGNLSE_DENSE_GEMM_DEFAULT_MEMORY_FRACTION = 0.5

function _cuda_mmgnlse_dense_gemm_tile_time(nt::Int)
    requested = parse(Int, get(
        ENV, "PULSEPROPAGATION_CUDA_DENSE_GEMM_TILE_TIME",
        string(_CUDA_MMGNLSE_DENSE_GEMM_DEFAULT_TILE_TIME)))
    requested > 0 || throw(ArgumentError(
        "PULSEPROPAGATION_CUDA_DENSE_GEMM_TILE_TIME must be positive."))
    return min(nt, requested)
end

function _cuda_mmgnlse_dense_gemm_memory_fraction()
    fraction = parse(Float64, get(
        ENV, "PULSEPROPAGATION_CUDA_DENSE_GEMM_MAX_MEMORY_FRACTION",
        string(_CUDA_MMGNLSE_DENSE_GEMM_DEFAULT_MEMORY_FRACTION)))
    0 < fraction <= 1 || throw(ArgumentError(
        "PULSEPROPAGATION_CUDA_DENSE_GEMM_MAX_MEMORY_FRACTION must be " *
        "in (0, 1]."))
    return fraction
end

function _cuda_mmgnlse_dense_gemm_bytes(
    algorithm::Symbol, nt::Int, width::Int, nchannels::Int,
    raman_active::Bool, ::Type{T}, tensor_entries::Int,
) where {T<:AbstractFloat}
    complex_bytes = 2 * sizeof(T)
    tile_time = _cuda_mmgnlse_dense_gemm_tile_time(nt)
    tensor_bytes = tensor_entries * complex_bytes
    if algorithm === :gemm
        tile_entries = tile_time * nchannels * width * width
        pairs_entries = raman_active ? nt * nchannels * width * width :
                        tile_entries
        # Raman retains the full pair history for its temporal FFT and uses
        # two tile buffers.  Instantaneous-only propagation reuses `pairs` as
        # the contracted tile and therefore needs just two tile buffers.
        workspace_entries = pairs_entries + tile_entries +
                            (raman_active ? tile_entries : 0)
    elseif algorithm === :symmetric_gemm
        npairs = width * (width + 1) ÷ 2
        workspace_entries = tile_time * nchannels * (width * width + npairs)
    else
        return 0
    end
    return tensor_bytes + workspace_entries * complex_bytes
end

function _cuda_mmgnlse_dense_contraction_choice(
    parameters, cpu_cache, ::Type{T}; requested::Union{Nothing,Symbol}=nothing,
) where {T<:AbstractFloat}
    overlap = parameters.S
    nt, nm, np = PulsePropagation.parameter_field_shape(parameters)
    spatial = overlap isa PulsePropagation.SpatialOverlap
    width = spatial ? nm : nm * np
    polarized = spatial && np == 2
    physical_symmetric = spatial &&
        _cuda_mmgnlse_physical_spatial_tensor(overlap.values)
    can_reduce = physical_symmetric && iszero(cpu_cache.raman.fraction)
    override = requested === nothing ? Symbol(lowercase(get(
        ENV, "PULSEPROPAGATION_CUDA_DENSE_CONTRACTION", "auto"))) : requested
    allowed = requested === nothing ?
              (:auto, :custom, :gemm, :symmetric_gemm) :
              (:auto, :custom, :gemm, :symmetric_gemm, :none)
    override in allowed ||
        throw(ArgumentError(
            "PULSEPROPAGATION_CUDA_DENSE_CONTRACTION must be " *
            "auto, custom, gemm, or symmetric_gemm; got $(repr(override))"))
    override === :none && return :none, physical_symmetric
    override === :custom && return :custom, physical_symmetric
    if override === :symmetric_gemm
        can_reduce || throw(ArgumentError(
            "symmetric_gemm requires a physical symmetric spatial tensor " *
            "with Raman disabled"))
        return :symmetric_gemm, physical_symmetric
    elseif override === :gemm
        return :gemm, physical_symmetric
    end

    minimum_width = polarized ?
                    _CUDA_MMGNLSE_DENSE_GEMM_MIN_POLARIZED_WIDTH :
                    _CUDA_MMGNLSE_DENSE_GEMM_MIN_SCALAR_WIDTH
    minimum_work = polarized ?
                   _CUDA_MMGNLSE_DENSE_GEMM_MIN_POLARIZED_WORK :
                   _CUDA_MMGNLSE_DENSE_GEMM_MIN_SCALAR_WORK
    (width < minimum_width || nt * width * width < minimum_work) &&
        return :custom, physical_symmetric

    algorithm = can_reduce ? :symmetric_gemm : :gemm
    nchannels = algorithm === :symmetric_gemm ? (polarized ? 3 : 1) :
                (polarized ? 4 : 1)
    tensor_entries = algorithm === :symmetric_gemm ?
                     width * width * (width * (width + 1) ÷ 2) :
                     width^4
    required = _cuda_mmgnlse_dense_gemm_bytes(
        algorithm, nt, width, nchannels,
        !iszero(cpu_cache.raman.fraction), T, tensor_entries)
    cached = coalesce(CUDA.cached_memory(), 0)
    used = coalesce(CUDA.used_memory(), 0)
    reclaimable = max(cached - used, 0)
    available = CUDA.free_memory() + reclaimable
    budget = floor(Int, _cuda_mmgnlse_dense_gemm_memory_fraction() * available)
    if required > budget
        message = "Falling back to the custom dense CUDA contraction because " *
                  "the estimated GEMM tensor/workspace footprint exceeds the " *
                  "configured free-memory budget."
        @warn message algorithm=algorithm required_bytes=required budget_bytes=budget width=width nt=nt nchannels=nchannels
        return :custom, physical_symmetric
    end
    return algorithm, physical_symmetric
end

function _cuda_mmgnlse_overlap(
    overlap::PulsePropagation.SpatialOverlap,
    algorithm::Symbol,
    ::Type{T},
    physical_symmetric::Bool,
) where {T<:AbstractFloat}
    storage_type = eltype(overlap.values) <: Real ? T : Complex{T}
    nm = size(overlap.values, 1)
    CT = Complex{T}
    values = algorithm === :custom ?
             CUDA.CuArray(storage_type.(overlap.values)) :
             CUDA.zeros(storage_type, 0, 0, 0, 0)
    gemm_values = algorithm === :gemm ?
                  CUDA.CuArray(reshape(
                      CT.(overlap.values), nm * nm, nm * nm)) :
                  CUDA.zeros(CT, 0, 0)
    symmetric_values = algorithm === :symmetric_gemm ?
                       _cuda_mmgnlse_symmetric_spatial_matrix(
                           overlap.values, T, physical_symmetric) :
                       CUDA.zeros(CT, 0, 0)
    return CUDAMMGNLSEDenseSpatial(
        values, gemm_values, symmetric_values, physical_symmetric, algorithm)
end

function _cuda_mmgnlse_overlap(
    overlap::PulsePropagation.ComponentOverlap,
    algorithm::Symbol,
    ::Type{T},
    ::Bool,
) where {T<:AbstractFloat}
    flattened = PulsePropagation.flatten_component_tensor(overlap.values)
    storage_type = eltype(flattened) <: Real ? T : Complex{T}
    nc = size(flattened, 1)
    CT = Complex{T}
    values = algorithm === :custom ?
             CUDA.CuArray(storage_type.(flattened)) :
             CUDA.zeros(storage_type, 0, 0, 0, 0)
    gemm_values = algorithm === :gemm ?
                  CUDA.CuArray(reshape(CT.(flattened), nc * nc, nc * nc)) :
                  CUDA.zeros(CT, 0, 0)
    return CUDAMMGNLSEDenseComponent(
        values, gemm_values, size(overlap.values, 1),
        size(overlap.values, 2), algorithm)
end

_cuda_mmgnlse_overlap(overlap::PulsePropagation.SpatialOverlap) =
    _cuda_mmgnlse_overlap(
        overlap, :custom, Float64,
        _cuda_mmgnlse_physical_spatial_tensor(overlap.values))

_cuda_mmgnlse_overlap(overlap::PulsePropagation.ComponentOverlap) =
    _cuda_mmgnlse_overlap(overlap, :custom, Float64, false)

_cuda_mmgnlse_overlap(
    overlap::Union{
        PulsePropagation.SpatialOverlap,
        PulsePropagation.ComponentOverlap,
    },
    ::Symbol,
) = _cuda_mmgnlse_overlap(overlap)

function _cuda_mmgnlse_overlap(
    overlap::PulsePropagation.MMGNLSECPDecomposition,
    optimization::Symbol=:baseline,
)
    return _cuda_mmgnlse_overlap(overlap, optimization, Float64)
end

function _cuda_mmgnlse_overlap(
    overlap::PulsePropagation.MMGNLSECPDecomposition,
    optimization::Symbol,
    ::Type{T},
) where {T<:AbstractFloat}
    CT = Complex{T}
    real_factors = eltype(overlap.λ) <: Real &&
                   all(eltype(factor) <: Real for factor in overlap.U)
    symmetric_factors = real_factors &&
                        all(overlap.U[index] == overlap.U[1] for index in 2:4)
    if optimization === :symmetric_experimental && !symmetric_factors
        throw(ArgumentError(
            "The experimental symmetric CUDA CP backend requires real " *
            "weights and four exactly equal real factor matrices."))
    end
    factors_cpu = ntuple(index -> CT.(overlap.U[index]), 4)
    weights = CT.(overlap.λ)
    weighted_u1 = factors_cpu[1] .* reshape(weights, 1, :)
    factors = ntuple(index -> CUDA.CuArray(factors_cpu[index]), 4)
    wu1t = CUDA.CuArray(Matrix(transpose(weighted_u1)))
    cwu1 = CUDA.CuArray(conj.(weighted_u1))
    projection_factors =
        if optimization === :rank_tuned &&
           length(weights) <= _CUDA_MMGNLSE_CP_STACKED_PROJECTION_MAX_RANK
        CUDA.CuArray(hcat(
            factors_cpu[2], factors_cpu[3], conj.(factors_cpu[4])))
    else
        CUDA.zeros(CT, size(factors_cpu[2], 1), 0)
    end
    backprojection_factors =
        if optimization === :rank_tuned &&
           _cuda_mmgnlse_cp_rank_uses_stacked_backprojection(length(weights))
            CUDA.CuArray(vcat(
                Matrix(adjoint(factors_cpu[2])),
                Matrix(adjoint(factors_cpu[3])),
                Matrix(transpose(factors_cpu[4])),
            ))
    else
        CUDA.zeros(CT, 0, size(factors_cpu[2], 1))
    end
    real_projection_factors =
        if real_factors && optimization !== :baseline
            projected = symmetric_factors ? overlap.U[2] :
                        hcat(overlap.U[2], overlap.U[3], overlap.U[4])
            CUDA.CuArray(T.(projected))
        else
            CUDA.zeros(T, size(factors_cpu[2], 1), 0)
        end
    real_weighted_u1 =
        if real_factors && optimization !== :baseline
            weighted = overlap.U[1] .* reshape(overlap.λ, 1, :)
            CUDA.CuArray(Matrix{T}(transpose(weighted)))
        else
            CUDA.zeros(T, 0, size(factors_cpu[1], 1))
        end
    return CUDAMMGNLSECP(
        factors, wu1t, cwu1, projection_factors, backprojection_factors,
        real_projection_factors, real_weighted_u1,
        overlap.metadata.layout, overlap.metadata.nmodes,
        overlap.metadata.npolarizations, length(weights), optimization,
        real_factors, symmetric_factors)
end

function _cuda_mmgnlse_forward_cache(
    parameters,
    cp_optimization::Symbol=:baseline,
    ; dense_contraction::Union{Nothing,Symbol}=nothing,
)
    return _cuda_mmgnlse_forward_cache(
        parameters, cp_optimization, Float64; dense_contraction)
end

function _cuda_mmgnlse_forward_cache(
    parameters,
    cp_optimization::Symbol,
    ::Type{T},
    ; dense_contraction::Union{Nothing,Symbol}=nothing,
) where {T<:AbstractFloat}
    CT = Complex{T}
    cpu_cache = PulsePropagation._mmgnlse_solver_cache(parameters)
    combined_raman = cpu_cache.raman.ha .+ cpu_cache.raman.hb
    combined_raman_r2c = isempty(combined_raman) ? ComplexF64[] :
        conj.(@view combined_raman[1:(length(combined_raman) ÷ 2 + 1)])
    raman = (
        ha=CUDA.CuArray(CT.(cpu_cache.raman.ha)),
        hb=CUDA.CuArray(CT.(cpu_cache.raman.hb)),
        combined=CUDA.CuArray(CT.(combined_raman)),
        combined_r2c=CUDA.CuArray(CT.(combined_raman_r2c)),
        fraction=T(cpu_cache.raman.fraction),
        ha_active=any(value -> !iszero(value), cpu_cache.raman.ha),
        hb_active=any(value -> !iszero(value), cpu_cache.raman.hb),
    )
    overlap = if parameters.S isa PulsePropagation.MMGNLSECPDecomposition
        _cuda_mmgnlse_overlap(parameters.S, cp_optimization, T)
    else
        algorithm, physical_symmetric =
            _cuda_mmgnlse_dense_contraction_choice(
                parameters, cpu_cache, T; requested=dense_contraction)
        _cuda_mmgnlse_overlap(
            parameters.S, algorithm, T, physical_symmetric)
    end
    alpha = _cuda_mmgnlse_profile(parameters.alpha, parameters)
    gain = _cuda_mmgnlse_profile(parameters.gain, parameters)
    return CUDAMMGNLSEForwardCache(
        CUDA.CuArray(CT.(cpu_cache.beta_operator)),
        alpha,
        gain,
        CUDA.CuArray(CT.(cpu_cache.nonlinear_prefactor)),
        raman,
        overlap,
        cpu_cache.nonlinear_active,
        size(alpha.values, 4) == 1 && size(gain.values, 4) == 1,
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

function _cuda_mmgnlse_linear_propagator!(
    out, cache::CUDAMMGNLSEForwardCache, z0::Real, z1::Real, scratch,
)
    step = z1 - z0
    if cache.constant_passive
        @views out .= exp.(
            cache.beta_operator .* step .+
            (cache.gain.values[:, :, :, 1] .-
             cache.alpha.values[:, :, :, 1]) .* (step / 2))
    else
        _cuda_mmgnlse_passive_integral!(scratch, cache, z0, z1)
        out .= exp.(cache.beta_operator .* step .+ scratch)
    end
    return out
end

function _cuda_mmgnlse_dense_spatial_kernel!(
    out, field, S, nt::Int, nm::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        two_thirds = RT(2) / RT(3)
        one_third = RT(1) / RT(3)
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
                        two_thirds * field[t, i2, 1] * field[t, i3, 2] *
                        conj(field[t, i4, 2]) +
                        one_third * field[t, i2, 2] * field[t, i3, 2] *
                        conj(field[t, i4, 1]))
                end
            else
                accumulator += s * (
                    field[t, i2, 2] * field[t, i3, 2] *
                    conj(field[t, i4, 2]) +
                    two_thirds * field[t, i2, 2] * field[t, i3, 1] *
                    conj(field[t, i4, 1]) +
                    one_third * field[t, i2, 1] * field[t, i3, 1] *
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

# Keep the dense Raman contraction pair-first, as in the legacy CUDA solver:
# each GPU thread owns one (time, output, input) channel and reduces only the
# two coherence indices.  The temporal Raman filter is then applied to these
# contracted channels before the inexpensive input-mode combine below.
function _cuda_mmgnlse_dense_raman_pairs_kernel!(
    pairs, field, S, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nc + 1
        i2 = remainder ÷ nc + 1
        accumulator = zero(eltype(pairs))
        @inbounds for i4 in 1:nc, i3 in 1:nc
            accumulator += S[i1, i2, i3, i4] *
                           field[t, i3] * conj(field[t, i4])
        end
        @inbounds pairs[t, i1, i2] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_pair_products_kernel!(
    products, field, nt::Int, nc::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        remainder = (index - 1) ÷ tile_nt
        i3 = remainder % nc + 1
        i4 = remainder ÷ nc + 1
        @inbounds products[local_t, 1, i3, i4] =
            field[t, i3] * conj(field[t, i4])
    end
    return nothing
end

function _cuda_mmgnlse_dense_polarized_pair_products_kernel!(
    products, field, nt::Int, nm::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        remainder = (index - 1) ÷ tile_nt
        channel = remainder % 4 + 1
        remainder ÷= 4
        i3 = remainder % nm + 1
        i4 = remainder ÷ nm + 1
        @inbounds begin
            x3 = field[t, i3, 1]
            y3 = field[t, i3, 2]
            x4 = conj(field[t, i4, 1])
            y4 = conj(field[t, i4, 2])
            products[local_t, channel, i3, i4] =
                channel == 1 ? x3 * x4 :
                channel == 2 ? y3 * y4 :
                channel == 3 ? x3 * y4 : y3 * x4
        end
    end
    return nothing
end

function _cuda_mmgnlse_dense_symmetric_products_kernel!(
    products, field, nt::Int, nm::Int, np::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        remainder = (index - 1) ÷ tile_nt
        i2 = remainder % nm + 1
        i3 = remainder ÷ nm + 1
        if i2 <= i3
            pair = i2 + (i3 - 1) * i3 ÷ 2
            @inbounds begin
                x2 = field[t, i2, 1]
                x3 = field[t, i3, 1]
                products[local_t, 1, pair] =
                    i2 == i3 ? x2 * x3 : x2 * x3 + x3 * x2
                if np == 2
                    y2 = field[t, i2, 2]
                    y3 = field[t, i3, 2]
                    products[local_t, 2, pair] =
                        i2 == i3 ? y2 * y3 : y2 * y3 + y3 * y2
                    products[local_t, 3, pair] =
                        i2 == i3 ? x2 * y3 : x2 * y3 + x3 * y2
                end
            end
        end
    end
    return nothing
end

function _cuda_mmgnlse_dense_symmetric_combine_kernel!(
    out, field, pairs, nt::Int, nm::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        two_thirds = RT(2) / RT(3)
        one_third = RT(1) / RT(3)
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm
            if polarization == 1
                accumulator += pairs[t, 1, i1, i4] * conj(field[t, i4, 1])
                if np == 2
                    accumulator +=
                        two_thirds * pairs[t, 3, i1, i4] *
                            conj(field[t, i4, 2]) +
                        one_third * pairs[t, 2, i1, i4] *
                            conj(field[t, i4, 1])
                end
            else
                accumulator +=
                    pairs[t, 2, i1, i4] * conj(field[t, i4, 2]) +
                    two_thirds * pairs[t, 3, i1, i4] *
                        conj(field[t, i4, 1]) +
                    one_third * pairs[t, 1, i1, i4] *
                        conj(field[t, i4, 2])
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_raman_combine_kernel!(
    out, field, filtered_pairs, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        i1 = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i2 in 1:nc
            accumulator += filtered_pairs[t, i1, i2] * field[t, i2]
        end
        @inbounds out[t, i1] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_polarized_raman_pairs_kernel!(
    qa, qxx, qyy, qxy, field, S, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        i2 = remainder ÷ nm + 1
        accumulator_xx = zero(eltype(qa))
        accumulator_yy = zero(eltype(qa))
        accumulator_xy = zero(eltype(qa))
        @inbounds for i4 in 1:nm, i3 in 1:nm
            s = S[i1, i2, i3, i4]
            x3 = field[t, i3, 1]
            y3 = field[t, i3, 2]
            x4 = conj(field[t, i4, 1])
            y4 = conj(field[t, i4, 2])
            accumulator_xx += s * x3 * x4
            accumulator_yy += s * y3 * y4
            accumulator_xy += s * (x3 * y4 + y3 * x4)
        end
        @inbounds begin
            qa[t, i1, i2] = accumulator_xx + accumulator_yy
            qxx[t, i1, i2] = accumulator_xx
            qyy[t, i1, i2] = accumulator_yy
            qxy[t, i1, i2] = accumulator_xy
        end
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_polarized_raman_combine_kernel!(
    out, field, qa, qxx, qyy, qxy, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        half = RT(1) / RT(2)
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i2 in 1:nm
            if polarization == 1
                accumulator +=
                    field[t, i2, 1] *
                    (qa[t, i1, i2] + qxx[t, i1, i2]) +
                    half * field[t, i2, 2] * qxy[t, i1, i2]
            else
                accumulator +=
                    field[t, i2, 2] *
                    (qa[t, i1, i2] + qyy[t, i1, i2]) +
                    half * field[t, i2, 1] * qxy[t, i1, i2]
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_polarized_kerr_pairs_kernel!(
    pxx, pyy, pxy, pyx, field, S, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        i2 = remainder ÷ nm + 1
        accumulator_xx = zero(eltype(pxx))
        accumulator_yy = zero(eltype(pxx))
        accumulator_xy = zero(eltype(pxx))
        accumulator_yx = zero(eltype(pxx))
        @inbounds for i4 in 1:nm, i3 in 1:nm
            s = S[i1, i2, i3, i4]
            x3 = field[t, i3, 1]
            y3 = field[t, i3, 2]
            x4 = conj(field[t, i4, 1])
            y4 = conj(field[t, i4, 2])
            accumulator_xx += s * x3 * x4
            accumulator_yy += s * y3 * y4
            accumulator_xy += s * x3 * y4
            accumulator_yx += s * y3 * x4
        end
        @inbounds begin
            pxx[t, i1, i2] = accumulator_xx
            pyy[t, i1, i2] = accumulator_yy
            pxy[t, i1, i2] = accumulator_xy
            pyx[t, i1, i2] = accumulator_yx
        end
    end
    return nothing
end

function _cuda_mmgnlse_dense_polarized_pair_rearrange_kernel!(
    pairs, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        i2 = remainder ÷ nm + 1
        @inbounds begin
            xx = pairs[t, 1, i1, i2]
            yy = pairs[t, 2, i1, i2]
            xy = pairs[t, 3, i1, i2]
            yx = pairs[t, 4, i1, i2]
            pairs[t, 1, i1, i2] = xx + yy
            pairs[t, 2, i1, i2] = xx
            pairs[t, 3, i1, i2] = yy
            pairs[t, 4, i1, i2] = xy + yx
        end
    end
    return nothing
end

function _cuda_mmgnlse_dense_polarized_filter_kernel!(
    pairs, ha, hb, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        channel = ((index - 1) ÷ nt) % 4 + 1
        @inbounds pairs[index] *= channel == 1 ? ha[t] : hb[t]
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_polarized_kerr_combine_kernel!(
    out, field, pxx, pyy, pxy, pyx, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        two_thirds = RT(2) / RT(3)
        one_third = RT(1) / RT(3)
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i2 in 1:nm
            if polarization == 1
                accumulator +=
                    field[t, i2, 1] *
                    (pxx[t, i1, i2] + two_thirds * pyy[t, i1, i2]) +
                    one_third * field[t, i2, 2] * pyx[t, i1, i2]
            else
                accumulator +=
                    field[t, i2, 2] *
                    (pyy[t, i1, i2] + two_thirds * pxx[t, i1, i2]) +
                    one_third * field[t, i2, 1] * pxy[t, i1, i2]
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_scalar_combine_tiled_kernel!(
    out, field, pairs, nt::Int, nc::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        i1 = (index - 1) ÷ tile_nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i2 in 1:nc
            accumulator += pairs[local_t, 1, i1, i2] * field[t, i2]
        end
        @inbounds out[t, i1] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_polarized_combine_tiled_kernel!(
    out, field, pairs, nt::Int, nm::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        two_thirds = RT(2) / RT(3)
        one_third = RT(1) / RT(3)
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        remainder = (index - 1) ÷ tile_nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i2 in 1:nm
            if polarization == 1
                accumulator +=
                    field[t, i2, 1] *
                    (pairs[local_t, 1, i1, i2] +
                     two_thirds * pairs[local_t, 2, i1, i2]) +
                    one_third * field[t, i2, 2] *
                    pairs[local_t, 4, i1, i2]
            else
                accumulator +=
                    field[t, i2, 2] *
                    (pairs[local_t, 2, i1, i2] +
                     two_thirds * pairs[local_t, 1, i1, i2]) +
                    one_third * field[t, i2, 1] *
                    pairs[local_t, 3, i1, i2]
            end
        end
        @inbounds out[t, i1, polarization] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_symmetric_combine_tiled_kernel!(
    out, field, pairs, nt::Int, nm::Int, np::Int, tile_nt::Int,
    time_offset::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        RT = typeof(real(zero(eltype(out))))
        two_thirds = RT(2) / RT(3)
        one_third = RT(1) / RT(3)
        local_t = (index - 1) % tile_nt + 1
        t = local_t + time_offset
        remainder = (index - 1) ÷ tile_nt
        i1 = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm
            if polarization == 1
                accumulator += pairs[local_t, 1, i1, i4] *
                               conj(field[t, i4, 1])
                if np == 2
                    accumulator +=
                        two_thirds * pairs[local_t, 3, i1, i4] *
                            conj(field[t, i4, 2]) +
                        one_third * pairs[local_t, 2, i1, i4] *
                            conj(field[t, i4, 1])
                end
            else
                accumulator +=
                    pairs[local_t, 2, i1, i4] * conj(field[t, i4, 2]) +
                    two_thirds * pairs[local_t, 3, i1, i4] *
                        conj(field[t, i4, 1]) +
                    one_third * pairs[local_t, 1, i1, i4] *
                        conj(field[t, i4, 2])
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
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        flattened = reshape(field, nt, nm)
        pairs = CUDA.zeros(ComplexF64, nt, nm, nm)
        _launch_1d(
            _cuda_mmgnlse_dense_raman_pairs_kernel!,
            length(pairs), pairs, flattened, overlap.values, nt, nm)
        filtered = _cuda_mmgnlse_filter_raman(
            pairs, raman.ha .+ raman.hb)
        _launch_1d(
            _cuda_mmgnlse_dense_raman_combine_kernel!,
            nt * nm, reshape(out, nt, nm), flattened, filtered, nt, nm)
        return out
    end

    qa = CUDA.zeros(ComplexF64, nt, nm, nm)
    qxx = similar(qa)
    qyy = similar(qa)
    qxy = similar(qa)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_polarized_raman_pairs_kernel!,
        length(qa), qa, qxx, qyy, qxy, field, overlap.values, nt, nm)
    qa = _cuda_mmgnlse_filter_raman(qa, raman.ha)
    qxx = _cuda_mmgnlse_filter_raman(qxx, raman.hb)
    qyy = _cuda_mmgnlse_filter_raman(qyy, raman.hb)
    qxy = _cuda_mmgnlse_filter_raman(qxy, raman.hb)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_polarized_raman_combine_kernel!,
        length(out), out, field, qa, qxx, qyy, qxy,
        nt, nm)
    return out
end

function _cuda_mmgnlse_delayed(
    field, overlap::CUDAMMGNLSEDenseComponent, raman,
)
    nt, nm, np = size(field)
    nc = nm * np
    flattened = reshape(field, nt, nc)
    pairs = CUDA.zeros(ComplexF64, nt, nc, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_raman_pairs_kernel!,
        length(pairs), pairs, flattened, overlap.values, nt, nc)
    filtered = _cuda_mmgnlse_filter_raman(
        pairs, raman.ha .+ raman.hb)
    out = CUDA.zeros(ComplexF64, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_raman_combine_kernel!, length(out),
        out, flattened, filtered, nt, nc)
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

function _cuda_mmgnlse_dense_forward_workspace(
    field::CUDA.CuArray{Complex{T},3},
) where {T<:AbstractFloat}
    arrays = ntuple(_ -> similar(field), 12)
    field_t, nonlinear, next_field, midpoint_base, stage,
        k1, k2, k3, k4, to_midpoint, from_midpoint,
        passive_integral = arrays
    return CUDAMMGNLSEDenseForwardWorkspace(
        field_t, nonlinear, next_field, midpoint_base, stage,
        k1, k2, k3, k4, to_midpoint, from_midpoint,
        passive_integral,
        plan_fft!(field_t, 1), plan_ifft!(nonlinear, 1), NaN, false)
end

function _cuda_mmgnlse_dense_contraction_algorithm(field, cache)
    overlap = cache.overlap
    overlap isa Union{CUDAMMGNLSEDenseSpatial,CUDAMMGNLSEDenseComponent} ||
        throw(ArgumentError("Dense contraction requested for a CP overlap."))
    return overlap.algorithm
end

function _cuda_mmgnlse_dense_raman_forward_workspace(
    field::CUDA.CuArray{Complex{T},3},
    cache::CUDAMMGNLSEForwardCache,
) where {T<:AbstractFloat}
    CT = Complex{T}
    base = _cuda_mmgnlse_dense_forward_workspace(field)
    nt, nm, np = size(field)
    overlap = cache.overlap
    width = overlap isa CUDAMMGNLSEDenseSpatial ? nm : nm * np
    polarized = overlap isa CUDAMMGNLSEDenseSpatial && np == 2
    algorithm = _cuda_mmgnlse_dense_contraction_algorithm(field, cache)
    algorithm === :none && throw(ArgumentError(
        "A state-only dense cache cannot evaluate the nonlinear RHS."))
    nchannels = algorithm === :symmetric_gemm ? (polarized ? 3 : 1) :
                (polarized ? 4 : 1)
    raman_active = !iszero(cache.raman.fraction)
    tile_time = algorithm in (:gemm, :symmetric_gemm) ?
                _cuda_mmgnlse_dense_gemm_tile_time(nt) : nt
    pairs = if algorithm === :custom || raman_active
        CUDA.zeros(CT, nt, nchannels, width, width)
    else
        CUDA.zeros(CT, tile_time, nchannels, width, width)
    end
    products = if algorithm === :gemm
        CUDA.zeros(CT, tile_time, nchannels, width, width)
    elseif algorithm === :symmetric_gemm
        CUDA.zeros(
            CT, tile_time, nchannels, width * (width + 1) ÷ 2)
    else
        CUDA.zeros(CT, 0, 0, 0, 0)
    end
    contracted = if algorithm === :gemm && raman_active
        CUDA.zeros(CT, tile_time, nchannels, width, width)
    elseif algorithm in (:gemm, :symmetric_gemm)
        pairs
    else
        CUDA.zeros(CT, 0, 0, 0, 0)
    end
    fft_pairs!, ifft_pairs! = if !raman_active
        nothing, nothing
    else
        plan_fft!(pairs, 1), plan_ifft!(pairs, 1)
    end
    return CUDAMMGNLSEDenseRamanForwardWorkspace(
        base, similar(field), pairs, products, contracted, algorithm,
        tile_time,
        fft_pairs!, ifft_pairs!)
end

function _cuda_mmgnlse_dense_raman_forward_workspace(
    field::CUDA.CuArray{Complex{T},3},
    overlap::AbstractCUDAMMGNLSEOverlap,
) where {T<:AbstractFloat}
    CT = Complex{T}
    base = _cuda_mmgnlse_dense_forward_workspace(field)
    nt, nm, np = size(field)
    width = overlap isa CUDAMMGNLSEDenseSpatial ? nm : nm * np
    polarized = overlap isa CUDAMMGNLSEDenseSpatial && np == 2
    nchannels = polarized ? 4 : 1
    pairs = CUDA.zeros(CT, nt, nchannels, width, width)
    products = CUDA.zeros(CT, 0, 0, 0, 0)
    contracted = CUDA.zeros(CT, 0, 0, 0, 0)
    return CUDAMMGNLSEDenseRamanForwardWorkspace(
        base, similar(field), pairs, products, contracted, :custom, nt,
        plan_fft!(pairs, 1), plan_ifft!(pairs, 1))
end

@inline _cuda_mmgnlse_dense_pair(workspace, channel::Int) =
    @view workspace.pairs[:, channel, :, :]

_cuda_mmgnlse_dense_base(workspace::CUDAMMGNLSEDenseForwardWorkspace) =
    workspace
_cuda_mmgnlse_dense_base(
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace) = workspace.base
_cuda_mmgnlse_dense_base(
    workspace::CUDAMMGNLSEMixedDenseForwardWorkspace) =
        _cuda_mmgnlse_dense_base(workspace.state)

function _cuda_mmgnlse_dense_forward_workspace(
    field::CUDA.CuArray{Complex{T},3},
    cache::CUDAMMGNLSEForwardCache,
) where {T<:AbstractFloat}
    algorithm = _cuda_mmgnlse_dense_contraction_algorithm(field, cache)
    return iszero(cache.raman.fraction) && algorithm === :custom ?
           _cuda_mmgnlse_dense_forward_workspace(field) :
           _cuda_mmgnlse_dense_raman_forward_workspace(field, cache)
end

function _cuda_mmgnlse_dense_precision_workspace(
    initial_field::CUDA.CuArray{ComplexF64,3},
    state_cache::CUDAMMGNLSEForwardCache,
    rhs_cache::CUDAMMGNLSEForwardCache,
    precision::Symbol,
)
    state = precision === :float64 ?
            _cuda_mmgnlse_dense_forward_workspace(initial_field, state_cache) :
            _cuda_mmgnlse_dense_forward_workspace(initial_field)
    precision === :float64 && return state
    rhs_field = similar(initial_field, ComplexF32)
    rhs_field .= initial_field
    rhs = _cuda_mmgnlse_dense_forward_workspace(rhs_field, rhs_cache)
    return CUDAMMGNLSEMixedDenseForwardWorkspace(state, rhs)
end

_cuda_mmgnlse_dense_state_workspace(
    workspace::CUDAMMGNLSEDenseForwardWorkspace) = workspace
_cuda_mmgnlse_dense_state_workspace(
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace) = workspace.base
_cuda_mmgnlse_dense_state_workspace(
    workspace::CUDAMMGNLSEMixedDenseForwardWorkspace) =
        _cuda_mmgnlse_dense_base(workspace.state)

function _cuda_mmgnlse_dense_instantaneous!(
    out, field, overlap::CUDAMMGNLSEDenseSpatial,
)
    nt, nm, np = size(field)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_kernel!, length(out),
        out, field, overlap.values, nt, nm, np)
    return out
end

function _cuda_mmgnlse_dense_instantaneous!(
    out, field, overlap::CUDAMMGNLSEDenseComponent,
)
    nt, nm, np = size(field)
    nc = nm * np
    _launch_1d(
        _cuda_mmgnlse_dense_component_kernel!, length(out),
        reshape(out, nt, nc), reshape(field, nt, nc),
        overlap.values, nt, nc)
    return out
end

# Materialize and contract one time tile.  Raman propagation copies each
# contracted tile into the full pair history needed by the temporal FFT;
# instantaneous-only propagation combines the tile immediately.
function _cuda_mmgnlse_dense_gemm_tile!(
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
    field,
    overlap::Union{CUDAMMGNLSEDenseSpatial,CUDAMMGNLSEDenseComponent},
    time_offset::Int,
    active_time::Int,
)
    nt = size(field, 1)
    width = size(workspace.contracted, 3)
    nchannels = size(workspace.contracted, 2)
    tile_time = workspace.tile_time
    if nchannels == 1
        _launch_1d(
            _cuda_mmgnlse_dense_pair_products_kernel!,
            active_time * width * width, workspace.products,
            reshape(field, nt, width), nt, width, active_time,
            time_offset)
    else
        _launch_1d(
            _cuda_mmgnlse_dense_polarized_pair_products_kernel!,
            active_time * nchannels * width * width, workspace.products,
            field, nt, width, active_time, time_offset)
    end
    mul!(
        reshape(workspace.contracted, tile_time * nchannels, width * width),
        reshape(workspace.products, tile_time * nchannels, width * width),
        transpose(overlap.gemm_values))
    if size(workspace.pairs, 1) == nt
        @views workspace.pairs[
            time_offset + 1:time_offset + active_time, :, :, :] .=
            workspace.contracted[1:active_time, :, :, :]
    end
    return workspace.contracted
end

function _cuda_mmgnlse_dense_symmetric_gemm!(
    out,
    field,
    overlap::CUDAMMGNLSEDenseSpatial,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    nt, nm, np = size(field)
    nchannels = size(workspace.contracted, 2)
    npairs = nm * (nm + 1) ÷ 2
    tile_time = workspace.tile_time
    for time_offset in 0:tile_time:nt-1
        active_time = min(tile_time, nt - time_offset)
        _launch_1d(
            _cuda_mmgnlse_dense_symmetric_products_kernel!,
            active_time * nm * nm, workspace.products, field, nt, nm, np,
            active_time, time_offset)
        mul!(
            reshape(workspace.contracted,
                    tile_time * nchannels, nm * nm),
            reshape(workspace.products,
                    tile_time * nchannels, npairs),
            overlap.symmetric_values)
        _launch_1d(
            _cuda_mmgnlse_dense_symmetric_combine_tiled_kernel!,
            active_time * nm * np, out, field, workspace.contracted,
            nt, nm, np, active_time, time_offset)
    end
    return out
end

function _cuda_mmgnlse_dense_gemm_instantaneous!(
    out, field,
    overlap::Union{CUDAMMGNLSEDenseSpatial,CUDAMMGNLSEDenseComponent},
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    nt, nm, np = size(field)
    width = overlap isa CUDAMMGNLSEDenseSpatial ? nm : nm * np
    tile_time = workspace.tile_time
    for time_offset in 0:tile_time:nt-1
        active_time = min(tile_time, nt - time_offset)
        contracted = _cuda_mmgnlse_dense_gemm_tile!(
            workspace, field, overlap, time_offset, active_time)
        if overlap isa CUDAMMGNLSEDenseSpatial && np == 2
            _launch_1d(
                _cuda_mmgnlse_dense_polarized_combine_tiled_kernel!,
                active_time * nm * np, out, field, contracted,
                nt, nm, active_time, time_offset)
        else
            flattened = reshape(field, nt, width)
            _launch_1d(
                _cuda_mmgnlse_dense_scalar_combine_tiled_kernel!,
                active_time * width, reshape(out, nt, width), flattened,
                contracted, nt, width, active_time, time_offset)
        end
    end
    return out
end

function _cuda_mmgnlse_dense_pair_instantaneous!(
    out,
    field,
    overlap::CUDAMMGNLSEDenseSpatial,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    nt, nm, np = size(field)
    if workspace.algorithm === :symmetric_gemm
        return _cuda_mmgnlse_dense_symmetric_gemm!(
            out, field, overlap, workspace)
    elseif workspace.algorithm === :gemm
        return _cuda_mmgnlse_dense_gemm_instantaneous!(
            out, field, overlap, workspace)
    end
    if np == 1
        flattened = reshape(field, nt, nm)
        pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
        if workspace.algorithm === :custom
            _launch_1d(
                _cuda_mmgnlse_dense_raman_pairs_kernel!,
                length(pair1), pair1, flattened,
                overlap.values, nt, nm)
        end
        _launch_1d(
            _cuda_mmgnlse_dense_raman_combine_kernel!,
            nt * nm, reshape(out, nt, nm), flattened,
            pair1, nt, nm)
    else
        pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
        pair2 = _cuda_mmgnlse_dense_pair(workspace, 2)
        pair3 = _cuda_mmgnlse_dense_pair(workspace, 3)
        pair4 = _cuda_mmgnlse_dense_pair(workspace, 4)
        if workspace.algorithm === :custom
            _launch_1d(
                _cuda_mmgnlse_dense_spatial_polarized_kerr_pairs_kernel!,
                length(pair1), pair1, pair2, pair3, pair4, field,
                overlap.values, nt, nm)
        end
        _launch_1d(
            _cuda_mmgnlse_dense_spatial_polarized_kerr_combine_kernel!,
            length(out), out, field, pair1, pair2, pair3, pair4, nt, nm)
    end
    return out
end

function _cuda_mmgnlse_dense_pair_instantaneous!(
    out,
    field,
    overlap::CUDAMMGNLSEDenseComponent,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    nt, nm, np = size(field)
    nc = nm * np
    flattened = reshape(field, nt, nc)
    if workspace.algorithm === :gemm
        return _cuda_mmgnlse_dense_gemm_instantaneous!(
            out, field, overlap, workspace)
    end
    pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
    _launch_1d(
        _cuda_mmgnlse_dense_raman_pairs_kernel!,
        length(pair1), pair1, flattened,
        overlap.values, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_raman_combine_kernel!,
        length(out), reshape(out, nt, nc), flattened,
        pair1, nt, nc)
    return out
end

function _cuda_mmgnlse_dense_nonlinear_spectral!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEDenseForwardWorkspace,
)
    if !cache.nonlinear_active
        fill!(out, zero(eltype(out)))
        return out
    end
    workspace.field_t .= field_w
    workspace.fft_field! * workspace.field_t
    _cuda_mmgnlse_dense_instantaneous!(
        workspace.nonlinear, workspace.field_t, cache.overlap)
    workspace.ifft_field! * workspace.nonlinear
    out .= cache.nonlinear_prefactor .* workspace.nonlinear
    return out
end

function _cuda_mmgnlse_filter_pairs!(
    kernel,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    workspace.ifft_pairs! * workspace.pairs
    workspace.pairs .*= reshape(kernel, :, 1, 1, 1)
    workspace.fft_pairs! * workspace.pairs
    return workspace.pairs
end

function _cuda_mmgnlse_filter_polarized_pairs!(
    ha,
    hb,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    workspace.ifft_pairs! * workspace.pairs
    nt, _, nm, _ = size(workspace.pairs)
    _launch_1d(
        _cuda_mmgnlse_dense_polarized_filter_kernel!,
        length(workspace.pairs), workspace.pairs, ha, hb, nt, nm)
    workspace.fft_pairs! * workspace.pairs
    return workspace.pairs
end

function _cuda_mmgnlse_dense_nonlinear_spectral!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEDenseRamanForwardWorkspace,
)
    base = workspace.base
    if !cache.nonlinear_active
        fill!(out, zero(eltype(out)))
        return out
    end
    base.field_t .= field_w
    base.fft_field! * base.field_t
    _cuda_mmgnlse_dense_pair_instantaneous!(
        base.nonlinear, base.field_t, cache.overlap, workspace)

    fraction = cache.raman.fraction
    if iszero(fraction)
        base.ifft_field! * base.nonlinear
        out .= cache.nonlinear_prefactor .* base.nonlinear
        return out
    end

    nt, nm, np = size(base.field_t)
    if cache.overlap isa CUDAMMGNLSEDenseSpatial
        if np == 1
            flattened = reshape(base.field_t, nt, nm)
            pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
            _cuda_mmgnlse_filter_pairs!(cache.raman.combined, workspace)
            _launch_1d(
                _cuda_mmgnlse_dense_raman_combine_kernel!,
                nt * nm, reshape(workspace.delayed, nt, nm), flattened,
                pair1, nt, nm)
        else
            # Kerr leaves (xx, yy, xy, yx) in the four pair buffers.
            # Form the anisotropic Raman channels without repeating the
            # O(Nt * Nm^4) overlap contraction.
            _launch_1d(
                _cuda_mmgnlse_dense_polarized_pair_rearrange_kernel!,
                length(workspace.pairs) ÷ 4, workspace.pairs, nt, nm)
            _cuda_mmgnlse_filter_polarized_pairs!(
                cache.raman.ha, cache.raman.hb, workspace)
            pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
            pair2 = _cuda_mmgnlse_dense_pair(workspace, 2)
            pair3 = _cuda_mmgnlse_dense_pair(workspace, 3)
            pair4 = _cuda_mmgnlse_dense_pair(workspace, 4)
            _launch_1d(
                _cuda_mmgnlse_dense_spatial_polarized_raman_combine_kernel!,
                length(workspace.delayed), workspace.delayed, base.field_t,
                pair1, pair2, pair3, pair4, nt, nm)
        end
    else
        nc = nm * np
        flat = reshape(base.field_t, nt, nc)
        pair1 = _cuda_mmgnlse_dense_pair(workspace, 1)
        _cuda_mmgnlse_filter_pairs!(cache.raman.combined, workspace)
        _launch_1d(
            _cuda_mmgnlse_dense_raman_combine_kernel!,
            length(workspace.delayed), reshape(workspace.delayed, nt, nc),
            flat, pair1, nt, nc)
    end

    base.nonlinear .=
        (1 - fraction) .* base.nonlinear .+
        fraction .* workspace.delayed
    base.ifft_field! * base.nonlinear
    out .= cache.nonlinear_prefactor .* base.nonlinear
    return out
end

function _cuda_mmgnlse_dense_integrator_rhs!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::Union{
        CUDAMMGNLSEDenseForwardWorkspace,
        CUDAMMGNLSEDenseRamanForwardWorkspace,
    },
)
    return _cuda_mmgnlse_dense_nonlinear_spectral!(
        out, field_w, cache, workspace)
end

function _cuda_mmgnlse_dense_integrator_rhs!(
    out::CUDA.CuArray{ComplexF64,3},
    field_w::CUDA.CuArray{ComplexF64,3},
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEMixedDenseForwardWorkspace,
)
    rhs = workspace.rhs
    return _cuda_mmgnlse_dense_nonlinear_spectral!(
        out, field_w, cache, rhs)
end

function _cuda_mmgnlse_dense_propagators!(
    workspace,
    cache::CUDAMMGNLSEForwardCache,
    z0::Real,
    z1::Real,
)
    base = _cuda_mmgnlse_dense_base(workspace)
    step = Float64(z1 - z0)
    midpoint = (z0 + z1) / 2
    if cache.constant_passive
        if !base.propagator_valid || base.propagator_step != step
            _cuda_mmgnlse_linear_propagator!(
                base.to_midpoint, cache, z0, midpoint,
                base.passive_integral)
            base.from_midpoint .= base.to_midpoint
            base.propagator_step = step
            base.propagator_valid = true
        end
    else
        _cuda_mmgnlse_linear_propagator!(
            base.to_midpoint, cache, z0, midpoint,
            base.passive_integral)
        _cuda_mmgnlse_linear_propagator!(
            base.from_midpoint, cache, midpoint, z1,
            base.passive_integral)
    end
    return workspace
end

function _cuda_mmgnlse_dense_rk4ip_step!(
    out,
    field_w,
    state_cache::CUDAMMGNLSEForwardCache,
    rhs_cache::CUDAMMGNLSEForwardCache,
    workspace,
    z0::Real,
    z1::Real,
)
    state_workspace = workspace isa CUDAMMGNLSEMixedDenseForwardWorkspace ?
                      workspace.state : workspace
    base = _cuda_mmgnlse_dense_base(state_workspace)
    _cuda_mmgnlse_dense_propagators!(
        state_workspace, state_cache, z0, z1)
    if !state_cache.nonlinear_active
        out .= base.from_midpoint .* base.to_midpoint .* field_w
        return out
    end
    step = z1 - z0
    base.midpoint_base .= base.to_midpoint .* field_w
    _cuda_mmgnlse_dense_integrator_rhs!(
        base.k1, field_w, rhs_cache, workspace)
    base.k1 .*= base.to_midpoint
    base.stage .=
        base.midpoint_base .+ (step / 2) .* base.k1
    _cuda_mmgnlse_dense_integrator_rhs!(
        base.k2, base.stage, rhs_cache, workspace)
    base.stage .=
        base.midpoint_base .+ (step / 2) .* base.k2
    _cuda_mmgnlse_dense_integrator_rhs!(
        base.k3, base.stage, rhs_cache, workspace)
    base.stage .= base.from_midpoint .*
                  (base.midpoint_base .+ step .* base.k3)
    _cuda_mmgnlse_dense_integrator_rhs!(
        base.k4, base.stage, rhs_cache, workspace)
    out .= base.from_midpoint .* (
               base.midpoint_base .+
               (step / 6) .* (
                   base.k1 .+ 2 .* base.k2 .+
                   2 .* base.k3)) .+
           (step / 6) .* base.k4
    return out
end

function _cuda_mmgnlse_dense_rk4ip_step!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace,
    z0::Real,
    z1::Real,
)
    return _cuda_mmgnlse_dense_rk4ip_step!(
        out, field_w, cache, cache, workspace, z0, z1)
end

function _cuda_mmgnlse_cp_forward_workspace(
    field::CUDA.CuArray{Complex{T},3},
    overlap::CUDAMMGNLSECP,
) where {T<:AbstractFloat}
    CT = Complex{T}
    nt, nm, np = size(field)
    ncomponents = overlap.layout == :spatial ? np : 1
    component_width = overlap.layout == :spatial ? nm : nm * np
    field_t = similar(field)
    nonlinear = similar(field)
    instantaneous = similar(field)
    delayed = similar(field)
    next_field = similar(field)
    midpoint_base = similar(field)
    stage = similar(field)
    k1 = similar(field)
    k2 = similar(field)
    k3 = similar(field)
    k4 = similar(field)
    to_midpoint = similar(field)
    from_midpoint = similar(field)
    passive_integral = similar(field)
    projections = CUDA.zeros(
        CT, nt, 3 * overlap.rank, ncomponents)
    b2 = @view projections[:, 1:overlap.rank, :]
    b3 = @view projections[:, overlap.rank + 1:2 * overlap.rank, :]
    b4 = @view projections[:, 2 * overlap.rank + 1:3 * overlap.rank, :]
    conjugated = CUDA.zeros(CT, nt, component_width)
    stacked_rows = overlap.layout == :spatial ? nt * np : nt
    stacked_field = CUDA.zeros(CT, stacked_rows, component_width)
    stacked_projections = CUDA.zeros(CT, stacked_rows, 3 * overlap.rank)
    stacked_rank = CUDA.zeros(CT, stacked_rows, overlap.rank)
    raman_channels = CUDA.zeros(CT, nt, 4 * overlap.rank)
    rank1 = CUDA.zeros(CT, nt, overlap.rank)
    rank2 = similar(rank1)
    rank3 = similar(rank1)
    rank4 = similar(rank1)
    use_real_raman = _cuda_mmgnlse_cp_use_symmetric(overlap) &&
                     _cuda_mmgnlse_cp_use_real_gemm(overlap) &&
                     (overlap.layout != :spatial || np == 1)
    if use_real_raman
        # The scalar symmetric Raman path needs one real rank buffer and one
        # half-spectrum. Reuse rank3/rank4 storage, which that path does not
        # use, instead of increasing the persistent workspace footprint.
        real_rank_storage = reshape(
            reinterpret(T, rank3), nt, 2 * overlap.rank)
        real_rank = @view real_rank_storage[:, 1:overlap.rank]
        half_length = (nt ÷ 2 + 1) * overlap.rank
        half_rank = reshape(
            @view(reshape(rank4, :)[1:half_length]),
            nt ÷ 2 + 1, overlap.rank)
        rfft_rank! = plan_rfft(real_rank, 1)
        irfft_rank! = plan_irfft(half_rank, nt, 1)
    else
        real_rank = CUDA.zeros(T, 0, 0)
        half_rank = CUDA.zeros(CT, 0, 0)
        rfft_rank! = nothing
        irfft_rank! = nothing
    end
    rank3_channels = @view raman_channels[:, overlap.rank + 1:4 * overlap.rank]
    fft_field! = plan_fft!(field_t, 1)
    ifft_field! = plan_ifft!(nonlinear, 1)
    fft_rank! = plan_fft!(rank1, 1)
    ifft_rank! = plan_ifft!(rank1, 1)
    fft_rank3! = plan_fft!(rank3_channels, 1)
    ifft_rank3! = plan_ifft!(rank3_channels, 1)
    fft_rank4! = plan_fft!(raman_channels, 1)
    ifft_rank4! = plan_ifft!(raman_channels, 1)
    return CUDAMMGNLSECPForwardWorkspace(
        field_t, nonlinear, instantaneous, delayed, next_field,
        midpoint_base, stage, k1, k2, k3, k4, to_midpoint,
        from_midpoint, passive_integral, projections, b2, b3, b4, conjugated,
        stacked_field, stacked_projections, stacked_rank, raman_channels,
        rank1, rank2, rank3, rank4, real_rank, half_rank,
        fft_field!, ifft_field!, fft_rank!, ifft_rank!,
        fft_rank3!, ifft_rank3!, fft_rank4!, ifft_rank4!,
        rfft_rank!, irfft_rank!,
        NaN, false)
end

function _cuda_mmgnlse_cp_precision_workspace(
    initial_field::CUDA.CuArray{ComplexF64,3},
    state_cache::CUDAMMGNLSEForwardCache,
    rhs_cache::CUDAMMGNLSEForwardCache,
    precision::Symbol,
)
    if precision === :float64
        return _cuda_mmgnlse_cp_forward_workspace(
            initial_field, state_cache.overlap)
    end
    next_field = similar(initial_field)
    midpoint_base = similar(initial_field)
    stage = similar(initial_field)
    k1 = similar(initial_field)
    k2 = similar(initial_field)
    k3 = similar(initial_field)
    k4 = similar(initial_field)
    to_midpoint = similar(initial_field)
    from_midpoint = similar(initial_field)
    passive_integral = similar(initial_field)
    # k4 is dead between completed steps, so it also serves as the spectral
    # save scratch without adding another FP64 field-sized allocation.
    field_t = k4
    fft_field! = plan_fft!(field_t, 1)
    state = CUDAMMGNLSECPStateWorkspace(
        field_t, next_field, midpoint_base, stage, k1, k2, k3, k4,
        to_midpoint, from_midpoint, passive_integral, fft_field!, NaN, false)
    rhs_field = similar(initial_field, ComplexF32)
    rhs = _cuda_mmgnlse_cp_forward_workspace(
        rhs_field, rhs_cache.overlap)
    return CUDAMMGNLSEMixedCPForwardWorkspace(state, rhs)
end

_cuda_mmgnlse_cp_state_workspace(
    workspace::CUDAMMGNLSECPForwardWorkspace) = workspace
_cuda_mmgnlse_cp_state_workspace(
    workspace::CUDAMMGNLSEMixedCPForwardWorkspace) = workspace.state

function _cuda_mmgnlse_cp_pack_polarizations_kernel!(
    packed, field, nt::Int, nm::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        mode = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        packed_index = t + (polarization - 1) * nt +
                       (mode - 1) * nt * np
        @inbounds packed[packed_index] = field[index]
    end
    return nothing
end

function _cuda_mmgnlse_cp_unpack_projections_kernel!(
    projections, packed, nt::Int, rank::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        channel = remainder % (3 * rank) + 1
        polarization = remainder ÷ (3 * rank) + 1
        packed_index = t + (polarization - 1) * nt +
                       (channel - 1) * nt * np
        @inbounds value = packed[packed_index]
        @inbounds projections[index] =
            channel > 2 * rank ? conj(value) : value
    end
    return nothing
end

@inline _cuda_mmgnlse_cp_use_real_gemm(overlap::CUDAMMGNLSECP) =
    overlap.real_factors &&
    overlap.optimization === :symmetric_experimental

function _cuda_mmgnlse_cp_real_mul!(out, input, factors)
    T = typeof(real(zero(eltype(input))))
    input_real = reshape(
        reinterpret(T, input), 2 * size(input, 1), size(input, 2))
    out_real = reshape(
        reinterpret(T, out), 2 * size(out, 1), size(out, 2))
    mul!(out_real, input_real, factors)
    return out
end

function _cuda_mmgnlse_cp_backproject!(out, rank_values, overlap)
    if _cuda_mmgnlse_cp_use_real_gemm(overlap)
        return _cuda_mmgnlse_cp_real_mul!(
            out, rank_values, overlap.real_wu1t)
    end
    mul!(out, rank_values, overlap.wu1t)
    return out
end

function _cuda_mmgnlse_cp_project!(
    workspace::CUDAMMGNLSECPForwardWorkspace,
    field,
    overlap::CUDAMMGNLSECP,
)
    nt, nm, np = size(field)
    if _cuda_mmgnlse_cp_use_real_gemm(overlap) &&
       (overlap.layout != :spatial || np == 1)
        component = overlap.layout == :spatial ?
                    @view(field[:, :, 1]) : reshape(field, nt, nm * np)
        if overlap.symmetric_factors
            b2 = @view workspace.b2[:, :, 1]
            b3 = @view workspace.b3[:, :, 1]
            b4 = @view workspace.b4[:, :, 1]
            T = typeof(real(zero(eltype(component))))
            component_real = reshape(
                reinterpret(T, component), 2 * nt, size(component, 2))
            projections_real = reshape(
                reinterpret(T, workspace.projections),
                2 * nt, 3 * overlap.rank)
            b2_real = @view projections_real[:, 1:overlap.rank]
            mul!(b2_real, component_real, overlap.real_projection_factors)
        else
            projections = @view workspace.projections[:, :, 1]
            _cuda_mmgnlse_cp_real_mul!(
                projections, component, overlap.real_projection_factors)
            b4 = @view workspace.b4[:, :, 1]
            b4 .= conj.(b4)
        end
        return workspace
    end
    if _cuda_mmgnlse_cp_is_rank_tuned(overlap) &&
       overlap.rank <= _CUDA_MMGNLSE_CP_STACKED_PROJECTION_MAX_RANK
        if overlap.layout == :spatial
            if np == 1
                component = @view field[:, :, 1]
                projections = @view workspace.projections[:, :, 1]
                b4 = @view workspace.b4[:, :, 1]
                mul!(projections, component, overlap.projection_factors)
                b4 .= conj.(b4)
            else
                _launch_1d(
                    _cuda_mmgnlse_cp_pack_polarizations_kernel!,
                    length(field), workspace.stacked_field, field,
                    nt, nm, np)
                mul!(
                    workspace.stacked_projections,
                    workspace.stacked_field,
                    overlap.projection_factors,
                )
                _launch_1d(
                    _cuda_mmgnlse_cp_unpack_projections_kernel!,
                    length(workspace.projections), workspace.projections,
                    workspace.stacked_projections, nt, overlap.rank, np)
            end
        else
            component = reshape(field, nt, nm * np)
            mul!(
                @view(workspace.projections[:, :, 1]),
                component,
                overlap.projection_factors,
            )
            b4 = @view workspace.b4[:, :, 1]
            b4 .= conj.(b4)
        end
        return workspace
    end

    _, U2, U3, U4 = overlap.U
    if overlap.layout == :spatial
        @inbounds for polarization in 1:np
            component = @view field[:, :, polarization]
            mul!(@view(workspace.b2[:, :, polarization]), component, U2)
            mul!(@view(workspace.b3[:, :, polarization]), component, U3)
            workspace.conjugated .= conj.(component)
            mul!(
                @view(workspace.b4[:, :, polarization]),
                workspace.conjugated,
                U4,
            )
        end
    else
        component = reshape(field, nt, nm * np)
        mul!(@view(workspace.b2[:, :, 1]), component, U2)
        mul!(@view(workspace.b3[:, :, 1]), component, U3)
        workspace.conjugated .= conj.(component)
        mul!(@view(workspace.b4[:, :, 1]), workspace.conjugated, U4)
    end
    return workspace
end

function _cuda_mmgnlse_cp_pair_copy_kernel!(
    pair, filtered_pair, b3, b4, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        @inbounds value = b3[index] * b4[index]
        @inbounds pair[index] = value
        @inbounds filtered_pair[index] = value
    end
    return nothing
end

function _cuda_mmgnlse_cp_pair_copy!(pair, filtered_pair, b3, b4)
    _launch_1d(
        _cuda_mmgnlse_cp_pair_copy_kernel!,
        length(pair), pair, filtered_pair, b3, b4)
    return pair, filtered_pair
end

function _cuda_mmgnlse_cp_symmetric_real_pair_kernel!(
    pair, projected, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        @inbounds pair[index] = abs2(projected[index])
    end
    return nothing
end

function _cuda_mmgnlse_cp_symmetric_real_pair!(pair, projected)
    _launch_1d(
        _cuda_mmgnlse_cp_symmetric_real_pair_kernel!,
        length(pair), pair, projected)
    return pair
end

function _cuda_mmgnlse_filter_real_rank!(
    workspace::CUDAMMGNLSECPForwardWorkspace,
    kernel_half,
)
    mul!(workspace.half_rank, workspace.rfft_rank!, workspace.real_rank)
    workspace.half_rank .*= reshape(kernel_half, :, 1)
    mul!(workspace.real_rank, workspace.irfft_rank!, workspace.half_rank)
    return workspace.real_rank
end

function _cuda_mmgnlse_cp_polarized_pairs_kernel!(
    channels, b3x, b4x, b3y, b4y,
    ha_active::Bool, hb_active::Bool, channel_stride::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        @inbounds begin
            x3 = b3x[index]
            x4 = b4x[index]
            y3 = b3y[index]
            y4 = b4y[index]
            xx = x3 * x4
            yy = y3 * y4
            xy = x3 * y4
            yx = y3 * x4
            ha_active && (channels[index] = xx + yy)
            if hb_active
                channels[index + channel_stride] = xx
                channels[index + 2 * channel_stride] = yy
                channels[index + 3 * channel_stride] = xy + yx
            end
            # Preserve the four pair products in projection storage after
            # the projections themselves have had their final use.
            b3x[index] = xx
            b3y[index] = yy
            b4x[index] = xy
            b4y[index] = yx
        end
    end
    return nothing
end

function _cuda_mmgnlse_cp_rank_scale_kernel!(
    channels, kernel, nt::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        @inbounds channels[index] *= kernel[t]
    end
    return nothing
end

function _cuda_mmgnlse_cp_rank4_scale_kernel!(
    channels, ha, hb, nt::Int, rank::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        channel = (index - 1) ÷ (nt * rank)
        @inbounds channels[index] *= channel == 0 ? ha[t] : hb[t]
    end
    return nothing
end

function _cuda_mmgnlse_filter_polarized_rank!(workspace, raman)
    nt = size(workspace.raman_channels, 1)
    rank = size(workspace.rank1, 2)
    if raman.ha_active && raman.hb_active
        workspace.ifft_rank4! * workspace.raman_channels
        _launch_1d(
            _cuda_mmgnlse_cp_rank4_scale_kernel!,
            length(workspace.raman_channels), workspace.raman_channels,
            raman.ha, raman.hb, nt, rank)
        workspace.fft_rank4! * workspace.raman_channels
    elseif raman.ha_active
        ha_channel = @view workspace.raman_channels[:, 1:rank]
        workspace.ifft_rank! * ha_channel
        _launch_1d(
            _cuda_mmgnlse_cp_rank_scale_kernel!,
            length(ha_channel), ha_channel, raman.ha, nt)
        workspace.fft_rank! * ha_channel
    elseif raman.hb_active
        hb_channels = @view workspace.raman_channels[:, rank + 1:4 * rank]
        workspace.ifft_rank3! * hb_channels
        _launch_1d(
            _cuda_mmgnlse_cp_rank_scale_kernel!,
            length(hb_channels), hb_channels, raman.hb, nt)
        workspace.fft_rank3! * hb_channels
    end
    return workspace
end

function _cuda_mmgnlse_cp_polarized_mix_kernel!(
    output_x, output_y, b2x, pair_xx, pair_xy,
    b2y, pair_yy, pair_yx, qa, qxx, qyy, qxy,
    instantaneous, fraction, ha_active::Bool, hb_active::Bool,
    nt::Int, pairs_ready::Bool, stacked::Bool, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        @inbounds begin
            if pairs_ready
                xx = pair_xx[index]
                yy = pair_yy[index]
                xy = pair_xy[index]
                yx = pair_yx[index]
            else
                x3 = pair_xx[index]
                x4 = pair_xy[index]
                y3 = pair_yy[index]
                y4 = pair_yx[index]
                xx = x3 * x4
                yy = y3 * y4
                xy = x3 * y4
                yx = y3 * x4
            end
            one_value = one(xx)
            instant_x = zero(xx)
            instant_y = zero(xx)
            if !iszero(instantaneous)
                instant_x = b2x[index] * xx +
                            (2 * one_value / 3) * b2x[index] * yy +
                            (one_value / 3) * b2y[index] * yx
                instant_y = b2y[index] * yy +
                            (2 * one_value / 3) * b2y[index] * xx +
                            (one_value / 3) * b2x[index] * xy
            end
            delayed_x = zero(xx)
            delayed_y = zero(xx)
            if ha_active
                delayed_x += b2x[index] * qa[index]
                delayed_y += b2y[index] * qa[index]
            end
            if hb_active
                delayed_x += b2x[index] * qxx[index] +
                             (one_value / 2) * b2y[index] * qxy[index]
                delayed_y += b2y[index] * qyy[index] +
                             (one_value / 2) * b2x[index] * qxy[index]
            end
            value_x = instantaneous * instant_x + fraction * delayed_x
            value_y = instantaneous * instant_y + fraction * delayed_y
            if stacked
                t = (index - 1) % nt + 1
                rank_index = (index - 1) ÷ nt
                base = t + rank_index * 2 * nt
                output_x[base] = value_x
                output_x[base + nt] = value_y
            else
                output_x[index] = value_x
                output_y[index] = value_y
            end
        end
    end
    return nothing
end

function _cuda_mmgnlse_cp_unpack_polarizations_kernel!(
    field, packed, nt::Int, nm::Int, np::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        mode = remainder % nm + 1
        polarization = remainder ÷ nm + 1
        packed_index = t + (polarization - 1) * nt +
                       (mode - 1) * nt * np
        @inbounds field[index] = packed[packed_index]
    end
    return nothing
end

function _cuda_mmgnlse_cp_backproject_polarized!(
    out, workspace, overlap,
)
    nt, nm, np = size(out)
    _cuda_mmgnlse_cp_backproject!(
        workspace.stacked_field, workspace.stacked_rank, overlap)
    _launch_1d(
        _cuda_mmgnlse_cp_unpack_polarizations_kernel!,
        length(out), out, workspace.stacked_field, nt, nm, np)
    return out
end

function _cuda_mmgnlse_cp_instantaneous!(
    out,
    workspace::CUDAMMGNLSECPForwardWorkspace,
    overlap::CUDAMMGNLSECP,
)
    nt, nm, np = size(out)
    if overlap.layout == :component
        b2 = @view workspace.b2[:, :, 1]
        b3 = @view workspace.b3[:, :, 1]
        b4 = @view workspace.b4[:, :, 1]
        workspace.rank1 .= b2 .* b3 .* b4
        _cuda_mmgnlse_cp_backproject!(
            reshape(out, nt, nm * np), workspace.rank1, overlap)
        return out
    end

    b2x = @view workspace.b2[:, :, 1]
    b3x = @view workspace.b3[:, :, 1]
    b4x = @view workspace.b4[:, :, 1]
    if np == 1
        workspace.rank1 .= b2x .* b3x .* b4x
        _cuda_mmgnlse_cp_backproject!(
            @view(out[:, :, 1]), workspace.rank1, overlap)
        return out
    end

    b2y = @view workspace.b2[:, :, 2]
    b3y = @view workspace.b3[:, :, 2]
    b4y = @view workspace.b4[:, :, 2]
    workspace.rank1 .= b2x .* b3x .* b4x .+
                       (2 / 3) .* b2x .* b3y .* b4y .+
                       (1 / 3) .* b2y .* b3y .* b4x
    _cuda_mmgnlse_cp_backproject!(
        @view(out[:, :, 1]), workspace.rank1, overlap)
    workspace.rank1 .= b2y .* b3y .* b4y .+
                       (2 / 3) .* b2y .* b3x .* b4x .+
                       (1 / 3) .* b2x .* b3x .* b4y
    _cuda_mmgnlse_cp_backproject!(
        @view(out[:, :, 2]), workspace.rank1, overlap)
    return out
end

function _cuda_mmgnlse_filter_rank!(
    channels,
    kernel,
    workspace::CUDAMMGNLSECPForwardWorkspace,
)
    workspace.ifft_rank! * channels
    channels .*= reshape(kernel, :, 1)
    workspace.fft_rank! * channels
    return channels
end

function _cuda_mmgnlse_cp_delayed!(
    out,
    workspace::CUDAMMGNLSECPForwardWorkspace,
    overlap::CUDAMMGNLSECP,
    raman,
)
    nt, nm, np = size(out)
    if overlap.layout == :component
        b2 = @view workspace.b2[:, :, 1]
        b3 = @view workspace.b3[:, :, 1]
        b4 = @view workspace.b4[:, :, 1]
        workspace.rank1 .= b3 .* b4
        _cuda_mmgnlse_filter_rank!(
            workspace.rank1, raman.combined, workspace)
        workspace.rank1 .*= b2
        _cuda_mmgnlse_cp_backproject!(
            reshape(out, nt, nm * np), workspace.rank1, overlap)
        return out
    end

    b2x = @view workspace.b2[:, :, 1]
    b3x = @view workspace.b3[:, :, 1]
    b4x = @view workspace.b4[:, :, 1]
    if np == 1
        workspace.rank1 .= b3x .* b4x
        _cuda_mmgnlse_filter_rank!(
            workspace.rank1, raman.combined, workspace)
        workspace.rank1 .*= b2x
        _cuda_mmgnlse_cp_backproject!(
            @view(out[:, :, 1]), workspace.rank1, overlap)
        return out
    end

    b2y = @view workspace.b2[:, :, 2]
    b3y = @view workspace.b3[:, :, 2]
    b4y = @view workspace.b4[:, :, 2]

    if !(raman.ha_active || raman.hb_active)
        fill!(out, zero(eltype(out)))
        return out
    end
    channel_stride = length(workspace.rank1)
    _launch_1d(
        _cuda_mmgnlse_cp_polarized_pairs_kernel!,
        channel_stride, workspace.raman_channels,
        b3x, b4x, b3y, b4y,
        raman.ha_active, raman.hb_active, channel_stride)
    _cuda_mmgnlse_filter_polarized_rank!(workspace, raman)
    rank = overlap.rank
    qa = @view workspace.raman_channels[:, 1:rank]
    qxx = @view workspace.raman_channels[:, rank + 1:2 * rank]
    qyy = @view workspace.raman_channels[:, 2 * rank + 1:3 * rank]
    qxy = @view workspace.raman_channels[:, 3 * rank + 1:4 * rank]
    _launch_1d(
        _cuda_mmgnlse_cp_polarized_mix_kernel!,
        channel_stride, workspace.rank1, workspace.rank2,
        b2x, b3x, b4x, b2y, b3y, b4y,
        qa, qxx, qyy, qxy,
        zero(raman.fraction), one(raman.fraction),
        raman.ha_active, raman.hb_active, nt, true, false)
    _cuda_mmgnlse_cp_backproject!(
        @view(out[:, :, 1]), workspace.rank1, overlap)
    _cuda_mmgnlse_cp_backproject!(
        @view(out[:, :, 2]), workspace.rank2, overlap)
    return out
end

function _cuda_mmgnlse_cp_combined!(
    out,
    workspace::CUDAMMGNLSECPForwardWorkspace,
    overlap::CUDAMMGNLSECP,
    raman,
)
    nt, nm, np = size(out)
    fraction = raman.fraction
    instantaneous = 1 - fraction

    if overlap.layout == :component
        b2 = @view workspace.b2[:, :, 1]
        b3 = @view workspace.b3[:, :, 1]
        b4 = @view workspace.b4[:, :, 1]
        if iszero(fraction)
            if overlap.symmetric_factors &&
               _cuda_mmgnlse_cp_use_real_gemm(overlap)
                workspace.rank1 .= b2 .* abs2.(b2)
            else
                workspace.rank1 .= b2 .* b3 .* b4
            end
        else
            if overlap.symmetric_factors &&
               _cuda_mmgnlse_cp_use_real_gemm(overlap)
                _cuda_mmgnlse_cp_symmetric_real_pair!(
                    workspace.real_rank, b2)
                _cuda_mmgnlse_filter_real_rank!(
                    workspace, raman.combined_r2c)
                workspace.rank1 .= b2 .* (
                    instantaneous .* abs2.(b2) .+
                    fraction .* workspace.real_rank)
            else
                _cuda_mmgnlse_cp_pair_copy!(
                    workspace.rank1, workspace.rank2, b3, b4)
                _cuda_mmgnlse_filter_rank!(
                    workspace.rank2, raman.combined, workspace)
                workspace.rank1 .= b2 .* (
                    instantaneous .* workspace.rank1 .+
                    fraction .* workspace.rank2)
            end
        end
        _cuda_mmgnlse_cp_backproject!(
            reshape(out, nt, nm * np), workspace.rank1, overlap)
        return out
    end

    b2x = @view workspace.b2[:, :, 1]
    b3x = @view workspace.b3[:, :, 1]
    b4x = @view workspace.b4[:, :, 1]
    if np == 1
        if iszero(fraction)
            if overlap.symmetric_factors &&
               _cuda_mmgnlse_cp_use_real_gemm(overlap)
                workspace.rank1 .= b2x .* abs2.(b2x)
            else
                workspace.rank1 .= b2x .* b3x .* b4x
            end
        else
            if overlap.symmetric_factors &&
               _cuda_mmgnlse_cp_use_real_gemm(overlap)
                _cuda_mmgnlse_cp_symmetric_real_pair!(
                    workspace.real_rank, b2x)
                _cuda_mmgnlse_filter_real_rank!(
                    workspace, raman.combined_r2c)
                workspace.rank1 .= b2x .* (
                    instantaneous .* abs2.(b2x) .+
                    fraction .* workspace.real_rank)
            else
                _cuda_mmgnlse_cp_pair_copy!(
                    workspace.rank1, workspace.rank2, b3x, b4x)
                _cuda_mmgnlse_filter_rank!(
                    workspace.rank2, raman.combined, workspace)
                workspace.rank1 .= b2x .* (
                    instantaneous .* workspace.rank1 .+
                    fraction .* workspace.rank2)
            end
        end
        _cuda_mmgnlse_cp_backproject!(
            @view(out[:, :, 1]), workspace.rank1, overlap)
        return out
    end

    b2y = @view workspace.b2[:, :, 2]
    b3y = @view workspace.b3[:, :, 2]
    b4y = @view workspace.b4[:, :, 2]
    if iszero(fraction)
        if _cuda_mmgnlse_cp_is_rank_tuned(overlap)
            _launch_1d(
                _cuda_mmgnlse_cp_polarized_mix_kernel!,
                length(workspace.rank1),
                workspace.stacked_rank, workspace.stacked_rank,
                b2x, b3x, b4x, b2y, b3y, b4y,
                workspace.rank1, workspace.rank2,
                workspace.rank3, workspace.rank4,
                one(fraction), zero(fraction), false, false,
                nt, false, true)
            _cuda_mmgnlse_cp_backproject_polarized!(
                out, workspace, overlap)
        else
            workspace.rank2 .=
                b2x .* b3x .* b4x .+
                (2 / 3) .* b2x .* b3y .* b4y .+
                (1 / 3) .* b2y .* b3y .* b4x
            _cuda_mmgnlse_cp_backproject!(
                @view(out[:, :, 1]), workspace.rank2, overlap)
            workspace.rank2 .=
                b2y .* b3y .* b4y .+
                (2 / 3) .* b2y .* b3x .* b4x .+
                (1 / 3) .* b2x .* b3x .* b4y
            _cuda_mmgnlse_cp_backproject!(
                @view(out[:, :, 2]), workspace.rank2, overlap)
        end
        return out
    end

    channels_active = raman.ha_active || raman.hb_active
    channel_stride = length(workspace.rank1)
    if channels_active
        _launch_1d(
            _cuda_mmgnlse_cp_polarized_pairs_kernel!,
            channel_stride, workspace.raman_channels,
            b3x, b4x, b3y, b4y,
            raman.ha_active, raman.hb_active, channel_stride)
        _cuda_mmgnlse_filter_polarized_rank!(workspace, raman)
    end
    stacked = _cuda_mmgnlse_cp_is_rank_tuned(overlap)
    output_x = stacked ? workspace.stacked_rank : workspace.rank1
    output_y = stacked ? workspace.stacked_rank : workspace.rank2
    rank = overlap.rank
    qa = @view workspace.raman_channels[:, 1:rank]
    qxx = @view workspace.raman_channels[:, rank + 1:2 * rank]
    qyy = @view workspace.raman_channels[:, 2 * rank + 1:3 * rank]
    qxy = @view workspace.raman_channels[:, 3 * rank + 1:4 * rank]
    _launch_1d(
        _cuda_mmgnlse_cp_polarized_mix_kernel!,
        channel_stride, output_x, output_y,
        b2x, b3x, b4x, b2y, b3y, b4y,
        qa, qxx, qyy, qxy,
        instantaneous, fraction, raman.ha_active, raman.hb_active,
        nt, channels_active, stacked)
    if stacked
        _cuda_mmgnlse_cp_backproject_polarized!(out, workspace, overlap)
    else
        _cuda_mmgnlse_cp_backproject!(
            @view(out[:, :, 1]), workspace.rank1, overlap)
        _cuda_mmgnlse_cp_backproject!(
            @view(out[:, :, 2]), workspace.rank2, overlap)
    end
    return out
end

function _cuda_mmgnlse_cp_nonlinear_spectral!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPForwardWorkspace,
)
    if !cache.nonlinear_active
        fill!(out, zero(eltype(out)))
        return out
    end
    workspace.field_t .= field_w
    workspace.fft_field! * workspace.field_t
    _cuda_mmgnlse_cp_project!(
        workspace, workspace.field_t, cache.overlap)
    if _cuda_mmgnlse_cp_is_optimized(cache.overlap)
        _cuda_mmgnlse_cp_combined!(
            workspace.nonlinear, workspace, cache.overlap, cache.raman)
    else
        _cuda_mmgnlse_cp_instantaneous!(
            workspace.instantaneous, workspace, cache.overlap)
        fraction = cache.raman.fraction
        if iszero(fraction)
            workspace.nonlinear .= workspace.instantaneous
        else
            _cuda_mmgnlse_cp_delayed!(
                workspace.delayed, workspace, cache.overlap, cache.raman)
            workspace.nonlinear .=
                (1 - fraction) .* workspace.instantaneous .+
                fraction .* workspace.delayed
        end
    end
    workspace.ifft_field! * workspace.nonlinear
    out .= cache.nonlinear_prefactor .* workspace.nonlinear
    return out
end

function _cuda_mmgnlse_cp_integrator_rhs!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPForwardWorkspace,
)
    return _cuda_mmgnlse_cp_nonlinear_spectral!(
        out, field_w, cache, workspace)
end

function _cuda_mmgnlse_cp_integrator_rhs!(
    out::CUDA.CuArray{ComplexF64,3},
    field_w::CUDA.CuArray{ComplexF64,3},
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEMixedCPForwardWorkspace,
)
    rhs = workspace.rhs
    rhs.next_field .= field_w
    _cuda_mmgnlse_cp_nonlinear_spectral!(
        rhs.k1, rhs.next_field, cache, rhs)
    out .= rhs.k1
    return out
end

function _cuda_mmgnlse_cp_propagators!(
    workspace::Union{
        CUDAMMGNLSECPForwardWorkspace,
        CUDAMMGNLSECPStateWorkspace,
    },
    cache::CUDAMMGNLSEForwardCache,
    z0::Real,
    z1::Real,
)
    step = Float64(z1 - z0)
    midpoint = (z0 + z1) / 2
    if cache.constant_passive
        if !workspace.propagator_valid || workspace.propagator_step != step
            _cuda_mmgnlse_linear_propagator!(
                workspace.to_midpoint, cache, z0, midpoint,
                workspace.passive_integral)
            workspace.from_midpoint .= workspace.to_midpoint
            workspace.propagator_step = step
            workspace.propagator_valid = true
        end
    else
        _cuda_mmgnlse_linear_propagator!(
            workspace.to_midpoint, cache, z0, midpoint,
            workspace.passive_integral)
        _cuda_mmgnlse_linear_propagator!(
            workspace.from_midpoint, cache, midpoint, z1,
            workspace.passive_integral)
    end
    return workspace
end

_cuda_mmgnlse_forward_state_workspace(
    workspace::CUDAMMGNLSECPForwardWorkspace) = workspace
_cuda_mmgnlse_forward_state_workspace(
    workspace::CUDAMMGNLSEMixedCPForwardWorkspace) = workspace.state
_cuda_mmgnlse_forward_state_workspace(
    workspace::Union{
        CUDAMMGNLSEDenseForwardWorkspace,
        CUDAMMGNLSEDenseRamanForwardWorkspace,
        CUDAMMGNLSEMixedDenseForwardWorkspace,
    }) = _cuda_mmgnlse_dense_state_workspace(workspace)

function _cuda_mmgnlse_forward_propagators!(
    workspace::Union{
        CUDAMMGNLSECPForwardWorkspace,
        CUDAMMGNLSEMixedCPForwardWorkspace,
    },
    cache::CUDAMMGNLSEForwardCache,
    z0::Real,
    z1::Real,
)
    return _cuda_mmgnlse_cp_propagators!(
        _cuda_mmgnlse_cp_state_workspace(workspace), cache, z0, z1)
end

function _cuda_mmgnlse_forward_propagators!(
    workspace::Union{
        CUDAMMGNLSEDenseForwardWorkspace,
        CUDAMMGNLSEDenseRamanForwardWorkspace,
        CUDAMMGNLSEMixedDenseForwardWorkspace,
    },
    cache::CUDAMMGNLSEForwardCache,
    z0::Real,
    z1::Real,
)
    state_workspace = workspace isa CUDAMMGNLSEMixedDenseForwardWorkspace ?
                      workspace.state : workspace
    return _cuda_mmgnlse_dense_propagators!(
        state_workspace, cache, z0, z1)
end

function _cuda_mmgnlse_forward_integrator_rhs!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::Union{
        CUDAMMGNLSECPForwardWorkspace,
        CUDAMMGNLSEMixedCPForwardWorkspace,
    },
)
    return _cuda_mmgnlse_cp_integrator_rhs!(
        out, field_w, cache, workspace)
end

function _cuda_mmgnlse_forward_integrator_rhs!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::Union{
        CUDAMMGNLSEDenseForwardWorkspace,
        CUDAMMGNLSEDenseRamanForwardWorkspace,
        CUDAMMGNLSEMixedDenseForwardWorkspace,
    },
)
    return _cuda_mmgnlse_dense_integrator_rhs!(
        out, field_w, cache, workspace)
end

function _cuda_mmgnlse_cp_rk4ip_step!(
    out,
    field_w,
    state_cache::CUDAMMGNLSEForwardCache,
    rhs_cache::CUDAMMGNLSEForwardCache,
    workspace,
    z0::Real,
    z1::Real,
)
    state = _cuda_mmgnlse_cp_state_workspace(workspace)
    _cuda_mmgnlse_cp_propagators!(state, state_cache, z0, z1)
    if !state_cache.nonlinear_active
        out .= state.from_midpoint .* state.to_midpoint .* field_w
        return out
    end
    step = z1 - z0
    state.midpoint_base .= state.to_midpoint .* field_w
    _cuda_mmgnlse_cp_integrator_rhs!(
        state.k1, field_w, rhs_cache, workspace)
    state.k1 .*= state.to_midpoint
    state.stage .=
        state.midpoint_base .+ (step / 2) .* state.k1
    _cuda_mmgnlse_cp_integrator_rhs!(
        state.k2, state.stage, rhs_cache, workspace)
    state.stage .=
        state.midpoint_base .+ (step / 2) .* state.k2
    _cuda_mmgnlse_cp_integrator_rhs!(
        state.k3, state.stage, rhs_cache, workspace)
    state.stage .= state.from_midpoint .*
                   (state.midpoint_base .+ step .* state.k3)
    _cuda_mmgnlse_cp_integrator_rhs!(
        state.k4, state.stage, rhs_cache, workspace)
    out .= state.from_midpoint .* (
               state.midpoint_base .+
               (step / 6) .* (
                   state.k1 .+ 2 .* state.k2 .+
                   2 .* state.k3)) .+
           (step / 6) .* state.k4
    return out
end

function _cuda_mmgnlse_cp_rk4ip_step!(
    out,
    field_w,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPForwardWorkspace,
    z0::Real,
    z1::Real,
)
    return _cuda_mmgnlse_cp_rk4ip_step!(
        out, field_w, cache, cache, workspace, z0, z1)
end

function _cuda_mmgnlse_adaptive_workspace(workspace, initial_field)
    return CUDAMMGNLSEAdaptiveWorkspace(
        workspace, similar(initial_field), similar(initial_field),
        similar(initial_field))
end

"""
Form one classical RK4IP candidate and its embedded third-order estimate.

The extra endpoint evaluation supplies the fifth-stage derivative for the
embedded weights `(1/6, 1/3, 1/3, 0, 1/6)`. The local error is therefore
`h/6 * (k4 - k5)`. On acceptance, `k5` is the raw nonlinear derivative at
the next step's start and is reused FSAL-style, so the steady accepted-step
cost remains four nonlinear right-hand-side evaluations.
"""
function _cuda_mmgnlse_rk4ip_adaptive_trial!(
    out,
    field_w,
    state_cache::CUDAMMGNLSEForwardCache,
    rhs_cache::CUDAMMGNLSEForwardCache,
    adaptive::CUDAMMGNLSEAdaptiveWorkspace,
    z0::Real,
    z1::Real,
    reltol::Float64,
    abstol::Float64,
)
    workspace = adaptive.integrator
    state = _cuda_mmgnlse_forward_state_workspace(workspace)
    _cuda_mmgnlse_forward_propagators!(
        workspace, state_cache, z0, z1)
    if !state_cache.nonlinear_active
        out .= state.from_midpoint .* state.to_midpoint .* field_w
        fill!(adaptive.endpoint_rhs, zero(eltype(adaptive.endpoint_rhs)))
        fill!(adaptive.error, zero(eltype(adaptive.error)))
        return 0.0
    end

    step = Float64(z1 - z0)
    state.midpoint_base .= state.to_midpoint .* field_w
    state.k1 .= state.to_midpoint .* adaptive.fsal
    state.stage .=
        state.midpoint_base .+ (step / 2) .* state.k1
    _cuda_mmgnlse_forward_integrator_rhs!(
        state.k2, state.stage, rhs_cache, workspace)
    state.stage .=
        state.midpoint_base .+ (step / 2) .* state.k2
    _cuda_mmgnlse_forward_integrator_rhs!(
        state.k3, state.stage, rhs_cache, workspace)
    state.stage .= state.from_midpoint .*
                   (state.midpoint_base .+ step .* state.k3)
    _cuda_mmgnlse_forward_integrator_rhs!(
        state.k4, state.stage, rhs_cache, workspace)
    out .= state.from_midpoint .* (
               state.midpoint_base .+
               (step / 6) .* (
                   state.k1 .+ 2 .* state.k2 .+
                   2 .* state.k3)) .+
           (step / 6) .* state.k4
    _cuda_mmgnlse_forward_integrator_rhs!(
        adaptive.endpoint_rhs, out, rhs_cache, workspace)

    adaptive.error .= (step / 6) .* (
        state.k4 .- adaptive.endpoint_rhs)
    adaptive.error .= adaptive.error ./ (
        abstol .+ reltol .* max.(abs.(field_w), abs.(out)))
    return Float64(norm(adaptive.error)) / sqrt(length(adaptive.error))
end

@inline function _cuda_mmgnlse_adaptive_factor(error_norm::Float64,
                                                accepted::Bool)
    iszero(error_norm) && return accepted ? 5.0 : 0.2
    isfinite(error_norm) || return 0.2
    upper = accepted ? 5.0 : 0.9
    return clamp(0.9 * error_norm^(-0.25), 0.2, upper)
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
    cp_optimization::Symbol, precision::Symbol,
)
    dense_state_only = precision === :mixed &&
                       !(parameters.S isa
                         PulsePropagation.MMGNLSECPDecomposition)
    gpu_cache = _cuda_mmgnlse_forward_cache(
        parameters, cp_optimization;
        dense_contraction=dense_state_only ? :none : nothing)
    rhs_cache = precision === :mixed ?
                _cuda_mmgnlse_forward_cache(
                    parameters, cp_optimization, Float32) : gpu_cache
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
    workspace = if gpu_cache.overlap isa CUDAMMGNLSECP
        _cuda_mmgnlse_cp_precision_workspace(
            initial_field, gpu_cache, rhs_cache, precision)
    else
        _cuda_mmgnlse_dense_precision_workspace(
            initial_field, gpu_cache, rhs_cache, precision)
    end
    if workspace === nothing
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
    else
        for index in 1:length(steps)-1
            z0, z1 = steps[index], steps[index + 1]
            if workspace isa Union{
                CUDAMMGNLSECPForwardWorkspace,
                CUDAMMGNLSEMixedCPForwardWorkspace,
            }
                state = _cuda_mmgnlse_cp_state_workspace(workspace)
                _cuda_mmgnlse_cp_rk4ip_step!(
                    state.next_field, field_w,
                    gpu_cache, rhs_cache, workspace, z0, z1)
            else
                dense_base = _cuda_mmgnlse_dense_base(workspace)
                _cuda_mmgnlse_dense_rk4ip_step!(
                    dense_base.next_field, field_w,
                    gpu_cache, rhs_cache, workspace, z0, z1)
            end
            if workspace isa Union{
                CUDAMMGNLSECPForwardWorkspace,
                CUDAMMGNLSEMixedCPForwardWorkspace,
            }
                state = _cuda_mmgnlse_cp_state_workspace(workspace)
                field_w, state.next_field = state.next_field, field_w
            else
                dense_base = _cuda_mmgnlse_dense_base(workspace)
                field_w, dense_base.next_field =
                    dense_base.next_field, field_w
            end
            if save_index <= length(saved) &&
               abs(z1 - saved[save_index]) <= tolerance
                save_workspace =
                    _cuda_mmgnlse_forward_state_workspace(workspace)
                save_workspace.field_t .= field_w
                save_workspace.fft_field! * save_workspace.field_t
                @views fields[:, :, :, save_index] .=
                    save_workspace.field_t
                save_index += 1
            end
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
               cuda=gpu_cache, rhs_cuda=rhs_cache,
               workspace=workspace, precision, adaptive=false,
               accepted_steps=length(steps) - 1,
               rejected_steps=0,
               rhs_evaluations=gpu_cache.nonlinear_active ?
                   4 * (length(steps) - 1) : 0))
end

function _cuda_mmgnlse_solve_rk4ip_adaptive(
    initial_field,
    parameters,
    dz::Float64,
    saveat,
    cp_optimization::Symbol,
    precision::Symbol;
    reltol::Float64,
    abstol::Float64,
    dzmin::Float64,
    dzmax::Float64,
    maxiters::Int,
)
    saveat === :steps && throw(ArgumentError(
        "saveat=:steps is not supported with adaptive RK4IP; pass an " *
        "explicit spacing or vector of save coordinates."))
    dense_state_only = precision === :mixed &&
                       !(parameters.S isa
                         PulsePropagation.MMGNLSECPDecomposition)
    gpu_cache = _cuda_mmgnlse_forward_cache(
        parameters, cp_optimization, Float64;
        dense_contraction=dense_state_only ? :none : nothing)
    rhs_cache = precision === :mixed ?
                _cuda_mmgnlse_forward_cache(
                    parameters, cp_optimization, Float32) : gpu_cache
    targets, _ = PulsePropagation._mmgnlse_save_targets(
        Float64(parameters.length), saveat)
    nt, nm, np = size(initial_field)
    fields = CUDA.zeros(ComplexF64, nt, nm, np, length(targets))
    @views fields[:, :, :, 1] .= initial_field
    field_w = ifft(initial_field, 1)

    integrator = if gpu_cache.overlap isa CUDAMMGNLSECP
        _cuda_mmgnlse_cp_precision_workspace(
            initial_field, gpu_cache, rhs_cache, precision)
    else
        _cuda_mmgnlse_dense_precision_workspace(
            initial_field, gpu_cache, rhs_cache, precision)
    end
    state = _cuda_mmgnlse_forward_state_workspace(integrator)
    adaptive_workspace = _cuda_mmgnlse_adaptive_workspace(
        integrator, initial_field)
    rhs_evaluations = 0
    if gpu_cache.nonlinear_active
        _cuda_mmgnlse_forward_integrator_rhs!(
            adaptive_workspace.fsal, field_w,
            rhs_cache, integrator)
        rhs_evaluations = 1
    else
        fill!(adaptive_workspace.fsal,
              zero(eltype(adaptive_workspace.fsal)))
    end

    length_m = Float64(parameters.length)
    tolerance = 32eps(Float64) * max(1.0, length_m)
    minimum_floor = max(dzmin, 32eps(Float64) * max(1.0, length_m))
    step_size = min(dz, dzmax, length_m)
    integration_z = Float64[0.0]
    z = 0.0
    target_index = 2
    attempts = 0
    accepted_steps = 0
    rejected_steps = 0
    accepted_step_sum = 0.0
    accepted_step_min = Inf
    accepted_step_max = 0.0
    last_error = 0.0
    max_error = 0.0

    while z < length_m - tolerance
        attempts += 1
        attempts <= maxiters || error(
            "Adaptive RK4IP exceeded maxiters=$maxiters at z=$z m.")
        next_target = targets[target_index]
        remaining = min(next_target, length_m) - z
        trial_step = min(step_size, remaining)
        trial_step > 0 || error(
            "Adaptive RK4IP could not construct a positive step at z=$z m.")
        z1 = z + trial_step
        abs(z1 - next_target) <= tolerance && (z1 = next_target)
        abs(z1 - length_m) <= tolerance && (z1 = length_m)
        trial_step = z1 - z

        error_norm = _cuda_mmgnlse_rk4ip_adaptive_trial!(
            state.next_field, field_w, gpu_cache, rhs_cache,
            adaptive_workspace, z, z1, reltol, abstol)
        gpu_cache.nonlinear_active && (rhs_evaluations += 4)
        last_error = error_norm
        isfinite(error_norm) && (max_error = max(max_error, error_norm))
        accepted = isfinite(error_norm) && error_norm <= 1.0

        if accepted
            field_w, state.next_field = state.next_field, field_w
            adaptive_workspace.fsal, adaptive_workspace.endpoint_rhs =
                adaptive_workspace.endpoint_rhs, adaptive_workspace.fsal
            z = z1
            push!(integration_z, z)
            accepted_steps += 1
            accepted_step_sum += trial_step
            accepted_step_min = min(accepted_step_min, trial_step)
            accepted_step_max = max(accepted_step_max, trial_step)

            if abs(z - next_target) <= tolerance
                z = next_target
                state.field_t .= field_w
                state.fft_field! * state.field_t
                @views fields[:, :, :, target_index] .= state.field_t
                target_index += 1
            end
            step_size = min(
                dzmax,
                trial_step * _cuda_mmgnlse_adaptive_factor(
                    error_norm, true),
            )
        else
            rejected_steps += 1
            trial_step > minimum_floor || error(
                "Adaptive RK4IP could not satisfy reltol=$reltol and " *
                "abstol=$abstol at z=$z m before reaching dzmin=$dzmin.")
            step_size = max(
                minimum_floor,
                trial_step * _cuda_mmgnlse_adaptive_factor(
                    error_norm, false),
            )
        end
    end

    target_index == length(targets) + 1 || error(
        "Not every requested adaptive save plane was reached.")
    cpu_core = PulsePropagation._mmgnlse_solver_cache(parameters)
    mean_step = accepted_steps == 0 ? 0.0 :
                accepted_step_sum / accepted_steps
    min_step = accepted_steps == 0 ? 0.0 : accepted_step_min
    return PulsePropagation.MMGNLSESolution(
        z=Float64.(targets), fields=fields, parameters=parameters,
        initial_field=copy(initial_field), dz=dz,
        method=PulsePropagation.RK4IP(),
        cache=(integration_z, core=cpu_core, cuda=gpu_cache,
               rhs_cuda=rhs_cache, workspace=adaptive_workspace,
               precision, adaptive=true, accepted_steps, rejected_steps,
               rhs_evaluations, min_step, max_step=accepted_step_max,
               mean_step, last_error, max_error, reltol, abstol,
               dzmin, dzmax))
end

function _cuda_mmgnlse_solve_tsit5(
    initial_field, parameters, dz::Float64, saveat, method,
    cp_optimization::Symbol,
)
    gpu_cache = _cuda_mmgnlse_forward_cache(parameters, cp_optimization)
    targets, every_step = PulsePropagation._mmgnlse_save_targets(
        Float64(parameters.length), saveat)
    steps, saved = PulsePropagation._mmgnlse_step_grid(
        Float64(parameters.length), dz, targets, every_step)
    field_w0 = ifft(initial_field, 1)
    workspace = if gpu_cache.overlap isa CUDAMMGNLSECP
        _cuda_mmgnlse_cp_forward_workspace(initial_field, gpu_cache.overlap)
    else
        _cuda_mmgnlse_dense_forward_workspace(initial_field, gpu_cache)
    end
    function rhs!(derivative, field_w, _, z)
        if workspace === nothing
            derivative .= _cuda_mmgnlse_rhs_spectral(
                field_w, gpu_cache, z)
        else
            if gpu_cache.constant_passive
                derivative .= gpu_cache.beta_operator .* field_w
            else
                derivative .=
                    _cuda_mmgnlse_linear_operator(gpu_cache, z) .* field_w
            end
            if workspace isa CUDAMMGNLSECPForwardWorkspace
                _cuda_mmgnlse_cp_nonlinear_spectral!(
                    workspace.k1, field_w, gpu_cache, workspace)
                derivative .+= workspace.k1
            else
                dense_base = _cuda_mmgnlse_dense_base(workspace)
                _cuda_mmgnlse_dense_nonlinear_spectral!(
                    dense_base.k1, field_w, gpu_cache, workspace)
                derivative .+= dense_base.k1
            end
        end
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
               cuda=gpu_cache, workspace=workspace,
               ode_solution=ode_solution))
end

function _cuda_mmgnlse_solve_device(
    initial_field::CUDA.CuArray{T,3},
    parameters::PulsePropagation.MMGNLSEParameters,
    dz::Float64;
    method=PulsePropagation.RK4IP(),
    saveat=nothing,
    cp_optimization::Symbol=:baseline,
    precision::Symbol=:float64,
    adaptive::Bool=false,
    reltol::Float64=1e-6,
    abstol::Float64=1e-12,
    dzmin::Float64=0.0,
    dzmax::Float64=Inf,
    maxiters::Int=10^7,
) where {T<:Number}
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    field = _cuda_mmgnlse_validate_initial(initial_field, parameters)
    precision in (:float64, :mixed) || throw(ArgumentError(
        "precision must be :float64 or :mixed; got $(repr(precision))."))
    (precision === :mixed || adaptive) &&
        !(method isa PulsePropagation.RK4IP) && throw(ArgumentError(
            "precision=:mixed and adaptive stepping require the forward " *
            "CUDA RK4IP solver."))
    isfinite(reltol) && reltol >= 0 || throw(ArgumentError(
        "reltol must be finite and nonnegative."))
    isfinite(abstol) && abstol >= 0 || throw(ArgumentError(
        "abstol must be finite and nonnegative."))
    reltol > 0 || abstol > 0 || throw(ArgumentError(
        "At least one of reltol and abstol must be positive."))
    isfinite(dzmin) && dzmin >= 0 || throw(ArgumentError(
        "dzmin must be finite and nonnegative."))
    (isfinite(dzmax) || dzmax == Inf) && dzmax > 0 ||
        throw(ArgumentError("dzmax must be positive."))
    dzmin <= dzmax || throw(ArgumentError(
        "dzmin cannot exceed dzmax."))
    maxiters > 0 || throw(ArgumentError("maxiters must be positive."))
    if method isa PulsePropagation.RK4IP
        adaptive && return _cuda_mmgnlse_solve_rk4ip_adaptive(
            field, parameters, dz, saveat, cp_optimization, precision;
            reltol, abstol, dzmin, dzmax, maxiters)
        return _cuda_mmgnlse_solve_rk4ip(
            field, parameters, dz, saveat, cp_optimization, precision)
    elseif nameof(typeof(method)) == :Tsit5
        return _cuda_mmgnlse_solve_tsit5(
            field, parameters, dz, saveat, method, cp_optimization)
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
    backend::Symbol=:cuda,
    precision::Symbol=:float64,
    adaptive::Bool=false,
    reltol::Float64=1e-6,
    abstol::Float64=1e-12,
    dzmin::Float64=0.0,
    dzmax::Float64=Inf,
    maxiters::Int=10^7,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    step = Float64(dz)
    isfinite(step) && step > 0 || throw(ArgumentError(
        "dz must be finite and positive."))
    cp_optimization = _cuda_mmgnlse_cp_optimization(backend)
    device_solution = _cuda_mmgnlse_solve_device(
        CUDA.CuArray(ComplexF64.(initial_field)),
        parameters, step;
        method, saveat, cp_optimization, precision, adaptive,
        reltol, abstol, dzmin, dzmax, maxiters)
    synchronize && CUDA.synchronize()
    device_cache = device_solution.cache
    host_cache =
        if method isa PulsePropagation.RK4IP
            (
                backend,
                integration_z=device_cache.integration_z,
                core=device_cache.core,
                device=string(CUDA.device()),
                precision=device_cache.precision,
                adaptive=device_cache.adaptive,
                accepted_steps=device_cache.accepted_steps,
                rejected_steps=device_cache.rejected_steps,
                rhs_evaluations=device_cache.rhs_evaluations,
                step_statistics=device_cache.adaptive ? (
                    min=device_cache.min_step,
                    max=device_cache.max_step,
                    mean=device_cache.mean_step,
                    last_error=device_cache.last_error,
                    max_error=device_cache.max_error,
                    reltol=device_cache.reltol,
                    abstol=device_cache.abstol,
                    dzmin=device_cache.dzmin,
                    dzmax=device_cache.dzmax,
                ) : nothing,
            )
        else
            (
                backend,
                integration_z=device_cache.integration_z,
                core=device_cache.core,
                device=string(CUDA.device()),
            )
        end
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
