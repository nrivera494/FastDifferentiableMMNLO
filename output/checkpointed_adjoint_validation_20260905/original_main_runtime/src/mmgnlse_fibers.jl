import Arpack
import SpecialFunctions
using SparseArrays

const _FIBER_C_M_PER_PS = 2.99792458e-4

abstract type AbstractMMGNLSEFiber end

"""
    GRIN(; lambda0, core_radius, NA)

Standard weak-guidance graded-index fiber. The profile is parabolic in
permittivity,

``n^2(r, lambda) = n_core(lambda)^2 - NA^2 (r/core_radius)^2``

inside the core and equals `n_core(lambda)^2 - NA^2` outside it. Wavelengths
and `core_radius` are in metres. `NA` is held constant over a dispersion
stencil.
"""
struct GRIN{T<:Real} <: AbstractMMGNLSEFiber
    lambda0::T
    core_radius::T
    NA::T
end

function GRIN(; lambda0, core_radius, NA)
    T = promote_type(typeof(float(lambda0)), typeof(float(core_radius)),
                     typeof(float(NA)))
    fiber = GRIN{T}(T(lambda0), T(core_radius), T(NA))
    return _validate_analytic_fiber(fiber)
end

"""
    StepIndex(; lambda0, core_radius, NA)

Weak-guidance circular step-index fiber. The material supplies
`n_core(lambda)` and the cladding index is
`sqrt(n_core(lambda)^2 - NA^2)` at every wavelength.
"""
struct StepIndex{T<:Real} <: AbstractMMGNLSEFiber
    lambda0::T
    core_radius::T
    NA::T
end

function StepIndex(; lambda0, core_radius, NA)
    T = promote_type(typeof(float(lambda0)), typeof(float(core_radius)),
                     typeof(float(NA)))
    fiber = StepIndex{T}(T(lambda0), T(core_radius), T(NA))
    return _validate_analytic_fiber(fiber)
end

function _validate_analytic_fiber(fiber::Union{GRIN,StepIndex})
    isfinite(fiber.lambda0) && fiber.lambda0 > 0 || throw(ArgumentError(
        "lambda0 must be finite and positive (metres)."))
    isfinite(fiber.core_radius) && fiber.core_radius > 0 || throw(ArgumentError(
        "core_radius must be finite and positive (metres)."))
    isfinite(fiber.NA) && fiber.NA > 0 || throw(ArgumentError(
        "NA must be finite and positive."))
    return fiber
end

"""
    IndexProfile(x, y, nxy; lambda0)
    IndexProfile(n; lambda0, x, y)

Custom scalar refractive-index profile on a uniform Cartesian grid. `nxy`
has shape `length(y) x length(x)` and specifies the absolute index at
`lambda0`. At another wavelength the sampled form preserves its permittivity
offset from the material,

``n^2(x,y,lambda) = n_material(lambda)^2 +
                    nxy(x,y)^2 - n_material(lambda0)^2``.

The callable form is evaluated as `n(x, y, lambda)`, with all three spatial
arguments in metres. It can therefore supply arbitrary wavelength dependence.
"""
struct IndexProfile{T<:Real,A,F} <: AbstractMMGNLSEFiber
    lambda0::T
    x::Vector{T}
    y::Vector{T}
    samples::A
    model::F
end

function IndexProfile(x::AbstractVector, y::AbstractVector,
                      nxy::AbstractMatrix; lambda0)
    T = promote_type(typeof(float(lambda0)), float(eltype(x)), float(eltype(y)),
                     float(eltype(nxy)))
    xv = T.(x)
    yv = T.(y)
    values = T.(nxy)
    size(values) == (length(yv), length(xv)) || throw(DimensionMismatch(
        "nxy must have shape (length(y), length(x)) = " *
        "$((length(yv), length(xv))); got $(size(values))."))
    _validate_index_grid(xv, yv)
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError(
        "The sampled refractive index must be finite and positive."))
    wavelength = T(lambda0)
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError(
        "lambda0 must be finite and positive (metres)."))
    return IndexProfile{T,typeof(values),Nothing}(
        wavelength, xv, yv, values, nothing)
end

function IndexProfile(model::F; lambda0, x, y) where {F}
    T = promote_type(typeof(float(lambda0)), float(eltype(x)), float(eltype(y)))
    xv = T.(x)
    yv = T.(y)
    _validate_index_grid(xv, yv)
    applicable(model, first(xv), first(yv), T(lambda0)) || throw(ArgumentError(
        "The index model must be callable as n(x, y, lambda), in metres."))
    wavelength = T(lambda0)
    isfinite(wavelength) && wavelength > 0 || throw(ArgumentError(
        "lambda0 must be finite and positive (metres)."))
    profile = IndexProfile{T,Nothing,F}(wavelength, xv, yv, nothing, model)
    # Validate the callable over the actual grid, rather than only probing its
    # method table. This gives an early and local error for malformed profiles.
    _evaluate_callable_index(profile, wavelength)
    return profile
end

IndexProfile(model, x::AbstractVector, y::AbstractVector; lambda0) =
    IndexProfile(model; lambda0=lambda0, x=x, y=y)

function _validate_index_grid(x, y)
    length(x) >= 3 && length(y) >= 3 || throw(ArgumentError(
        "A finite-difference index grid requires at least three points per axis."))
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError(
        "Index-profile coordinates must be finite."))
    _uniform_grid_step(x; name=:x)
    _uniform_grid_step(y; name=:y)
    return nothing
end

function _uniform_grid_step(axis::AbstractVector; name=:axis)
    step = axis[2] - axis[1]
    step > 0 || throw(ArgumentError("$name must be strictly increasing."))
    tolerance = 256eps(float(eltype(axis))) * max(one(float(eltype(axis))),
                                                       maximum(abs, axis))
    all(delta -> isapprox(delta, step; rtol=256eps(float(eltype(axis))),
                          atol=tolerance), diff(axis)) || throw(ArgumentError(
        "$name must be uniformly sampled."))
    return step
end

"""
    silica_refractive_index(lambda_m)

Malitson Sellmeier index of fused silica. The wavelength is in metres.
"""
function silica_refractive_index(lambda_m::Real)
    isfinite(lambda_m) && lambda_m > 0 || throw(ArgumentError(
        "Wavelength must be finite and positive (metres)."))
    lambda_um_squared = (float(lambda_m) * 1e6)^2
    denominators = (lambda_um_squared - 0.0684043^2,
                    lambda_um_squared - 0.1162414^2,
                    lambda_um_squared - 9.896161^2)
    all(!iszero, denominators) || throw(DomainError(lambda_m,
        "The wavelength lies on a Sellmeier pole."))
    n_squared = 1 +
        0.6961663 * lambda_um_squared / denominators[1] +
        0.4079426 * lambda_um_squared / denominators[2] +
        0.8974794 * lambda_um_squared / denominators[3]
    n_squared > 0 || throw(DomainError(lambda_m,
        "The Malitson model is not real at this wavelength."))
    return sqrt(n_squared)
end

silica_refractive_index(lambda_m::AbstractArray{<:Real}) =
    silica_refractive_index.(lambda_m)

function _is_silica_material(material)
    type_name = nameof(typeof(material))
    return type_name === :Silica
end

"""Return the material index at a wavelength in metres."""
function material_refractive_index(material, lambda_m::Real)
    _is_silica_material(material) || throw(ArgumentError(
        "No refractive-index model is registered for $(typeof(material))."))
    return silica_refractive_index(lambda_m)
end

function _material_n2(material)
    if hasproperty(material, :n2)
        value = float(getproperty(material, :n2))
        isfinite(value) && value > 0 || throw(ArgumentError(
            "The material n2 must be finite and positive."))
        return value
    end
    _is_silica_material(material) && return 2.3e-20
    throw(ArgumentError("Material $(typeof(material)) does not provide n2."))
end

function _material_raman(material)
    hasproperty(material, :raman) && return getproperty(material, :raman)
    _is_silica_material(material) && return AnisotropicRaman()
    throw(ArgumentError("Material $(typeof(material)) does not provide a Raman response."))
end

function _material_index_checked(material, lambda_m)
    index = material_refractive_index(material, lambda_m)
    isfinite(index) && index > 0 || throw(DomainError(index,
        "The material index must be finite and positive."))
    return index
end

function _cladding_index(material, lambda_m, NA)
    core = _material_index_checked(material, lambda_m)
    core^2 > NA^2 || throw(DomainError(NA,
        "NA must be smaller than the material refractive index."))
    return sqrt(core^2 - NA^2)
end

"""
    evaluate_index_profile(profile, material, lambda; x=nothing, y=nothing)

Evaluate a fiber's scalar refractive index. For analytic profiles, `x` and
`y` are required. An `IndexProfile` carries its own grid.
"""
function evaluate_index_profile(fiber::GRIN, material, lambda_m::Real;
                                x=nothing, y=nothing)
    x === nothing && throw(ArgumentError("x is required for a GRIN profile."))
    y === nothing && throw(ArgumentError("y is required for a GRIN profile."))
    core = _material_index_checked(material, lambda_m)
    clad_squared = core^2 - fiber.NA^2
    clad_squared > 0 || throw(DomainError(fiber.NA,
        "NA must be smaller than the material refractive index."))
    values = Matrix{typeof(core)}(undef, length(y), length(x))
    @inbounds for ix in eachindex(x), iy in eachindex(y)
        rho_squared = (x[ix]^2 + y[iy]^2) / fiber.core_radius^2
        values[iy, ix] = sqrt(max(clad_squared,
            core^2 - fiber.NA^2 * rho_squared))
    end
    return values
end

function evaluate_index_profile(fiber::StepIndex, material, lambda_m::Real;
                                x=nothing, y=nothing)
    x === nothing && throw(ArgumentError("x is required for a StepIndex profile."))
    y === nothing && throw(ArgumentError("y is required for a StepIndex profile."))
    core = _material_index_checked(material, lambda_m)
    clad = _cladding_index(material, lambda_m, fiber.NA)
    values = Matrix{typeof(core)}(undef, length(y), length(x))
    radius_squared = fiber.core_radius^2
    @inbounds for ix in eachindex(x), iy in eachindex(y)
        values[iy, ix] = x[ix]^2 + y[iy]^2 <= radius_squared ? core : clad
    end
    return values
end

function evaluate_index_profile(profile::IndexProfile, material, lambda_m::Real;
                                x=nothing, y=nothing)
    x === nothing || x == profile.x || throw(ArgumentError(
        "A sampled IndexProfile is evaluated on its stored x grid."))
    y === nothing || y == profile.y || throw(ArgumentError(
        "A sampled IndexProfile is evaluated on its stored y grid."))
    if profile.samples === nothing
        return _evaluate_callable_index(profile, lambda_m)
    end
    reference_material = _material_index_checked(material, profile.lambda0)
    material_index = _material_index_checked(material, lambda_m)
    epsilon = profile.samples .^ 2 .-
              reference_material^2 .+ material_index^2
    all(>(zero(eltype(epsilon))), epsilon) || throw(DomainError(lambda_m,
        "The constant-permittivity-offset profile is non-positive."))
    return sqrt.(epsilon)
end

function _evaluate_callable_index(profile::IndexProfile, lambda_m::Real)
    isfinite(lambda_m) && lambda_m > 0 || throw(ArgumentError(
        "Wavelength must be finite and positive (metres)."))
    values = Matrix{float(eltype(profile.x))}(
        undef, length(profile.y), length(profile.x))
    @inbounds for ix in eachindex(profile.x), iy in eachindex(profile.y)
        value = profile.model(profile.x[ix], profile.y[iy], lambda_m)
        value isa Real || throw(ArgumentError(
            "The callable refractive-index profile must return real values."))
        values[iy, ix] = value
    end
    all(value -> isfinite(value) && value > 0, values) || throw(ArgumentError(
        "The callable refractive-index profile must be finite and positive."))
    return values
end

"""Carrier-wavelength mode samples and propagation constants."""
struct FiberModeData{T<:Real,F<:Number,L}
    x::Vector{T}
    y::Vector{T}
    fields::Array{F,3}
    beta0::Vector{T}
    labels::L
    wavelength::T
end

function FiberModeData(x::AbstractVector, y::AbstractVector,
                       fields::AbstractArray{F,3}, beta0::AbstractVector,
                       labels, wavelength) where {F<:Number}
    T = promote_type(float(eltype(x)), float(eltype(y)), float(eltype(beta0)),
                     typeof(float(wavelength)))
    xv, yv, betav = T.(x), T.(y), T.(beta0)
    field_values = Array(fields)
    size(field_values, 1) == length(yv) &&
        size(field_values, 2) == length(xv) || throw(DimensionMismatch(
        "Mode fields must have shape (length(y), length(x), num_modes)."))
    size(field_values, 3) == length(betav) || throw(DimensionMismatch(
        "Mode-field and beta0 mode counts differ."))
    length(labels) == length(betav) || throw(DimensionMismatch(
        "Mode labels and beta0 mode counts differ."))
    return FiberModeData{T,eltype(field_values),typeof(labels)}(
        xv, yv, field_values, betav, labels, T(wavelength))
end

"""
    FiberProperties

Result of `compute_fiber_properties`. `beta` is a `TaylorBeta`, `S` is a
spatial overlap in `m^-2`, `raman` comes from the material, and `modes`
contains power-normalized carrier fields. `n2` and `omega0` remain separate
from `S` for direct construction of `MMGNLSEParameters`.
"""
struct FiberProperties{B,O,R,M,C,T<:Real}
    beta::B
    S::O
    raman::R
    modes::M
    config::C
    n2::T
    omega0::T
end

function FiberProperties(beta, S, raman, modes, config, n2, omega0)
    T = promote_type(typeof(float(n2)), typeof(float(omega0)))
    return FiberProperties{typeof(beta),typeof(S),typeof(raman),typeof(modes),
                           typeof(config),T}(
        beta, S, raman, modes, config, T(n2), T(omega0))
end

mode_grid(properties::FiberProperties) =
    (x=properties.modes.x, y=properties.modes.y)
mode_fields(properties::FiberProperties) = properties.modes.fields
propagation_constants(properties::FiberProperties) = properties.modes.beta0
spatial_overlap(properties::FiberProperties) =
    properties.S isa SpatialOverlap ? properties.S.values : properties.S

function _normalize_polarization(value)
    symbol = value isa Symbol ? value : Symbol(lowercase(String(value)))
    symbol == :polarized && (symbol = :linear)
    symbol == :xy && (symbol = :linear)
    symbol in (:scalar, :linear) || throw(ArgumentError(
        "polarization must be :scalar or :linear (x/y)."))
    return symbol
end

function _normalize_mode_basis(fiber, mode_basis)
    if mode_basis === nothing
        return fiber isa GRIN ? :HG : fiber isa StepIndex ? :LP : :FD
    end
    basis = Symbol(uppercase(String(mode_basis)))
    fiber isa GRIN && basis in (:HG, :LP) && return basis
    fiber isa StepIndex && basis == :LP && return basis
    fiber isa IndexProfile && basis == :FD && return basis
    allowed = fiber isa GRIN ? ":HG or :LP" :
              fiber isa StepIndex ? ":LP" : ":FD"
    throw(ArgumentError("$(typeof(fiber)) supports mode_basis=$allowed."))
end

function _analytic_grid(fiber::GRIN, labels; grid_size, grid_half_width)
    grid_size >= 17 || throw(ArgumentError("grid_size must be at least 17."))
    half_width = grid_half_width === nothing ? fiber.core_radius :
                 float(grid_half_width)
    half_width > 0 || throw(ArgumentError("grid_half_width must be positive."))
    x = collect(range(-half_width, half_width; length=grid_size))
    return x, copy(x)
end

function _analytic_grid(fiber::StepIndex, labels; grid_size, grid_half_width)
    grid_size >= 17 || throw(ArgumentError("grid_size must be at least 17."))
    half_width = grid_half_width === nothing ? 3 * fiber.core_radius :
                 float(grid_half_width)
    half_width > fiber.core_radius || throw(ArgumentError(
        "Step-index grid_half_width must exceed core_radius."))
    x = collect(range(-half_width, half_width; length=grid_size))
    return x, copy(x)
end

function _hermite_polynomial(order::Integer, x)
    order == 0 && return one(x)
    order == 1 && return 2x
    previous, current = one(x), 2x
    for n in 1:order-1
        previous, current = current, 2x * current - 2n * previous
    end
    return current
end

function _associated_laguerre(order::Integer, alpha::Integer, x)
    order == 0 && return one(x)
    order == 1 && return one(x) + alpha - x
    previous, current = one(x), one(x) + alpha - x
    for n in 1:order-1
        following = ((2n + 1 + alpha - x) * current -
                     (n + alpha) * previous) / (n + 1)
        previous, current = current, following
    end
    return current
end

function _grin_hg_labels(num_modes)
    labels = NamedTuple[]
    total_order = 0
    while length(labels) < num_modes
        for nx in total_order:-1:0
            ny = total_order - nx
            push!(labels, (family=:HG, nx=nx, ny=ny, order=total_order,
                           parity=:cartesian))
            length(labels) == num_modes && return labels
        end
        total_order += 1
    end
    return labels
end

function _grin_lp_labels(num_modes)
    labels = NamedTuple[]
    total_order = 0
    while length(labels) < num_modes
        for azimuthal in total_order:-2:0
            radial = (total_order - azimuthal) ÷ 2
            if azimuthal == 0
                push!(labels, (family=:LP, l=azimuthal, m=radial + 1,
                               p=radial, order=total_order, parity=:axisymmetric))
            else
                push!(labels, (family=:LP, l=azimuthal, m=radial + 1,
                               p=radial, order=total_order, parity=:cos))
                length(labels) == num_modes && return labels
                push!(labels, (family=:LP, l=azimuthal, m=radial + 1,
                               p=radial, order=total_order, parity=:sin))
            end
            length(labels) == num_modes && return labels
        end
        total_order += 1
    end
    return labels
end

function _grin_beta(fiber::GRIN, material, label, wavelength)
    k0 = 2pi / wavelength
    core = _material_index_checked(material, wavelength)
    oscillator = 2k0 * fiber.NA / fiber.core_radius * (label.order + 1)
    beta_squared = (k0 * core)^2 - oscillator
    cutoff_squared = (k0 * _cladding_index(material, wavelength, fiber.NA))^2
    beta_squared > cutoff_squared || throw(ArgumentError(
        "Requested GRIN mode order $(label.order) is not guided at wavelength $wavelength m."))
    return sqrt(beta_squared)
end

function _grin_fields(fiber::GRIN, material, labels, basis, x, y)
    k0 = 2pi / fiber.lambda0
    q = k0 * fiber.NA / fiber.core_radius
    scale = sqrt(q)
    fields = zeros(Float64, length(y), length(x), length(labels))
    @inbounds for (mode, label) in enumerate(labels), ix in eachindex(x), iy in eachindex(y)
        xx, yy = x[ix], y[iy]
        gaussian = exp(-q * (xx^2 + yy^2) / 2)
        if basis == :HG
            fields[iy, ix, mode] =
                _hermite_polynomial(label.nx, scale * xx) *
                _hermite_polynomial(label.ny, scale * yy) * gaussian
        else
            radius = hypot(xx, yy)
            angle = atan(yy, xx)
            radial = (scale * radius)^label.l *
                _associated_laguerre(label.p, label.l, q * radius^2) * gaussian
            angular = label.parity == :axisymmetric ? 1.0 :
                      label.parity == :cos ? cos(label.l * angle) :
                      sin(label.l * angle)
            fields[iy, ix, mode] = radial * angular
        end
    end
    return _normalize_modes!(fields, x, y)
end

function _hermite_polynomial_pair(order::Integer, x)
    order > 0 || throw(ArgumentError("order must be positive."))
    previous = one(x)
    order == 1 && return 2x, previous
    current = 2x
    for degree in 1:order-1
        previous, current =
            current, 2x * current - 2degree * previous
    end
    return current, previous
end

function _gauss_hermite_rule(order::Integer; precision_bits::Integer=256)
    order > 0 || throw(ArgumentError(
        "Gauss-Hermite order must be positive."))
    precision_bits >= 128 || throw(ArgumentError(
        "Gauss-Hermite precision_bits must be at least 128."))
    diagonal = zeros(Float64, order)
    off_diagonal = sqrt.(collect(1:order-1) ./ 2)
    decomposition = eigen(SymTridiagonal(diagonal, off_diagonal))
    initial_nodes = decomposition.values
    nodes = zeros(Float64, order)
    weights = zeros(Float64, order)

    setprecision(BigFloat, Int(precision_bits)) do
        numerator = BigFloat(2)^(order - 1) *
                    BigFloat(factorial(big(order))) * sqrt(big(pi))
        half = order ÷ 2
        for offset in 1:half
            positive_index = order - offset + 1
            negative_index = offset
            root = BigFloat(abs(initial_nodes[positive_index]))
            for _ in 1:50
                polynomial, previous = _hermite_polynomial_pair(order, root)
                correction = polynomial / (2order * previous)
                root -= correction
                abs(correction) <= 32eps(BigFloat) * max(abs(root), one(root)) &&
                    break
            end
            _, previous = _hermite_polynomial_pair(order, root)
            weight = numerator / (BigFloat(order)^2 * previous^2)
            nodes[negative_index] = -Float64(root)
            nodes[positive_index] = Float64(root)
            weights[negative_index] = Float64(weight)
            weights[positive_index] = Float64(weight)
        end
        if isodd(order)
            center = half + 1
            root = zero(BigFloat)
            _, previous = _hermite_polynomial_pair(order, root)
            nodes[center] = 0.0
            weights[center] = Float64(
                numerator / (BigFloat(order)^2 * previous^2))
        end
    end
    return nodes, weights
end

function _scaled_hermite_values(maximum_order::Integer, nodes)
    maximum_order >= 0 || throw(ArgumentError(
        "maximum_order must be nonnegative."))
    values = zeros(Float64, maximum_order + 1, length(nodes))
    values[1, :] .= 1
    maximum_order == 0 && return values
    values[2, :] .= nodes
    for order in 1:maximum_order-1
        values[order + 2, :] .=
            nodes .* values[order + 1, :] ./ sqrt(order + 1) .-
            sqrt(order / (order + 1)) .* values[order, :]
    end
    return values
end

"""
    grin_hg_overlap_quadrature_cp(fiber, num_modes; quadrature_order=48)

Construct the scalar GRIN Hermite-Gaussian overlap tensor directly as a
fully permutation-symmetric CP quadrature. No dense `num_modes^4` tensor is
formed. Columns are normalized and the quadrature weights retain the physical
`m^-2` units of the overlap.

For a basis with maximum total HG order `g`, four-mode products have degree at
most `4g` in either transverse coordinate. The function therefore requires
Gauss-Hermite order `2g+1` or greater, which integrates the analytic infinite-
domain HG overlaps exactly up to floating-point roundoff. For the complete
190-mode Eslami basis (`g=18`), the minimum exact order is 37.
"""
function grin_hg_overlap_quadrature_cp(fiber::GRIN,
                                       num_modes::Integer;
                                       quadrature_order::Integer=48,
                                       precision_bits::Integer=256)
    num_modes > 0 || throw(ArgumentError("num_modes must be positive."))
    labels = _grin_hg_labels(Int(num_modes))
    maximum_order = maximum(label.order for label in labels)
    minimum_exact_order = 2 * maximum_order + 1
    quadrature_order >= minimum_exact_order || throw(ArgumentError(
        "quadrature_order=$quadrature_order is insufficient for maximum HG " *
        "order $maximum_order; use at least $minimum_exact_order."))

    nodes, node_weights = _gauss_hermite_rule(
        Int(quadrature_order); precision_bits)
    hermite_values = _scaled_hermite_values(maximum_order, nodes)
    source_rank = Int(quadrature_order)^2
    factor = Matrix{Float64}(undef, Int(num_modes), source_rank)
    weights = Vector{Float64}(undef, source_rank)

    k0 = 2pi / fiber.lambda0
    oscillator_q = k0 * fiber.NA / fiber.core_radius
    overlap_scale = oscillator_q / (2pi^2)
    column = 0
    @inbounds for y_node in eachindex(nodes), x_node in eachindex(nodes)
        column += 1
        weights[column] = overlap_scale *
                          node_weights[x_node] * node_weights[y_node]
        for (mode, label) in enumerate(labels)
            factor[mode, column] =
                hermite_values[label.nx + 1, x_node] *
                hermite_values[label.ny + 1, y_node]
        end
    end

    floor_norm = sqrt(eps(Float64))
    @inbounds for column in axes(factor, 2)
        column_norm = norm(view(factor, :, column))
        column_norm > floor_norm || throw(ErrorException(
            "Gauss-Hermite source column $column has negligible norm."))
        factor[:, column] ./= column_norm
        weights[column] *= column_norm^4
    end
    factors = ntuple(_ -> copy(factor), 4)
    return MMGNLSECPDecomposition(weights, factors;
        layout=:spatial, nmodes=Int(num_modes), npolarizations=1,
        relative_error=0.0, seed=0, iterations=0, converged=true,
        zero_tensor=false)
end

function _gauss_legendre_rule(order::Integer, lower::Real, upper::Real)
    order >= 2 || throw(ArgumentError(
        "quadrature_order must be at least two."))
    isfinite(lower) && isfinite(upper) && lower < upper ||
        throw(ArgumentError("Quadrature bounds must be finite and increasing."))
    off_diagonal = [
        index / sqrt(4index^2 - 1)
        for index in 1:Int(order)-1
    ]
    decomposition = eigen(SymTridiagonal(
        zeros(Float64, Int(order)), off_diagonal))
    scale = (float(upper) - float(lower)) / 2
    shift = (float(upper) + float(lower)) / 2
    nodes = scale .* decomposition.values .+ shift
    weights = 2scale .* abs2.(decomposition.vectors[1, :])
    return nodes, weights
end

function _uniform_mode_interpolation_indices(axis, nodes, name)
    step = _uniform_grid_step(axis; name)
    first_value = first(axis)
    last_value = last(axis)
    tolerance = 64eps(float(eltype(axis))) *
                max(one(float(eltype(axis))), maximum(abs, axis))
    all(node -> first_value - tolerance <= node <= last_value + tolerance,
        nodes) || throw(ArgumentError(
            "$name quadrature nodes lie outside the sampled mode grid."))
    left = Vector{Int}(undef, length(nodes))
    fraction = Vector{Float64}(undef, length(nodes))
    @inbounds for index in eachindex(nodes)
        coordinate = clamp((nodes[index] - first_value) / step,
                           0.0, length(axis) - 1.0)
        left[index] = min(floor(Int, coordinate) + 1, length(axis) - 1)
        fraction[index] = coordinate - (left[index] - 1)
    end
    return left, fraction
end

@inline function _cubic_mode_interpolate(p0, p1, p2, p3, fraction)
    t = fraction
    return 0.5 * (
        2p1 +
        (-p0 + p2) * t +
        (2p0 - 5p1 + 4p2 - p3) * t^2 +
        (-p0 + 3p1 - 3p2 + p3) * t^3)
end


"""
    mode_overlap_quadrature_cp(fields, x, y; quadrature_order=48)
    mode_overlap_quadrature_cp(modes::FiberModeData; quadrature_order=48)

Construct a factorized CP quadrature for the scalar spatial overlap of any
real sampled mode basis. `fields` has axes `(y, x, mode)` and must be sampled
on uniform `x` and `y` grids in metres. Tensor-product Gauss--Legendre nodes
are evaluated by local bicubic interpolation, producing source rank
`quadrature_order^2` without allocating the dense `Nm^4` overlap tensor.

This is the general sampled-mode counterpart of
`grin_hg_overlap_quadrature_cp`. It applies to GRIN, step-index, and arbitrary
finite-difference index profiles, as well as externally supplied real scalar
modes. Use `cp_compress_mode_overlap` when a convergence-audited compressed
solver tensor is desired.
"""
function mode_overlap_quadrature_cp(
    fields::AbstractArray{<:Real,3},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    quadrature_order::Integer=48,
)
    size(fields, 1) == length(y) && size(fields, 2) == length(x) ||
        throw(DimensionMismatch(
            "fields must have shape (length(y), length(x), num_modes)."))
    size(fields, 3) > 0 || throw(ArgumentError(
        "At least one sampled mode is required."))
    all(isfinite, fields) || throw(ArgumentError(
        "Sampled mode fields must be finite."))
    x_nodes, x_weights = _gauss_legendre_rule(
        Int(quadrature_order), first(x), last(x))
    y_nodes, y_weights = _gauss_legendre_rule(
        Int(quadrature_order), first(y), last(y))
    x_left, x_fraction = _uniform_mode_interpolation_indices(
        x, x_nodes, :x)
    y_left, y_fraction = _uniform_mode_interpolation_indices(
        y, y_nodes, :y)

    nmodes = size(fields, 3)
    source_rank = Int(quadrature_order)^2
    factor = Matrix{Float64}(undef, nmodes, source_rank)
    weights = Vector{Float64}(undef, source_rank)
    retained = trues(source_rank)
    floor_norm = sqrt(eps(Float64))
    column = 0
    @inbounds for jy in eachindex(y_nodes), ix in eachindex(x_nodes)
        column += 1
        i0, j0 = x_left[ix], y_left[jy]
        tx, ty = x_fraction[ix], y_fraction[jy]
        for mode in 1:nmodes
            y0, y1, y2, y3 = max(j0 - 1, 1), j0, j0 + 1,
                             min(j0 + 2, length(y))
            x0, x1, x2, x3 = max(i0 - 1, 1), i0, i0 + 1,
                             min(i0 + 2, length(x))
            row0 = _cubic_mode_interpolate(
                fields[y0, x0, mode], fields[y0, x1, mode],
                fields[y0, x2, mode], fields[y0, x3, mode], tx)
            row1 = _cubic_mode_interpolate(
                fields[y1, x0, mode], fields[y1, x1, mode],
                fields[y1, x2, mode], fields[y1, x3, mode], tx)
            row2 = _cubic_mode_interpolate(
                fields[y2, x0, mode], fields[y2, x1, mode],
                fields[y2, x2, mode], fields[y2, x3, mode], tx)
            row3 = _cubic_mode_interpolate(
                fields[y3, x0, mode], fields[y3, x1, mode],
                fields[y3, x2, mode], fields[y3, x3, mode], tx)
            factor[mode, column] = _cubic_mode_interpolate(
                row0, row1, row2, row3, ty)
        end
        column_norm = norm(view(factor, :, column))
        if column_norm <= floor_norm
            retained[column] = false
            weights[column] = 0.0
        else
            factor[:, column] ./= column_norm
            weights[column] = x_weights[ix] * y_weights[jy] * column_norm^4
        end
    end
    any(retained) || throw(ArgumentError(
        "Every quadrature node has negligible sampled modal amplitude."))
    retained_factor = factor[:, retained]
    retained_weights = weights[retained]
    factors = ntuple(_ -> copy(retained_factor), 4)
    return MMGNLSECPDecomposition(retained_weights, factors;
        layout=:spatial, nmodes, npolarizations=1,
        relative_error=0.0, seed=0, iterations=0, converged=true,
        zero_tensor=false)
end

mode_overlap_quadrature_cp(modes::FiberModeData; kwargs...) =
    mode_overlap_quadrature_cp(
        modes.fields, modes.x, modes.y; kwargs...)

function mode_overlap_quadrature_cp(
    fields::AbstractArray{<:Complex,3}, x, y; kwargs...)
    throw(ArgumentError(
        "Dense-free factorized compression currently requires real scalar mode fields."))
end


"""
    cp_compress_mode_overlap(fields, x, y;
                             target_error, quadrature_order=48,
                             authority_order=64,
                             quadrature_tolerance=target_error/10,
                             backend=:cpu, ...)

Build and compress a sampled-mode overlap without ever materializing its dense
four-index tensor. A higher-order quadrature authority is constructed first;
compression proceeds only when the working quadrature differs from it by no
more than `quadrature_tolerance`. The final CP is then checked directly against
that authority and must satisfy `target_error`.

All remaining keywords are forwarded to factorized `cp_compress`. The returned
CP records its total error against the authority, so its `relative_error`
includes both quadrature and compression error.
"""
function cp_compress_mode_overlap(
    fields::AbstractArray{<:Real,3},
    x::AbstractVector{<:Real},
    y::AbstractVector{<:Real};
    target_error::Real=1e-4,
    quadrature_order::Integer=48,
    authority_order::Integer=max(Int(quadrature_order) + 16,
                                 cld(4 * Int(quadrature_order), 3)),
    quadrature_tolerance::Real=float(target_error) / 10,
    backend::Symbol=:cpu,
    error_block_size::Integer=256,
    max_rank::Integer=min(size(fields, 3)^2, Int(quadrature_order)^2),
    min_rank::Integer=1,
    rank_step::Integer=max(1, cld(Int(max_rank), 16)),
    kwargs...,
)
    0 < target_error < 1 || throw(ArgumentError(
        "target_error must lie in (0, 1)."))
    0 <= quadrature_tolerance < target_error || throw(ArgumentError(
        "quadrature_tolerance must lie in [0, target_error)."))
    authority_order > quadrature_order || throw(ArgumentError(
        "authority_order must exceed quadrature_order."))
    haskey(kwargs, :error) && throw(ArgumentError(
        "Use target_error, not the cp_compress error alias, with " *
        "cp_compress_mode_overlap."))
    working = mode_overlap_quadrature_cp(
        fields, x, y; quadrature_order)
    authority = mode_overlap_quadrature_cp(
        fields, x, y; quadrature_order=authority_order)
    device_value = haskey(kwargs, :device) ? kwargs[:device] : nothing
    synchronize_value = haskey(kwargs, :synchronize) ?
        kwargs[:synchronize] : true
    quadrature_error = cp_relative_error(
        working, authority; block_size=error_block_size,
        backend, device=device_value, synchronize=synchronize_value)
    quadrature_error <= quadrature_tolerance || throw(ArgumentError(
        "The sampled-mode quadrature is not converged: working order " *
        "$quadrature_order differs from authority order $authority_order by " *
        "$quadrature_error, exceeding quadrature_tolerance=$quadrature_tolerance. " *
        "Increase the mode-grid resolution and/or quadrature orders."))
    compression_target = max(
        float(target_error) - quadrature_error,
        16eps(Float64))
    forwarded = (; kwargs...)
    if !haskey(forwarded, :source_checksum) &&
       !haskey(forwarded, :checksum_fn)
        authority_checksum = _canonical_value_checksum((
            x=collect(x), y=collect(y), fields=Array(fields),
            quadrature_order=Int(authority_order),
        ))
        forwarded = merge((source_checksum=authority_checksum,), forwarded)
    end
    compressed = cp_compress(
        working; target_error=compression_target, backend,
        error_block_size, max_rank, min_rank, rank_step, forwarded...)
    total_error = cp_relative_error(
        compressed, authority; block_size=error_block_size,
        backend, device=device_value, synchronize=synchronize_value)
    total_error <= target_error || throw(ErrorException(
        "Compressed overlap error $total_error exceeds target_error=$target_error " *
        "against the authority quadrature."))
    metadata = compressed.metadata
    return MMGNLSECPDecomposition(
        compressed.λ, compressed.U;
        layout=metadata.layout, nmodes=metadata.nmodes,
        npolarizations=metadata.npolarizations,
        source_checksum=metadata.source_checksum,
        relative_error=total_error, seed=compressed.seed,
        iterations=compressed.iterations, converged=true,
        zero_tensor=compressed.zero_tensor)
end

cp_compress_mode_overlap(modes::FiberModeData; kwargs...) =
    cp_compress_mode_overlap(
        modes.fields, modes.x, modes.y; kwargs...)

function _step_characteristic(l::Integer, u, V)
    u <= 0 && return SpecialFunctions.besselj(l, zero(u))
    u >= V && return -u * SpecialFunctions.besselj(l + 1, u)
    w = sqrt(max(V^2 - u^2, zero(V)))
    # The common exp(-w) scale cancels in K_(l+1)(w) / K_l(w).
    # Using the scaled functions prevents underflow for high-V fibers.
    kl_scaled = SpecialFunctions.besselkx(l, w)
    ratio = SpecialFunctions.besselkx(l + 1, w) / kl_scaled
    return w * ratio * SpecialFunctions.besselj(l, u) -
           u * SpecialFunctions.besselj(l + 1, u)
end

function _bisect_root(f, lo, hi; iterations=90)
    flo, fhi = f(lo), f(hi)
    flo == 0 && return lo
    fhi == 0 && return hi
    flo * fhi <= 0 || throw(ArgumentError("Root interval does not bracket a root."))
    for _ in 1:iterations
        middle = (lo + hi) / 2
        fmiddle = f(middle)
        if flo * fmiddle <= 0
            hi, fhi = middle, fmiddle
        else
            lo, flo = middle, fmiddle
        end
    end
    return (lo + hi) / 2
end

function _step_lp_roots(l::Integer, V)
    V > 0 || return Float64[]
    upper = V * (1 - 128eps(float(V)))
    samples = max(2000, ceil(Int, 400V))
    axis = collect(range(max(eps(float(V)), V * 1e-10), upper; length=samples))
    values = [_step_characteristic(l, u, V) for u in axis]
    roots = Float64[]
    for index in 1:length(axis)-1
        fleft, fright = values[index], values[index + 1]
        isfinite(fleft) && isfinite(fright) || continue
        if fleft == 0
            candidate = axis[index]
        elseif fleft * fright < 0
            candidate = _bisect_root(
                u -> _step_characteristic(l, u, V), axis[index], axis[index + 1])
        else
            continue
        end
        isempty(roots) || abs(candidate - roots[end]) > 1e-7 || continue
        push!(roots, candidate)
    end
    return roots
end

function _step_labels(fiber::StepIndex, num_modes)
    V = 2pi * fiber.core_radius * fiber.NA / fiber.lambda0
    candidates = NamedTuple[]
    max_l = max(1, ceil(Int, V))
    for l in 0:max_l
        for (root_index, u) in enumerate(_step_lp_roots(l, V))
            if l == 0
                push!(candidates, (family=:LP, l=l, m=root_index,
                                   parity=:axisymmetric, u=u))
            else
                push!(candidates, (family=:LP, l=l, m=root_index,
                                   parity=:cos, u=u))
                push!(candidates, (family=:LP, l=l, m=root_index,
                                   parity=:sin, u=u))
            end
        end
    end
    sort!(candidates; by=label -> (label.u, label.l,
                                   label.parity == :sin ? 1 : 0))
    length(candidates) >= num_modes || throw(ArgumentError(
        "The step-index fiber supports only $(length(candidates)) scalar LP modes " *
        "at lambda0; requested $num_modes."))
    return candidates[1:num_modes]
end

function _tracked_step_root(label, V)
    width = max(0.02, 0.002 * label.u)
    f(u) = _step_characteristic(label.l, u, V)
    for _ in 1:28
        lo = max(V * 1e-10, label.u - width)
        hi = min(V * (1 - 128eps(float(V))), label.u + width)
        lo < hi || break
        flo, fhi = f(lo), f(hi)
        if isfinite(flo) && isfinite(fhi) && flo * fhi <= 0
            return _bisect_root(f, lo, hi)
        end
        width *= 1.4
    end
    roots = _step_lp_roots(label.l, V)
    length(roots) >= label.m || throw(ArgumentError(
        "LP$(label.l)$(label.m) reaches cutoff in the dispersion stencil."))
    return roots[label.m]
end

function _step_beta(fiber::StepIndex, material, label, wavelength)
    V = 2pi * fiber.core_radius * fiber.NA / wavelength
    u = _tracked_step_root(label, V)
    k0 = 2pi / wavelength
    core = _material_index_checked(material, wavelength)
    beta_squared = (k0 * core)^2 - (u / fiber.core_radius)^2
    cutoff = k0 * _cladding_index(material, wavelength, fiber.NA)
    beta_squared > cutoff^2 || throw(ArgumentError(
        "LP$(label.l)$(label.m) is not guided at wavelength $wavelength m."))
    return sqrt(beta_squared)
end

function _step_fields(fiber::StepIndex, labels, x, y)
    V = 2pi * fiber.core_radius * fiber.NA / fiber.lambda0
    fields = zeros(Float64, length(y), length(x), length(labels))
    @inbounds for (mode, label) in enumerate(labels)
        u = label.u
        w = sqrt(V^2 - u^2)
        boundary_value = SpecialFunctions.besselj(label.l, u)
        boundary_k_scaled = SpecialFunctions.besselkx(label.l, w)
        for ix in eachindex(x), iy in eachindex(y)
            radius = hypot(x[ix], y[iy])
            angle = atan(y[iy], x[ix])
            rho = radius / fiber.core_radius
            radial = if rho <= 1
                SpecialFunctions.besselj(label.l, u * rho)
            else
                boundary_value *
                (SpecialFunctions.besselkx(label.l, w * rho) /
                 boundary_k_scaled) * exp(-w * (rho - 1))
            end
            angular = label.parity == :axisymmetric ? 1.0 :
                      label.parity == :cos ? cos(label.l * angle) :
                      sin(label.l * angle)
            fields[iy, ix, mode] = radial * angular
        end
    end
    return _normalize_modes!(fields, x, y)
end

function _normalize_modes!(fields, x, y)
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    weight = dx * dy
    for mode in axes(fields, 3)
        field = @view fields[:, :, mode]
        normalization = sqrt(sum(abs2, field) * weight)
        isfinite(normalization) && normalization > 0 || throw(ArgumentError(
            "Mode $mode has zero or non-finite norm."))
        field ./= normalization
        peak = field[argmax(abs.(field))]
        if peak isa Real
            peak < 0 && (field .*= -1)
        else
            field ./= peak / abs(peak)
        end
    end
    return fields
end

function _mode_gram(fields, x, y)
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    flat = reshape(fields, :, size(fields, 3))
    return adjoint(flat) * flat * (dx * dy)
end

function _spatial_overlap(fields, x, y;
                          chunk_points=4096,
                          max_workspace_bytes=512 * 1024^2,
                          precision::Type{<:AbstractFloat}=Float64,
                          max_tensor_bytes=nothing)
    chunk_points > 0 || throw(ArgumentError("overlap_chunk_points must be positive."))
    max_workspace_bytes === nothing || max_workspace_bytes > 0 ||
        throw(ArgumentError(
            "overlap_max_workspace_bytes must be positive or nothing."))
    max_tensor_bytes === nothing || max_tensor_bytes > 0 ||
        throw(ArgumentError(
            "overlap_max_tensor_bytes must be positive or nothing."))
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    flat = reshape(fields, :, size(fields, 3))
    points, modes = size(flat)
    pair_count = modes^2
    T = eltype(fields) <: Real ? precision : Complex{precision}
    tensor_bytes = Base.checked_mul(
        Base.checked_mul(pair_count, pair_count), sizeof(T))
    if max_tensor_bytes !== nothing && tensor_bytes > max_tensor_bytes
        throw(ArgumentError(
            "The dense $modes-mode overlap requires $tensor_bytes bytes, " *
            "exceeding overlap_max_tensor_bytes=$(Int(max_tensor_bytes)). " *
            "Use overlap_precision=Float32, raise the limit, or reduce the " *
            "mode count."))
    end
    bytes_per_point = Base.checked_mul(pair_count, sizeof(T))
    workspace_points = max_workspace_bytes === nothing ? Int(chunk_points) :
        min(Int(chunk_points), Int(max_workspace_bytes) ÷ bytes_per_point)
    workspace_points > 0 || throw(ArgumentError(
        "overlap_max_workspace_bytes=$(Int(max_workspace_bytes)) cannot hold " *
        "one pair-product row ($bytes_per_point bytes at $modes modes)."))

    real_symmetric = eltype(fields) <: Real
    tensor = zeros(T, modes, modes, modes, modes)
    gram = real_symmetric ? reshape(tensor, pair_count, pair_count) :
           zeros(T, pair_count, pair_count)
    pairs_workspace = Matrix{T}(undef, workspace_points, pair_count)
    for first_point in 1:workspace_points:points
        last_point = min(points, first_point + workspace_points - 1)
        count = last_point - first_point + 1
        pairs = @view pairs_workspace[1:count, :]
        column = 0
        for last_mode in 1:modes, first_mode in 1:modes
            column += 1
            @views pairs[:, column] .=
                flat[first_point:last_point, first_mode] .*
                flat[first_point:last_point, last_mode]
        end
        mul!(gram, adjoint(pairs), pairs, dx * dy, one(T))
    end
    # gram axes are (output, conjugated-input, first-input, second-input)
    # after the pair-product construction. Restore solver order
    # (output, first-input, second-input, conjugated-input).
    if !real_symmetric
        permutedims!(
            tensor, reshape(gram, modes, modes, modes, modes), (1, 3, 4, 2))
    end
    return SpatialOverlap(tensor; copy_values=false)
end

function _fd_operator(profile::IndexProfile, material, wavelength)
    x, y = profile.x, profile.y
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    nx, ny = length(x), length(y)
    tx = spdiagm(-1 => ones(nx - 1), 0 => fill(-2.0, nx),
                 1 => ones(nx - 1)) / dx^2
    ty = spdiagm(-1 => ones(ny - 1), 0 => fill(-2.0, ny),
                 1 => ones(ny - 1)) / dy^2
    laplacian = kron(sparse(I, nx, nx), ty) +
                kron(tx, sparse(I, ny, ny))
    index = evaluate_index_profile(profile, material, wavelength)
    k0 = 2pi / wavelength
    operator = laplacian + spdiagm(0 => k0^2 .* vec(index .^ 2))
    return operator
end

function _solve_fd_modes(profile::IndexProfile, material, wavelength, num_modes;
                         arpack_tol, arpack_maxiter, arpack_ncv,
                         tracking_padding=0)
    dimension = length(profile.x) * length(profile.y)
    requested = num_modes + tracking_padding
    requested < dimension - 1 || throw(ArgumentError(
        "The finite-difference grid is too small for $requested eigenmodes."))
    ncv = arpack_ncv === nothing ? max(2requested + 8, 20) : Int(arpack_ncv)
    ncv = min(dimension, ncv)
    ncv > requested || throw(ArgumentError("Arpack ncv must exceed the requested mode count."))
    operator = _fd_operator(profile, material, wavelength)
    values, vectors, nconv, _, _, _ = Arpack.eigs(
        Symmetric(operator); nev=requested, ncv=ncv, which=:LR,
        tol=arpack_tol, maxiter=arpack_maxiter)
    nconv >= requested || throw(ErrorException(
        "Arpack converged $nconv of $requested requested modes."))
    order = sortperm(real.(values); rev=true)
    eigenvalues = real.(values[order])
    all(>(0), eigenvalues) || throw(DomainError(minimum(eigenvalues),
        "The requested finite-difference modes do not have real propagation constants."))
    vectors_real = real.(vectors[:, order])
    fields = reshape(vectors_real, length(profile.y), length(profile.x), requested)
    _normalize_modes!(fields, profile.x, profile.y)
    beta = sqrt.(eigenvalues)
    return fields, beta
end

function _match_fd_modes(reference, candidates, candidate_beta, x, y)
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    ref_flat = reshape(reference, :, size(reference, 3))
    candidate_flat = reshape(candidates, :, size(candidates, 3))
    overlaps = abs.(adjoint(ref_flat) * candidate_flat) .* (dx * dy)
    available = trues(size(candidates, 3))
    matched = similar(candidate_beta, size(reference, 3))
    for mode in axes(reference, 3)
        scores = copy(@view overlaps[mode, :])
        scores[.!available] .= -Inf
        selected = argmax(scores)
        isfinite(scores[selected]) || throw(ErrorException(
            "Could not track custom-profile mode $mode across wavelength."))
        available[selected] = false
        matched[mode] = candidate_beta[selected]
    end
    return matched
end

function _dispersion_offsets(lambda0, beta_order, span, samples)
    beta_order >= 0 || throw(ArgumentError("beta_order must be nonnegative."))
    span > 0 || throw(ArgumentError("dispersion_span must be positive (rad/ps)."))
    samples >= beta_order + 1 || throw(ArgumentError(
        "dispersion_samples must be at least beta_order + 1."))
    isodd(samples) || throw(ArgumentError(
        "dispersion_samples must be odd so the carrier is sampled exactly."))
    omega0 = 2pi * _FIBER_C_M_PER_PS / lambda0
    span < omega0 || throw(ArgumentError(
        "dispersion_span must be smaller than the carrier angular frequency."))
    return collect(range(-float(span), float(span); length=samples)), omega0
end

function _fit_taylor_beta(offsets, values, beta_order, fit_order)
    samples, num_modes = size(values)
    fit_order >= beta_order || throw(ArgumentError(
        "dispersion_fit_order must be at least beta_order."))
    fit_order < samples || throw(ArgumentError(
        "dispersion_fit_order must be smaller than dispersion_samples."))
    span = maximum(abs, offsets)
    normalized = offsets ./ span
    vandermonde = hcat([normalized .^ order for order in 0:fit_order]...)
    center = findmin(abs.(offsets))[2]
    centered_values = values .- transpose(values[center, :])
    polynomial = vandermonde \ centered_values
    coefficients = zeros(eltype(values), beta_order + 1, num_modes)
    coefficients[1, :] .= values[center, :]
    for order in 1:beta_order
        coefficients[order + 1, :] .=
            factorial(order) .* polynomial[order + 1, :] ./ span^order
    end
    return TaylorBeta(coefficients)
end

function _analytic_beta_samples(fiber::Union{GRIN,StepIndex}, material, labels,
                                offsets, omega0)
    values = zeros(Float64, length(offsets), length(labels))
    for (sample, offset) in enumerate(offsets)
        wavelength = 2pi * _FIBER_C_M_PER_PS / (omega0 + offset)
        for (mode, label) in enumerate(labels)
            values[sample, mode] = fiber isa GRIN ?
                _grin_beta(fiber, material, label, wavelength) :
                _step_beta(fiber, material, label, wavelength)
        end
    end
    return values
end

function _fd_beta_samples(profile::IndexProfile, material, reference_fields,
                          num_modes, offsets, omega0;
                          arpack_tol, arpack_maxiter, arpack_ncv,
                          tracking_padding)
    values = zeros(Float64, length(offsets), num_modes)
    for (sample, offset) in enumerate(offsets)
        wavelength = 2pi * _FIBER_C_M_PER_PS / (omega0 + offset)
        if iszero(offset)
            _, beta = _solve_fd_modes(profile, material, wavelength, num_modes;
                arpack_tol=arpack_tol, arpack_maxiter=arpack_maxiter,
                arpack_ncv=arpack_ncv, tracking_padding=0)
            values[sample, :] .= beta
        else
            candidates, beta = _solve_fd_modes(profile, material, wavelength,
                num_modes; arpack_tol=arpack_tol,
                arpack_maxiter=arpack_maxiter, arpack_ncv=arpack_ncv,
                tracking_padding=tracking_padding)
            values[sample, :] .= _match_fd_modes(
                reference_fields, candidates, beta, profile.x, profile.y)
        end
    end
    return values
end

function _profile_kind(fiber)
    fiber isa GRIN && return :grin
    fiber isa StepIndex && return :step_index
    fiber isa IndexProfile && fiber.samples !== nothing && return :custom_sampled
    return :custom_callable
end

function _fiber_config(fiber, material, modes, polarization, basis, beta_order,
                       dispersion_span, dispersion_samples,
                       dispersion_fit_order, overlap_settings)
    core_radius = fiber isa Union{GRIN,StepIndex} ? fiber.core_radius : nothing
    numerical_aperture = fiber isa Union{GRIN,StepIndex} ? fiber.NA : nothing
    wavelength_rule = fiber isa IndexProfile ?
        (fiber.samples === nothing ? :callable : :constant_permittivity_offset) :
        :constant_NA
    return (
        profile=_profile_kind(fiber),
        material=Symbol(nameof(typeof(material))),
        lambda0=fiber.lambda0,
        core_radius=core_radius,
        NA=numerical_aperture,
        num_modes=size(modes.fields, 3),
        polarization=polarization,
        mode_basis=basis,
        beta_order=beta_order,
        dispersion_span=dispersion_span,
        dispersion_samples=dispersion_samples,
        dispersion_fit_order=dispersion_fit_order,
        overlap=overlap_settings,
        wavelength_rule=wavelength_rule,
        grid=(nx=length(modes.x), ny=length(modes.y),
              xmin=first(modes.x), xmax=last(modes.x),
              ymin=first(modes.y), ymax=last(modes.y)),
    )
end

"""
    compute_fiber_properties(fiber, material, num_modes, polarization=:scalar;
                             mode_basis=nothing, beta_order=5,
                             overlap_representation=:dense, ...)

Build carrier modes, Taylor dispersion, and the dense spatial overlap tensor
for an analytic GRIN/step-index fiber or a custom finite-difference profile.
Modes obey `sum(abs2, mode) * dx * dy == 1`, so `S` has units `m^-2`.

For GRIN, `mode_basis` must be `:HG` or `:LP`; for StepIndex it is `:LP`;
custom profiles use `:FD`. `polarization=:linear` records two independent
linear x/y components while retaining the same scalar spatial modes and `S`.
Set `overlap_representation=:quadrature_cp` to return an exact factorized
Gauss--Legendre source on the sampled modes instead of allocating dense `S`;
`overlap_quadrature_order` controls its one-dimensional order. This source can
be passed directly to `cp_compress`, or use `cp_compress_mode_overlap` to add
an authority-order convergence check before compression.
`overlap_max_workspace_bytes` bounds the pair-product buffer used during
overlap construction (512 MiB by default), and `overlap_precision=Float32`
halves dense overlap storage when that accuracy is sufficient.
`overlap_max_tensor_bytes` optionally rejects an oversized dense `N^4`
allocation before it is attempted.
"""
function compute_fiber_properties(fiber::AbstractMMGNLSEFiber, material,
                                  num_modes::Integer,
                                  polarization;
                                  mode_basis=nothing,
                                  beta_order::Integer=5,
                                  dispersion_span::Real=35.0,
                                  dispersion_samples::Integer=max(13, 2beta_order + 3),
                                  dispersion_fit_order::Integer=min(12, dispersion_samples - 1),
                                  grid_size::Integer=129,
                                  grid_half_width=nothing,
                                  overlap_representation::Symbol=:dense,
                                  overlap_quadrature_order::Integer=48,
                                  overlap_chunk_points::Integer=4096,
                                  overlap_max_workspace_bytes::Union{Nothing,Integer}=512 * 1024^2,
                                  overlap_precision::Type{<:AbstractFloat}=Float64,
                                  overlap_max_tensor_bytes::Union{Nothing,Integer}=nothing,
                                  arpack_tol::Real=1e-10,
                                  arpack_maxiter::Integer=5000,
                                  arpack_ncv=nothing,
                                  tracking_padding::Integer=2)
    num_modes > 0 || throw(ArgumentError("num_modes must be positive."))
    overlap_representation in (:dense, :quadrature_cp) ||
        throw(ArgumentError(
            "overlap_representation must be :dense or :quadrature_cp."))
    polarization_value = _normalize_polarization(polarization)
    basis = _normalize_mode_basis(fiber, mode_basis)
    offsets, omega0 = _dispersion_offsets(
        fiber.lambda0, beta_order, dispersion_span, dispersion_samples)
    dispersion_fit_order >= beta_order || throw(ArgumentError(
        "dispersion_fit_order must be at least beta_order."))
    dispersion_fit_order < dispersion_samples || throw(ArgumentError(
        "dispersion_fit_order must be smaller than dispersion_samples."))

    if fiber isa GRIN
        labels = basis == :HG ? _grin_hg_labels(num_modes) :
                               _grin_lp_labels(num_modes)
        # The carrier check gives a direct supported-mode error before any
        # overlap work is done.
        beta0 = [_grin_beta(fiber, material, label, fiber.lambda0)
                 for label in labels]
        x, y = _analytic_grid(fiber, labels; grid_size=grid_size,
                              grid_half_width=grid_half_width)
        fields = _grin_fields(fiber, material, labels, basis, x, y)
        beta_samples = _analytic_beta_samples(
            fiber, material, labels, offsets, omega0)
    elseif fiber isa StepIndex
        labels = _step_labels(fiber, num_modes)
        beta0 = [_step_beta(fiber, material, label, fiber.lambda0)
                 for label in labels]
        x, y = _analytic_grid(fiber, labels; grid_size=grid_size,
                              grid_half_width=grid_half_width)
        fields = _step_fields(fiber, labels, x, y)
        beta_samples = _analytic_beta_samples(
            fiber, material, labels, offsets, omega0)
    else
        grid_half_width === nothing || throw(ArgumentError(
            "A custom IndexProfile already carries its grid; do not pass grid_half_width."))
        x, y = fiber.x, fiber.y
        fields, beta0 = _solve_fd_modes(fiber, material, fiber.lambda0,
            num_modes; arpack_tol=arpack_tol, arpack_maxiter=arpack_maxiter,
            arpack_ncv=arpack_ncv, tracking_padding=0)
        labels = [(family=:FD, index=mode, parity=:numerical)
                  for mode in 1:num_modes]
        beta_samples = _fd_beta_samples(
            fiber, material, fields, num_modes, offsets, omega0;
            arpack_tol=arpack_tol, arpack_maxiter=arpack_maxiter,
            arpack_ncv=arpack_ncv, tracking_padding=tracking_padding)
    end

    modes = FiberModeData(x, y, fields, beta0, labels, fiber.lambda0)
    beta = _fit_taylor_beta(offsets, beta_samples, beta_order,
                            dispersion_fit_order)
    overlap = if overlap_representation === :dense
        _spatial_overlap(fields, x, y;
                         chunk_points=overlap_chunk_points,
                         max_workspace_bytes=overlap_max_workspace_bytes,
                         precision=overlap_precision,
                         max_tensor_bytes=overlap_max_tensor_bytes)
    else
        mode_overlap_quadrature_cp(
            fields, x, y; quadrature_order=overlap_quadrature_order)
    end
    overlap_settings = (
        representation=overlap_representation,
        quadrature_order=Int(overlap_quadrature_order),
        chunk_points=Int(overlap_chunk_points),
        max_workspace_bytes=overlap_max_workspace_bytes,
        precision=Symbol(nameof(overlap_precision)),
        max_tensor_bytes=overlap_max_tensor_bytes,
    )
    config = _fiber_config(
        fiber, material, modes, polarization_value, basis, beta_order,
        float(dispersion_span), dispersion_samples, dispersion_fit_order,
        overlap_settings)
    return FiberProperties(beta, overlap, _material_raman(material), modes,
                           config, _material_n2(material), omega0)
end

function compute_fiber_properties(fiber::AbstractMMGNLSEFiber, material,
                                  num_modes::Integer;
                                  polarization=:scalar, kwargs...)
    return compute_fiber_properties(
        fiber, material, num_modes, polarization; kwargs...)
end
