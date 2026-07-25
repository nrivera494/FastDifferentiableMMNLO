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
spatial_overlap(properties::FiberProperties) = properties.S.values

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

function _step_characteristic(l::Integer, u, V)
    u <= 0 && return SpecialFunctions.besselj(l, zero(u))
    u >= V && return -u * SpecialFunctions.besselj(l + 1, u)
    w = sqrt(max(V^2 - u^2, zero(V)))
    kl = SpecialFunctions.besselk(l, w)
    ratio = SpecialFunctions.besselk(l + 1, w) / kl
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
        boundary_scale = SpecialFunctions.besselj(label.l, u) /
                         SpecialFunctions.besselk(label.l, w)
        for ix in eachindex(x), iy in eachindex(y)
            radius = hypot(x[ix], y[iy])
            angle = atan(y[iy], x[ix])
            rho = radius / fiber.core_radius
            radial = rho <= 1 ? SpecialFunctions.besselj(label.l, u * rho) :
                     boundary_scale * SpecialFunctions.besselk(label.l, w * rho)
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

function _spatial_overlap(fields, x, y; chunk_points=4096)
    chunk_points > 0 || throw(ArgumentError("overlap_chunk_points must be positive."))
    dx = _uniform_grid_step(x; name=:x)
    dy = _uniform_grid_step(y; name=:y)
    flat = reshape(fields, :, size(fields, 3))
    points, modes = size(flat)
    pair_count = modes^2
    T = promote_type(eltype(fields), Float64)
    gram = zeros(T, pair_count, pair_count)
    for first_point in 1:chunk_points:points
        last_point = min(points, first_point + chunk_points - 1)
        count = last_point - first_point + 1
        pairs = Matrix{T}(undef, count, pair_count)
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
    tensor = permutedims(reshape(gram, modes, modes, modes, modes),
                         (1, 3, 4, 2))
    return SpatialOverlap(tensor)
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
                       dispersion_fit_order)
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
        wavelength_rule=wavelength_rule,
        grid=(nx=length(modes.x), ny=length(modes.y),
              xmin=first(modes.x), xmax=last(modes.x),
              ymin=first(modes.y), ymax=last(modes.y)),
    )
end

"""
    compute_fiber_properties(fiber, material, num_modes, polarization=:scalar;
                             mode_basis=nothing, beta_order=5, ...)

Build carrier modes, Taylor dispersion, and the dense spatial overlap tensor
for an analytic GRIN/step-index fiber or a custom finite-difference profile.
Modes obey `sum(abs2, mode) * dx * dy == 1`, so `S` has units `m^-2`.

For GRIN, `mode_basis` must be `:HG` or `:LP`; for StepIndex it is `:LP`;
custom profiles use `:FD`. `polarization=:linear` records two independent
linear x/y components while retaining the same scalar spatial modes and `S`.
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
                                  overlap_chunk_points::Integer=4096,
                                  arpack_tol::Real=1e-10,
                                  arpack_maxiter::Integer=5000,
                                  arpack_ncv=nothing,
                                  tracking_padding::Integer=2)
    num_modes > 0 || throw(ArgumentError("num_modes must be positive."))
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
    overlap = _spatial_overlap(fields, x, y;
                               chunk_points=overlap_chunk_points)
    config = _fiber_config(
        fiber, material, modes, polarization_value, basis, beta_order,
        float(dispersion_span), dispersion_samples, dispersion_fit_order)
    return FiberProperties(beta, overlap, _material_raman(material), modes,
                           config, _material_n2(material), omega0)
end

function compute_fiber_properties(fiber::AbstractMMGNLSEFiber, material,
                                  num_modes::Integer;
                                  polarization=:scalar, kwargs...)
    return compute_fiber_properties(
        fiber, material, num_modes, polarization; kwargs...)
end
