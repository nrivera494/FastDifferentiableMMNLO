const MMGNLSE_PROFILE_AXES = (:frequency, :mode, :polarization, :z)
const _MMGNLSE_DOF_ORDER = (:time, :space, :polarization)

"""
    DegreesOfFreedom(names...)
    degrees_of_freedom(names...)

The active public MMGNLSE field axes. Supported names are `:time`, `:space`,
and `:polarization`. At least one degree of freedom must be active. Names are
stored in canonical order, independently of the order in which they are given.
"""
struct DegreesOfFreedom
    names::Tuple{Vararg{Symbol}}

    function DegreesOfFreedom(names::Tuple{Vararg{Symbol}})
        isempty(names) && throw(ArgumentError(
            "DegreesOfFreedom cannot be empty; select at least one of $(_MMGNLSE_DOF_ORDER)."))
        all(name -> name in _MMGNLSE_DOF_ORDER, names) || throw(ArgumentError(
            "DegreesOfFreedom accepts only $(_MMGNLSE_DOF_ORDER); got $names."))
        length(unique(names)) == length(names) || throw(ArgumentError(
            "DegreesOfFreedom names must be unique; got $names."))
        canonical = Tuple(name for name in _MMGNLSE_DOF_ORDER if name in names)
        return new(canonical)
    end
end

DegreesOfFreedom(names::Symbol...) = DegreesOfFreedom(names)
DegreesOfFreedom(names::AbstractVector{Symbol}) = DegreesOfFreedom(Tuple(names))
degrees_of_freedom(names...) = DegreesOfFreedom(names...)

Base.length(dofs::DegreesOfFreedom) = length(dofs.names)
Base.iterate(dofs::DegreesOfFreedom, state...) = iterate(dofs.names, state...)
Base.in(name::Symbol, dofs::DegreesOfFreedom) = name in dofs.names
Base.:(==)(a::DegreesOfFreedom, b::DegreesOfFreedom) = a.names == b.names
Base.hash(dofs::DegreesOfFreedom, h::UInt) = hash(dofs.names, h)
Base.show(io::IO, dofs::DegreesOfFreedom) =
    print(io, "DegreesOfFreedom(", join((repr(name) for name in dofs.names), ", "), ")")

has_dof(dofs::DegreesOfFreedom, name::Symbol) = name in dofs

"""
    TransverseGrid(x, y)
    TransverseGrid(; Nx, Ny=Nx, dx, dy=dx)

Uniform Cartesian transverse discretization in metres. Public full-field arrays
use the axis order `(x, y)`. The coordinate vectors must be finite, strictly
increasing, and uniformly sampled.
"""
struct TransverseGrid{T<:Real}
    x::Vector{T}
    y::Vector{T}
    dx::T
    dy::T
end

function _mmgnlse_uniform_spacing(values::AbstractVector{<:Real}, name)
    length(values) >= 2 || throw(ArgumentError(
        "$name must contain at least two samples."))
    all(isfinite, values) || throw(ArgumentError(
        "$name samples must be finite."))
    differences = diff(values)
    all(>(zero(eltype(differences))), differences) || throw(ArgumentError(
        "$name samples must be strictly increasing."))
    spacing = first(differences)
    tolerance = 64eps(typeof(float(spacing))) *
                max(one(float(spacing)), maximum(abs, values))
    all(value -> isapprox(value, spacing;
                          rtol=64eps(typeof(float(spacing))),
                          atol=tolerance), differences) || throw(ArgumentError(
        "$name must be uniformly sampled."))
    return spacing
end

function TransverseGrid(x::AbstractVector{<:Real},
                        y::AbstractVector{<:Real})
    T = promote_type(float(eltype(x)), float(eltype(y)))
    x_values = T.(x)
    y_values = T.(y)
    dx = T(_mmgnlse_uniform_spacing(x_values, :x))
    dy = T(_mmgnlse_uniform_spacing(y_values, :y))
    return TransverseGrid{T}(x_values, y_values, dx, dy)
end

function TransverseGrid(; Nx::Integer, Ny::Integer=Nx, dx, dy=dx)
    Nx >= 2 || throw(ArgumentError("Nx must be at least two."))
    Ny >= 2 || throw(ArgumentError("Ny must be at least two."))
    dx_value = float(dx)
    dy_value = float(dy)
    isfinite(dx_value) && dx_value > 0 || throw(ArgumentError(
        "dx must be finite and positive."))
    isfinite(dy_value) && dy_value > 0 || throw(ArgumentError(
        "dy must be finite and positive."))
    return TransverseGrid(centered_time_grid(Int(Nx), dx_value),
                          centered_time_grid(Int(Ny), dy_value))
end

"""
    MMGNLSEDomain(dofs, time_grid; transverse_grid=nothing)
    MMGNLSEDomain(dofs; time_grid=nothing, transverse_grid=nothing)

Discretization metadata associated with an MMGNLSE problem. A `TimeGrid` is
required exactly when `:time` is active. If time is suppressed, the public time
axis is a retained singleton. A `TransverseGrid` may be attached when `:space`
is active; modal parameters do not require one, while full-field parameters do.
"""
struct MMGNLSEDomain{G,S}
    dofs::DegreesOfFreedom
    time_grid::G
    transverse_grid::S
end

function MMGNLSEDomain(dofs::DegreesOfFreedom, time_grid;
                       transverse_grid=nothing)
    if has_dof(dofs, :time)
        time_grid isa TimeGrid || throw(ArgumentError(
            "MMGNLSEDomain requires a TimeGrid when :time is active."))
    elseif time_grid !== nothing
        throw(ArgumentError(
            "MMGNLSEDomain does not accept a TimeGrid when :time is inactive."))
    end
    transverse_grid === nothing || transverse_grid isa TransverseGrid ||
        throw(ArgumentError(
            "transverse_grid must be a TransverseGrid or nothing."))
    !has_dof(dofs, :space) && transverse_grid !== nothing &&
        throw(ArgumentError(
            "MMGNLSEDomain accepts a transverse grid only when :space is active."))
    return MMGNLSEDomain{typeof(time_grid),typeof(transverse_grid)}(
        dofs, time_grid, transverse_grid)
end

MMGNLSEDomain(dofs::DegreesOfFreedom; time_grid=nothing,
              transverse_grid=nothing) =
    MMGNLSEDomain(dofs, time_grid; transverse_grid)

MMGNLSEDomain(names::Symbol...; time_grid=nothing,
              transverse_grid=nothing) =
    MMGNLSEDomain(degrees_of_freedom(names...), time_grid; transverse_grid)

frequency_count(domain::MMGNLSEDomain) =
    has_dof(domain.dofs, :time) ? domain.time_grid.nt : 1
polarization_count(domain::MMGNLSEDomain) = has_dof(domain.dofs, :polarization) ? 2 : 1

abstract type AbstractBeta end

"""
    TaylorBeta(coefficients)

Taylor coefficients `beta[k+1, mode, polarization] = beta_k` about `omega0`.
The leading axis is coefficient order (`beta0`, `beta1`, ...); omitted mode and
polarization axes are treated as singletons. Coefficients use `ps^k / m`.
"""
struct TaylorBeta{T<:Real,N} <: AbstractBeta
    coefficients::Array{T,N}

    function TaylorBeta(coefficients::AbstractArray{T,N}) where {T<:Real,N}
        1 <= N <= 3 || throw(ArgumentError(
            "TaylorBeta coefficients must have 1 to 3 axes (order, mode, polarization)."))
        isempty(coefficients) && throw(ArgumentError("TaylorBeta coefficients cannot be empty."))
        all(isfinite, coefficients) || throw(ArgumentError(
            "TaylorBeta coefficients must be finite."))
        return new{T,N}(Array(coefficients))
    end
end

TaylorBeta(coefficient::Real) = TaylorBeta([coefficient])
TaylorBeta(; coefficients) = TaylorBeta(coefficients)
beta_order(beta::TaylorBeta) = size(beta.coefficients, 1) - 1

"""
    SampledBeta(omega, values)

Propagation constants sampled on a strictly increasing, centered public
angular-frequency grid in `rad / ps`. `values` has axis order
`frequency x mode x polarization`, with singleton trailing axes permitted.
"""
struct SampledBeta{Tω<:Real,Tβ<:Real,N} <: AbstractBeta
    omega::Vector{Tω}
    values::Array{Tβ,N}

    function SampledBeta(omega::AbstractVector{Tω},
                         values::AbstractArray{Tβ,N}) where {Tω<:Real,Tβ<:Real,N}
        1 <= N <= 3 || throw(ArgumentError(
            "SampledBeta values must have 1 to 3 axes (frequency, mode, polarization)."))
        isempty(omega) && throw(ArgumentError("SampledBeta omega cannot be empty."))
        length(omega) == size(values, 1) || throw(DimensionMismatch(
            "SampledBeta has $(length(omega)) frequencies but $(size(values, 1)) value samples."))
        all(isfinite, omega) || throw(ArgumentError("SampledBeta omega must be finite."))
        all(diff(omega) .> zero(Tω)) || throw(ArgumentError(
            "SampledBeta omega must be strictly increasing (negative-to-positive public order)."))
        all(isfinite, values) || throw(ArgumentError("SampledBeta values must be finite."))
        return new{Tω,Tβ,N}(collect(omega), Array(values))
    end
end

SampledBeta(; omega, values) = SampledBeta(omega, values)

abstract type AbstractOverlap end

"""
    SpatialOverlap(S)

Spatial overlap `S[m,n,p,q]` in `m^-2`. All four axes must have the same
number of spatial modes. Polarization contractions are supplied separately by
the isotropic-material model.
"""
struct SpatialOverlap{T<:Number} <: AbstractOverlap
    values::Array{T,4}

    function SpatialOverlap(values::AbstractArray{T,4};
                            copy_values::Bool=true) where {T<:Number}
        nmode = size(values, 1)
        nmode > 0 || throw(ArgumentError("SpatialOverlap cannot be empty."))
        size(values) == (nmode, nmode, nmode, nmode) || throw(DimensionMismatch(
            "SpatialOverlap must have shape (Nm, Nm, Nm, Nm); got $(size(values))."))
        all(_mmgnlse_isfinite, values) || throw(ArgumentError(
            "SpatialOverlap entries must be finite."))
        stored = copy_values || !(values isa Array{T,4}) ?
                 Array(values) : values
        return new{T}(stored)
    end
end

SpatialOverlap(value::Number) = SpatialOverlap(fill(value, 1, 1, 1, 1))
SpatialOverlap(; S) = SpatialOverlap(S)

"""
    ComponentOverlap(S)

Arbitrary space-polarization overlap tensor with public shape
`Nm x 2 x Nm x 2 x Nm x 2 x Nm x 2`. Unlike `SpatialOverlap`, no analytic
polarization coefficient is applied to this tensor.
"""
struct ComponentOverlap{T<:Number} <: AbstractOverlap
    values::Array{T,8}

    function ComponentOverlap(values::AbstractArray{T,8};
                              copy_values::Bool=true) where {T<:Number}
        nmode = size(values, 1)
        expected = (nmode, 2, nmode, 2, nmode, 2, nmode, 2)
        nmode > 0 || throw(ArgumentError("ComponentOverlap cannot be empty."))
        size(values) == expected || throw(DimensionMismatch(
            "ComponentOverlap must have shape $expected; got $(size(values))."))
        all(_mmgnlse_isfinite, values) || throw(ArgumentError(
            "ComponentOverlap entries must be finite."))
        stored = copy_values || !(values isa Array{T,8}) ?
                 Array(values) : values
        return new{T}(stored)
    end
end

ComponentOverlap(; S) = ComponentOverlap(S)

_mmgnlse_isfinite(value::Real) = isfinite(value)
_mmgnlse_isfinite(value::Complex) = isfinite(real(value)) && isfinite(imag(value))

num_spatial_modes(overlap::SpatialOverlap) = size(overlap.values, 1)
num_spatial_modes(overlap::ComponentOverlap) = size(overlap.values, 1)
has_component_overlap(::SpatialOverlap) = false
has_component_overlap(::ComponentOverlap) = true

abstract type AbstractRamanResponse end

"""Disable the delayed Raman response."""
struct NoRaman <: AbstractRamanResponse end

"""
    AgarwalRaman(; fraction=0.18, tau1=0.0122, tau2=0.0320)

Single-response Agarwal silica model. Time constants are in picoseconds. The
response kernel is generated from these parameters; sampled `h(t)` is not a
public input. The public constructor is `AgarwalRaman`.
"""
struct AgarwalRaman{T<:Real} <: AbstractRamanResponse
    fraction::T
    tau1::T
    tau2::T
end

function AgarwalRaman(; fraction=0.18, tau1=0.0122, tau2=0.0320)
    T = promote_type(typeof(float(fraction)), typeof(float(tau1)), typeof(float(tau2)))
    response = AgarwalRaman{T}(T(fraction), T(tau1), T(tau2))
    return _validate_raman(response)
end

"""
    AnisotropicRaman(; fraction=0.245, fa=0.75, fb=0.21, fc=0.04,
                       tau1=0.0122, tau2=0.0320, taub=0.0960)

Anisotropic silica response used by the WISE model. Time constants are in
picoseconds. `ha` and `hb` are constructed internally from these parameters.
"""
struct AnisotropicRaman{T<:Real} <: AbstractRamanResponse
    fraction::T
    fa::T
    fb::T
    fc::T
    tau1::T
    tau2::T
    taub::T
end

function AnisotropicRaman(; fraction=0.245, fa=0.75, fb=0.21, fc=0.04,
                          tau1=0.0122, tau2=0.0320, taub=0.0960)
    T = promote_type(typeof(float(fraction)), typeof(float(fa)), typeof(float(fb)),
                     typeof(float(fc)), typeof(float(tau1)), typeof(float(tau2)),
                     typeof(float(taub)))
    response = AnisotropicRaman{T}(T(fraction), T(fa), T(fb), T(fc),
                                   T(tau1), T(tau2), T(taub))
    return _validate_raman(response)
end

function _validate_raman(response::AgarwalRaman)
    all(isfinite, (response.fraction, response.tau1, response.tau2)) ||
        throw(ArgumentError("Agarwal Raman parameters must be finite."))
    zero(response.fraction) <= response.fraction <= one(response.fraction) ||
        throw(ArgumentError("Raman fraction must lie in [0, 1]."))
    response.tau1 > 0 && response.tau2 > 0 || throw(ArgumentError(
        "Agarwal Raman time constants must be positive."))
    return response
end

function _validate_raman(response::AnisotropicRaman)
    all(isfinite, (response.fraction, response.fa, response.fb, response.fc,
                   response.tau1, response.tau2, response.taub)) ||
        throw(ArgumentError("Anisotropic Raman parameters must be finite."))
    zero(response.fraction) <= response.fraction <= one(response.fraction) ||
        throw(ArgumentError("Raman fraction must lie in [0, 1]."))
    all(coefficient -> coefficient >= 0, (response.fa, response.fb, response.fc)) ||
        throw(ArgumentError("Anisotropic Raman weights must be nonnegative."))
    isapprox(response.fa + response.fb + response.fc, one(response.fa);
             rtol=8eps(typeof(response.fa)), atol=8eps(typeof(response.fa))) ||
        throw(ArgumentError("Anisotropic Raman weights fa + fb + fc must equal one."))
    response.tau1 > 0 && response.tau2 > 0 && response.taub > 0 ||
        throw(ArgumentError("Anisotropic Raman time constants must be positive."))
    return response
end

raman_fraction(::NoRaman) = 0.0
raman_fraction(response::AbstractRamanResponse) = response.fraction

"""
    raman_response(response, t)

Construct causal Raman response samples on a time vector `t` in picoseconds.
Returns `(ha=..., hb=...)`. The material fraction is intentionally not folded
into either kernel.
"""
function raman_response(::NoRaman, t::AbstractVector{<:Real})
    T = float(eltype(t))
    return (ha=zeros(T, length(t)), hb=zeros(T, length(t)))
end

function raman_response(response::AgarwalRaman, t::AbstractVector{<:Real})
    T = promote_type(typeof(response.tau1), float(eltype(t)))
    ha = zeros(T, length(t))
    prefactor = (response.tau1^2 + response.tau2^2) /
                (response.tau1 * response.tau2^2)
    for index in eachindex(t)
        t[index] < 0 && continue
        time = T(t[index])
        ha[index] = prefactor * exp(-time / response.tau2) * sin(time / response.tau1)
    end
    return (ha=ha, hb=zeros(T, length(t)))
end

function raman_response(response::AnisotropicRaman, t::AbstractVector{<:Real})
    T = promote_type(typeof(response.tau1), float(eltype(t)))
    ha = zeros(T, length(t))
    hb = zeros(T, length(t))
    oscillator_prefactor = (response.tau1^2 + response.tau2^2) /
                           (response.tau1 * response.tau2^2)
    for index in eachindex(t)
        t[index] < 0 && continue
        time = T(t[index])
        oscillator = oscillator_prefactor * exp(-time / response.tau2) *
                     sin(time / response.tau1)
        boson = (2 * response.taub - time) / response.taub^2 * exp(-time / response.taub)
        ha[index] = response.fa * oscillator
        hb[index] = response.fc * oscillator + response.fb * boson
    end
    return (ha=ha, hb=hb)
end

if !isdefined(@__MODULE__, :AbstractMaterial)
    @eval abstract type AbstractMaterial end
end

"""
    Silica(; n2=2.3e-20, raman=AnisotropicRaman())

Silica material metadata. The default Raman response is the anisotropic
silica model and temperature is deliberately not material state. The public
constructor is `Silica`.
"""
struct Silica{T<:Real,R<:AbstractRamanResponse} <: AbstractMaterial
    n2::T
    raman::R
end

function Silica(; n2=2.3e-20, raman=AnisotropicRaman())
    n2_value = float(n2)
    isfinite(n2_value) && n2_value > 0 || throw(ArgumentError(
        "Silica n2 must be finite and positive."))
    raman isa AbstractRamanResponse || throw(ArgumentError(
        "Silica raman must be an AbstractRamanResponse."))
    return Silica{typeof(n2_value),typeof(raman)}(n2_value, raman)
end

material_name(::Silica) = "silica"
default_raman_fraction(material::Silica) = raman_fraction(material.raman)

"""
    profile_specification(value; name=:parameter)

Convert a scalar or positional array into a four-axis lazy profile
specification ordered as `frequency x mode x polarization x z`. Scalars become
`1 x 1 x 1 x 1`; vectors always mean frequency dependence. An `N`-dimensional
array occupies the first `N` axes, so singleton dimensions explicitly select
later axes.
"""
function profile_specification(value::Number; name=:parameter)
    _mmgnlse_isfinite(value) || throw(ArgumentError("$name must be finite."))
    return reshape([value], 1, 1, 1, 1)
end

function profile_specification(values::AbstractArray{T,N}; name=:parameter) where {T<:Number,N}
    1 <= N <= 4 || throw(ArgumentError(
        "$name must be a scalar or an array with at most four axes ordered as $(MMGNLSE_PROFILE_AXES)."))
    isempty(values) && throw(ArgumentError("$name cannot be empty."))
    all(_mmgnlse_isfinite, values) || throw(ArgumentError("$name must be finite."))
    padded_shape = ntuple(axis -> axis <= N ? size(values, axis) : 1, 4)
    return reshape(Array(values), padded_shape)
end

function profile_specification(value; name=:parameter)
    throw(ArgumentError(
        "$name must be a real scalar or numeric positional array; got $(typeof(value))."))
end

"""
    validate_profile_specification(value, target; name=:parameter, allow_sampled_z=true)

Validate a lazy profile against `(Nfrequency, Nmode, Npolarization, Nz)`. Every
axis must be a singleton or match its target. When `allow_sampled_z=true`, a
non-singleton z axis may have any length and represents uniform samples on the
fiber interval.
"""
function validate_profile_specification(value, target::NTuple{4,<:Integer};
                                        name=:parameter, allow_sampled_z::Bool=true)
    all(>(0), target) || throw(ArgumentError("Target profile dimensions must be positive; got $target."))
    specification = profile_specification(value; name=name)
    for axis in 1:4
        source_size = size(specification, axis)
        target_size = Int(target[axis])
        if axis == 4 && allow_sampled_z
            source_size > 0 || throw(DimensionMismatch("$name has an empty z axis."))
        elseif source_size != 1 && source_size != target_size
            throw(DimensionMismatch(
                "$name axis $(MMGNLSE_PROFILE_AXES[axis]) has size $source_size; expected 1 or $target_size."))
        end
    end
    return specification
end

"""
    broadcast_profile(value, target; name=:parameter)

Materialize a lazy profile at the exact four-axis `target`. This operation is
for profiles whose z samples already match the target; longitudinal
interpolation is a solver responsibility.
"""
function broadcast_profile(value, target::NTuple{4,<:Integer}; name=:parameter)
    specification = validate_profile_specification(value, target;
        name=name, allow_sampled_z=false)
    result = similar(specification, target)
    result .= specification
    return result
end

function public_field_shape(domain::MMGNLSEDomain, nmodes::Integer)
    nmodes > 0 || throw(ArgumentError("The number of spatial modes must be positive."))
    !has_dof(domain.dofs, :space) && nmodes != 1 && throw(ArgumentError(
        "A domain without :space must retain exactly one spatial-mode singleton."))
    return (frequency_count(domain), Int(nmodes), polarization_count(domain))
end

function validate_public_field(field::AbstractArray, domain::MMGNLSEDomain,
                               nmodes::Integer; nz::Union{Nothing,Integer}=nothing,
                               name=:field)
    expected = public_field_shape(domain, nmodes)
    if nz === nothing
        ndims(field) == 3 || throw(DimensionMismatch(
            "$name must retain public axes (time, mode, polarization); got $(ndims(field)) dimensions."))
        size(field) == expected || throw(DimensionMismatch(
            "$name must have shape $expected; got $(size(field))."))
    else
        nz > 0 || throw(ArgumentError("The number of saved z values must be positive."))
        expected_solution = (expected..., Int(nz))
        ndims(field) == 4 || throw(DimensionMismatch(
            "$name must retain public axes (time, mode, polarization, z); got $(ndims(field)) dimensions."))
        size(field) == expected_solution || throw(DimensionMismatch(
            "$name must have shape $expected_solution; got $(size(field))."))
    end
    return field
end

"""
    MMGNLSEParameters(domain; length, alpha=0, gain=0, beta=TaylorBeta(0),
                      S, n2, omega0, raman=NoRaman())

Canonical physical parameters for the new facade. `S` is the spatial overlap
in `m^-2` (or an explicitly component-resolved overlap); `n2` and `omega0` are
separate and are never folded into `S`. Raw 4D/8D arrays are defensively
copied by default. Set `copy_overlap=false` to transfer ownership of a large
ordinary `Array` without duplicating it; callers must then not mutate that
array while the parameters are in use. Existing `AbstractOverlap` and CP
objects are always retained without copying.
"""
struct MMGNLSEParameters{D,T,A,G,B,O,N,W,R}
    domain::D
    length::T
    alpha::A
    gain::G
    beta::B
    S::O
    n2::N
    omega0::W
    raman::R
end

function MMGNLSEParameters(domain::MMGNLSEDomain;
                           length,
                           alpha=0,
                           gain=0,
                           beta=TaylorBeta(0),
                           S,
                           n2,
                           omega0,
                           raman=NoRaman(),
                           copy_overlap::Bool=true)
    length_value = float(length)
    n2_value = float(n2)
    omega0_value = float(omega0)
    isfinite(length_value) && length_value > 0 || throw(ArgumentError(
        "MMGNLSEParameters length must be finite and positive (metres)."))
    isfinite(n2_value) && n2_value >= 0 || throw(ArgumentError(
        "MMGNLSEParameters n2 must be finite and nonnegative (m^2/W)."))
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "MMGNLSEParameters omega0 must be finite and positive (rad/ps)."))

    beta_value = _coerce_beta(beta)
    overlap = _coerce_overlap(S; copy_values=copy_overlap)
    raman isa AbstractRamanResponse || throw(ArgumentError(
        "raman must be NoRaman(), AgarwalRaman(...), or AnisotropicRaman(...)."))

    nmodes = num_spatial_modes(overlap)
    !has_dof(domain.dofs, :space) && nmodes != 1 && throw(ArgumentError(
        "A domain without :space cannot use an overlap with $nmodes spatial modes."))
    has_component_overlap(overlap) && !has_dof(domain.dofs, :polarization) &&
        throw(ArgumentError("ComponentOverlap requires the :polarization degree of freedom."))

    _validate_beta(beta_value, domain, nmodes)
    target = (frequency_count(domain), nmodes, polarization_count(domain), 1)
    alpha_specification = validate_profile_specification(alpha, target;
        name=:alpha, allow_sampled_z=true)
    gain_specification = validate_profile_specification(gain, target;
        name=:gain, allow_sampled_z=true)
    eltype(alpha_specification) <: Real || throw(ArgumentError("alpha must be real-valued."))
    eltype(gain_specification) <: Real || throw(ArgumentError("gain must be real-valued."))

    if !has_dof(domain.dofs, :time)
        _beta_iszero(beta_value) || throw(ArgumentError(
            "Dispersion must be zero when :time is inactive."))
        raman isa NoRaman || throw(ArgumentError(
            "Raman response requires the :time degree of freedom."))
    end

    T = promote_type(typeof(length_value), typeof(n2_value), typeof(omega0_value))
    return MMGNLSEParameters{typeof(domain),T,typeof(alpha_specification),
        typeof(gain_specification),typeof(beta_value),typeof(overlap),T,T,typeof(raman)}(
        domain, T(length_value), alpha_specification, gain_specification,
        beta_value, overlap, T(n2_value), T(omega0_value), raman)
end

function MMGNLSEParameters(; domain, kwargs...)
    return MMGNLSEParameters(domain; kwargs...)
end

_coerce_beta(beta::AbstractBeta) = beta
_coerce_beta(beta::Real) = TaylorBeta(beta)
function _coerce_beta(beta)
    throw(ArgumentError("beta must be TaylorBeta, SampledBeta, or a real scalar."))
end

_coerce_overlap(overlap::AbstractOverlap) = overlap
_coerce_overlap(overlap::Number) = SpatialOverlap(overlap)
_coerce_overlap(overlap::AbstractArray{<:Number,4}) = SpatialOverlap(overlap)
_coerce_overlap(overlap::AbstractArray{<:Number,8}) = ComponentOverlap(overlap)
function _coerce_overlap(overlap)
    throw(ArgumentError(
        "S must be a scalar, SpatialOverlap, ComponentOverlap, or a 4D/8D numeric array."))
end

_coerce_overlap(overlap; copy_values::Bool=true) = _coerce_overlap(overlap)
_coerce_overlap(overlap::AbstractArray{<:Number,4}; copy_values::Bool=true) =
    SpatialOverlap(overlap; copy_values)
_coerce_overlap(overlap::AbstractArray{<:Number,8}; copy_values::Bool=true) =
    ComponentOverlap(overlap; copy_values)

_beta_values(beta::TaylorBeta) = beta.coefficients
_beta_values(beta::SampledBeta) = beta.values
_beta_iszero(beta::AbstractBeta) = all(iszero, _beta_values(beta))

function _validate_beta(beta::TaylorBeta, domain::MMGNLSEDomain, nmodes::Integer)
    coefficients = beta.coefficients
    size(coefficients, 2) in (1, nmodes) || throw(DimensionMismatch(
        "TaylorBeta mode axis must have size 1 or $nmodes; got $(size(coefficients, 2))."))
    npolarizations = polarization_count(domain)
    size(coefficients, 3) in (1, npolarizations) || throw(DimensionMismatch(
        "TaylorBeta polarization axis must have size 1 or $npolarizations; got $(size(coefficients, 3))."))
    return beta
end

function _validate_beta(beta::SampledBeta, domain::MMGNLSEDomain, nmodes::Integer)
    values = beta.values
    nfrequency = frequency_count(domain)
    size(values, 1) == nfrequency || throw(DimensionMismatch(
        "SampledBeta frequency axis must have size $nfrequency; got $(size(values, 1))."))
    size(values, 2) in (1, nmodes) || throw(DimensionMismatch(
        "SampledBeta mode axis must have size 1 or $nmodes; got $(size(values, 2))."))
    npolarizations = polarization_count(domain)
    size(values, 3) in (1, npolarizations) || throw(DimensionMismatch(
        "SampledBeta polarization axis must have size 1 or $npolarizations; got $(size(values, 3))."))
    return beta
end

parameter_mode_count(parameters::MMGNLSEParameters) = num_spatial_modes(parameters.S)
parameter_field_shape(parameters::MMGNLSEParameters) =
    public_field_shape(parameters.domain, parameter_mode_count(parameters))
