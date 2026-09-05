abstract type AbstractBasis end
struct SingleMode <: AbstractBasis end
SingleModeField() = SingleMode()

Base.@kwdef struct ModalBasis <: AbstractBasis
    modes::Vector{Int}
end
ModalBasis(modes) = ModalBasis(; modes=collect(Int, modes))
ModalField(; modes) = ModalBasis(modes)
ModalField(modes) = ModalBasis(modes)

Base.@kwdef struct PolarizedModalBasis <: AbstractBasis
    spatial_modes::Vector{Int}
    polarizations::Vector{Symbol} = [:x, :y]
end
PolarizedModalBasis(spatial_modes; polarizations=[:x, :y]) =
    PolarizedModalBasis(; spatial_modes=collect(Int, spatial_modes),
                        polarizations=collect(Symbol, polarizations))
PolarizedModalField(; spatial_modes, polarizations=[:x, :y]) =
    PolarizedModalBasis(spatial_modes; polarizations=polarizations)
PolarizedModalField(spatial_modes; polarizations=[:x, :y]) =
    PolarizedModalBasis(spatial_modes; polarizations=polarizations)

num_components(::SingleMode) = 1
num_components(b::ModalBasis) = length(b.modes)
num_components(b::PolarizedModalBasis) = length(b.spatial_modes) * length(b.polarizations)

abstract type AbstractNormalization end
struct PowerNormalized <: AbstractNormalization end
struct PhotonNormalized <: AbstractNormalization end

abstract type AbstractDomain end
struct TimeDomain <: AbstractDomain end
struct FrequencyDomain <: AbstractDomain end

Base.@kwdef struct PulseState{T,B<:AbstractBasis,N<:AbstractNormalization,D<:AbstractDomain}
    grid::TimeGrid{T}
    fields::Array{Complex{T},3}
    basis::B = SingleMode()
    normalization::N = PowerNormalized()
    domain::D = TimeDomain()
    z::T = zero(T)
end

function PulseState(grid::TimeGrid{T}, fields;
                    basis::AbstractBasis=SingleMode(),
                    normalization::AbstractNormalization=PowerNormalized(),
                    domain::AbstractDomain=TimeDomain(),
                    z=zero(T)) where {T}
    f3 = as_3d_fields(Complex{T}.(fields))
    return PulseState{T,typeof(basis),typeof(normalization),typeof(domain)}(
        ; grid=grid, fields=f3, basis=basis, normalization=normalization,
        domain=domain, z=T(z))
end

last_fields(s::PulseState) = s.fields[:, :, end]
