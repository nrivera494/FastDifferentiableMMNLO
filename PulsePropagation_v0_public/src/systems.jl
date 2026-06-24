abstract type AbstractMaterial end
struct Silica <: AbstractMaterial end
Base.@kwdef struct AgarwalRaman{T} <: AbstractMaterial
    fraction::T = T(0.18)
    tau1_fs::T = T(12.2)
    tau2_fs::T = T(32.0)
    temperature_K::T = T(300.0)
end
AgarwalRaman(; fraction=0.18, tau1_fs=12.2, tau2_fs=32.0, temperature_K=300.0) =
    AgarwalRaman{Float64}(; fraction=Float64(fraction),
                           tau1_fs=Float64(tau1_fs),
                           tau2_fs=Float64(tau2_fs),
                           temperature_K=Float64(temperature_K))
struct Chalcogenide <: AbstractMaterial end

material_name(::Silica) = "silica"
material_name(::AgarwalRaman) = "agarwal"
material_name(::Chalcogenide) = "chalcogenide"
default_raman_fraction(::Silica) = 0.0
default_raman_fraction(m::AgarwalRaman) = m.fraction
default_raman_fraction(::Chalcogenide) = 0.0

function material_from_string(s::AbstractString)
    s == "silica" && return Silica()
    s == "agarwal" && return AgarwalRaman()
    s == "chalcogenide" && return Chalcogenide()
    error("Unsupported material string: $s")
end

abstract type AbstractDispersion end
Base.@kwdef struct TaylorDispersion{T} <: AbstractDispersion
    betas::Array{T}
end
TaylorDispersion(; betas) = TaylorDispersion{eltype(betas)}(; betas=Array(betas))

Base.@kwdef struct GridDispersion{T} <: AbstractDispersion
    betas::Array{T}
end
GridDispersion(; betas) = GridDispersion{eltype(betas)}(; betas=Array(betas))

abstract type AbstractGeometry end
Base.@kwdef struct ModeOverlapTensor{T} <: AbstractGeometry
    sr::Array{T,4}
end
ModeOverlapTensor(; sr) = ModeOverlapTensor{eltype(sr)}(; sr=Array(sr))

Base.@kwdef struct EffectiveArea{T} <: AbstractGeometry
    mfd::Union{Nothing,T} = nothing
end
EffectiveArea(; mfd=nothing) =
    EffectiveArea{mfd === nothing ? Float64 : typeof(float(mfd))}(; mfd=mfd)

abstract type AbstractPropagationSystem end

Base.@kwdef struct PassiveFiber{T,M<:AbstractMaterial,D<:AbstractDispersion,G<:AbstractGeometry} <: AbstractPropagationSystem
    length::T
    lambda0::T
    f0::T = T(2.99792458e-4) / lambda0
    material::M = Silica()
    dispersion::D
    geometry::G
    n2::T = T(2.3e-20)
    raman_fraction::T = zero(T)
    source_path::Union{Nothing,String} = nothing
end

function PassiveFiber(; length, lambda0, f0=2.99792458e-4 / lambda0,
                      material=Silica(), dispersion=nothing, geometry=nothing,
                      betas=nothing, sr=nothing, mfd=nothing,
                      n2=2.3e-20, raman_fraction=nothing, source_path=nothing)
    if dispersion === nothing
        betas === nothing && error("PassiveFiber requires either dispersion or betas.")
        dispersion = dispersion_from_betas(Array(betas))
    end
    if geometry === nothing
        geometry = sr === nothing ? EffectiveArea(; mfd=mfd) :
                                    ModeOverlapTensor(; sr=Array(sr))
    end
    rf = raman_fraction === nothing ? default_raman_fraction(material) : raman_fraction
    T = promote_type(typeof(float(length)), typeof(float(lambda0)),
                     typeof(float(f0)), typeof(float(n2)),
                     typeof(float(rf)))
    return PassiveFiber{T,typeof(material),typeof(dispersion),typeof(geometry)}(
        ; length=T(length), lambda0=T(lambda0), f0=T(f0),
        material=material, dispersion=dispersion, geometry=geometry,
        n2=T(n2), raman_fraction=T(rf), source_path=source_path)
end

function _read_csv_matrix(path::AbstractString)
    isfile(path) || error("Could not find CSV file: $path")
    return Matrix{Float64}(readdlm(path, ','))
end

function _read_csv_overlap_tensor(path::AbstractString, modes)
    isfile(path) || error("Could not find CSV file: $path")
    rows = Matrix{Float64}(readdlm(path, ',', Float64; skipstart=1))
    selected = collect(Int, modes)
    max_idx = maximum(Int.(rows[:, 1:4]))
    nfull = max(max_idx, maximum(selected))
    full = zeros(Float64, nfull, nfull, nfull, nfull)
    @inbounds for r in axes(rows, 1)
        i, j, k, l = Int.(rows[r, 1:4])
        full[i, j, k, l] = rows[r, 5]
    end
    return full[selected, selected, selected, selected]
end

function load_fiber_system(folder::AbstractString; modes,
                           length,
                           lambda0,
                           material=Silica(),
                           betas_filename="betas.csv",
                           s_tensors_filename="S_tensors_10modes.csv",
                           scalar::Bool=true,
                           n2=2.3e-20,
                           raman_fraction=nothing)
    selected = collect(Int, modes)
    beta_all = _read_csv_matrix(joinpath(folder, betas_filename))
    size(beta_all, 2) >= maximum(selected) ||
        error("Betas file has $(size(beta_all, 2)) modes, but modes=$selected was requested.")
    betas_selected = beta_all[:, selected]
    sr_selected = _read_csv_overlap_tensor(joinpath(folder, s_tensors_filename), selected)
    f0 = 2.99792458e-4 / lambda0
    system = PassiveFiber(; length=length, lambda0=lambda0, f0=f0,
                          material=material, betas=betas_selected,
                          sr=sr_selected, n2=n2,
                          raman_fraction=raman_fraction,
                          source_path=folder)
    dofs = scalar ? ModalField(modes) : PolarizedModalField(modes)
    return system, dofs
end

Base.@kwdef struct GainFiber{T,M<:AbstractMaterial,D<:AbstractDispersion,G<:AbstractGeometry,R} <: AbstractPropagationSystem
    passive::PassiveFiber{T,M,D,G}
    gain_model::Int
    gain_rate_eqn::R = nothing
end

function GainFiber(; passive, gain_model, gain_rate_eqn=nothing)
    T = typeof(passive.length)
    return GainFiber{T,typeof(passive.material),typeof(passive.dispersion),
                     typeof(passive.geometry),typeof(gain_rate_eqn)}(
        ; passive=passive, gain_model=gain_model, gain_rate_eqn=gain_rate_eqn)
end

function dispersion_from_betas(betas)
    size(betas, 1) < 20 && return TaylorDispersion(; betas=betas)
    return GridDispersion(; betas=betas)
end

betas(d::TaylorDispersion) = d.betas
betas(d::GridDispersion) = d.betas
sr(g::ModeOverlapTensor) = g.sr
sr(g::EffectiveArea) = reshape([1.0], 1, 1, 1, 1)
