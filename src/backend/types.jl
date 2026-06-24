Base.@kwdef struct Simulation{T}
    lambda0::T
    f0::T = T(2.99792458e-4) / lambda0
    dz::T = T(1000e-6)
    save_period::T = zero(T)
    midx::Vector{Int} = [1]
    scalar::Bool = true
    ellipticity::T = zero(T)
    include_Raman::Bool = true
    gain_model::Int = 0
    pulse_centering::Bool = true
    progress_bar::Bool = false
    step_method::String = "RK4IP"
    cs::Int = 1
    cs_model::Int = 1
    betas::Union{Nothing,Vector{T}} = nothing
    source::Union{Nothing,Array{Complex{T},3}} = nothing
end

Base.@kwdef struct Fiber{T}
    betas::Array{T}
    sr::Array{T,4}
    L0::T = T(2)
    n2::T = T(2.3e-20)
    material::String = "silica"
    mfd::Union{Nothing,T} = nothing
    mm_folder::Union{Nothing,String} = nothing
    fr::T = zero(T)
end

Base.@kwdef struct InitialCondition{T}
    dt::T
    fields::Array{Complex{T},3}
end

Base.@kwdef struct PropagationOutput{T}
    z::Vector{T}
    fields::Array{Complex{T},3}
    dt::T
    betas::Vector{T}
    dz::Vector{T}
    t_delay::Vector{T}
    ode_sol::Any = nothing
    linear_gain::Any = nothing
end

Base.@kwdef struct AdjointOutput{T}
    z::Vector{T}
    lambdaw::Array{Complex{T},3}
    dt::T
    frame::Symbol = :lab
end

Base.@kwdef struct SRSKInfo{T}
    sk::Vector{T}
    sk_indices::Matrix{Int}
    sra::Vector{T}
    sra_indices::Matrix{Int}
    sra_indices34::Matrix{Int}
    srb::Vector{T}
    srb_indices::Matrix{Int}
    srb_indices34::Matrix{Int}
end

last_fields(ic::InitialCondition) = ic.fields[:, :, end]

function as_3d_fields(fields::AbstractMatrix{<:Complex})
    out = Array{eltype(fields),3}(undef, size(fields, 1), size(fields, 2), 1)
    out[:, :, 1] = fields
    return out
end

function as_3d_fields(fields::Array{<:Complex,3})
    return fields
end
