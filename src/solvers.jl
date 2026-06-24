abstract type AbstractStepControl end
Base.@kwdef struct FixedStep{T} <: AbstractStepControl
    dz::T
end
FixedStep(; dz) = FixedStep{typeof(float(dz))}(; dz=float(dz))

Base.@kwdef struct AdaptiveStep{T} <: AbstractStepControl
    initial_dz::T
    max_dz::T
    min_dz::T = zero(T)
    rtol::T = T(1e-6)
    atol::T = T(1e-9)
end
function AdaptiveStep(; initial_dz, max_dz, min_dz=0.0, rtol=1e-6, atol=1e-9)
    T = promote_type(typeof(float(initial_dz)), typeof(float(max_dz)),
                     typeof(float(min_dz)), typeof(float(rtol)),
                     typeof(float(atol)))
    return AdaptiveStep{T}(; initial_dz=T(initial_dz), max_dz=T(max_dz),
                           min_dz=T(min_dz), rtol=T(rtol), atol=T(atol))
end

abstract type AbstractSaveSpec end
Base.@kwdef struct SaveAt{T} <: AbstractSaveSpec
    z::Vector{T}
end
SaveAt(z) = SaveAt{Float64}(; z=Float64.(collect(z)))
SaveAt(z::Real) =
    error("SaveAt expects a vector/range of saved z positions. For a scalar save spacing, use SaveEvery(; dz=$z), or use SaveAt(collect(range(0.0, L; length=N_z))).")

Base.@kwdef struct SaveEvery{T} <: AbstractSaveSpec
    dz::T
end
SaveEvery(dz) = SaveEvery{typeof(float(dz))}(; dz=float(dz))
SaveEvery(; dz) = SaveEvery(dz)

Base.@kwdef struct CPCompression
    tensor::Any = nothing
    target_error::Float64 = 1e-4
    initial_rank::Int = 16
    max_rank::Int = 128
    rank_growth::Int = 2
    maxiter::Int = 120
    tol::Float64 = 1e-8
    ridge::Float64 = 1e-8
    check_every::Int = 5
    seed::Int = 3
    verbose::Bool = false
end

Base.@kwdef struct CachedCPRHS
    fftw_threads::Int = 1
    fftw_flags::Symbol = :MEASURE
    blas_threads::Union{Nothing,Int} = nothing
end

Base.@kwdef struct FixedRK4IPCPRHS
    fftw_threads::Int = 1
    fftw_flags::Symbol = :MEASURE
    blas_threads::Union{Nothing,Int} = nothing
end

abstract type AbstractExecutionBackend end
struct CPUBackend <: AbstractExecutionBackend end
Base.@kwdef struct CUDABackend <: AbstractExecutionBackend
    device::Any = nothing
    synchronize::Bool = true
end

Base.@kwdef struct RK4IPSolver{T,S<:AbstractStepControl,V<:AbstractSaveSpec,C,H,G,B<:AbstractExecutionBackend}
    stepping::S
    saveat::V
    compression::C = nothing
    cp_rhs::H = nothing
    linear_gain::G = nothing
    backend::B = CPUBackend()
    reltol::T = T(1e-10)
    abstol::T = T(1e-10)
end

function RK4IPSolver(; stepping, saveat, compression=nothing,
                     cp_rhs=nothing, linear_gain=nothing, backend=CPUBackend(),
                     reltol=1e-10, abstol=1e-10)
    return RK4IPSolver{Float64,typeof(stepping),typeof(saveat),typeof(compression),typeof(cp_rhs),typeof(linear_gain),typeof(backend)}(
        ; stepping=stepping, saveat=saveat,
        compression=compression, cp_rhs=cp_rhs, linear_gain=linear_gain,
        backend=backend,
        reltol=Float64(reltol), abstol=Float64(abstol))
end

Base.@kwdef struct PulsePropagationProblem{S<:PulseState,M<:PropagationModel,V<:RK4IPSolver}
    initial_state::S
    model::M
    solver::V
end

function PulsePropagationProblem(; initial_state=nothing, model=nothing,
                                 solver,
                                 grid=nothing, system=nothing,
                                 dofs=SingleMode(),
                                 terms=PropagationTerms(Dispersion(), Kerr()),
                                 fields=nothing,
                                 normalization=PowerNormalized(),
                                 domain=TimeDomain(), z=0.0)
    if initial_state === nothing
        grid === nothing && error("PulsePropagationProblem requires grid when initial_state is not supplied.")
        fields === nothing && error("PulsePropagationProblem requires fields when initial_state is not supplied.")
        initial_state = PulseState(grid, fields; basis=dofs, normalization=normalization,
                                   domain=domain, z=z)
    end
    if model === nothing
        system === nothing && error("PulsePropagationProblem requires system when model is not supplied.")
        model = PropagationModel(; system=system, dofs=dofs, terms=terms)
    end
    return PulsePropagationProblem{typeof(initial_state),typeof(model),typeof(solver)}(
        ; initial_state=initial_state, model=model, solver=solver)
end

struct Adjoint end
struct AutomaticDifferentiation
    backend::Symbol
end
AutomaticDifferentiation() = AutomaticDifferentiation(:zygote)
struct InitialField end
