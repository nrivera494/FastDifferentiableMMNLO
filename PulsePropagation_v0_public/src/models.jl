abstract type AbstractModelTerm end
struct DispersionTerm <: AbstractModelTerm end
struct KerrTerm <: AbstractModelTerm end
Base.@kwdef struct RamanTerm <: AbstractModelTerm
    model::Symbol = :system
end
Base.@kwdef struct GainTerm <: AbstractModelTerm
    model::Int = 0
end

const Dispersion = DispersionTerm
const Kerr = KerrTerm
Raman(model::Symbol=:system) = RamanTerm(; model=model)
Gain(model::Int=0) = GainTerm(; model=model)

PropagationTerms(terms::AbstractModelTerm...) = AbstractModelTerm[terms...]
PropagationTerms(terms::AbstractVector{<:AbstractModelTerm}) = AbstractModelTerm[terms...]

Base.@kwdef struct PropagationModel{S<:AbstractPropagationSystem}
    system::S
    basis::AbstractBasis = SingleMode()
    terms::Vector{AbstractModelTerm} = AbstractModelTerm[DispersionTerm(), KerrTerm()]
end

function PropagationModel(; system, dofs=SingleMode(), basis=dofs,
                          terms=PropagationTerms(Dispersion(), Kerr()))
    terms2 = PropagationTerms(terms)
    if default_raman_fraction(system isa GainFiber ? system.passive.material : system.material) > 0 &&
       !any(t -> t isa RamanTerm, terms2)
        push!(terms2, Raman(:system))
    end
    return PropagationModel{typeof(system)}(; system=system, basis=basis,
                                            terms=terms2)
end

has_term(model::PropagationModel, ::Type{T}) where {T<:AbstractModelTerm} =
    any(t -> t isa T, model.terms)

function _raman_term(model::PropagationModel)
    for term in model.terms
        term isa RamanTerm && return term
    end
    return nothing
end
