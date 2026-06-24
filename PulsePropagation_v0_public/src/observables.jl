abstract type AbstractObservable end

Base.@kwdef struct SpectralPhotonNumber{F}
    filter::F
    modes::Any = :all
    shifted::Bool = false
end

function value(obs::SpectralPhotonNumber, sol::PulseTrajectory; z=:final)
    filter = _filter_for_modes(obs.filter, field(sol; z=z, domain=:time), obs.modes)
    return photon_number(sol, filter; z=z, shifted=obs.shifted)
end

function terminal_condition(obs::SpectralPhotonNumber, sol::PulseTrajectory; z=:final)
    fields_t = field(sol; z=z, domain=:time)
    uω = inverse_fft(fields_t, dims=1)
    weights = photon_bin_weights(size(fields_t, 1), sol.output.dt,
                                         _sim(sol).f0; shifted=obs.shifted)
    filt = _filter_for_modes(obs.filter, fields_t, obs.modes)
    return filt .* weights .* uω
end

function _filter_for_modes(filter, fields, modes)
    nt, nm = size(fields)
    f = Float64.(filter)
    if ndims(f) == 1
        out = zeros(Float64, nt, nm)
        selected = modes === :all ? (1:nm) : collect(Int, modes)
        for m in selected
            out[:, m] .= f
        end
        return out
    end
    return f
end

Base.@kwdef struct FilterEnergy{F} <: AbstractObservable
    filter::F
    domain::Symbol = :frequency
    shifted::Bool = false
    modes::Any = :all
end

Base.@kwdef struct BinEnergy <: AbstractObservable
    index::Any
    domain::Symbol = :frequency
    shifted::Bool = false
end

function BinEnergy(; bin::Integer, mode::Integer=1, domain::Symbol=:frequency,
                   shifted::Bool=false)
    return BinEnergy((Int(bin), Int(mode)), domain, shifted)
end

Base.@kwdef struct QuadratureObservable{MT} <: AbstractObservable
    M::MT
    domain::Symbol = :frequency
    shifted::Bool = false
    component::Any = nothing
    weights::Any = nothing
end

Base.@kwdef struct ProjectedEnergy{UT,FT} <: AbstractObservable
    U::UT
    filter::FT = nothing
    domain::Symbol = :frequency
    shifted::Bool = false
end

Base.@kwdef struct ProjectedPhotonNumber{UT,WT,FT} <: AbstractObservable
    U::UT
    weights::WT
    filter::FT = nothing
    domain::Symbol = :frequency
    shifted::Bool = false
end

Base.@kwdef struct TemporalMoment <: AbstractObservable
    order::Int = 1
    center::Any = 0.0
    normalized::Bool = true
    modes::Any = :all
end

Base.@kwdef struct SpectralMoment <: AbstractObservable
    order::Int = 1
    center::Any = 0.0
    normalized::Bool = true
    shifted::Bool = false
    modes::Any = :all
end

Base.@kwdef struct RatioObservable{N,D} <: AbstractObservable
    numerator::N
    denominator::D
end

RatioObservable(numerator, denominator) =
    RatioObservable{typeof(numerator),typeof(denominator)}(numerator, denominator)

TemporalCentroid(; modes=:all) =
    TemporalMoment(; order=1, center=0.0, normalized=true, modes=modes)
SpectralCentroid(; shifted::Bool=false, modes=:all) =
    SpectralMoment(; order=1, center=0.0, normalized=true, shifted=shifted, modes=modes)

function value(obs::FilterEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    f = _observable_filter(obs.filter, size(u), obs.modes)
    return real(sum(f .* abs2.(u)))
end

function terminal_condition(obs::FilterEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return terminal_condition(obs, u)
end

function value(obs::FilterEnergy, u::AbstractArray)
    f = _observable_filter(obs.filter, size(u), obs.modes)
    return real(sum(f .* abs2.(u)))
end

function terminal_condition(obs::FilterEnergy, u::AbstractArray)
    f = _observable_filter(obs.filter, size(u), obs.modes)
    return f .* u
end

function value(obs::BinEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return value(obs, u)
end

function terminal_condition(obs::BinEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return terminal_condition(obs, u)
end

function value(obs::BinEnergy, u::AbstractArray)
    return real(abs2(u[obs.index...]))
end

function terminal_condition(obs::BinEnergy, u::AbstractArray)
    g = zeros(eltype(u), size(u))
    g[obs.index...] = u[obs.index...]
    return g
end

function value(obs::QuadratureObservable, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return value(obs, u)
end

function terminal_condition(obs::QuadratureObservable, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return terminal_condition(obs, u)
end

function value(obs::QuadratureObservable, u::AbstractArray)
    M = _quadrature_matrix(obs.M, length(u))
    x = M * vec(u) .+ conj.(M) * conj.(vec(u))
    xr = real.(x)
    obs.component !== nothing && return xr[Int(obs.component)]
    obs.weights !== nothing && return real(sum(_quadrature_weights(obs.weights, length(xr)) .* xr))
    length(xr) == 1 && return xr[1]
    return xr
end

function terminal_condition(obs::QuadratureObservable, u::AbstractArray)
    M = _quadrature_matrix(obs.M, length(u))
    row = if obs.component !== nothing
        vec(conj.(M[Int(obs.component), :]))
    elseif obs.weights !== nothing
        vec(conj.(transpose(_quadrature_weights(obs.weights, size(M, 1))) * M))
    elseif size(M, 1) == 1
        vec(conj.(M[1, :]))
    else
        error("QuadratureObservable terminal_condition requires component or weights for multi-row M.")
    end
    return reshape(row, size(u))
end

function value(obs::ProjectedEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return value(obs, u)
end

function terminal_condition(obs::ProjectedEnergy, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return terminal_condition(obs, u)
end

function value(obs::ProjectedEnergy, u::AbstractArray)
    up = _project_field(obs.U, u)
    f = obs.filter === nothing ? ones(real(eltype(up)), size(up)) :
                                  _observable_filter(obs.filter, size(up), :all)
    return real(sum(f .* abs2.(up)))
end

function terminal_condition(obs::ProjectedEnergy, u::AbstractArray)
    up = _project_field(obs.U, u)
    f = obs.filter === nothing ? ones(real(eltype(up)), size(up)) :
                                  _observable_filter(obs.filter, size(up), :all)
    return _project_adjoint(obs.U, f .* up)
end

function value(obs::ProjectedPhotonNumber, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return value(obs, u)
end

function terminal_condition(obs::ProjectedPhotonNumber, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=obs.domain, shifted=obs.shifted)
    return terminal_condition(obs, u)
end

function value(obs::ProjectedPhotonNumber, u::AbstractArray)
    up = _project_field(obs.U, u)
    f = _projected_photon_filter(obs.weights, obs.filter, size(up))
    return real(sum(f .* abs2.(up)))
end

function terminal_condition(obs::ProjectedPhotonNumber, u::AbstractArray)
    up = _project_field(obs.U, u)
    f = _projected_photon_filter(obs.weights, obs.filter, size(up))
    return _project_adjoint(obs.U, f .* up)
end

function value(obs::TemporalMoment, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=:time, shifted=false)
    return value(obs, u; axis=time_axis(sol.problem.initial_state.grid))
end

function terminal_condition(obs::TemporalMoment, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=:time, shifted=false)
    return terminal_condition(obs, u; axis=time_axis(sol.problem.initial_state.grid))
end

function value(obs::TemporalMoment, u::AbstractArray; axis)
    return _moment_value(u, axis, obs.order, obs.center, obs.normalized, obs.modes)
end

function terminal_condition(obs::TemporalMoment, u::AbstractArray; axis)
    return _moment_terminal(u, axis, obs.order, obs.center, obs.normalized, obs.modes)
end

function value(obs::SpectralMoment, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=:frequency, shifted=obs.shifted)
    axis = frequency_axis(sol.problem.initial_state.grid; shifted=obs.shifted)
    return value(obs, u; axis=axis)
end

function terminal_condition(obs::SpectralMoment, sol::PulseTrajectory; z=:final)
    u = _observable_field(sol; z=z, domain=:frequency, shifted=obs.shifted)
    axis = frequency_axis(sol.problem.initial_state.grid; shifted=obs.shifted)
    return terminal_condition(obs, u; axis=axis)
end

function value(obs::SpectralMoment, u::AbstractArray; axis)
    return _moment_value(u, axis, obs.order, obs.center, obs.normalized, obs.modes)
end

function terminal_condition(obs::SpectralMoment, u::AbstractArray; axis)
    return _moment_terminal(u, axis, obs.order, obs.center, obs.normalized, obs.modes)
end

function value(obs::RatioObservable, sol::PulseTrajectory; z=:final)
    den = value(obs.denominator, sol; z=z)
    den == 0 && error("RatioObservable denominator is zero.")
    return value(obs.numerator, sol; z=z) / den
end

function terminal_condition(obs::RatioObservable, sol::PulseTrajectory; z=:final)
    num = value(obs.numerator, sol; z=z)
    den = value(obs.denominator, sol; z=z)
    den == 0 && error("RatioObservable denominator is zero.")
    gnum = terminal_condition(obs.numerator, sol; z=z)
    gden = terminal_condition(obs.denominator, sol; z=z)
    size(gnum) == size(gden) ||
        error("RatioObservable terminal conditions must have matching sizes.")
    return (den .* gnum .- num .* gden) ./ den^2
end

function value(obs::RatioObservable, u::AbstractArray; kwargs...)
    den = value(obs.denominator, u; kwargs...)
    den == 0 && error("RatioObservable denominator is zero.")
    return value(obs.numerator, u; kwargs...) / den
end

function terminal_condition(obs::RatioObservable, u::AbstractArray; kwargs...)
    num = value(obs.numerator, u; kwargs...)
    den = value(obs.denominator, u; kwargs...)
    den == 0 && error("RatioObservable denominator is zero.")
    gnum = terminal_condition(obs.numerator, u; kwargs...)
    gden = terminal_condition(obs.denominator, u; kwargs...)
    size(gnum) == size(gden) ||
        error("RatioObservable terminal conditions must have matching sizes.")
    return (den .* gnum .- num .* gden) ./ den^2
end

function _observable_field(sol::PulseTrajectory; z=:final, domain::Symbol, shifted::Bool=false)
    if domain === :time
        shifted && error("shifted=true is only meaningful for frequency-domain observables.")
        return field(sol; z=z, domain=:time)
    elseif domain === :frequency
        u = field(sol; z=z, domain=:frequency)
        return shifted ? fftshift(u, 1) : u
    else
        error("domain must be :time or :frequency.")
    end
end

function _observable_filter(filter, dims::Tuple, modes)
    T = Float64
    selected = _selected_modes(modes, dims[end])
    f = filter === nothing ? ones(T, dims) : Float64.(filter)
    if ndims(f) == 0
        out = fill(T(f), dims)
    elseif size(f) == dims
        out = T.(f)
    elseif length(dims) == 2 && ndims(f) == 1 && length(f) == dims[1]
        out = repeat(reshape(f, dims[1], 1), 1, dims[2])
    elseif length(dims) == 2 && ndims(f) == 1 && length(f) == dims[2]
        out = repeat(reshape(f, 1, dims[2]), dims[1], 1)
    else
        out = f .+ zeros(T, dims)
    end
    if length(dims) == 2 && modes !== :all
        mode_weights = [m in selected ? one(T) : zero(T) for m in 1:dims[2]]
        mask = repeat(reshape(mode_weights, 1, dims[2]), dims[1], 1)
        out = out .* mask
    end
    return out
end

_selected_modes(::Colon, nm) = 1:nm
_selected_modes(modes, nm) = modes === :all ? (1:nm) : collect(Int, modes)

function _quadrature_matrix(M, n)
    if ndims(M) == 1 && length(M) == n
        return reshape(collect(M), 1, n)
    end
    A = Matrix(M)
    size(A, 2) == n && return A
    error("Quadrature matrix has $(size(A, 2)) columns but field has length $n.")
end

_quadrature_weights(weights, n) = length(weights) == n ? collect(weights) :
    error("Quadrature weights length must match number of quadrature rows.")

function _project_field(U, u)
    A = Matrix(U)
    if ndims(u) == 2 && size(A, 2) == size(u, 2)
        return u * transpose(A)
    elseif size(A, 2) == length(u)
        return reshape(A * vec(u), size(A, 1), 1)
    end
    error("Projection matrix dimensions are incompatible with field size $(size(u)).")
end

function _project_adjoint(U, gp)
    A = Matrix(U)
    if ndims(gp) == 2 && size(A, 1) == size(gp, 2)
        return gp * conj.(A)
    elseif size(A, 1) == length(gp)
        return reshape(adjoint(A) * vec(gp), size(A, 2), 1)
    end
    error("Projection adjoint dimensions are incompatible with projected field size $(size(gp)).")
end

function _projected_photon_filter(weights, filter, dims::Tuple)
    length(dims) == 2 ||
        error("ProjectedPhotonNumber expects a 2D frequency/bin-by-projection field.")
    length(weights) == dims[1] ||
        error("ProjectedPhotonNumber weights length $(length(weights)) must match first dimension $(dims[1]).")
    base = filter === nothing ? ones(Float64, dims) :
                                _observable_filter(filter, dims, :all)
    return base .* repeat(reshape(Float64.(weights), dims[1], 1), 1, dims[2])
end

function _moment_value(u, axis, order, center, normalized, modes)
    selected = _selected_modes(modes, size(u, 2))
    return _moment_value_selected(u, axis, order, center, normalized, selected)
end

function _moment_value_selected(u, axis, order, center, normalized, selected)
    vals = map(selected) do m
        intensity = abs2.(u[:, m])
        E = sum(intensity)
        c = center === :centroid ? sum(axis .* intensity) / E : Float64(center)
        μ = sum(((axis .- c) .^ order) .* intensity)
        normalized ? μ / E : μ
    end
    return length(vals) == 1 ? vals[1] : vals
end

function _moment_terminal(u, axis, order, center, normalized, modes)
    selected = _selected_modes(modes, size(u, 2))
    g = zeros(eltype(u), size(u))
    for m in selected
        intensity = abs2.(u[:, m])
        E = sum(intensity)
        c = center === :centroid ? sum(axis .* intensity) / E : Float64(center)
        w = (axis .- c) .^ order
        if normalized
            μ = sum(w .* intensity) / E
            g[:, m] .= ((w .- μ) ./ E) .* u[:, m]
        else
            g[:, m] .= w .* u[:, m]
        end
    end
    return g
end
