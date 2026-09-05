Base.@kwdef struct PulseTrajectory{T,O}
    output::O
    problem::Any
    solver_cache::Any = nothing
end

PulseTrajectory(; output, problem, solver_cache=nothing) =
    PulseTrajectory{eltype(output.z),typeof(output)}(
        ; output=output, problem=problem, solver_cache=solver_cache)

function field(sol::PulseTrajectory; z=:final, domain=:time)
    idx = _z_index(sol, z)
    fields_t = sol.output.fields[:, :, idx]
    if domain === :time
        return fields_t
    elseif domain === :frequency
        return inverse_fft(fields_t, dims=1)
    else
        error("domain must be :time or :frequency")
    end
end

function spectrum(sol::PulseTrajectory; z=:final, shifted::Bool=true, sum_modes::Bool=false)
    aω = field(sol; z=z, domain=:frequency)
    shifted && (aω = fftshift(aω, 1))
    s = abs2.(aω)
    return sum_modes ? vec(sum(s; dims=2)) : s
end

function z_spectrum(sol::PulseTrajectory; shifted::Bool=true, sum_modes::Bool=false)
    fields_t = sol.output.fields
    nt, nm, nz = size(fields_t)
    out = sum_modes ? zeros(real(eltype(fields_t)), nt, nz) :
                      zeros(real(eltype(fields_t)), nt, nm, nz)
    for iz in 1:nz
        aω = inverse_fft(fields_t[:, :, iz], dims=1)
        shifted && (aω = fftshift(aω, 1))
        s = abs2.(aω)
        if sum_modes
            out[:, iz] .= vec(sum(s; dims=2))
        else
            out[:, :, iz] .= s
        end
    end
    return out
end

function photon_spectrum(sol::PulseTrajectory; z=:final, shifted::Bool=true,
                         sum_modes::Bool=false, normalization::Symbol=:bin)
    fields_t = field(sol; z=z, domain=:time)
    return photon_spectrum(fields_t, sol.output.dt,
                                   _sim(sol).f0; shifted=shifted,
                                   sum_modes=sum_modes,
                                   normalization=normalization)
end

function photon_number(sol::PulseTrajectory, filter; z=:final, shifted::Bool=false)
    fields_t = field(sol; z=z, domain=:time)
    return filtered_photon_number(fields_t, sol.output.dt, _sim(sol).f0,
                                          filter; shifted=shifted)
end

function compressed_tensor(sol::PulseTrajectory)
    cache = sol.solver_cache
    cache === nothing && return nothing
    hasproperty(cache, :compression) || return nothing
    return cache.compression.tensor
end

function compression_rank(sol::PulseTrajectory)
    cache = sol.solver_cache
    cache === nothing && return nothing
    hasproperty(cache, :compression) || return nothing
    return cache.compression.rank
end

function compression_error(sol::PulseTrajectory)
    cache = sol.solver_cache
    cache === nothing && return nothing
    hasproperty(cache, :compression) || return nothing
    return cache.compression.error
end

function modal_energy(sol::PulseTrajectory; z=:final)
    fields_t = field(sol; z=z, domain=:time)
    return vec(sum(abs2, fields_t; dims=1)) .* sol.output.dt ./ 1e3
end

function _z_index(sol::PulseTrajectory, z)
    z === :initial && return 1
    z === :final && return size(sol.output.fields, 3)
    if z isa Integer
        return Int(z)
    end
    _, idx = findmin(abs.(sol.output.z .- z))
    return idx
end

_sim(sol::PulseTrajectory) = backend_objects(sol.problem).sim
