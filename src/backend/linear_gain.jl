Base.@kwdef struct LinearGain{T}
    values::Array{T}
    axis::Symbol = :scalar
    z::Union{Nothing,Vector{T}} = nothing
end

LinearGain(g::Real) = LinearGain{typeof(float(g))}(; values=reshape([float(g)], 1, 1),
                                                    axis=:scalar, z=nothing)
LinearGain(values::AbstractArray; axis::Symbol=:auto, z=nothing) =
    LinearGain{Float64}(; values=Float64.(Array(values)), axis=axis,
                        z=z === nothing ? nothing : Float64.(collect(z)))

function _canonical_linear_gain(linear_gain, nt::Integer, nm::Integer, L::T;
                                zsave=nothing) where {T}
    linear_gain === nothing && return nothing
    if linear_gain isa LinearGain
        values = T.(linear_gain.values)
        z = linear_gain.z === nothing ? nothing : T.(linear_gain.z)
        axis = linear_gain.axis
    elseif linear_gain isa Real
        return LinearGain{T}(; values=reshape([T(linear_gain)], 1, 1),
                             axis=:scalar, z=nothing)
    else
        values = T.(Array(linear_gain))
        z = nothing
        axis = :auto
    end

    axis === :auto && (axis = _infer_linear_gain_axis(values, nt))
    axis in (:scalar, :frequency, :z) ||
        error("linear_gain axis must be :scalar, :frequency, :z, or :auto.")

    if axis === :scalar
        length(values) == 1 ||
            error("Scalar linear_gain must contain exactly one value.")
        values = reshape([only(vec(values))], 1, 1)
    elseif axis === :frequency
        values = _linear_gain_frequency_values(values, nt, nm)
    else
        values = _linear_gain_z_values(values, nm)
        if z === nothing
            n = size(values, 1)
            n >= 2 || error("z-dependent linear_gain needs at least two samples.")
            z = collect(range(zero(T), L; length=n))
        else
            length(z) == size(values, 1) ||
                error("linear_gain z grid length must match the number of z samples.")
            first(z) == zero(T) ||
                error("linear_gain z grid must start at 0.")
            last(z) == L ||
                error("linear_gain z grid must end at the fiber length.")
            all(diff(z) .> zero(T)) ||
                error("linear_gain z grid must be strictly increasing.")
        end
        zsave !== nothing && length(zsave) == size(values, 1) &&
            maximum(abs.(T.(zsave) .- z)) <= sqrt(eps(T)) * max(one(T), L)
    end
    return LinearGain{T}(; values=values, axis=axis, z=z)
end

function _infer_linear_gain_axis(values::AbstractArray, nt::Integer)
    length(values) == 1 && return :scalar
    size(values, 1) == nt && return :frequency
    return :z
end

function _linear_gain_frequency_values(values::AbstractArray{T}, nt::Integer,
                                       nm::Integer) where {T}
    if ndims(values) == 1
        length(values) == nt ||
            error("Frequency-dependent linear_gain vector must have length Nt.")
        return reshape(collect(values), nt, 1)
    elseif ndims(values) == 2
        size(values, 1) == nt ||
            error("Frequency-dependent linear_gain matrix must have Nt rows.")
        size(values, 2) in (1, nm) ||
            error("Frequency-dependent linear_gain matrix must have 1 or Nm columns.")
        return Array(values)
    else
        error("Frequency-dependent linear_gain must be a vector or matrix.")
    end
end

function _linear_gain_z_values(values::AbstractArray{T}, nm::Integer) where {T}
    if ndims(values) == 1
        return reshape(collect(values), length(values), 1)
    elseif ndims(values) == 2
        size(values, 2) in (1, nm) ||
            error("z-dependent linear_gain matrix must have 1 or Nm columns.")
        return Array(values)
    else
        error("z-dependent linear_gain must be a vector or matrix.")
    end
end

function _linear_gain_rate(gain::Union{Nothing,LinearGain{T}}, z::T,
                           nt::Integer, nm::Integer) where {T}
    gain === nothing && return zeros(T, nt, nm)
    if gain.axis === :scalar
        return fill(gain.values[1], nt, nm)
    elseif gain.axis === :frequency
        return repeat(gain.values, 1, nm ÷ size(gain.values, 2))
    else
        vals = _linear_gain_z_interp(gain, z)
        return repeat(reshape(vals, 1, :), nt, nm ÷ length(vals))
    end
end

function _linear_gain_z_interp(gain::LinearGain{T}, z::T) where {T}
    zg = gain.z::Vector{T}
    z <= zg[1] && return vec(gain.values[1, :])
    z >= zg[end] && return vec(gain.values[end, :])
    q = searchsortedlast(zg, z)
    q = clamp(q, 1, length(zg) - 1)
    θ = (z - zg[q]) / (zg[q + 1] - zg[q])
    return vec((one(T) - θ) .* gain.values[q, :] .+ θ .* gain.values[q + 1, :])
end

function _linear_gain_power_integral(gain::Union{Nothing,LinearGain{T}}, z::T,
                                     nt::Integer, nm::Integer) where {T}
    gain === nothing && return zeros(T, nt, nm)
    if gain.axis === :scalar
        return fill(gain.values[1] * z, nt, nm)
    elseif gain.axis === :frequency
        return repeat(gain.values .* z, 1, nm ÷ size(gain.values, 2))
    else
        vals = _linear_gain_z_integral(gain, z)
        return repeat(reshape(vals, 1, :), nt, nm ÷ length(vals))
    end
end

function _linear_gain_z_integral(gain::LinearGain{T}, z::T) where {T}
    zg = gain.z::Vector{T}
    zc = clamp(z, zg[1], zg[end])
    out = zeros(T, size(gain.values, 2))
    zc <= zg[1] && return out
    qmax = searchsortedlast(zg, zc)
    qmax = clamp(qmax, 1, length(zg) - 1)
    @inbounds for q in 1:qmax-1
        dz = zg[q + 1] - zg[q]
        out .+= (gain.values[q, :] .+ gain.values[q + 1, :]) .* (dz / 2)
    end
    zc > zg[qmax] || return out
    g0 = gain.values[qmax, :]
    g1 = _linear_gain_z_interp(gain, zc)
    out .+= (g0 .+ g1) .* ((zc - zg[qmax]) / 2)
    return out
end

function _linear_exponent(d_op::AbstractMatrix{Complex{T}}, z::T,
                          gain::Union{Nothing,LinearGain{T}}) where {T}
    nt, nm = size(d_op)
    return d_op .* z .+ Complex{T}.(_linear_gain_power_integral(gain, z, nt, nm) ./ 2)
end

function _linear_gain_matches(a, b, nt::Integer, nm::Integer, L::T) where {T}
    ac = _canonical_linear_gain(a, nt, nm, L)
    bc = _canonical_linear_gain(b, nt, nm, L)
    ac === nothing && bc === nothing && return true
    (ac === nothing || bc === nothing) && return false
    ac.axis == bc.axis || return false
    size(ac.values) == size(bc.values) || return false
    maximum(abs.(ac.values .- bc.values)) <= sqrt(eps(T)) * max(one(T), maximum(abs.(ac.values)))
end
