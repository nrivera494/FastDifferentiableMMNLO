struct TimeGrid{T}
    nt::Int
    dt::T

    function TimeGrid{T}(nt::Integer, dt::T) where {T<:Real}
        nt > 0 || throw(ArgumentError("nt must be positive."))
        isfinite(dt) && dt > 0 || throw(ArgumentError(
            "dt must be finite and positive."))
        return new{T}(Int(nt), dt)
    end
end

TimeGrid{T}(; nt::Integer, dt) where {T<:Real} = TimeGrid{T}(nt, T(dt))

function TimeGrid(; Nt::Integer, window)
    Nt > 0 || throw(ArgumentError("Nt must be positive."))
    window_value = Float64(window)
    isfinite(window_value) && window_value > 0 || throw(ArgumentError(
        "window must be finite and positive."))
    return TimeGrid{Float64}(Int(Nt), window_value / Int(Nt))
end

TimeGrid(Nt::Int, window::T) where {T<:Real} = TimeGrid(; Nt=Nt, window=window)
TimeGrid(Nt::Integer, window::Real) = TimeGrid(; Nt=Nt, window=window)

time_window(g::TimeGrid) = g.nt * g.dt

time_axis(g::TimeGrid) = centered_time_grid(g.nt, g.dt)
"""Centered angular-frequency detuning axis in rad/ps."""
frequency_axis(g::TimeGrid) =
    2π .* frequency_grid_THz(g.nt, g.dt; shifted=true)

"""Centered absolute-wavelength axis in metres for carrier `omega0` in rad/ps."""
function wavelength_axis(g::TimeGrid, omega0)
    omega0_value = float(omega0)
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "omega0 must be finite and positive."))
    omega_absolute = omega0_value .+ frequency_axis(g)
    c_m_per_ps = 2.99792458e-4
    return ifelse.(omega_absolute .> 0,
                   2π * c_m_per_ps ./ omega_absolute,
                   oftype.(omega_absolute, Inf))
end

"""Centered photon-number weights for carrier `omega0` in rad/ps."""
function photon_weights(g::TimeGrid, omega0)
    omega0_value = float(omega0)
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "omega0 must be finite and positive."))
    return photon_bin_weights(g.nt, g.dt, omega0_value / (2π); shifted=true)
end
