Base.@kwdef struct TimeGrid{T}
    nt::Int
    dt::T
end

function TimeGrid(; Nt::Integer, window)
    return TimeGrid{Float64}(; nt=Int(Nt), dt=Float64(window) / Int(Nt))
end

TimeGrid(Nt::Int, window::T) where {T<:Real} = TimeGrid(; Nt=Nt, window=window)
TimeGrid(Nt::Integer, window::Real) = TimeGrid(; Nt=Nt, window=window)

time_window(g::TimeGrid) = g.nt * g.dt

time_axis(g::TimeGrid) = centered_time_grid(g.nt, g.dt)
frequency_axis(g::TimeGrid; shifted::Bool=true) =
    frequency_grid_THz(g.nt, g.dt; shifted=shifted)
wavelength_axis(g::TimeGrid, f0; shifted::Bool=true) =
    wavelength_grid_nm(g.nt, g.dt, f0; shifted=shifted)
photon_weights(g::TimeGrid, f0; shifted::Bool=true) =
    photon_bin_weights(g.nt, g.dt, f0; shifted=shifted)
