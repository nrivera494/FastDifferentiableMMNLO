function _centered_time(grid::TimeGrid)
    return centered_time_grid(grid.nt, grid.dt)
end

asech(x) = log((1 + sqrt(1 - x^2)) / x)

function sech_pulse(grid::TimeGrid; peak_power, fwhm, dofs=SingleModeField(),
                    coefficients=nothing,
                    transform::Tuple{String,<:Real}=("ifft", 0.0),
                    time_offset=0.0)
    nm = num_components(dofs)
    t = _centered_time(grid)
    T0 = fwhm / (2asech(1 / sqrt(2)))
    profile = sqrt(float(peak_power)) ./ cosh.((t .- time_offset) ./ T0)
    kind, shift = transform
    shift != 0 && (profile = profile .* _phase_factor(kind, shift, t))
    modal = _modal_coefficients(coefficients, nm)
    fields = zeros(ComplexF64, grid.nt, nm, 1)
    for m in 1:nm
        fields[:, m, 1] .= profile .* modal[m]
    end
    return fields
end

function gaussian_pulse(grid::TimeGrid; fwhm, dofs=SingleModeField(),
                        peak_power=nothing, total_energy=nothing,
                        coefficients=nothing,
                        transform::Tuple{String,<:Real}=("ifft", 0.0),
                        time_offset=0.0)
    (peak_power === nothing) == (total_energy === nothing) &&
        error("Provide exactly one of peak_power or total_energy.")
    nm = num_components(dofs)
    t = _centered_time(grid)
    sigma = fwhm / (2sqrt(log(2)))
    shape = exp.(-((t .- time_offset).^2) ./ (2sigma^2))
    amp = if peak_power !== nothing
        sqrt(float(peak_power))
    else
        sqrt(float(total_energy) * 1e3 / (sum(abs2, shape) * grid.dt))
    end
    profile = amp .* shape
    kind, shift = transform
    shift != 0 && (profile = profile .* _phase_factor(kind, shift, t))
    modal = _modal_coefficients(coefficients, nm)
    fields = zeros(ComplexF64, grid.nt, nm, 1)
    for m in 1:nm
        fields[:, m, 1] .= profile .* modal[m]
    end
    return fields
end

function _modal_coefficients(coefficients, nm::Integer)
    coeffs = coefficients === nothing ? ones(ComplexF64, nm) : ComplexF64.(coefficients)
    length(coeffs) <= nm || error("coefficients length exceeds number of components.")
    modal = zeros(ComplexF64, nm)
    modal[1:length(coeffs)] .= coeffs
    norm(modal) > 0 || error("coefficients must not be all zero.")
    modal ./= norm(modal)
    return modal
end

function _phase_factor(kind::AbstractString, shift, t)
    if kind == "ifft"
        return exp.(-1im * 2π * shift .* t)
    elseif kind == "fft"
        return exp.(1im * 2π * shift .* t)
    end
    error("transform kind must be \"ifft\" or \"fft\".")
end
