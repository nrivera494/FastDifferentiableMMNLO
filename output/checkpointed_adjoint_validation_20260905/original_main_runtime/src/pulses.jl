function _centered_time(grid::TimeGrid)
    return centered_time_grid(grid.nt, grid.dt)
end

asech(x) = log((1 + sqrt(1 - x^2)) / x)

function _mmgnlse_pulse_coefficients(coefficients, nmodes::Int,
                                     npolarizations::Int,
                                     mode::Int, polarization::Int)
    values = if coefficients === nothing
        1 <= mode <= nmodes || throw(ArgumentError(
            "mode must lie in 1:$nmodes; got $mode."))
        1 <= polarization <= npolarizations || throw(ArgumentError(
            "polarization must lie in 1:$npolarizations; got $polarization."))
        result = zeros(ComplexF64, nmodes, npolarizations)
        result[mode, polarization] = 1
        result
    elseif coefficients isa AbstractVector
        npolarizations == 1 || throw(DimensionMismatch(
            "Vector coefficients are accepted only for scalar polarization; " *
            "use a mode × polarization matrix."))
        length(coefficients) == nmodes || throw(DimensionMismatch(
            "coefficients must contain $nmodes mode entries."))
        reshape(ComplexF64.(coefficients), nmodes, 1)
    elseif coefficients isa AbstractMatrix
        size(coefficients) == (nmodes, npolarizations) ||
            throw(DimensionMismatch(
                "coefficients must have shape ($nmodes, $npolarizations)."))
        ComplexF64.(coefficients)
    else
        throw(ArgumentError(
            "coefficients must be a mode vector or mode × polarization matrix."))
    end
    all(_mmgnlse_isfinite, values) || throw(ArgumentError(
        "coefficients must be finite."))
    coefficient_norm = norm(values)
    coefficient_norm > 0 || throw(ArgumentError(
        "coefficients must not be identically zero."))
    return values ./ coefficient_norm
end

"""
    gaussian_pulse(grid::TimeGrid, nmodes; peak_power, fwhm, ...)
    gaussian_pulse(domain::MMGNLSEDomain, nmodes; peak_power, fwhm, ...)

Construct a power-normalized Gaussian pulse with public field axes
`(time, mode, polarization)`. `peak_power` is in watts and `fwhm` is the
intensity full width at half maximum in ps.

By default the pulse is launched in mode 1 and x polarization. Pass
`coefficients` as a mode vector for scalar polarization or as a
`mode × polarization` matrix; coefficients are normalized so `peak_power`
remains the total peak power. `time_offset` is in ps and `omega_offset` is an
angular-frequency detuning in rad/ps.
"""
function gaussian_pulse(grid::TimeGrid, nmodes::Integer;
                        peak_power,
                        fwhm,
                        npolarizations::Integer=1,
                        mode::Integer=1,
                        polarization::Integer=1,
                        coefficients=nothing,
                        time_offset=0,
                        omega_offset=0)
    nm = Int(nmodes)
    np = Int(npolarizations)
    nm > 0 || throw(ArgumentError("nmodes must be positive."))
    np in (1, 2) || throw(ArgumentError(
        "npolarizations must be one or two."))
    power = float(peak_power)
    width = float(fwhm)
    offset = float(time_offset)
    detuning = float(omega_offset)
    isfinite(power) && power >= 0 || throw(ArgumentError(
        "peak_power must be finite and nonnegative."))
    isfinite(width) && width > 0 || throw(ArgumentError(
        "fwhm must be finite and positive."))
    isfinite(offset) || throw(ArgumentError("time_offset must be finite."))
    isfinite(detuning) || throw(ArgumentError("omega_offset must be finite."))

    component_weights = _mmgnlse_pulse_coefficients(
        coefficients, nm, np, Int(mode), Int(polarization))
    time = time_axis(grid)
    profile = sqrt(power) .* exp.(-2log(2) .* ((time .- offset) ./ width) .^ 2)
    profile = ComplexF64.(profile .* exp.(-1im .* detuning .* time))
    return reshape(profile, grid.nt, 1, 1) .*
           reshape(component_weights, 1, nm, np)
end

function gaussian_pulse(domain::MMGNLSEDomain, nmodes::Integer; kwargs...)
    has_dof(domain.dofs, :time) || throw(ArgumentError(
        "gaussian_pulse requires the :time degree of freedom."))
    has_dof(domain.dofs, :space) || nmodes == 1 || throw(ArgumentError(
        "nmodes must be one when the domain has no :space degree of freedom."))
    return gaussian_pulse(
        domain.time_grid, nmodes;
        npolarizations=polarization_count(domain), kwargs...)
end

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
