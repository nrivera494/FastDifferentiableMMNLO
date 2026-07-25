"""
Observable methods for `MMGNLSESolution`.

Frequency-domain fields and every terminal condition returned here use the
public centered, negative-to-positive angular-frequency ordering.  Terminal
conditions are Wirtinger derivatives with respect to the conjugate of that
centered spectral field and can therefore be passed directly to
`solve_adjoint`.
"""

function _mmgnlse_reconstruct_spatiotemporal(modal_field,
                                              properties::FiberProperties)
    ndims(modal_field) in (3, 4) || throw(DimensionMismatch(
        "A modal field must have axes (time/frequency, mode, polarization) " *
        "with an optional trajectory axis."))
    nt, nm, np = size(modal_field, 1), size(modal_field, 2), size(modal_field, 3)
    modes = mode_fields(properties)
    size(modes, 3) == nm || throw(DimensionMismatch(
        "The solution has $nm modes but the fiber properties contain " *
        "$(size(modes, 3)) mode fields."))
    ny, nx = size(modes, 1), size(modes, 2)
    ntrajectory = ndims(modal_field) == 4 ? size(modal_field, 4) : 1
    T = promote_type(eltype(modal_field), eltype(modes))
    result = Array{T}(undef, nt, nx, ny, np, ntrajectory)
    flattened_modes = reshape(modes, ny * nx, nm)
    for trajectory in 1:ntrajectory, polarization in 1:np
        amplitudes = ndims(modal_field) == 4 ?
            @view(modal_field[:, :, polarization, trajectory]) :
            @view(modal_field[:, :, polarization])
        reconstructed = amplitudes * transpose(flattened_modes)
        result[:, :, :, polarization, trajectory] .=
            permutedims(reshape(reconstructed, nt, ny, nx), (1, 3, 2))
    end
    return ndims(modal_field) == 4 ? result : dropdims(result; dims=5)
end

"""
    get_spatiotemporal_field(solution, properties; z=:final, domain=:time)

Reconstruct a polarization-resolved field from its modal amplitudes and the
carrier mode profiles in `properties`. The returned axes are
`time/frequency × x × y × polarization`; stochastic output adds trajectory
last. `domain=:frequency` uses the centered angular-frequency ordering.
"""
function get_spatiotemporal_field(
    solution::Union{MMGNLSESolution,MMGNLSEAdjointSolution},
    properties::FiberProperties; z=:final, domain::Symbol=:time)
    modal = if domain === :time
        get_temporal_field(solution; z)
    elseif domain === :frequency
        get_spectral_field(solution; z)
    else
        throw(ArgumentError("domain must be :time or :frequency."))
    end
    return _mmgnlse_reconstruct_spatiotemporal(modal, properties)
end

function get_spatiotemporal_field(
    solution::MMGNLSEStochasticSolution,
    properties::FiberProperties; z=:final, domain::Symbol=:time,
    trajectory=:all)
    modal = if domain === :time
        get_temporal_field(solution; z, trajectory)
    elseif domain === :frequency
        get_spectral_field(solution; z, trajectory)
    else
        throw(ArgumentError("domain must be :time or :frequency."))
    end
    return _mmgnlse_reconstruct_spatiotemporal(modal, properties)
end

function _mmgnlse_spatial_sample(parameters; time, wavelength)
    (time === nothing) != (wavelength === nothing) || throw(ArgumentError(
        "Specify exactly one of time or wavelength."))
    grid = parameters.domain.time_grid
    grid isa TimeGrid || throw(ArgumentError(
        "Spatial fields at a time or wavelength require the :time degree of freedom."))

    if time !== nothing
        time isa Real && isfinite(time) || throw(ArgumentError(
            "time must be a finite real value in ps."))
        _, index = findmin(abs.(time_axis(grid) .- time))
        return :time, index
    end

    wavelength isa Real && isfinite(wavelength) && wavelength > 0 ||
        throw(ArgumentError(
            "wavelength must be a finite positive real value in metres."))
    wavelengths = wavelength_axis(grid, parameters.omega0)
    _, index = findmin(abs.(wavelengths .- wavelength))
    return :frequency, index
end

function _mmgnlse_polarization_index(polarization, npolarizations::Integer;
                                     required::Bool)
    if polarization === nothing
        required && npolarizations > 1 && throw(ArgumentError(
            "polarization must be specified as :x, :y, 1, or 2 for a " *
            "polarization-resolved complex spatial field."))
        return 1
    end

    index = if polarization isa Integer
        Int(polarization)
    elseif polarization isa Symbol || polarization isa AbstractString
        symbol = Symbol(lowercase(String(polarization)))
        symbol === :x ? 1 : symbol === :y ? 2 : throw(ArgumentError(
            "polarization must be :x, :y, 1, or 2."))
    else
        throw(ArgumentError("polarization must be :x, :y, 1, or 2."))
    end
    1 <= index <= npolarizations || throw(ArgumentError(
        "polarization $index is not present in a field with " *
        "$npolarizations polarization component(s)."))
    return index
end

function _mmgnlse_spatial_polarizations(polarization,
                                        npolarizations::Integer)
    polarization === nothing && return 1:npolarizations
    if polarization isa Symbol || polarization isa AbstractString
        Symbol(lowercase(String(polarization))) === :all &&
            return 1:npolarizations
    end
    return (_mmgnlse_polarization_index(
        polarization, npolarizations; required=false),)
end

function _mmgnlse_reconstruct_spatial_sample(modal, sample::Integer,
                                             polarization::Integer,
                                             properties::FiberProperties)
    ndims(modal) >= 3 || throw(DimensionMismatch(
        "A modal field must retain frequency/time, mode, and polarization axes."))
    nmodes = size(modal, 2)
    modes = mode_fields(properties)
    size(modes, 3) == nmodes || throw(DimensionMismatch(
        "The solution has $nmodes modes but the fiber properties contain " *
        "$(size(modes, 3)) mode fields."))

    trailing_indices = ntuple(_ -> Colon(), ndims(modal) - 3)
    amplitudes = @view modal[sample, :, polarization, trailing_indices...]
    trailing_shape = size(amplitudes)[2:end]
    ny, nx = size(modes, 1), size(modes, 2)
    flattened_modes = reshape(modes, ny * nx, nmodes)
    reconstructed = flattened_modes * reshape(amplitudes, nmodes, :)
    shaped = reshape(reconstructed, ny, nx, trailing_shape...)
    permutation = (2, 1, ntuple(axis -> axis + 2, length(trailing_shape))...)
    return permutedims(shaped, permutation)
end

function _mmgnlse_get_spatial_field(
    solution, properties::FiberProperties; z, time, wavelength,
    polarization, units::Symbol, trajectory=nothing)
    domain, sample = _mmgnlse_spatial_sample(
        solution.parameters; time, wavelength)
    modal = if solution isa MMGNLSEStochasticSolution
        domain === :time ?
            get_temporal_field(solution; z, trajectory, units) :
            get_spectral_field(solution; z, trajectory, units)
    else
        domain === :time ?
            get_temporal_field(solution; z, units) :
            get_spectral_field(solution; z, units)
    end
    selected = _mmgnlse_polarization_index(
        polarization, size(modal, 3); required=true)
    return _mmgnlse_reconstruct_spatial_sample(
        modal, sample, selected, properties)
end

"""
    get_spatial_field(solution, properties;
                      time=nothing, wavelength=nothing,
                      polarization=nothing, z=:final, units=:power)

Reconstruct the complex transverse field at one time in ps or one absolute
wavelength in metres. Exactly one of `time` and `wavelength` must be supplied;
the nearest public grid sample is used. The result has axes `x × y`. With
`z=:all`, saved z is retained after y. Stochastic output similarly retains any
selected z and trajectory axes.

For a scalar-polarization field, `polarization` may be omitted. For a linear
x/y field it must be `:x`, `:y`, `1`, or `2`. The carrier mode profiles stored
in `properties` are used at both time and wavelength samples.
"""
function get_spatial_field(
    solution::MMGNLSESolution, properties::FiberProperties;
    time=nothing, wavelength=nothing, polarization=nothing,
    z=:final, units::Symbol=:power)
    return _mmgnlse_get_spatial_field(
        solution, properties; z, time, wavelength, polarization, units)
end

function get_spatial_field(
    solution::MMGNLSEAdjointSolution, properties::FiberProperties;
    time=nothing, wavelength=nothing, polarization=nothing,
    z=:initial, units::Symbol=:power)
    return _mmgnlse_get_spatial_field(
        solution, properties; z, time, wavelength, polarization, units)
end

function get_spatial_field(
    solution::MMGNLSEStochasticSolution, properties::FiberProperties;
    time=nothing, wavelength=nothing, polarization=nothing,
    z=:final, trajectory=:all, units::Symbol=:power)
    return _mmgnlse_get_spatial_field(
        solution, properties;
        z, time, wavelength, polarization, units, trajectory)
end

function _mmgnlse_spectral_spatial_intensity(modal,
                                             properties::FiberProperties,
                                             polarization)
    ndims(modal) >= 3 || throw(DimensionMismatch(
        "A modal field must retain frequency, mode, and polarization axes."))
    nfrequency, nmodes, npolarizations = size(modal, 1), size(modal, 2),
        size(modal, 3)
    modes = mode_fields(properties)
    size(modes, 3) == nmodes || throw(DimensionMismatch(
        "The solution has $nmodes modes but the fiber properties contain " *
        "$(size(modes, 3)) mode fields."))

    trailing_shape = size(modal)[4:end]
    nbatch = isempty(trailing_shape) ? 1 : prod(trailing_shape)
    modal_batches = reshape(modal, nfrequency, nmodes, npolarizations, nbatch)
    ny, nx = size(modes, 1), size(modes, 2)
    flattened_modes = reshape(modes, ny * nx, nmodes)
    conjugated_modes = conj.(flattened_modes)
    T = promote_type(eltype(modal), eltype(modes))
    R = typeof(abs2(zero(T)))
    intensity = zeros(R, ny * nx, nbatch)
    selected = _mmgnlse_spatial_polarizations(
        polarization, npolarizations)

    for batch in 1:nbatch, pol in selected
        amplitudes = @view modal_batches[:, :, pol, batch]
        coherence = adjoint(amplitudes) * amplitudes
        contribution = real.(sum(
            conjugated_modes .*
            (flattened_modes * transpose(coherence));
            dims=2))
        intensity[:, batch] .+= vec(contribution)
    end
    intensity .= max.(intensity, zero(R))

    shaped = reshape(intensity, ny, nx, trailing_shape...)
    permutation = (2, 1, ntuple(axis -> axis + 2, length(trailing_shape))...)
    return permutedims(shaped, permutation)
end

function _mmgnlse_get_spectrally_summed_spatial_intensity(
    solution, properties::FiberProperties; z, polarization,
    units::Symbol, trajectory=nothing)
    modal = solution isa MMGNLSEStochasticSolution ?
        get_spectral_field(solution; z, trajectory, units) :
        get_spectral_field(solution; z, units)
    return _mmgnlse_spectral_spatial_intensity(
        modal, properties, polarization)
end

"""
    get_spectrally_summed_spatial_intensity(
        solution, properties;
        polarization=nothing, z=:final, units=:power)

Return `sum(abs2(field(x, y, omega, polarization)), omega)` with axes `x × y`.
If a linear x/y polarization is present and `polarization` is omitted (or
`:all`), the result is summed incoherently over both polarizations, as for a
polarization-independent broadband detector. Pass `:x`, `:y`, `1`, or `2` to
select one polarization. Saved-z and stochastic trajectory axes, when
requested, follow y.

The sum uses the package's centered spectral field and its discrete FFT
normalization without an additional frequency-bin factor. The carrier mode
profiles stored in `properties` are used at every frequency.
"""
function get_spectrally_summed_spatial_intensity(
    solution::MMGNLSESolution, properties::FiberProperties;
    polarization=nothing, z=:final, units::Symbol=:power)
    return _mmgnlse_get_spectrally_summed_spatial_intensity(
        solution, properties; z, polarization, units)
end

function get_spectrally_summed_spatial_intensity(
    solution::MMGNLSEAdjointSolution, properties::FiberProperties;
    polarization=nothing, z=:initial, units::Symbol=:power)
    return _mmgnlse_get_spectrally_summed_spatial_intensity(
        solution, properties; z, polarization, units)
end

function get_spectrally_summed_spatial_intensity(
    solution::MMGNLSEStochasticSolution, properties::FiberProperties;
    polarization=nothing, z=:final, trajectory=:all,
    units::Symbol=:power)
    return _mmgnlse_get_spectrally_summed_spatial_intensity(
        solution, properties; z, polarization, units, trajectory)
end

function _mmgnlse_observable_field(solution::MMGNLSESolution;
                                    z=:final, domain::Symbol)
    if domain === :frequency
        return get_spectral_field(solution; z=z)
    elseif domain === :time
        return get_temporal_field(solution; z=z)
    end
    throw(ArgumentError("observable domain must be :time or :frequency."))
end

function _mmgnlse_observable_field(solution::MMGNLSEStochasticSolution;
                                    domain::Symbol)
    if domain === :frequency
        return get_spectral_field(solution)
    elseif domain === :time
        return get_temporal_field(solution)
    end
    throw(ArgumentError("observable domain must be :time or :frequency."))
end

function _mmgnlse_selected_modes(modes, nmodes::Integer)
    selected = if modes === :all || modes isa Colon
        collect(1:Int(nmodes))
    elseif modes isa Integer
        [Int(modes)]
    else
        try
            Int.(collect(modes))
        catch
            throw(ArgumentError(
                "modes must be :all, :, an integer, or a collection of integers."))
        end
    end
    isempty(selected) && throw(ArgumentError("modes cannot be empty."))
    all(mode -> 1 <= mode <= nmodes, selected) || throw(ArgumentError(
        "selected modes $selected are not contained in 1:$nmodes."))
    length(unique(selected)) == length(selected) || throw(ArgumentError(
        "modes must not contain duplicate indices."))
    return selected
end

function _mmgnlse_observable_filter(filter, target::NTuple{3,<:Integer}, modes;
                                     name=:filter)
    specification = if filter === nothing
        reshape([1.0], 1, 1, 1)
    elseif filter isa Real
        reshape([float(filter)], 1, 1, 1)
    elseif filter isa AbstractArray{<:Real}
        1 <= ndims(filter) <= 3 || throw(ArgumentError(
            "$name must have at most three positional axes " *
            "(frequency, mode, polarization)."))
        isempty(filter) && throw(ArgumentError("$name cannot be empty."))
        reshape(float.(Array(filter)), size(filter)...,
                ntuple(_ -> 1, 3 - ndims(filter))...)
    else
        throw(ArgumentError("$name must be a real scalar or real positional array."))
    end
    all(isfinite, specification) || throw(ArgumentError(
        "$name entries must be finite."))
    for axis in 1:3
        size(specification, axis) in (1, target[axis]) || throw(DimensionMismatch(
            "$name axis $axis has size $(size(specification, axis)); " *
            "expected 1 or $(target[axis])."))
    end
    result = similar(specification, target)
    result .= specification

    selected = _mmgnlse_selected_modes(modes, target[2])
    if length(selected) != target[2]
        keep = falses(target[2])
        keep[selected] .= true
        result .*= reshape(keep, 1, target[2], 1)
    end
    return result
end

function _mmgnlse_time_gradient_to_centered_spectral(gradient_t)
    nt = size(gradient_t, 1)
    return fftshift(nt .* ifft(gradient_t, 1), 1)
end

function _mmgnlse_observable_axis(
    solution::Union{MMGNLSESolution,MMGNLSEStochasticSolution},
    domain::Symbol)
    has_dof(solution.parameters.domain.dofs, :time) || throw(ArgumentError(
        "temporal and spectral moments require the :time degree of freedom."))
    grid = solution.parameters.domain.time_grid
    if domain === :time
        return time_axis(grid)
    elseif domain === :frequency
        return fftshift(ifft_frequency_grid(grid.nt, grid.dt))
    end
    throw(ArgumentError("observable domain must be :time or :frequency."))
end

function _mmgnlse_terminal_in_centered_spectrum(gradient, domain::Symbol)
    domain === :frequency && return gradient
    domain === :time && return _mmgnlse_time_gradient_to_centered_spectral(gradient)
    throw(ArgumentError("observable domain must be :time or :frequency."))
end

function _mmgnlse_trajectory_values(f, field)
    ndims(field) == 4 || throw(DimensionMismatch(
        "An ensemble observable field must have axes " *
        "(time/frequency, mode, polarization, trajectory)."))
    return [f(@view(field[:, :, :, trajectory]))
            for trajectory in axes(field, 4)]
end

function _mmgnlse_spectral_photon_number(observable::SpectralPhotonNumber,
                                          spectrum, parameters)
    filter = _mmgnlse_observable_filter(
        observable.filter, size(spectrum), observable.modes)
    weights = reshape(_mmgnlse_photon_weights(parameters), :, 1, 1)
    return real(sum(filter .* weights .* abs2.(spectrum)))
end

function value(observable::SpectralPhotonNumber,
               solution::MMGNLSESolution; z=:final)
    spectrum = get_spectral_field(solution; z=z)
    return _mmgnlse_spectral_photon_number(
        observable, spectrum, solution.parameters)
end

function value(observable::SpectralPhotonNumber,
               solution::MMGNLSEStochasticSolution)
    spectrum = get_spectral_field(solution)
    target = ntuple(axis -> size(spectrum, axis), 3)
    filter = _mmgnlse_observable_filter(
        observable.filter, target, observable.modes)
    weights = reshape(_mmgnlse_photon_weights(solution.parameters), :, 1, 1)
    return _mmgnlse_trajectory_values(spectrum) do trajectory
        real(sum(filter .* weights .* abs2.(trajectory)))
    end
end

function terminal_condition(observable::SpectralPhotonNumber,
                            solution::MMGNLSESolution; z=:final)
    spectrum = get_spectral_field(solution; z=z)
    filter = _mmgnlse_observable_filter(
        observable.filter, size(spectrum), observable.modes)
    weights = reshape(_mmgnlse_photon_weights(solution.parameters), :, 1, 1)
    return filter .* weights .* spectrum
end

function value(observable::FilterEnergy,
               solution::MMGNLSESolution; z=:final)
    field = _mmgnlse_observable_field(
        solution; z=z, domain=observable.domain)
    filter = _mmgnlse_observable_filter(
        observable.filter, size(field), observable.modes)
    return real(sum(filter .* abs2.(field)))
end

function value(observable::FilterEnergy,
               solution::MMGNLSEStochasticSolution)
    field = _mmgnlse_observable_field(
        solution; domain=observable.domain)
    target = ntuple(axis -> size(field, axis), 3)
    filter = _mmgnlse_observable_filter(
        observable.filter, target, observable.modes)
    return _mmgnlse_trajectory_values(field) do trajectory
        real(sum(filter .* abs2.(trajectory)))
    end
end

function terminal_condition(observable::FilterEnergy,
                            solution::MMGNLSESolution; z=:final)
    field = _mmgnlse_observable_field(
        solution; z=z, domain=observable.domain)
    filter = _mmgnlse_observable_filter(
        observable.filter, size(field), observable.modes)
    gradient = filter .* field
    return _mmgnlse_terminal_in_centered_spectrum(gradient, observable.domain)
end

function _mmgnlse_bin_index(index, target::NTuple{3,<:Integer})
    entries = if index isa CartesianIndex
        collect(Tuple(index))
    elseif index isa Integer
        [Int(index)]
    else
        try
            Int.(collect(index))
        catch
            throw(ArgumentError(
                "BinEnergy index must contain frequency/time, mode, and polarization indices."))
        end
    end
    1 <= length(entries) <= 3 || throw(ArgumentError(
        "BinEnergy index must have one to three entries."))
    append!(entries, ones(Int, 3 - length(entries)))
    all(axis -> 1 <= entries[axis] <= target[axis], 1:3) ||
        throw(BoundsError(target, Tuple(entries)))
    return Tuple(entries)
end

function value(observable::BinEnergy,
               solution::MMGNLSESolution; z=:final)
    field = _mmgnlse_observable_field(
        solution; z=z, domain=observable.domain)
    index = _mmgnlse_bin_index(observable.index, size(field))
    return real(abs2(field[index...]))
end

function value(observable::BinEnergy,
               solution::MMGNLSEStochasticSolution)
    field = _mmgnlse_observable_field(
        solution; domain=observable.domain)
    target = ntuple(axis -> size(field, axis), 3)
    index = _mmgnlse_bin_index(observable.index, target)
    return _mmgnlse_trajectory_values(field) do trajectory
        real(abs2(trajectory[index...]))
    end
end

function terminal_condition(observable::BinEnergy,
                            solution::MMGNLSESolution; z=:final)
    field = _mmgnlse_observable_field(
        solution; z=z, domain=observable.domain)
    index = _mmgnlse_bin_index(observable.index, size(field))
    gradient = zeros(eltype(field), size(field))
    gradient[index...] = field[index...]
    return _mmgnlse_terminal_in_centered_spectrum(gradient, observable.domain)
end

function _mmgnlse_moment_value_and_gradient(field, axis, order::Integer,
                                             center, normalized::Bool, modes)
    order >= 0 || throw(ArgumentError("moment order must be nonnegative."))
    length(axis) == size(field, 1) || throw(DimensionMismatch(
        "moment axis length must match the first field axis."))
    selected = _mmgnlse_selected_modes(modes, size(field, 2))
    mask = zeros(Float64, 1, size(field, 2), 1)
    mask[:, selected, :] .= 1
    intensity = abs2.(field) .* mask
    energy = real(sum(intensity))
    energy > 0 || throw(DomainError(energy,
        "a moment is undefined for zero energy in the selected modes."))

    axis3 = reshape(axis, :, 1, 1)
    center_value = if center === :centroid
        real(sum(axis3 .* intensity)) / energy
    elseif center isa Real && isfinite(center)
        float(center)
    else
        throw(ArgumentError("moment center must be finite or :centroid."))
    end
    offset = axis3 .- center_value
    weights = offset .^ order
    numerator = real(sum(weights .* intensity))
    result = normalized ? numerator / energy : numerator

    coefficient = copy(weights)
    if center === :centroid && order > 0
        derivative_center = -order * real(sum((offset .^ (order - 1)) .* intensity))
        coefficient .+= derivative_center .* offset ./ energy
    end
    if normalized
        coefficient .= (coefficient .- result) ./ energy
    end
    gradient = coefficient .* field .* mask
    return result, gradient
end

function value(observable::TemporalMoment,
               solution::MMGNLSESolution; z=:final)
    field = get_temporal_field(solution; z=z)
    axis = _mmgnlse_observable_axis(solution, :time)
    result, _ = _mmgnlse_moment_value_and_gradient(
        field, axis, observable.order, observable.center,
        observable.normalized, observable.modes)
    return result
end

function value(observable::TemporalMoment,
               solution::MMGNLSEStochasticSolution)
    field = get_temporal_field(solution)
    axis = _mmgnlse_observable_axis(solution, :time)
    return _mmgnlse_trajectory_values(field) do trajectory
        result, _ = _mmgnlse_moment_value_and_gradient(
            trajectory, axis, observable.order, observable.center,
            observable.normalized, observable.modes)
        result
    end
end

function terminal_condition(observable::TemporalMoment,
                            solution::MMGNLSESolution; z=:final)
    field = get_temporal_field(solution; z=z)
    axis = _mmgnlse_observable_axis(solution, :time)
    _, gradient = _mmgnlse_moment_value_and_gradient(
        field, axis, observable.order, observable.center,
        observable.normalized, observable.modes)
    return _mmgnlse_time_gradient_to_centered_spectral(gradient)
end

function value(observable::SpectralMoment,
               solution::MMGNLSESolution; z=:final)
    field = get_spectral_field(solution; z=z)
    axis = _mmgnlse_observable_axis(solution, :frequency)
    result, _ = _mmgnlse_moment_value_and_gradient(
        field, axis, observable.order, observable.center,
        observable.normalized, observable.modes)
    return result
end

function value(observable::SpectralMoment,
               solution::MMGNLSEStochasticSolution)
    field = get_spectral_field(solution)
    axis = _mmgnlse_observable_axis(solution, :frequency)
    return _mmgnlse_trajectory_values(field) do trajectory
        result, _ = _mmgnlse_moment_value_and_gradient(
            trajectory, axis, observable.order, observable.center,
            observable.normalized, observable.modes)
        result
    end
end

function terminal_condition(observable::SpectralMoment,
                            solution::MMGNLSESolution; z=:final)
    field = get_spectral_field(solution; z=z)
    axis = _mmgnlse_observable_axis(solution, :frequency)
    _, gradient = _mmgnlse_moment_value_and_gradient(
        field, axis, observable.order, observable.center,
        observable.normalized, observable.modes)
    return gradient
end

function value(observable::RatioObservable,
               solution::MMGNLSESolution; z=:final)
    denominator = value(observable.denominator, solution; z=z)
    iszero(denominator) && throw(DomainError(
        denominator, "RatioObservable denominator is zero."))
    return value(observable.numerator, solution; z=z) / denominator
end

function value(observable::RatioObservable,
               solution::MMGNLSEStochasticSolution)
    numerator = value(observable.numerator, solution)
    denominator = value(observable.denominator, solution)
    zero_index = findfirst(iszero, denominator)
    isnothing(zero_index) || throw(DomainError(
        denominator[zero_index],
        "RatioObservable denominator is zero for trajectory $zero_index."))
    return numerator ./ denominator
end

function terminal_condition(observable::RatioObservable,
                            solution::MMGNLSESolution; z=:final)
    numerator = value(observable.numerator, solution; z=z)
    denominator = value(observable.denominator, solution; z=z)
    iszero(denominator) && throw(DomainError(
        denominator, "RatioObservable denominator is zero."))
    numerator_gradient = terminal_condition(
        observable.numerator, solution; z=z)
    denominator_gradient = terminal_condition(
        observable.denominator, solution; z=z)
    size(numerator_gradient) == size(denominator_gradient) ||
        throw(DimensionMismatch(
            "RatioObservable terminal conditions must have matching sizes."))
    return (denominator .* numerator_gradient .-
            numerator .* denominator_gradient) ./ denominator^2
end

"""Return the centered photon count in each frequency-mode-polarization bin."""
function photon_spectrum(solution::MMGNLSESolution; z=:final,
                         sum_modes::Bool=false,
                         sum_polarizations::Bool=false,
                         normalization::Symbol=:bin)
    normalization in (:bin, :density) || throw(ArgumentError(
        "normalization must be :bin or :density."))
    spectrum = get_spectral_field(solution; z=z)
    weights = reshape(_mmgnlse_photon_weights(solution.parameters), :, 1, 1)
    photons = weights .* abs2.(spectrum)
    if normalization === :density
        grid = solution.parameters.domain.time_grid
        # Delta-omega = 2pi/(Nt*dt), so density is photons per rad/ps.
        photons .*= grid.nt * grid.dt / (2π)
    end
    sum_modes && (photons = sum(photons; dims=2))
    sum_polarizations && (photons = sum(photons; dims=3))
    return photons
end

"""Return the photon number selected by a lazy frequency-mode-polarization filter."""
function photon_number(solution::MMGNLSESolution, filter=nothing;
                       z=:final, modes=:all)
    spectrum = get_spectral_field(solution; z=z)
    expanded = _mmgnlse_observable_filter(filter, size(spectrum), modes)
    weights = reshape(_mmgnlse_photon_weights(solution.parameters), :, 1, 1)
    return real(sum(expanded .* weights .* abs2.(spectrum)))
end

"""
Return pulse energy in nJ for each `(mode, polarization)` component.  With
`sum_polarizations=true`, return one total per spatial mode.
"""
function modal_energy(solution::MMGNLSESolution; z=:final,
                      sum_polarizations::Bool=false)
    has_dof(solution.parameters.domain.dofs, :time) || throw(ArgumentError(
        "modal energy requires the :time degree of freedom."))
    field = get_temporal_field(solution; z=z)
    energy = dropdims(sum(abs2, field; dims=1); dims=1) .*
             solution.parameters.domain.time_grid.dt ./ 1e3
    return sum_polarizations ? vec(sum(energy; dims=2)) : energy
end
