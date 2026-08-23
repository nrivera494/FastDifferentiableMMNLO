using CUDA
using FFTW
using LinearAlgebra
using NPZ
using PulsePropagation
using Random

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const C_M_PER_PS = 2.99792458e-4

function environment_value(name, default, ::Type{T}) where {T}
    return parse(T, get(ENV, name, string(default)))
end

function eslami_dispersion(time)
    # Published at 1700 nm. Input units are s^n/m; convert to ps^n/m.
    coefficients_si = (
        5.93e-26,
        3.10e-40,
        -7.35e-55,
        2.15e-69,
        -3.60e-84,
        2.84e-99,
        1.35e-115,
    )
    omega = frequency_axis(time)
    dispersion = zeros(Float64, length(omega))
    for (index, coefficient_si) in enumerate(coefficients_si)
        order = index + 1
        coefficient_ps = coefficient_si * 1e12^order
        dispersion .+= coefficient_ps .* omega .^ order ./ factorial(order)
    end
    return dispersion
end

function eslami_spectral_loss(time, omega0)
    # Smooth continuous absorber representing the paper's super-Gaussian
    # long-wavelength suppression beyond 2800 nm. The attenuation is zero at
    # the quoted edge and saturates over a 10 THz transition outside it.
    cutoff = 2pi * C_M_PER_PS / 2800e-9
    transition = 2pi * 10.0 # rad/ps = 10 THz
    normalized_shortfall = max.(
        (cutoff .- (omega0 .+ frequency_axis(time))) ./ transition, 0.0)
    return 5.0e4 .* min.(normalized_shortfall .^ 8, 1.0)
end

function eslami_geometry(time; transverse_count=64)
    wavelength = 1700e-9
    omega0 = 2pi * C_M_PER_PS / wavelength
    ncore = 1.885
    nclad = 1.866
    core_radius = 40e-6
    relative_index = (ncore - nclad) / ncore
    grid = FullField3DGrid(
        time;
        Nx=transverse_count,
        Ny=transverse_count,
        dx=160e-6 / transverse_count,
    )
    curvature = (ncore - nclad) / core_radius^2
    index_yx = [
        hypot(x, y) <= core_radius ?
            ncore - curvature * (x^2 + y^2) : nclad
        for y in grid.y, x in grid.x
    ]
    beta0 = ncore * omega0 / C_M_PER_PS
    potential = paraxial_index_potential(
        grid, index_yx; beta0, omega0, reference_index=ncore)
    self_imaging_period = pi * core_radius / sqrt(2relative_index)
    return (;
        grid,
        omega0,
        beta0,
        potential,
        self_imaging_period,
        ncore,
        nclad,
        core_radius,
        relative_index,
    )
end

function eslami_initial_field(geometry, peak_power; seed=20220819)
    grid = geometry.grid
    time = grid.time
    rng = MersenneTwister(seed)
    pulse_fwhm = 0.350 # ps, intensity FWHM
    time_values = time_axis(time)
    temporal_intensity = exp.(-4log(2) .* (time_values ./ pulse_fwhm) .^ 2)
    # One deterministic realization of the reported 0.2% temporal RIN.
    temporal_intensity .*= max.(0.0, 1 .+ 0.002 .* randn(rng, time.nt))
    temporal_amplitude = sqrt.(temporal_intensity)

    beam_radius = 25e-6 # 1/e^2 intensity radius
    spatial = ComplexF32[
        exp(-(x^2 + y^2) / beam_radius^2)
        for x in grid.x, y in grid.y
    ]
    spatial ./= sqrt(sum(abs2, spatial) * grid.dx * grid.dy)
    if get(ENV, "ESLAMI_RANDOM_SPATIAL_PHASE", "false") == "true"
        spatial .*= cis.(Float32(pi) .* rand(rng, Float32, size(spatial)))
    end

    return ComplexF32.(
        sqrt(peak_power) .*
        reshape(temporal_amplitude, time.nt, 1, 1) .*
        reshape(spatial, 1, length(grid.x), length(grid.y)))
end

function spatial_fluence(field, grid)
    return dropdims(sum(abs2, field; dims=1); dims=1) .* grid.time.dt
end

function effective_area(fluence, grid)
    integral = sum(fluence) * grid.dx * grid.dy
    return integral^2 / (sum(abs2, fluence) * grid.dx * grid.dy)
end

function normalized_image_error(candidate, reference)
    candidate_normalized = candidate ./ sum(candidate)
    reference_normalized = reference ./ sum(reference)
    return norm(candidate_normalized - reference_normalized) /
           norm(reference_normalized)
end

function spectral_diagnostics(field, grid, omega0)
    spectral_field = fftshift(ifft(field, 1), 1)
    density = vec(sum(abs2, spectral_field; dims=(2, 3))) .* (grid.dx * grid.dy)
    density ./= maximum(density)
    detuning_thz = frequency_axis(grid.time) ./ (2pi)
    wavelength_nm = wavelength_axis(grid.time, omega0) .* 1e9
    weight = density ./ sum(density)
    centroid = sum(weight .* detuning_thz)
    rms_width = sqrt(sum(weight .* (detuning_thz .- centroid) .^ 2))
    supported = findall(
        index -> isfinite(wavelength_nm[index]) && density[index] >= 1e-4,
        eachindex(density))
    wavelength_span = isempty(supported) ? (NaN, NaN) :
        extrema(wavelength_nm[supported])
    return (; density, detuning_thz, wavelength_nm, centroid, rms_width,
            wavelength_span)
end

nt = environment_value("ESLAMI_NT", 16384, Int)
transverse_count = environment_value("ESLAMI_NX", 64, Int)
propagation_length = environment_value("ESLAMI_LENGTH", 0.20, Float64)
step = environment_value("ESLAMI_DZ", 37e-6, Float64)
peak_power = environment_value("ESLAMI_PEAK_POWER", 5.85e5, Float64)
time = TimeGrid(nt, 20.0)
geometry = eslami_geometry(time; transverse_count)
initial = eslami_initial_field(geometry, peak_power)
initial_fluence = spatial_fluence(initial, geometry.grid)
pulse_energy_nj = sum(full_field_power(initial, geometry.grid)) * time.dt * 1e-3
spectral_filter_enabled = get(ENV, "ESLAMI_SPECTRAL_FILTER", "true") == "true"

@info "Eslami-like 3+1D gate" nt transverse_count propagation_length step peak_power pulse_energy_nj self_imaging_period=geometry.self_imaging_period included=(
        "published grid/window",
        "350 fs Gaussian pulse and 25 um beam",
        "published parabolic index and beta2-beta8",
        "Kerr coefficient and 0.2% RIN",
        spectral_filter_enabled ?
            "super-Gaussian attenuation beyond 2800 nm" :
            "spectral absorber disabled by environment",
    ) omitted=(
        "shock/self-steepening",
        "absorbing transverse boundary/leakage model",
        "one-photon-per-mode noise",
        "Raman (negligible and omitted in the paper production runs)",
    )

linear_image_error = NaN
if get(ENV, "ESLAMI_LINEAR_GATE", "true") == "true"
    linear_parameters = FullField3DParameters(
        geometry.grid;
        length=geometry.self_imaging_period,
        beta0=geometry.beta0,
        potential=geometry.potential,
        omega0=geometry.omega0,
    )
    linear_seconds = @elapsed linear_solution = solve_full_field_3d(
        initial, linear_parameters, step; precision=:float32)
    linear_final = @view linear_solution.fields[:, :, :, end]
    linear_fluence = spatial_fluence(linear_final, geometry.grid)
    linear_image_error = normalized_image_error(linear_fluence, initial_fluence)
    linear_diagnostics = (;
        linear_seconds,
        linear_image_error,
        accepted_steps=linear_solution.cache.accepted_steps,
        linear_fft_pairs=linear_solution.cache.linear_fft_pairs,
    )
    @info "Eslami linear self-imaging gate" linear_diagnostics
    linear_solution = nothing
    linear_final = nothing
    linear_fluence = nothing
    GC.gc()
    CUDA.reclaim()
end

parameters = FullField3DParameters(
    geometry.grid;
    length=propagation_length,
    beta0=geometry.beta0,
    dispersion=eslami_dispersion(time),
    spectral_loss=spectral_filter_enabled ?
        eslami_spectral_loss(time, geometry.omega0) : zeros(time.nt),
    potential=geometry.potential,
    n2=1.92e-19,
    omega0=geometry.omega0,
    raman=NoRaman(),
)

wall_seconds = @elapsed solution = solve_full_field_3d(
    initial, parameters, step; precision=:float32)
final_field = @view solution.fields[:, :, :, end]
final_fluence = spatial_fluence(final_field, geometry.grid)
initial_spectrum = spectral_diagnostics(initial, geometry.grid, geometry.omega0)
final_spectrum = spectral_diagnostics(final_field, geometry.grid, geometry.omega0)
initial_energy = sum(full_field_power(initial, geometry.grid)) * time.dt
final_energy = sum(full_field_power(final_field, geometry.grid)) * time.dt
power_drift = abs(final_energy / initial_energy - 1)

nonlinear_diagnostics = (;
    wall_seconds,
    accepted_steps=solution.cache.accepted_steps,
    linear_fft_pairs=solution.cache.linear_fft_pairs,
    workspace_gib=solution.cache.workspace_bytes / 2.0^30,
    power_drift,
    initial_rms_thz=initial_spectrum.rms_width,
    final_rms_thz=final_spectrum.rms_width,
    spectral_broadening=final_spectrum.rms_width / initial_spectrum.rms_width,
    wavelength_span_nm=final_spectrum.wavelength_span,
    initial_effective_area_um2=
        effective_area(initial_fluence, geometry.grid) * 1e12,
    final_effective_area_um2=
        effective_area(final_fluence, geometry.grid) * 1e12,
)
@info "Eslami-like nonlinear result" nonlinear_diagnostics

output_path = get(ENV, "ESLAMI_OUTPUT", "")
if !isempty(output_path)
    npzwrite(output_path, Dict(
        "frequency_detuning_THz" => final_spectrum.detuning_thz,
        "wavelength_nm" => final_spectrum.wavelength_nm,
        "input_spectrum_normalized" => initial_spectrum.density,
        "output_spectrum_normalized" => final_spectrum.density,
        "x_um" => geometry.grid.x .* 1e6,
        "y_um" => geometry.grid.y .* 1e6,
        "input_fluence_normalized" => initial_fluence ./ maximum(initial_fluence),
        "output_fluence_normalized" => final_fluence ./ maximum(final_fluence),
        "linear_self_image_error" => [linear_image_error],
        "wall_seconds" => [wall_seconds],
        "power_drift" => [power_drift],
    ))
    @info "wrote Eslami-like diagnostics" output_path
end
