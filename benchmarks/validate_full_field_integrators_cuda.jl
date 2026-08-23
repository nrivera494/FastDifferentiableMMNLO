using CUDA
using FFTW
using LinearAlgebra
using NPZ
using PulsePropagation
using Test

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const CUDA_EXT = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
const C_M_PER_PS_FF = 2.99792458e-4
const OMEGA0_FF = 2pi * C_M_PER_PS_FF / 1.55e-6
const OUTPUT_DIR_FF = get(
    ENV,
    "FULL_FIELD_INTEGRATOR_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache",
             "full_field_integrator_validation"),
)

final_field_ff(solution) = Array(@view solution.fields[:, :, :, end])
relative_l2_ff(candidate, reference) =
    norm(vec(candidate .- reference)) / norm(vec(reference))

function deterministic_field(grid; peak_power=4.0e5)
    t = time_axis(grid.time)
    temporal = sqrt(peak_power) .* exp.(
        -2log(2) .* (t ./ 0.28) .^ 2) .* cis.(0.35 .* (t ./ 0.28) .^ 2)
    spatial = ComplexF64[
        exp(-0.5 * ((x / 4.5e-6)^2 + (y / 3.8e-6)^2)) *
        cis(0.22 * x / 4.5e-6 - 0.17 * y / 3.8e-6)
        for x in grid.x, y in grid.y
    ]
    spatial ./= sqrt(sum(abs2, spatial) * grid.dx * grid.dy)
    return reshape(temporal, :, 1, 1) .* reshape(
        spatial, 1, length(grid.x), length(grid.y))
end

function cpu_residual_reference(field, parameters)
    nt = size(field, 1)
    intensity = abs2.(field)
    fraction = PulsePropagation.raman_fraction(parameters.raman)
    response = PulsePropagation._full_field_raman_spectrum(parameters, Float64)
    delayed = if iszero(fraction)
        intensity
    else
        irfft(rfft(intensity, 1) .* reshape(response, :, 1, 1), nt, 1)
    end
    nonlinear_response = (1 - fraction) .* intensity .+ fraction .* delayed
    polarization = field .* nonlinear_response
    omega = PulsePropagation.ifft_frequency_grid(
        nt, parameters.grid.time.dt)
    damp = parameters.self_steepening ?
        PulsePropagation.damped_freq_window(nt) : ones(nt)
    nonlinear_prefactor = reshape(
        1im .* parameters.n2 .* (parameters.self_steepening ?
            omega .+ parameters.omega0 : fill(parameters.omega0, nt)) ./
        C_M_PER_PS_FF .* damp,
        nt, 1, 1)
    potential_rhs = 1im .* reshape(parameters.potential, 1,
                                    size(field, 2), size(field, 3)) .* field
    return ifft(potential_rhs, (1, 2, 3)) .+
           nonlinear_prefactor .* ifft(polarization, (1, 2, 3))
end

@testset "Full-field self-steepening RHS" begin
    time = TimeGrid(64, 2.0)
    grid = FullField3DGrid(time; Nx=4, Ny=3, dx=3.0e-6, dy=3.5e-6)
    field = deterministic_field(grid; peak_power=2.0e5)
    potential = [350.0 + 80.0ix - 55.0iy
                 for ix in eachindex(grid.x), iy in eachindex(grid.y)]
    parameters = FullField3DParameters(
        grid;
        length=1.0e-6,
        beta0=8.0e6,
        potential,
        n2=2.3e-20,
        omega0=OMEGA0_FF,
        raman=AgarwalRaman(),
        self_steepening=true,
    )
    expected = cpu_residual_reference(field, parameters)
    workspace = CUDA_EXT._cuda_full_field_workspace(
        field, parameters, Float64, :rk4ip)
    mul!(workspace.field, workspace.to_spectral!, workspace.field)
    CUDA_EXT._cuda_full_field_residual_spectral!(
        workspace.derivative, workspace.field, workspace)
    CUDA.synchronize()
    measured = Array(workspace.derivative)
    @test relative_l2_ff(measured, expected) < 2e-10
end

function mixed_case(nsteps, method)
    time = TimeGrid(64, 2.0)
    grid = FullField3DGrid(time; Nx=16, Ny=16, dx=3.0e-6)
    field = deterministic_field(grid)
    beta0 = 8.0e6
    oscillator_rate = 2.2e3
    potential = [
        -0.5 * beta0 * oscillator_rate^2 *
        ((1.05x)^2 + (0.90y)^2) + 180.0 * x / 8e-6
        for x in grid.x, y in grid.y
    ]
    dispersion = -0.012 .* frequency_axis(time) .^ 2
    length_m = 4.0e-4
    parameters = FullField3DParameters(
        grid;
        length=length_m,
        beta0,
        dispersion,
        potential,
        n2=2.3e-20,
        omega0=OMEGA0_FF,
        raman=AgarwalRaman(),
        self_steepening=true,
    )
    solution = solve_full_field_3d(
        field, parameters, length_m / nsteps;
        precision=:float64, method)
    return final_field_ff(solution)
end

@testset "Full-field Strang/RK4IP longitudinal order" begin
    reference64 = mixed_case(64, :rk4ip)
    reference128 = mixed_case(128, :rk4ip)
    reference_gap = relative_l2_ff(reference64, reference128)

    steps = (4, 8, 16)
    strang_fields = map(n -> mixed_case(n, :strang), steps)
    rk4ip_fields = map(n -> mixed_case(n, :rk4ip), steps)
    strang_errors = map(field -> relative_l2_ff(field, reference128), strang_fields)
    rk4ip_errors = map(field -> relative_l2_ff(field, reference128), rk4ip_fields)
    strang_order = log2(strang_errors[2] / strang_errors[3])
    rk4ip_order = log2(rk4ip_errors[2] / rk4ip_errors[3])
    cross_errors = map(
        index -> relative_l2_ff(strang_fields[index], rk4ip_fields[index]),
        eachindex(steps))
    cross_order = log2(cross_errors[2] / cross_errors[3])

    println("reference gap = ", reference_gap)
    println("Strang errors = ", strang_errors, " observed order = ", strang_order)
    println("RK4IP errors = ", rk4ip_errors, " observed order = ", rk4ip_order)
    println("same-step Strang/RK4IP errors = ", cross_errors,
            " observed order = ", cross_order)

    @test all(diff(collect(strang_errors)) .< 0)
    @test all(diff(collect(rk4ip_errors)) .< 0)
    @test all(diff(collect(cross_errors)) .< 0)
    @test reference_gap < 0.05 * rk4ip_errors[3]
    @test 1.65 < strang_order < 2.35
    @test 3.3 < rk4ip_order < 4.7
    @test 1.5 < cross_order < 2.5
    @test rk4ip_errors[3] < 0.2 * strang_errors[3]

    mkpath(OUTPUT_DIR_FF)
    comparison_grid = FullField3DGrid(
        TimeGrid(64, 2.0); Nx=16, Ny=16, dx=3.0e-6)
    dxdy = comparison_grid.dx * comparison_grid.dy
    dt = comparison_grid.time.dt
    spectrum(field) = vec(sum(
        abs2, fftshift(ifft(field, 1), 1); dims=(2, 3))) .* dxdy
    fluence(field) = dropdims(sum(abs2, field; dims=1); dims=1) .* dt
    reference_power = vec(sum(abs2, reference128; dims=(2, 3))) .* dxdy
    peak_index = argmax(reference_power)
    npzwrite(joinpath(OUTPUT_DIR_FF, "integrator_comparison.npz"), Dict(
        "step_counts" => collect(steps),
        "dz_m" => 4.0e-4 ./ collect(steps),
        "strang_errors" => collect(strang_errors),
        "rk4ip_errors" => collect(rk4ip_errors),
        "cross_errors" => collect(cross_errors),
        "reference_gap" => [reference_gap],
        "strang_order" => [strang_order],
        "rk4ip_order" => [rk4ip_order],
        "cross_order" => [cross_order],
        "frequency_detuning_THz" =>
            frequency_axis(comparison_grid.time) ./ (2pi),
        "x_m" => comparison_grid.x,
        "y_m" => comparison_grid.y,
        "strang_spectrum" => spectrum(strang_fields[end]),
        "rk4ip_spectrum" => spectrum(rk4ip_fields[end]),
        "reference_spectrum" => spectrum(reference128),
        "strang_fluence" => fluence(strang_fields[end]),
        "rk4ip_fluence" => fluence(rk4ip_fields[end]),
        "reference_fluence" => fluence(reference128),
        "strang_peak_intensity" => abs2.(
            strang_fields[end][peak_index, :, :]),
        "rk4ip_peak_intensity" => abs2.(
            rk4ip_fields[end][peak_index, :, :]),
        "reference_peak_intensity" => abs2.(
            reference128[peak_index, :, :]),
        "peak_time_index" => [peak_index],
    ))
    open(joinpath(OUTPUT_DIR_FF, "convergence.csv"), "w") do io
        println(io, "steps,dz_m,strang_error,rk4ip_error,cross_error")
        for index in eachindex(steps)
            println(io, join((
                steps[index], 4.0e-4 / steps[index],
                strang_errors[index], rk4ip_errors[index],
                cross_errors[index]), ','))
        end
    end
    println("integrator artifacts = ", OUTPUT_DIR_FF)
end

println("validate_full_field_integrators_cuda: PASS gpu=", CUDA.name(CUDA.device()))
