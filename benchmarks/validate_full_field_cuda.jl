using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Test

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const C_M_PER_PS = 2.99792458e-4
const OMEGA0 = 2pi * C_M_PER_PS / 1.55e-6

final_field(solution) = @view solution.fields[:, :, :, end]

@testset "Experimental full-field CUDA solver" begin
    @testset "Public API matches the validated kernel" begin
        time = TimeGrid(8, 1.0)
        transverse = TransverseGrid(Nx=6, Ny=4, dx=2e-6, dy=2.5e-6)
        domain = MMGNLSEDomain(
            degrees_of_freedom(:time, :space), time;
            transverse_grid=transverse)
        beta0 = 8.0e6
        beta = TaylorBeta([beta0, 4.8e3, -2.0e-2])
        potential = [150.0 - 2e11 * (x^2 + y^2)
                     for x in transverse.x, y in transverse.y]
        parameters = FullFieldMMGNLSEParameters(
            domain;
            length=2e-5,
            alpha=10.0,
            gain=2.0,
            beta,
            potential,
            n2=2.3e-20,
            omega0=OMEGA0,
            self_steepening=false,
        )
        rng = MersenneTwister(0x44f1)
        initial = randn(rng, ComplexF64, full_field_shape(domain)) .* 2e5

        public_solution = solve_mmgnlse_full_field(
            initial, parameters, 5e-6;
            method=StrangSplit(), precision=:float64, saveat=:steps)
        internal_parameters = PulsePropagation._full_field_legacy_parameters(
            parameters)
        internal_solution = solve_full_field_3d(
            @view(initial[:, :, :, 1]), internal_parameters, 5e-6;
            method=:strang, precision=:float64, saveat=:steps)

        @test size(public_solution.fields) == (8, 6, 4, 1, 5)
        @test public_solution.method isa StrangSplit
        @test public_solution.z == internal_solution.z
        @test public_solution.fields[:, :, :, 1, :] ==
              internal_solution.fields
        @test get_temporal_field(public_solution; z=:final) ==
              public_solution.fields[:, :, :, :, end]
        @test size(get_spectral_field(public_solution; z=:final)) ==
              (8, 6, 4, 1)
        @test size(full_field_power(public_solution; z=:final)) == (8, 1)

        rk_solution = solve_mmgnlse_full_field(
            initial, parameters, 5e-6; precision=:float64)
        @test rk_solution.method isa RK4IP
        @test rk_solution.cache.method == :rk4ip
    end

    @testset "Linear Fourier eigenmode and precision" begin
        time = TimeGrid(16, 2.0)
        grid = FullField3DGrid(
            time; Nx=16, Ny=12, dx=2.0e-6, dy=2.5e-6)
        beta0 = 8.0e6
        propagation_length = 1.0e-3
        mx, my = 2, -1
        initial = Array{ComplexF64}(undef, full_field_shape(grid))
        @inbounds for iy in eachindex(grid.y), ix in eachindex(grid.x),
                      it in 1:time.nt
            initial[it, ix, iy] = cis(2pi * (
                mx * (ix - 1) / length(grid.x) +
                my * (iy - 1) / length(grid.y)))
        end
        parameters = FullField3DParameters(
            grid;
            length=propagation_length,
            beta0,
            omega0=OMEGA0,
        )
        kx = 2pi * mx / (length(grid.x) * grid.dx)
        ky = 2pi * my / (length(grid.y) * grid.dy)
        eigenvalue = -(kx^2 + ky^2) / (2beta0)
        expected = initial .* cis(eigenvalue * propagation_length)

        solution64 = solve_full_field_3d(
            initial, parameters, propagation_length / 5;
            precision=:float64)
        solution64_rk4ip = solve_full_field_3d(
            initial, parameters, propagation_length / 5;
            precision=:float64, method=:rk4ip)
        solution32 = solve_full_field_3d(
            initial, parameters, propagation_length / 5;
            precision=:float32)
        @test norm(final_field(solution64) - expected) / norm(expected) < 2e-11
        @test norm(final_field(solution64_rk4ip) - expected) / norm(expected) < 2e-11
        @test norm(final_field(solution32) - expected) / norm(expected) < 5e-5
        @test solution64.cache.linear_fft_pairs == 6
        @test solution64.cache.accepted_steps == 5
    end

    @testset "Exact constant potential and Kerr phase" begin
        time = TimeGrid(16, 2.0)
        grid = FullField3DGrid(time; Nx=8, Ny=8, dx=2e-6)
        propagation_length = 2.5e-4
        beta0 = 8.0e6
        potential_value = 350.0
        n2 = 2.3e-20
        amplitude = 5.0e7 + 2.0e7im
        initial = fill(ComplexF64(amplitude), full_field_shape(grid))
        parameters = FullField3DParameters(
            grid;
            length=propagation_length,
            beta0,
            potential=fill(potential_value, length(grid.x), length(grid.y)),
            n2,
            omega0=OMEGA0,
        )
        coefficient = OMEGA0 * n2 / C_M_PER_PS
        angle = propagation_length * (
            potential_value + coefficient * abs2(amplitude))
        expected = initial .* cis(angle)
        solution = solve_full_field_3d(
            initial, parameters, propagation_length / 7;
            precision=:float64)
        @test norm(final_field(solution) - expected) / norm(expected) < 2e-11
        @test solution.cache.nonlinear_steps == 7
    end

    @testset "Centered spectral attenuation" begin
        time = TimeGrid(16, 2.0)
        grid = FullField3DGrid(time; Nx=4, Ny=4, dx=3e-6)
        propagation_length = 0.01
        attenuation = 150.0
        initial = fill(ComplexF64(2.0, -0.5), full_field_shape(grid))
        parameters = FullField3DParameters(
            grid;
            length=propagation_length,
            beta0=8e6,
            spectral_loss=fill(attenuation, time.nt),
            omega0=OMEGA0,
        )
        solution = solve_full_field_3d(
            initial, parameters, propagation_length / 4;
            precision=:float64)
        solution_rk4ip = solve_full_field_3d(
            initial, parameters, propagation_length / 4;
            precision=:float64, method=:rk4ip)
        expected = initial .* exp(-attenuation * propagation_length / 2)
        @test norm(final_field(solution) - expected) / norm(expected) < 2e-11
        @test norm(final_field(solution_rk4ip) - expected) / norm(expected) < 2e-11
    end

    @testset "Scalar Raman convolution normalization" begin
        time = TimeGrid(64, 4.0)
        grid = FullField3DGrid(time; Nx=4, Ny=4, dx=3e-6)
        propagation_length = 1.0e-5
        beta0 = 8.0e6
        n2 = 2.3e-20
        amplitude = 4.0e7 - 1.5e7im
        response_model = AgarwalRaman()
        initial = fill(ComplexF64(amplitude), full_field_shape(grid))
        parameters = FullField3DParameters(
            grid;
            length=propagation_length,
            beta0,
            n2,
            omega0=OMEGA0,
            raman=response_model,
        )
        response = raman_response(
            response_model, time.dt .* collect(0:time.nt-1))
        response_integral = time.dt * sum(response.ha .+ response.hb)
        fraction = response_model.fraction
        coefficient = OMEGA0 * n2 / C_M_PER_PS
        nonlinear_intensity = abs2(amplitude) * (
            (1 - fraction) + fraction * response_integral)
        expected = initial .* cis(
            propagation_length * coefficient * nonlinear_intensity)
        solution = solve_full_field_3d(
            initial, parameters, propagation_length / 4;
            precision=:float64)
        @test norm(final_field(solution) - expected) / norm(expected) < 5e-10
        @test solution.cache.raman_fft_pairs == 4
    end

    @testset "Unitary evolution and save semantics" begin
        rng = MersenneTwister(0x3dff)
        time = TimeGrid(32, 3.2)
        grid = FullField3DGrid(time; Nx=12, Ny=10, dx=1.5e-6)
        initial = randn(rng, ComplexF64, full_field_shape(grid)) .* 2e6
        dispersion = 5e-4 .* frequency_axis(time) .^ 2
        potential = [
            -2e3 * ((x / 8e-6)^2 + (y / 8e-6)^2)
            for x in grid.x, y in grid.y
        ]
        parameters = FullField3DParameters(
            grid;
            length=4e-5,
            beta0=8e6,
            dispersion,
            potential,
            n2=2.3e-20,
            omega0=OMEGA0,
            raman=AgarwalRaman(),
        )
        solution = solve_full_field_3d(
            initial, parameters, 1e-5;
            precision=:float64, saveat=:steps)
        initial_norm = sum(abs2, initial)
        for index in axes(solution.fields, 4)
            relative_drift = abs(
                sum(abs2, @view(solution.fields[:, :, :, index])) /
                initial_norm - 1)
            @test relative_drift < 3e-11
        end
        @test solution.z ≈ collect(0.0:1e-5:4e-5) rtol=0 atol=1e-18
    end

    @testset "Parabolic GRIN ground mode" begin
        time = TimeGrid(4, 1.0)
        grid = FullField3DGrid(time; Nx=64, Ny=64, dx=1.0e-6)
        beta0 = 8.0e6
        oscillator_rate = 2.0e3
        width = inv(sqrt(beta0 * oscillator_rate))
        potential = [
            -0.5 * beta0 * oscillator_rate^2 * (x^2 + y^2)
            for x in grid.x, y in grid.y
        ]
        spatial = [
            exp(-(x^2 + y^2) / (2width^2))
            for x in grid.x, y in grid.y
        ]
        spatial ./= sqrt(sum(abs2, spatial) * grid.dx * grid.dy)
        initial = repeat(reshape(spatial, 1, length(grid.x), length(grid.y)),
                         time.nt, 1, 1)
        propagation_length = 2.0e-4
        parameters = FullField3DParameters(
            grid;
            length=propagation_length,
            beta0,
            potential,
            omega0=OMEGA0,
        )
        solution = solve_full_field_3d(
            initial, parameters, 2e-6;
            precision=:float64)
        expected = initial .* cis(-oscillator_rate * propagation_length)
        @test phase_invariant_relative_error(
            final_field(solution), expected) < 8e-4
        phase = dot(vec(initial), vec(final_field(solution)))
        measured = angle(phase)
        @test abs(measured + oscillator_rate * propagation_length) < 2e-3
    end
end

println("validate_full_field_cuda: PASS gpu=", CUDA.name(CUDA.device()))
