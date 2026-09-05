using FFTW
using LinearAlgebra
using Random

function _checkpoint_test_parameters(; raman=NoRaman(), n2=2.3e-20,
                                     length_m=6.0e-4, cp=false)
    grid = TimeGrid(16, 2.0)
    domain = MMGNLSEDomain(degrees_of_freedom(:time), grid)
    overlap = cp ?
        MMGNLSECPDecomposition(
            [1.0e10], ntuple(_ -> ones(1, 1), 4);
            layout=:spatial, nmodes=1, npolarizations=1,
            relative_error=0.0) :
        reshape([1.0e10], 1, 1, 1, 1)
    parameters = MMGNLSEParameters(
        domain;
        length=length_m,
        beta=TaylorBeta(reshape([0.0, 0.0, -0.02], 3, 1)),
        S=overlap,
        n2,
        omega0=2pi * 193.4,
        raman,
    )
    t = time_axis(grid)
    initial = reshape(
        ComplexF64.(sqrt(2.0e3) .* exp.(-(t ./ 0.32) .^ 2) .*
                   exp.(0.17im .* t)),
        length(t), 1, 1,
    )
    return parameters, initial
end

function _centered_spectrum(field_t)
    return fftshift(ifft(field_t, 1), 1)
end

function _directional_derivative_check(parameters, initial, dz; seed)
    rng = MersenneTwister(seed)
    direction = randn(rng, ComplexF64, size(initial))
    direction ./= norm(direction)
    terminal = randn(rng, ComplexF64, size(initial))
    terminal ./= norm(terminal)
    epsilon = 2.0e-6 * norm(initial)

    objective(input) = real(sum(conj.(terminal) .* _centered_spectrum(
        solve_mmgnlse(input, parameters, dz).fields[:, :, :, end])))
    finite_difference = (
        objective(initial .+ epsilon .* direction) -
        objective(initial .- epsilon .* direction)
    ) / (2epsilon)

    adjoint = solve_adjoint(
        terminal, parameters, dz;
        initial_field=initial,
        dz_forward=dz,
        adjoint_mode=:discrete_rk4ip,
        checkpoint_stride=2,
    )
    initial_cotangent = get_spectral_field(adjoint; z=:initial)
    predicted = real(sum(
        conj.(initial_cotangent) .* _centered_spectrum(direction)))
    return finite_difference, predicted, adjoint
end

@testset "Checkpointed discrete RK4IP adjoint" begin
    @testset "single-step VJP directional differences" begin
        for (raman, cp, seed) in (
            (NoRaman(), false, 101),
            (AnisotropicRaman(), false, 151),
            (AnisotropicRaman(), true, 202),
        )
            parameters, initial = _checkpoint_test_parameters(
                ; raman, cp, length_m=1.0e-4)
            cache = PulsePropagation._mmgnlse_solver_cache(parameters)
            field_w = ifft(initial, 1)
            rng = MersenneTwister(seed)
            direction = randn(rng, ComplexF64, size(field_w))
            direction ./= norm(direction)
            cotangent = randn(rng, ComplexF64, size(field_w))
            cotangent ./= norm(cotangent)
            epsilon = 2.0e-6 * norm(field_w)
            pairing(input) = real(sum(conj.(cotangent) .*
                PulsePropagation._mmgnlse_rk4ip_step(
                    input, parameters, 0.0, 1.0e-4, cache)))
            finite_difference = (
                pairing(field_w .+ epsilon .* direction) -
                pairing(field_w .- epsilon .* direction)
            ) / (2epsilon)
            vjp = PulsePropagation._mmgnlse_rk4ip_step_vjp(
                field_w, cotangent, parameters, 0.0, 1.0e-4, cache)
            predicted = real(sum(conj.(vjp) .* direction))
            @test predicted ≈ finite_difference rtol=2e-6 atol=2e-8
        end
    end

    @testset "full-map differences and checkpoint invariance" begin
        for (raman, cp, seed) in (
            (NoRaman(), false, 303),
            (AnisotropicRaman(), false, 353),
            (AnisotropicRaman(), true, 404),
        )
            parameters, initial = _checkpoint_test_parameters(; raman, cp)
            finite_difference, predicted, adjoint =
                _directional_derivative_check(
                    parameters, initial, 1.0e-4; seed)
            @test predicted ≈ finite_difference rtol=3e-6 atol=3e-8
            @test adjoint.cache.checkpoint_stride == 2
            @test adjoint.cache.replay_peak_planes == 3

            terminal = get_spectral_field(adjoint; z=:final)
            stride_one = solve_adjoint(
                terminal, parameters, 1.0e-4;
                initial_field=initial, dz_forward=1.0e-4,
                adjoint_mode=:discrete_rk4ip, checkpoint_stride=1)
            stride_full = solve_adjoint(
                terminal, parameters, 1.0e-4;
                initial_field=initial, dz_forward=1.0e-4,
                adjoint_mode=:discrete_rk4ip, checkpoint_stride=100)
            @test get_spectral_field(stride_one; z=:initial) ≈
                  get_spectral_field(stride_full; z=:initial) rtol=2e-13
        end
    end

    @testset "continuous limit and validation" begin
        linear_parameters, initial = _checkpoint_test_parameters(
            ; n2=0.0, length_m=6.0e-4)
        terminal = reshape(ComplexF64.(range(0.2, 1.1; length=16)), 16, 1, 1)
        forward = solve_mmgnlse(
            initial, linear_parameters, 1.0e-4; saveat=:steps)
        discrete = solve_adjoint(
            terminal, linear_parameters, 1.0e-4;
            forward_solution=forward,
            adjoint_mode=:discrete_rk4ip,
            checkpoint_stride=3)
        continuous = solve_adjoint(
            terminal, linear_parameters, 1.0e-4;
            forward_solution=forward)
        @test get_spectral_field(discrete; z=:initial) ≈
              get_spectral_field(continuous; z=:initial) rtol=5e-13

        nonlinear_parameters, nonlinear_initial = _checkpoint_test_parameters(
            ; raman=AnisotropicRaman(), cp=true, length_m=8.0e-4)
        errors = Float64[]
        for dz in (2.0e-4, 1.0e-4)
            nonlinear_forward = solve_mmgnlse(
                nonlinear_initial, nonlinear_parameters, dz; saveat=:steps)
            discrete = solve_adjoint(
                terminal, nonlinear_parameters, dz;
                forward_solution=nonlinear_forward,
                adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=2)
            continuous = solve_adjoint(
                terminal, nonlinear_parameters, dz;
                forward_solution=nonlinear_forward)
            numerator = norm(
                get_spectral_field(discrete; z=:initial) .-
                get_spectral_field(continuous; z=:initial))
            denominator = norm(get_spectral_field(discrete; z=:initial))
            push!(errors, numerator / denominator)
        end
        @test errors[2] < errors[1]
        @test errors[2] < 2e-4

        @test_throws ArgumentError solve_adjoint(
            terminal, nonlinear_parameters, 2.0e-4;
            initial_field=nonlinear_initial, dz_forward=1.0e-4,
            adjoint_mode=:discrete_rk4ip)
        @test_throws ArgumentError solve_adjoint(
            terminal, nonlinear_parameters, 1.0e-4;
            initial_field=nonlinear_initial, dz_forward=1.0e-4,
            adjoint_mode=:discrete_rk4ip, saveat=[0.0, 1.5e-4, 8.0e-4])
    end
end
