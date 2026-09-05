using CUDA
using FFTW
using LinearAlgebra
using Random
using Test
using PulsePropagation

CUDA.functional() || error("A functional CUDA device is required.")
CUDA.allowscalar(false)

function _cuda_checkpoint_parameters(; raman=NoRaman(), cp=false,
                                     length_m=6.0e-4)
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
        n2=2.3e-20,
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

_cuda_centered_spectrum(field_t) = fftshift(ifft(field_t, 1), 1)

function _cuda_checkpoint_directional_check(parameters, initial, dz,
                                            backend; seed)
    rng = MersenneTwister(seed)
    direction = randn(rng, ComplexF64, size(initial))
    direction ./= norm(direction)
    terminal = randn(rng, ComplexF64, size(initial))
    terminal ./= norm(terminal)
    epsilon = 2.0e-6 * norm(initial)
    objective(input) = real(sum(conj.(terminal) .*
        get_spectral_field(solve_mmgnlse(
            input, parameters, dz; backend=backend))))
    finite_difference = (
        objective(initial .+ epsilon .* direction) -
        objective(initial .- epsilon .* direction)
    ) / (2epsilon)
    adjoint = solve_adjoint(
        terminal, parameters, dz;
        initial_field=initial,
        dz_forward=dz,
        backend=backend,
        adjoint_mode=:discrete_rk4ip,
        checkpoint_stride=2,
    )
    predicted = real(sum(
        conj.(get_spectral_field(adjoint; z=:initial)) .*
        _cuda_centered_spectrum(direction)))
    return finite_difference, predicted, terminal, adjoint
end

@testset "CUDA checkpointed discrete RK4IP adjoint" begin
    for (name, raman, cp, backend, seed) in (
        ("dense Kerr", NoRaman(), false, :cuda, 501),
        ("dense anisotropic Raman", AnisotropicRaman(), false,
         :cuda, 503),
        ("CP anisotropic Raman", AnisotropicRaman(), true,
         :cuda_cp_optimized, 502),
    )
        @testset "$name directional derivative" begin
            parameters, initial = _cuda_checkpoint_parameters(; raman, cp)
            finite_difference, predicted, terminal, adjoint =
                _cuda_checkpoint_directional_check(
                    parameters, initial, 1.0e-4, backend; seed)
            @test predicted ≈ finite_difference rtol=2e-5 atol=2e-7
            @test adjoint.cache.checkpoint_count == 4
            @test adjoint.cache.replay_peak_planes == 3

            stride_one = solve_adjoint(
                terminal, parameters, 1.0e-4;
                initial_field=initial, dz_forward=1.0e-4,
                backend=backend, adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=1)
            stride_full = solve_adjoint(
                terminal, parameters, 1.0e-4;
                initial_field=initial, dz_forward=1.0e-4,
                backend=backend, adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=100)
            @test get_spectral_field(stride_one; z=:initial) ≈
                  get_spectral_field(stride_full; z=:initial) rtol=2e-11

            cpu = solve_adjoint(
                terminal, parameters, 1.0e-4;
                initial_field=initial, dz_forward=1.0e-4,
                backend=:cpu, adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=2)
            @test get_spectral_field(adjoint; z=:initial) ≈
                  get_spectral_field(cpu; z=:initial) rtol=2e-10
        end
    end

    @testset "continuous limit" begin
        parameters, initial = _cuda_checkpoint_parameters(
            ; raman=AnisotropicRaman(), cp=true, length_m=8.0e-4)
        terminal = reshape(ComplexF64.(range(0.2, 1.1; length=16)),
                           16, 1, 1)
        errors = Float64[]
        for dz in (2.0e-4, 1.0e-4)
            forward = solve_mmgnlse(
                initial, parameters, dz;
                backend=:cuda_cp_optimized, saveat=:steps)
            discrete = solve_adjoint(
                terminal, parameters, dz;
                forward_solution=forward,
                backend=:cuda_cp_optimized,
                adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=2)
            continuous = solve_adjoint(
                terminal, parameters, dz;
                forward_solution=forward,
                backend=:cuda_cp_optimized)
            initial_discrete = get_spectral_field(discrete; z=:initial)
            push!(errors, norm(initial_discrete .-
                               get_spectral_field(continuous; z=:initial)) /
                          norm(initial_discrete))
        end
        @test errors[2] < errors[1]
        @test errors[2] < 2e-4
    end
end


include("cuda_compatibility_fixtures.jl")

@testset "Multimode and polarization checkpointed CUDA adjoints" begin
    for (name,np,cp,raman) in (
        ("dense multimode Kerr",1,false,NoRaman()),
        ("dense isotropic Raman",1,false,AgarwalRaman()),
        ("CP isotropic Raman",1,true,AgarwalRaman()),
        ("dense polarized Raman",2,false,AnisotropicRaman()),
        ("CP multimode Raman",1,true,AnisotropicRaman()),
        ("CP polarized Raman",2,true,AnisotropicRaman()),
    )
        @testset "$name" begin
            parameters,initial = cuda_compatibility_fixture(;nt=32,nm=3,np,cp,raman)
            backend = cp ? :cuda_cp_optimized : :cuda
            dz = 5e-4
            finite_difference,predicted,terminal,adjoint =
                _cuda_checkpoint_directional_check(parameters,initial,dz,backend;seed=963)
            @test predicted ≈ finite_difference rtol=2e-5 atol=2e-7
            cpu = solve_adjoint(terminal,parameters,dz;
                initial_field=initial,dz_forward=dz,adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=3)
            @test get_spectral_field(adjoint;z=:initial) ≈
                get_spectral_field(cpu;z=:initial) rtol=2e-9
            forward = solve_mmgnlse(initial,parameters,dz;backend)
            replayed = solve_adjoint(terminal,parameters,dz;
                forward_solution=forward,backend,adjoint_mode=:discrete_rk4ip,
                checkpoint_stride=3,saveat=:steps)
            @test get_spectral_field(adjoint;z=:initial) ≈
                get_spectral_field(replayed;z=:initial) rtol=2e-11
            @test replayed.cache.replay_peak_planes <= 4
            @test replayed.cache.checkpoint_count < length(replayed.cache.integration_z)
        end
    end
end


@testset "Checkpointed replay rejects mixed-precision forward maps" begin
    parameters,initial = cuda_compatibility_fixture(;nt=32,nm=3,cp=true)
    forward = solve_mmgnlse(initial,parameters,5e-4;
        backend=:cuda_cp_optimized,precision=:mixed)
    terminal = ones(ComplexF64,size(initial))
    @test_throws ArgumentError solve_adjoint(terminal,parameters,5e-4;
        forward_solution=forward,backend=:cuda_cp_optimized,
        adjoint_mode=:discrete_rk4ip)
    @test_throws ArgumentError solve_adjoint(terminal,parameters,5e-4;
        forward_solution=forward,backend=:cpu,adjoint_mode=:discrete_rk4ip)
end
