using LinearAlgebra
using PulsePropagation
using Test

@testset "Experimental full-field host contract" begin
    time = TimeGrid(32, 2.0)
    grid = FullField3DGrid(time; Nx=12, Ny=10, dx=0.75e-6, dy=0.9e-6)
    @test full_field_shape(grid) == (32, 12, 10)
    @test_throws ArgumentError FullField3DGrid(
        time, [0.0, 1.0, 2.1], [0.0, 1.0])

    parameters = FullField3DParameters(
        grid;
        length=0.01,
        beta0=8.0e6,
        omega0=2pi * 2.99792458e-4 / 1.55e-6,
    )
    @test size(parameters.potential) == (12, 10)
    @test length(parameters.dispersion) == 32
    @test iszero(sum(parameters.spectral_loss))
    @test !parameters.self_steepening
    steep_parameters = FullField3DParameters(
        grid;
        length=0.01,
        beta0=8.0e6,
        omega0=parameters.omega0,
        self_steepening=true,
    )
    @test steep_parameters.self_steepening
    @test_throws DimensionMismatch FullField3DParameters(
        grid;
        length=0.01,
        beta0=8.0e6,
        spectral_loss=zeros(31),
        omega0=parameters.omega0,
    )
    @test_throws ArgumentError FullField3DParameters(
        grid;
        length=0.01,
        beta0=8.0e6,
        spectral_loss=-ones(32),
        omega0=parameters.omega0,
    )

    dxdy = grid.dx * grid.dy
    normalization = inv(sqrt(length(grid.x) * length(grid.y) * dxdy))
    modes = Array{ComplexF64}(undef, length(grid.y), length(grid.x), 2)
    modes[:, :, 1] .= normalization
    transverse_period = length(grid.x) * grid.dx
    @inbounds for ix in eachindex(grid.x), iy in eachindex(grid.y)
        modes[iy, ix, 2] = normalization *
            cis(2pi * (ix - 1) * grid.dx / transverse_period)
    end

    amplitudes = zeros(ComplexF64, time.nt, 2)
    amplitudes[:, 1] .= 1 .+ 0.25im
    amplitudes[:, 2] .= range(0.1, 0.5; length=time.nt) .- 0.2im
    field = reconstruct_full_field(amplitudes, modes, grid)
    projected = project_full_field(field, modes, grid)
    @test norm(projected - amplitudes) / norm(amplitudes) < 1e-12
    @test full_field_subspace_leakage(field, modes, grid) < 1e-12
    @test maximum(abs.(full_field_power(field, grid) .-
                       vec(sum(abs2, amplitudes; dims=2)))) < 1e-11
    @test phase_invariant_relative_error(
        cis(0.731) .* field, field) < 1e-12

    nref = 1.45
    beta0 = parameters.omega0 / 2.99792458e-4 * nref
    index_yx = fill(nref, length(grid.y), length(grid.x))
    potential = paraxial_index_potential(
        grid, index_yx; beta0, omega0=parameters.omega0)
    @test maximum(abs, potential) < 1e-8

    @test_throws ArgumentError solve_full_field_3d(
        field, parameters, 1e-4; backend=:cpu)
    @test_throws ArgumentError solve_full_field_3d(
        field, parameters, 1e-4; method=:not_an_integrator)
    unstable_parameters = FullField3DParameters(
        grid;
        length=0.01,
        beta0=8.0e6,
        potential=fill(4.0e4, length(grid.x), length(grid.y)),
        omega0=parameters.omega0,
    )
    @test_throws ArgumentError solve_full_field_3d(
        field, unstable_parameters, 1e-4; method=:rk4ip)
end

@testset "Public full-field MMGNLSE contract" begin
    time = TimeGrid(16, 1.0)
    transverse = TransverseGrid(Nx=8, Ny=6, dx=1.2e-6, dy=1.5e-6)
    dofs = degrees_of_freedom(:time, :space)
    domain = MMGNLSEDomain(dofs, time; transverse_grid=transverse)
    @test domain.time_grid === time
    @test domain.transverse_grid === transverse
    @test full_field_shape(domain) == (16, 8, 6, 1)

    omega0 = 2pi * 2.99792458e-4 / 1.55e-6
    reference_index = 1.45
    beta0 = reference_index * omega0 / 2.99792458e-4
    beta = TaylorBeta([beta0, 4.9e3, -2.1e-2, 1.2e-4])
    potential = [-(x^2 + y^2) * 1e12 for x in transverse.x,
                                              y in transverse.y]
    parameters = FullFieldMMGNLSEParameters(
        domain;
        length=1e-3,
        alpha=0.2,
        gain=zeros(time.nt),
        beta,
        potential,
        n2=2.3e-20,
        omega0,
        self_steepening=false,
    )
    @test parameters.reference_beta == beta0
    @test isapprox(parameters.reference_index, reference_index; rtol=1e-14)
    @test !parameters.self_steepening
    @test size(parameters.potential) == (8, 6)
    @test all(parameters.alpha .== 0.2)

    dispersion = PulsePropagation._full_field_temporal_dispersion(parameters)
    omega = frequency_axis(time)
    expected = beta.coefficients[3] .* omega .^ 2 ./ 2 .+
               beta.coefficients[4] .* omega .^ 3 ./ 6
    @test maximum(abs.(dispersion .- expected)) < 1e-10

    index_yx = fill(reference_index, length(transverse.y),
                    length(transverse.x))
    @test maximum(abs, paraxial_index_potential(
        domain, index_yx; beta, omega0)) < 1e-8

    field = zeros(ComplexF64, full_field_shape(domain))
    @test size(full_field_power(field, domain)) == (time.nt, 1)
    mock_solution = FullFieldMMGNLSESolution(
        z=[0.0],
        fields=reshape(field, size(field)..., 1),
        parameters=parameters,
        initial_field=field,
        dz=1e-5,
        method=RK4IP(),
    )
    @test size(get_spatiotemporal_field(mock_solution)) == size(field)
    @test size(get_spatial_field(mock_solution; time=0.0)) == (8, 6)
    @test size(get_spatial_field(mock_solution; wavelength=1.55e-6)) ==
          (8, 6)
    @test_throws DimensionMismatch solve_mmgnlse_full_field(
        dropdims(field; dims=4), parameters, 1e-5)
    @test_throws ArgumentError solve_mmgnlse_full_field(
        field, parameters, 1e-5; backend=:cpu)
    @test_throws ArgumentError solve_mmgnlse_full_field(
        field, parameters, 1e-5; method=:rk4ip)

    polarization_domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space, :polarization), time;
        transverse_grid=transverse)
    @test_throws ArgumentError FullFieldMMGNLSEParameters(
        polarization_domain;
        length=1e-3, beta, n2=0, omega0)
    @test_throws ArgumentError FullFieldMMGNLSEParameters(
        MMGNLSEDomain(dofs, time);
        length=1e-3, beta, n2=0, omega0)

    # The established positional domain constructor remains unchanged.
    modal_domain = MMGNLSEDomain(degrees_of_freedom(:time), time)
    @test modal_domain.time_grid === time
    @test modal_domain.transverse_grid === nothing
    modal_parameters = MMGNLSEParameters(
        modal_domain;
        length=1e-3,
        beta=TaylorBeta([0.0]),
        S=1.0,
        n2=0.0,
        omega0,
    )
    @test PulsePropagation.parameter_field_shape(modal_parameters) ==
          (time.nt, 1, 1)
end

println("validate_full_field_host: PASS")
