using LinearAlgebra
using PulsePropagation
using Test

@testset "Generic dense-free mode-overlap CP" begin
    x = collect(range(-8e-6, 8e-6; length=65))
    y = collect(range(-7e-6, 7e-6; length=61))
    fields = Array{Float64}(undef, length(y), length(x), 3)
    @inbounds for ix in eachindex(x), iy in eachindex(y)
        envelope = exp(-((x[ix] / 4e-6)^2 + (y[iy] / 3.5e-6)^2) / 2)
        fields[iy, ix, 1] = envelope
        fields[iy, ix, 2] = (x[ix] / 4e-6) * envelope
        fields[iy, ix, 3] = (y[iy] / 3.5e-6) * envelope
    end
    PulsePropagation._normalize_modes!(fields, x, y)

    working = mode_overlap_quadrature_cp(
        fields, x, y; quadrature_order=20)
    authority = mode_overlap_quadrature_cp(
        fields, x, y; quadrature_order=28)
    quadrature_error = cp_relative_error(working, authority)
    @test cp_rank(working) <= 20^2
    @test quadrature_error < 2e-5

    dense = PulsePropagation._spatial_overlap(fields, x, y).values
    reconstructed = cp_reconstruct_mmgnlse(authority)
    @test norm(reconstructed - dense) / norm(dense) < 2e-3

    compressed = cp_compress_mode_overlap(
        fields, x, y;
        target_error=5e-4,
        quadrature_order=20,
        authority_order=28,
        quadrature_tolerance=2e-5,
        max_rank=20,
        rank_step=2,
        maxiter=100,
        check_every=2,
        restarts=1,
    )
    @test compressed isa MMGNLSECPDecomposition
    @test compressed.converged
    @test compressed.relative_error <= 5e-4
    @test compressed.metadata.source_checksum !== nothing

    fiber = GRIN(lambda0=1550e-9, core_radius=25e-6, NA=0.08)
    properties = compute_fiber_properties(
        fiber, Silica(), 3;
        grid_size=33,
        overlap_representation=:quadrature_cp,
        overlap_quadrature_order=12,
        beta_order=2,
        dispersion_samples=7,
        dispersion_fit_order=4,
    )
    @test properties.S isa MMGNLSECPDecomposition
    @test cp_rank(properties.S) <= 12^2
    @test spatial_overlap(properties) === properties.S

    mktempdir() do directory
        data_path = joinpath(directory, "fiber_cp.jld2")
        metadata_path = joinpath(directory, "fiber_cp.toml")
        save_fiber_properties(data_path, metadata_path, properties)
        restored = load_fiber_properties(data_path, metadata_path)
        @test restored.S isa MMGNLSECPDecomposition
        @test cp_relative_error(restored.S, properties.S) < 1e-12
        @test fiber_properties_checksum(restored) ==
              fiber_properties_checksum(properties)
    end
end

println("validate_generic_mode_cp: PASS")
