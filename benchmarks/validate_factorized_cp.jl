using LinearAlgebra
using PulsePropagation
using Random
using Test

fiber = GRIN(
    lambda0=1700e-9,
    core_radius=40e-6,
    NA=sqrt(1.885^2 - 1.866^2),
)

@testset "Gauss-Hermite overlap CP" begin
    working = grin_hg_overlap_quadrature_cp(
        fiber, 6; quadrature_order=5)
    authority = grin_hg_overlap_quadrature_cp(
        fiber, 6; quadrature_order=7)
    @test cp_rank(working) == 25
    @test cp_rank(authority) == 49
    @test cp_is_symmetric(working)
    @test cp_relative_error(working, authority) <= 5e-8

    k0 = 2pi / fiber.lambda0
    expected_fundamental =
        (k0 * fiber.NA / fiber.core_radius) / (2pi)
    dense = cp_reconstruct_mmgnlse(working)
    @test isapprox(
        dense[1, 1, 1, 1], expected_fundamental; rtol=2e-14)
    @test isapprox(cp_norm(working), norm(dense); rtol=2e-14)
    @test_throws ArgumentError grin_hg_overlap_quadrature_cp(
        fiber, 6; quadrature_order=4)
end

@testset "Dense-free factorized CP compression" begin
    source = grin_hg_overlap_quadrature_cp(
        fiber, 6; quadrature_order=5)
    fitted = cp_compress(
        source;
        error=0.9,
        min_rank=2,
        max_rank=2,
        maxiter=30,
        check_every=2,
        tolerance=1e-9,
        seed=7,
    )
    verified_error = cp_relative_error(fitted, source)
    @test cp_rank(fitted) == 2
    @test fitted.relative_error <= 0.9
    @test isapprox(
        fitted.relative_error, verified_error; rtol=5e-13, atol=5e-13)

    rng = MersenneTwister(9)
    field = randn(rng, ComplexF64, 4, 6)
    source_action = PulsePropagation.cp_spatial_isotropic_cubic(field, source)
    dense_action = PulsePropagation.dense_spatial_isotropic_cubic(
        field, cp_reconstruct_mmgnlse(source))
    @test norm(source_action - dense_action) / norm(dense_action) <= 2e-14

    @test_throws PulsePropagation.CPCompressionFailure cp_compress(
        source;
        error=0.1,
        min_rank=2,
        max_rank=2,
        maxiter=5,
        check_every=1,
        seed=7,
    )
end

println("validate_factorized_cp: PASS")
