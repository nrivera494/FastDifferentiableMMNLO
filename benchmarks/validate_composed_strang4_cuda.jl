using CUDA
using LinearAlgebra
using PulsePropagation
using Test

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const EXT_COMPOSED = Base.get_extension(
    PulsePropagation, :PulsePropagationCUDAExt)
const C_M_PER_PS_COMPOSED = 2.99792458e-4
const OMEGA0_COMPOSED =
    2pi * C_M_PER_PS_COMPOSED / 1.55e-6

relative_error_composed(candidate, reference) =
    norm(vec(candidate .- reference)) / norm(vec(reference))

function composed_test_problem(; length_m=4.0e-4)
    time = TimeGrid(64, 2.0)
    grid = FullField3DGrid(time; Nx=16, Ny=16, dx=3.0e-6)
    t = time_axis(time)
    temporal = sqrt(4.0e5) .* exp.(
        -2log(2) .* (t ./ 0.28) .^ 2) .*
        cis.(0.35 .* (t ./ 0.28) .^ 2)
    spatial = ComplexF64[
        exp(-0.5 * ((x / 4.5e-6)^2 + (y / 3.8e-6)^2)) *
        cis(0.22 * x / 4.5e-6 - 0.17 * y / 3.8e-6)
        for x in grid.x, y in grid.y
    ]
    spatial ./= sqrt(sum(abs2, spatial) * grid.dx * grid.dy)
    field = reshape(temporal, :, 1, 1) .* reshape(
        spatial, 1, length(grid.x), length(grid.y))

    beta0 = 8.0e6
    oscillator_rate = 2.2e3
    potential = [
        -0.5 * beta0 * oscillator_rate^2 *
        ((1.05x)^2 + (0.90y)^2) + 180.0 * x / 8e-6
        for x in grid.x, y in grid.y
    ]
    dispersion = -0.012 .* frequency_axis(time) .^ 2
    parameters = FullField3DParameters(
        grid;
        length=length_m,
        beta0,
        dispersion,
        potential,
        n2=2.3e-20,
        omega0=OMEGA0_COMPOSED,
        raman=AgarwalRaman(),
        self_steepening=true,
    )
    return field, parameters
end

function composed_solution(nsteps, method)
    field, parameters = composed_test_problem()
    solution = solve_full_field_3d(
        field, parameters, parameters.length / nsteps;
        precision=:float64, method)
    return Array(@view solution.fields[:, :, :, end]), solution
end

@testset "ComposedStrang4 public dispatch" begin
    time = TimeGrid(16, 1.0)
    transverse = TransverseGrid(Nx=8, Ny=8, dx=3e-6)
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), time;
        transverse_grid=transverse)
    parameters = FullFieldMMGNLSEParameters(
        domain;
        length=2e-5,
        beta=TaylorBeta([8e6, 0.0, -0.02]),
        potential=[200.0 - 1e11 * (x^2 + y^2)
                   for x in transverse.x, y in transverse.y],
        n2=2.3e-20,
        omega0=OMEGA0_COMPOSED,
        raman=AgarwalRaman(),
        self_steepening=true,
    )
    initial = fill(ComplexF64(2e7, -1e7), full_field_shape(domain))
    solution = solve_mmgnlse_full_field(
        initial, parameters, 1e-5;
        method=ComposedStrang4(), precision=:float64)
    @test solution.method isa ComposedStrang4
    @test solution.cache.method == :composed_strang4
    @test solution.cache.self_steepening
    @test solution.cache.residual_rhs_evaluations == 40
end

@testset "Shock midpoint near-reversibility" begin
    field, parameters = composed_test_problem(; length_m=1e-5)
    workspace = EXT_COMPOSED._cuda_full_field_workspace(
        field, parameters, Float64, :composed_strang4)
    initial = copy(workspace.field)
    EXT_COMPOSED._cuda_full_field_shock_midpoint!(workspace, 1e-5)
    EXT_COMPOSED._cuda_full_field_shock_midpoint!(workspace, -1e-5)
    CUDA.synchronize()
    reversal_error = relative_error_composed(
        Array(workspace.field), Array(initial))
    println("shock midpoint reversal error = ", reversal_error)
    @test reversal_error < 2e-9
end

@testset "ComposedStrang4 longitudinal order" begin
    reference, _ = composed_solution(256, :rk4ip)
    reference_half, _ = composed_solution(128, :rk4ip)
    reference_gap = relative_error_composed(reference_half, reference)

    steps = (4, 8, 16, 32)
    fields = map(n -> first(composed_solution(
        n, :composed_strang4)), steps)
    errors = map(field -> relative_error_composed(field, reference), fields)
    observed_orders = log2.(errors[1:end-1] ./ errors[2:end])

    println("reference gap = ", reference_gap)
    println("composed errors = ", errors)
    println("composed observed orders = ", observed_orders)
    @test all(diff(collect(errors)) .< 0)
    @test reference_gap < 0.1 * errors[end]
    @test observed_orders[end] > 3.3
end

println(
    "validate_composed_strang4_cuda: PASS gpu=",
    CUDA.name(CUDA.device()),
)
