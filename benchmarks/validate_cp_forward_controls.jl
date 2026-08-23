using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Test

CUDA.functional() || error("CUDA is not functional.")

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))

function timed_solve(args...; kwargs...)
    CUDA.synchronize()
    start = time_ns()
    solution = solve_mmgnlse(args...; kwargs...)
    CUDA.synchronize()
    return solution, (time_ns() - start) / 1e9
end

rng = MersenneTwister(0xada47)
nt, nm, rank = 128, 3, 8
length_m = 2e-3
domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space), TimeGrid(nt, 5.0))
factor = randn(rng, Float64, nm, rank) ./ sqrt(nm)
cp = MMGNLSECPDecomposition(
    fill(2e12, rank),
    (copy(factor), copy(factor), copy(factor), copy(factor));
    layout=:spatial,
    nmodes=nm,
    npolarizations=1,
)
beta_coefficients = zeros(4, nm)
beta_coefficients[2, :] .= [0.0, 0.2, -0.15]
beta_coefficients[3, :] .= [-0.02, -0.024, -0.017]
parameters = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=TaylorBeta(beta_coefficients),
    S=cp,
    n2=2.3e-20,
    omega0=2π * 2.99792458e-4 / 1550e-9,
    raman=AnisotropicRaman(),
)
dense_parameters = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=TaylorBeta(beta_coefficients),
    S=SpatialOverlap(cp_reconstruct_mmgnlse(cp)),
    n2=2.3e-20,
    omega0=2π * 2.99792458e-4 / 1550e-9,
    raman=AnisotropicRaman(),
)
dense_no_raman_parameters = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=TaylorBeta(beta_coefficients),
    S=SpatialOverlap(cp_reconstruct_mmgnlse(cp)),
    n2=2.3e-20,
    omega0=2π * 2.99792458e-4 / 1550e-9,
    raman=NoRaman(),
)
coefficients = ComplexF64[1.0, 0.7im, -0.5]
initial = gaussian_pulse(
    domain, nm;
    peak_power=1e5,
    fwhm=0.25,
    time_offset=-0.3,
    coefficients,
)

# Compile every new route before recording validation timings.
solve_mmgnlse(initial, parameters, 2e-5;
    backend=:cuda_cp_optimized, precision=:float64)
solve_mmgnlse(initial, parameters, 2e-5;
    backend=:cuda_cp_optimized, precision=:mixed)
solve_mmgnlse(initial, parameters, 1e-4;
    backend=:cuda_cp_optimized, adaptive=true,
    reltol=1e-5, abstol=1e-8)
solve_mmgnlse(initial, dense_parameters, 2e-5;
    backend=:cuda, precision=:mixed)
solve_mmgnlse(initial, dense_parameters, 1e-4;
    backend=:cuda, adaptive=true,
    reltol=1e-5, abstol=1e-8)

reference, reference_seconds = timed_solve(
    initial, parameters, 1e-6;
    backend=:cuda_cp_optimized,
    precision=:float64,
)
default_solution, default_seconds = timed_solve(
    initial, parameters, 1e-5;
    backend=:cuda_cp_optimized,
)
fp64_solution, fp64_seconds = timed_solve(
    initial, parameters, 1e-5;
    backend=:cuda_cp_optimized,
    precision=:float64,
)
mixed_solution, mixed_seconds = timed_solve(
    initial, parameters, 1e-5;
    backend=:cuda_cp_optimized,
    precision=:mixed,
)
adaptive_solution, adaptive_seconds = timed_solve(
    initial, parameters, 1e-4;
    backend=:cuda_cp_optimized,
    precision=:float64,
    adaptive=true,
    reltol=1e-6,
    abstol=1e-9,
    dzmax=5e-4,
)
adaptive_mixed_solution, adaptive_mixed_seconds = timed_solve(
    initial, parameters, 1e-4;
    backend=:cuda_cp_optimized,
    precision=:mixed,
    adaptive=true,
    reltol=1e-6,
    abstol=1e-9,
    dzmax=5e-4,
)
rejection_solution = solve_mmgnlse(
    initial, parameters, length_m;
    backend=:cuda_cp_optimized,
    adaptive=true,
    reltol=1e-10,
    abstol=1e-13,
    dzmax=length_m,
)
saved_solution = solve_mmgnlse(
    initial, parameters, 1e-4;
    backend=:cuda_cp_optimized,
    adaptive=true,
    reltol=1e-6,
    abstol=1e-9,
    saveat=[0.0, length_m / 3, length_m],
)
dense_fp64_solution = solve_mmgnlse(
    initial, dense_parameters, 1e-5;
    backend=:cuda,
    precision=:float64,
)
dense_mixed_solutions = Dict(
    backend => solve_mmgnlse(
        initial, dense_parameters, 1e-5;
        backend,
        precision=:mixed,
    ) for backend in (:cuda, :cuda_optimized)
)
dense_adaptive_solutions = Dict(
    backend => solve_mmgnlse(
        initial, dense_parameters, 1e-4;
        backend,
        adaptive=true,
        reltol=1e-6,
        abstol=1e-9,
        dzmax=5e-4,
    ) for backend in (:cuda, :cuda_optimized)
)
dense_no_raman_mixed = solve_mmgnlse(
    initial, dense_no_raman_parameters, 1e-5;
    backend=:cuda,
    precision=:mixed,
)
dense_no_raman_adaptive = solve_mmgnlse(
    initial, dense_no_raman_parameters, 1e-4;
    backend=:cuda,
    adaptive=true,
    reltol=1e-6,
    abstol=1e-9,
    dzmax=5e-4,
)
cp_mixed_solutions = Dict(
    backend => solve_mmgnlse(
        initial, parameters, 1e-5;
        backend,
        precision=:mixed,
    ) for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
)
cp_adaptive_solutions = Dict(
    backend => solve_mmgnlse(
        initial, parameters, 1e-4;
        backend,
        adaptive=true,
        reltol=1e-6,
        abstol=1e-9,
        dzmax=5e-4,
    ) for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
)
cp_tsit5_solutions = Dict(
    backend => solve_mmgnlse(
        initial, parameters, 1e-5;
        backend,
        method=Tsit5(),
    ) for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
)

@testset "forward CUDA precision and adaptivity" begin
    @test default_solution.fields == fp64_solution.fields
    @test default_solution.cache.precision === :float64
    @test !default_solution.cache.adaptive
    @test mixed_solution.cache.precision === :mixed
    @test eltype(mixed_solution.fields) === ComplexF64
    @test adaptive_solution.cache.adaptive
    @test adaptive_solution.cache.accepted_steps > 0
    @test adaptive_solution.cache.rhs_evaluations ==
          1 + 4 * (adaptive_solution.cache.accepted_steps +
                   adaptive_solution.cache.rejected_steps)
    @test rejection_solution.cache.rejected_steps > 0
    @test saved_solution.z == [0.0, length_m / 3, length_m]
    @test size(saved_solution.fields, 4) == 3
    for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
        cp_mixed = cp_mixed_solutions[backend]
        cp_adaptive = cp_adaptive_solutions[backend]
        @test cp_mixed.cache.backend === backend
        @test cp_mixed.cache.precision === :mixed
        @test cp_adaptive.cache.backend === backend
        @test cp_adaptive.cache.adaptive
        @test cp_adaptive.cache.accepted_steps > 0
        @test relative_l2(
            cp_tsit5_solutions[backend].fields,
            cp_tsit5_solutions[:cuda].fields,
        ) <= 2e-11
    end
    for backend in (:cuda, :cuda_optimized)
        dense_mixed = dense_mixed_solutions[backend]
        dense_adaptive = dense_adaptive_solutions[backend]
        @test dense_mixed.cache.backend === backend
        @test dense_mixed.cache.precision === :mixed
        @test eltype(dense_mixed.fields) === ComplexF64
        @test relative_l2(
            dense_mixed.fields, dense_fp64_solution.fields) <= 1e-3
        @test dense_adaptive.cache.backend === backend
        @test dense_adaptive.cache.adaptive
        @test dense_adaptive.cache.accepted_steps > 0
        @test dense_adaptive.cache.rhs_evaluations ==
              1 + 4 * (dense_adaptive.cache.accepted_steps +
                       dense_adaptive.cache.rejected_steps)
    end
    @test dense_no_raman_mixed.cache.precision === :mixed
    @test dense_no_raman_adaptive.cache.adaptive
    @test dense_no_raman_adaptive.cache.accepted_steps > 0
    @test_throws ArgumentError solve_mmgnlse(
        initial, parameters, 1e-4;
        backend=:cuda_cp_optimized,
        adaptive=true,
        saveat=:steps,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, dense_parameters, 1e-4;
        backend=:cuda_cp_optimized,
        precision=:mixed,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, parameters, 1e-4;
        backend=:cpu,
        precision=:mixed,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, parameters, 1e-4;
        backend=:cpu,
        adaptive=true,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, parameters, 1e-4;
        backend=:cuda_optimized,
        method=Tsit5(),
        precision=:mixed,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, parameters, 1e-4;
        backend=:cuda_optimized,
        method=Tsit5(),
        adaptive=true,
    )
end

println("gpu=", CUDA.name(CUDA.device()))
println("reference_seconds=", reference_seconds)
for (label, solution, seconds) in (
    ("fixed_default", default_solution, default_seconds),
    ("fixed_fp64", fp64_solution, fp64_seconds),
    ("fixed_mixed", mixed_solution, mixed_seconds),
    ("adaptive_fp64", adaptive_solution, adaptive_seconds),
    ("adaptive_mixed", adaptive_mixed_solution, adaptive_mixed_seconds),
)
    println(
        "variant=", label,
        " seconds=", seconds,
        " relative_l2_vs_reference=",
        relative_l2(solution.fields, reference.fields),
        " accepted_steps=", solution.cache.accepted_steps,
        " rejected_steps=", solution.cache.rejected_steps,
        " rhs_evaluations=", solution.cache.rhs_evaluations,
    )
end
