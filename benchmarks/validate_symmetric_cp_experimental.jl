using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Test

const EXT = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
EXT === nothing && error("PulsePropagation CUDA extension did not load.")
CUDA.functional() || error("CUDA is not functional.")

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))

rng = MersenneTwister(0x51cc)
nm, rank = 4, 3
factor = randn(rng, Float64, nm, rank)
weights = [1.0, 0.7, -0.4]
exact_cp = MMGNLSECPDecomposition(
    weights,
    ntuple(_ -> copy(factor), 4);
    layout=:spatial,
    nmodes=nm,
    npolarizations=1,
)
source = cp_reconstruct_mmgnlse(exact_cp)

@testset "experimental symmetric CP fitting API" begin
    @test cp_is_symmetric(exact_cp)
    fitted_cpu = cp_compress_symmetric_experimental(
        source, rank;
        initial=exact_cp,
        backend=:cpu,
        maxiter=5,
        check_every=1,
        learning_rate=1e-6,
        ridge=1e-12,
        tolerance=0,
        target_error=1e-8,
    )
    @test cp_is_symmetric(fitted_cpu)
    @test fitted_cpu.relative_error <= 1e-7

    fitted_cuda = cp_compress_symmetric_experimental(
        source, rank;
        initial=exact_cp,
        backend=:cuda,
        maxiter=5,
        check_every=1,
        learning_rate=1e-6,
        ridge=1e-12,
        tolerance=0,
        target_error=1e-8,
    )
    @test cp_is_symmetric(fitted_cuda)
    @test fitted_cuda.relative_error <= 1e-7
    @test relative_l2(
        cp_reconstruct_mmgnlse(fitted_cuda), source) <= 1e-7

    nonsymmetric = copy(source)
    nonsymmetric[1, 2, 3, 4] += 1e-2
    @test_throws ArgumentError cp_compress_symmetric_experimental(
        nonsymmetric, rank; initial=exact_cp, maxiter=1)
end

solver_cp = MMGNLSECPDecomposition(
    2e12 .* exact_cp.λ,
    exact_cp.U;
    layout=:spatial,
    nmodes=nm,
    npolarizations=1,
)
nt = 64
domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space), TimeGrid(nt, 5.0))
parameters = MMGNLSEParameters(
    domain;
    length=2e-4,
    beta=TaylorBeta(zeros(4, nm)),
    S=solver_cp,
    n2=2.3e-20,
    omega0=2π * 2.99792458e-4 / 1550e-9,
    raman=AnisotropicRaman(),
)
initial = 10 .* randn(rng, ComplexF64, nt, nm, 1)
terminal = randn(rng, ComplexF64, nt, nm, 1)
terminal ./= norm(terminal)
dz = 2e-5

@testset "experimental symmetric CUDA backend" begin
    baseline = solve_mmgnlse(
        initial, parameters, dz;
        backend=:cuda,
        precision=:float64,
        saveat=:steps,
    )
    experimental = solve_mmgnlse(
        initial, parameters, dz;
        backend=:cuda_cp_symmetric_experimental,
        precision=:float64,
        saveat=:steps,
    )
    @test experimental.cache.backend ===
          :cuda_cp_symmetric_experimental
    @test relative_l2(experimental.fields, baseline.fields) <= 2e-11

    mixed = solve_mmgnlse(
        initial, parameters, dz;
        backend=:cuda_cp_symmetric_experimental,
        precision=:mixed,
        saveat=:steps,
    )
    @test relative_l2(mixed.fields, experimental.fields) <= 2e-5

    adaptive = solve_mmgnlse(
        initial, parameters, 5dz;
        backend=:cuda_cp_symmetric_experimental,
        precision=:float64,
        adaptive=true,
        reltol=1e-6,
        abstol=1e-9,
        dzmax=5dz,
        saveat=parameters.length,
    )
    @test adaptive.cache.adaptive
    @test adaptive.cache.accepted_steps > 0

    baseline_tsit5 = solve_mmgnlse(
        initial, parameters, dz;
        method=Tsit5(),
        backend=:cuda,
        saveat=:steps,
    )
    experimental_tsit5 = solve_mmgnlse(
        initial, parameters, dz;
        method=Tsit5(),
        backend=:cuda_cp_symmetric_experimental,
        saveat=:steps,
    )
    @test relative_l2(
        experimental_tsit5.fields, baseline_tsit5.fields) <= 2e-11

    legacy_cache = EXT._cuda_mmgnlse_forward_cache(
        parameters, :rank_tuned, Float64)
    experimental_cache = EXT._cuda_mmgnlse_forward_cache(
        parameters, :symmetric_experimental, Float64)
    @test !EXT._cuda_mmgnlse_cp_use_real_gemm(legacy_cache.overlap)
    @test EXT._cuda_mmgnlse_cp_use_real_gemm(experimental_cache.overlap)
    workspace = EXT._cuda_mmgnlse_cp_forward_workspace(
        CuArray(initial), experimental_cache.overlap)
    @test workspace.rfft_rank! !== nothing
    @test workspace.irfft_rank! !== nothing

    @test_throws ArgumentError solve_adjoint(
        terminal, parameters, dz;
        forward_solution=baseline,
        backend=:cuda_cp_symmetric_experimental,
        saveat=:steps,
    )

    untied_factors = ntuple(index -> begin
        candidate = copy(exact_cp.U[index])
        index == 4 && (candidate[1, 1] += eps(Float64))
        candidate
    end, 4)
    untied_cp = MMGNLSECPDecomposition(
        exact_cp.λ, untied_factors;
        layout=:spatial, nmodes=nm, npolarizations=1)
    untied_parameters = MMGNLSEParameters(
        domain;
        length=parameters.length,
        beta=parameters.beta,
        S=untied_cp,
        n2=parameters.n2,
        omega0=parameters.omega0,
        raman=parameters.raman,
    )
    @test_throws ArgumentError solve_mmgnlse(
        initial, untied_parameters, dz;
        backend=:cuda_cp_symmetric_experimental)
end
