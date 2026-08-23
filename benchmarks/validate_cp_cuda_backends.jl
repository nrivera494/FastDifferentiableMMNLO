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

rng = MersenneTwister(0xc0ba)
nt, nm, rank = 32, 3, 12
domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space), TimeGrid(nt, 5.0))

# A real, permutation-symmetric CP tensor exercises both the CP backends and
# the optimized physical dense-spatial CUDA kernels after reconstruction.
factor = randn(rng, Float64, nm, rank) ./ sqrt(nm)
cp = MMGNLSECPDecomposition(
    ones(Float64, rank),
    (copy(factor), copy(factor), copy(factor), copy(factor));
    layout=:spatial,
    nmodes=nm,
    npolarizations=1,
)
dense = SpatialOverlap(cp_reconstruct_mmgnlse(cp))
common = (
    length=1e-5,
    beta=TaylorBeta(zeros(4, nm)),
    n2=2.3e-20,
    omega0=2π * 2.99792458e-4 / 1550e-9,
    raman=AnisotropicRaman(),
)
cp_parameters = MMGNLSEParameters(domain; S=cp, common...)
dense_parameters = MMGNLSEParameters(domain; S=dense, common...)
initial = 10 .* randn(rng, ComplexF64, nt, nm, 1)
terminal = randn(rng, ComplexF64, nt, nm, 1)
terminal ./= norm(terminal)
dz = cp_parameters.length

@testset "CUDA CP backend policies" begin
    cpu_forward = solve_mmgnlse(
        initial, cp_parameters, dz; backend=:cpu, saveat=:steps)
    cp_forwards = Dict{Symbol,Any}()
    for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
        solution = solve_mmgnlse(
            initial, cp_parameters, dz; backend, saveat=:steps)
        @test solution.cache.backend === backend
        @test relative_l2(solution.fields, cpu_forward.fields) <= 2e-11
        cp_forwards[backend] = solution
    end
    @test relative_l2(
        cp_forwards[:cuda_cp_optimized].fields,
        cp_forwards[:cuda].fields,
    ) <= 2e-11
    @test relative_l2(
        cp_forwards[:cuda_optimized].fields,
        cp_forwards[:cuda].fields,
    ) <= 2e-11

    cpu_adjoint = solve_adjoint(
        terminal, cp_parameters, dz;
        forward_solution=cpu_forward, backend=:cpu, saveat=:steps)
    cp_adjoints = Dict{Symbol,Any}()
    for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
        solution = solve_adjoint(
            terminal, cp_parameters, dz;
            forward_solution=cpu_forward, backend, saveat=:steps)
        @test solution.cache.backend === backend
        @test relative_l2(solution.fields, cpu_adjoint.fields) <= 2e-11
        cp_adjoints[backend] = solution
    end
    @test relative_l2(
        cp_adjoints[:cuda_cp_optimized].fields,
        cp_adjoints[:cuda].fields,
    ) <= 2e-11
    @test relative_l2(
        cp_adjoints[:cuda_optimized].fields,
        cp_adjoints[:cuda].fields,
    ) <= 2e-11
end

@testset "Dense CUDA routing and CP-only rejection" begin
    @test_throws ArgumentError solve_mmgnlse(
        initial, dense_parameters, dz;
        backend=:cuda_cp_optimized)

    dense_cpu = solve_mmgnlse(
        initial, dense_parameters, dz; backend=:cpu, saveat=:steps)
    dense_cuda = solve_mmgnlse(
        initial, dense_parameters, dz; backend=:cuda, saveat=:steps)
    dense_cuda_optimized = solve_mmgnlse(
        initial, dense_parameters, dz;
        backend=:cuda_optimized, saveat=:steps)
    @test dense_cuda.cache.backend === :cuda
    @test dense_cuda_optimized.cache.backend === :cuda_optimized
    @test relative_l2(dense_cuda.fields, dense_cpu.fields) <= 2e-11
    @test relative_l2(
        dense_cuda_optimized.fields, dense_cuda.fields) <= 2e-11

    dense_cache = EXT._cuda_mmgnlse_forward_cache(
        dense_parameters, :baseline)
    @test dense_cache.overlap isa EXT.CUDAMMGNLSEDenseSpatial
    @test dense_cache.overlap.physical_symmetric
    dense_workspace = EXT._cuda_mmgnlse_dense_raman_forward_workspace(
        CuArray(initial), dense_cache.overlap)
    @test dense_workspace isa EXT.CUDAMMGNLSEDenseRamanForwardWorkspace

    @test_throws ArgumentError solve_adjoint(
        terminal, dense_parameters, dz;
        forward_solution=dense_cpu,
        backend=:cuda_cp_optimized)
end
