using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Test

CUDA.functional() || error("CUDA is not functional.")
const CUDA_EXT = Base.get_extension(
    PulsePropagation, :PulsePropagationCUDAExt)
CUDA_EXT === nothing && error("PulsePropagation CUDA extension did not load.")

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))

rng = MersenneTwister(0x6d656d)
nmodes = parse(Int, get(ENV, "CP_MEMORY_TEST_MODES", "8"))
rank = parse(Int, get(ENV, "CP_MEMORY_TEST_RANK", "5"))
factor = randn(rng, Float64, nmodes, rank)
weights = randn(rng, Float64, rank)
exact = MMGNLSECPDecomposition(
    weights,
    ntuple(_ -> copy(factor), 4);
    layout=:spatial,
    nmodes,
    npolarizations=1,
)
source = cp_reconstruct_mmgnlse(exact)

@testset "Overlap construction and ownership controls" begin
    x = collect(range(-1, 1; length=5))
    y = collect(range(-1, 1; length=4))
    fields = randn(rng, Float64, length(x), length(y), 3)
    overlap_small_workspace = PulsePropagation._spatial_overlap(
        fields, x, y;
        chunk_points=3,
        max_workspace_bytes=3 * 3^2 * sizeof(Float64),
    )
    overlap_large_workspace = PulsePropagation._spatial_overlap(
        fields, x, y;
        chunk_points=length(x) * length(y),
        max_workspace_bytes=nothing,
    )
    @test overlap_small_workspace.values ≈
          overlap_large_workspace.values rtol=2e-14 atol=2e-14
    @test_throws ArgumentError PulsePropagation._spatial_overlap(
        fields, x, y;
        max_tensor_bytes=3^4 * sizeof(Float64) - 1,
    )

    raw_overlap = copy(overlap_small_workspace.values)
    domain = MMGNLSEDomain(degrees_of_freedom(:space))
    common_parameters = (
        length=1.0,
        beta=TaylorBeta(zeros(1, 3)),
        n2=1.0,
        omega0=1.0,
    )
    retained = MMGNLSEParameters(
        domain; S=raw_overlap, copy_overlap=false, common_parameters...)
    copied = MMGNLSEParameters(
        domain; S=raw_overlap, common_parameters...)
    @test retained.S.values === raw_overlap
    @test copied.S.values !== raw_overlap
end

@testset "CP compression memory controls" begin
    estimate = cp_compression_memory_estimate(
        nmodes, rank; symmetric=true, rank_block_size=1)
    @test estimate.blocked_khatri_rao_bytes == nmodes^3 * sizeof(Float64)
    @test estimate.blocked_khatri_rao_bytes <=
          estimate.full_khatri_rao_bytes

    blocked = cp_compress_symmetric_experimental(
        source, rank;
        initial=exact,
        backend=:cuda,
        maxiter=3,
        check_every=1,
        learning_rate=1e-6,
        ridge=1e-12,
        tolerance=0,
        rank_block_size=1,
        workspace_memory_fraction=0.25,
        verbose=true,
    )
    unblocked = cp_compress_symmetric_experimental(
        source, rank;
        initial=exact,
        backend=:cuda,
        maxiter=3,
        check_every=1,
        learning_rate=1e-6,
        ridge=1e-12,
        tolerance=0,
        rank_block_size=rank,
        workspace_memory_fraction=0.5,
        verbose=true,
    )
    @test blocked.relative_error ≈ unblocked.relative_error rtol=1e-11 atol=1e-12
    @test relative_l2(
        cp_reconstruct_mmgnlse(blocked),
        cp_reconstruct_mmgnlse(unblocked),
    ) <= 1e-11

    warm_blocked, history_blocked = PulsePropagation.cp_als_warm_cuda(
        source, rank;
        init=exact,
        maxiter=2,
        check_every=1,
        rng=MersenneTwister(99),
        rank_block_size=1,
        symmetric_source=true,
    )
    warm_unblocked, history_unblocked = PulsePropagation.cp_als_warm_cuda(
        source, rank;
        init=exact,
        maxiter=2,
        check_every=1,
        rng=MersenneTwister(99),
        rank_block_size=rank,
        symmetric_source=true,
    )
    @test history_blocked ≈ history_unblocked rtol=1e-7 atol=5e-8
    @test relative_l2(
        PulsePropagation.cp_reconstruct(warm_blocked, size(source)),
        PulsePropagation.cp_reconstruct(warm_unblocked, size(source)),
    ) <= 1e-11

    complex_blocked = CUDA_EXT._cp_als_complex_cuda(
        source, rank;
        seed=101,
        initial=exact,
        maxiter=2,
        check_every=1,
        rank_block_size=1,
    )
    complex_unblocked = CUDA_EXT._cp_als_complex_cuda(
        source, rank;
        seed=101,
        initial=exact,
        maxiter=2,
        check_every=1,
        rank_block_size=rank,
    )
    @test complex_blocked[3] ≈ complex_unblocked[3] rtol=1e-7 atol=5e-8
    @test relative_l2(
        cp_reconstruct_mmgnlse(MMGNLSECPDecomposition(
            complex_blocked[1], complex_blocked[2])),
        cp_reconstruct_mmgnlse(MMGNLSECPDecomposition(
            complex_unblocked[1], complex_unblocked[2])),
    ) <= 1e-10

    @test_throws ArgumentError cp_compress_symmetric_experimental(
        source, rank;
        initial=exact,
        backend=:cuda,
        maxiter=1,
        max_workspace_bytes=nmodes^3 * sizeof(Float64) - 1,
    )
end

println(
    "gpu=", CUDA.name(CUDA.device()),
    " free_bytes=", CUDA.free_memory(),
    " nmodes=", nmodes,
    " rank=", rank,
)
