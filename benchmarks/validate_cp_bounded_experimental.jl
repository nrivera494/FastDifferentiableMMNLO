using LinearAlgebra
using PulsePropagation
using Random
using Test

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))

rng = MersenneTwister(0x626f756e64)
nmodes = parse(Int, get(ENV, "CP_BOUNDED_TEST_MODES", "4"))
factor = randn(rng, Float64, nmodes, 1)
exact = MMGNLSECPDecomposition(
    [1.25],
    ntuple(_ -> copy(factor), 4);
    layout=:spatial,
    nmodes,
)
source = real.(cp_reconstruct_mmgnlse(exact))
nonsymmetric = copy(source)
nonsymmetric[1, min(2, nmodes), 1, 1] += 0.25

@testset "Experimental bounded CP API" begin
    @test CPMemoryBudget().rank_block_size === :auto
    @test_throws ArgumentError CPMemoryBudget(rank_block_size=0)
    @test_throws ArgumentError CPMemoryBudget(rank_block_size=:full)
    @test_throws ArgumentError CPMemoryBudget(max_workspace_bytes=0)
    @test_throws ArgumentError CPMemoryBudget(workspace_memory_fraction=1.1)

    bytes_per_column_f64 = nmodes^3 * sizeof(ComplexF64)
    budget = CPMemoryBudget(
        rank_block_size=:auto,
        max_workspace_bytes=2bytes_per_column_f64,
        workspace_memory_fraction=0.4,
        reclaim_memory=false,
    )
    general_plan = cp_compression_plan(
        source, 5; source_structure=:general, memory=budget)
    symmetric_plan = cp_compression_plan(
        source, 5; source_structure=:permutation_symmetric, memory=budget)
    auto_plan = cp_compression_plan(
        source, 5; source_structure=:auto, memory=budget)
    float32_plan = cp_compression_plan(
        source, 5;
        fit_precision=:float32,
        source_structure=:permutation_symmetric,
        memory=CPMemoryBudget(max_workspace_bytes=bytes_per_column_f64),
    )

    @test general_plan.source_structure === :general
    @test general_plan.unfolding_bytes == general_plan.source_bytes
    @test symmetric_plan.source_structure === :permutation_symmetric
    @test symmetric_plan.unfolding_bytes == 0
    @test auto_plan.source_structure === :permutation_symmetric
    @test cp_compression_plan(
        nonsymmetric, 2; source_structure=:auto).source_structure === :general
    @test_throws ArgumentError cp_compression_plan(
        nonsymmetric, 2; source_structure=:permutation_symmetric)
    @test cp_compression_plan(
        nonsymmetric, 2;
        source_structure=:permutation_symmetric,
        verify_source_structure=false,
    ).source_structure === :permutation_symmetric
    @test general_plan.resolved_rank_block_size == 2
    @test general_plan.khatri_rao_workspace_bytes == 2bytes_per_column_f64
    @test float32_plan.source_bytes * 2 == symmetric_plan.source_bytes
    @test float32_plan.fit_precision === :float32
    @test float32_plan.unfolding_bytes == 0

    common = (
        error=2.0,
        min_rank=1,
        max_rank=1,
        maxiter=2,
        check_every=1,
        seed=2026,
        backend=:cpu,
    )
    legacy = cp_compress(source; common...)
    bounded = cp_compress_bounded(
        source;
        common...,
        memory=CPMemoryBudget(
            rank_block_size=1,
            workspace_memory_fraction=0.1,
            reclaim_memory=false,
        ),
        source_structure=:general,
    )
    @test bounded isa MMGNLSECPDecomposition
    @test bounded.relative_error ≈ legacy.relative_error rtol=1e-13 atol=1e-14
    @test relative_l2(
        cp_reconstruct_mmgnlse(bounded),
        cp_reconstruct_mmgnlse(legacy),
    ) <= 1e-12

    checksum_eltype = Ref{DataType}()
    bounded32 = cp_compress_bounded(
        source;
        common...,
        fit_precision=:float32,
        source_structure=:permutation_symmetric,
        checksum_fn=S -> begin
            checksum_eltype[] = eltype(S)
            "original-source"
        end,
        memory=CPMemoryBudget(rank_block_size=1),
    )
    @test checksum_eltype[] === Float64
    @test bounded32.metadata.source_checksum == "original-source"
    @test eltype(bounded32.U[1]) === ComplexF32
end

if get(ENV, "CP_BOUNDED_TEST_CUDA", "0") == "1"
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional.")

    @testset "Experimental bounded CP API on CUDA" begin
        budget = CPMemoryBudget(
            rank_block_size=:auto,
            max_workspace_bytes=nmodes^3 * sizeof(ComplexF32),
            workspace_memory_fraction=0.25,
            reclaim_memory=true,
        )
        plan = cp_compression_plan(
            source, 4;
            backend=:cuda,
            fit_precision=:float32,
            source_structure=:permutation_symmetric,
            memory=budget,
        )
        @test plan.free_device_bytes !== nothing
        @test plan.estimated_headroom_bytes >= 0
        @test plan.resolved_rank_block_size == 1
        @test plan.cuda_addressable

        fitted = cp_compress_bounded(
            source;
            error=2.0,
            min_rank=1,
            max_rank=1,
            maxiter=2,
            check_every=1,
            seed=2026,
            backend=:cuda,
            fit_precision=:float32,
            source_structure=:permutation_symmetric,
            memory=budget,
        )
        @test fitted isa MMGNLSECPDecomposition
        @test eltype(fitted.U[1]) === ComplexF32

        general_fit = cp_compress_bounded(
            source;
            error=2.0,
            min_rank=1,
            max_rank=1,
            maxiter=2,
            check_every=1,
            seed=2026,
            backend=:cuda,
            fit_precision=:float32,
            source_structure=:general,
            memory=budget,
        )
        @test general_fit isa MMGNLSECPDecomposition
        @test eltype(general_fit.U[1]) === ComplexF32
        CUDA.synchronize()
        CUDA.reclaim()
        println(
            "gpu=", CUDA.name(CUDA.device()),
            " free_bytes=", CUDA.free_memory(),
            " resolved_rank_block_size=", plan.resolved_rank_block_size,
        )
    end

    large_modes = parse(Int, get(ENV, "CP_BOUNDED_LARGE_MODES", "0"))
    if large_modes > 0
        @testset "Large trusted symmetric source on CUDA" begin
            large_factor = randn(rng, Float32, large_modes)
            large_source = reshape(large_factor, :, 1, 1, 1) .*
                           reshape(large_factor, 1, :, 1, 1) .*
                           reshape(large_factor, 1, 1, :, 1) .*
                           reshape(large_factor, 1, 1, 1, :)
            large_budget = CPMemoryBudget(
                rank_block_size=:auto,
                max_workspace_bytes=256 * 1024^2,
                workspace_memory_fraction=0.25,
                reclaim_memory=true,
            )
            large_plan = cp_compression_plan(
                large_source, 1;
                backend=:cuda,
                fit_precision=:float32,
                source_structure=:permutation_symmetric,
                verify_source_structure=false,
                memory=large_budget,
            )
            elapsed = @elapsed large_fit = cp_compress_bounded(
                large_source;
                error=2.0,
                min_rank=1,
                max_rank=1,
                maxiter=1,
                check_every=1,
                backend=:cuda,
                fit_precision=:float32,
                source_structure=:permutation_symmetric,
                verify_source_structure=false,
                memory=large_budget,
            )
            @test large_fit isa MMGNLSECPDecomposition
            @test eltype(large_fit.U[1]) === ComplexF32
            @test large_plan.unfolding_bytes == 0
            @test large_plan.estimated_headroom_bytes >= 0
            println(
                "large_modes=", large_modes,
                " source_bytes=", sizeof(large_source),
                " estimated_peak_bytes=", large_plan.estimated_peak_bytes,
                " elapsed_seconds=", elapsed,
                " relative_error=", large_fit.relative_error,
            )
            large_fit = nothing
            large_source = nothing
            GC.gc(true)
            CUDA.reclaim()
        end
    end
end
