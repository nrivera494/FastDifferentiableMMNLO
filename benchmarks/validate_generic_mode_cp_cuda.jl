using CUDA
using PulsePropagation
using Test

CUDA.functional() || error("CUDA is not functional.")

@testset "Generic dense-free mode-overlap CP on CUDA" begin
    x = collect(range(-8e-6, 8e-6; length=49))
    y = collect(range(-7e-6, 7e-6; length=47))
    fields = Array{Float64}(undef, length(y), length(x), 3)
    @inbounds for ix in eachindex(x), iy in eachindex(y)
        envelope = exp(-((x[ix] / 4e-6)^2 +
                         (y[iy] / 3.5e-6)^2) / 2)
        fields[iy, ix, 1] = envelope
        fields[iy, ix, 2] = (x[ix] / 4e-6) * envelope
        fields[iy, ix, 3] = (y[iy] / 3.5e-6) * envelope
    end
    PulsePropagation._normalize_modes!(fields, x, y)

    compressed = cp_compress_mode_overlap(
        fields, x, y;
        target_error=1e-3,
        quadrature_order=16,
        authority_order=22,
        quadrature_tolerance=1e-4,
        backend=:cuda,
        max_rank=20,
        rank_step=2,
        maxiter=100,
        check_every=2,
        restarts=1,
    )
    @test compressed isa MMGNLSECPDecomposition
    @test compressed.converged
    @test compressed.relative_error <= 1e-3
    @test compressed.metadata.source_checksum !== nothing
end

CUDA.synchronize()
println(
    "validate_generic_mode_cp_cuda: PASS gpu=", CUDA.name(CUDA.device()))
