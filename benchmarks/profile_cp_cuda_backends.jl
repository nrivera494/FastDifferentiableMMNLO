using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Statistics

const EXT = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
EXT === nothing && error("PulsePropagation CUDA extension did not load.")
CUDA.functional() || error("CUDA is not functional.")

const NT = parse(Int, get(ENV, "NT", "8192"))
const NM = parse(Int, get(ENV, "NM", "40"))
const RANKS = parse.(Int, split(get(ENV, "RANKS", "50,100,200,484"), ","))
const NSAMPLES = parse(Int, get(ENV, "NSAMPLES", "15"))
const NWARMUP = parse(Int, get(ENV, "NWARMUP", "5"))

function measure(f)
    for _ in 1:NWARMUP
        f()
        CUDA.synchronize()
    end
    samples = Float64[]
    for _ in 1:NSAMPLES
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        push!(samples, (time_ns() - start) / 1e6)
    end
    return samples
end

function report(f, backend, rank, stage)
    samples = measure(f)
    println(
        "backend=", backend,
        " rank=", rank,
        " stage=", stage,
        " median_ms=", median(samples),
        " minimum_ms=", minimum(samples),
        " maximum_ms=", maximum(samples),
    )
end

function notebook_initial_condition(domain)
    coefficients = zeros(ComplexF64, NM)
    coefficients[6:15] .= ComplexF64[
        0.042233154627023345 + 0.9991077822989120im,
       -0.9548483373690423   + 0.29709367651226704im,
        0.8131419320624027   - 0.5820654587946469im,
        0.18628501477914675  + 0.9824957472013469im,
        0.6319768780431223   + 0.7749872422297470im,
       -0.9492435495998746   + 0.31454202190332287im,
       -0.06792170771520316  - 0.9976906542716788im,
        0.3907153673065814   - 0.9205115435183218im,
        0.9992441657975882   - 0.03887283266604082im,
        0.6421650166589653   - 0.7665664298541848im,
    ]
    return gaussian_pulse(
        domain, NM;
        peak_power=751549.7920240875,
        fwhm=0.2499694427126389,
        time_offset=-30.002441147314094,
        coefficients,
    )
end

function benchmark(rank)
    rng = MersenneTwister(0xbacc + rank)
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), TimeGrid(NT, 100.0))
    factors = ntuple(4) do _
        randn(rng, ComplexF64, NM, rank) ./ sqrt(NM)
    end
    cp = MMGNLSECPDecomposition(
        ones(Float64, rank), factors;
        layout=:spatial, nmodes=NM, npolarizations=1)
    parameters = MMGNLSEParameters(
        domain;
        length=1e-5,
        beta=TaylorBeta(zeros(4, NM)),
        S=cp,
        n2=2.3e-20,
        omega0=2π * 2.99792458e-4 / 1550e-9,
        raman=AnisotropicRaman(),
    )
    initial = notebook_initial_condition(domain)
    field_t = CuArray(ComplexF64.(initial))
    field_w = CUDA.ifft(field_t, 1)
    cotangent = randn(rng, ComplexF64, size(initial))
    cotangent ./= norm(cotangent)
    cotangent_t = CuArray(cotangent)

    for backend in (:cuda, :cuda_cp_optimized, :cuda_optimized)
        policy = EXT._cuda_mmgnlse_cp_optimization(backend)
        cache = EXT._cuda_mmgnlse_forward_cache(parameters, policy)
        forward_workspace =
            EXT._cuda_mmgnlse_cp_forward_workspace(field_t, cache.overlap)
        forward_out = similar(field_t)
        adjoint_workspace =
            EXT._cuda_mmgnlse_cp_adjoint_workspace(field_t, cache.overlap)
        adjoint_out = similar(field_t)

        report(backend, rank, "forward_rhs") do
            EXT._cuda_mmgnlse_cp_nonlinear_spectral!(
                forward_out, field_w, cache, forward_workspace)
        end
        report(backend, rank, "forward_rk4ip_step") do
            EXT._cuda_mmgnlse_cp_rk4ip_step!(
                forward_workspace.next_field, field_w, cache,
                forward_workspace, 0.0, 5e-7)
        end
        report(backend, rank, "adjoint_vjp") do
            EXT._cuda_mmgnlse_cp_vjp_nonlinear!(
                adjoint_out, field_t, cotangent_t, cache,
                adjoint_workspace)
        end
    end
end

println(
    "gpu=", CUDA.name(CUDA.device()),
    " nt=", NT,
    " nm=", NM,
    " ranks=", join(RANKS, ","),
    " samples=", NSAMPLES,
)
for rank in RANKS
    benchmark(rank)
end
