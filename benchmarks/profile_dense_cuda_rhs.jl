using CUDA
using LinearAlgebra
using PulsePropagation
using Random
using Serialization
using Statistics

const PPCExt = Base.get_extension(
    PulsePropagation, :PulsePropagationCUDAExt)
PPCExt === nothing && error("PulsePropagation CUDA extension did not load.")
CUDA.functional() || error("CUDA is not functional.")

const NT = parse(Int, get(ENV, "NT", "8192"))
const NMODES = parse(Int, get(ENV, "NMODES", "40"))
const NSAMPLES = parse(Int, get(ENV, "NSAMPLES", "7"))
const NWARMUP = parse(Int, get(ENV, "NWARMUP", "2"))
const LABEL = get(ENV, "LABEL", "dense_cuda")

function synchronized_samples(f, nsamples::Int, nwarmup::Int)
    for _ in 1:nwarmup
        f()
        CUDA.synchronize()
    end
    times = Vector{Float64}(undef, nsamples)
    for index in eachindex(times)
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        times[index] = (time_ns() - start) / 1e9
    end
    return times
end

function report_samples(name, times)
    println(
        name,
        " median_s=", median(times),
        " minimum_s=", minimum(times),
        " maximum_s=", maximum(times),
        " samples_s=", join(times, ","),
    )
end

function benchmark_problem(nt::Int, nmodes::Int)
    rng = MersenneTwister(0x5eed)
    grid = TimeGrid(nt, 100.0)
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), grid)
    overlap = rand(rng, Float64, nmodes, nmodes, nmodes, nmodes)

    coefficients = zeros(ComplexF64, nmodes)
    launch = ComplexF64[
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
    launch_start = min(6, max(1, nmodes - length(launch) + 1))
    launch_stop = min(nmodes, launch_start + length(launch) - 1)
    coefficients[launch_start:launch_stop] .=
        launch[1:(launch_stop - launch_start + 1)]

    initial = gaussian_pulse(
        domain,
        nmodes;
        peak_power=751549.7920240875,
        fwhm=0.2499694427126389,
        time_offset=-30.002441147314094,
        coefficients,
    )
    parameters = MMGNLSEParameters(
        domain;
        length=1e-5,
        beta=TaylorBeta(zeros(4, nmodes)),
        S=overlap,
        n2=2.3e-20,
        omega0=2π * 2.99792458e-4 / 1550e-9,
        raman=AnisotropicRaman(),
    )
    return initial, parameters
end

initial, parameters = benchmark_problem(NT, NMODES)
cache = PPCExt._cuda_mmgnlse_forward_cache(parameters)
initial_gpu = CUDA.CuArray(ComplexF64.(initial))
field_w = PulsePropagation.ifft(initial_gpu, 1)
workspace = PPCExt._cuda_mmgnlse_dense_forward_workspace(field_w, cache)
base = PPCExt._cuda_mmgnlse_dense_base(workspace)
out = similar(field_w)

base.field_t .= field_w
base.fft_field! * base.field_t
CUDA.synchronize()

instantaneous!() = workspace isa
    PPCExt.CUDAMMGNLSEDenseRamanForwardWorkspace ?
    PPCExt._cuda_mmgnlse_dense_pair_instantaneous!(
        base.nonlinear, base.field_t, cache.overlap, workspace) :
    PPCExt._cuda_mmgnlse_dense_instantaneous!(
        base.nonlinear, base.field_t, cache.overlap)

rhs!() = PPCExt._cuda_mmgnlse_dense_nonlinear_spectral!(
    out, field_w, cache, workspace)

println(
    "label=", LABEL,
    " gpu=", CUDA.name(CUDA.device()),
    " nt=", NT,
    " nmodes=", NMODES,
    " samples=", NSAMPLES,
    " warmup=", NWARMUP,
    " contraction=", cache.overlap.algorithm,
)
report_samples(
    "instantaneous_contraction",
    synchronized_samples(instantaneous!, NSAMPLES, NWARMUP),
)
rhs_times = synchronized_samples(rhs!, NSAMPLES, NWARMUP)
report_samples("nonlinear_rhs", rhs_times)

rhs!()
CUDA.synchronize()
host_output = Array(out)
println(
    "output_norm2=", sum(abs2, host_output),
    " output_maxabs=", maximum(abs, host_output),
)

if haskey(ENV, "REFERENCE_OUTPUT")
    reference = deserialize(ENV["REFERENCE_OUTPUT"])
    difference = host_output .- reference
    println(
        "reference_max_abs_error=", maximum(abs, difference),
        " reference_relative_l2_error=",
        norm(difference) / max(norm(reference), eps(Float64)),
    )
end
if haskey(ENV, "SAVE_OUTPUT")
    serialize(ENV["SAVE_OUTPUT"], host_output)
    println("saved_output=", ENV["SAVE_OUTPUT"])
end
