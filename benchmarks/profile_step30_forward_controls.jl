using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation
using Serialization

CUDA.functional() || error("CUDA is not functional.")

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(
    ENV,
    "STEP30_DATA_DIR",
    joinpath(ROOT, "example_notebooks", "benchmark_data",
             "step_index_soliton"),
)
const CP_PATH = get(
    ENV,
    "STEP30_CP_PATH",
    joinpath(ROOT, "benchmark_cache", "step30_cp_1e-4.jls"),
)
const OUTPUT_PATH = get(ENV, "STEP30_OUTPUT_PATH", "")
const REFERENCE_RESULTS_PATH = get(ENV, "REFERENCE_RESULTS_PATH", "")
const LENGTH_M = parse(Float64, get(ENV, "LENGTH_M", "0.01"))
const BASELINE_DZ = parse(Float64, get(ENV, "BASELINE_DZ", "5e-5"))
const FINE_DZ = parse(Float64, get(ENV, "FINE_DZ", "0"))
const ADAPT_INITIAL_DZ = parse(
    Float64, get(ENV, "ADAPT_INITIAL_DZ", string(BASELINE_DZ)))
const ADAPT_DZMAX = parse(
    Float64, get(ENV, "ADAPT_DZMAX", "0.01"))
const ADAPT_ABSTOL = parse(
    Float64, get(ENV, "ADAPT_ABSTOL", "1e-12"))
const ADAPT_RTOLS = parse.(Float64, split(
    get(ENV, "ADAPT_RTOLS", "1e-4,3e-5,1e-5,3e-6,1e-6"), ","))
const RUN_FIXED64 = get(ENV, "RUN_FIXED64", "1") == "1"
const RUN_MIXED = get(ENV, "RUN_MIXED", "1") == "1"
const RUN_ADAPT64 = get(ENV, "RUN_ADAPT64", "1") == "1"
const RUN_ADAPT_MIXED = get(ENV, "RUN_ADAPT_MIXED", "1") == "1"
const FORCE_COMPLEX_CP = get(ENV, "FORCE_COMPLEX_CP", "0") == "1"
const SOLVER_BACKEND = Symbol(get(
    ENV, "STEP30_BACKEND", "cuda_cp_optimized"))
const CUDA_FAST_PRECISION = get(ENV, "CUDA_FAST_PRECISION", "")

if !isempty(CUDA_FAST_PRECISION)
    CUDA.math_mode!(CUDA.FAST_MATH; precision=Symbol(CUDA_FAST_PRECISION))
end

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))

function intensity_relative_l2(candidate, reference)
    candidate_intensity = abs2.(candidate)
    reference_intensity = abs2.(reference)
    return relative_l2(candidate_intensity, reference_intensity)
end

function modal_energy(field)
    return vec(sum(abs2, field; dims=(1, 3)))
end

function timed_solve(initial, parameters, dz; kwargs...)
    CUDA.synchronize()
    start = time_ns()
    solution = solve_mmgnlse(initial, parameters, dz; kwargs...)
    CUDA.synchronize()
    return solution, (time_ns() - start) / 1e9
end

function initial_condition(domain)
    coefficients = zeros(ComplexF64, 30)
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
        domain, 30;
        peak_power=751549.7920240875,
        fwhm=0.2499694427126389,
        time_offset=-30.002441147314094,
        coefficients,
    )
end

function benchmark_parameters(length_m)
    nt = 8192
    lambda0 = 1550e-9
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), TimeGrid(nt, 100.0))
    betas40 = npzread(joinpath(DATA_DIR, "betas_40modes.npy"))
    beta30 = TaylorBeta(permutedims(betas40[1:30, :]))
    material = Silica()
    fiber = StepIndex(lambda0, 52.5e-6, 0.1)
    properties = compute_fiber_properties(
        fiber, material, 30;
        polarization=:scalar,
        mode_basis=:LP,
        beta_order=3,
    )
    cp = deserialize(CP_PATH)
    if FORCE_COMPLEX_CP
        cp = MMGNLSECPDecomposition(
            complex.(cp.λ),
            ntuple(index -> complex.(cp.U[index]), 4);
            layout=cp.metadata.layout,
            nmodes=cp.metadata.nmodes,
            npolarizations=cp.metadata.npolarizations,
            relative_error=cp.relative_error,
            seed=cp.seed,
            iterations=cp.iterations,
            converged=cp.converged,
        )
    end
    parameters = MMGNLSEParameters(
        domain;
        length=length_m,
        beta=beta30,
        S=cp,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman,
    )
    return domain, parameters, cp
end

domain, parameters, cp = benchmark_parameters(LENGTH_M)
initial = initial_condition(domain)
torch_input = reshape(
    permutedims(npzread(joinpath(
        DATA_DIR, "input_fields_30_5e-05.npy"))),
    8192, 30, 1,
)
torch_output = reshape(
    permutedims(npzread(joinpath(
        DATA_DIR, "output_fields_30_5e-05.npy"))),
    8192, 30, 1,
)

println(
    "configuration gpu=", CUDA.name(CUDA.device()),
    " length_m=", LENGTH_M,
    " baseline_dz=", BASELINE_DZ,
    " cp_rank=", cp_rank(cp),
    " cp_error=", cp.relative_error,
    " force_complex_cp=", FORCE_COMPLEX_CP,
    " backend=", SOLVER_BACKEND,
    " math_mode=", CUDA.math_mode(),
    " math_precision=", CUDA.math_precision(),
    " input_relative_l2_vs_torch=", relative_l2(initial, torch_input),
)

# Compile all selected paths on a short instance before timing.
warmup_length = min(LENGTH_M, max(2BASELINE_DZ, 1e-4))
_, warmup_parameters, _ = benchmark_parameters(warmup_length)
solve_mmgnlse(initial, warmup_parameters, BASELINE_DZ;
    backend=SOLVER_BACKEND, precision=:float64)
RUN_MIXED && solve_mmgnlse(
    initial, warmup_parameters, BASELINE_DZ;
    backend=SOLVER_BACKEND, precision=:mixed)
RUN_ADAPT64 && solve_mmgnlse(
    initial, warmup_parameters, ADAPT_INITIAL_DZ;
    backend=SOLVER_BACKEND, precision=:float64,
    adaptive=true, reltol=first(ADAPT_RTOLS), abstol=ADAPT_ABSTOL,
    dzmax=ADAPT_DZMAX)
RUN_ADAPT_MIXED && solve_mmgnlse(
    initial, warmup_parameters, ADAPT_INITIAL_DZ;
    backend=SOLVER_BACKEND, precision=:mixed,
    adaptive=true, reltol=first(ADAPT_RTOLS), abstol=ADAPT_ABSTOL,
    dzmax=ADAPT_DZMAX)

results = Dict{String,Any}()
timings = Dict{String,Float64}()

if !isempty(REFERENCE_RESULTS_PATH)
    reference_payload = deserialize(REFERENCE_RESULTS_PATH)
    haskey(reference_payload.results, "fixed_fp64") || error(
        "REFERENCE_RESULTS_PATH does not contain fixed_fp64.")
    results["external_fixed_fp64"] =
        reference_payload.results["fixed_fp64"]
    timings["external_fixed_fp64"] =
        reference_payload.timings["fixed_fp64"]
end

if FINE_DZ > 0
    results["fine_fp64"], timings["fine_fp64"] = timed_solve(
        initial, parameters, FINE_DZ;
        backend=SOLVER_BACKEND, precision=:float64)
end
if RUN_FIXED64
    results["fixed_fp64"], timings["fixed_fp64"] = timed_solve(
        initial, parameters, BASELINE_DZ;
        backend=SOLVER_BACKEND, precision=:float64)
end
if RUN_MIXED
    results["fixed_mixed"], timings["fixed_mixed"] = timed_solve(
        initial, parameters, BASELINE_DZ;
        backend=SOLVER_BACKEND, precision=:mixed)
end
if RUN_ADAPT64
    for reltol in ADAPT_RTOLS
        label = "adaptive_fp64_rtol_$(reltol)"
        results[label], timings[label] = timed_solve(
            initial, parameters, ADAPT_INITIAL_DZ;
            backend=SOLVER_BACKEND, precision=:float64,
            adaptive=true, reltol, abstol=ADAPT_ABSTOL,
            dzmax=ADAPT_DZMAX)
    end
end
if RUN_ADAPT_MIXED
    for reltol in ADAPT_RTOLS
        label = "adaptive_mixed_rtol_$(reltol)"
        results[label], timings[label] = timed_solve(
            initial, parameters, ADAPT_INITIAL_DZ;
            backend=SOLVER_BACKEND, precision=:mixed,
            adaptive=true, reltol, abstol=ADAPT_ABSTOL,
            dzmax=ADAPT_DZMAX)
    end
end

reference_label = haskey(results, "fine_fp64") ? "fine_fp64" :
                  haskey(results, "external_fixed_fp64") ?
                  "external_fixed_fp64" : "fixed_fp64"
reference = results[reference_label].fields[:, :, :, end]
initial_energy = sum(abs2, initial)
for label in sort!(collect(keys(results)))
    solution = results[label]
    final = solution.fields[:, :, :, end]
    modal_error = relative_l2(modal_energy(final), modal_energy(reference))
    torch_field_error = LENGTH_M == 10.0 ?
                        relative_l2(final, torch_output) : NaN
    torch_intensity_error = LENGTH_M == 10.0 ?
                            intensity_relative_l2(final, torch_output) : NaN
    println(
        "result variant=", label,
        " seconds=", timings[label],
        " speedup_vs_fixed_fp64=",
        haskey(timings, "fixed_fp64") ?
            timings["fixed_fp64"] / timings[label] : NaN,
        " reference=", reference_label,
        " field_relative_l2=", relative_l2(final, reference),
        " intensity_relative_l2=", intensity_relative_l2(final, reference),
        " modal_energy_relative_l2=", modal_error,
        " energy_relative_drift=", (sum(abs2, final) - initial_energy) /
                                    initial_energy,
        " torch_field_relative_l2=", torch_field_error,
        " torch_intensity_relative_l2=", torch_intensity_error,
        " accepted_steps=", solution.cache.accepted_steps,
        " rejected_steps=", solution.cache.rejected_steps,
        " rhs_evaluations=", solution.cache.rhs_evaluations,
        " step_statistics=", solution.cache.step_statistics,
    )
end

if haskey(results, "fixed_fp64")
    fixed_final = results["fixed_fp64"].fields[:, :, :, end]
    for label in sort!(collect(keys(results)))
        label == "fixed_fp64" && continue
        final = results[label].fields[:, :, :, end]
        println(
            "comparison variant=", label,
            " reference=fixed_fp64",
            " field_relative_l2=", relative_l2(final, fixed_final),
            " intensity_relative_l2=",
            intensity_relative_l2(final, fixed_final),
        )
    end
end

if !isempty(OUTPUT_PATH)
    mkpath(dirname(OUTPUT_PATH))
    open(OUTPUT_PATH, "w") do io
        serialize(io, (; configuration=(
            length_m=LENGTH_M,
            baseline_dz=BASELINE_DZ,
            fine_dz=FINE_DZ,
            adaptive_initial_dz=ADAPT_INITIAL_DZ,
            adaptive_dzmax=ADAPT_DZMAX,
            adaptive_abstol=ADAPT_ABSTOL,
            adaptive_rtols=ADAPT_RTOLS,
            backend=SOLVER_BACKEND,
            cp_rank=cp_rank(cp),
            cp_error=cp.relative_error,
        ), timings, results))
    end
    println("serialized_results=", OUTPUT_PATH)
end
