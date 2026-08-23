using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation
using Serialization
using Statistics

const EXT = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
EXT === nothing && error("PulsePropagation CUDA extension did not load.")
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
    joinpath(ROOT, "benchmark_cache", "step30_real_rank252.jls"),
)
const NSAMPLES = parse(Int, get(ENV, "NSAMPLES", "20"))
const NWARMUP = parse(Int, get(ENV, "NWARMUP", "5"))
const CUDA_FAST_PRECISION = get(ENV, "CUDA_FAST_PRECISION", "")
const R2C_ONLY = get(ENV, "R2C_ONLY", "0") == "1"

if !isempty(CUDA_FAST_PRECISION)
    CUDA.math_mode!(CUDA.FAST_MATH; precision=Symbol(CUDA_FAST_PRECISION))
end

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
    return (; median=median(samples), minimum=minimum(samples),
            maximum=maximum(samples))
end

function report(f, backend, stage)
    timing = measure(f)
    println(
        "backend=", backend,
        " stage=", stage,
        " median_ms=", timing.median,
        " minimum_ms=", timing.minimum,
        " maximum_ms=", timing.maximum,
    )
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

nt = 8192
domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space), TimeGrid(nt, 100.0))
betas40 = npzread(joinpath(DATA_DIR, "betas_40modes.npy"))
beta30 = TaylorBeta(permutedims(betas40[1:30, :]))
material = Silica()
fiber = StepIndex(1550e-9, 52.5e-6, 0.1)
properties = compute_fiber_properties(
    fiber, material, 30;
    polarization=:scalar,
    mode_basis=:LP,
    beta_order=3,
)
cp_real = deserialize(CP_PATH)
cp_complex = MMGNLSECPDecomposition(
    complex.(cp_real.λ),
    ntuple(index -> complex.(cp_real.U[index]), 4);
    layout=cp_real.metadata.layout,
    nmodes=cp_real.metadata.nmodes,
    npolarizations=cp_real.metadata.npolarizations,
    relative_error=cp_real.relative_error,
    seed=cp_real.seed,
    iterations=cp_real.iterations,
    converged=cp_real.converged,
)
initial = initial_condition(domain)

println(
    "gpu=", CUDA.name(CUDA.device()),
    " nt=", nt,
    " nm=30 rank=", cp_rank(cp_real),
    " cp_error=", cp_real.relative_error,
    " math_mode=", CUDA.math_mode(),
    " math_precision=", CUDA.math_precision(),
)

reference_outputs = Dict{DataType,Array}()
for T in (Float64, Float32)
    field_t = CuArray(Complex{T}.(initial))
    field_w = CUDA.ifft(field_t, 1)
    representations = R2C_ONLY ? ((:real, cp_real),) :
        ((:complex_cast, cp_complex), (:real, cp_real))
    for (representation, cp) in representations
        parameters = MMGNLSEParameters(
            domain;
            length=0.01,
            beta=beta30,
            S=cp,
            n2=properties.n2,
            omega0=properties.omega0,
            raman=properties.raman,
        )
        backends = R2C_ONLY ? (:cuda_optimized,) :
            (:cuda, :cuda_cp_optimized, :cuda_optimized)
        for backend in backends
            policy = EXT._cuda_mmgnlse_cp_optimization(backend)
            cache = EXT._cuda_mmgnlse_forward_cache(parameters, policy, T)
            workspace = EXT._cuda_mmgnlse_cp_forward_workspace(
                field_t, cache.overlap)
            out = similar(field_t)

            workspace.field_t .= field_w
            workspace.fft_field! * workspace.field_t
            EXT._cuda_mmgnlse_cp_project!(
                workspace, workspace.field_t, cache.overlap)

            label = "$(representation)_$(T)_$(backend)"
            report(label, "modal_fft") do
                workspace.field_t .= field_w
                workspace.fft_field! * workspace.field_t
            end
            report(label, "projection") do
                EXT._cuda_mmgnlse_cp_project!(
                    workspace, workspace.field_t, cache.overlap)
            end
            report(label, "rank_mix_raman_backprojection") do
                EXT._cuda_mmgnlse_cp_combined!(
                    workspace.nonlinear, workspace,
                    cache.overlap, cache.raman)
            end
            report(label, "complete_rhs") do
                EXT._cuda_mmgnlse_cp_nonlinear_spectral!(
                    out, field_w, cache, workspace)
            end
            report(label, "rk4ip_step") do
                EXT._cuda_mmgnlse_cp_rk4ip_step!(
                    workspace.next_field, field_w, cache,
                    workspace, 0.0, 5e-5)
            end

            if representation === :real &&
               backend === :cuda_optimized &&
               cache.overlap.symmetric_factors
                projected = @view workspace.b2[:, :, 1]
                report(label, "symmetric_pair") do
                    workspace.rank2 .= abs2.(projected)
                end
                report(label, "raman_ifft") do
                    workspace.ifft_rank! * workspace.rank2
                end
                report(label, "raman_scale") do
                    workspace.rank2 .*=
                        reshape(cache.raman.combined, :, 1)
                end
                report(label, "raman_fft") do
                    workspace.fft_rank! * workspace.rank2
                end
                report(label, "symmetric_mix") do
                    workspace.rank1 .= projected .* (
                        (1 - cache.raman.fraction) .* abs2.(projected) .+
                        cache.raman.fraction .* workspace.rank2)
                end
                output_view = @view workspace.nonlinear[:, :, 1]
                report(label, "real_backprojection") do
                    EXT._cuda_mmgnlse_cp_backproject!(
                        output_view, workspace.rank1, cache.overlap)
                end

                # The symmetric scalar pair |B|^2 is real.  Probe an R2C/C2R
                # Raman filter without changing the production workspace yet.
                pair_real = CUDA.zeros(T, nt, cache.overlap.rank)
                half_spectrum = CUDA.zeros(
                    Complex{T}, nt ÷ 2 + 1, cache.overlap.rank)
                filtered_real = similar(pair_real)
                rfft_rank = CUDA.plan_rfft(pair_real, 1)
                irfft_rank = CUDA.plan_irfft(half_spectrum, nt, 1)
                kernel_half = conj.(
                    @view cache.raman.combined[1:(nt ÷ 2 + 1)])
                pair_real .= abs2.(projected)
                report(label, "raman_r2c_forward") do
                    mul!(half_spectrum, rfft_rank, pair_real)
                end
                report(label, "raman_r2c_scale") do
                    half_spectrum .*= reshape(kernel_half, :, 1)
                end
                report(label, "raman_c2r_inverse") do
                    mul!(filtered_real, irfft_rank, half_spectrum)
                end
                report(label, "raman_r2c_filter") do
                    mul!(half_spectrum, rfft_rank, pair_real)
                    half_spectrum .*= reshape(kernel_half, :, 1)
                    mul!(filtered_real, irfft_rank, half_spectrum)
                end
                mul!(half_spectrum, rfft_rank, pair_real)
                half_spectrum .*= reshape(kernel_half, :, 1)
                mul!(filtered_real, irfft_rank, half_spectrum)

                # Recompute the complex reference once from a fresh pair.
                # The per-stage timing loops above intentionally mutate their
                # input repeatedly and therefore cannot serve as a reference.
                workspace.rank2 .= pair_real
                workspace.ifft_rank! * workspace.rank2
                workspace.rank2 .*= reshape(
                    cache.raman.combined, :, 1)
                workspace.fft_rank! * workspace.rank2
                filtered_complex = Array(workspace.rank2)
                filtered_r2c = Array(filtered_real)
                println(
                    "correctness variant=", label,
                    " raman_r2c_relative_l2=",
                    norm(Complex{T}.(filtered_r2c) .- filtered_complex) /
                    max(norm(filtered_complex), eps(T)),
                    " raman_filtered_imag_relative_l2=",
                    norm(imag.(filtered_complex)) /
                    max(norm(filtered_complex), eps(T)),
                )
            end

            EXT._cuda_mmgnlse_cp_nonlinear_spectral!(
                out, field_w, cache, workspace)
            output = Array(out)
            if (representation === :complex_cast && backend === :cuda) ||
               (R2C_ONLY && representation === :real && T === Float64)
                reference_outputs[T] = output
            else
                reference = get(reference_outputs, T, nothing)
                reference === nothing || println(
                    "correctness variant=", label,
                    " relative_l2=", norm(output .- reference) /
                                      max(norm(reference), eps(T)),
                )
            end
            if T === Float32 && haskey(reference_outputs, Float64)
                reference64 = reference_outputs[Float64]
                println(
                    "cross_precision variant=", label,
                    " relative_l2_vs_fp64=",
                    norm(ComplexF64.(output) .- reference64) /
                    max(norm(reference64), eps(Float64)),
                )
            end
        end
    end
end
