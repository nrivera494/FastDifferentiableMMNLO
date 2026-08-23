using CUDA
using FFTW
using LinearAlgebra
using NPZ
using PulsePropagation
using Statistics

include(joinpath(@__DIR__, "grin_210_common.jl"))

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const GRIN210_CUDA_EXT =
    Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
GRIN210_CUDA_EXT === nothing && error(
    "PulsePropagation CUDA extension did not load.")

parse_csv(::Type{T}, value) where {T} =
    parse.(T, filter(!isempty, strip.(split(value, ','))))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in
                          ("1", "true", "yes", "on")

const MODE_COUNTS = parse_csv(
    Int,
    get(ENV, "GRIN210_MODE_COUNTS",
        join(GRIN210_DEFAULT_MODE_COUNTS, ',')),
)
const TARGET_ERRORS = parse_csv(
    Float64,
    get(ENV, "GRIN210_CP_TARGET_ERRORS",
        join(GRIN210_DEFAULT_TARGET_ERRORS, ',')),
)
const NT = parse(Int, get(ENV, "GRIN210_NT", "4096"))
const TIME_WINDOW_PS = parse(
    Float64, get(ENV, "GRIN210_TIME_WINDOW_PS", "10.0"))
const TRANSVERSE_COUNT = parse(
    Int, get(ENV, "GRIN210_TRANSVERSE_COUNT", "96"))
const TRANSVERSE_HALF_WIDTH_UM = parse(
    Float64, get(ENV, "GRIN210_TRANSVERSE_HALF_WIDTH_UM", "120.0"))
const DZ_M = parse(Float64, get(ENV, "GRIN210_DZ_M", "1.0e-5"))
const PROPAGATION_LENGTH_M = parse(
    Float64, get(ENV, "GRIN210_LENGTH_M", "0.025"))
const STRANG_DZ_M = parse(
    Float64, get(ENV, "GRIN210_STRANG_DZ_M", string(DZ_M)))
const TIMING_WARMUP = parse(
    Int, get(ENV, "GRIN210_TIMING_WARMUP", "3"))
const TIMING_SAMPLES = parse(
    Int, get(ENV, "GRIN210_TIMING_SAMPLES", "15"))
const RUN_TIMINGS = env_bool("GRIN210_RUN_TIMINGS", true)
const RUN_SOLUTIONS = env_bool("GRIN210_RUN_SOLUTIONS", false)
const RUN_STRANG_SOLUTION = env_bool(
    "GRIN210_RUN_STRANG_SOLUTION", true)
const CP_BACKEND = Symbol(lowercase(get(
    ENV, "GRIN210_CP_BACKEND", "cuda_cp_optimized")))
const PRECISION = Symbol(lowercase(get(
    ENV, "GRIN210_PRECISION", "float64")))
const CP_DIR = get(
    ENV,
    "GRIN210_CP_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_cp"),
)
const OUTPUT_DIR = get(
    ENV,
    "GRIN210_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_crossover"),
)

isempty(MODE_COUNTS) && error("GRIN210_MODE_COUNTS cannot be empty.")
issorted(MODE_COUNTS) || error("GRIN210_MODE_COUNTS must be increasing.")
all(mode -> mode >= 15, MODE_COUNTS) || error(
    "Every retained basis must contain launch modes 6:15.")
foreach(grin210_complete_group_count, MODE_COUNTS)
TARGET_ERRORS == sort(TARGET_ERRORS; rev=true) || error(
    "GRIN210_CP_TARGET_ERRORS must be ordered loose to tight.")
all(target -> 0 < target < 1, TARGET_ERRORS) || error(
    "Every CP target must lie in (0, 1).")
NT >= 16 || error("GRIN210_NT must be at least 16.")
TRANSVERSE_COUNT >= 16 || error(
    "GRIN210_TRANSVERSE_COUNT must be at least 16.")
TIME_WINDOW_PS > 0 || error("GRIN210_TIME_WINDOW_PS must be positive.")
TRANSVERSE_HALF_WIDTH_UM > 0 || error(
    "GRIN210_TRANSVERSE_HALF_WIDTH_UM must be positive.")
DZ_M > 0 || error("GRIN210_DZ_M must be positive.")
PROPAGATION_LENGTH_M > 0 || error("GRIN210_LENGTH_M must be positive.")
STRANG_DZ_M > 0 || error("GRIN210_STRANG_DZ_M must be positive.")
TIMING_WARMUP >= 0 || error("GRIN210_TIMING_WARMUP must be nonnegative.")
TIMING_SAMPLES > 0 || error("GRIN210_TIMING_SAMPLES must be positive.")
PRECISION === :float64 || error(
    "The matched crossover currently requires GRIN210_PRECISION=float64; " *
    "full-field Float32 and modal mixed precision are not equivalent policies.")
CP_BACKEND in (:cuda, :cuda_cp_optimized, :cuda_optimized) || error(
    "Unsupported GRIN210_CP_BACKEND=$CP_BACKEND.")

mkpath(OUTPUT_DIR)

function load_grin210_cp(mode_count, target)
    paths = grin210_cp_cache_paths(CP_DIR, mode_count, target)
    isfile(paths.data) && isfile(paths.metadata) || error(
        "Missing CP cache for modes=$mode_count target=$target. Run " *
        "benchmarks/build_grin_210_cp_cuda.jl first. Expected $(paths.data).")
    cp = load_cp_decomposition(paths.data, paths.metadata)
    cp.metadata.nmodes == mode_count || error(
        "Cached CP mode count does not match $mode_count: $(paths.data)")
    cp.relative_error <= target || error(
        "Cached CP misses target $target: $(paths.data)")
    return cp
end

function measure_cuda(operation, prepare=()->nothing;
                      warmup::Integer=TIMING_WARMUP,
                      samples::Integer=TIMING_SAMPLES)
    for _ in 1:Int(warmup)
        prepare()
        operation()
        CUDA.synchronize()
    end
    milliseconds = Vector{Float64}(undef, Int(samples))
    for index in eachindex(milliseconds)
        prepare()
        CUDA.synchronize()
        start = time_ns()
        operation()
        CUDA.synchronize()
        milliseconds[index] = (time_ns() - start) / 1e6
    end
    return (;
        median_ms=median(milliseconds),
        mean_ms=mean(milliseconds),
        minimum_ms=minimum(milliseconds),
        maximum_ms=maximum(milliseconds),
        std_ms=std(milliseconds; corrected=false),
        samples=length(milliseconds),
        raw_ms=milliseconds,
    )
end

function time_full_field_step(initial, parameters)
    workspace = GRIN210_CUDA_EXT._cuda_full_field_workspace(
        initial, parameters, Float64, :rk4ip)
    mul!(workspace.field, workspace.to_spectral!, workspace.field)
    seed_field = copy(workspace.field)
    prepare!() = (workspace.field .= seed_field)
    step!() = GRIN210_CUDA_EXT._cuda_full_field_rk4ip_step!(
        workspace, DZ_M)
    timing = measure_cuda(step!, prepare!)
    used_bytes = Int(coalesce(CUDA.used_memory(), 0))
    return timing, used_bytes
end

function time_cp_step(cp, time)
    initial = grin210_modal_initial(time, cp.metadata.nmodes)
    parameters = grin210_modal_parameters(time, cp; length=DZ_M)
    optimization = GRIN210_CUDA_EXT._cuda_mmgnlse_cp_optimization(CP_BACKEND)
    state_cache = GRIN210_CUDA_EXT._cuda_mmgnlse_forward_cache(
        parameters, optimization)
    initial_device = CUDA.CuArray(ComplexF64.(initial))
    field_w = ifft(initial_device, 1)
    workspace = GRIN210_CUDA_EXT._cuda_mmgnlse_cp_precision_workspace(
        initial_device, state_cache, state_cache, :float64)
    state = GRIN210_CUDA_EXT._cuda_mmgnlse_cp_state_workspace(workspace)
    step!() = GRIN210_CUDA_EXT._cuda_mmgnlse_cp_rk4ip_step!(
        state.next_field, field_w, state_cache, state_cache,
        workspace, 0.0, DZ_M)
    timing = measure_cuda(step!)
    used_bytes = Int(coalesce(CUDA.used_memory(), 0))
    return timing, used_bytes
end

function write_timing_csv(path, full_timing, full_bytes, rows)
    open(path, "w") do io
        println(io,
            "method,modes,groups,target,rank,cp_error,median_ms,mean_ms," *
            "minimum_ms,maximum_ms,std_ms,samples,device_used_bytes")
        println(io, join((
            "full_field_rk4ip",
            GRIN210_MODE_COUNT,
            GRIN210_GROUP_COUNT,
            "",
            "",
            "",
            full_timing.median_ms,
            full_timing.mean_ms,
            full_timing.minimum_ms,
            full_timing.maximum_ms,
            full_timing.std_ms,
            full_timing.samples,
            full_bytes,
        ), ','))
        for row in rows
            println(io, join((
                "modal_cp_rk4ip",
                row.modes,
                row.groups,
                row.target,
                row.rank,
                row.cp_error,
                row.timing.median_ms,
                row.timing.mean_ms,
                row.timing.minimum_ms,
                row.timing.maximum_ms,
                row.timing.std_ms,
                row.timing.samples,
                row.device_used_bytes,
            ), ','))
        end
    end
end

function gpu_integrated_spectrum(field; spatial_weight::Real=1.0)
    device = CUDA.CuArray(field)
    spectral = fftshift(ifft(device, 1), 1)
    values = Array(vec(sum(abs2, spectral; dims=(2, 3)))) .* spatial_weight
    device = nothing
    spectral = nothing
    CUDA.reclaim()
    return values
end

function gpu_project_modes(field, modes_yxm, grid)
    # Keep both GEMM operands complex; cuBLAS does not provide the mixed
    # ComplexF64-by-Float64 matrix product implied by the real HG profiles.
    mode_matrix = ComplexF64.(
        PulsePropagation._full_field_mode_matrix(modes_yxm, grid))
    device_field = CUDA.CuArray(field)
    device_modes = CUDA.CuArray(mode_matrix)
    flattened = reshape(device_field, grid.time.nt, :)
    amplitudes = flattened * device_modes
    amplitudes .*= Float64(grid.dx * grid.dy)
    result = Array(amplitudes)
    device_field = nothing
    device_modes = nothing
    amplitudes = nothing
    CUDA.reclaim()
    return result
end

function normalized(values)
    scale = maximum(values)
    return scale > 0 ? values ./ scale : copy(values)
end

function save_solution_artifacts(time, grid, initial, modes)
    full_parameters = grin210_full_field_parameters(
        grid; length=PROPAGATION_LENGTH_M)
    println(
        "stage=full_solution_start method=rk4ip",
        " length_m=", PROPAGATION_LENGTH_M,
        " dz_m=", DZ_M,
    )
    rk_seconds = @elapsed rk_solution = solve_full_field_3d(
        initial, full_parameters, DZ_M;
        precision=:float64,
        method=:rk4ip,
        saveat=PROPAGATION_LENGTH_M,
    )
    rk_final = Array(@view rk_solution.fields[:, :, :, end])
    all(isfinite, rk_final) || error(
        "Full-field RK4IP produced a non-finite value; reduce GRIN210_DZ_M.")
    spatial_weight = grid.dx * grid.dy
    rk_spectrum = gpu_integrated_spectrum(
        rk_final; spatial_weight)
    projected_rk = gpu_project_modes(rk_final, modes, grid)
    projected_rk_spectral = fftshift(ifft(projected_rk, 1), 1)
    projected_rk_spectrum = vec(sum(abs2, projected_rk_spectral; dims=2))
    projected_rk_energy = vec(sum(abs2, projected_rk; dims=1)) .* time.dt
    total_rk_energy = sum(abs2, rk_final) * grid.dx * grid.dy * time.dt
    retained_rk_energy = sum(abs2, projected_rk) * time.dt
    leakage = max(0.0, 1 - retained_rk_energy / total_rk_energy)
    rk_fluence = dropdims(sum(abs2, rk_final; dims=1); dims=1) .* time.dt
    peak_index = argmax(full_field_power(rk_final, grid))
    rk_peak_intensity = abs2.(@view rk_final[peak_index, :, :])

    reference_path = joinpath(OUTPUT_DIR, "full_field_solution.npz")
    reference_payload = Dict{String,Any}(
        "frequency_detuning_THz" => frequency_axis(time) ./ (2pi),
        "input_spectrum" => gpu_integrated_spectrum(
            initial; spatial_weight),
        "rk4ip_output_spectrum" => rk_spectrum,
        "rk4ip_output_spectrum_normalized" => normalized(rk_spectrum),
        "rk4ip_projected_spectrum" => projected_rk_spectrum,
        "rk4ip_projected_spectrum_normalized" =>
            normalized(projected_rk_spectrum),
        "rk4ip_projected_modal_energy" => projected_rk_energy,
        "rk4ip_fluence" => rk_fluence,
        "rk4ip_peak_time_intensity" => rk_peak_intensity,
        "rk4ip_peak_time_index" => [peak_index],
        "rk4ip_subspace_leakage" => [leakage],
        "rk4ip_wall_seconds" => [rk_seconds],
        "x_m" => grid.x,
        "y_m" => grid.y,
        "dz_m" => [DZ_M],
        "length_m" => [PROPAGATION_LENGTH_M],
    )

    if RUN_STRANG_SOLUTION
        println(
            "stage=full_solution_start method=strang",
            " length_m=", PROPAGATION_LENGTH_M,
            " dz_m=", STRANG_DZ_M,
        )
        strang_seconds = @elapsed strang_solution = solve_full_field_3d(
            initial, full_parameters, STRANG_DZ_M;
            precision=:float64,
            method=:strang,
            saveat=PROPAGATION_LENGTH_M,
        )
        strang_final = Array(@view strang_solution.fields[:, :, :, end])
        all(isfinite, strang_final) || error(
            "Full-field Strang produced a non-finite value; reduce " *
            "GRIN210_STRANG_DZ_M.")
        strang_spectrum = gpu_integrated_spectrum(
            strang_final; spatial_weight)
        strang_fluence = dropdims(
            sum(abs2, strang_final; dims=1); dims=1) .* time.dt
        strang_peak_intensity = abs2.(@view strang_final[peak_index, :, :])
        reference_payload["strang_output_spectrum"] = strang_spectrum
        reference_payload["strang_output_spectrum_normalized"] =
            normalized(strang_spectrum)
        reference_payload["strang_fluence"] = strang_fluence
        reference_payload["strang_peak_time_intensity"] =
            strang_peak_intensity
        reference_payload["strang_wall_seconds"] = [strang_seconds]
        reference_payload["strang_dz_m"] = [STRANG_DZ_M]
        reference_payload["strang_rk4ip_field_error"] = [
            phase_invariant_relative_error(strang_final, rk_final)]
        reference_payload["strang_rk4ip_spectrum_error"] = [
            norm(strang_spectrum - rk_spectrum) / norm(rk_spectrum)]
        strang_solution = nothing
        strang_final = nothing
    end
    npzwrite(reference_path, reference_payload)
    println(
        "stage=full_solution_complete",
        " wall_seconds=", rk_seconds,
        " leakage=", leakage,
        " output_path=", reference_path,
    )

    for target in TARGET_ERRORS
        cp = load_grin210_cp(GRIN210_MODE_COUNT, target)
        parameters = grin210_modal_parameters(
            time, cp; length=PROPAGATION_LENGTH_M)
        modal_initial = grin210_modal_initial(time, GRIN210_MODE_COUNT)
        println(
            "stage=cp_solution_start",
            " target=", target,
            " rank=", cp_rank(cp),
        )
        seconds = @elapsed solution = solve_mmgnlse(
            modal_initial, parameters, DZ_M;
            backend=CP_BACKEND,
            precision=:float64,
            adaptive=false,
            saveat=PROPAGATION_LENGTH_M,
        )
        final = Array(get_temporal_field(solution; z=:final))[:, :, 1]
        all(isfinite, final) || error(
            "Modal CP solution produced a non-finite value for target=$target.")
        spectral = fftshift(ifft(final, 1), 1)
        spectrum = vec(sum(abs2, spectral; dims=2))
        modal_energy = vec(sum(abs2, final; dims=1)) .* time.dt
        tag = grin210_target_tag(target)
        output_path = joinpath(OUTPUT_DIR, "modal_solution_$(tag).npz")
        npzwrite(output_path, Dict(
            "frequency_detuning_THz" => frequency_axis(time) ./ (2pi),
            "output_spectrum" => spectrum,
            "output_spectrum_normalized" => normalized(spectrum),
            "output_modal_energy" => modal_energy,
            "projected_full_field_spectrum" => projected_rk_spectrum,
            "projected_full_field_modal_energy" => projected_rk_energy,
            "field_relative_error" => [
                phase_invariant_relative_error(final, projected_rk)],
            "spectrum_relative_error" => [
                norm(spectrum - projected_rk_spectrum) /
                norm(projected_rk_spectrum)],
            "modal_energy_relative_error" => [
                norm(modal_energy - projected_rk_energy) /
                norm(projected_rk_energy)],
            "target_error" => [target],
            "reported_cp_error" => [cp.relative_error],
            "cp_rank" => [cp_rank(cp)],
            "wall_seconds" => [seconds],
            "dz_m" => [DZ_M],
            "length_m" => [PROPAGATION_LENGTH_M],
        ))
        println(
            "stage=cp_solution_complete",
            " target=", target,
            " seconds=", seconds,
            " output_path=", output_path,
        )
        solution = nothing
        CUDA.reclaim()
    end
    return nothing
end

time = TimeGrid(NT, TIME_WINDOW_PS)
half_width = TRANSVERSE_HALF_WIDTH_UM * 1e-6
grid = FullField3DGrid(
    time;
    Nx=TRANSVERSE_COUNT,
    Ny=TRANSVERSE_COUNT,
    dx=2half_width / TRANSVERSE_COUNT,
)
initial, modes = grin210_full_field_initial(grid)

println(
    "stage=configuration",
    " gpu=", CUDA.name(CUDA.device()),
    " nt=", NT,
    " time_window_ps=", TIME_WINDOW_PS,
    " nx=", TRANSVERSE_COUNT,
    " half_width_um=", TRANSVERSE_HALF_WIDTH_UM,
    " dx_um=", grid.dx * 1e6,
    " dz_m=", DZ_M,
    " length_m=", PROPAGATION_LENGTH_M,
    " cp_backend=", CP_BACKEND,
    " precision=", PRECISION,
    " run_timings=", RUN_TIMINGS,
    " run_solutions=", RUN_SOLUTIONS,
)

if RUN_TIMINGS
    full_parameters = grin210_full_field_parameters(grid; length=DZ_M)
    full_timing, full_bytes = time_full_field_step(initial, full_parameters)
    println(
        "stage=timing method=full_field_rk4ip",
        " median_ms=", full_timing.median_ms,
        " mean_ms=", full_timing.mean_ms,
        " minimum_ms=", full_timing.minimum_ms,
        " device_used_bytes=", full_bytes,
    )
    GC.gc(true)
    CUDA.reclaim()

    rows = NamedTuple[]
    median_table = fill(NaN, length(MODE_COUNTS), length(TARGET_ERRORS))
    rank_table = fill(-1, size(median_table))
    error_table = fill(NaN, size(median_table))
    for (mode_index, mode_count) in pairs(MODE_COUNTS)
        for (target_index, target) in pairs(TARGET_ERRORS)
            cp = load_grin210_cp(mode_count, target)
            timing, used_bytes = time_cp_step(cp, time)
            median_table[mode_index, target_index] = timing.median_ms
            rank_table[mode_index, target_index] = cp_rank(cp)
            error_table[mode_index, target_index] = cp.relative_error
            row = (;
                modes=mode_count,
                groups=grin210_complete_group_count(mode_count),
                target,
                rank=cp_rank(cp),
                cp_error=cp.relative_error,
                timing,
                device_used_bytes=used_bytes,
            )
            push!(rows, row)
            println(
                "stage=timing method=modal_cp_rk4ip",
                " modes=", mode_count,
                " target=", target,
                " rank=", cp_rank(cp),
                " median_ms=", timing.median_ms,
                " mean_ms=", timing.mean_ms,
                " minimum_ms=", timing.minimum_ms,
                " device_used_bytes=", used_bytes,
            )
            cp = nothing
            GC.gc(true)
            CUDA.reclaim()
        end
    end

    timing_csv = joinpath(OUTPUT_DIR, "crossover_timings.csv")
    write_timing_csv(timing_csv, full_timing, full_bytes, rows)
    npzwrite(joinpath(OUTPUT_DIR, "crossover_timings.npz"), Dict(
        "mode_counts" => MODE_COUNTS,
        "target_errors" => TARGET_ERRORS,
        "cp_median_ms" => median_table,
        "cp_ranks" => rank_table,
        "cp_reported_errors" => error_table,
        "full_field_median_ms" => [full_timing.median_ms],
        "full_field_raw_ms" => full_timing.raw_ms,
        "dz_m" => [DZ_M],
        "nt" => [NT],
        "transverse_count" => [TRANSVERSE_COUNT],
        "transverse_half_width_um" => [TRANSVERSE_HALF_WIDTH_UM],
    ))
    println("stage=timings_complete output_path=", timing_csv)
end

RUN_SOLUTIONS && save_solution_artifacts(time, grid, initial, modes)

open(joinpath(OUTPUT_DIR, "run_summary.txt"), "w") do io
    configuration = grin210_configuration()
    for name in propertynames(configuration)
        println(io, name, '=', getproperty(configuration, name))
    end
    println(io, "gpu=", CUDA.name(CUDA.device()))
    println(io, "nt=", NT)
    println(io, "time_window_ps=", TIME_WINDOW_PS)
    println(io, "transverse_count=", TRANSVERSE_COUNT)
    println(io, "transverse_half_width_um=", TRANSVERSE_HALF_WIDTH_UM)
    println(io, "dx_um=", grid.dx * 1e6)
    println(io, "dz_m=", DZ_M)
    println(io, "strang_dz_m=", STRANG_DZ_M)
    println(io, "length_m=", PROPAGATION_LENGTH_M)
    println(io, "mode_counts=", join(MODE_COUNTS, ','))
    println(io, "target_errors=", join(TARGET_ERRORS, ','))
    println(io, "cp_backend=", CP_BACKEND)
    println(io, "precision=", PRECISION)
    println(io, "timing_warmup=", TIMING_WARMUP)
    println(io, "timing_samples=", TIMING_SAMPLES)
    println(io, "run_timings=", RUN_TIMINGS)
    println(io, "run_solutions=", RUN_SOLUTIONS)
    println(io, "run_strang_solution=", RUN_STRANG_SOLUTION)
end

println("stage=complete output_dir=", OUTPUT_DIR)
