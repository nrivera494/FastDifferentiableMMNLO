using CUDA
using FFTW
using LinearAlgebra
using NPZ
using Printf
using PulsePropagation

include(joinpath(@__DIR__, "grin_210_common.jl"))

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const GRIN210_ACCURACY_EXT =
    Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
GRIN210_ACCURACY_EXT === nothing && error(
    "PulsePropagation CUDA extension did not load.")

parse_csv(::Type{T}, value) where {T} =
    parse.(T, filter(!isempty, strip.(split(value, ','))))
env_bool(name, default) = lowercase(get(ENV, name, string(default))) in
                          ("1", "true", "yes", "on")

const NT = parse(Int, get(ENV, "GRIN210_NT", "4096"))
const TIME_WINDOW_PS = parse(
    Float64, get(ENV, "GRIN210_TIME_WINDOW_PS", "10.0"))
const TRANSVERSE_COUNT = parse(
    Int, get(ENV, "GRIN210_TRANSVERSE_COUNT", "96"))
const TRANSVERSE_HALF_WIDTH_UM = parse(
    Float64, get(ENV, "GRIN210_TRANSVERSE_HALF_WIDTH_UM", "120.0"))
const PROPAGATION_LENGTH_M = parse(
    Float64, get(ENV, "GRIN210_LENGTH_M", "0.025"))
const DZ_VALUES = parse_csv(
    Float64, get(ENV, "GRIN210_ACCURACY_DZS", "1e-5,5e-6"))
const LINEAR_LENGTH_M = parse(
    Float64, get(ENV, "GRIN210_LINEAR_LENGTH_M", "1e-3"))
const TIGHT_TARGET = parse(
    Float64, get(ENV, "GRIN210_ACCURACY_CP_TARGET", "1e-4"))
const CP_BACKEND = Symbol(lowercase(get(
    ENV, "GRIN210_CP_BACKEND", "cuda_cp_optimized")))
const RUN_FULL = env_bool("GRIN210_ACCURACY_RUN_FULL", false)
const RUN_MODAL = env_bool("GRIN210_ACCURACY_RUN_MODAL", false)
const RUN_LINEAR = env_bool("GRIN210_ACCURACY_RUN_LINEAR", false)
const RUN_BRIDGE = env_bool("GRIN210_ACCURACY_RUN_BRIDGE", false)
const RUN_SWEEP = env_bool("GRIN210_ACCURACY_RUN_SWEEP", false)
const CP_DIR = get(
    ENV, "GRIN210_CP_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_cp"))
const TIMING_CSV = get(
    ENV, "GRIN210_TIMING_CSV",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_crossover",
             "crossover_timings.csv"))
const OUTPUT_DIR = get(
    ENV, "GRIN210_ACCURACY_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_accuracy"))

length(DZ_VALUES) == 2 || error(
    "GRIN210_ACCURACY_DZS must contain coarse and fine dz values.")
DZ_VALUES[1] > DZ_VALUES[2] > 0 || error(
    "GRIN210_ACCURACY_DZS must be ordered coarse to fine and positive.")
isapprox(DZ_VALUES[1] / DZ_VALUES[2], 2; rtol=1e-12) || error(
    "The accuracy driver currently requires a factor-two dz refinement.")
PROPAGATION_LENGTH_M > 0 || error("GRIN210_LENGTH_M must be positive.")
LINEAR_LENGTH_M > 0 || error("GRIN210_LINEAR_LENGTH_M must be positive.")
TIGHT_TARGET in GRIN210_DEFAULT_TARGET_ERRORS || error(
    "The tight CP target must be one of $(GRIN210_DEFAULT_TARGET_ERRORS).")
CP_BACKEND in (:cuda, :cuda_cp_optimized, :cuda_optimized) || error(
    "Unsupported CP backend $CP_BACKEND.")

mkpath(OUTPUT_DIR)

dz_tag(dz) = replace(@sprintf("%.6g", Float64(dz) * 1e6), '.' => 'p') * "um"
full_path(dz) = joinpath(OUTPUT_DIR, "full_dz_$(dz_tag(dz)).npz")
modal_path(label, dz) =
    joinpath(OUTPUT_DIR, "modal_$(label)_dz_$(dz_tag(dz)).npz")

function load_grin210_cp(mode_count, target)
    paths = grin210_cp_cache_paths(CP_DIR, mode_count, target)
    isfile(paths.data) && isfile(paths.metadata) || error(
        "Missing CP cache for modes=$mode_count target=$target.")
    cp = load_cp_decomposition(paths.data, paths.metadata)
    cp.metadata.nmodes == mode_count || error(
        "Cached CP mode count does not match $mode_count.")
    return cp
end

function gpu_project_modes(field, modes_yxm, grid)
    mode_matrix = ComplexF64.(
        PulsePropagation._full_field_mode_matrix(modes_yxm, grid))
    device_field = CUDA.CuArray(field)
    device_modes = CUDA.CuArray(mode_matrix)
    amplitudes = reshape(device_field, grid.time.nt, :) * device_modes
    amplitudes .*= Float64(grid.dx * grid.dy)
    result = Array(amplitudes)
    device_field = nothing
    device_modes = nothing
    amplitudes = nothing
    CUDA.reclaim()
    return result
end

function gpu_integrated_spectrum(field, grid)
    device = CUDA.CuArray(field)
    spectral = fftshift(ifft(device, 1), 1)
    result = Array(vec(sum(abs2, spectral; dims=(2, 3)))) .*
             Float64(grid.dx * grid.dy)
    device = nothing
    spectral = nothing
    CUDA.reclaim()
    return result
end

function modal_spectrum(field)
    spectral = fftshift(ifft(field, 1), 1)
    return vec(sum(abs2, spectral; dims=2))
end

function phase_aligned_difference_norm(candidate, reference)
    overlap = dot(vec(reference), vec(candidate))
    phase = iszero(overlap) ? one(overlap) : overlap / abs(overlap)
    return norm(vec(candidate .- phase .* reference))
end

relative_error(candidate, reference) = norm(vec(candidate - reference)) /
                                       norm(vec(reference))

function basis_consistency(initial, modes, grid)
    mode_matrix = PulsePropagation._full_field_mode_matrix(modes, grid)
    device_modes = CUDA.CuArray(mode_matrix)
    gram = Array(adjoint(device_modes) * device_modes) .* (grid.dx * grid.dy)
    device_modes = nothing
    CUDA.reclaim()
    identity_matrix = Matrix{Float64}(I, size(gram))
    offdiagonal = gram - Diagonal(diag(gram))
    modal_initial = @view grin210_modal_initial(
        grid.time, GRIN210_MODE_COUNT)[:, :, 1]
    projected_initial = gpu_project_modes(initial, modes, grid)
    metrics = (
        gram_relative_error=norm(gram - identity_matrix) /
                            norm(identity_matrix),
        maximum_diagonal_error=maximum(abs.(diag(gram) .- 1)),
        maximum_offdiagonal=maximum(abs, offdiagonal),
        initial_projection_error=phase_invariant_relative_error(
            projected_initial, modal_initial),
    )
    npzwrite(joinpath(OUTPUT_DIR, "basis_consistency.npz"), Dict(
        "gram" => gram,
        "gram_relative_error" => [metrics.gram_relative_error],
        "maximum_diagonal_error" => [metrics.maximum_diagonal_error],
        "maximum_offdiagonal" => [metrics.maximum_offdiagonal],
        "initial_projection_error" => [metrics.initial_projection_error],
    ))
    println("stage=basis_consistency", " metrics=", metrics)
    return metrics
end

function run_full_solution(initial, modes, grid, dz;
                           length=PROPAGATION_LENGTH_M,
                           n2=GRIN210_N2_M2_PER_W,
                           raman=GRIN210_MATERIAL.raman,
                           save_midpoint=true,
                           output_path=full_path(dz))
    parameters = grin210_full_field_parameters(
        grid; length, n2, raman, self_steepening=true)
    save_coordinates = save_midpoint ? [length / 2, length] : length
    println("stage=full_start", " dz_m=", dz, " length_m=", length,
            " n2=", n2)
    wall_seconds = @elapsed solution = solve_full_field_3d(
        initial, parameters, dz;
        precision=:float64, method=:rk4ip, saveat=save_coordinates)
    final = Array(get_temporal_field(solution; z=:final))
    midpoint = save_midpoint ?
               Array(get_temporal_field(solution; z=length / 2)) : nothing
    projected_initial = gpu_project_modes(initial, modes, grid)
    projected_final = gpu_project_modes(final, modes, grid)
    projected_midpoint = save_midpoint ?
                         gpu_project_modes(midpoint, modes, grid) :
                         zeros(ComplexF64, 0, 0)
    output_spectrum = gpu_integrated_spectrum(final, grid)
    projected_spectrum = modal_spectrum(projected_final)
    modal_energy = vec(sum(abs2, projected_final; dims=1)) .* grid.time.dt
    total_energy = sum(abs2, final) * grid.dx * grid.dy * grid.time.dt
    retained_energy = sum(abs2, projected_final) * grid.time.dt
    leakage = max(0.0, 1 - retained_energy / total_energy)
    npzwrite(output_path, Dict(
        "projected_initial" => projected_initial,
        "projected_midpoint" => projected_midpoint,
        "projected_final" => projected_final,
        "output_spectrum" => output_spectrum,
        "projected_spectrum" => projected_spectrum,
        "projected_modal_energy" => modal_energy,
        "total_energy" => [total_energy],
        "retained_energy" => [retained_energy],
        "subspace_leakage" => [leakage],
        "wall_seconds" => [wall_seconds],
        "dz_m" => [dz],
        "length_m" => [length],
        "nt" => [grid.time.nt],
        "transverse_count" => [Base.length(grid.x)],
    ))
    solution = nothing
    midpoint = nothing
    GC.gc(true)
    CUDA.reclaim()
    println("stage=full_complete", " dz_m=", dz,
            " wall_seconds=", wall_seconds, " leakage=", leakage,
            " output_path=", output_path)
    return (;
        final, projected_final, output_spectrum, projected_spectrum,
        modal_energy, total_energy, retained_energy, leakage, wall_seconds,
    )
end

function run_full_convergence(initial, modes, grid)
    coarse = run_full_solution(initial, modes, grid, DZ_VALUES[1])
    fine = run_full_solution(initial, modes, grid, DZ_VALUES[2])
    field_difference = phase_invariant_relative_error(coarse.final, fine.final)
    projected_difference = phase_invariant_relative_error(
        coarse.projected_final, fine.projected_final)
    spectrum_difference = relative_error(
        coarse.output_spectrum, fine.output_spectrum)
    projected_spectrum_difference = relative_error(
        coarse.projected_spectrum, fine.projected_spectrum)
    richardson_field = field_difference / 15
    richardson_projected = projected_difference / 15
    open(joinpath(OUTPUT_DIR, "full_convergence.csv"), "w") do io
        println(io, "metric,value")
        println(io, "field_coarse_fine,$field_difference")
        println(io, "projected_field_coarse_fine,$projected_difference")
        println(io, "spectrum_coarse_fine,$spectrum_difference")
        println(io, "projected_spectrum_coarse_fine,$projected_spectrum_difference")
        println(io, "richardson_fine_field,$richardson_field")
        println(io, "richardson_fine_projected_field,$richardson_projected")
    end
    println("stage=full_convergence_complete",
            " field_difference=", field_difference,
            " projected_difference=", projected_difference,
            " richardson_field=", richardson_field)
    coarse = nothing
    fine = nothing
    GC.gc(true)
    return nothing
end

function run_modal_solution(overlap, time, dz, label;
                            length=PROPAGATION_LENGTH_M,
                            save_midpoint=true)
    parameters = grin210_modal_parameters(time, overlap; length)
    initial = grin210_modal_initial(time, overlap.metadata.nmodes)
    save_coordinates = save_midpoint ? [length / 2, length] : length
    println("stage=modal_start", " label=", label,
            " rank=", cp_rank(overlap), " dz_m=", dz)
    wall_seconds = @elapsed solution = solve_mmgnlse(
        initial, parameters, dz;
        backend=CP_BACKEND, precision=:float64, adaptive=false,
        saveat=save_coordinates)
    final = Array(get_temporal_field(solution; z=:final))[:, :, 1]
    midpoint = save_midpoint ?
               Array(get_temporal_field(solution; z=length / 2))[:, :, 1] :
               zeros(ComplexF64, 0, 0)
    output_path = modal_path(label, dz)
    npzwrite(output_path, Dict(
        "initial" => Array(@view initial[:, :, 1]),
        "midpoint" => midpoint,
        "final" => final,
        "output_spectrum" => modal_spectrum(final),
        "modal_energy" => vec(sum(abs2, final; dims=1)) .* time.dt,
        "wall_seconds" => [wall_seconds],
        "dz_m" => [dz],
        "length_m" => [length],
        "rank" => [cp_rank(overlap)],
    ))
    solution = nothing
    GC.gc(true)
    CUDA.reclaim()
    println("stage=modal_complete", " label=", label,
            " wall_seconds=", wall_seconds, " output_path=", output_path)
    return (; final, midpoint, wall_seconds)
end

function run_modal_convergence(authority, tight, time)
    tight_coarse = run_modal_solution(
        tight, time, DZ_VALUES[1], "tight")
    tight_fine = run_modal_solution(
        tight, time, DZ_VALUES[2], "tight")
    authority_fine = run_modal_solution(
        authority, time, DZ_VALUES[2], "authority")
    integrator_difference = phase_invariant_relative_error(
        tight_coarse.final, tight_fine.final)
    compression_difference = phase_invariant_relative_error(
        tight_fine.final, authority_fine.final)
    authority_spectrum = modal_spectrum(authority_fine.final)
    compression_spectrum = relative_error(
        modal_spectrum(tight_fine.final), authority_spectrum)
    open(joinpath(OUTPUT_DIR, "modal_convergence.csv"), "w") do io
        println(io, "metric,value")
        println(io, "tight_coarse_fine_field,$integrator_difference")
        println(io, "tight_richardson_fine_field,$(integrator_difference / 15)")
        println(io, "tight_authority_field,$compression_difference")
        println(io, "tight_authority_spectrum,$compression_spectrum")
        println(io, "tight_rank,$(cp_rank(tight))")
        println(io, "authority_rank,$(cp_rank(authority))")
    end
    println("stage=modal_convergence_complete",
            " integrator_difference=", integrator_difference,
            " compression_difference=", compression_difference,
            " compression_spectrum=", compression_spectrum)
    return nothing
end

function modal_nonlinear_rhs(states, authority, time)
    parameters = grin210_modal_parameters(time, authority; length=1e-3)
    optimization = GRIN210_ACCURACY_EXT._cuda_mmgnlse_cp_optimization(
        CP_BACKEND)
    cache = GRIN210_ACCURACY_EXT._cuda_mmgnlse_forward_cache(
        parameters, optimization)
    seed = CUDA.CuArray(reshape(states[1], time.nt,
                                authority.metadata.nmodes, 1))
    field_w = ifft(seed, 1)
    workspace = GRIN210_ACCURACY_EXT._cuda_mmgnlse_cp_precision_workspace(
        seed, cache, cache, :float64)
    output_w = similar(field_w)
    results = Matrix{ComplexF64}[]
    for state in states
        copyto!(seed, reshape(state, size(seed)))
        field_w .= ifft(seed, 1)
        GRIN210_ACCURACY_EXT._cuda_mmgnlse_cp_integrator_rhs!(
            output_w, field_w, cache, workspace)
        push!(results, Array(fft(output_w, 1))[:, :, 1])
    end
    seed = nothing
    field_w = nothing
    output_w = nothing
    workspace = nothing
    cache = nothing
    GC.gc(true)
    CUDA.reclaim()
    return results
end

function full_nonlinear_rhs(states, modes, grid)
    parameters = grin210_full_field_parameters(grid; length=1e-3)
    results = Matrix{ComplexF64}[]
    for (index, state) in pairs(states)
        field = reconstruct_full_field(state, modes, grid)
        workspace = GRIN210_ACCURACY_EXT._cuda_full_field_workspace(
            field, parameters, Float64, :rk4ip)
        GRIN210_ACCURACY_EXT._cuda_full_field_nonlinear_rhs_time!(
            workspace.derivative, workspace.field, workspace)
        rhs_field = Array(workspace.derivative)
        push!(results, gpu_project_modes(rhs_field, modes, grid))
        workspace = nothing
        field = nothing
        rhs_field = nothing
        GC.gc(true)
        CUDA.reclaim()
        println("stage=full_rhs_complete state_index=", index)
    end
    return results
end

function run_bridge(authority, modes, grid)
    fine_data = npzread(full_path(DZ_VALUES[2]))
    states = Matrix{ComplexF64}[
        ComplexF64.(fine_data["projected_initial"]),
        ComplexF64.(fine_data["projected_midpoint"]),
        ComplexF64.(fine_data["projected_final"]),
    ]
    full_rhs = full_nonlinear_rhs(states, modes, grid)
    modal_rhs = modal_nonlinear_rhs(states, authority, grid.time)
    labels = ("initial", "midpoint", "final")
    open(joinpath(OUTPUT_DIR, "nonlinear_bridge.csv"), "w") do io
        println(io, "state,relative_error,norm_ratio,overlap_phase_rad")
        for index in eachindex(states)
            error = relative_error(full_rhs[index], modal_rhs[index])
            ratio = norm(full_rhs[index]) / norm(modal_rhs[index])
            overlap = dot(vec(modal_rhs[index]), vec(full_rhs[index]))
            phase = angle(overlap)
            println(io, join((labels[index], error, ratio, phase), ','))
            println("stage=nonlinear_bridge", " state=", labels[index],
                    " relative_error=", error, " norm_ratio=", ratio,
                    " phase=", phase)
        end
    end
    return nothing
end

function run_linear_consistency(initial, modes, grid, tight)
    full_results = NamedTuple[]
    for dz in DZ_VALUES
        path = joinpath(OUTPUT_DIR, "linear_full_dz_$(dz_tag(dz)).npz")
        push!(full_results, run_full_solution(
            initial, modes, grid, dz;
            length=LINEAR_LENGTH_M, n2=0.0, raman=NoRaman(),
            save_midpoint=false, output_path=path))
    end
    parameters = grin210_modal_parameters(
        grid.time, tight; length=LINEAR_LENGTH_M,
        n2=0.0, raman=NoRaman())
    modal_initial = grin210_modal_initial(grid.time, GRIN210_MODE_COUNT)
    modal_solution = solve_mmgnlse(
        modal_initial, parameters, DZ_VALUES[1];
        backend=CP_BACKEND, precision=:float64, adaptive=false,
        saveat=LINEAR_LENGTH_M)
    modal_final = Array(get_temporal_field(
        modal_solution; z=:final))[:, :, 1]
    coarse_modal = phase_invariant_relative_error(
        full_results[1].projected_final, modal_final)
    fine_modal = phase_invariant_relative_error(
        full_results[2].projected_final, modal_final)
    coarse_fine = phase_invariant_relative_error(
        full_results[1].projected_final,
        full_results[2].projected_final)
    open(joinpath(OUTPUT_DIR, "linear_consistency.csv"), "w") do io
        println(io, "metric,value")
        println(io, "full_coarse_vs_modal,$coarse_modal")
        println(io, "full_fine_vs_modal,$fine_modal")
        println(io, "full_coarse_vs_fine,$coarse_fine")
        println(io, "full_fine_richardson,$(coarse_fine / 15)")
    end
    println("stage=linear_consistency_complete",
            " coarse_modal=", coarse_modal,
            " fine_modal=", fine_modal,
            " coarse_fine=", coarse_fine)
    return nothing
end

function timing_lookup(path)
    isfile(path) || error("Missing timing CSV $path.")
    lookup = Dict{Tuple{Int,Float64},Float64}()
    lines = readlines(path)
    header = split(first(lines), ',')
    indices = Dict(name => findfirst(==(name), header) for name in header)
    for line in Iterators.drop(lines, 1)
        columns = split(line, ',')
        columns[indices["method"]] == "modal_cp_rk4ip" || continue
        modes = parse(Int, columns[indices["modes"]])
        target = parse(Float64, columns[indices["target"]])
        median = parse(Float64, columns[indices["median_ms"]])
        lookup[(modes, target)] = median
    end
    return lookup
end

function run_accuracy_sweep(time)
    reference = npzread(full_path(DZ_VALUES[2]))
    projected_reference = ComplexF64.(reference["projected_final"])
    total_energy = only(reference["total_energy"])
    full_spectrum = vec(reference["output_spectrum"])
    timing = timing_lookup(TIMING_CSV)
    output_path = joinpath(OUTPUT_DIR, "accuracy_sweep.csv")
    open(output_path, "w") do io
        println(io,
            "modes,target,rank,median_step_ms,projected_field_error," *
            "truncation_field_floor,total_field_error,spectrum_error," *
            "modal_energy_error,qualifies_1e_2,qualifies_1e_3," *
            "qualifies_1e_4,wall_seconds")
        for mode_count in GRIN210_DEFAULT_MODE_COUNTS
            reference_m = @view projected_reference[:, 1:mode_count]
            retained_energy = sum(abs2, reference_m) * time.dt
            tail_energy = max(total_energy - retained_energy, 0.0)
            truncation_floor = sqrt(tail_energy / total_energy)
            reference_modal_energy = vec(sum(abs2, reference_m; dims=1)) .* time.dt
            for target in GRIN210_DEFAULT_TARGET_ERRORS
                cp = load_grin210_cp(mode_count, target)
                parameters = grin210_modal_parameters(
                    time, cp; length=PROPAGATION_LENGTH_M)
                initial = grin210_modal_initial(time, mode_count)
                start = time_ns()
                solution = solve_mmgnlse(
                    initial, parameters, DZ_VALUES[1];
                    backend=CP_BACKEND, precision=:float64,
                    adaptive=false, saveat=PROPAGATION_LENGTH_M)
                CUDA.synchronize()
                wall_seconds = (time_ns() - start) / 1e9
                final = Array(get_temporal_field(
                    solution; z=:final))[:, :, 1]
                difference_norm = phase_aligned_difference_norm(
                    final, reference_m)
                projected_error = difference_norm / norm(reference_m)
                total_error = sqrt(
                    (difference_norm^2 * time.dt + tail_energy) /
                    total_energy)
                spectrum_error = relative_error(
                    modal_spectrum(final), full_spectrum)
                modal_energy = vec(sum(abs2, final; dims=1)) .* time.dt
                modal_energy_error = relative_error(
                    modal_energy, reference_modal_energy)
                median_ms = timing[(mode_count, target)]
                qualifies = total_error .<= (1e-2, 1e-3, 1e-4)
                println(io, join((
                    mode_count, target, cp_rank(cp), median_ms,
                    projected_error, truncation_floor, total_error,
                    spectrum_error, modal_energy_error,
                    qualifies[1], qualifies[2], qualifies[3],
                    wall_seconds,
                ), ','))
                flush(io)
                println("stage=accuracy_sweep", " modes=", mode_count,
                        " target=", target, " total_error=", total_error,
                        " truncation_floor=", truncation_floor,
                        " wall_seconds=", wall_seconds)
                solution = nothing
                final = nothing
                cp = nothing
                GC.gc(true)
                CUDA.reclaim()
            end
        end
    end
    println("stage=accuracy_sweep_complete output_path=", output_path)
    return nothing
end

println("stage=setup_time_grid_start")
time = TimeGrid(NT, TIME_WINDOW_PS)
println("stage=setup_time_grid_complete")
half_width = TRANSVERSE_HALF_WIDTH_UM * 1e-6
println("stage=setup_full_grid_start")
grid = FullField3DGrid(
    time; Nx=TRANSVERSE_COUNT, Ny=TRANSVERSE_COUNT,
    dx=2half_width / TRANSVERSE_COUNT)
println("stage=setup_full_grid_complete")
println("stage=setup_initial_start")
initial, modes = grin210_full_field_initial(grid)
println("stage=setup_initial_complete")

println("stage=configuration", " gpu=", CUDA.name(CUDA.device()),
        " nt=", NT, " nx=", TRANSVERSE_COUNT,
        " dz_values=", DZ_VALUES, " length_m=", PROPAGATION_LENGTH_M,
        " run_full=", RUN_FULL, " run_modal=", RUN_MODAL,
        " run_linear=", RUN_LINEAR, " run_bridge=", RUN_BRIDGE,
        " run_sweep=", RUN_SWEEP)

basis_consistency(initial, modes, grid)
tight = (RUN_MODAL || RUN_LINEAR) ?
        load_grin210_cp(GRIN210_MODE_COUNT, TIGHT_TARGET) : nothing
authority = (RUN_MODAL || RUN_BRIDGE) ?
            grin210_quadrature_source(
                GRIN210_MODE_COUNT;
                quadrature_order=GRIN210_AUTHORITY_QUADRATURE_ORDER) : nothing

RUN_FULL && run_full_convergence(initial, modes, grid)
RUN_MODAL && run_modal_convergence(authority, tight, time)
RUN_LINEAR && run_linear_consistency(initial, modes, grid, tight)
RUN_BRIDGE && run_bridge(authority, modes, grid)
RUN_SWEEP && run_accuracy_sweep(time)

println("stage=complete output_dir=", OUTPUT_DIR)
