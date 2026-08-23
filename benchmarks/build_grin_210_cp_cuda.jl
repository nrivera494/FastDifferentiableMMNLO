using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation
using Random

include(joinpath(@__DIR__, "grin_210_common.jl"))

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

parse_csv(::Type{T}, value) where {T} =
    parse.(T, filter(!isempty, strip.(split(value, ','))))

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
const QUADRATURE_ORDER = parse(
    Int, get(ENV, "GRIN210_CP_QUADRATURE_ORDER",
             string(GRIN210_WORKING_QUADRATURE_ORDER)))
const AUTHORITY_QUADRATURE_ORDER = parse(
    Int, get(ENV, "GRIN210_CP_AUTHORITY_ORDER",
             string(GRIN210_AUTHORITY_QUADRATURE_ORDER)))
const QUADRATURE_TOLERANCE = parse(
    Float64, get(ENV, "GRIN210_CP_QUADRATURE_TOLERANCE", "1e-6"))
const USER_MIN_RANK = parse(Int, get(ENV, "GRIN210_CP_MIN_RANK", "0"))
const MIN_RANK_FACTOR = parse(
    Float64, get(ENV, "GRIN210_CP_MIN_RANK_FACTOR", "2.0"))
const RANK_STEP = parse(Int, get(ENV, "GRIN210_CP_RANK_STEP", "64"))
const MAX_RANK = parse(Int, get(ENV, "GRIN210_CP_MAX_RANK", "2304"))
const MAXITER = parse(Int, get(ENV, "GRIN210_CP_MAXITER", "150"))
const ALS_TOLERANCE = parse(
    Float64, get(ENV, "GRIN210_CP_ALS_TOLERANCE", "1e-8"))
const RIDGE = parse(Float64, get(ENV, "GRIN210_CP_RIDGE", "1e-10"))
const CHECK_EVERY = parse(Int, get(ENV, "GRIN210_CP_CHECK_EVERY", "5"))
const RESTARTS = parse(Int, get(ENV, "GRIN210_CP_RESTARTS", "1"))
const ERROR_BLOCK_SIZE = parse(
    Int, get(ENV, "GRIN210_CP_ERROR_BLOCK_SIZE", "256"))
const RANDOM_ACTIONS = parse(
    Int, get(ENV, "GRIN210_CP_RANDOM_ACTIONS", "4"))
const SEED = parse(Int, get(ENV, "GRIN210_CP_SEED", "20220819"))
const OUTPUT_DIR = get(
    ENV,
    "GRIN210_CP_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_cp"),
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
AUTHORITY_QUADRATURE_ORDER > QUADRATURE_ORDER || error(
    "The authority quadrature order must exceed the working order.")
RANK_STEP > 0 || error("GRIN210_CP_RANK_STEP must be positive.")
USER_MIN_RANK >= 0 || error("GRIN210_CP_MIN_RANK must be nonnegative.")
MIN_RANK_FACTOR > 0 || error("GRIN210_CP_MIN_RANK_FACTOR must be positive.")
MAX_RANK > 0 || error("GRIN210_CP_MAX_RANK must be positive.")
RANDOM_ACTIONS > 0 || error("GRIN210_CP_RANDOM_ACTIONS must be positive.")

mkpath(OUTPUT_DIR)

function first_rank(mode_count, source_rank)
    requested = if USER_MIN_RANK > 0
        USER_MIN_RANK
    else
        RANK_STEP * ceil(Int, MIN_RANK_FACTOR * mode_count / RANK_STEP)
    end
    return clamp(requested, 1, min(MAX_RANK, source_rank))
end

function write_progress(path, rows)
    open(path, "w") do io
        println(io,
            "modes,groups,target,rank,reported_error,authority_error," *
            "random_action_error,fit_seconds,quadrature_error,source_rank," *
            "authority_rank,source_checksum,data_path,metadata_path")
        for row in rows
            println(io, join((
                row.modes,
                row.groups,
                row.target,
                row.rank,
                row.reported_error,
                row.authority_error,
                row.random_action_error,
                row.fit_seconds,
                row.quadrature_error,
                row.source_rank,
                row.authority_rank,
                row.source_checksum,
                row.data_path,
                row.metadata_path,
            ), ','))
        end
    end
end

rank_table = fill(-1, length(MODE_COUNTS), length(TARGET_ERRORS))
reported_error_table = fill(NaN, size(rank_table))
authority_error_table = fill(NaN, size(rank_table))
action_error_table = fill(NaN, size(rank_table))
fit_seconds_table = fill(NaN, size(rank_table))
quadrature_error_vector = fill(NaN, length(MODE_COUNTS))
rows = NamedTuple[]

for (mode_index, mode_count) in pairs(MODE_COUNTS)
    groups = grin210_complete_group_count(mode_count)
    println(
        "stage=quadrature_construct",
        " modes=", mode_count,
        " groups=", groups,
        " working_order=", QUADRATURE_ORDER,
        " authority_order=", AUTHORITY_QUADRATURE_ORDER,
    )
    source = grin210_quadrature_source(
        mode_count; quadrature_order=QUADRATURE_ORDER)
    authority = grin210_quadrature_source(
        mode_count; quadrature_order=AUTHORITY_QUADRATURE_ORDER)
    quadrature_error = cp_relative_error(
        source, authority;
        block_size=ERROR_BLOCK_SIZE,
        backend=:cuda,
    )
    quadrature_error <= QUADRATURE_TOLERANCE || error(
        "Quadrature source for $mode_count modes is not converged: " *
        "$quadrature_error > $QUADRATURE_TOLERANCE.")
    quadrature_error_vector[mode_index] = quadrature_error
    source_checksum = cp_decomposition_checksum(source)

    rng = MersenneTwister(SEED + 10_000mode_count)
    action_fields = randn(rng, ComplexF64, RANDOM_ACTIONS, mode_count)
    authority_actions = PulsePropagation.cp_spatial_isotropic_cubic(
        action_fields, authority)
    previous = nothing

    println(
        "stage=quadrature_converged",
        " modes=", mode_count,
        " relative_error=", quadrature_error,
        " source_rank=", cp_rank(source),
        " authority_rank=", cp_rank(authority),
        " source_checksum=", source_checksum,
    )

    for (target_index, target) in pairs(TARGET_ERRORS)
        paths = grin210_cp_cache_paths(OUTPUT_DIR, mode_count, target)
        start_time = time_ns()
        candidate = if isfile(paths.data) && isfile(paths.metadata)
            cached = load_cp_decomposition(paths.data, paths.metadata)
            cached.metadata.source_checksum == source_checksum || error(
                "Cached source checksum does not match $mode_count-mode " *
                "working quadrature: $(paths.data)")
            cached
        elseif isfile(paths.data) || isfile(paths.metadata)
            error("Incomplete cache pair at $(paths.data) / $(paths.metadata).")
        else
            minimum_rank = previous === nothing ?
                           first_rank(mode_count, cp_rank(source)) :
                           cp_rank(previous)
            fitted = cp_compress(
                source;
                error=target,
                backend=:cuda,
                min_rank=minimum_rank,
                rank_step=RANK_STEP,
                max_rank=min(MAX_RANK, cp_rank(source)),
                seed=SEED,
                initial=previous,
                maxiter=MAXITER,
                tolerance=ALS_TOLERANCE,
                ridge=RIDGE,
                check_every=CHECK_EVERY,
                restarts=RESTARTS,
                source_checksum,
                synchronize=true,
                error_block_size=ERROR_BLOCK_SIZE,
                verbose=true,
            )
            save_cp_decomposition(
                paths.data, paths.metadata, fitted; overwrite=false)
            fitted
        end
        CUDA.synchronize()
        elapsed = (time_ns() - start_time) / 1e9

        authority_error = cp_relative_error(
            candidate, authority;
            block_size=ERROR_BLOCK_SIZE,
            backend=:cuda,
        )
        candidate_actions = PulsePropagation.cp_spatial_isotropic_cubic(
            action_fields, candidate)
        action_error = norm(candidate_actions - authority_actions) /
                       norm(authority_actions)
        candidate.relative_error <= target || error(
            "The $mode_count-mode candidate misses working target $target.")
        authority_error <= target + QUADRATURE_TOLERANCE || error(
            "The $mode_count-mode candidate misses target $target against " *
            "the independent authority: $authority_error.")

        rank_table[mode_index, target_index] = cp_rank(candidate)
        reported_error_table[mode_index, target_index] =
            candidate.relative_error
        authority_error_table[mode_index, target_index] = authority_error
        action_error_table[mode_index, target_index] = action_error
        fit_seconds_table[mode_index, target_index] = elapsed
        previous = candidate

        row = (;
            modes=mode_count,
            groups,
            target,
            rank=cp_rank(candidate),
            reported_error=candidate.relative_error,
            authority_error,
            random_action_error=action_error,
            fit_seconds=elapsed,
            quadrature_error,
            source_rank=cp_rank(source),
            authority_rank=cp_rank(authority),
            source_checksum,
            data_path=paths.data,
            metadata_path=paths.metadata,
        )
        push!(rows, row)

        npzwrite(joinpath(OUTPUT_DIR, "mode_counts.npy"), MODE_COUNTS)
        npzwrite(joinpath(OUTPUT_DIR, "target_errors.npy"), TARGET_ERRORS)
        npzwrite(joinpath(OUTPUT_DIR, "ranks.npy"), rank_table)
        npzwrite(joinpath(OUTPUT_DIR, "reported_errors.npy"),
                 reported_error_table)
        npzwrite(joinpath(OUTPUT_DIR, "authority_errors.npy"),
                 authority_error_table)
        npzwrite(joinpath(OUTPUT_DIR, "random_action_errors.npy"),
                 action_error_table)
        npzwrite(joinpath(OUTPUT_DIR, "fit_seconds.npy"), fit_seconds_table)
        npzwrite(joinpath(OUTPUT_DIR, "quadrature_errors.npy"),
                 quadrature_error_vector)
        write_progress(joinpath(OUTPUT_DIR, "cp_summary.csv"), rows)

        println(
            "stage=cp_complete",
            " modes=", mode_count,
            " target=", target,
            " rank=", cp_rank(candidate),
            " reported_error=", candidate.relative_error,
            " authority_error=", authority_error,
            " random_action_error=", action_error,
            " seconds=", elapsed,
            " data_path=", paths.data,
        )
    end

    source = nothing
    authority = nothing
    previous = nothing
    CUDA.reclaim()
end

open(joinpath(OUTPUT_DIR, "run_summary.txt"), "w") do io
    configuration = grin210_configuration()
    for name in propertynames(configuration)
        println(io, name, '=', getproperty(configuration, name))
    end
    println(io, "mode_counts=", join(MODE_COUNTS, ','))
    println(io, "target_errors=", join(TARGET_ERRORS, ','))
    println(io, "quadrature_order=", QUADRATURE_ORDER)
    println(io, "authority_quadrature_order=", AUTHORITY_QUADRATURE_ORDER)
    println(io, "quadrature_tolerance=", QUADRATURE_TOLERANCE)
    println(io, "user_min_rank=", USER_MIN_RANK)
    println(io, "min_rank_factor=", MIN_RANK_FACTOR)
    println(io, "rank_step=", RANK_STEP)
    println(io, "max_rank=", MAX_RANK)
    println(io, "maxiter=", MAXITER)
    println(io, "als_tolerance=", ALS_TOLERANCE)
    println(io, "ridge=", RIDGE)
    println(io, "check_every=", CHECK_EVERY)
    println(io, "restarts=", RESTARTS)
    println(io, "seed=", SEED)
    println(io, "random_actions=", RANDOM_ACTIONS)
    println(io, "gpu=", CUDA.name(CUDA.device()))
end

println("stage=complete output_dir=", OUTPUT_DIR)
