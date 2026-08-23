using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation
using Random

include(joinpath(@__DIR__, "eslami_190_modal_common.jl"))

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

parse_csv(::Type{T}, value) where {T} =
    parse.(T, filter(!isempty, strip.(split(value, ','))))

const TARGET_ERRORS = parse_csv(
    Float64, get(ENV, "ESLAMI_CP_TARGET_ERRORS", "1e-2,1e-3,1e-4"))
const QUADRATURE_ORDER = parse(
    Int, get(ENV, "ESLAMI_CP_QUADRATURE_ORDER", "48"))
const AUTHORITY_QUADRATURE_ORDER = parse(
    Int, get(ENV, "ESLAMI_CP_AUTHORITY_ORDER", "64"))
const QUADRATURE_TOLERANCE = parse(
    Float64, get(ENV, "ESLAMI_CP_QUADRATURE_TOLERANCE", "1e-6"))
const MIN_RANK = parse(Int, get(ENV, "ESLAMI_CP_MIN_RANK", "512"))
const RANK_STEP = parse(Int, get(ENV, "ESLAMI_CP_RANK_STEP", "64"))
const MAX_RANK = parse(Int, get(ENV, "ESLAMI_CP_MAX_RANK", "2304"))
const MAXITER = parse(Int, get(ENV, "ESLAMI_CP_MAXITER", "150"))
const ALS_TOLERANCE = parse(
    Float64, get(ENV, "ESLAMI_CP_ALS_TOLERANCE", "1e-8"))
const RIDGE = parse(Float64, get(ENV, "ESLAMI_CP_RIDGE", "1e-10"))
const CHECK_EVERY = parse(Int, get(ENV, "ESLAMI_CP_CHECK_EVERY", "5"))
const RESTARTS = parse(Int, get(ENV, "ESLAMI_CP_RESTARTS", "1"))
const ERROR_BLOCK_SIZE = parse(
    Int, get(ENV, "ESLAMI_CP_ERROR_BLOCK_SIZE", "256"))
const SEED = parse(Int, get(ENV, "ESLAMI_CP_SEED", "20220819"))
const RANDOM_ACTIONS = parse(
    Int, get(ENV, "ESLAMI_CP_RANDOM_ACTIONS", "8"))
const OUTPUT_DIR = get(
    ENV,
    "ESLAMI_CP_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "eslami_190_cp"),
)

TARGET_ERRORS == sort(TARGET_ERRORS; rev=true) || error(
    "ESLAMI_CP_TARGET_ERRORS must be ordered from loose to tight.")
all(target -> 0 < target < 1, TARGET_ERRORS) || error(
    "Every CP target must lie in (0, 1).")
AUTHORITY_QUADRATURE_ORDER > QUADRATURE_ORDER || error(
    "The authority quadrature order must exceed the working order.")
mkpath(OUTPUT_DIR)

fiber = eslami_modal_fiber()
println(
    "stage=quadrature_construct",
    " modes=", ESLAMI_MODE_COUNT,
    " working_order=", QUADRATURE_ORDER,
    " authority_order=", AUTHORITY_QUADRATURE_ORDER,
)
source = grin_hg_overlap_quadrature_cp(
    fiber, ESLAMI_MODE_COUNT; quadrature_order=QUADRATURE_ORDER)
authority = grin_hg_overlap_quadrature_cp(
    fiber, ESLAMI_MODE_COUNT;
    quadrature_order=AUTHORITY_QUADRATURE_ORDER)
quadrature_error = cp_relative_error(
    source, authority; block_size=ERROR_BLOCK_SIZE, backend=:cuda)
quadrature_error <= QUADRATURE_TOLERANCE || error(
    "Quadrature source is not converged: relative discrepancy " *
    "$quadrature_error exceeds $QUADRATURE_TOLERANCE.")
source_checksum = cp_decomposition_checksum(source)
println(
    "stage=quadrature_converged",
    " source_rank=", cp_rank(source),
    " authority_rank=", cp_rank(authority),
    " relative_error=", quadrature_error,
    " source_checksum=", source_checksum,
)

rng = MersenneTwister(SEED + 1)
action_fields = randn(
    rng, ComplexF64, RANDOM_ACTIONS, ESLAMI_MODE_COUNT)
authority_actions = PulsePropagation.cp_spatial_isotropic_cubic(
    action_fields, authority)

ranks = fill(-1, length(TARGET_ERRORS))
reported_errors = fill(NaN, length(TARGET_ERRORS))
authority_errors = fill(NaN, length(TARGET_ERRORS))
action_errors = fill(NaN, length(TARGET_ERRORS))
fit_seconds = fill(NaN, length(TARGET_ERRORS))
previous = nothing

for (target_index, target) in pairs(TARGET_ERRORS)
    global previous
    tag = eslami_target_tag(target)
    data_path = joinpath(OUTPUT_DIR, "eslami_190_cp_$(tag).jld2")
    metadata_path = joinpath(OUTPUT_DIR, "eslami_190_cp_$(tag).toml")
    start_time = time_ns()
    candidate = if isfile(data_path) && isfile(metadata_path)
        cached = load_cp_decomposition(data_path, metadata_path)
        cached.metadata.source_checksum == source_checksum || error(
            "Cached CP source checksum does not match the working quadrature.")
        cached
    else
        fitted = cp_compress(
            source;
            error=target,
            backend=:cuda,
            min_rank=previous === nothing ? MIN_RANK : cp_rank(previous),
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
            data_path, metadata_path, fitted; overwrite=false)
        fitted
    end
    elapsed = (time_ns() - start_time) / 1e9

    authority_error = cp_relative_error(
        candidate, authority; block_size=ERROR_BLOCK_SIZE, backend=:cuda)
    candidate_actions = PulsePropagation.cp_spatial_isotropic_cubic(
        action_fields, candidate)
    action_error = norm(candidate_actions - authority_actions) /
                   norm(authority_actions)
    candidate.relative_error <= target || error(
        "Candidate misses its working-source target $target.")
    authority_error <= target + QUADRATURE_TOLERANCE || error(
        "Candidate misses target against independent quadrature authority: " *
        "$authority_error > $(target + QUADRATURE_TOLERANCE).")

    ranks[target_index] = cp_rank(candidate)
    reported_errors[target_index] = candidate.relative_error
    authority_errors[target_index] = authority_error
    action_errors[target_index] = action_error
    fit_seconds[target_index] = elapsed
    previous = candidate

    npzwrite(joinpath(OUTPUT_DIR, "target_errors.npy"), TARGET_ERRORS)
    npzwrite(joinpath(OUTPUT_DIR, "ranks.npy"), ranks)
    npzwrite(joinpath(OUTPUT_DIR, "reported_errors.npy"), reported_errors)
    npzwrite(joinpath(OUTPUT_DIR, "authority_errors.npy"), authority_errors)
    npzwrite(joinpath(OUTPUT_DIR, "random_action_errors.npy"), action_errors)
    npzwrite(joinpath(OUTPUT_DIR, "fit_seconds.npy"), fit_seconds)
    println(
        "stage=cp_complete",
        " target=", target,
        " rank=", cp_rank(candidate),
        " reported_error=", candidate.relative_error,
        " authority_error=", authority_error,
        " random_action_error=", action_error,
        " seconds=", elapsed,
        " data_path=", data_path,
    )
end

open(joinpath(OUTPUT_DIR, "run_summary.txt"), "w") do io
    println(io, "mode_count=", ESLAMI_MODE_COUNT)
    println(io, "maximum_hg_order=", ESLAMI_MAXIMUM_HG_ORDER)
    println(io, "quadrature_order=", QUADRATURE_ORDER)
    println(io, "authority_quadrature_order=", AUTHORITY_QUADRATURE_ORDER)
    println(io, "quadrature_relative_error=", quadrature_error)
    println(io, "source_rank=", cp_rank(source))
    println(io, "authority_rank=", cp_rank(authority))
    println(io, "source_checksum=", source_checksum)
    println(io, "targets=", join(TARGET_ERRORS, ','))
    println(io, "ranks=", join(ranks, ','))
    println(io, "reported_errors=", join(reported_errors, ','))
    println(io, "authority_errors=", join(authority_errors, ','))
    println(io, "random_action_errors=", join(action_errors, ','))
    println(io, "fit_seconds=", join(fit_seconds, ','))
    println(io, "gpu=", CUDA.name(CUDA.device()))
end

println("stage=complete output_dir=", OUTPUT_DIR)
