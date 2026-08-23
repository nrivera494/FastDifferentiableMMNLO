using LinearAlgebra
using NPZ
using PulsePropagation
using Serialization

const ROOT = normpath(joinpath(@__DIR__, ".."))

parse_csv(::Type{T}, value) where {T} =
    parse.(T, filter(!isempty, strip.(split(value, ','))))

const MODE_COUNTS = parse_csv(
    Int, get(ENV, "ELLIPSE_MODE_COUNTS", "10,20,30,40,50"))
const TARGET_ERRORS = parse_csv(
    Float64, get(ENV, "ELLIPSE_TARGET_ERRORS", "1e-2,1e-3,1e-4"))
const GRID_SIZE = parse(Int, get(ENV, "ELLIPSE_GRID_SIZE", "161"))
const GRID_HALF_WIDTH_UM = parse(
    Float64, get(ENV, "ELLIPSE_GRID_HALF_WIDTH_UM", "60"))
const BACKEND = Symbol(lowercase(get(ENV, "ELLIPSE_CP_BACKEND", "cuda")))
const MIN_RANK = parse(Int, get(ENV, "ELLIPSE_CP_MIN_RANK", "2"))
const RANK_STEP = parse(Int, get(ENV, "ELLIPSE_CP_RANK_STEP", "25"))
const MAX_RANK = parse(Int, get(ENV, "ELLIPSE_CP_MAX_RANK", "1002"))
const MAXITER = parse(Int, get(ENV, "ELLIPSE_CP_MAXITER", "120"))
const ALS_TOLERANCE = parse(
    Float64, get(ENV, "ELLIPSE_CP_ALS_TOLERANCE", "1e-6"))
const RIDGE = parse(Float64, get(ENV, "ELLIPSE_CP_RIDGE", "1e-8"))
const CHECK_EVERY = parse(Int, get(ENV, "ELLIPSE_CP_CHECK_EVERY", "5"))
const RESTARTS = parse(Int, get(ENV, "ELLIPSE_CP_RESTARTS", "1"))
const SEED = parse(Int, get(ENV, "ELLIPSE_CP_SEED", "11"))
const RANK_BLOCK_SIZE = parse(
    Int, get(ENV, "ELLIPSE_CP_RANK_BLOCK_SIZE", "16"))
const OUTPUT_DIR = get(
    ENV,
    "ELLIPSE_OUTPUT_DIR",
    joinpath(ROOT, "benchmark_cache", "perturbed_ellipse_pilot"),
)

isempty(MODE_COUNTS) && error("ELLIPSE_MODE_COUNTS cannot be empty.")
isempty(TARGET_ERRORS) && error("ELLIPSE_TARGET_ERRORS cannot be empty.")
issorted(MODE_COUNTS) || error("ELLIPSE_MODE_COUNTS must be increasing.")
all(>(0), MODE_COUNTS) || error("Every mode count must be positive.")
all(error -> 0 < error < 1, TARGET_ERRORS) ||
    error("Every target error must lie in (0, 1).")
isodd(GRID_SIZE) || error("ELLIPSE_GRID_SIZE must be odd.")
BACKEND in (:cpu, :cuda) ||
    error("ELLIPSE_CP_BACKEND must be cpu or cuda.")

if BACKEND === :cuda
    @eval using CUDA
    CUDA.functional() || error("CUDA is not functional.")
end

mkpath(OUTPUT_DIR)

const LAMBDA0 = 1550e-9
const NA = 0.275
const AX = 31.25e-6
const AY = 24.0e-6
const DEFECT_STRENGTH = 0.06
const DEFECT_WIDTH = 4.0e-6
const DEFECT_X = 7.0e-6
const DEFECT_Y = -4.0e-6

material = Silica(raman=NoRaman())
ncore = silica_refractive_index(LAMBDA0)
nclad2 = ncore^2 - NA^2
nclad = sqrt(nclad2)

half_width = GRID_HALF_WIDTH_UM * 1e-6
x = collect(range(-half_width, half_width; length=GRID_SIZE))
y = copy(x)
X = reshape(x, 1, length(x))
Y = reshape(y, length(y), 1)
rho2 = @. (X / AX)^2 + (Y / AY)^2
defect = @. exp(-((X - DEFECT_X)^2 + (Y - DEFECT_Y)^2) /
                  (2DEFECT_WIDTH^2))
n2 = @. max(
    nclad2,
    nclad2 + NA^2 * max(1 - rho2, 0) -
    DEFECT_STRENGTH * NA^2 * defect,
)
nxy = sqrt.(n2)
profile = IndexProfile(x, y, nxy; lambda0=LAMBDA0)
index_contrast = n2 .- nclad2
index_contrast_norm = norm(index_contrast)
x_reflection_defect = norm(index_contrast .- reverse(index_contrast; dims=2)) /
                      index_contrast_norm
y_reflection_defect = norm(index_contrast .- reverse(index_contrast; dims=1)) /
                      index_contrast_norm
quarter_turn_defect = norm(index_contrast .- reverse(permutedims(index_contrast); dims=1)) /
                      index_contrast_norm

max_modes = maximum(MODE_COUNTS)
println(
    "stage=construct_overlap",
    " max_modes=", max_modes,
    " grid_size=", GRID_SIZE,
    " backend=", BACKEND,
    " output_dir=", OUTPUT_DIR,
)

overlap_start = time_ns()
properties = compute_fiber_properties(
    profile,
    material,
    max_modes;
    beta_order=0,
    dispersion_samples=3,
    dispersion_fit_order=0,
    overlap_chunk_points=1024,
    overlap_precision=Float64,
    arpack_tol=1e-9,
    arpack_maxiter=10_000,
    tracking_padding=4,
)
Smax = spatial_overlap(properties)
overlap_seconds = (time_ns() - overlap_start) / 1e9

fields = mode_fields(properties)
grid = mode_grid(properties)
dx = grid.x[2] - grid.x[1]
dy = grid.y[2] - grid.y[1]
flat_fields = reshape(fields, :, max_modes)
mode_gram = adjoint(flat_fields) * flat_fields * (dx * dy)
orthogonality_defect = norm(
    mode_gram - Matrix{eltype(mode_gram)}(I, max_modes, max_modes), Inf)
beta0 = propagation_constants(properties)
beta_clad = 2pi / LAMBDA0 * nclad
guided_margin = minimum(beta0 .- beta_clad)
symmetry_defect = PulsePropagation._cp_symmetric_tensor_defect(Smax)

guided_margin > 0 || error(
    "At least one requested mode is not guided: minimum beta-beta_clad = " *
    string(guided_margin))
orthogonality_defect <= 1e-7 || error(
    "Finite-difference modes failed the orthogonality check: defect = " *
    string(orthogonality_defect))
symmetry_defect <= 1e-12 || error(
    "Spatial overlap failed the permutation-symmetry check: defect = " *
    string(symmetry_defect))

npzwrite(joinpath(OUTPUT_DIR, "profile_nxy.npy"), nxy)
npzwrite(joinpath(OUTPUT_DIR, "mode_x_m.npy"), x)
npzwrite(joinpath(OUTPUT_DIR, "mode_y_m.npy"), y)
npzwrite(joinpath(OUTPUT_DIR, "mode_fields.npy"), fields)
npzwrite(joinpath(OUTPUT_DIR, "propagation_constants_m_inv.npy"), beta0)
npzwrite(joinpath(OUTPUT_DIR, "S_$(max_modes)modes.npy"), Smax)

rank_table = fill(-1, length(MODE_COUNTS), length(TARGET_ERRORS))
error_table = fill(NaN, length(MODE_COUNTS), length(TARGET_ERRORS))
seconds_table = fill(NaN, length(MODE_COUNTS), length(TARGET_ERRORS))

function target_tag(target)
    exponent = round(Int, -log10(target))
    isapprox(target, 10.0^(-exponent); rtol=1e-12) && return "1em$(exponent)"
    return replace(string(target), '.' => 'p', '-' => 'm')
end

for (mode_index, modes) in pairs(MODE_COUNTS)
    source_view = @view Smax[1:modes, 1:modes, 1:modes, 1:modes]
    # The CUDA staging path requires contiguous host storage. Materializing
    # these nested slices also avoids scalar-indexing fallback in CUDA.jl.
    source = BACKEND === :cuda ? Array(source_view) : source_view
    for (target_index, target) in pairs(TARGET_ERRORS)
        cp_path = joinpath(
            OUTPUT_DIR,
            "cp_modes$(lpad(modes, 3, '0'))_target$(target_tag(target)).jls",
        )

        fit_start = time_ns()
        cp = if isfile(cp_path)
            cached = open(deserialize, cp_path)
            size(cached.U[1], 1) == modes || error(
                "Cached CP mode count does not match $modes: $cp_path")
            cached.relative_error <= target || error(
                "Cached CP misses target $target: $cp_path")
            cached
        else
            candidate = cp_compress(
                source;
                error=target,
                backend=BACKEND,
                min_rank=MIN_RANK,
                rank_step=RANK_STEP,
                max_rank=MAX_RANK,
                seed=SEED,
                maxiter=MAXITER,
                tolerance=ALS_TOLERANCE,
                ridge=RIDGE,
                check_every=CHECK_EVERY,
                restarts=RESTARTS,
                rank_block_size=RANK_BLOCK_SIZE,
                symmetric_source=true,
            )
            open(cp_path, "w") do io
                serialize(io, candidate)
            end
            candidate
        end
        BACKEND === :cuda && CUDA.synchronize()
        elapsed_seconds = (time_ns() - fit_start) / 1e9

        rank_table[mode_index, target_index] = cp_rank(cp)
        error_table[mode_index, target_index] = cp.relative_error
        seconds_table[mode_index, target_index] = elapsed_seconds

        npzwrite(joinpath(OUTPUT_DIR, "mode_number_list.npy"), MODE_COUNTS)
        npzwrite(joinpath(OUTPUT_DIR, "target_error_list.npy"), TARGET_ERRORS)
        npzwrite(joinpath(OUTPUT_DIR, "cp_rank_table.npy"), rank_table)
        npzwrite(joinpath(OUTPUT_DIR, "cp_error_table.npy"), error_table)
        npzwrite(joinpath(OUTPUT_DIR, "cp_fit_seconds_table.npy"), seconds_table)

        println(
            "stage=cp_fit",
            " modes=", modes,
            " target_error=", target,
            " rank=", cp_rank(cp),
            " relative_error=", cp.relative_error,
            " iterations=", cp.iterations,
            " elapsed_seconds=", elapsed_seconds,
            " cp_path=", cp_path,
        )
    end
end

open(joinpath(OUTPUT_DIR, "run_summary.txt"), "w") do io
    println(io, "profile=perturbed_elliptical_grin")
    println(io, "lambda0_m=", LAMBDA0)
    println(io, "NA=", NA)
    println(io, "ax_m=", AX)
    println(io, "ay_m=", AY)
    println(io, "defect_strength=", DEFECT_STRENGTH)
    println(io, "defect_width_m=", DEFECT_WIDTH)
    println(io, "defect_x_m=", DEFECT_X)
    println(io, "defect_y_m=", DEFECT_Y)
    println(io, "grid_size=", GRID_SIZE)
    println(io, "grid_half_width_um=", GRID_HALF_WIDTH_UM)
    println(io, "mode_counts=", join(MODE_COUNTS, ','))
    println(io, "target_errors=", join(TARGET_ERRORS, ','))
    println(io, "backend=", BACKEND)
    println(io, "min_rank=", MIN_RANK)
    println(io, "rank_step=", RANK_STEP)
    println(io, "max_rank=", MAX_RANK)
    println(io, "maxiter=", MAXITER)
    println(io, "als_tolerance=", ALS_TOLERANCE)
    println(io, "ridge=", RIDGE)
    println(io, "check_every=", CHECK_EVERY)
    println(io, "restarts=", RESTARTS)
    println(io, "seed=", SEED)
    println(io, "rank_block_size=", RANK_BLOCK_SIZE)
    println(io, "overlap_seconds=", overlap_seconds)
    println(io, "orthogonality_defect_inf=", orthogonality_defect)
    println(io, "permutation_symmetry_defect=", symmetry_defect)
    println(io, "minimum_beta_minus_beta_clad_m_inv=", guided_margin)
    println(io, "x_reflection_index_defect=", x_reflection_defect)
    println(io, "y_reflection_index_defect=", y_reflection_defect)
    println(io, "quarter_turn_index_defect=", quarter_turn_defect)
    if BACKEND === :cuda
        println(io, "cuda_device=", CUDA.name(CUDA.device()))
    end
end

println(
    "stage=complete",
    " overlap_seconds=", overlap_seconds,
    " orthogonality_defect=", orthogonality_defect,
    " symmetry_defect=", symmetry_defect,
    " guided_margin_m_inv=", guided_margin,
    " x_reflection_index_defect=", x_reflection_defect,
    " y_reflection_index_defect=", y_reflection_defect,
    " quarter_turn_index_defect=", quarter_turn_defect,
    " output_dir=", OUTPUT_DIR,
)
