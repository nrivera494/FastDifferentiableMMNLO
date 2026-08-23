using CUDA
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
const INITIAL_PATH = get(
    ENV,
    "STEP30_REAL_CP_PATH",
    joinpath(ROOT, "benchmark_cache", "step30_real_rank252.jls"),
)
const RANK = parse(Int, get(ENV, "CP_RANK", "252"))
const MAXITER = parse(Int, get(ENV, "MAXITER", "500"))
const CHECK_EVERY = parse(Int, get(ENV, "CHECK_EVERY", "10"))
const LEARNING_RATE = parse(Float64, get(ENV, "LEARNING_RATE", "0.02"))
const RIDGE = parse(Float64, get(ENV, "RIDGE", "1e-8"))
const TOLERANCE = parse(Float64, get(ENV, "TOLERANCE", "0"))
const SEED = parse(Int, get(ENV, "CP_SEED", "1297303374"))
const OUTPUT_PATH = get(
    ENV,
    "STEP30_SYMMETRIC_CP_PATH",
    joinpath(ROOT, "benchmark_cache", "step30_symmetric_rank$(RANK).jls"),
)

source40 = npzread(joinpath(DATA_DIR, "S_40modes.npy"))
source = source40[1:30, 1:30, 1:30, 1:30]
initial = isfile(INITIAL_PATH) ? deserialize(INITIAL_PATH) : nothing

CUDA.synchronize()
start = time_ns()
cp = cp_compress_symmetric_experimental(
    source,
    RANK;
    initial,
    backend=:cuda,
    maxiter=MAXITER,
    check_every=CHECK_EVERY,
    learning_rate=LEARNING_RATE,
    ridge=RIDGE,
    tolerance=TOLERANCE,
    seed=SEED,
    layout=:spatial,
    nmodes=30,
    npolarizations=1,
    verbose=true,
)
CUDA.synchronize()
elapsed_seconds = (time_ns() - start) / 1e9

mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    serialize(io, cp)
end
println(
    "gpu=", CUDA.name(CUDA.device()),
    " rank=", cp_rank(cp),
    " relative_error=", cp.relative_error,
    " iterations=", cp.iterations,
    " converged=", cp.converged,
    " symmetric=", cp_is_symmetric(cp),
    " elapsed_seconds=", elapsed_seconds,
    " output=", OUTPUT_PATH,
)
