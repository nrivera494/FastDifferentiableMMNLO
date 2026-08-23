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
const OUTPUT_PATH = get(
    ENV,
    "STEP30_CP_PATH",
    joinpath(ROOT, "benchmark_cache", "step30_cp_1e-4.jls"),
)

if isfile(OUTPUT_PATH)
    cp = deserialize(OUTPUT_PATH)
    println(
        "existing_cp_path=", OUTPUT_PATH,
        " rank=", cp_rank(cp),
        " relative_error=", cp.relative_error,
        " iterations=", cp.iterations,
    )
    exit()
end

S40 = npzread(joinpath(DATA_DIR, "S_40modes.npy"))
S30 = S40[1:30, 1:30, 1:30, 1:30]
CUDA.synchronize()
start = time_ns()
cp = cp_compress(
    S30;
    error=1e-4,
    backend=:cuda,
    min_rank=2,
    rank_step=50,
    max_rank=1000,
    maxiter=120,
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)
CUDA.synchronize()
elapsed_seconds = (time_ns() - start) / 1e9
mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    serialize(io, cp)
end
println(
    "cp_path=", OUTPUT_PATH,
    " gpu=", CUDA.name(CUDA.device()),
    " rank=", cp_rank(cp),
    " relative_error=", cp.relative_error,
    " iterations=", cp.iterations,
    " elapsed_seconds=", elapsed_seconds,
)
