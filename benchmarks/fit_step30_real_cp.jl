using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation
using Random
using Serialization

CUDA.functional() || error("CUDA is not functional.")

const ROOT = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR = get(
    ENV,
    "STEP30_DATA_DIR",
    joinpath(ROOT, "example_notebooks", "benchmark_data",
             "step_index_soliton"),
)
const RANK = parse(Int, get(ENV, "CP_RANK", "252"))
const MAXITER = parse(Int, get(ENV, "MAXITER", "75"))
const CHECK_EVERY = parse(Int, get(ENV, "CHECK_EVERY", "5"))
const SEED = parse(Int, get(ENV, "CP_SEED", "1297303374"))
const OUTPUT_PATH = get(
    ENV,
    "STEP30_REAL_CP_PATH",
    joinpath(ROOT, "benchmark_cache", "step30_real_rank$(RANK).jls"),
)

source40 = npzread(joinpath(DATA_DIR, "S_40modes.npy"))
source = source40[1:30, 1:30, 1:30, 1:30]
rng = MersenneTwister(SEED)

CUDA.synchronize()
start = time_ns()
legacy_cp, history = PulsePropagation.cp_als_warm_cuda(
    source,
    RANK;
    maxiter=MAXITER,
    tol=1e-7,
    ridge=1e-8,
    check_every=CHECK_EVERY,
    rng,
    verbose=true,
)
CUDA.synchronize()
elapsed_seconds = (time_ns() - start) / 1e9

relative_error = isempty(history) ? NaN : last(history)
cp = MMGNLSECPDecomposition(
    legacy_cp.λ,
    legacy_cp.U;
    layout=:spatial,
    nmodes=30,
    npolarizations=1,
    relative_error,
    seed=SEED,
    iterations=MAXITER,
    converged=false,
)

mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    serialize(io, cp)
end

println(
    "gpu=", CUDA.name(CUDA.device()),
    " rank=", cp_rank(cp),
    " relative_error=", cp.relative_error,
    " history=", join(history, ','),
    " elapsed_seconds=", elapsed_seconds,
    " output=", OUTPUT_PATH,
)
