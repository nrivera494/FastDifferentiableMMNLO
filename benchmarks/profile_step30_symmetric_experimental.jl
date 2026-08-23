const ROOT = normpath(joinpath(@__DIR__, ".."))

get!(ENV, "STEP30_BACKEND", "cuda_cp_symmetric_experimental")
get!(ENV, "STEP30_CP_PATH",
     joinpath(ROOT, "benchmark_cache", "step30_symmetric_rank252.jls"))

include(joinpath(@__DIR__, "profile_step30_forward_controls.jl"))
