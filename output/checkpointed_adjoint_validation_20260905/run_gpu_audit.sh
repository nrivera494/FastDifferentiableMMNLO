#!/bin/bash
set -euo pipefail
AUDIT=/tmp/mmgnlse_compat_20260905
export JULIA_DEPOT_PATH="$AUDIT/depot"
export JULIA_NUM_PRECOMPILE_TASKS=4
export OPENBLAS_NUM_THREADS=1
JULIA="$AUDIT/julia-1.12.5/bin/julia"
"$JULIA" --project="$AUDIT/main" --startup-file=no "$AUDIT/candidate/test/cuda_compatibility.jl" "$AUDIT/results/main_gpu.jld2" > "$AUDIT/results/main_gpu.log" 2>&1
"$JULIA" --project="$AUDIT/candidate" --startup-file=no "$AUDIT/candidate/test/cuda_compatibility.jl" "$AUDIT/results/candidate_gpu.jld2" > "$AUDIT/results/candidate_gpu.log" 2>&1
"$JULIA" --project="$AUDIT/candidate" --startup-file=no "$AUDIT/candidate/test/cuda_checkpointed_adjoint.jl" > "$AUDIT/results/checkpointed_gpu.log" 2>&1
"$JULIA" --project="$AUDIT/candidate" --startup-file=no "$AUDIT/compare_gpu_results.jl" "$AUDIT/results/main_gpu.jld2" "$AUDIT/results/candidate_gpu.jld2" "$AUDIT/results/gpu_comparison.toml" > "$AUDIT/results/comparison.log" 2>&1
