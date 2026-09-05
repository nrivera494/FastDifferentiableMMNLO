#!/bin/bash
set -euo pipefail
AUDIT=/tmp/mmgnlse_compat_20260905
mkdir -p "$AUDIT/main" "$AUDIT/candidate" "$AUDIT/results"
tar --no-same-owner -xf /workspace/mmgnlse_compat_20260905/main_runtime.tar -C "$AUDIT/main"
tar --no-same-owner -xf /workspace/mmgnlse_compat_20260905/candidate_runtime.tar -C "$AUDIT/candidate"
export JULIA_DEPOT_PATH="$AUDIT/depot"
export JULIA_NUM_PRECOMPILE_TASKS=4
export JULIA_PKG_PRECOMPILE_AUTO=0
export PYTHON=/usr/bin/python3
"$AUDIT/julia-1.12.5/bin/julia" --project="$AUDIT/main" --startup-file=no -e 'using Pkg; Pkg.instantiate(;allow_autoprecomp=false); Pkg.add(PackageSpec(name="CUDA",version="6")); Pkg.precompile()'
cp "$AUDIT/main/Project.toml" "$AUDIT/candidate/Project.toml"
cp "$AUDIT/main/Manifest.toml" "$AUDIT/candidate/Manifest.toml"
"$AUDIT/julia-1.12.5/bin/julia" --project="$AUDIT/main" --startup-file=no -e 'using CUDA; CUDA.versioninfo(); @assert CUDA.functional()'
