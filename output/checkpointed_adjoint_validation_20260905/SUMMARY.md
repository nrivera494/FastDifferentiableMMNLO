# Checkpointed-adjoint integration and validation

Completed 2026-09-05. The primary project is on local `main` at
`7c7f0e4bbc4883567c94ee1f8a8be18310272925`, one commit ahead of `origin/main`.
No push was performed.

## What was integrated

The checkpointed discrete RK4IP adjoint from `8808631` was integrated directly
onto original main `d39bf98eac832533644dbbde650f0fd933579afe`, with associated
CPU/CUDA tests. Only two production files changed:
`src/mmgnlse_adjoint_stochastic.jl` and `ext/mmgnlse_cuda_adjoint.jl`.

The default adjoint remains continuous. Checkpointed replay is selected with
`adjoint_mode=:discrete_rk4ip` and `checkpoint_stride`. It currently requires
fixed-step RK4IP in Float64, with matching nominal forward/adjoint step sizes.
A guard rejects mixed-precision forward solutions in the new mode because
Float64 replay would otherwise differentiate a different numerical map.
Existing mixed-precision forward and continuous-adjoint behavior is unchanged.

The self-steepening controls, terminal-only stochastic storage, and Wigner Raman
normalization commits were excluded. No reproduction notebook changes were
integrated. Original notebooks, parameter schema, stochastic implementations,
forward solver files, Project.toml, and Manifest.toml are unchanged from old main.

## Results

| Check | Result |
| --- | --- |
| Existing CPU behavior versus old main | 234/234 recorded outputs exactly equal; zero errors |
| Old portable and raw JLD2 archives | Reload and replay successfully, with equal fields and preserved checksums |
| CPU checkpointed-adjoint suite | 20/20 assertions passed |
| Existing GPU behavior versus old main | 590/590 recorded outputs exactly equal; zero differences |
| GPU compatibility harness | 206/206 assertions passed on each revision |
| CUDA checkpointed-adjoint suite | 49/49 assertions passed |
| Tested remote source versus committed source | SHA-256 hashes match |
| Primary-checkout Julia import | Loads `/Users/nr462/Desktop/FastDifferentiableMMNLO/src/PulsePropagation.jl` |

GPU environment: NVIDIA H100 80GB HBM3, driver 580.159.04, Julia 1.12.5,
CUDA.jl 6.3.1. Both revisions used the same GPU and dependency environment.
The GPU-only dependency setup was confined to the pod; local dependencies
were not modified.

Small GPU calculations cover dense and CP backends, baseline and optimized
paths, multimode fields and polarization, Kerr/isotropic/anisotropic Raman,
Float64 and mixed precision, fixed and adaptive steps, deliberate rejected
steps, continuous adjoints, seeded stochastic ensembles, forced GEMM
contractions, odd tile boundaries, CP compression, and full-field propagation.
New adjoints were checked against directional finite differences and CPU
results, across checkpoint strides and sparse forward storage, and for
convergence toward the continuous adjoint as the step decreases.

An initial diagnostic on unchanged main found preexisting CPU/GPU continuous
adjoint differences up to 8.89e-8 relative. The diagnostic smoke bound was
therefore set to 1e-6, and the error values were included in the strict
cross-revision comparison. All 590 before/after GPU outputs, including those
diagnostics, matched exactly. The initial log is retained for transparency.

These are representative short regression calculations, not full-scale
executions of every notebook. They establish equivalence for the tested
paths; they do not prove universal bitwise equivalence, production-scale GPU
memory requirements, or long-distance physical accuracy.

## Preserved work and current checkout

The primary checkout remains at:
`/Users/nr462/Desktop/FastDifferentiableMMNLO` on `main`.
Its tracked working tree is clean. Preexisting untracked benchmark arrays,
manuscript files, and presentation remain there, along with this audit folder.

The reproduction branch remains at `040dee79cd14ea54b815056ec5e290c78485d6f2`
in a separate persistent worktree:
`/Users/nr462/Desktop/FastDifferentiableMMNLO-six-reproduction-examples`.
All six originally modified tracked files were restored there and verified
by SHA-256, including the four edited reproduction notebooks. The untracked
Corney plan and reproduction README were moved there too. All 66 original
untracked files were verified present with their original sizes and mtimes.

An additional backup of the tracked edits remains in Git stash
`2a62f4bf7a9b8d7a29735cd3786de6bb952506a6`.
The existing `codex/checkpointed-cuda-adjoint` branch was left intact.

Restart any already-running Julia/Jupyter kernel before continuing in main,
so it loads the current source instead of retaining modules from the previous
branch. New Julia sessions using this project have been verified to load it.

## Saved evidence and RunPod shutdown

All GPU results, logs, and exact-comparison reports are in `gpu_results/`.
The remote Project.toml, Manifest.toml, setup log, source/result checksums,
and version records are in `gpu_environment/`. Downloaded files were verified
against the pod's SHA-256 checksums before shutdown.

CPU numeric results, archive fixtures, comparison scripts, original main
runtime snapshot, initial full-branch audit, and the final integration patch
are saved alongside this report. `final_preservation_and_git_checks.json`
records preservation and branch-ancestry checks. `AUDIT.md` is the historical
audit of the entire checkpointed branch, before the narrower integration;
this report describes the completed outcome.

RunPod `helpful_orange_woodpecker` (`84vnl6ydpgu3y6`) was stopped after all
downloads. The console confirmed Compute: Not running and displayed its
Start control. The pod was not terminated or deleted.
