# Checkpointed adjoint tests

The default adjoint remains continuous. Select the fixed-step discrete RK4IP
adjoint explicitly with `adjoint_mode=:discrete_rk4ip`; set
`checkpoint_stride` to control the number of replay steps held at once.
The nominal adjoint and forward step sizes must agree. Adaptive forward
solutions are rejected. Checkpointed replay uses Float64, and rejects a
mixed-precision forward solution. This restriction does not change the
existing mixed-precision forward or continuous-adjoint pathways.

Run the CPU suite with:

```sh
julia --project=. test/runtests.jl
```

In an environment with this package and a functional CUDA installation, run:

```sh
julia --project=. test/cuda_checkpointed_adjoint.jl
```

The CUDA suite checks directional finite differences, CPU/GPU agreement,
checkpoint-stride invariance, continuous-limit convergence, sparse-forward
replay, multimode propagation, and polarization.

`cuda_compatibility.jl` records small calculations using only the preexisting
API, so the same script can run under both an old main checkout and a proposed
integration. Use the same Julia and CUDA dependencies and the same GPU:

```sh
julia --project=/path/to/main /path/to/candidate/test/cuda_compatibility.jl main_gpu.jld2
julia --project=/path/to/candidate /path/to/candidate/test/cuda_compatibility.jl candidate_gpu.jld2
julia --project=/path/to/candidate /path/to/candidate/test/compare_gpu_results.jl main_gpu.jld2 candidate_gpu.jld2 gpu_comparison.toml
```

The comparison requires exact equality of the `results` dictionaries;
timings and source paths are metadata, not
numerical outputs. Both `errors` dictionaries must be empty. The audit records
fields, saved coordinates, integration grids, adaptive counters, and noise
samples. It covers dense and CP backends, both forward precisions, adaptive
and fixed steps, forced GEMM contractions, stochastic ensembles, compression,
and the Cartesian full-field solver. These short runs provide regression
coverage; they do not establish production-scale memory requirements or
long-distance accuracy.
