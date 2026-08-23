# CUDA backend validation — A100 80 GB PCIe — 2026-07-28

These measurements were collected on the RunPod A100 used for CUDA solver
development. Timings use `CUDA.synchronize()` before and after every sample.
No timing below includes Julia compilation.

## Backend mapping

- `backend=:cuda` with a dense overlap: optimized dense CUDA implementation.
- `backend=:cuda` with a CP overlap: baseline CUDA CP implementation.
- `backend=:cuda_cp_optimized`: rank-agnostic optimized CUDA CP implementation;
  dense overlaps are rejected.
- `backend=:cuda_optimized`: rank-tuned optimized CUDA CP implementation. With a
  dense overlap it uses the same optimized dense kernels as `backend=:cuda`.

## Dense nonlinear RHS

Command:

```sh
NSAMPLES=7 NWARMUP=3 LABEL=final_backend_cuda_dense \
  julia --project=/root/bench_env benchmarks/profile_dense_cuda_rhs.jl
```

Problem size: 8192 time samples and 40 spatial modes.

| Measurement | Median |
|---|---:|
| Serialized Kerr | 117.436 ms |
| Pair/physical Kerr | 62.744 ms |
| Raman standalone | 63.919 ms |
| Complete nonlinear RHS | **64.170 ms** |

The complete RHS result reproduces the previously observed 64.72 ms optimized
dense CUDA pathway within ordinary run-to-run variation.

## CP solver backend comparison

Command:

```sh
RANKS=50,100,200,484 NSAMPLES=5 NWARMUP=2 \
  julia --project=/root/bench_env benchmarks/profile_cp_cuda_backends.jl
```

Problem size: 8192 time samples and 40 spatial modes. Entries are median
milliseconds.

| Rank | Backend | Forward RHS | RK4IP step | Adjoint VJP |
|---:|---|---:|---:|---:|
| 50 | `:cuda` | 0.480 | 1.856 | 0.815 |
| 50 | `:cuda_cp_optimized` | 0.327 | 1.239 | 0.577 |
| 50 | `:cuda_optimized` | 0.274 | 1.050 | 0.411 |
| 100 | `:cuda` | 0.480 | 1.953 | 0.964 |
| 100 | `:cuda_cp_optimized` | 0.397 | 1.672 | 0.717 |
| 100 | `:cuda_optimized` | 0.372 | 1.515 | 0.751 |
| 200 | `:cuda` | 0.784 | 2.986 | 1.631 |
| 200 | `:cuda_cp_optimized` | 0.668 | 2.546 | 1.223 |
| 200 | `:cuda_optimized` | 0.679 | 2.542 | 1.220 |
| 484 | `:cuda` | 1.663 | 6.256 | 3.455 |
| 484 | `:cuda_cp_optimized` | 1.392 | 5.440 | 2.672 |
| 484 | `:cuda_optimized` | 1.394 | 5.409 | 2.595 |

## Correctness and regression checks

- `benchmarks/validate_cp_cuda_backends.jl`: 25/25 assertions passed.
  This checks all three CP routes against CPU for forward and adjoint solves,
  verifies cache backend tags, checks dense CUDA agreement with CPU, confirms
  the optimized dense spatial/Raman workspace, and verifies that
  `:cuda_cp_optimized` rejects a dense overlap.
- Existing optional CUDA MMGNLSE API suite: 153/153 tests passed.

