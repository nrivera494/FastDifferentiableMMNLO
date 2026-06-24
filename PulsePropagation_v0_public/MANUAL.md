# PulsePropagation v0 Public Notes

This public v0 package is self-contained: it does not depend on the private
development package or legacy fiber loaders. Multimode examples use CSV fiber
assets bundled with this folder.

## Capability Matrix

| Regime | Forward | Adjoint | AD | Stochastic | CP | GPU |
|---|---:|---:|---:|---:|---:|---:|
| Single-mode scalar | yes | yes | yes | initial, Raman, linear-gain noise | n/a | planned |
| Multimode scalar | yes | yes | yes | initial noise | CPU forward/adjoint/ALS | planned |
| Multimode vector | yes | yes | yes | initial noise | not in v0 | planned |

All current CPU solvers use interaction-picture propagation with fixed-step or
adaptive Tsit5 forward solves and Vern9 adjoint solves. Fixed-step cached RK4IP
is available for scalar CP propagation.

## Data

Public fiber assets are stored as CSV:

- `GRIN_62.5um_wavelength1550nm_csv`
- `step_wavelength1550nm_10mode_csv`

Use `load_fiber_system(folder; modes, length, lambda0, ...)` to construct a
`PassiveFiber` and modal basis from these assets.

## Examples

The retained public notebooks are:

- `example_notebooks/squeezing_spectral_filter.ipynb`
- `example_notebooks/birefringent_mm_soliton.ipynb`
- `example_notebooks/step_index_soliton_10mode_cached_cp.ipynb`

The 10-mode CP notebook retains `.npy` benchmark arrays so CPU CP validity can
be compared against reference data.
