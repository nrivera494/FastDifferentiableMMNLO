# ComposedStrang4 step-index H100 validation (2026-08-13)

GPU: NVIDIA H100 80GB HBM3 (SXM)

The full-size pilot uses the physical and numerical transverse system from
`example_notebooks/full_field_vs_cp.ipynb`: `Nt=8192`, `Nx=Ny=64`, a 125 µm
transverse window, the 52.5 µm-radius/NA=0.1 step-index potential, the first
100 eigenvectors of the sampled Fourier-Laplacian-plus-index operator, modes
6:15 in the launch, the 0.249969 ps pulse, Raman, and self-steepening. The pilot
length is 1 mm. Timings exclude compilation and include the final host copy.

The reference is `StrangSplit()` at 1.25 µm. Errors are evaluated at the final
distance. The modal error is the relative Euclidean error between projected
modal spectral intensities. The field error is the phase-aligned relative
error of the complete spatially dependent spectral field.

| method | dz (µm) | time (s) | modal-spectrum error | spectral-field error | nonlinear RHS evaluations |
|---|---:|---:|---:|---:|---:|
| Strang reference | 1.25 | 12.256 | reference | reference | 1600 |
| StrangSplit | 5 | 4.740 | 1.789e-5 | 2.882e-4 | 400 |
| ComposedStrang4 (Suzuki five-stage) | 40 | 4.842 | 8.137e-3 | 7.730e-2 | 500 |
| ComposedStrang4 (Suzuki five-stage) | 20 | 7.963 | 3.219e-4 | 1.500e-2 | 1000 |
| ComposedStrang4 (Suzuki five-stage) | 10 | 14.537 | 2.459e-6 | 6.357e-5 | 2000 |
| ComposedStrang4 (Suzuki five-stage) | 5 | 27.724 | 1.255e-6 | 2.025e-5 | 4000 |

The discrete modal Gram-matrix error was `5.36e-14`. The 10 µm composed run is
more accurate than the 5 µm Strang run by both reported measures, but is about
3.1 times slower. Thus this implementation is useful as an independent
fourth-order accuracy check, but it does not beat the current Strang path on
time to matched accuracy for this pilot.

The selected Suzuki composition was also tested on a smooth self-steepening
problem. Errors at 4, 8, 16, and 32 steps were `1.057e-3`, `4.347e-5`,
`1.717e-6`, and `1.008e-7`; the observed refinement orders were 4.60, 4.66,
and 4.09. The RK4IP reference gap was `1.98e-9`.

For comparison, the preliminary three-stage triple jump at 10 µm took 9.29 s
but had modal-spectrum error `1.271e-4` and spectral-field error `7.116e-4`.
The five-stage Suzuki version is slower per step but much more accurate and has
a smaller-magnitude negative substep, so it is the retained implementation.
