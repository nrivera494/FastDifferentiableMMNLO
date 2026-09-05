# Main to checkpointed-adjoint compatibility audit

Date: 2026-09-05. No repository source changes, commits, merges, or branch switches were made.

## Compared revisions

- main: d39bf98eac832533644dbbde650f0fd933579afe
- codex/checkpointed-cuda-adjoint: 88086319063eff09351159330b8db6ee93f0a1d9
- The original project remains on codex/six-reproduction-examples at 040dee79cd14ea54b815056ec5e290c78485d6f2 with its preexisting uncommitted work.

The checkpointed branch contains four commits beyond main:

1. 9579421: optional self-steepening controls, cherry-picked from 5e29fad on the reproduction branch.
2. d5670ad: terminal-only stochastic ensemble storage, cherry-picked from 4d41444 on the reproduction branch.
3. 0810a41: opt-in Wigner Raman normalization, cherry-picked from ac38834 on the reproduction branch.
4. 8808631: checkpointed discrete RK4IP adjoint.

It contains none of the reproduction notebook additions, but it DOES contain three shared changes originating on that branch.

## Passing evidence

- All tracked original example notebooks are byte-for-byte unchanged between the compared revisions.
- Project.toml and Manifest.toml are unchanged.
- Existing CUDA forward implementation, nonlinear VJP implementation, full-field implementation, and MMGNLSE autodiff implementation are unchanged.
- Default adjoint mode remains continuous. Discrete RK4IP checkpointing is opt-in.
- Candidate CPU test suite: 85 / 85 assertions passed.
- An isolated main source snapshot and the candidate were executed under the same Julia 1.12.5 environment, BLAS threads=1, FFTW threads=1, and fixed RNG seeds.
- 15 physical/layout combinations: Kerr, Agarwal Raman, anisotropic Raman; scalar dense, scalar CP, multimode dense, multimode CP, and two-mode/two-polarization dense.
- These exercised RK4IP and Tsit5 forward propagation; continuous adjoints from full and sparse forward storage; photon-unit gradients; photon observables and terminal conditions; initial-shot-noise stochastic ensembles; nonlinear prefactors; saved z grids.
- All 195 notebook-facing numerical outputs compared exactly, not merely within a tolerance.
- Across the full comparison there were 226 exact outputs out of 234 and zero harness case failures. Six changed field arrays belong to the older distributed-Raman API; two changed strings are parameter/solution checksums.
- A main-generated archive written with save_mmgnlse_solution loads with the expected old checksum on the candidate; stored fields and a replayed calculation match exactly.

## Compatibility failures

### 1. Older distributed-Raman API changes behavior

src/backend/stochastic.jl changes include_self_steepening's default from false to true. On main, the deterministic nonlinear term always includes self-steepening; the flag only controls the stochastic kick. On the candidate, the flag controls both terms.

Consequences verified numerically with identical random seeds:

- Default, nonzero Raman noise: different field arrays for both ODE and RK4IP deterministic steppers. Relative array error in the small test was approximately 3.73e-8. This is a deliberate equation/default difference, not random-seed variability.
- Explicit include_self_steepening=false: different deterministic and stochastic field arrays. Relative array error was approximately 1.61e-5 even with noise_scale=0.
- Therefore setting the new flag to false does not restore the old combination of deterministic shock-on and stochastic shock-off.
- Raman spectrum normalization itself is identical under the default legacy convention.

### 2. Raw JLD2 object compatibility is broken

The added self_steepening field changes the MMGNLSEParameters structure. A raw JLD2 archive written by main containing forward and stochastic solution objects reloads on the candidate with a JLD2.ReconstructedMutable parameter object, not an MMGNLSEParameters instance. Array data remain readable, but passing the restored parameters to solve_mmgnlse raises a MethodError.

This differs from the dedicated save_mmgnlse_solution/load_mmgnlse_solution format, whose backward compatibility passed. The original GPU stochastic notebook uses raw JLD2.jldsave for its ensemble, making this relevant to the existing workflow.

Parameter and solution checksums also change when otherwise identical parameters gain the default true field; old portable-archive checksum verification is explicitly supported and passed.

## Limits

These were representative small CPU numerical tests, not full executions of every original notebook at its production resolution. CUDA calculations were not executed on this Mac. Unchanged CUDA forward source and unchanged default prefactors are supporting static evidence, not GPU runtime validation. No claim of universal bitwise equivalence is made.

## Recommendation

Do not merge the entire checkpointed branch under an identical-behavior requirement.

Import only the checkpointed-adjoint implementation in commit 8808631 and its tests onto main, leaving the three earlier optional-feature changes and the reproduction branch separate. The source-only patch for its two implementation files passes git apply --check against the main snapshot. Its tests would need their redundant self_steepening=true keyword removed and a standalone test entry point, because main predates those additions. This narrower variant has not been applied or numerically validated in this audit.

After authorization of that narrower scope: construct and validate the restricted integration, preserve the current dirty reproduction work separately, update main locally, and switch the original project checkout to main. Do not discard existing untracked or modified files.

## Reproducible artifacts

- compare_paths.jl: temporary comparison harness
- main_results.jld2 / checkpointed_results.jld2: numeric outputs
- summarize.jl: comparison and archive-loading checks
- audit_results.toml: machine-readable results
- main_archive.jld2 and main_raw_archive.jld2: portable/raw compatibility fixtures
- checkpoint_only.patch: unapplied source-only patch; application checked against main
- main/: isolated main source and environment snapshot

All artifacts are under /private/tmp/mmgnlse-main-compatibility-20260905.
