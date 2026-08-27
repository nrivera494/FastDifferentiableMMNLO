---
layout: default
title: FastDifferentiableMMNLO.jl
description: Fast and differentiable simulations of ultrafast multimode nonlinear dynamics
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']]
  }
};
</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

<style>
table {
  display: block;
  overflow-x: auto;
  white-space: nowrap;
}
th, td {
  text-align: center;
  vertical-align: middle;
}
th:first-child, td:first-child {
  position: sticky;
  left: 0;
  text-align: left;
  background: #fff;
}
.supported {
  color: #0969da;
  font-weight: 700;
}
</style>

# FastDifferentiableMMNLO.jl

## A package for fast and differentiable simulations of ultrafast multimode nonlinear dynamics

**Nicholas Rivera**  
School of Applied and Engineering Physics, Cornell University, Ithaca, NY 14853

[View the source code](https://github.com/nrivera494/FastDifferentiableMMNLO) · [Browse the example notebooks](https://github.com/nrivera494/FastDifferentiableMMNLO/tree/main/example_notebooks)

## What it is

FastDifferentiableMMNLO.jl features routines for solving ultrafast pulse propagation in the presence of dispersion, diffraction, Kerr nonlinearity, Raman nonlinearity, and self-steepening. The equations that are solved are:

1. The modal representation, called the multimode generalized nonlinear Schrödinger equation (MMGNLSE), written as

   $$
   \partial_z A_m = \sum_{k=0}^{\infty} \frac{i^{k+1}\beta_{m,k}}{k!}\,\partial_t^k A_m
   + i\left(1+i\tau_S\partial_t\right)\sum_{npq}\gamma_{mnpq}
   \left((1-f_R)A_n^*A_p+f_R\left[h_R\ast A_n^*A_p\right]\right)A_q.
   $$

2. The full-field (3+1)D representation,

   $$
   \partial_z A = \sum_{k=0}^{\infty} \frac{i^{k+1}\beta_k}{k!}\,\partial_t^k A
   + \left[\frac{1}{2\beta_0}\nabla_{\perp}^2+\frac{\omega_0}{c}\Delta n(\boldsymbol{\rho})\right]A
   + i\left(1+i\tau_S\partial_t\right)\sum_{npq}\gamma_{mnpq}
   \left((1-f_R)A_n^*A_p+f_R\left[h_R\ast A_n^*A_p\right]\right)A_q.
   $$

The capabilities of the code, in outline, are:

1. Forward multimode propagation via MMGNLSE or full-field approaches.
2. CPU and GPU (CUDA) implementations for each method.
3. Kerr, Raman, self-steepening, dispersion, fixed linear gain/loss, and polarization.
4. Support for stochastic trajectories with an ensemble of initial conditions and distributed noise such as Raman noise (truncated Wigner approximation).
5. Gradients of observables with respect to initial conditions for optimization and classical or quantum noise analysis via quantum sensitivity analysis.

## Why FastDifferentiableMMNLO.jl?

Besides its capacity for handling quantum noise and differentiability, the forward solver itself is fast and can accommodate a significantly larger number of modes than existing solvers. The MMGNLSE has a runtime that scales quartically in the number of modes. We have implemented a new algorithm that is approximately quadratic in the number of modes.

As shown in the examples, this enables nontrivial simulations retaining more than 1,000 modes that run on a one-hour timescale, as well as stochastic ensembles of such systems. The CPU implementation of the CP decomposition makes it feasible to consider at least 40–50 modes; a 20-mode example that runs in 10 minutes is included in the example notebooks. Behind this is an algorithmic improvement in which the four-index nonlinear overlap tensor undergoes canonical polyadic decomposition before solving the MMGNLSE.

## Capabilities and examples

Each column heading links directly to the corresponding example notebook. A <span class="supported">✓</span> indicates that the capability is explicitly demonstrated in that example.

| Capability / feature | [Raman soliton filtering](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/squeezing_spectral_filter.ipynb) | [Soliton jitter](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/soliton_jitter.ipynb) | [Step-index squeezing](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/step_index_squeezing.ipynb) | [Normal-dispersion CP](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/om4_1030.ipynb) | [Birefringent solitons](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/birefringent_mm_soliton.ipynb) | [Stochastic fission](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/gpu_examples/stochastic_example_gpu.ipynb) | [CP mode-count scaling](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/gpu_examples/cp_decomp_modecount_gpu.ipynb) | [Full-field vs. CP](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/gpu_examples/full_field_vs_cp_gpu.ipynb) | [1,035-mode propagation](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/gpu_examples/manymode_prop_gpu.ipynb) | [30-mode CP benchmark](https://github.com/nrivera494/FastDifferentiableMMNLO/blob/main/example_notebooks/gpu_examples/cp_decomposition_30mode_soliton_gpu.ipynb) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Dense forward | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | — | <span class="supported">✓</span> | <span class="supported">✓</span> | — | <span class="supported">✓</span> |
| CP forward | — | — | <span class="supported">✓</span> | <span class="supported">✓</span> | — | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> |
| Full-field forward | — | — | — | — | — | — | — | <span class="supported">✓</span> | — | — |
| Dense adjoint | <span class="supported">✓</span> | <span class="supported">✓</span> | — | — | <span class="supported">✓</span> | — | — | — | — | — |
| CP adjoint | — | — | <span class="supported">✓</span> | — | — | — | — | — | — | — |
| Stochastic | — | — | <span class="supported">✓</span> | — | — | <span class="supported">✓</span> | — | — | — | — |
| Distributed noise | <span class="supported">✓</span> | <span class="supported">✓</span> | — | — | — | — | — | — | — | — |
| Time | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> |
| Spatial modes | — | — | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> |
| Polarization | — | — | — | — | <span class="supported">✓</span> | — | — | — | — | — |
| GPU | — | — | — | — | — | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> |
| External validity | — | <span class="supported">✓</span> | — | — | <span class="supported">✓</span> | — | — | — | — | <span class="supported">✓</span> |
| Internal validity | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | — | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> | <span class="supported">✓</span> |

The examples correspond to notebooks in [`example_notebooks/`](https://github.com/nrivera494/FastDifferentiableMMNLO/tree/main/example_notebooks). GPU examples are in [`example_notebooks/gpu_examples/`](https://github.com/nrivera494/FastDifferentiableMMNLO/tree/main/example_notebooks/gpu_examples).

**Internal validity** means that multiple implementations or limiting cases were compared within FastDifferentiableMMNLO.jl. **External validity** means that results were compared with independent code or analytical predictions.
