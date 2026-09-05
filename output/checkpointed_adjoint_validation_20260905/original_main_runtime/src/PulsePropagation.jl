__precompile__(false)

module PulsePropagation

using FFTW
using LinearAlgebra
using Random
using Statistics
using Zygote
using DelimitedFiles
using DifferentialEquations
using ChainRulesCore
using ForwardDiff

import Optim

include("backend/types.jl")
include("backend/fft_utils.jl")
include("backend/linear_gain.jl")
include("backend/dispersion.jl")
include("backend/srsk.jl")
include("backend/raman.jl")
include("backend/nonlinear.jl")
include("backend/compressed.jl")
include("backend/cp_rhs_cached.jl")
include("backend/rk4ip.jl")
include("backend/propagate.jl")
include("backend/adjoint.jl")
include("backend/ad_propagate.jl")
include("backend/stochastic.jl")
include("backend/cuda_api.jl")
include("grids.jl")
include("states.jl")
include("systems.jl")
include("mmgnlse_types.jl")
include("mmgnlse_cp.jl")
include("mmgnlse_fibers.jl")
include("mmgnlse_cache.jl")
include("mmgnlse_solver.jl")
include("experimental_full_field.jl")
include("mmgnlse_vjp.jl")
include("mmgnlse_adjoint_stochastic.jl")
include("mmgnlse_solution_cache.jl")
include("pulses.jl")
include("models.jl")
include("solvers.jl")
include("trajectories.jl")
include("observables.jl")
include("mmgnlse_observables.jl")
include("mmgnlse_autodiff.jl")
include("gradients.jl")
include("variance_objectives.jl")
include("optimization.jl")
include("backend_bridge.jl")
include("stochastic.jl")

export TimeGrid,
       DegreesOfFreedom,
       degrees_of_freedom,
       TransverseGrid,
       MMGNLSEDomain,
       TaylorBeta,
       SampledBeta,
       beta_order,
       SpatialOverlap,
       ComponentOverlap,
       NoRaman,
       AgarwalRaman,
       AnisotropicRaman,
       Silica,
       raman_response,
       MMGNLSEParameters,
       MMGNLSECPDecomposition,
       CPMemoryBudget,
       cp_compress,
       cp_compress_bounded,
       cp_compression_plan,
       cp_compress_symmetric_experimental,
       cp_compression_memory_estimate,
       cp_inner_product,
       cp_norm,
       cp_relative_error,
       cp_rank,
       cp_iszero,
       cp_is_symmetric,
       cp_reconstruct_mmgnlse,
       cp_source_matches,
       validate_cp_source,
       GRIN,
       grin_hg_overlap_quadrature_cp,
       StepIndex,
       IndexProfile,
       FiberModeData,
       FiberProperties,
       silica_refractive_index,
       material_refractive_index,
       evaluate_index_profile,
       compute_fiber_properties,
       mode_grid,
       mode_fields,
       propagation_constants,
       spatial_overlap,
       mode_overlap_quadrature_cp,
       cp_compress_mode_overlap,
       canonical_array_checksum,
       fiber_properties_checksum,
       save_fiber_properties,
       load_fiber_properties,
       cp_decomposition_checksum,
       save_cp_decomposition,
       load_cp_decomposition,
       mmgnlse_parameters_checksum,
       mmgnlse_solution_checksum,
       save_mmgnlse_solution,
       load_mmgnlse_solution,
       RK4IP,
       Tsit5,
       Vern9,
       MMGNLSESolution,
       MMGNLSEAdjointSolution,
       MMGNLSEStochasticSolution,
       solve_mmgnlse,
       StrangSplit,
       ComposedStrang4,
       FullFieldMMGNLSEParameters,
       FullFieldMMGNLSESolution,
       solve_mmgnlse_full_field,
       FullField3DGrid,
       FullField3DParameters,
       FullField3DSolution,
       full_field_shape,
       paraxial_index_potential,
       reconstruct_full_field,
       project_full_field,
       full_field_power,
       full_field_subspace_leakage,
       phase_invariant_relative_error,
       solve_full_field_3d,
       solve_adjoint,
       get_autodiff_adjoint,
       solve_mmgnlse_stochastic,
       gaussian_pulse,
       get_temporal_field,
       get_spectral_field,
       get_spatiotemporal_field,
       get_spatial_field,
       get_spectrally_summed_spatial_intensity,
       AbstractObservable,
       SpectralPhotonNumber,
       FilterEnergy,
       BinEnergy,
       RatioObservable,
       TemporalMoment,
       SpectralMoment,
       TemporalCentroid,
       SpectralCentroid,
       photon_spectrum,
       modal_energy,
       photon_number,
       value,
       terminal_condition,
       time_axis,
       frequency_axis,
       wavelength_axis,
       photon_weights

end
