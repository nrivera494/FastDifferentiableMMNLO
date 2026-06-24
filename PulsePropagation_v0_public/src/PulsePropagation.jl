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
include("pulses.jl")
include("models.jl")
include("solvers.jl")
include("trajectories.jl")
include("observables.jl")
include("gradients.jl")
include("variance_objectives.jl")
include("optimization.jl")
include("backend_bridge.jl")
include("stochastic.jl")

export TimeGrid,
       SingleMode,
       ModalBasis,
       PolarizedModalBasis,
       SingleModeField,
       ModalField,
       PolarizedModalField,
       PowerNormalized,
       PhotonNormalized,
       TimeDomain,
       FrequencyDomain,
       PulseState,
       PassiveFiber,
       GainFiber,
       load_fiber_system,
       TaylorDispersion,
       GridDispersion,
       ModeOverlapTensor,
       EffectiveArea,
       Silica,
       AgarwalRaman,
       Chalcogenide,
       LinearGain,
       PropagationModel,
       PropagationTerms,
       Dispersion,
       Kerr,
       Raman,
       Gain,
       DispersionTerm,
       KerrTerm,
       RamanTerm,
       GainTerm,
       FixedStep,
       AdaptiveStep,
       SaveAt,
       SaveEvery,
       CPCompression,
       CachedCPRHS,
       FixedRK4IPCPRHS,
       CPUBackend,
       CUDABackend,
       RK4IPSolver,
       PulsePropagationProblem,
       PulseTrajectory,
       Adjoint,
       AutomaticDifferentiation,
       InitialField,
       AbstractObservable,
       SpectralPhotonNumber,
       FilterEnergy,
       BinEnergy,
       QuadratureObservable,
       ProjectedEnergy,
       ProjectedPhotonNumber,
       RatioObservable,
       TemporalMoment,
       SpectralMoment,
       TemporalCentroid,
       SpectralCentroid,
       VarianceObjective,
       variance_result,
       variance_directional_derivative,
       fano,
       ShotNoise,
       RamanLangevin,
       GainLangevin,
       StochasticEnsemble,
       StochasticTrajectory,
       stochastic_ensemble,
       solve_stochastic,
       stochastic_values,
       stochastic_raman_added_variance,
       fano_factor,
       PropagationOptimizationProblem,
       AdjointGradient,
       AutomaticDifferentiationGradient,
       FiniteDifferenceGradient,
       NoGradient,
       DifferentiableRegularizer,
       GradientDescent,
       CoordinateSearch,
       OptimizationResult,
       optimization_value,
       optimization_loss,
       optimization_gradient,
       value_and_gradient,
       optimize,
       solve,
       field,
       spectrum,
       z_spectrum,
       photon_spectrum,
       compressed_tensor,
       compression_rank,
       compression_error,
       modal_energy,
       photon_number,
       value,
       terminal_condition,
       gradient,
       photon_normalized_adjoint_trajectory,
       import_backend,
       backend_problem,
       backend_objects,
       sech_pulse,
       gaussian_pulse,
       time_axis,
       frequency_axis,
       wavelength_axis,
       photon_weights

end
