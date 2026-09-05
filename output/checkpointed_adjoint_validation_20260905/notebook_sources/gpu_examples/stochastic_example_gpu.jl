# CELL 1
using Sockets
using CUDA
using PulsePropagation
using CSV
using DelimitedFiles
using NPZ
using FFTW
using LinearAlgebra
using Arpack
using JLD2

println("Host: ", gethostname())
println("Slurm job: ", get(ENV, "SLURM_JOB_ID", "missing"))
println("GPU: ", CUDA.name(CUDA.device()))
println("CUDA functional: ", CUDA.functional())

# CELL 2
ENV["GKSwstype"] = "100"

using Plots
Plots.gr()

# CELL 3
c_light = 2.99792458e-4; # m/ps -- see manual for unit system info
Nt = 2^14;
time_window = 100; # ps
λ0 = 1550e-9; # m
ω0 = 2*π*c_light/λ0;

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
λ = wavelength_axis(time_grid,ω0); # wavelength
λ_nm = 1e9 * λ;

dofs = degrees_of_freedom(:time, :space)
domain = MMGNLSEDomain(dofs, time_grid);

# CELL 4
core_radius = 52.5e-6;
NA = 0.275;
material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
grin_fiber = GRIN(λ0, core_radius, NA);

num_groups = 14;
num_modes = Int((num_groups)*(num_groups+1)/2);

properties = compute_fiber_properties(grin_fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=3, overlap_representation=:quadrature_cp,
    overlap_quadrature_order=128, grid_half_width=1.3*core_radius,
    grid_size=256)

# CELL 5
S_compressed = cp_compress(
                properties.S;
                error=1e-3,
                backend=:cuda,
                min_rank=100,       # choose using previous experience
                rank_step=50,       # or 64
                max_rank=6000,
                maxiter=120,
                tolerance=1e-6,
                ridge=1e-8,
                check_every=5,
                )


# CELL 6
coefficients = zeros(ComplexF64, num_modes) 
coefficients[1:20] .= 1.0+0.0im;

initial_field = gaussian_pulse(
        domain,
        num_modes;
        peak_power=3e5,    # total peak power across all modes, W
        fwhm=0.25,          # intensity FWHM, ps
        time_offset=0,  # ps
        coefficients=coefficients,
    );

dz_cp = 50e-6;
length_m = 1;

parameters_compressed = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=properties.beta,
    S=S_compressed,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp = solve_mmgnlse(initial_field, parameters_compressed, dz_cp; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=false)

Aω_0p01 = get_spectral_field(solution_cp);
At_0p01 = get_temporal_field(solution_cp);

# CELL 7
num_ensemble = 120;
ensemble = solve_mmgnlse_stochastic(initial_field, parameters_compressed, dz_cp, num_ensemble;backend=:cuda_cp_optimized, precision=:mixed, noise = 1e6)

# CELL 8
p = plot(λ_nm, abs2.(get_spectral_field(ensemble))[:,1:20,1,1],color=:red)
plot!(λ_nm, abs2.(get_spectral_field(ensemble))[:,1:20,1,3],color=:blue)
xlims!(p, (1100, 2300))
display(p)

# CELL 9
JLD2.jldsave(
    "ensemble.jld2";
    ensemble,
    z=ensemble.z,
    fields=ensemble.fields,
    fano=ensemble.noise.fano,
    photon_noise=ensemble.noise.photon_noise,
    weights=ensemble.noise.weights,
)

npzwrite("λ_stochastic.npy",λ_nm)

# CELL 10
