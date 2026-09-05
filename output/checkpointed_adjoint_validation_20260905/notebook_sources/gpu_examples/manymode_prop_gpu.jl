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
Nt = 2^13;
time_window = 80; # ps
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
core_radius = 100e-6;
NA = 0.275;
material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
grin_fiber = GRIN(λ0, core_radius, NA);

num_groups = 45;
num_modes = Int((num_groups)*(num_groups+1)/2);

properties = compute_fiber_properties(grin_fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=3, overlap_representation=:quadrature_cp,
    overlap_quadrature_order=256, grid_half_width=1.3*core_radius,
    grid_size=384)

# CELL 5
S_compressed_1em1 = cp_compress(
                properties.S;
                error=1e-1,
                backend=:cuda,
                min_rank=100,       # choose using previous experience
                rank_step=300,       # or 64
                max_rank=6000,
                maxiter=120,
                tolerance=1e-6,
                ridge=1e-8,
                check_every=5,
                )

S_compressed_1em3 = cp_compress(
                properties.S;
                error=1e-3,
                backend=:cuda,
                min_rank=500,       # choose using previous experience
                rank_step=500,       # or 64
                max_rank=10000,
                maxiter=120,
                tolerance=1e-6,
                ridge=1e-8,
                check_every=5,
                )

# CELL 6
coefficients = zeros(ComplexF64, num_modes) 
coefficients[1:120] .= 1.0+0.0im;

initial_field = gaussian_pulse(
        domain,
        num_modes;
        peak_power=7.5e5,    # total peak power across all modes, W
        fwhm=0.25,          # intensity FWHM, ps
        time_offset=0,  # ps
        coefficients=coefficients,
    );

dz_cp = 100e-6;
length_m = 1;


parameters_compressed_1em1 = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=properties.beta,
    S=S_compressed_1em1,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp_1em1 = solve_mmgnlse(initial_field, parameters_compressed_1em1, dz_cp; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=false)


# CELL 7
dz_cp = 100e-6;
length_m = 1;

parameters_compressed_1em3 = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=properties.beta,
    S=S_compressed_1em3,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp_1em3 = solve_mmgnlse(initial_field, parameters_compressed_1em3, dz_cp; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=false)

# CELL 8
Aω_0p1 = get_spectral_field(solution_cp_1em1);
Aω_0p001 = get_spectral_field(solution_cp_1em3);
sol_err = norm(abs2.(Aω_0p1[:,:,1]) - abs2.(Aω_0p001[:,:,1]))/norm(abs2.(Aω_0p001[:,:,1]))

p = plot(λ_nm, abs2.(Aω_0p1[:,1:120,1]); size=(800, 500), linestyle=:dash, color=:red, label=false,alpha=0.6)
plot!(λ_nm, abs2.(Aω_0p001[:,1:120,1]); size=(800, 500), color=:black, label=false, alpha=0.2)
xlims!(p, (1400, 2100))
display(p)

# CELL 9
npzwrite("spectral_fields_cp_1035modes_0p1.npy", Aω_0p1)
npzwrite("spectral_fields_cp_1035modes_0p001.npy", Aω_0p001)
npzwrite("wavelengths_1035modes.npy", λ_nm)

# CELL 10
Ixy_in = get_spectrally_summed_spatial_intensity(
    solution_cp_1em1,
    properties;
    z=:initial,
)

Ixy_0p1 = get_spectrally_summed_spatial_intensity(
    solution_cp_1em1,
    properties;
    z=:final,
)

Ixy_0p001 = get_spectrally_summed_spatial_intensity(
    solution_cp_1em3,
    properties;
    z=:final,
)

# CELL 11
x = properties.modes.x
y = properties.modes.y

heatmap(
    x .* 1e6,
    y .* 1e6,
    permutedims(Ixy_in);
)

# CELL 12
x = properties.modes.x
y = properties.modes.y

heatmap(
    x .* 1e6,
    y .* 1e6,
    permutedims(Ixy_0p1);
)

# CELL 13

heatmap(
    x .* 1e6,
    y .* 1e6,
    permutedims(Ixy_0p001);
)

# CELL 14
npzwrite("xvals_1035modes.npy", x .* 1e6)
npzwrite("Iwxy_cp_1035modes_0p1.npy", Ixy_0p1)
npzwrite("Iwxy_cp_1035modes_0p001.npy", Ixy_0p001)
npzwrite("Iwxy_cp_1035modes_initial.npy", Ixy_in)

# CELL 15
