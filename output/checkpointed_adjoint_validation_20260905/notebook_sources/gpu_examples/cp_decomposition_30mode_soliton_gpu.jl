# CELL 3
using Sockets
using CUDA
using PulsePropagation
using CSV
using DelimitedFiles
using NPZ
using FFTW
using LinearAlgebra

println("Host: ", gethostname())
println("Slurm job: ", get(ENV, "SLURM_JOB_ID", "missing"))
println("GPU: ", CUDA.name(CUDA.device()))
println("CUDA functional: ", CUDA.functional())

# CELL 4
ENV["GKSwstype"] = "100"

using Plots
Plots.gr()

# CELL 5
## This loads the externally calculated data and the four-wave mixing tensors and dispersion for their system.

data_dir = joinpath(
    pwd(),
    "example_notebooks",
    "benchmark_data",
    "step_index_soliton",
)

S40 = npzread(joinpath(data_dir, "S_40modes.npy"))
betas40 = npzread(joinpath(data_dir, "betas_40modes.npy"))
torch_soln = npzread(joinpath(data_dir,"output_fields_30_5e-05.npy"));
torch_spec = abs2.(fftshift(ifft(torch_soln,2)));

S30 = S40[1:30, 1:30, 1:30, 1:30]

beta30 = TaylorBeta(permutedims(betas40[1:30, :]))

@show size(S30)                 # (30, 30, 30, 30)
@show size(beta30.coefficients) # (4, 30)


# CELL 6
c_light = 2.99792458e-4; # m/ps -- see manual for unit system info
Nt = 2^13;
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

# CELL 7
core_radius = 52.5e-6;
NA = 0.1;
material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
step_fiber = StepIndex(λ0, core_radius, NA);
L_fiber = 10;
num_modes = 30;

coefficients = zeros(ComplexF64, num_modes)

coefficients[6:15] .= ComplexF64[
     0.042233154627023345 + 0.9991077822989120im,
    -0.9548483373690423   + 0.29709367651226704im,
     0.8131419320624027   - 0.5820654587946469im,
     0.18628501477914675  + 0.9824957472013469im,
     0.6319768780431223   + 0.7749872422297470im,
    -0.9492435495998746   + 0.31454202190332287im,
    -0.06792170771520316  - 0.9976906542716788im,
     0.3907153673065814   - 0.9205115435183218im,
     0.9992441657975882   - 0.03887283266604082im,
     0.6421650166589653   - 0.7665664298541848im,
    ]

initial_field = gaussian_pulse(
        domain,
        num_modes;
        peak_power=751549.7920240875,    # total peak power across all modes, W
        fwhm=0.2499694427126389,          # intensity FWHM, ps
        time_offset=-30.002441147314094,  # ps
        coefficients=coefficients,
    )


properties = compute_fiber_properties(step_fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=3)

## generating compressed tensor for a small number of errors:

S_compressed_0p01 = cp_compress(
    S30;
    error=1e-2,
    backend=:cuda,
    min_rank=2,       # choose using previous experience
    rank_step=50,       # or 64
    max_rank=1000,
    maxiter=120,
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

parameters_0p01 = MMGNLSEParameters(domain;
        length=L_fiber,
        beta=beta30,
        S=S_compressed_0p01,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman)


S_compressed_0p001 = cp_compress(
    S30;
    error=1e-3,
    backend=:cuda,
    min_rank=2,       # choose using previous experience
    rank_step=50,       # or 64
    max_rank=1000,
    maxiter=120,
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

parameters_0p001 = MMGNLSEParameters(domain;
        length=L_fiber,
        beta=beta30,
        S=S_compressed_0p001,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman)

S_compressed_0p0001 = cp_compress(
    S30;
    error=1e-4,
    backend=:cuda,
    min_rank=2,       # choose using previous experience
    rank_step=50,       # or 64
    max_rank=1000,
    maxiter=120,
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

parameters_0p0001 = MMGNLSEParameters(domain;
        length=L_fiber,
        beta=beta30,
        S=S_compressed_0p0001,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman)

S_compressed_0p00001 = cp_compress(
    S30;
    error=1e-5,
    backend=:cuda,
    min_rank=2,       # choose using previous experience
    rank_step=50,       # or 64
    max_rank=1000,
    maxiter=120,
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

parameters_0p00001 = MMGNLSEParameters(domain;
        length=L_fiber,
        beta=beta30,
        S=S_compressed_0p00001,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman)

        dz_forward = 50e-6;

parameters_dense = MMGNLSEParameters(domain;
        length=L_fiber,
        beta=beta30,
        S=S30,
        n2=properties.n2,
        omega0=properties.omega0,
        raman=properties.raman)


# CELL 8
dz_forward = 50e-6;

solution_dense = solve_mmgnlse(
    initial_field, parameters_dense, dz_forward;
    backend=:cuda,
    precision=:mixed,     # or :float64
    adaptive=true,    
    reltol=1e-5,
    abstol=1e-12,
    dzmax=2e-3,
)

solution_0p01 = solve_mmgnlse(initial_field, parameters_0p01, dz_forward; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=true,
    reltol=1e-5,
    abstol=1e-12,
    dzmax=2e-3,)

solution_0p001 = solve_mmgnlse(initial_field, parameters_0p001, dz_forward; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=true,
    reltol=1e-5,
    abstol=1e-12,
    dzmax=2e-3,)

solution_0p0001 = solve_mmgnlse(initial_field, parameters_0p0001, dz_forward; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=true,
    reltol=1e-5,
    abstol=1e-12,
    dzmax=2e-3,)

solution_0p00001 = solve_mmgnlse(initial_field, parameters_0p00001, dz_forward; backend=:cuda_cp_optimized, precision=:mixed,
    adaptive=true,
    reltol=1e-5,
    abstol=1e-12,
    dzmax=2e-3,)

# CELL 9
Aω_dense = get_spectral_field(solution_dense);
Aω_0p01 = get_spectral_field(solution_0p01);
Aω_0p001 = get_spectral_field(solution_0p001);
Aω_0p0001 = get_spectral_field(solution_0p0001);
Aω_0p00001 = get_spectral_field(solution_0p00001);

# CELL 10
p = plot(λ_nm, abs2.(Aω_0p01[:,:,1]); size=(800, 500), linestyle=:dash, color=:red, label=false)
plot!(λ_nm, abs2.(Aω_0p001[:,:,1]), alpha=1, linestyle=:dashdot,label=false, color=:blue);
plot!(λ_nm, abs2.(Aω_0p0001[:,:,1]), alpha=1, linestyle=:dot, label=false, color=:green);
plot!(λ_nm, abs2.(Aω_0p00001[:,:,1]), alpha=1, linestyle=:dot, label=false, color=:orange);
plot!(λ_nm, abs2.(Aω_dense[:,:,1]), alpha=0.5, label=false, color=:purple);
plot!(λ_nm, torch_spec[:,:]', alpha=0.5, label=false, color=:black);
xlims!(p, (1480, 1920))
ylims!(p, (0,1.4))
display(p)

# CELL 11
npzwrite("wavelengths.npy", λ_nm)
npzwrite("spectral_fields_0p01.npy", Aω_0p01)
npzwrite("spectral_fields_0p001.npy", Aω_0p001)
npzwrite("spectral_fields_0p0001.npy", Aω_0p0001)
npzwrite("spectral_fields_0p00001.npy", Aω_0p00001)
npzwrite("spectral_fields_dense.npy", Aω_dense)