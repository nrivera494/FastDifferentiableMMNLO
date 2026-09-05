# CELL 4
using Pkg

ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using PulsePropagation
using FFTW
using LinearAlgebra
using PyPlot
using DelimitedFiles
using CSV

# CELL 6
betas_biref = [5.847462729085851e+06	5.851435323667164e+06	5.841335866995176e+06	5.845308461576490e+06
4.878764128935511e+03	0	4.878836422580576e+03	0
-0.028100040576544	0	-0.028248131742058	0
1.518979580892188e-04	0	1.524304694210461e-04	0
-4.917494281128627e-07	0	-4.940312405234015e-07	0
2.315909039461211e-09	0	2.327945659939789e-09	0];

S_vals = reshape(readdlm("benchmark_data/birefringent_mm_soliton/S_tensors_10modes.csv",',',Float64;skipstart=1,)[:, 5],10, 10, 10, 10);

wavelengths = readdlm("benchmark_data/birefringent_mm_soliton/wavelength_nm.csv");
spec1 = readdlm("benchmark_data/birefringent_mm_soliton/spectrum_1.csv");
spec2 = readdlm("benchmark_data/birefringent_mm_soliton/spectrum_2.csv");
spec3 = readdlm("benchmark_data/birefringent_mm_soliton/spectrum_3.csv");
spec4 = readdlm("benchmark_data/birefringent_mm_soliton/spectrum_4.csv");

# CELL 8
c_light = 2.99792458e-4; # m/ps -- see manual for unit system info
Nt = 2^11;
time_window = 50; # ps
λ0 = 1550e-9; # m
ω0 = 2*π*c_light/λ0;

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
λ = wavelength_axis(time_grid,ω0); # wavelength
λ_nm = 1e9*λ;

dofs = degrees_of_freedom(:time, :space, :polarization)
domain = MMGNLSEDomain(dofs, time_grid);


# CELL 10
num_modes = 2;
L_fiber = 2;
material = Silica()

S_tensor = S_vals[1:2,1:2,1:2,1:2];

betas = zeros((size(betas_biref,1)),2,2);
betas[:,1,1] = betas_biref[:,1];
betas[:,2,1] = betas_biref[:,3];
betas[:,1,2] = betas_biref[:,2];
betas[:,2,2] = betas_biref[:,4];

parameters = MMGNLSEParameters(domain; length=L_fiber, beta=TaylorBeta(betas), S=S_tensor, n2=material.n2, omega0 = ω0, raman = material.raman);


pulse_fwhm = 0.500
peak_power = 9394.39; # close enough to 5 nJ.
mode_coefficients = ones(2,2)

initial_field = gaussian_pulse(
    domain, num_modes;
    peak_power=peak_power, # W
    fwhm=pulse_fwhm,
    coefficients = mode_coefficients
);

# CELL 11
dz = 0.0003;
solution = solve_mmgnlse(initial_field, parameters, dz; saveat=0.1);

# CELL 12
At = get_temporal_field(solution,z=:final)
spec_out = get_spectral_field(solution,z=:final);

# CELL 13
figure(figsize=(3,3))
plot(λ_nm,abs2.(spec_out[:,:,1]))
plot(λ_nm,abs2.(spec_out[:,:,2]))
plot(wavelengths, spec1, color="k", "--", alpha=0.5)
plot(wavelengths, spec2, color="k", "--", alpha=0.5)
plot(wavelengths, spec3, color="k", "--", alpha=0.5)
plot(wavelengths, spec4, color="k", "--", alpha=0.5)
xlim(1500,1600)
display(gcf())

# CELL 14
maxind = argmax(abs2.(spec_out[:,1,1]))
filter = zero(spec_out[:,:,:]);
filter[maxind,1,1] = 1.0;
max_filt = real(filter);

obs = SpectralPhotonNumber(filter=max_filt)
n_out = value(obs, solution)

# CELL 15
dz_adj = dz;
lambda_L = terminal_condition(obs, solution);
adjoint = solve_adjoint(lambda_L, parameters, dz; forward_solution = solution, units = :photon)

# CELL 16
spectral_sensitivity = get_spectral_field(adjoint; z = :initial, units = :photon);

# CELL 17
figure(figsize=(3,3))
semilogy(λ_nm, abs2.(spectral_sensitivity[:,:,1]))
semilogy(λ_nm, abs2.(spectral_sensitivity[:,:,2]))
xlim(1500,1600)
display(gcf())

# CELL 18
dz_ad = 0.0003;
spectral_sensitivity_ad  = get_autodiff_adjoint(
    obs,
    initial_field,
    parameters,
    dz_ad; units=:photon
)

# CELL 20
figure(figsize=(3,3))
semilogy(λ_nm, abs2.(spectral_sensitivity[:,:,1]))
semilogy(λ_nm, abs2.(spectral_sensitivity[:,:,2]))
semilogy(λ_nm, abs2.(spectral_sensitivity_ad[:,:,1]),"--")
semilogy(λ_nm, abs2.(spectral_sensitivity_ad[:,:,2]),"--")
xlim(1500,1600)
ylim(1e-4,1e8)
display(gcf())