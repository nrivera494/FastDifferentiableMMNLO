# CELL 3
using Pkg

ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using PulsePropagation
using FFTW
using LinearAlgebra
using PyPlot
using DelimitedFiles

# CELL 5
c_light = 2.99792458e-4; # m/ps -- see manual for unit system info
Nt = 2^12;
time_window = 50; # ps
λ0 = 1550e-9; # m
ω0 = 2*π*c_light/λ0;

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
f = ω / (2*π);
λ = wavelength_axis(time_grid,ω0); # wavelength
λ_nm = λ/1e-9; # wavelength

dofs = degrees_of_freedom(:time)
domain = MMGNLSEDomain(dofs, time_grid);


# CELL 7
material = Silica(raman=AgarwalRaman()) # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
mfd = 9.2e-6; # mode-field diameter.
Aeff= π * (mfd / 2)^2;
fiber_length = 10;

beta2 = -2.06e-26 * 1e24;
beta3 = 8.1038e-41 * 1e36;
betas = reshape([0.0, 0.0, beta2, beta3], :, 1);

# CELL 9
peak_power = 3655.7;
pulse_fwhm = 0.22;
pulse_tau = pulse_fwhm / (2*acosh(sqrt(2)));

initial_field = zeros(Nt,1,1);
initial_field[:,1,1] .= sqrt(peak_power)*sech.(t/pulse_tau);

# CELL 11
parameters = MMGNLSEParameters(domain;
length=fiber_length,
beta=TaylorBeta(betas),
S=1/Aeff,
n2=2.3e-20,
omega0=2*π*c_light / λ0,
raman=AgarwalRaman())

# CELL 13
dz = 0.001;
solution = solve_mmgnlse(initial_field, parameters, dz; saveat=0.001);

At_final = get_temporal_field(solution)
Aomega_final = get_spectral_field(solution)
z_saved = solution.z;

# CELL 14
figure(figsize=(3,3))
plot(λ_nm,abs2.(Aomega_final[:,1,1]))
xlim(1500,1700)
xlabel("λ (nm)")
ylabel("Spectrum (a.u.)")
display(gcf())

# CELL 15
spec_z = abs2.(ifftshift(ifft(solution.fields, 1),1))
figure(figsize=(3,3))
contourf(λ_nm,solution.z,10log10.(spec_z[:,1,1,:]'),levels=[-110:2.5:0;])
clim(-40,0)
xlim(1400,1700)
colorbar()
display(gcf())

# CELL 17
peak_pass_ind = argmax(abs2.(Aomega_final[:,1,1]))
filter = zeros(length(λ));
filter[peak_pass_ind] = 1.0;

figure(figsize=(3,3))
plot(λ_nm, filter)
xlim(1500,1700)
display(gcf())

# CELL 19
peak_photon_number =  SpectralPhotonNumber(filter=filter, modes=:all);
filter_mean = value(peak_photon_number, solution)
lambda_L = terminal_condition(peak_photon_number, solution);

# CELL 20
dz_adj = 0.001;
adjoint = solve_adjoint(lambda_L, parameters, dz_adj; forward_solution=solution)
peak_sensitivity = get_spectral_field(adjoint; units=:photon)[:,:,1];

# CELL 21
filter_var = sum(abs2.(peak_sensitivity));
fano = filter_var / filter_mean
println("The Fano factor of the Raman soliton is $(fano)")

# CELL 22
figure(figsize=(3,3))
semilogy(λ_nm, abs2.(peak_sensitivity))
xlim(1500,1700)
ylim(1)
xlabel("λ (nm)")
ylabel("Raman peak sensitivity |∂n/∂α|²")
display(gcf())

# CELL 24
opt_filt_trunc = readdlm("benchmark_data/squeezing_spectral_filter/optimal_filter.csv");
opt_filt = zeros(Nt);
opt_filt[1196:2371] = reverse(opt_filt_trunc); 

figure(figsize=(3,3))
plot(λ_nm, opt_filt)
xlim(1550,1700)
display(gcf())

# CELL 25
filtered_photon_number = SpectralPhotonNumber(filter=opt_filt);
filter_mean = value(filtered_photon_number, solution)
lambda_L = terminal_condition(filtered_photon_number, solution);

dz_adj = 0.001;
adjoint = solve_adjoint(lambda_L, parameters, dz_adj; forward_solution=solution, saveat=0.01)
filtered_sensitivity = get_spectral_field(adjoint; units=:photon)[:,:,1];

# CELL 26
figure(figsize=(3,3))
semilogy(λ_nm, abs2.(filtered_sensitivity))
xlim(1500,1700)
ylim(1)
xlabel("λ (nm)")
ylabel("Raman peak sensitivity (a.u)")
display(gcf())

# CELL 27
filter_var = sum(abs2.(filtered_sensitivity));
fano = filter_var / filter_mean
println("The Fano factor of the Raman soliton is $(fano)")

# CELL 29
F0, λ0, Δλ = 9, 1550, 20;
fano_list = F0*exp.(-(λ_nm .- λ0) .^2 ./ (Δλ)^2) .+ 1;

figure(figsize=(3,3))
plot(λ_nm,fano_list)
xlim(1500,1600)
xlabel("λ (nm)")
ylabel("Fano factor")
display(gcf())

# CELL 30
figure(figsize=(3,3))
semilogy(λ_nm, abs2.(filtered_sensitivity))
semilogy(λ_nm, fano_list .* abs2.(filtered_sensitivity), "--")
xlim(1500,1700)
ylim(1)
xlabel("λ (nm)")
ylabel("Raman peak sensitivity (a.u)")
display(gcf())

# CELL 31
var_with_noise = sum(fano_list .* abs2.(filtered_sensitivity));
fano_with_noise = var_with_noise/filter_mean;

# CELL 33
filtered_photon_number = SpectralPhotonNumber(filter=opt_filt);
filter_mean = value(filtered_photon_number, solution)
lambda_L = terminal_condition(filtered_photon_number, solution);

dz_adj = 0.001;
adjoint = solve_adjoint(lambda_L, parameters, dz_adj; forward_solution=solution, saveat=0.001)
filtered_sensitivity = get_spectral_field(adjoint)[:,:,1];

# CELL 34
gamma = parameters.n2 * parameters.omega0 / c_light / Aeff  # W^-1 m^-1
dt = diff(t)[1];
f0 = ω0/(2*π);

fR = parameters.raman.fraction
tau1 = parameters.raman.tau1;
tau2 = parameters.raman.tau2;

temp = 300.0;

Cω = PulsePropagation.raman_langevin_spectrum(
    Nt, dt, f0, gamma;
    raman_fraction=parameters.raman.fraction,
    temperature_K=temp,
)

Cω = ifftshift(Cω)

figure(figsize=(3,3))
plot(abs.(Cω))
display(gcf())

# CELL 35
Azt = get_temporal_field(solution; z=:all, units=:power);
λzt = conj(get_temporal_field(adjoint; z=:all, units=:power));
Bzt = -2*imag(Azt .* λzt);
Bzω = ifftshift(ifft(Bzt,1));
Δz = diff(adjoint.z)[1];
dn2_raman = gamma^2 * sum(abs2.(Bzω) .* Cω) * (diff(f)[1] * 1e12) * Δz

# CELL 36
fano_with_raman = (filter_var + dn2_raman)/filter_mean

# CELL 37


# CELL 38
