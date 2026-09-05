# CELL 3
using Pkg

ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(ROOT)

using PulsePropagation
using FFTW
using LinearAlgebra
using PyPlot
using Statistics

# CELL 5
c_light = 2.99792458e-4; # m/ps -- see manual for unit system info
Nt = 2^11;
time_window = 10; # ps
λ0 = 1379.2e-9; # m
ω0 = 2*π*c_light/λ0;

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
λ = wavelength_axis(time_grid,ω0); # wavelength

dofs = degrees_of_freedom(:time, :space)
domain = MMGNLSEDomain(dofs, time_grid);


# CELL 7
core_radius = 52.5e-6;
NA = 0.1;
num_modes = 3;
fiber_length = 2.86;

material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
step_fiber = StepIndex(λ0, core_radius, NA);

properties = compute_fiber_properties(step_fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=5)

# CELL 9
x = y = properties.modes.x
modes = properties.modes.fields; 
betas = properties.beta.coefficients;
S_tensor = properties.S.values;

# CELL 11
figure(figsize=(18,6))

for ii = 1:3
    subplot(1,3,ii)
    pcolormesh(x*1e6,y*1e6,modes[:,:,ii],cmap="seismic")
end
display(gcf())

# CELL 13
peak_power = 100000;
pulse_fwhm = 0.2;

# pulse_tau = pulse_fwhm / (2*acosh(sqrt(2)));
# initial_field = zeros(Nt,num_modes,1);
# initial_field[:,1,1] .= sqrt(peak_power)*sech.(t/pulse_tau);

coefficient_vector = zeros(num_modes);
coefficient_vector[1] = 1.0;

initial_field = gaussian_pulse(
    domain, num_modes;
    peak_power=peak_power, # W
    fwhm=pulse_fwhm,
    coefficients = coefficient_vector
);

# CELL 15
parameters = MMGNLSEParameters(domain;
length=fiber_length,
beta=properties.beta,
S=properties.S,
n2=properties.n2,
omega0=properties.omega0,
raman=properties.raman)

# CELL 17
dz = fiber_length/4000;
solution = solve_mmgnlse(initial_field, parameters, dz; saveat=0.1);

# CELL 19
At_final = get_temporal_field(solution)
Aomega_final = get_spectral_field(solution)
z_saved = solution.z;

# CELL 20
figure(figsize=(3,3))
plot(t,abs2.(At_final[:,1,1]))
display(gcf())

# CELL 21
figure(figsize=(3,3))
plot(λ,abs2.(Aomega_final[:,1,1]))
xlim(1340e-9,1440e-9)
display(gcf())

# CELL 24
S_compressed = cp_compress(properties.S,error=1e-3);

# CELL 25
parameters_cp = MMGNLSEParameters(domain;
length=fiber_length,
beta=properties.beta,
S=S_compressed,
n2=properties.n2,
omega0=properties.omega0,
raman=properties.raman)

dz = 1e-3;

solution_cp = solve_mmgnlse(initial_field, parameters_cp, dz; saveat=0.1);

# CELL 26
At_final_cp = get_temporal_field(solution_cp)
Aomega_final_cp = get_spectral_field(solution_cp)
z_saved_cp = solution_cp.z;

# CELL 27
figure(figsize=(3,3))
plot(t,abs2.(At_final[:,1,1]))
plot(t,abs2.(At_final_cp[:,1,1]),"--")
display(gcf())

# CELL 28
figure(figsize=(3,3))
plot(λ,abs2.(Aomega_final[:,1,1]))
plot(λ,abs2.(Aomega_final_cp[:,1,1]),"--")
xlim(1340e-9,1440e-9)
display(gcf())

# CELL 30
long_pass_ind = argmin(abs.(λ .-  1370.6998e-9))
filter = zeros(length(λ));
filter[1:long_pass_ind] .= 1.0;

figure(figsize=(3,3))
plot(λ, filter)
xlim(1350e-9,1450e-9)
display(gcf())

# CELL 31
longpass_photon_number =  SpectralPhotonNumber(filter=filter, modes=:all);
filter_mean = value(longpass_photon_number, solution_cp)
lambda_L = terminal_condition(longpass_photon_number, solution_cp);

# CELL 32
dz_adj = fiber_length/4000;

adjoint = solve_adjoint(lambda_L, parameters_cp, dz_adj; forward_solution=solution_cp);
# adjoint emits super verbose stuff now -- fix

# CELL 33
filter_var = sum(abs2.(get_spectral_field(adjoint; units=:photon)[:,:,1]));

# CELL 34
fano = filter_var / filter_mean
10log10(fano)

# CELL 36
dz_ad = 1e-2;
adjoint_ad = get_autodiff_adjoint(
    longpass_photon_number,
    initial_field,
    parameters_cp,
    dz_ad; units=:photon
)

# CELL 37
filter_var_ad = sum(abs2.(adjoint_ad))
fano_ad = 10log10.(filter_var_ad/filter_mean);

# CELL 39
num_ensemble = 100;
ensemble = solve_mmgnlse_stochastic(initial_field, parameters_cp, dz, num_ensemble)

# CELL 40
fano_stochastic = 10log10(var(value(longpass_photon_number, ensemble)) / filter_mean)

# CELL 41
