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
Nt = 2^13;
time_window = 10; # ps
λ0 = 1030e-9; # m
ω0 = 2*π*c_light/λ0;

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
λ = wavelength_axis(time_grid,ω0); # wavelength

dofs = degrees_of_freedom(:time, :space)
domain = MMGNLSEDomain(dofs, time_grid);


# CELL 7
core_radius = 168e-6;
NA = 0.1;
num_modes = 6;
fiber_length = 1;

material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
step_fiber = GRIN(λ0, core_radius, NA);

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
peak_power = 50000;
pulse_fwhm = 0.5;

coefficient_vector = ones(num_modes) .* exp.(1im .* randn(num_modes));
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
dz = 1e-4;
solution = solve_mmgnlse(initial_field, parameters, dz; saveat=0.1);

# CELL 19
At_final = get_temporal_field(solution)
Aomega_final = get_spectral_field(solution)
z_saved = solution.z;

# CELL 20
figure(figsize=(3,3))
plot(t,abs2.(At_final[:,:,end]))
xlim(-2,2)
display(gcf())

# CELL 21
figure(figsize=(3,3))
plot(λ,abs2.(Aomega_final[:,:,1]))
xlim(1020e-9,1040e-9)
display(gcf())

# CELL 24
S_compressed_1em1 = cp_compress(properties.S,error=1e-1);
S_compressed_1em3 = cp_compress(properties.S,error=1e-3);

parameters_cp_1em1 = MMGNLSEParameters(domain;
length=fiber_length,
beta=properties.beta,
S=S_compressed_1em1,
n2=properties.n2,
omega0=properties.omega0,
raman=properties.raman)

parameters_cp_1em3 = MMGNLSEParameters(domain;
length=fiber_length,
beta=properties.beta,
S=S_compressed_1em3,
n2=properties.n2,
omega0=properties.omega0,
raman=properties.raman)

# CELL 25
dz = 1e-4;

solution_cp_1em1 = solve_mmgnlse(initial_field, parameters_cp_1em1, dz; saveat=0.1);
solution_cp_1em3 = solve_mmgnlse(initial_field, parameters_cp_1em3, dz; saveat=0.1);

# CELL 26
At_final_cp_1em1 = get_temporal_field(solution_cp_1em1)
Aomega_final_cp_1em1 = get_spectral_field(solution_cp_1em1)
At_final_cp_1em3 = get_temporal_field(solution_cp_1em3)
Aomega_final_cp_1em3 = get_spectral_field(solution_cp_1em3)

# CELL 27
figure(figsize=(3,3))
plot(t,abs2.(At_final[:,:,1]))
plot(t,abs2.(At_final_cp_1em1[:,:,1]),"--")
plot(t,abs2.(At_final_cp_1em3[:,:,1]),"-.")
xlim(-2,2)
display(gcf())

# CELL 28
figure(figsize=(3,3))
plot(λ,abs2.(Aomega_final[:,:,1]),alpha=0.1)
plot(λ,abs2.(Aomega_final_cp_1em1[:,:,1]),"--")
plot(λ,abs2.(Aomega_final_cp_1em3[:,:,1]),"-.")
xlim(1020e-9,1040e-9)
display(gcf())

# CELL 29
## timing difference is ~55 min for uncompressed simulation versus 1.25 min at error 1e-3 for compressed at just six modes.

# CELL 31
core_radius = 168e-6;
NA = 0.1;
num_modes = 21;
fiber_length = 1;

material = Silica() # The default Silica picks an anisotropic Raman model. See other notebooks such as soliton_jitter for other Raman models;
step_fiber = GRIN(λ0, core_radius, NA);

properties = compute_fiber_properties(step_fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=5)

x = y = properties.modes.x
modes = properties.modes.fields; 
betas = properties.beta.coefficients;
S_tensor = properties.S.values;

# CELL 32
figure(figsize=(21,7))

for ii = 1:num_modes
    subplot(3,7,ii)
    pcolormesh(x*1e6,y*1e6,modes[:,:,ii],cmap="seismic")
end
display(gcf())

# CELL 33
## Same excitation as 6 mode case, may not provide same output spectrum if other modes beyond 1:6 can be addressed by dynamics
coefficients = 1.0im*zeros(num_modes)
coefficients[1:6] .= coefficient_vector;

# CELL 34
peak_power = 50000;
pulse_fwhm = 0.5;

initial_field = gaussian_pulse(
    domain, num_modes;
    peak_power=peak_power, # W
    fwhm=pulse_fwhm,
    coefficients = coefficients
);

# CELL 35
S_compressed = cp_compress(properties.S,error=1e-3; min_rank=50,
    rank_step=25,
    max_rank=300,
    maxiter=50, #120
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5);

parameters_cp = MMGNLSEParameters(domain;
length=fiber_length,
beta=properties.beta,
S=S_compressed,
n2=properties.n2,
omega0=properties.omega0,
raman=properties.raman)

# CELL 36
dz = 1e-4;
solution_cp = solve_mmgnlse(initial_field, parameters_cp, dz);

# CELL 37
At_final_cp = get_temporal_field(solution_cp)
Aomega_final_cp = get_spectral_field(solution_cp)

# CELL 38
figure(figsize=(3,3))
plot(t,abs2.(At_final_cp[:,:,1]))
xlim(-2,2)
display(gcf())

# CELL 39
