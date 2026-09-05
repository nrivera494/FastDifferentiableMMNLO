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
time_window = 100; # ps
λ0 = 1550e-9; # m
ω0 = 2*π*c_light/λ0;
nref = 1.444
beta0 = nref * ω0 / c_light # m^-1
beta2 = -0.0217 # ps^2/m, illustrative
beta = TaylorBeta([beta0, 0.0, beta2])

time_grid = TimeGrid(Nt,time_window);
t = time_axis(time_grid);
Ω = frequency_axis(time_grid); # constructs relative frequency
ω = ω0 .+ Ω; # absolute frequency;
λ = wavelength_axis(time_grid,ω0); # wavelength
λ_nm = 1e9 * λ;

num_grid_points_1d = 64;
L_sim = 125e-6;
Δx = L_sim / num_grid_points_1d;
xy_grid = TransverseGrid(Nx=num_grid_points_1d , Ny=num_grid_points_1d , dx=Δx, dy=Δx)

dofs = degrees_of_freedom(:time, :space)
domain = MMGNLSEDomain(dofs, time_grid; transverse_grid=xy_grid);

# CELL 4
fiber = StepIndex(;
    lambda0 = λ0,
    core_radius = 52.5e-6,
    NA = 0.275,
)

material = Silica()

ncore0 = material_refractive_index(material, λ0)
nclad0 = sqrt(ncore0^2 - fiber.NA^2)

ncore_ω = material_refractive_index.(Ref(material), λ)
nclad_ω = sqrt.(ncore_ω.^2 .- fiber.NA^2)

beta_reference = SampledBeta(
    Ω,
    nclad_ω .* ω ./ c_light,
)

# evaluate_index_profile returns (y,x); the helper converts to (x,y).
index_yx = evaluate_index_profile(
    fiber,
    material,
    λ0;
    x = xy_grid.x,
    y = xy_grid.y,
)

potential = paraxial_index_potential(
    domain,
    index_yx;
    beta = beta_reference,
    omega0 = ω0,
    reference_index = nclad0,
);

# CELL 5
# Matrix-free representation of the exact transverse operator used by
# the full-field solver: spectral diffraction plus sampled potential.
mutable struct FFTTransverseOperator <: AbstractMatrix{Float64}
    nx::Int
    ny::Int
    kinetic::Matrix{Float64}
    potential::Matrix{Float64}
    work::Matrix{ComplexF64}
end

Base.size(H::FFTTransverseOperator) = (H.nx * H.ny, H.nx * H.ny)
LinearAlgebra.issymmetric(::FFTTransverseOperator) = true

function LinearAlgebra.mul!(
    y::AbstractVector,
    H::FFTTransverseOperator,
    x::AbstractVector,
)
    copyto!(vec(H.work), x)

    ifft!(H.work)
    H.work .*= H.kinetic
    fft!(H.work)

    w = vec(H.work)
    V = vec(H.potential)

    @inbounds for j in eachindex(x)
        y[j] = real(w[j]) + V[j] * x[j]
    end

    return y
end

# ----------------------------------------------------------------------
# Discrete transverse eigensystem
# ----------------------------------------------------------------------

Nm = 150
nx, ny = length(xy_grid.x), length(xy_grid.y)

# Must be exactly the reference beta used for full-field diffraction.
βref0 = nclad0 * ω0 / c_light

kx = PulsePropagation._full_field_wavenumbers(nx, xy_grid.dx)
ky = PulsePropagation._full_field_wavenumbers(ny, xy_grid.dy)

kinetic = -(
    reshape(kx.^2, nx, 1) .+
    reshape(ky.^2, 1, ny)
) ./ (2βref0)

Hxy = FFTTransverseOperator(
    nx,
    ny,
    Matrix(kinetic),
    Matrix(potential),
    zeros(ComplexF64, nx, ny),
)

λraw, Uraw, nconv, _, _, _ = Arpack.eigs(
    Hxy;
    nev=Nm,
    ncv=min(nx * ny, max(2Nm + 20, 220)),
    which=:LR,
    tol=1e-10,
    maxiter=5000,
)

@assert nconv == Nm

order = sortperm(real.(λraw); rev=true)
λperp = real.(λraw[order])       # transverse eigenvalues, m^-1
U = real.(Uraw[:, order])

# Positive eigenvalues correspond to modes guided relative to the
# cladding-reference potential.
@show extrema(λperp)
@assert minimum(λperp) > 0

# Convert (point,mode) eigenvectors to the package's (y,x,mode) convention.
modes_xym = Array{Float64}(undef, nx, ny, Nm)

for m in 1:Nm
    u = reshape(copy(@view(U[:, m])), nx, ny)

    # Power normalization: ∫|u|² dxdy = 1.
    u ./= sqrt(sum(abs2, u) * xy_grid.dx * xy_grid.dy)

    # Deterministic real sign convention.
    pivot = argmax(abs.(u))
    u[pivot] < 0 && (u .*= -1)

    modes_xym[:, :, m] .= u
end

modes_discrete = permutedims(modes_xym, (2, 1, 3))

# These modes should be orthonormal under the same rectangle rule used
# for reconstruction and projection.
Φ = reshape(modes_discrete, :, Nm)
gram = (xy_grid.dx * xy_grid.dy) .* (Φ' * Φ)

@show norm(gram - I, 2)
@show cond(gram)

# ----------------------------------------------------------------------
# Matched modal dispersion
# ----------------------------------------------------------------------

βref_samples = vec(beta_reference.values)

β_discrete = SampledBeta(
    beta_reference.omega,
    reshape(βref_samples, :, 1) .+ reshape(λperp, 1, :),
)

# CELL 6
# ----------------------------------------------------------------------
# Dense overlap and CP compression
# ----------------------------------------------------------------------

# About 0.8 GB for 100^4 Float64 entries.
S_discrete_dense = PulsePropagation._spatial_overlap(
    modes_discrete,
    xy_grid.x,
    xy_grid.y;
    chunk_points=4096,
    max_workspace_bytes=512 * 1024^2,
    precision=Float32,
    max_tensor_bytes=10 * 1024^3,
)

S_compressed_1em2 = cp_compress(
    S_discrete_dense;
    error=1e-2,
    backend=:cuda,
    min_rank=100,
    rank_step=100,
    max_rank=2000,
    maxiter=50, #120
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

S_compressed_1em4 = cp_compress(
    S_discrete_dense;
    error=1e-4,
    backend=:cuda,
    min_rank=100,
    rank_step=100,
    max_rank=2000,
    maxiter=50, #120
    tolerance=1e-6,
    ridge=1e-8,
    check_every=5,
)

# CELL 7
# ----------------------------------------------------------------------
# Common launch for both solvers
# ----------------------------------------------------------------------

num_modes = 150;
#properties = compute_fiber_properties(fiber, material, num_modes, polarization=:scalar, mode_basis=:LP, beta_order=3)
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

# modal field
initial_field = gaussian_pulse(
        domain,
        num_modes;
        peak_power=751549.7920240875,    # total peak power across all modes, W
        fwhm=0.2499694427126389,          # intensity FWHM, ps
        time_offset=0,  # ps
        coefficients=coefficients,
    )

# Reuse the existing coefficient array, including the modes 6:15 launch.
initial_field_discrete = copy(initial_field)

grid = FullField3DGrid(
    time_grid,
    xy_grid.x,
    xy_grid.y,
)

A_txy = reconstruct_full_field(
    initial_field_discrete[:, :, 1],
    modes_discrete,
    grid,
)

initial_field_full = reshape(A_txy, size(A_txy)..., 1)

# Use modes_discrete, not the analytic step-index modes, for projection.
modes_target = modes_discrete;

# CELL 8
# dz are slightly different to get to comparable accuracy
dz_cp_1em2 = 200e-6;
dz_cp_1em4 = 240e-6;
length_m = 0.25;

parameters_compressed_1em2 = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=β_discrete,
    S=S_compressed_1em2,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp_1em2 = solve_mmgnlse(initial_field, parameters_compressed_1em2, dz_cp_1em2; backend=:cuda_cp_optimized, precision=:float64,
    adaptive=false)

# floor = 6.8,6.7,7.1,6.8,6.7,6.8 -> 6.82 \pm 0.15
parameters_compressed_1em4 = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=β_discrete,
    S=S_compressed_1em4,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp_1em4 = solve_mmgnlse(initial_field, parameters_compressed_1em4, dz_cp_1em4; backend=:cuda_cp_optimized, precision=:float64,
    adaptive=false)

# CELL 9
dz_dense = 200e-6;

parameters_dense = MMGNLSEParameters(
    domain;
    length=length_m,
    beta=β_discrete,
    S=S_discrete_dense,
    n2=material.n2,        # use 0 initially for the linear validation
    omega0=ω0,
    raman=material.raman,
)

solution_cp_dense = solve_mmgnlse(initial_field, parameters_dense, dz_dense; backend=:cuda, precision=:float64,
    adaptive=false)

# CELL 10
dz_ff = 1.25e-6;

parameters_ff = FullFieldMMGNLSEParameters(
domain;
length=length_m, # m
beta=beta_reference,
potential=potential, # axes (x, y), m^-1
n2=material.n2, # m^2/W
omega0=ω0,
raman=material.raman,
self_steepening=true,
)

solution_ff = solve_mmgnlse_full_field(
initial_field_full,
parameters_ff,
dz_ff;
method=StrangSplit(),
precision=:float64,
saveat=nothing,
backend=:cuda,
device=nothing,
synchronize=true,
)

# CELL 11
# CP-modal result: (frequency, mode, polarization)
Aω_modal_3d_1em2 = Array(get_spectral_field(
    solution_cp_1em2;
    z=:final,
))

Aω_modal_3d_1em4 = Array(get_spectral_field(
    solution_cp_1em4;
    z=:final,
))

Aω_modal_3d_dense = Array(get_spectral_field(
    solution_cp_dense;
    z=:final,
))

# Full-field result: (frequency, x, y, polarization)
Aω_full_4d = Array(get_spectral_field(
    solution_ff;
    z=:final,
))


Aω_modal_1em2 = @view Aω_modal_3d_1em2[:, :, 1];
Aω_modal_1em4 = @view Aω_modal_3d_1em4[:, :, 1];
Aω_modal_dense = @view Aω_modal_3d_dense[:, :, 1];
Aω_full_xy = @view Aω_full_4d[:, :, :, 1];

Aω_full_projected = project_full_field(
    Aω_full_xy,
    modes_target,
    grid,
)

Aω_full_coefficients =
    Aω_full_projected / transpose(gram);

spectral_scale = time_grid.nt * time_grid.dt / 1e3

Eωm_modal_1em2 = spectral_scale .* abs2.(Aω_modal_1em2) 
Eωm_modal_1em4 = spectral_scale .* abs2.(Aω_modal_1em4)             # nJ/bin
Eωm_modal_dense = spectral_scale .* abs2.(Aω_modal_dense) 
Eωm_full  = spectral_scale .* abs2.(Aω_full_coefficients)    # nJ/bin


modal_spectrum_error_cp1em2_dense =
    norm(Eωm_modal_dense - Eωm_modal_1em2) /
    norm(Eωm_modal_dense)

modal_spectrum_error_cp1em4_dense =
    norm(Eωm_modal_dense - Eωm_modal_1em4) /
    norm(Eωm_modal_dense)

modal_spectrum_error_full_dense =
    norm(Eωm_modal_dense - Eωm_full) /
    norm(Eωm_modal_dense)


@show modal_spectrum_error_cp1em2_dense
@show modal_spectrum_error_cp1em4_dense
@show modal_spectrum_error_full_dense

# CELL 12
plot(λ_nm,abs2.(Aω_modal_1em2),color=:red,label=:false)
plot!(λ_nm,abs2.(Aω_modal_1em4),color=:blue,linestyle=:dash,label=:false,alpha=0.5)
plot!(λ_nm,abs2.(Aω_modal_dense),color=:green,linestyle=:dash,label=:false,alpha=0.5)
plot!(λ_nm,abs2.(Aω_full_projected),color=:black,linestyle=:dash,label=:false,alpha=0.5)
xlims!(1450,1700)
# ylims!(0.9,1.1)

# CELL 13
npzwrite("cpff_wavelengths.npy", λ_nm)
npzwrite("cpff_spectral_fields_0p01.npy", Aω_modal_1em2)
npzwrite("cpff_spectral_fields_0p0001.npy", Aω_modal_1em4)
npzwrite("cpff_spectral_fields_dense.npy", Aω_modal_dense)
npzwrite("cpff_spectral_fields_ff.npy",Aω_full_projected)