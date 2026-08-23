using CUDA
using LinearAlgebra
using PulsePropagation
using Random

const PPCExt = Base.get_extension(
    PulsePropagation, :PulsePropagationCUDAExt)
PPCExt === nothing && error("PulsePropagation CUDA extension did not load.")
CUDA.functional() || error("CUDA is not functional.")

function physical_spatial_tensor(rng, nmodes::Int; rank::Int=5)
    factors = randn(rng, nmodes, rank)
    overlap = zeros(Float64, ntuple(_ -> nmodes, 4))
    for component in axes(factors, 2)
        vector = @view factors[:, component]
        overlap .+= reshape(vector, nmodes, 1, 1, 1) .*
                    reshape(vector, 1, nmodes, 1, 1) .*
                    reshape(vector, 1, 1, nmodes, 1) .*
                    reshape(vector, 1, 1, 1, nmodes)
    end
    return overlap
end

function compare_cpu_gpu(label, domain, overlap, field, cotangent)
    parameters = MMGNLSEParameters(
        domain;
        length=1e-5,
        beta=0.0,
        S=overlap,
        n2=2.3e-20,
        omega0=2π * 2.99792458e-4 / 1550e-9,
        raman=AnisotropicRaman(),
    )
    cpu_cache = PulsePropagation._mmgnlse_solver_cache(parameters)
    cpu = PulsePropagation._mmgnlse_vjp_nonlinear(
        field, cotangent, parameters, cpu_cache)

    gpu_cache = PPCExt._cuda_mmgnlse_forward_cache(parameters)
    gpu_field = CuArray(ComplexF64.(field))
    gpu_cotangent = CuArray(ComplexF64.(cotangent))
    workspace = PPCExt._cuda_mmgnlse_dense_raman_adjoint_workspace(
        gpu_field, gpu_cache.overlap)
    gpu_out = similar(gpu_field)
    PPCExt._cuda_mmgnlse_dense_vjp_nonlinear!(
        gpu_out, gpu_field, gpu_cotangent, gpu_cache, workspace)
    CUDA.synchronize()
    gpu = Array(gpu_out)

    difference = gpu .- cpu
    relative_l2 =
        norm(difference) / max(norm(cpu), eps(Float64))
    max_abs = maximum(abs, difference)
    println(
        label,
        " relative_l2_error=", relative_l2,
        " max_abs_error=", max_abs,
        " cpu_norm=", norm(cpu),
        " gpu_norm=", norm(gpu),
    )
    relative_l2 <= 5e-12 || error(
        "$label CPU/GPU VJP relative error $relative_l2 exceeds tolerance.")
    return parameters
end

rng = MersenneTwister(0xad1017)
nt = 32
nmodes = 3
scalar_domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space), TimeGrid(nt, 5.0))
scalar_field = randn(rng, ComplexF64, nt, nmodes, 1)
scalar_cotangent = randn(rng, ComplexF64, size(scalar_field))

scalar_parameters = compare_cpu_gpu(
    "scalar_physical_real",
    scalar_domain,
    physical_spatial_tensor(rng, nmodes),
    scalar_field,
    scalar_cotangent,
)
compare_cpu_gpu(
    "scalar_generic_complex",
    scalar_domain,
    randn(rng, ComplexF64, ntuple(_ -> nmodes, 4)),
    scalar_field,
    scalar_cotangent,
)

component_domain = MMGNLSEDomain(
    degrees_of_freedom(:time, :space, :polarization),
    TimeGrid(nt, 5.0),
)
component_field = randn(rng, ComplexF64, nt, nmodes, 2)
component_cotangent = randn(rng, ComplexF64, size(component_field))
component_overlap =
    randn(rng, ComplexF64, nmodes, 2, nmodes, 2, nmodes, 2, nmodes, 2)
compare_cpu_gpu(
    "component_generic_complex",
    component_domain,
    component_overlap,
    component_field,
    component_cotangent,
)

# Exercise the same kernel through the public forward/adjoint solver boundary.
forward = solve_mmgnlse(
    scalar_field, scalar_parameters, 1e-5; saveat=:steps)
terminal = randn(rng, ComplexF64, size(scalar_field))
cpu_adjoint = solve_adjoint(
    terminal, scalar_parameters, 1e-5;
    forward_solution=forward, saveat=:steps)
gpu_adjoint = solve_adjoint(
    terminal, scalar_parameters, 1e-5;
    forward_solution=forward, saveat=:steps, backend=:cuda)
public_difference = gpu_adjoint.fields .- cpu_adjoint.fields
public_relative_l2 =
    norm(public_difference) / max(norm(cpu_adjoint.fields), eps(Float64))
public_max_abs = maximum(abs, public_difference)
println(
    "public_scalar_adjoint",
    " relative_l2_error=", public_relative_l2,
    " max_abs_error=", public_max_abs,
)
public_relative_l2 <= 5e-11 || error(
    "Public CPU/GPU adjoint relative error $public_relative_l2 exceeds tolerance.")
