using Arpack
using CUDA
using FFTW
using LinearAlgebra
using PulsePropagation

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const C_STEP_COMPOSED = 2.99792458e-4
const LAMBDA0_STEP_COMPOSED = 1550e-9
const OMEGA0_STEP_COMPOSED =
    2pi * C_STEP_COMPOSED / LAMBDA0_STEP_COMPOSED

mutable struct StepComposedTransverseOperator <: AbstractMatrix{Float64}
    nx::Int
    ny::Int
    kinetic::Matrix{Float64}
    potential::Matrix{Float64}
    work::Matrix{ComplexF64}
end

Base.size(H::StepComposedTransverseOperator) = (H.nx * H.ny, H.nx * H.ny)
LinearAlgebra.issymmetric(::StepComposedTransverseOperator) = true

function LinearAlgebra.mul!(
    y::AbstractVector,
    H::StepComposedTransverseOperator,
    x::AbstractVector,
)
    copyto!(vec(H.work), x)
    ifft!(H.work)
    H.work .*= H.kinetic
    fft!(H.work)
    work = vec(H.work)
    potential = vec(H.potential)
    @inbounds for index in eachindex(x)
        y[index] = real(work[index]) + potential[index] * x[index]
    end
    return y
end

function step_composed_problem(; length_m=1e-3)
    time = TimeGrid(2^13, 100.0)
    transverse = TransverseGrid(Nx=64, Ny=64, dx=125e-6 / 64)
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), time;
        transverse_grid=transverse)
    grid = FullField3DGrid(time, transverse)

    fiber = StepIndex(
        lambda0=LAMBDA0_STEP_COMPOSED,
        core_radius=52.5e-6,
        NA=0.1)
    material = Silica()
    lambda = wavelength_axis(time, OMEGA0_STEP_COMPOSED)
    omega = OMEGA0_STEP_COMPOSED .+ frequency_axis(time)
    ncore0 = material_refractive_index(material, LAMBDA0_STEP_COMPOSED)
    nclad0 = sqrt(ncore0^2 - fiber.NA^2)
    ncore_omega = material_refractive_index.(Ref(material), lambda)
    nclad_omega = sqrt.(ncore_omega .^ 2 .- fiber.NA^2)
    beta_reference = SampledBeta(
        frequency_axis(time), nclad_omega .* omega ./ C_STEP_COMPOSED)

    index_yx = evaluate_index_profile(
        fiber, material, LAMBDA0_STEP_COMPOSED;
        x=transverse.x, y=transverse.y)
    potential = paraxial_index_potential(
        domain, index_yx;
        beta=beta_reference,
        omega0=OMEGA0_STEP_COMPOSED,
        reference_index=nclad0)

    nx, ny = length(transverse.x), length(transverse.y)
    beta_ref0 = nclad0 * OMEGA0_STEP_COMPOSED / C_STEP_COMPOSED
    kx = PulsePropagation._full_field_wavenumbers(nx, transverse.dx)
    ky = PulsePropagation._full_field_wavenumbers(ny, transverse.dy)
    kinetic = -(
        reshape(kx .^ 2, nx, 1) .+
        reshape(ky .^ 2, 1, ny)) ./ (2beta_ref0)
    operator = StepComposedTransverseOperator(
        nx, ny, Matrix(kinetic), Matrix(potential),
        zeros(ComplexF64, nx, ny))
    values, vectors, nconv, _, _, _ = Arpack.eigs(
        operator;
        nev=100,
        ncv=220,
        which=:LR,
        tol=1e-10,
        maxiter=5000)
    nconv == 100 || error("Only $nconv of 100 transverse modes converged.")
    order = sortperm(real.(values); rev=true)
    modes_xy = Array{Float64}(undef, nx, ny, 100)
    for mode in 1:100
        profile = reshape(copy(@view(vectors[:, order[mode]])), nx, ny)
        profile = real.(profile)
        profile ./= sqrt(sum(abs2, profile) * transverse.dx * transverse.dy)
        profile[argmax(abs.(profile))] < 0 && (profile .*= -1)
        modes_xy[:, :, mode] .= profile
    end
    modes_yx = permutedims(modes_xy, (2, 1, 3))
    phi = reshape(modes_yx, :, 100)
    gram = (transverse.dx * transverse.dy) .* (phi' * phi)

    coefficients = zeros(ComplexF64, 100)
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
    modal_field = gaussian_pulse(
        domain, 100;
        peak_power=751549.7920240875,
        fwhm=0.2499694427126389,
        coefficients)
    field_txy = reconstruct_full_field(
        @view(modal_field[:, :, 1]), modes_yx, grid)
    initial_field = reshape(field_txy, size(field_txy)..., 1)

    parameters = FullFieldMMGNLSEParameters(
        domain;
        length=length_m,
        beta=beta_reference,
        potential,
        n2=material.n2,
        omega0=OMEGA0_STEP_COMPOSED,
        raman=material.raman,
        self_steepening=true)
    return (; initial_field, parameters, grid, modes_yx, gram)
end

function run_step_composed(problem, dz, method)
    solution = nothing
    elapsed = @elapsed begin
        solution = solve_mmgnlse_full_field(
            problem.initial_field,
            problem.parameters,
            dz;
            method,
            precision=:float64,
            saveat=nothing,
            backend=:cuda,
            synchronize=true)
    end
    field_omega = Array(get_spectral_field(solution; z=:final))[:, :, :, 1]
    coefficients = project_full_field(
        field_omega, problem.modes_yx, problem.grid) /
        transpose(problem.gram)
    cache = solution.cache
    solution = nothing
    GC.gc()
    CUDA.reclaim()
    return (; field_omega, coefficients, elapsed, cache)
end

intensity_error(candidate, reference) =
    norm(abs2.(candidate) .- abs2.(reference)) / norm(abs2.(reference))

function phase_aligned_error(candidate, reference)
    phase = dot(vec(reference), vec(candidate))
    iszero(phase) || (phase /= abs(phase))
    return norm(vec(candidate .- phase .* reference)) / norm(vec(reference))
end

problem = step_composed_problem()
println("gpu=", CUDA.name(CUDA.device()))
println("shape=", size(problem.initial_field), " length_m=", problem.parameters.length)
println("gram_error=", norm(problem.gram - I, 2))

# Compile both paths on the real notebook shape without including JIT time.
warm_problem = merge(problem, (parameters=FullFieldMMGNLSEParameters(
    problem.parameters.domain;
    length=1e-6,
    beta=problem.parameters.beta,
    potential=problem.parameters.potential,
    n2=problem.parameters.n2,
    omega0=problem.parameters.omega0,
    raman=problem.parameters.raman,
    self_steepening=true),))
run_step_composed(warm_problem, 1e-6, StrangSplit())
run_step_composed(warm_problem, 1e-6, ComposedStrang4())

reference = run_step_composed(problem, 1.25e-6, StrangSplit())
println("reference method=strang dz_um=1.25 elapsed_s=", reference.elapsed)

for (method, dz) in (
    (StrangSplit(), 5e-6),
    (ComposedStrang4(), 40e-6),
    (ComposedStrang4(), 20e-6),
    (ComposedStrang4(), 10e-6),
    (ComposedStrang4(), 5e-6),
)
    result = run_step_composed(problem, dz, method)
    println(
        "method=", nameof(typeof(method)),
        " dz_um=", dz * 1e6,
        " elapsed_s=", result.elapsed,
        " modal_spectrum_error=",
        intensity_error(result.coefficients, reference.coefficients),
        " spectral_field_error=",
        phase_aligned_error(result.field_omega, reference.field_omega),
        " rhs_evaluations=", result.cache.residual_rhs_evaluations)
    result = nothing
    GC.gc()
end
