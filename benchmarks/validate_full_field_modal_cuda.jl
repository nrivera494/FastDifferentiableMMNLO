using CUDA
using FFTW
using LinearAlgebra
using PulsePropagation
using Test

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const C_M_PER_PS = 2.99792458e-4
const OMEGA0 = 2pi * C_M_PER_PS / 1.55e-6

"""Lowest sampled modes of the same spectral transverse operator as the solver."""
function sampled_transverse_modes(grid, beta0, potential, mode_count)
    nx, ny = length(grid.x), length(grid.y)
    point_count = nx * ny
    kx = PulsePropagation._full_field_wavenumbers(nx, grid.dx)
    ky = PulsePropagation._full_field_wavenumbers(ny, grid.dy)
    kinetic = -(
        reshape(kx .^ 2, nx, 1) .+ reshape(ky .^ 2, 1, ny)) ./ (2beta0)

    operator = Matrix{ComplexF64}(undef, point_count, point_count)
    basis = zeros(ComplexF64, nx, ny)
    for column in 1:point_count
        fill!(basis, 0)
        basis[column] = 1
        spectral = ifft(basis, (1, 2))
        spectral .*= kinetic
        @views operator[:, column] .= vec(
            fft(spectral, (1, 2)) .+ potential .* basis)
    end
    decomposition = eigen(Hermitian((operator + operator') / 2))
    selected = sortperm(decomposition.values; rev=true)[1:mode_count]
    beta = decomposition.values[selected]

    modes_xym = Array{Float64}(undef, nx, ny, mode_count)
    for mode in 1:mode_count
        vector = decomposition.vectors[:, selected[mode]]
        pivot = argmax(abs.(vector))
        vector .*= cis(-angle(vector[pivot]))
        real_vector = real.(vector)
        real_vector ./= sqrt(sum(abs2, real_vector) * grid.dx * grid.dy)
        @views modes_xym[:, :, mode] .= reshape(real_vector, nx, ny)
    end
    modes_yxm = permutedims(modes_xym, (2, 1, 3))
    return beta, modes_yxm
end

function sampled_overlap(modes_yxm, grid)
    mode_count = size(modes_yxm, 3)
    modes = reshape(permutedims(modes_yxm, (2, 1, 3)), :, mode_count)
    overlap = Array{Float64}(undef, ntuple(_ -> mode_count, 4))
    weight = grid.dx * grid.dy
    @inbounds for q in 1:mode_count, p in 1:mode_count,
                  n in 1:mode_count, m in 1:mode_count
        overlap[m, n, p, q] = weight * sum(
            modes[:, m] .* modes[:, n] .* modes[:, p] .* modes[:, q])
    end
    return overlap
end

function modal_full_field_case(n2)
    time = TimeGrid(8, 2.0)
    grid = FullField3DGrid(time; Nx=24, Ny=24, dx=1.5e-6)
    beta0 = 8.0e6
    oscillator_rate = 2.0e3
    potential = [
        -0.5 * beta0 * oscillator_rate^2 * (x^2 + y^2)
        for x in grid.x, y in grid.y
    ]
    mode_count = 3
    beta, modes = sampled_transverse_modes(
        grid, beta0, potential, mode_count)
    overlap = sampled_overlap(modes, grid)

    powers = 1.0e6 .* [0.55, 0.30, 0.15]
    phases = [0.0, 0.37, -0.61]
    modal_trace = sqrt.(powers) .* cis.(phases)
    modal_initial = repeat(reshape(modal_trace, 1, mode_count), time.nt, 1)
    full_initial = reconstruct_full_field(modal_initial, modes, grid)

    propagation_length = 2.0e-4
    dz = 2.0e-6
    full_parameters = FullField3DParameters(
        grid;
        length=propagation_length,
        beta0,
        potential,
        n2,
        omega0=OMEGA0,
    )
    modal_domain = MMGNLSEDomain(:time, :space; time_grid=time)
    modal_parameters = MMGNLSEParameters(
        modal_domain;
        length=propagation_length,
        beta=TaylorBeta(reshape(beta, 1, mode_count)),
        S=SpatialOverlap(overlap),
        n2,
        omega0=OMEGA0,
    )

    full_solution = solve_full_field_3d(
        full_initial, full_parameters, dz; precision=:float64)
    modal_solution = solve_mmgnlse(
        reshape(modal_initial, time.nt, mode_count, 1),
        modal_parameters, dz;
        backend=:cuda, precision=:float64)

    full_final = @view full_solution.fields[:, :, :, end]
    projected = project_full_field(full_final, modes, grid)
    modal_final = Array(get_temporal_field(modal_solution; z=:final))[:, :, 1]
    error = phase_invariant_relative_error(projected, modal_final)
    leakage = full_field_subspace_leakage(full_final, modes, grid)
    power_drift = maximum(abs.(
        full_field_power(full_final, grid) ./
        full_field_power(full_initial, grid) .- 1))
    return (; error, leakage, power_drift, beta)
end

@testset "Matched modal/full-field CUDA comparison" begin
    linear = modal_full_field_case(0.0)
    @test linear.error < 3e-5
    @test linear.leakage < 3e-7
    @test linear.power_drift < 3e-11

    nonlinear = modal_full_field_case(2.3e-20)
    @test nonlinear.error < 3e-3
    @test nonlinear.leakage < 2e-3
    @test nonlinear.power_drift < 3e-11

    @info "matched comparison" linear nonlinear
end

println("validate_full_field_modal_cuda: PASS gpu=", CUDA.name(CUDA.device()))
