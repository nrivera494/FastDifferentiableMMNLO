using .PulsePropagation
using LinearAlgebra
using Random
using Statistics

@testset "Analytic-GRIN multimode stochastic variance" begin
    folder = abspath(joinpath(@__DIR__, "..", "GRIN_62.5um_wavelength1550nm_csv"))
    nm = 6
    L = 1e-4
    grid = TimeGrid(; Nt=48, window=4.0)
    system, dofs = load_fiber_system(folder; modes=1:nm, length=L,
                                     lambda0=1.55e-6)
    fields = gaussian_pulse(grid; fwhm=0.35, total_energy=2.0,
                            dofs=dofs,
                            coefficients=ones(ComplexF64, nm) ./ sqrt(nm))
    solver = RK4IPSolver(; stepping=FixedStep(; dz=L),
                         saveat=SaveAt([0.0, L]),
                         reltol=1e-8, abstol=1e-8)
    problem = PulsePropagationProblem(; grid, system, dofs, fields, solver,
                                      terms=PropagationTerms(Dispersion(), Kerr()))
    sol = solve(problem)

    obs = SpectralPhotonNumber(; filter=ones(grid.nt), modes=4:6,
                               shifted=false)
    adj = variance_result(
        VarianceObjective(obs; method=PulsePropagation.Adjoint(),
                          normalization=PhotonNormalized(),
                          raman=:off,
                          reltol=1e-7,
                          abstol=1e-7,
                          return_lambdaw_zsave=false),
        problem,
        sol,
    )

    ntraj = 24
    ensemble = stochastic_ensemble(problem; noise=ShotNoise(; noise_scale=0.5),
                                   ntraj=ntraj,
                                   rng=MersenneTwister(1234))
    st = solve_stochastic(problem, ensemble; reltol=1e-7, abstol=1e-7)
    samples = stochastic_values(obs, st)
    mc_mean = mean(samples)
    mc_variance = var(samples; corrected=true)
    variance_stderr = sqrt(2 / (ntraj - 1)) * adj.variance

    @test size(ensemble.uω) == (grid.nt, nm, ntraj)
    @test size(st.fields) == (grid.nt, nm, 2, ntraj)
    @test isapprox(mc_mean, adj.value; rtol=5e-6, atol=0)
    @test abs(mc_variance - adj.variance) <= 4 * variance_stderr
    @test isapprox(adj.fano, 1.0; rtol=1e-6, atol=1e-6)
end

@testset "Vector multimode stochastic smoke" begin
    folder = abspath(joinpath(@__DIR__, "..", "GRIN_62.5um_wavelength1550nm_csv"))
    system, dofs = load_fiber_system(folder; modes=1:2, length=1e-5,
                                     lambda0=1.55e-6, scalar=false)
    grid = TimeGrid(; Nt=16, window=2.0)
    fields = gaussian_pulse(grid; fwhm=0.4, total_energy=0.1,
                            dofs=dofs,
                            coefficients=[1.0, 0.2, 0.3im, -0.1])
    solver = RK4IPSolver(; stepping=FixedStep(; dz=1e-5),
                         saveat=SaveAt([0.0, 1e-5]))
    problem = PulsePropagationProblem(; grid, system, dofs, fields, solver,
                                      terms=PropagationTerms(Dispersion(), Kerr()))
    ensemble = stochastic_ensemble(problem; noise=ShotNoise(; noise_scale=0.5),
                                   ntraj=3,
                                   rng=MersenneTwister(5678))
    st = solve_stochastic(problem, ensemble; reltol=1e-7, abstol=1e-7)

    @test size(ensemble.uω) == (grid.nt, 4, 3)
    @test size(st.fields) == (grid.nt, 4, 2, 3)
end
