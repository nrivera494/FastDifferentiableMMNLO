using .PulsePropagation
using LinearAlgebra

@testset "Native public backend" begin
    grid = TimeGrid(; Nt=32, window=4.0)
    fields = gaussian_pulse(grid; fwhm=0.5, peak_power=1.0)
    system = PassiveFiber(; length=1e-4, lambda0=1.55e-6,
                          betas=zeros(3, 1),
                          sr=reshape([1.0], 1, 1, 1, 1),
                          n2=1e-22)
    solver = RK4IPSolver(; stepping=FixedStep(; dz=1e-4),
                         saveat=SaveAt([0.0, 1e-4]))
    problem = PulsePropagationProblem(; grid, system, dofs=SingleMode(),
                                      fields, solver)
    sol = solve(problem)
    @test size(sol.output.fields) == (32, 1, 2)

    obs = SpectralPhotonNumber(; filter=ones(grid.nt))
    g_adj = gradient(problem, obs; method=PulsePropagation.Adjoint())
    g_ad = gradient(problem, obs; method=AutomaticDifferentiation())
    @test isapprox(g_adj.value, g_ad.value; rtol=1e-10, atol=1e-8)
    @test size(g_adj.initial_field_gradient) == (32, 1)

    dofs2 = ModalField(1:2)
    fields2 = gaussian_pulse(grid; fwhm=0.5, peak_power=1.0, dofs=dofs2,
                             coefficients=[1.0, 0.2])
    system2 = PassiveFiber(; length=1e-4, lambda0=1.55e-6,
                           betas=zeros(3, 2),
                           sr=fill(0.1, 2, 2, 2, 2),
                           n2=1e-22)
    problem2 = PulsePropagationProblem(; grid, system=system2, dofs=dofs2,
                                       fields=fields2, solver)
    ensemble = stochastic_ensemble(problem2; ntraj=4,
                                   noise=ShotNoise(; noise_scale=0.5))
    st = solve_stochastic(problem2, ensemble)
    @test size(ensemble.uω) == (32, 2, 4)
    @test size(st.fields) == (32, 2, 2, 4)

    cp_solver = RK4IPSolver(;
        stepping=FixedStep(; dz=1e-4),
        saveat=SaveAt([0.0, 1e-4]),
        compression=CPCompression(; target_error=1e-7, initial_rank=1,
                                  max_rank=2, maxiter=10),
        cp_rhs=FixedRK4IPCPRHS(; fftw_flags=:ESTIMATE),
    )
    cp_problem = PulsePropagationProblem(; grid, system=system2, dofs=dofs2,
                                         fields=fields2, solver=cp_solver)
    cp_sol = solve(cp_problem)
    @test size(cp_sol.output.fields) == (32, 2, 2)
    @test compression_rank(cp_sol) >= 1
    @test compression_error(cp_sol) <= 1e-6

    folder = abspath(joinpath(@__DIR__, "..", "GRIN_62.5um_wavelength1550nm_csv"))
    csv_system, csv_dofs = load_fiber_system(folder; modes=1:2,
                                             length=0.01,
                                             lambda0=1.55e-6)
    @test csv_dofs isa ModalBasis
    @test size(PulsePropagation.betas(csv_system.dispersion)) == (6, 2)
    @test size(PulsePropagation.sr(csv_system.geometry)) == (2, 2, 2, 2)
end
