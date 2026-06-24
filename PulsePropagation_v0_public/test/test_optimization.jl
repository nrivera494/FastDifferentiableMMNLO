using Test
using LinearAlgebra
import Optim
using .PulsePropagation

@testset "variance objective and optimization" begin
    grid = TimeGrid(8, 8.0)
    dofs = ModalField(1:2)
    system = PassiveFiber(; length=0.001,
                          lambda0=1550e-9,
                          material=Silica(),
                          betas=zeros(Float64, 3, 2),
                          sr=zeros(Float64, 2, 2, 2, 2),
                          n2=0.0,
                          raman_fraction=0.0)
    terms = PropagationTerms(Dispersion(), Kerr())
    solver = RK4IPSolver(; stepping=FixedStep(; dz=0.001),
                         saveat=SaveAt([0.0, 0.001]),
                         reltol=1e-4,
                         abstol=1e-4)
    pulse = sech_pulse(grid; peak_power=1.0, fwhm=0.5,
                       dofs=dofs, coefficients=ComplexF64[1, 0])

    build_problem = θ -> begin
        coeffs = ComplexF64[cos(θ[1]), sin(θ[1])]
        fields = sech_pulse(grid; peak_power=1.0, fwhm=0.5,
                            dofs=dofs, coefficients=coeffs)
        PulsePropagationProblem(; grid, system, dofs, terms, solver, fields)
    end

    p0 = build_problem([0.2])
    sol0 = solve(p0)
    obs = FilterEnergy(; filter=ones(grid.nt), domain=:frequency, modes=[1])
    grad = gradient(p0, obs; trajectory=sol0, method=PulsePropagation.Adjoint(),
                    normalization=PowerNormalized())
    var_obj = VarianceObjective(obs; method=PulsePropagation.Adjoint(),
                                normalization=PowerNormalized())
    vr = variance_result(var_obj, p0, sol0)
    @test vr.value ≈ value(obs, sol0)
    @test vr.variance ≈ sum(abs2, grad.initial_field_gradient)
    @test value(var_obj, p0, sol0) ≈ vr.variance

    uω0 = PulsePropagation.inverse_fft(p0.initial_state.fields[:, :, 1], dims=1)
    direction = randn(ComplexF64, size(uω0))
    direction ./= norm(direction)
    dvar = variance_directional_derivative(var_obj, p0, direction;
                                           trajectory=sol0,
                                           reltol=1e-4,
                                           abstol=1e-4,
                                           jvp_eps=1e-4)
    ε = 1e-6
    problem_from_uω = uω -> begin
        pert_fields = reshape(PulsePropagation.forward_fft(uω, dims=1),
                              size(uω, 1), size(uω, 2), 1)
        PulsePropagationProblem(; grid, system, dofs, terms, solver,
                                fields=pert_fields)
    end
    vp = value(var_obj, problem_from_uω(uω0 .+ ε .* direction))
    vm = value(var_obj, problem_from_uω(uω0 .- ε .* direction))
    @test dvar.directional_derivative ≈ (vp - vm) / (2ε) rtol=5e-2 atol=5e-2

    fd_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem,
        objective=obs,
        sense=:maximize,
        gradient=FiniteDifferenceGradient(; relstep=1e-4),
    )
    @test isfinite(optimization_value(fd_opt))
    @test length(optimization_gradient(fd_opt)) == 1

    phase_pullback = (θ, problem, sol, field_gradient, J) -> begin
        u0 = problem.initial_state.fields[:, :, 1]
        du = similar(u0, ComplexF64)
        du[:, 1] .= -sin(θ[1]) .* pulse[:, 1, 1]
        du[:, 2] .= cos(θ[1]) .* pulse[:, 1, 1]
        duω = PulsePropagation.inverse_fft(du, dims=1)
        [2real(sum(conj.(field_gradient) .* duω))]
    end
    adj_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem,
        objective=obs,
        sense=:maximize,
        gradient=AdjointGradient(; normalization=PowerNormalized()),
        pullback=phase_pullback,
    )
    gadj = optimization_gradient(adj_opt)
    gfd = optimization_gradient(fd_opt)
    @test gadj[1] ≈ gfd[1] rtol=5e-3 atol=5e-3

    variance_fd_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem,
        objective=var_obj,
        sense=:minimize,
        gradient=FiniteDifferenceGradient(; relstep=1e-4),
    )
    variance_pushforward = (θ, problem, sol) -> begin
        du = similar(problem.initial_state.fields[:, :, 1], ComplexF64)
        du[:, 1] .= -sin(θ[1]) .* pulse[:, 1, 1]
        du[:, 2] .= cos(θ[1]) .* pulse[:, 1, 1]
        [PulsePropagation.inverse_fft(du, dims=1)]
    end
    variance_adj_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem,
        objective=var_obj,
        sense=:minimize,
        gradient=AdjointGradient(; normalization=PowerNormalized(),
                                 reltol=1e-4,
                                 abstol=1e-4,
                                 jvp_eps=1e-4),
        pushforward=variance_pushforward,
    )
    gv_adj = optimization_gradient(variance_adj_opt)
    gv_fd = optimization_gradient(variance_fd_opt)
    @test gv_adj[1] ≈ gv_fd[1] rtol=5e-2 atol=5e-2

    reg = DifferentiableRegularizer(; value=θ -> sum(abs2, θ),
                                    gradient=θ -> 2 .* θ)
    reg_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem,
        objective=obs,
        sense=:maximize,
        gradient=FiniteDifferenceGradient(),
        regularizers=[reg],
    )
    @test optimization_loss(reg_opt) ≈ -optimization_value(reg_opt) + 0.04

    result = optimize(fd_opt, CoordinateSearch(; initial_step=0.1,
                                               maxiters=2,
                                               tol=1e-4))
    @test result.iterations >= 1
    @test isfinite(result.objective_value)

    solve_count = Ref(0)
    counted_build_problem = θ -> begin
        solve_count[] += 1
        build_problem(θ)
    end
    counted_opt = PropagationOptimizationProblem(;
        parameters=[0.2],
        build_problem=counted_build_problem,
        objective=obs,
        sense=:maximize,
        gradient=AdjointGradient(; normalization=PowerNormalized()),
        pullback=phase_pullback,
    )
    counted_result = optimize(counted_opt, GradientDescent(; stepsize=0.0,
                                                           maxiters=1,
                                                           tol=Inf))
    @test counted_result.iterations == 1
    @test solve_count[] == 1

    lbfgs_result = optimize(adj_opt, Optim.LBFGS();
                            options=Optim.Options(iterations=1))
    @test lbfgs_result.iterations >= 0
    @test isfinite(lbfgs_result.objective_value)
    @test lbfgs_result.history.optim_result isa Optim.MultivariateOptimizationResults
end
