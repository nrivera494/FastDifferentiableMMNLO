using Test
using LinearAlgebra
using Random
using FFTW
using .PulsePropagation

function _check_terminal(obs, u; atol=1e-6, rtol=1e-6)
    rng = MersenneTwister(7)
    δ = randn(rng, ComplexF64, size(u))
    ε = 1e-7
    fd = (value(obs, u .+ ε .* δ) - value(obs, u .- ε .* δ)) / (2ε)
    g = terminal_condition(obs, u)
    lin = 2real(sum(conj.(g) .* δ))
    @test fd ≈ lin atol=atol rtol=rtol
end

@testset "observable terminal conditions" begin
    rng = MersenneTwister(11)
    nt = 16
    nm = 3
    u = randn(rng, ComplexF64, nt, nm)

    filt = rand(rng, nt, nm)
    _check_terminal(FilterEnergy(; filter=filt, domain=:frequency), u)

    _check_terminal(FilterEnergy(; filter=rand(rng, nt), domain=:frequency,
                                 modes=[1, 3]), u)

    _check_terminal(RatioObservable(
        FilterEnergy(; filter=rand(rng, nt), domain=:frequency, modes=[1]),
        FilterEnergy(; filter=ones(nt), domain=:frequency, modes=:all)), u)

    _check_terminal(BinEnergy(; bin=5, mode=2, domain=:frequency), u)

    q = randn(rng, ComplexF64, length(u))
    _check_terminal(QuadratureObservable(; M=q, domain=:frequency), u)

    M = randn(rng, ComplexF64, 4, length(u))
    w = randn(rng, 4)
    _check_terminal(QuadratureObservable(; M=M, weights=w,
                                         domain=:frequency), u)

    Q, _ = qr(randn(rng, ComplexF64, nm, nm))
    _check_terminal(ProjectedEnergy(; U=Matrix(Q), domain=:frequency), u)

    nx = 5
    U = randn(rng, ComplexF64, nx, nm)
    photon_weights = rand(rng, nt)
    spatial_filter = rand(rng, nx)
    projected_photon = ProjectedPhotonNumber(; U, weights=photon_weights,
                                             filter=spatial_filter,
                                             domain=:frequency)
    _check_terminal(projected_photon, u)
    manual_filter = reshape(photon_weights, nt, 1) .* reshape(spatial_filter, 1, nx)
    manual_projected = ProjectedEnergy(; U, filter=manual_filter,
                                       domain=:frequency)
    @test value(projected_photon, u) ≈ value(manual_projected, u)
    @test terminal_condition(projected_photon, u) ≈ terminal_condition(manual_projected, u)

    identity_projected = ProjectedPhotonNumber(; U=Matrix{Float64}(I, nm, nm),
                                               weights=photon_weights,
                                               filter=ones(nm),
                                               domain=:frequency)
    identity_modal = FilterEnergy(; filter=photon_weights,
                                  domain=:frequency,
                                  modes=:all)
    @test value(identity_projected, u) ≈ value(identity_modal, u)
    @test terminal_condition(identity_projected, u) ≈ terminal_condition(identity_modal, u)

    axis = collect(range(-2.0, 2.0; length=nt))
    obs_t = TemporalMoment(; order=1, center=0.0, normalized=true, modes=[2])
    δ = randn(rng, ComplexF64, size(u))
    ε = 1e-7
    fd = (value(obs_t, u .+ ε .* δ; axis=axis) -
          value(obs_t, u .- ε .* δ; axis=axis)) / (2ε)
    g = terminal_condition(obs_t, u; axis=axis)
    lin = 2real(sum(conj.(g) .* δ))
    @test fd ≈ lin atol=1e-6 rtol=1e-6
end

function _identity_problem(fields)
    nt, nm, _ = size(fields)
    grid = TimeGrid(nt, 8.0)
    dofs = ModalField(1:nm)
    system = PassiveFiber(; length=0.01,
                          lambda0=1550e-9,
                          material=Silica(),
                          betas=zeros(Float64, 3, nm),
                          sr=zeros(Float64, nm, nm, nm, nm),
                          n2=0.0,
                          raman_fraction=0.0)
    terms = PropagationTerms(Dispersion(), Kerr())
    solver = RK4IPSolver(; stepping=FixedStep(; dz=0.01),
                         saveat=SaveAt([0.0, 0.01]))
    return PulsePropagationProblem(; grid, system, dofs, terms, solver, fields)
end

function _linear_gain_problem(fields; gain=0.7, length=1.0)
    nt, nm, _ = size(fields)
    grid = TimeGrid(nt, 8.0)
    dofs = ModalField(1:nm)
    system = PassiveFiber(; length,
                          lambda0=1550e-9,
                          material=Silica(),
                          betas=zeros(Float64, 3, nm),
                          sr=zeros(Float64, nm, nm, nm, nm),
                          n2=0.0,
                          raman_fraction=0.0)
    terms = PropagationTerms(Dispersion(), Kerr())
    solver = RK4IPSolver(; stepping=FixedStep(; dz=length),
                         saveat=SaveAt([0.0, length]),
                         linear_gain=gain)
    return PulsePropagationProblem(; grid, system, dofs, terms, solver, fields)
end

function _check_ad_initial_gradient(obs, fields; atol=1e-5, rtol=1e-5)
    rng = MersenneTwister(17)
    δ = randn(rng, ComplexF64, size(fields))
    ε = 1e-7
    p0 = _identity_problem(fields)
    ad = gradient(p0, obs; method=AutomaticDifferentiation(),
                  normalization=PowerNormalized())
    objective(f) = value(obs, solve(_identity_problem(f)))
    fd = (objective(fields .+ ε .* δ) - objective(fields .- ε .* δ)) / (2ε)
    lin = 2real(sum(conj.(ad.initial_field_gradient) .* δ[:, :, 1]))
    @test fd ≈ lin atol=atol rtol=rtol
    @test ad.method isa AutomaticDifferentiation
    @test isfinite(ad.value)
end

function _check_adjoint_fourier_gradient(obs, fields; atol=1e-5, rtol=1e-5)
    rng = MersenneTwister(19)
    p0 = _identity_problem(fields)
    uω0 = ifft(fields[:, :, 1], 1)
    δ = randn(rng, ComplexF64, size(uω0))
    ε = 1e-7
    objective(uω) = value(obs, solve(_identity_problem(reshape(fft(uω, 1),
                                                              size(fields)))))
    fd = (objective(uω0 .+ ε .* δ) - objective(uω0 .- ε .* δ)) / (2ε)
    adj = gradient(p0, obs; method=PulsePropagation.Adjoint(),
                   normalization=PowerNormalized(),
                   reltol=1e-9,
                   abstol=1e-9)
    lin = 2real(sum(conj.(adj.initial_field_gradient) .* δ))
    @test fd ≈ lin atol=atol rtol=rtol
    @test adj.method isa PulsePropagation.Adjoint
    @test size(adj.terminal_condition) == size(uω0)
end

function _check_linear_gain_adjoint_sign(fields; gain=0.7, length=1.0)
    rng = MersenneTwister(31)
    obs = FilterEnergy(; filter=ones(size(fields, 1), size(fields, 2)),
                       domain=:frequency)
    uω0 = ifft(fields[:, :, 1], 1)
    δ = randn(rng, ComplexF64, size(uω0))
    δ ./= norm(δ)
    ε = 1e-7
    objective(uω) = value(obs, solve(_linear_gain_problem(
        reshape(fft(uω, 1), size(fields)); gain, length)))
    fd = (objective(uω0 .+ ε .* δ) - objective(uω0 .- ε .* δ)) / (2ε)
    problem = _linear_gain_problem(fields; gain, length)
    sol = solve(problem)
    adj = gradient(problem, obs; trajectory=sol,
                   method=PulsePropagation.Adjoint(),
                   normalization=PowerNormalized(),
                   raman=:auto,
                   dz_adj=length,
                   reltol=1e-9,
                   abstol=1e-9)
    lin = 2real(sum(conj.(adj.initial_field_gradient) .* δ))
    @test fd ≈ lin atol=1e-5 rtol=1e-5
    @test norm(adj.initial_field_gradient) / norm(uω0) ≈ exp(gain * length) atol=1e-8 rtol=1e-8
end

function _linear_gain_soliton_timing_problem(; nt=256, gain=0.001, length=1000.0)
    grid = TimeGrid(nt, 100.0)
    lambda0 = 1550e-9
    f0 = 2.99792458e8 / lambda0 * 1e-12
    beta2_si = -2.04e-26
    beta2_ps2_m = beta2_si * 1e24
    betas = reshape([0.0, 0.0, beta2_ps2_m], :, 1)
    n2 = 2.3e-20
    mfd_um = 9.2
    aeff = π * (mfd_um * 1e-6 / 2)^2
    sr = reshape([1 / aeff], 1, 1, 1, 1)
    gamma = n2 * 2π / aeff / lambda0
    dofs = SingleModeField()
    system = PassiveFiber(; length, lambda0, f0, material=Silica(),
                          betas, sr, mfd=mfd_um, n2, raman_fraction=0.0,
                          source_path=nothing)
    terms = PropagationTerms(Dispersion(), Kerr())
    pulse_fwhm_ps = 4.0
    tau_sech = pulse_fwhm_ps / (2asech(1 / sqrt(2)))
    peak_power = abs(beta2_si) / gamma / (tau_sech * 1e-12)^2
    fields = sech_pulse(grid; peak_power, fwhm=pulse_fwhm_ps,
                        dofs, transform=("ifft", 0.0))
    solver = RK4IPSolver(;
        stepping=AdaptiveStep(; initial_dz=1.0, max_dz=10.0,
                              rtol=1e-7, atol=1e-8),
        saveat=SaveAt(collect(0.0:20.0:length)),
        linear_gain=gain,
        reltol=1e-8,
        abstol=1e-8,
    )
    return PulsePropagationProblem(; grid, system, dofs, terms, solver, fields)
end

function _check_linear_gain_timing_adjoint_frame()
    rng = MersenneTwister(43)
    problem = _linear_gain_soliton_timing_problem()
    obs = TemporalCentroid()
    sol = solve(problem)
    adj = gradient(problem, obs; trajectory=sol,
                   method=PulsePropagation.Adjoint(),
                   normalization=PowerNormalized(),
                   raman=:none,
                   dz_adj=20.0,
                   reltol=1e-7,
                   abstol=1e-7)
    fields0 = copy(problem.initial_state.fields)
    uω0 = ifft(fields0[:, :, 1], 1)
    δ = randn(rng, ComplexF64, size(uω0))
    δ .*= norm(uω0) / norm(δ)
    ε = 3e-7
    objective(uω) = value(obs, solve(PulsePropagationProblem(;
        initial_state=PulseState(problem.initial_state.grid,
                                 reshape(fft(uω, 1), size(fields0));
                                 basis=problem.initial_state.basis),
        model=problem.model,
        solver=problem.solver)))
    fd = (objective(uω0 .+ ε .* δ) - objective(uω0 .- ε .* δ)) / (2ε)
    lin = 2real(sum(conj.(adj.initial_field_gradient) .* δ))
    @test fd ≈ lin rtol=2e-4 atol=2e-4
end

@testset "observable AD gradients" begin
    rng = MersenneTwister(23)
    nt = 16
    nm = 2
    fields = randn(rng, ComplexF64, nt, nm, 1)

    _check_ad_initial_gradient(FilterEnergy(; filter=rand(rng, nt, nm),
                                           domain=:frequency), fields)
    _check_ad_initial_gradient(RatioObservable(
        FilterEnergy(; filter=rand(rng, nt), domain=:frequency, modes=[1]),
        FilterEnergy(; filter=ones(nt), domain=:frequency, modes=:all)), fields;
        atol=1e-4, rtol=1e-4)
    ad_photon = gradient(_identity_problem(fields),
                         FilterEnergy(; filter=rand(rng, nt, nm),
                                      domain=:frequency);
                         method=AutomaticDifferentiation(),
                         normalization=PhotonNormalized())
    @test ad_photon.method isa AutomaticDifferentiation
    @test size(ad_photon.initial_field_gradient) == (nt, nm)
    @test all(isfinite, real.(ad_photon.initial_field_gradient))
    adj_photon = gradient(_identity_problem(fields),
                          FilterEnergy(; filter=rand(rng, nt, nm),
                                       domain=:frequency);
                          method=PulsePropagation.Adjoint(),
                          normalization=PhotonNormalized())
    adj_photon_z = photon_normalized_adjoint_trajectory(adj_photon,
                                                        _identity_problem(fields))
    @test adj_photon_z.lambdaw[:, :, 1] ≈ adj_photon.initial_field_gradient
    @test adj_photon_z.z == adj_photon.adjoint_trajectory.z
    @test size(adj_photon_z.lambdaw) == size(adj_photon.adjoint_trajectory.lambdaw)
    _check_ad_initial_gradient(BinEnergy(; bin=4, mode=2,
                                        domain=:frequency), fields)
    _check_ad_initial_gradient(QuadratureObservable(; M=randn(rng, ComplexF64, nt * nm),
                                                    domain=:frequency), fields)
    Q, _ = qr(randn(rng, ComplexF64, nm, nm))
    _check_ad_initial_gradient(ProjectedEnergy(; U=Matrix(Q),
                                               domain=:frequency), fields)
    _check_ad_initial_gradient(ProjectedPhotonNumber(; U=Matrix(Q),
                                                     weights=rand(rng, nt),
                                                     filter=rand(rng, nm),
                                                     domain=:frequency), fields)
    _check_ad_initial_gradient(TemporalMoment(; order=1, center=0.0,
                                             normalized=true, modes=[1]), fields)
    _check_ad_initial_gradient(SpectralMoment(; order=1, center=0.0,
                                             normalized=true, shifted=false,
                                             modes=[1]), fields)
end

@testset "linear gain adjoint sign" begin
    rng = MersenneTwister(37)
    fields = randn(rng, ComplexF64, 16, 1, 1)
    _check_linear_gain_adjoint_sign(fields)
    _check_linear_gain_timing_adjoint_frame()
end

@testset "observable adjoint terminal coordinates" begin
    rng = MersenneTwister(31)
    nt = 16
    nm = 2
    fields = randn(rng, ComplexF64, nt, nm, 1)
    grid = TimeGrid(nt, 8.0)
    t = time_axis(grid)

    _check_adjoint_fourier_gradient(
        TemporalMoment(; order=1, center=0.0, normalized=false, modes=[1]),
        fields)
    _check_adjoint_fourier_gradient(
        TemporalMoment(; order=1, center=0.0, normalized=true, modes=[1]),
        fields; atol=1e-4, rtol=1e-4)
    _check_adjoint_fourier_gradient(
        SpectralMoment(; order=1, center=0.0, normalized=true,
                       shifted=true, modes=[1]),
        fields; atol=1e-4, rtol=1e-4)
    _check_adjoint_fourier_gradient(
        RatioObservable(
            FilterEnergy(; filter=t, domain=:time, modes=[1]),
            FilterEnergy(; filter=ones(nt), domain=:time, modes=[1])),
        fields; atol=1e-4, rtol=1e-4)
end

@testset "variance objective covers observable library" begin
    rng = MersenneTwister(29)
    nt = 16
    nm = 2
    fields = randn(rng, ComplexF64, nt, nm, 1)
    problem = _identity_problem(fields)
    sol = solve(problem)
    Q, _ = qr(randn(rng, ComplexF64, nm, nm))
    observables = Any[
        FilterEnergy(; filter=rand(rng, nt, nm), domain=:frequency),
        RatioObservable(
            FilterEnergy(; filter=rand(rng, nt), domain=:frequency, modes=[1]),
            FilterEnergy(; filter=ones(nt), domain=:frequency, modes=:all)),
        FilterEnergy(; filter=rand(rng, nt, nm), domain=:time),
        BinEnergy(; bin=4, mode=2, domain=:frequency),
        BinEnergy(; bin=4, mode=2, domain=:time),
        QuadratureObservable(; M=randn(rng, ComplexF64, nt * nm),
                             domain=:frequency),
        QuadratureObservable(; M=randn(rng, ComplexF64, nt * nm),
                             domain=:time),
        ProjectedEnergy(; U=Matrix(Q), domain=:frequency),
        ProjectedEnergy(; U=Matrix(Q), domain=:time),
        ProjectedPhotonNumber(; U=Matrix(Q), weights=rand(rng, nt),
                               filter=rand(rng, nm), domain=:frequency),
        TemporalMoment(; order=1, center=0.0, normalized=true, modes=[1]),
        SpectralMoment(; order=1, center=0.0, normalized=true,
                       shifted=false, modes=[1]),
    ]

    for obs in observables
        grad = gradient(problem, obs; trajectory=sol,
                        method=AutomaticDifferentiation(),
                        normalization=PowerNormalized())
        var_obj = VarianceObjective(obs; method=AutomaticDifferentiation(),
                                    normalization=PowerNormalized())
        vr = variance_result(var_obj, problem, sol)
        @test vr.value ≈ grad.value
        @test vr.variance ≈ sum(abs2, grad.initial_field_gradient)
        @test value(var_obj, problem, sol) ≈ vr.variance
    end

    photon_filter = rand(rng, nt, nm)
    photon_obs = SpectralPhotonNumber(; filter=photon_filter,
                                      modes=:all,
                                      shifted=false)
    grad = gradient(problem, photon_obs; trajectory=sol,
                    method=AutomaticDifferentiation(),
                    normalization=PhotonNormalized())
    var_obj = VarianceObjective(photon_obs; method=AutomaticDifferentiation(),
                                normalization=PhotonNormalized())
    vr = variance_result(var_obj, problem, sol)
    @test vr.value ≈ grad.value
    @test vr.variance ≈ sum(abs2, grad.initial_field_gradient)
end
