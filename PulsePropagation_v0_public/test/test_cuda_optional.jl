using .PulsePropagation
using LinearAlgebra
using Random

function _cuda_tiny_problem(; backend=CPUBackend(), compression=nothing,
                            cp_rhs=nothing)
    grid = TimeGrid(; Nt=16, window=2.0)
    fields = gaussian_pulse(grid; fwhm=0.35, peak_power=0.5)
    system = PassiveFiber(; length=1e-5, lambda0=1.55e-6,
                          betas=zeros(3, 1),
                          sr=reshape([1.0], 1, 1, 1, 1),
                          n2=1e-22)
    solver = RK4IPSolver(; stepping=FixedStep(; dz=1e-5),
                         saveat=SaveAt([0.0, 1e-5]),
                         compression=compression,
                         cp_rhs=cp_rhs,
                         backend=backend)
    return PulsePropagationProblem(; grid, system, dofs=SingleMode(),
                                   fields, solver)
end

function _cuda_two_mode_problem(; backend=CPUBackend(), compression=nothing,
                                cp_rhs=nothing)
    grid = TimeGrid(; Nt=24, window=2.0)
    dofs = ModalField(1:2)
    fields = gaussian_pulse(grid; fwhm=0.35, peak_power=0.5,
                            dofs=dofs, coefficients=[1.0, 0.35im])
    cp_exact = PulsePropagation.CPDecomposition(;
        λ=[0.8, 0.25],
        U=(
            [1.0 0.3; 0.2 1.1],
            [0.9 -0.2; 0.4 0.7],
            [1.1 0.5; -0.3 0.8],
            [0.7 -0.4; 0.6 1.0],
        ),
    )
    sr = abs.(PulsePropagation.cp_reconstruct(cp_exact, (2, 2, 2, 2)))
    system = PassiveFiber(; length=2e-5, lambda0=1.55e-6,
                          betas=zeros(3, 2), sr=sr, n2=1e-22)
    solver = RK4IPSolver(; stepping=FixedStep(; dz=1e-5),
                         saveat=SaveAt([0.0, 1e-5, 2e-5]),
                         compression=compression,
                         cp_rhs=cp_rhs,
                         backend=backend)
    problem = PulsePropagationProblem(; grid, system, dofs, fields, solver,
                                      terms=PropagationTerms(Dispersion(), Kerr()))
    return problem, cp_exact
end

function _cuda_available()
    try
        @eval using CUDA
        return CUDA.functional()
    catch
        return false
    end
end

@testset "Optional CUDA backend" begin
    @test CUDABackend() isa PulsePropagation.AbstractExecutionBackend

    cuda_problem = _cuda_tiny_problem(; backend=CUDABackend())
    if !_cuda_available()
        @test_throws ErrorException solve(cuda_problem)
    else
        cpu = solve(_cuda_tiny_problem())
        gpu = solve(cuda_problem)
        @test isapprox(gpu.output.fields, cpu.output.fields; rtol=1e-9, atol=1e-9)

        cp = CPCompression(; target_error=1e-12, initial_rank=1, max_rank=1,
                           maxiter=5)
        cpu_cp = solve(_cuda_tiny_problem(; compression=cp,
                                          cp_rhs=FixedRK4IPCPRHS(; fftw_flags=:ESTIMATE)))
        gpu_cp = solve(_cuda_tiny_problem(; backend=CUDABackend(),
                                          compression=cp,
                                          cp_rhs=FixedRK4IPCPRHS(; fftw_flags=:ESTIMATE)))
        @test isapprox(gpu_cp.output.fields, cpu_cp.output.fields; rtol=1e-9, atol=1e-9)
        @test compression_rank(gpu_cp) == compression_rank(cpu_cp)

        obs = SpectralPhotonNumber(; filter=ones(16), shifted=false)
        g_cpu = gradient(_cuda_tiny_problem(), obs; method=PulsePropagation.Adjoint())
        g_gpu = gradient(cuda_problem, obs; method=PulsePropagation.Adjoint())
        @test isapprox(g_gpu.initial_field_gradient,
                       g_cpu.initial_field_gradient; rtol=1e-8, atol=1e-8)

        ext = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
        factors = [[1.0 2.0; 3.0 4.0],
                   [0.5 -1.0; 1.5 2.5; -0.25 0.75],
                   [2.0 0.25; -1.0 1.25]]
        kr_cpu = PulsePropagation._khatri_rao_rows(factors)
        kr_gpu = Array(ext._khatri_rao_rows_cuda(CUDA.CuArray.(factors)))
        @test kr_gpu == kr_cpu

        tensor = reshape(Float64.(1:16), 2, 2, 2, 2) ./ 10
        cp_cpu, _ = PulsePropagation.cp_als_warm(tensor, 3;
            maxiter=12, check_every=3, rng=MersenneTwister(33))
        cp_gpu, hist_gpu = PulsePropagation.cp_als_warm_cuda(tensor, 3;
            maxiter=12, check_every=3, rng=MersenneTwister(33))
        err_cpu = norm(PulsePropagation.cp_reconstruct(cp_cpu, tensor) .- tensor) / norm(tensor)
        err_gpu = norm(PulsePropagation.cp_reconstruct(cp_gpu, tensor) .- tensor) / norm(tensor)
        @test isfinite(err_gpu)
        @test !isempty(hist_gpu)
        @test err_gpu <= 1.25 * err_cpu + 1e-8

        mm_problem, cp_exact = _cuda_two_mode_problem()
        mm_cuda_problem, _ = _cuda_two_mode_problem(; backend=CUDABackend())
        mm_sol = solve(mm_problem)
        objs = PulsePropagation.backend_objects(mm_problem)
        terminal = PulsePropagation.inverse_fft(mm_sol.output.fields[:, :, end], dims=1)
        adj_cpu = PulsePropagation.solve_adjoint(
            terminal, mm_sol.output, objs.fiber, objs.sim;
            raman=:off, dz_adj=5e-6, adaptive=false,
            return_lambdaw_zsave=false)
        adj_gpu = PulsePropagation.solve_adjoint_cuda(
            terminal, mm_sol.output, objs.fiber, objs.sim;
            raman=:off, dz_adj=2.5e-6,
            return_lambdaw_zsave=false)
        @test isapprox(adj_gpu.lambdaw, adj_cpu.lambdaw; rtol=5e-5, atol=5e-7)
        @test adj_gpu.frame == :interaction

        adj_cpu_z = PulsePropagation.solve_adjoint(
            terminal, mm_sol.output, objs.fiber, objs.sim;
            raman=:off, dz_adj=5e-6, adaptive=false,
            return_lambdaw_zsave=true)
        adj_gpu_z = PulsePropagation.solve_adjoint_cuda(
            terminal, mm_sol.output, objs.fiber, objs.sim;
            raman=:off, dz_adj=2.5e-6,
            return_lambdaw_zsave=true)
        @test adj_gpu_z.frame == :lab
        @test adj_gpu_z.z == adj_cpu_z.z
        @test isapprox(adj_gpu_z.lambdaw, adj_cpu_z.lambdaw; rtol=5e-5, atol=5e-7)

        cp_solver = CPCompression(; tensor=cp_exact)
        cpu_cp2 = solve(_cuda_two_mode_problem(; compression=cp_solver,
            cp_rhs=FixedRK4IPCPRHS(; fftw_flags=:ESTIMATE))[1])
        gpu_cp2 = solve(_cuda_two_mode_problem(; backend=CUDABackend(),
            compression=cp_solver,
            cp_rhs=FixedRK4IPCPRHS(; fftw_flags=:ESTIMATE))[1])
        @test isapprox(gpu_cp2.output.fields, cpu_cp2.output.fields;
                       rtol=5e-8, atol=5e-8)

        compressed = compressed_tensor(cpu_cp2)
        terminal_cp = PulsePropagation.inverse_fft(cpu_cp2.output.fields[:, :, end], dims=1)
        adj_cp_cpu = PulsePropagation.solve_adjoint_compressed_rankchannels(
            terminal_cp, cpu_cp2.output, objs.fiber, objs.sim, compressed;
            raman=:off, dz_adj=5e-6, adaptive=false,
            return_lambdaw_zsave=false)
        adj_cp_gpu = PulsePropagation.solve_adjoint_compressed_rankchannels_cuda(
            terminal_cp, cpu_cp2.output, objs.fiber, objs.sim, compressed;
            raman=:off, dz_adj=2.5e-6,
            return_lambdaw_zsave=false)
        @test isapprox(adj_cp_gpu.lambdaw, adj_cp_cpu.lambdaw;
                       rtol=5e-5, atol=5e-7)
    end
end
