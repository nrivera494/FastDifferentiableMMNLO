# Run the same script under each revision's project on one GPU:
# julia --project=/path/to/revision test/cuda_compatibility.jl output.jld2
# This records numerical results for cross-revision comparison; it does not
# require the checkpointed API, so the unchanged script also runs on old main.
using CUDA, FFTW, LinearAlgebra, Random, JLD2, Test
using PulsePropagation
include("cuda_compatibility_fixtures.jl")

CUDA.functional() || error("A functional CUDA device is required.")
CUDA.allowscalar(false)
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)
length(ARGS) == 1 || error("Supply a destination JLD2 path.")
results = Dict{String,Any}()
timings = Dict{String,Float64}()
errors = Dict{String,String}()

function record_solution!(results, key, solution)
    @test all(isfinite, solution.fields)
    results[key*"/fields"] = solution.fields
    results[key*"/z"] = solution.z
    if hasproperty(solution,:cache) && solution.cache !== nothing
        for property in (:integration_z,:accepted_steps,:rejected_steps,:rhs_evaluations)
            hasproperty(solution.cache,property) || continue
            results[key*"/"*string(property)] = getproperty(solution.cache,property)
        end
    end
end

function audit_case(f, key)
    println("START ",key); flush(stdout)
    started = time()
    try
        f()
        CUDA.synchronize()
        println("PASS ",key)
    catch err
        errors[key] = sprint(showerror,err,catch_backtrace())
        println("ERROR ",key,": ",sprint(showerror,err))
    finally
        timings[key] = time()-started
        flush(stdout)
    end
end

try
@testset "Existing GPU notebook pathways" begin
    for (rname,raman) in (("kerr",NoRaman()),("agarwal",AgarwalRaman()),
                         ("anisotropic",AnisotropicRaman()))
        for (name,nm,np,cp,backend) in (
            ("scalar",1,1,false,:cuda),
            ("dense",3,1,false,:cuda),
            ("cp_baseline",3,1,true,:cuda),
            ("cp_optimized",3,1,true,:cuda_cp_optimized),
            ("polarized_dense",2,2,false,:cuda),
            ("polarized_cp",2,2,true,:cuda_cp_optimized))
            parameters,initial = cuda_compatibility_fixture(;nm,np,cp,raman)
            dz = 5e-4
            cpu = solve_mmgnlse(initial,parameters,dz;saveat=[0.,parameters.length])
            for precision in (:float64,:mixed), adaptive in (false,true)
                key = "$rname/$name/$precision/adaptive$adaptive"
                audit_case(key) do
                    solution = solve_mmgnlse(initial,parameters,dz;
                        backend,precision,adaptive,saveat=[0.,parameters.length],
                        reltol=1e-7,abstol=1e-10)
                    record_solution!(results,key,solution)
                    # Independent CPU check. Cross-revision equality is assessed
                    # separately, without using this CPU/GPU precision tolerance.
                    tolerance = precision === :float64 ? 2e-7 : 2e-5
                    @test norm(solution.fields-cpu.fields)/norm(cpu.fields) < tolerance
                end
            end
            key = "$rname/$name/continuous_adjoint"
            audit_case(key) do
                terminal = randn(MersenneTwister(194),ComplexF64,size(initial))
                terminal ./= norm(terminal)
                forward = solve_mmgnlse(initial,parameters,dz;backend,saveat=:steps)
                adjoint = solve_adjoint(terminal,parameters,dz;
                    forward_solution=forward,backend,saveat=:steps)
                record_solution!(results,key,adjoint)
                cpu_forward = solve_mmgnlse(initial,parameters,dz;saveat=:steps)
                cpu_adjoint = solve_adjoint(terminal,parameters,dz;
                    forward_solution=cpu_forward,saveat=:steps)
                # Existing CPU/CUDA continuous solvers have small numerical
                # differences. Record these for exact before/after comparison;
                # the one-ppm smoke bound is not the regression tolerance.
                cpu_error = norm(adjoint.fields-cpu_adjoint.fields)/norm(cpu_adjoint.fields)
                results[key*"/cpu_relative_error"] = cpu_error
                @test cpu_error < 1e-6
            end
        end
    end

    # Explicitly select kernels normally chosen for much larger dense tensors.
    for np in (1,2), (contraction,raman) in
        (("gemm",AnisotropicRaman()),("symmetric_gemm",NoRaman())),
        precision in (:float64,:mixed)
        key = "dense_contraction/$contraction/polarizations$np/$precision"
        audit_case(key) do
            parameters,initial = cuda_compatibility_fixture(;np,raman)
            solution = withenv("PULSEPROPAGATION_CUDA_DENSE_CONTRACTION"=>contraction) do
                solve_mmgnlse(initial,parameters,5e-4;backend=:cuda,precision)
            end
            record_solution!(results,key,solution)
            cpu = solve_mmgnlse(initial,parameters,5e-4)
            @test norm(solution.fields-cpu.fields)/norm(cpu.fields) < 2e-5
        end
    end

    for cp in (false,true), precision in (:float64,:mixed)
        key = "stochastic/cp$cp/$precision"
        audit_case(key) do
            parameters,initial = cuda_compatibility_fixture(;cp)
            backend = cp ? :cuda_cp_optimized : :cuda
            ensemble = solve_mmgnlse_stochastic(initial,parameters,5e-4,3;
                backend,precision,noise=10.,seed=711)
            record_solution!(results,key,ensemble)
            results[key*"/noise"] = ensemble.noise.photon_noise
            results[key*"/weights"] = ensemble.noise.weights
        end
    end

    audit_case("adaptive_rejections") do
        parameters,initial = cuda_compatibility_fixture(;
            cp=true,length_m=0.05,peak_power=1e5)
        solution = solve_mmgnlse(initial,parameters,0.05;
            backend=:cuda_cp_optimized,adaptive=true,reltol=1e-8,abstol=1e-10)
        record_solution!(results,"adaptive_rejections",solution)
        @test solution.cache.rejected_steps > 0
    end

    audit_case("cp_tile_boundaries") do
        parameters,initial = cuda_compatibility_fixture(;nt=96,nm=17,rank=33,cp=true)
        solution = solve_mmgnlse(initial,parameters,5e-4;
            backend=:cuda_cp_optimized,precision=:mixed)
        record_solution!(results,"cp_tile_boundaries",solution)
    end

    audit_case("compression") do
        parameters,initial = cuda_compatibility_fixture(;nm=3,rank=3)
        compressed = cp_compress(parameters.S;backend=:cuda,error=1e-3,
            min_rank=3,rank_step=2,max_rank=9,maxiter=120,seed=442,
            tolerance=1e-8,ridge=1e-10)
        reconstructed = cp_reconstruct_mmgnlse(compressed)
        results["compression/tensor"] = reconstructed
        results["compression/relative_error"] = compressed.relative_error
        @test compressed.relative_error <= 1e-3
    end

    for (rname,raman) in (("kerr",NoRaman()),("anisotropic",AnisotropicRaman()))
        key = "full_field/$rname"
        audit_case(key) do
            grid = TimeGrid(32,4.)
            xy = TransverseGrid(Nx=8,Ny=8,dx=5e-6,dy=5e-6)
            domain = MMGNLSEDomain(degrees_of_freedom(:time,:space),grid;
                transverse_grid=xy)
            initial = reshape(exp.(-(time_axis(grid)./0.4).^2),32,1,1) .*
                reshape(exp.(-(xy.x./1e-5).^2),1,8,1) .*
                reshape(exp.(-(xy.y./1e-5).^2),1,1,8) .* ComplexF64(1e6)
            initial = reshape(initial,32,8,8,1)
            parameters = FullFieldMMGNLSEParameters(domain;length=4e-4,
                beta=TaylorBeta([5.8e6,0.,-0.02]),potential=zeros(8,8),
                n2=2.3e-20,omega0=2pi*193.4,raman)
            solution = solve_mmgnlse_full_field(initial,parameters,1e-4;
                backend=:cuda,method=StrangSplit(),precision=:float64)
            record_solution!(results,key,solution)
        end
    end
end

catch err
    errors["testset"] = sprint(showerror,err)
finally
metadata = (;julia=string(VERSION),cuda=string(Base.pkgversion(CUDA)),
    gpu=CUDA.name(CUDA.device()),source=pathof(PulsePropagation))
JLD2.jldsave(ARGS[1];results,timings,errors,metadata)
println("SAVED ",ARGS[1]," outputs=",length(results)," errors=",length(errors))
end
isempty(errors) || error("GPU audit cases failed; inspect the saved errors.")
