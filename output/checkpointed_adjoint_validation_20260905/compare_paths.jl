using PulsePropagation, FFTW, LinearAlgebra, Random, JLD2

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)
const PP = PulsePropagation
const audit_root = @__DIR__
const label = ARGS[1]
const results = Dict{String,Any}()
println("Loaded ", pathof(PP)); flush(stdout)

function fixture(nm, np, raman, cp)
    grid = TimeGrid(32, 4.0)
    axes = nm == 1 && np == 1 ? degrees_of_freedom(:time) :
           np == 1 ? degrees_of_freedom(:time, :space) :
           degrees_of_freedom(:time, :space, :polarization)
    domain = MMGNLSEDomain(axes, grid)
    rng = MersenneTwister(112)
    factors = 0.2 .+ rand(rng, nm, 2)
    weights = [1.0e10, 0.7e10]
    compressed = MMGNLSECPDecomposition(weights, ntuple(_ -> copy(factors), 4);
        layout=:spatial, nmodes=nm, npolarizations=np, relative_error=0.0)
    dense = zeros(nm, nm, nm, nm)
    for i in 1:nm, j in 1:nm, k in 1:nm, l in 1:nm, r in 1:2
        dense[i,j,k,l] += weights[r]*factors[i,r]*factors[j,r]*factors[k,r]*factors[l,r]
    end
    beta = zeros(3, nm, np)
    beta[3,:,:] .= -0.02
    beta[1,:,:] .= reshape(0.03 .* collect(0:nm*np-1), nm, np)
    parameters = MMGNLSEParameters(domain; length=6.0e-4,
        alpha=0.03, gain=0.01, beta=TaylorBeta(beta),
        S=cp ? compressed : dense, n2=2.3e-20,
        omega0=2pi*193.4, raman)
    t = time_axis(grid)
    initial = zeros(ComplexF64, 32, nm, np)
    for m in 1:nm, p in 1:np
        initial[:,m,p] .= sqrt(2000/(nm*np)) .* exp.(-(t./0.32).^2) .* exp.(0.17im.*t .+ 0.2im*m*p)
    end
    return parameters, initial
end

for (raman_name, raman) in (("kerr", NoRaman()), ("agarwal", AgarwalRaman()), ("anisotropic", AnisotropicRaman()))
    for (layout, nm, np, cp) in (("scalar_dense",1,1,false), ("scalar_cp",1,1,true),
                                ("multimode_dense",2,1,false), ("multimode_cp",2,1,true),
                                ("vector_dense",2,2,false))
        key = raman_name * "/" * layout
        try
        parameters, initial = fixture(nm, np, raman, cp)
        results[key*"/prefactor"] = PP._mmgnlse_solver_cache(parameters).nonlinear_prefactor
        for (method_name, method) in (("rk4ip",RK4IP()), ("tsit5",Tsit5()))
            sol = solve_mmgnlse(initial, parameters, 1.0e-4; method, saveat=:steps)
            results[key*"/forward/"*method_name] = sol.fields
            results[key*"/z/"*method_name] = sol.z
        end
        forward = solve_mmgnlse(initial, parameters, 1.0e-4; saveat=:steps)
        sparse = solve_mmgnlse(initial, parameters, 1.0e-4)
        obs = SpectralPhotonNumber(filter=ones(size(initial)))
        lambda = terminal_condition(obs, forward)
        results[key*"/terminal"] = lambda
        results[key*"/photons"] = [value(obs, forward)]
        for (source_name, source) in (("dense",forward), ("sparse",sparse))
            adjoint = solve_adjoint(lambda, parameters, 1.0e-4;
                forward_solution=source, saveat=:steps)
            results[key*"/adjoint/"*source_name] = adjoint.fields
            results[key*"/adjoint_photon/"*source_name] = get_spectral_field(adjoint; z=:all, units=:photon)
        end
        ensemble = solve_mmgnlse_stochastic(initial, parameters, 1.0e-4, 2;
            seed=246, saveat=:steps)
        results[key*"/stochastic"] = ensemble.fields
        results[key*"/photon_noise"] = ensemble.noise.photon_noise
        if key == "agarwal/scalar_dense"
            results["parameter_checksum"] = mmgnlse_parameters_checksum(parameters)
            results["solution_checksum"] = mmgnlse_solution_checksum(forward)
            save_mmgnlse_solution(joinpath(audit_root, label*"_archive.jld2"), forward)
            JLD2.jldsave(joinpath(audit_root, label*"_raw_archive.jld2"); forward, ensemble)
        end
        println("completed ", key); flush(stdout)
        catch err
            results[key*"/ERROR"] = sprint(showerror, err)
            println("FAILED ", key, ": ", sprint(showerror, err)); flush(stdout)
        end
    end
end

# The older distributed-Raman API is also part of main's public functionality.
nt = 32; dt = 0.025; lambda0 = 1.55e-6; f0 = 2.99792458e-4/lambda0
fiber = PP.Fiber{Float64}(betas=reshape([0.,0.,-0.02],3,1),
    sr=fill(1.0e10,1,1,1,1), L0=2.0e-3, n2=2.3e-20, material="agarwal", fr=0.18)
simulation = PP.Simulation{Float64}(lambda0=lambda0, f0=f0,
    dz=1.0e-3, save_period=1.0e-3, midx=[1], scalar=true,
    include_Raman=true, pulse_centering=false)
t = (collect(0:nt-1) .- nt÷2).*dt
field_t = reshape(ComplexF64.(sqrt(2000.).*exp.(-(t./0.15).^2)),nt,1)
field_w = PP.inverse_fft(field_t,dims=1)
ensemble_w = repeat(reshape(field_w,nt,1,1),1,1,2)
for stepper in (:ode,:rk4ip), noise_scale in (0.0,1.0), shock in (:default,:false,:true)
    kwargs = shock === :default ? (;) : (; include_self_steepening=shock === :true)
    output = PP.stochastic_solve_raman(fiber,ensemble_w,dt,simulation;
        zsave=[0.,1.0e-3,2.0e-3],rng=MersenneTwister(883),
        noise_scale,keep_noise=true,deterministic_stepper=stepper,kwargs...)
    key = "legacy/$(stepper)/noise$(noise_scale)/shock$(shock)"
    results[key*"/fields"] = output.fields
    results[key*"/noise"] = output.raman_noise
    results[key*"/spectrum"] = output.Cω
end
results["legacy/spectrum_default"] = PP.raman_langevin_spectrum(nt,dt,f0,1.0e-3)
JLD2.jldsave(joinpath(audit_root,label*"_results.jld2"); results)
println("SAVED ",label," ",length(results)," outputs"); flush(stdout)
