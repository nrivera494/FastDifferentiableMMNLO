# Small, deterministic physical inputs shared by the GPU regression checks.
function cuda_compatibility_fixture(; nt=64, nm=3, np=1, rank=5,
                                    raman=AnisotropicRaman(), cp=false,
                                    length_m=4e-3, peak_power=2e3)
    grid = TimeGrid(nt, 4.0)
    dofs = np == 2 ? degrees_of_freedom(:time, :space, :polarization) :
           nm > 1 ? degrees_of_freedom(:time, :space) :
                    degrees_of_freedom(:time)
    domain = MMGNLSEDomain(dofs, grid)
    rng = MersenneTwister(312)
    factor = 0.2 .+ rand(rng, nm, rank)
    weights = fill(1e10 / rank, rank)
    compressed = MMGNLSECPDecomposition(weights, ntuple(_ -> copy(factor), 4);
        layout=:spatial, nmodes=nm, npolarizations=np, relative_error=0.0)
    dense = zeros(nm, nm, nm, nm)
    for i in 1:nm, j in 1:nm, k in 1:nm, l in 1:nm, r in 1:rank
        dense[i,j,k,l] += weights[r]*factor[i,r]*factor[j,r]*factor[k,r]*factor[l,r]
    end
    beta = zeros(4, nm, np)
    for m in 1:nm, p in 1:np
        beta[:,m,p] .= (0.03*(m-1)+0.01*(p-1), 0.002*(m-1), -0.02, 1e-4)
    end
    parameters = MMGNLSEParameters(domain; length=length_m,
        alpha=0.02, gain=0.01, beta=TaylorBeta(beta),
        S=cp ? compressed : dense, n2=2.3e-20, omega0=2pi*193.4, raman)
    initial = zeros(ComplexF64, nt, nm, np)
    t = time_axis(grid)
    for m in 1:nm, p in 1:np
        initial[:,m,p] .= sqrt(peak_power/(nm*np)) .*
            exp.(-((t .- 0.03*(m-1))./0.32).^2) .* exp.(0.17im.*t .+ 0.2im*m*p)
    end
    return parameters, initial
end
