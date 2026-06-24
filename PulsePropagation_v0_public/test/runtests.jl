using Test
using Pkg

const ROOT = abspath(joinpath(@__DIR__, "..", ".."))
const PUBLIC_ROOT = abspath(joinpath(@__DIR__, ".."))
Pkg.activate(PUBLIC_ROOT)

include(joinpath(PUBLIC_ROOT, "src", "PulsePropagation.jl"))

@testset "PulsePropagation public v0" begin
    include("test_public_native_backend.jl")
    include("test_multimode_stochastic_analytic.jl")
    include("test_cuda_optional.jl")
    include("test_observables.jl")
    include("test_optimization.jl")
end
