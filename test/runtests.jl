using Test
using PulsePropagation

@testset "PulsePropagation" begin
    include("checkpointed_adjoint.jl")
end
