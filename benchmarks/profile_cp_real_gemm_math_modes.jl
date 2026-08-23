using CUDA
using LinearAlgebra
using Statistics

CUDA.functional() || error("CUDA is not functional.")

const NT = parse(Int, get(ENV, "NT", "8192"))
const NM = parse(Int, get(ENV, "NM", "30"))
const RANK = parse(Int, get(ENV, "CP_RANK", "252"))
const NSAMPLES = parse(Int, get(ENV, "NSAMPLES", "50"))
const NWARMUP = parse(Int, get(ENV, "NWARMUP", "10"))

function measure(f)
    for _ in 1:NWARMUP
        f()
    end
    CUDA.synchronize()
    samples = Float64[]
    for _ in 1:NSAMPLES
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        push!(samples, (time_ns() - start) / 1e6)
    end
    return median(samples)
end

projection_input = CUDA.rand(Float32, 2NT, NM)
projection_factor = CUDA.rand(Float32, NM, RANK)
projection_output = CUDA.zeros(Float32, 2NT, RANK)
backprojection_factor = CUDA.rand(Float32, RANK, NM)
backprojection_output = CUDA.zeros(Float32, 2NT, NM)

CUDA.math_mode!(CUDA.DEFAULT_MATH; precision=:TensorFloat32)
mul!(projection_output, projection_input, projection_factor)
projection_reference = Array(projection_output)
mul!(backprojection_output, projection_output, backprojection_factor)
backprojection_reference = Array(backprojection_output)

for (label, mode, precision) in (
    (:default, CUDA.DEFAULT_MATH, :TensorFloat32),
    (:tf32, CUDA.FAST_MATH, :TensorFloat32),
    (:bf16, CUDA.FAST_MATH, :BFloat16),
)
    CUDA.math_mode!(mode; precision)
    projection_ms = measure() do
        mul!(projection_output, projection_input, projection_factor)
    end
    projection_value = Array(projection_output)
    backprojection_ms = measure() do
        mul!(backprojection_output, projection_output, backprojection_factor)
    end
    backprojection_value = Array(backprojection_output)
    println(
        "gpu=", CUDA.name(CUDA.device()),
        " mode=", label,
        " projection_ms=", projection_ms,
        " projection_relative_l2=",
        norm(projection_value .- projection_reference) /
        norm(projection_reference),
        " backprojection_ms=", backprojection_ms,
        " backprojection_relative_l2=",
        norm(backprojection_value .- backprojection_reference) /
        norm(backprojection_reference),
    )
end
