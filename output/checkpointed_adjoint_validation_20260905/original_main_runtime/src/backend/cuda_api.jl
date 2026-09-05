function _cuda_unavailable_error(name)
    error("$name requires CUDA.jl. Load PulsePropagation in an environment with CUDA.jl available and a functional CUDA device.")
end

function cuda_cp_fast_rhs_cache(args...; kwargs...)
    _cuda_unavailable_error("cuda_cp_fast_rhs_cache")
end

function cuda_full_rhs_cache(args...; kwargs...)
    _cuda_unavailable_error("cuda_full_rhs_cache")
end

function n_op_rk4ip_full_cuda!(args...; kwargs...)
    _cuda_unavailable_error("n_op_rk4ip_full_cuda!")
end

function n_op_rk4ip_full_cuda(args...; kwargs...)
    _cuda_unavailable_error("n_op_rk4ip_full_cuda")
end

function propagate_rk4ip_full_cuda(args...; kwargs...)
    _cuda_unavailable_error("propagate_rk4ip_full_cuda")
end

function n_op_rk4ip_cp_cuda!(args...; kwargs...)
    _cuda_unavailable_error("n_op_rk4ip_cp_cuda!")
end

function n_op_rk4ip_cp_cuda(args...; kwargs...)
    _cuda_unavailable_error("n_op_rk4ip_cp_cuda")
end

function propagate_rk4ip_cp_cuda(args...; kwargs...)
    _cuda_unavailable_error("propagate_rk4ip_cp_cuda")
end

function cp_als_warm_cuda(args...; kwargs...)
    _cuda_unavailable_error("cp_als_warm_cuda")
end

function _cp_compress_cuda(args...; kwargs...)
    _cuda_unavailable_error("cp_compress(...; backend=:cuda)")
end

function _cp_compress_factorized_cuda(args...; kwargs...)
    _cuda_unavailable_error(
        "cp_compress(::MMGNLSECPDecomposition; backend=:cuda)")
end

function _cp_relative_error_factorized_cuda(args...; kwargs...)
    _cuda_unavailable_error(
        "cp_relative_error(...; backend=:cuda)")
end

function _cp_cuda_memory_status(args...; kwargs...)
    _cuda_unavailable_error("cp_compression_plan(...; backend=:cuda)")
end

function _cp_prepare_bounded_cuda_source(args...; kwargs...)
    _cuda_unavailable_error("cp_compress_bounded(...; backend=:cuda)")
end

function _cp_compress_symmetric_experimental_cuda(args...; kwargs...)
    _cuda_unavailable_error(
        "cp_compress_symmetric_experimental(...; backend=:cuda)")
end

function solve_adjoint_cuda(args...; kwargs...)
    _cuda_unavailable_error("solve_adjoint_cuda")
end

function solve_adjoint_compressed_rankchannels_cuda(args...; kwargs...)
    _cuda_unavailable_error("solve_adjoint_compressed_rankchannels_cuda")
end

# New public MMGNLSE facade hooks.  The CUDA extension adds methods with the
# same signatures; keeping the fallbacks in the core package lets callers use
# `backend=:cuda` without making CUDA.jl a required dependency.
function _mmgnlse_solve_cuda(args...; backend=:cuda, kwargs...)
    _cuda_unavailable_error(
        "solve_mmgnlse(...; backend=$(repr(backend)))")
end

function _mmgnlse_solve_adjoint_cuda(args...; backend=:cuda, kwargs...)
    _cuda_unavailable_error(
        "solve_adjoint(...; backend=$(repr(backend)))")
end
