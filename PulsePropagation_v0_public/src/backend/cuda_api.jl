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

function solve_adjoint_cuda(args...; kwargs...)
    _cuda_unavailable_error("solve_adjoint_cuda")
end

function solve_adjoint_compressed_rankchannels_cuda(args...; kwargs...)
    _cuda_unavailable_error("solve_adjoint_compressed_rankchannels_cuda")
end
