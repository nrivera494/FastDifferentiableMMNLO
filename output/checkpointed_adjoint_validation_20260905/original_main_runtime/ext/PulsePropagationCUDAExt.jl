module PulsePropagationCUDAExt

using CUDA
using FFTW: fft, ifft, ifftshift, plan_fft!, plan_ifft!, plan_rfft, plan_irfft
using LinearAlgebra: I, dot, mul!, norm
using Random
import PulsePropagation

struct CUDAFullRHSCache{T,A2,A3,PFM,PIM,PFR,PIR}
    nm::Int
    at::A2
    nonlinear::A2
    raman_pairs::A3
    sk::CUDA.CuArray{T,4}
    sra::CUDA.CuArray{T,4}
    fft_modes!::PFM
    ifft_modes!::PIM
    fft_pairs!::PFR
    ifft_pairs!::PIR
end

struct CUDACPFastRHSCache{T,A,PFM,PIM,PFR,PIR}
    rank::Int
    at::A
    conj_at::A
    b2::A
    b3::A
    b4::A
    p::A
    k::A
    conv::A
    nonlinear::A
    u2::A
    u3::A
    u4::A
    wu1t::A
    fft_modes!::PFM
    ifft_modes!::PIM
    fft_rank!::PFR
    ifft_rank!::PIR
end

struct CUDACPAdjointRamanCache{A,PFL,PFU,PFR,PIR,PIT}
    exp_p::A
    exp_m::A
    lambdat::A
    lambdawc_t::A
    ut::A
    rankwork::A
    term_t::A
    rhs::A
    fft_lambdat!::PFL
    fft_ut!::PFU
    fft_rank!::PFR
    ifft_rank!::PIR
    ifft_term!::PIT
end

function _cuda_cp_adjoint_raman_cache(nt::Integer, nm::Integer, rank::Integer,
                                      ::Type{T}) where {T}
    exp_p = CUDA.zeros(Complex{T}, nt, nm)
    exp_m = similar(exp_p)
    lambdat = similar(exp_p)
    lambdawc_t = similar(exp_p)
    ut = similar(exp_p)
    rankwork = CUDA.zeros(Complex{T}, nt, rank)
    term_t = similar(exp_p)
    rhs = similar(exp_p)
    fft_lambdat! = plan_fft!(lambdat, 1)
    fft_ut! = plan_fft!(ut, 1)
    fft_rank! = plan_fft!(rankwork, 1)
    ifft_rank! = plan_ifft!(rankwork, 1)
    ifft_term! = plan_ifft!(term_t, 1)
    return CUDACPAdjointRamanCache{typeof(exp_p),typeof(fft_lambdat!),
                                   typeof(fft_ut!),typeof(fft_rank!),
                                   typeof(ifft_rank!),typeof(ifft_term!)}(
        exp_p, exp_m, lambdat, lambdawc_t, ut, rankwork, term_t, rhs,
        fft_lambdat!, fft_ut!, fft_rank!, ifft_rank!, ifft_term!)
end

function PulsePropagation.cuda_full_rhs_cache(srsk::PulsePropagation.SRSKInfo{S}, nt::Integer,
                                     nm::Integer; T=S) where {S}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    sk = CUDA.CuArray(PulsePropagation._dense_tensor(T.(srsk.sk), srsk.sk_indices, nm))
    sra = CUDA.CuArray(PulsePropagation._dense_tensor(T.(srsk.sra), srsk.sra_indices, nm))

    at = CUDA.zeros(Complex{T}, nt, nm)
    nonlinear = similar(at)
    raman_pairs = CUDA.zeros(Complex{T}, nt, nm, nm)

    fft_modes! = plan_fft!(at, 1)
    ifft_modes! = plan_ifft!(nonlinear, 1)
    fft_pairs! = plan_fft!(raman_pairs, 1)
    ifft_pairs! = plan_ifft!(raman_pairs, 1)

    return CUDAFullRHSCache{T,typeof(at),typeof(raman_pairs),
                            typeof(fft_modes!),typeof(ifft_modes!),
                            typeof(fft_pairs!),typeof(ifft_pairs!)}(
        nm, at, nonlinear, raman_pairs, sk, sra,
        fft_modes!, ifft_modes!, fft_pairs!, ifft_pairs!)
end

function PulsePropagation.cuda_cp_fast_rhs_cache(cp, nt::Integer; T=Float64)
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    cpd = PulsePropagation._as_cp(cp)
    nm = size(cpd.U[1], 1)
    rank = length(cpd.λ)

    u1 = Complex{T}.(cpd.U[1])
    u2 = CUDA.CuArray(Complex{T}.(cpd.U[2]))
    u3 = CUDA.CuArray(Complex{T}.(cpd.U[3]))
    u4 = CUDA.CuArray(Complex{T}.(cpd.U[4]))
    λ = Complex{T}.(cpd.λ)
    wu1t = CUDA.CuArray(Matrix(transpose(u1 .* reshape(λ, 1, :))))

    at = CUDA.zeros(Complex{T}, nt, nm)
    conj_at = similar(at)
    b2 = CUDA.zeros(Complex{T}, nt, rank)
    b3 = similar(b2)
    b4 = similar(b2)
    p = similar(b2)
    k = similar(at)
    conv = similar(b2)
    nonlinear = similar(at)

    fft_modes! = plan_fft!(at, 1)
    ifft_modes! = plan_ifft!(nonlinear, 1)
    fft_rank! = plan_fft!(conv, 1)
    ifft_rank! = plan_ifft!(conv, 1)

    return CUDACPFastRHSCache{T,typeof(at),typeof(fft_modes!),typeof(ifft_modes!),
                              typeof(fft_rank!),typeof(ifft_rank!)}(
        rank, at, conj_at, b2, b3, b4, p, k, conv, nonlinear,
        u2, u3, u4, wu1t, fft_modes!, ifft_modes!, fft_rank!, ifft_rank!)
end

function _mode_unfold_cuda(X, n::Integer)
    order = (n, (k for k in 1:ndims(X) if k != n)...)
    return reshape(permutedims(X, order), size(X, n), :)
end

function _mode_unfold_cuda!(workspace, X, n::Integer)
    order = (n, (k for k in 1:ndims(X) if k != n)...)
    destination_shape = ntuple(index -> size(X, order[index]), ndims(X))
    destination = reshape(workspace, destination_shape)
    permutedims!(destination, X, order)
    return reshape(destination, size(X, n), :)
end

function _check_cuda_unfolding_addressability(X, symmetric_source::Bool)
    if !symmetric_source && length(X) > typemax(Cint)
        throw(ArgumentError(
            "The nonsymmetric CUDA CP path cannot safely permute a tensor " *
            "with $(length(X)) elements; the current CUDA permutation " *
            "backend is limited to $(typemax(Cint)) elements (215^4 fits, " *
            "216^4 does not). For a fully permutation-symmetric overlap, " *
            "set symmetric_source=true; otherwise use backend=:cpu or " *
            "reduce the mode count."))
    end
    return nothing
end

function _cuda_complex_source(X, ::Type{CT};
                              staging_bytes::Integer=64 * 1024^2) where {CT<:Complex}
    if X isa CUDA.CuArray{CT,4}
        return X
    elseif X isa CUDA.CuArray
        return CT.(X)
    elseif eltype(X) === CT
        return CUDA.CuArray(X)
    end

    destination = CUDA.CuArray{CT}(undef, size(X))
    source_flat = reshape(X, :)
    destination_flat = reshape(destination, :)
    source_eltype = eltype(X)
    staging_elements = min(
        length(source_flat),
        max(1, Int(staging_bytes) ÷ sizeof(source_eltype)))
    staging = CUDA.CuArray{source_eltype}(undef, staging_elements)
    GC.@preserve destination begin
        for first_element in 1:staging_elements:length(source_flat)
            last_element = min(
                first_element + staging_elements - 1,
                length(source_flat))
            count = last_element - first_element + 1
            source_range = first_element:last_element
            staging_view = view(staging, 1:count)
            copyto!(staging, 1, source_flat, first_element, count)
            view(destination_flat, source_range) .= CT.(staging_view)
        end
        CUDA.synchronize()
        staging = nothing
        GC.gc(false)
        CUDA.reclaim()
    end
    return destination
end

function PulsePropagation._cp_cuda_memory_status(;
    device=nothing,
    reclaim_memory::Bool=true,
)
    CUDA.functional() || Base.error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    reclaim_memory && CUDA.reclaim()
    cached = Int(coalesce(CUDA.cached_memory(), 0))
    used = Int(coalesce(CUDA.used_memory(), 0))
    reclaimable = max(cached - used, 0)
    raw_free = Int(CUDA.free_memory())
    return (;
        available_bytes=Base.checked_add(raw_free, reclaimable),
        raw_free_bytes=raw_free,
        reclaimable_bytes=reclaimable,
    )
end

function PulsePropagation._cp_prepare_bounded_cuda_source(
    S::AbstractArray{<:Number,4},
    ::Type{CT};
    device=nothing,
) where {CT<:Complex}
    CUDA.functional() || Base.error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    return _cuda_complex_source(S, CT)
end

function _khatri_rao_rows_cuda!(destination, factors;
                                conjugate_factors::Bool=false)
    length(factors) == 3 || throw(ArgumentError(
        "Fourth-order CP MTTKRP requires exactly three factors."))
    first_factor, second_factor, third_factor = factors
    rows1, rank = size(first_factor)
    rows2 = size(second_factor, 1)
    rows3 = size(third_factor, 1)
    size(destination) == (rows1 * rows2 * rows3, rank) ||
        throw(DimensionMismatch(
            "Khatri–Rao destination has size $(size(destination)); expected " *
            "$((rows1 * rows2 * rows3, rank))."))
    shaped_destination = reshape(
        destination, rows1, rows2, rows3, rank)
    if conjugate_factors
        shaped_destination .=
            conj.(reshape(first_factor, rows1, 1, 1, rank)) .*
            conj.(reshape(second_factor, 1, rows2, 1, rank)) .*
            conj.(reshape(third_factor, 1, 1, rows3, rank))
        return destination
    end
    shaped_destination .=
        reshape(first_factor, rows1, 1, 1, rank) .*
        reshape(second_factor, 1, rows2, 1, rank) .*
        reshape(third_factor, 1, 1, rows3, rank)
    return destination
end

function _cp_mttkrp_cuda(unfolded, factors, rest,
                         rank_block_size::Integer;
                         conjugate_factors::Bool=false)
    rank = size(factors[1], 2)
    block = min(Int(rank_block_size), rank)
    block > 0 || throw(ArgumentError(
        "rank_block_size must be positive."))
    result = similar(factors[1], size(unfolded, 1), rank)
    khatri_rao_workspace = similar(
        factors[1], size(unfolded, 2), block)
    for first_column in 1:block:rank
        last_column = min(first_column + block - 1, rank)
        columns = first_column:last_column
        khatri_rao = view(
            khatri_rao_workspace, :, 1:length(columns))
        _khatri_rao_rows_cuda!(
            khatri_rao,
            [view(factors[index], :, columns) for index in rest];
            conjugate_factors)
        destination = view(result, :, columns)
        # The classic cuBLAS interface uses 32-bit matrix dimensions/offsets.
        # A mode unfolding can exceed 2^31 elements well before it exhausts an
        # 80 GiB GPU. Split only the inner dimension in that case and retain
        # the same blocked Khatri–Rao workspace.
        maximum_inner = max(
            1, typemax(Cint) ÷ max(size(unfolded, 1), 1))
        if eltype(unfolded) <: Complex &&
           size(unfolded, 2) > maximum_inner
            first_inner = true
            for first_row in 1:maximum_inner:size(unfolded, 2)
                last_row = min(
                    first_row + maximum_inner - 1,
                    size(unfolded, 2))
                inner_rows = first_row:last_row
                mul!(
                    destination,
                    view(unfolded, :, inner_rows),
                    view(khatri_rao, inner_rows, :),
                    one(eltype(destination)),
                    first_inner ? zero(eltype(destination)) :
                                  one(eltype(destination)),
                )
                first_inner = false
            end
        else
            mul!(destination, unfolded, khatri_rao)
        end
    end
    return result
end

function _cuda_cp_rank_block_size(nmodes::Int, rank::Int, ::Type{T};
                                  rank_block_size=nothing,
                                  max_workspace_bytes=nothing,
                                  workspace_memory_fraction::Real=0.5,
                                  reclaim_memory::Bool=true,
                                  reserve_unfolding::Bool=false) where {T}
    PulsePropagation._cp_validate_memory_controls(
        rank_block_size, max_workspace_bytes, workspace_memory_fraction)
    reclaim_memory && CUDA.reclaim()
    cached = coalesce(CUDA.cached_memory(), 0)
    used = coalesce(CUDA.used_memory(), 0)
    reclaimable = max(cached - used, 0)
    available = CUDA.free_memory() + reclaimable
    unfolding_bytes = reserve_unfolding ? Base.checked_mul(
        Base.checked_mul(
            Base.checked_mul(
                Base.checked_mul(nmodes, nmodes), nmodes), nmodes),
        sizeof(T)) : 0
    available_for_khatri_rao = available - unfolding_bytes
    available_for_khatri_rao > 0 || throw(ArgumentError(
        "The generic CUDA CP path needs a $unfolding_bytes-byte mode " *
        "unfolding in addition to the resident source, but only $available " *
        "bytes are currently available. For a permutation-symmetric overlap, " *
        "set symmetric_source=true; otherwise use lower precision or fewer modes."))
    fraction_budget = min(
        floor(Int, workspace_memory_fraction * available),
        available_for_khatri_rao)
    budget = max_workspace_bytes === nothing ? fraction_budget :
             min(fraction_budget, Int(max_workspace_bytes))
    bytes_per_column = Base.checked_mul(
        Base.checked_mul(Base.checked_mul(nmodes, nmodes), nmodes), sizeof(T))
    affordable = budget ÷ bytes_per_column
    affordable > 0 || throw(ArgumentError(
        "The CUDA CP workspace budget ($budget bytes) cannot hold one " *
        "Khatri–Rao rank column ($bytes_per_column bytes at $nmodes modes). " *
        "Increase max_workspace_bytes/workspace_memory_fraction, use lower " *
        "precision, or reduce the mode count."))
    requested = rank_block_size === nothing ? rank : Int(rank_block_size)
    return min(rank, requested, affordable), budget, bytes_per_column
end

function _cp_relative_error_cuda(source, source_norm2, weights, factors,
                                 rank_block_size::Integer;
                                 symmetric_source::Bool=false)
    # Mode one is already contiguous in Julia's column-major storage.  Using
    # permutedims here would allocate an unnecessary second full tensor.
    unfolded = reshape(source, size(source, 1), :)
    mttkrp = _cp_mttkrp_cuda(
        unfolded, factors, (2, 3, 4), rank_block_size;
        conjugate_factors=eltype(source) <: Complex)
    cross = sum(weights .* vec(sum(conj.(factors[1]) .* mttkrp; dims=1)))
    gram = CUDA.ones(eltype(factors[1]), length(weights), length(weights))
    for factor in factors
        gram .*= adjoint(factor) * factor
    end
    gram_weights = eltype(gram).(weights)
    approximation_norm2 = real(dot(gram_weights, gram * gram_weights))
    residual2 = real(source_norm2 + approximation_norm2 - 2cross)
    return sqrt(max(Float64(residual2 / source_norm2), 0.0))
end

function _normalize_columns_cuda!(A)
    norms = vec(sqrt.(sum(abs2, A; dims=1)))
    safe = ifelse.(norms .<= eps(eltype(real(norms))), one(eltype(real(norms))), norms)
    A ./= reshape(safe, 1, :)
    return Array(safe)
end

function PulsePropagation.cp_als_warm_cuda(X::AbstractArray{T,4}, rank::Integer;
                                           init=nothing, maxiter::Integer=75,
                                           tol=1e-5, ridge=T(1e-8),
                                           check_every::Integer=5,
                                           rng=Random.default_rng(),
                                           verbose::Bool=false,
                                           device=nothing,
                                           synchronize::Bool=true,
                                           rank_block_size=nothing,
                                           max_workspace_bytes=nothing,
                                           workspace_memory_fraction::Real=0.5,
                                           reclaim_memory::Bool=true,
                                           symmetric_source::Bool=false) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    rank > 0 || error("rank must be positive.")
    _check_cuda_unfolding_addressability(X, symmetric_source)
    Xd = X isa CUDA.CuArray{T,4} ? X : CUDA.CuArray(T.(X))
    # cuBLAS nrm2 takes a 32-bit vector length and is unsafe once the source
    # tensor exceeds 2^31 elements. CUDA.jl's reduction remains valid there.
    Xnorm = sqrt(sum(abs2, Xd))
    Xnorm > zero(T) || error("Cannot decompose a zero tensor.")

    λ_cpu, U_cpu = PulsePropagation._warm_started_factors(
        X, rank; init=init, rng=rng, source_norm=Xnorm)
    λ = CUDA.CuArray(T.(λ_cpu))
    U = [CUDA.CuArray(T.(U_cpu[n])) for n in 1:4]
    resolved_block, workspace_budget, bytes_per_column =
        _cuda_cp_rank_block_size(
            size(X, 1), Int(rank), T;
            rank_block_size,
            max_workspace_bytes,
            workspace_memory_fraction,
            reclaim_memory,
            reserve_unfolding=!symmetric_source)
    if verbose
        @info "CUDA CP workspace" rank=rank rank_block_size=resolved_block workspace_budget_bytes=workspace_budget bytes_per_rank_column=bytes_per_column symmetric_source
    end
    history = Float64[]
    unfolding_workspace = symmetric_source ? nothing :
                          similar(Xd, length(Xd))
    for it in 1:maxiter
        for n in 1:4
            rest = [k for k in 1:4 if k != n]
            gram = CUDA.ones(T, rank, rank)
            for k in rest
                gram .*= U[k]' * U[k]
            end
            gram .+= CUDA.CuArray(ridge .* Matrix{T}(I, rank, rank))
            unfolded = symmetric_source || n == 1 ?
                         reshape(Xd, size(Xd, 1), :) :
                         _mode_unfold_cuda!(unfolding_workspace, Xd, n)
            mttkrp = _cp_mttkrp_cuda(
                unfolded, U, rest, resolved_block)
            U[n] = mttkrp / gram
            λ .= CUDA.CuArray(_normalize_columns_cuda!(U[n]))
        end
        if it == 1 || it % check_every == 0 || it == maxiter
            err = _cp_relative_error_cuda(
                Xd, Xnorm^2, λ, U, resolved_block; symmetric_source)
            synchronize && CUDA.synchronize()
            push!(history, Float64(err))
            verbose && @info "CUDA CP-ALS" iter=it rank=rank relerr=err
            if length(history) >= 2 &&
               abs(history[end - 1] - history[end]) <= tol * max(history[end - 1], eps())
                break
            end
        end
    end
    cp = PulsePropagation.CPDecomposition(;
        λ=Array(λ),
        U=ntuple(n -> Array(U[n]), 4),
    )
    return cp, history
end

function _normalize_mmgnlse_cp_columns_cuda!(factor, floor_norm)
    norms = vec(sqrt.(sum(abs2, factor; dims=1)))
    safe = ifelse.(norms .<= floor_norm, one(eltype(norms)), norms)
    factor ./= reshape(safe, 1, :)
    return safe
end

function _cp_factorized_mttkrp_cuda(source_weights, source_factors,
                                    target_factors, mode::Int)
    source_rank = length(source_weights)
    target_rank = size(target_factors[1], 2)
    T = eltype(source_weights)
    cross = CUDA.ones(T, source_rank, target_rank)
    for other_mode in 1:4
        other_mode == mode && continue
        cross .*= transpose(source_factors[other_mode]) *
                  target_factors[other_mode]
    end
    cross .*= reshape(source_weights, :, 1)
    return source_factors[mode] * cross
end

function _cp_factorized_inner_product_cuda(left_weights, left_factors,
                                           right_weights, right_factors;
                                           block_size::Integer,
                                           tied::Bool=false)
    (isempty(left_weights) || isempty(right_weights)) &&
        return zero(eltype(left_weights))
    result = zero(promote_type(eltype(left_weights), eltype(right_weights)))
    right_rank = length(right_weights)
    for first_column in 1:Int(block_size):right_rank
        last_column = min(first_column + Int(block_size) - 1, right_rank)
        columns = first_column:last_column
        cross = CUDA.ones(
            eltype(left_weights), length(left_weights), length(columns))
        if tied
            cross .= transpose(left_factors[1]) *
                     view(right_factors[1], :, columns)
            cross .^= 4
        else
            for mode in 1:4
                cross .*= transpose(left_factors[mode]) *
                          view(right_factors[mode], :, columns)
            end
        end
        result += dot(left_weights, cross * view(right_weights, columns))
    end
    return result
end

function PulsePropagation._cp_relative_error_factorized_cuda(
    candidate::PulsePropagation.MMGNLSECPDecomposition,
    reference::PulsePropagation.MMGNLSECPDecomposition;
    block_size::Integer=256,
    device=nothing,
    synchronize::Bool=true,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    PulsePropagation._cp_require_compatible(candidate, reference)
    candidate_weights_cpu, candidate_factors_cpu =
        PulsePropagation._cp_real_factorized_source(candidate)
    reference_weights_cpu, reference_factors_cpu =
        PulsePropagation._cp_real_factorized_source(reference)
    candidate_tied = all(
        factor -> factor == candidate_factors_cpu[1],
        candidate_factors_cpu[2:4])
    reference_tied = all(
        factor -> factor == reference_factors_cpu[1],
        reference_factors_cpu[2:4])
    candidate_weights = CUDA.CuArray(candidate_weights_cpu)
    reference_weights = CUDA.CuArray(reference_weights_cpu)
    candidate_factors = [CUDA.CuArray(factor) for factor in candidate_factors_cpu]
    reference_factors = [CUDA.CuArray(factor) for factor in reference_factors_cpu]
    reference_norm_squared = real(_cp_factorized_inner_product_cuda(
        reference_weights, reference_factors,
        reference_weights, reference_factors;
        block_size, tied=reference_tied))
    reference_norm_squared > 0 ||
        return PulsePropagation.cp_iszero(candidate) ? 0.0 : Inf
    candidate_norm_squared = real(_cp_factorized_inner_product_cuda(
        candidate_weights, candidate_factors,
        candidate_weights, candidate_factors;
        block_size, tied=candidate_tied))
    cross = _cp_factorized_inner_product_cuda(
        reference_weights, reference_factors,
        candidate_weights, candidate_factors;
        block_size, tied=reference_tied && candidate_tied)
    synchronize && CUDA.synchronize()
    residual_squared = reference_norm_squared + candidate_norm_squared -
                       2real(cross)
    return sqrt(max(
        Float64(residual_squared / reference_norm_squared), 0.0))
end

function _cp_factorized_relative_error_cuda(
    source_weights, source_factors, source_norm_squared,
    target_weights, target_factors; block_size::Integer)
    target_norm_squared = real(_cp_factorized_inner_product_cuda(
        target_weights, target_factors, target_weights, target_factors;
        block_size))
    cross = _cp_factorized_inner_product_cuda(
        source_weights, source_factors, target_weights, target_factors;
        block_size)
    residual_squared = source_norm_squared + target_norm_squared - 2real(cross)
    return sqrt(max(Float64(residual_squared / source_norm_squared), 0.0))
end

function _cp_als_factorized_real_cuda(source_weights, source_factors,
                                      rank::Integer;
                                      seed::Integer,
                                      initial=nothing,
                                      maxiter::Integer,
                                      tolerance::Real,
                                      ridge::Real,
                                      check_every::Integer,
                                      target_error::Real,
                                      source_norm_squared,
                                      synchronize::Bool,
                                      error_block_size::Integer,
                                      verbose::Bool)
    r = Int(rank)
    RT = eltype(source_weights)
    dims = ntuple(mode -> size(source_factors[mode], 1), 4)
    rng = Random.MersenneTwister(Int(seed))
    factors_cpu = PulsePropagation._cp_real_random_factors(
        rng, RT, dims, r; initial)
    factors = [CUDA.CuArray(factor) for factor in factors_cpu]
    weights = CUDA.ones(RT, r)
    source_norm_squared > zero(RT) || throw(ArgumentError(
        "Use factorized CP compression only for a nonzero source."))
    identity_matrix = CUDA.CuArray(Matrix{RT}(I, r, r))
    floor_norm = sqrt(eps(RT))
    history = Float64[]
    previous_error = Inf
    converged = false
    iterations = 0

    for iteration in 1:Int(maxiter)
        iterations = iteration
        for mode in 1:4
            gram = CUDA.ones(RT, r, r)
            for other_mode in 1:4
                other_mode == mode && continue
                gram .*= transpose(factors[other_mode]) *
                         factors[other_mode]
            end
            ridge_scale = max(maximum(abs, gram), one(RT))
            gram .+= RT(ridge) * ridge_scale .* identity_matrix
            updated = _cp_factorized_mttkrp_cuda(
                source_weights, source_factors, factors, mode) / gram
            weights = _normalize_mmgnlse_cp_columns_cuda!(
                updated, floor_norm)
            factors[mode] = updated
        end

        if iteration == 1 || iteration % Int(check_every) == 0 ||
           iteration == Int(maxiter)
            err = _cp_factorized_relative_error_cuda(
                source_weights, source_factors, source_norm_squared,
                weights, factors; block_size=error_block_size)
            synchronize && CUDA.synchronize()
            push!(history, err)
            verbose && @info "CUDA factorized CP-ALS" iteration rank=r relative_error=err
            if err <= target_error
                converged = true
                break
            end
            if isfinite(previous_error) &&
               abs(previous_error - err) <= tolerance * max(previous_error, 1.0)
                converged = true
                break
            end
            previous_error = err
        end
    end
    return Array(weights), ntuple(mode -> Array(factors[mode]), 4),
           history, iterations, converged
end

function PulsePropagation._cp_compress_factorized_cuda(
    source::PulsePropagation.MMGNLSECPDecomposition;
    target_error::Real,
    max_rank::Integer,
    min_rank::Integer,
    rank_step::Integer,
    seed::Integer,
    initial=nothing,
    maxiter::Integer,
    tolerance::Real,
    ridge::Real,
    check_every::Integer,
    restarts::Integer,
    source_checksum=nothing,
    device=nothing,
    synchronize::Bool=true,
    error_block_size::Integer=256,
    verbose::Bool=false,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    source_weights_cpu, source_factors_cpu =
        PulsePropagation._cp_real_factorized_source(source)
    source_weights = CUDA.CuArray(source_weights_cpu)
    source_factors = [CUDA.CuArray(factor) for factor in source_factors_cpu]
    source_tied = all(
        factor -> factor == source_factors_cpu[1], source_factors_cpu[2:4])
    source_norm_squared = if source_tied
        correlation = transpose(source_factors[1]) * source_factors[1]
        correlation .^= 4
        real(dot(source_weights, correlation * source_weights))
    else
        real(_cp_factorized_inner_product_cuda(
            source_weights, source_factors, source_weights, source_factors;
            block_size=error_block_size))
    end
    metadata = source.metadata
    best = nothing
    warm = initial

    for rank in PulsePropagation._cp_factorized_rank_sequence(
        Int(min_rank), Int(rank_step), Int(max_rank))
        if rank == PulsePropagation.cp_rank(source)
            return PulsePropagation.MMGNLSECPDecomposition(
                source_weights_cpu, source_factors_cpu;
                layout=metadata.layout, nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum, relative_error=0.0, seed,
                iterations=0, converged=true, zero_tensor=false)
        end
        rank_best = nothing
        for restart in 1:Int(restarts)
            restart_initial = restart == 1 ? warm : nothing
            local_seed = Int(seed) + 104729 * rank + 1009 * (restart - 1)
            weights, factors, history, iterations, _ =
                _cp_als_factorized_real_cuda(
                    source_weights, source_factors, rank;
                    seed=local_seed, initial=restart_initial, maxiter,
                    tolerance, ridge, check_every, target_error,
                    source_norm_squared, synchronize,
                    error_block_size, verbose)
            err = history[end]
            candidate = PulsePropagation.MMGNLSECPDecomposition(
                weights, factors;
                layout=metadata.layout, nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum, relative_error=err, seed,
                iterations, converged=err <= target_error,
                zero_tensor=false)
            if rank_best === nothing ||
               candidate.relative_error < rank_best.relative_error
                rank_best = candidate
            end
        end
        warm = rank_best
        if best === nothing || rank_best.relative_error < best.relative_error
            best = rank_best
        end
        rank_best.relative_error <= target_error && return rank_best
    end

    throw(PulsePropagation.CPCompressionFailure(
        Float64(target_error), Int(max_rank), best))
end

function _cp_als_complex_cuda(X::AbstractArray{<:Number,4}, rank::Integer;
                              seed::Integer,
                              initial=nothing,
                              maxiter::Integer=300,
                              tolerance::Real=1e-10,
                              ridge::Real=1e-12,
                              check_every::Integer=5,
                              target_error::Real=0.0,
                              device=nothing,
                              synchronize::Bool=true,
                              rank_block_size=nothing,
                              max_workspace_bytes=nothing,
                              workspace_memory_fraction::Real=0.5,
                              reclaim_memory::Bool=true,
                              symmetric_source::Bool=false,
                              verbose::Bool=false)
    CUDA.functional() || Base.error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    r = Int(rank)
    r > 0 || throw(ArgumentError("rank must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    check_every > 0 || throw(ArgumentError("check_every must be positive."))
    _check_cuda_unfolding_addressability(X, symmetric_source)

    RT = typeof(float(real(zero(eltype(X)))))
    CT = Complex{RT}
    source = _cuda_complex_source(X, CT)
    # Avoid cuBLAS nrm2's 32-bit vector-length ceiling for large N^4 tensors.
    source_norm = sqrt(sum(abs2, source))
    source_norm > zero(RT) ||
        throw(ArgumentError("Use cp_compress for a zero tensor."))

    rng = Random.MersenneTwister(Int(seed))
    factors_cpu = PulsePropagation._complex_random_factors(
        rng, CT, size(X), r; initial)
    factors = [CUDA.CuArray(factor) for factor in factors_cpu]
    weights = CUDA.ones(RT, r)
    resolved_block, workspace_budget, bytes_per_column =
        _cuda_cp_rank_block_size(
            size(X, 1), r, CT;
            rank_block_size,
            max_workspace_bytes,
            workspace_memory_fraction,
            reclaim_memory,
            reserve_unfolding=!symmetric_source)
    if verbose
        @info "CUDA complex CP workspace" rank=r rank_block_size=resolved_block workspace_budget_bytes=workspace_budget bytes_per_rank_column=bytes_per_column symmetric_source
    end
    history = Float64[]
    previous_error = Inf
    converged = false
    iterations = 0
    floor_norm = sqrt(eps(RT))
    unfolding_workspace = symmetric_source ? nothing :
                          similar(source, length(source))

    for iteration in 1:Int(maxiter)
        iterations = iteration
        for mode in 1:4
            rest = [index for index in 1:4 if index != mode]
            gram = CUDA.ones(CT, r, r)
            for index in rest
                gram .*= transpose(factors[index]) * conj.(factors[index])
            end
            ridge_scale = max(maximum(abs, gram), one(RT))
            gram .+= CUDA.CuArray(
                RT(ridge) * ridge_scale .* Matrix{CT}(I, r, r))
            unfolded = symmetric_source || mode == 1 ?
                         reshape(source, size(source, 1), :) :
                         _mode_unfold_cuda!(
                             unfolding_workspace, source, mode)
            mttkrp = _cp_mttkrp_cuda(
                unfolded, factors, rest, resolved_block;
                conjugate_factors=true)
            updated = mttkrp / gram
            weights = _normalize_mmgnlse_cp_columns_cuda!(
                updated, floor_norm)
            factors[mode] = updated
        end

        if iteration == 1 ||
           iteration % Int(check_every) == 0 ||
           iteration == Int(maxiter)
            err = _cp_relative_error_cuda(
                source, source_norm^2, weights, factors,
                resolved_block; symmetric_source)
            synchronize && CUDA.synchronize()
            push!(history, err)
            if err <= target_error
                converged = true
                break
            end
            if isfinite(previous_error) &&
               abs(previous_error - err) <=
               tolerance * max(previous_error, 1.0)
                converged = true
                break
            end
            previous_error = err
        end
    end

    return Array(weights),
           ntuple(index -> Array(factors[index]), 4),
           history,
           iterations,
           converged
end

function PulsePropagation._cp_compress_cuda(
    S::AbstractArray{<:Number,4};
    target_error::Real=1e-6,
    error=nothing,
    max_rank::Integer=1024,
    min_rank::Integer=1,
    rank_step::Integer=1,
    seed::Integer=0x4d4d474e,
    maxiter::Integer=300,
    tolerance::Real=1e-10,
    ridge::Real=1e-12,
    check_every::Integer=5,
    restarts::Integer=1,
    layout::Symbol=:spatial,
    nmodes::Union{Nothing,Integer}=nothing,
    npolarizations::Integer=1,
    source_checksum=nothing,
    checksum_fn=nothing,
    fftw_threads::Integer=1,
    blas_threads::Integer=1,
    device=nothing,
    synchronize::Bool=true,
    rank_block_size=nothing,
    max_workspace_bytes=nothing,
    workspace_memory_fraction::Real=0.5,
    reclaim_memory::Bool=true,
    symmetric_source::Bool=false,
    check_symmetry::Bool=true,
)
    resolved_error = error === nothing ? target_error : float(error)
    resolved_error >= 0 ||
        throw(ArgumentError("error must be nonnegative."))
    max_rank > 0 || throw(ArgumentError("max_rank must be positive."))
    min_rank > 0 || throw(ArgumentError("min_rank must be positive."))
    min_rank <= max_rank ||
        throw(ArgumentError("min_rank cannot exceed max_rank."))
    rank_step > 0 || throw(ArgumentError("rank_step must be positive."))
    restarts > 0 || throw(ArgumentError("restarts must be positive."))
    fftw_threads > 0 ||
        throw(ArgumentError("fftw_threads must be positive."))
    blas_threads > 0 ||
        throw(ArgumentError("blas_threads must be positive."))
    PulsePropagation._cp_validate_memory_controls(
        rank_block_size, max_workspace_bytes, workspace_memory_fraction)
    _check_cuda_unfolding_addressability(S, symmetric_source)
    all(isfinite, S) ||
        throw(ArgumentError("The overlap tensor contains non-finite values."))
    CUDA.functional() || Base.error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    RT = typeof(float(real(zero(eltype(S)))))
    CT = Complex{RT}
    if symmetric_source && check_symmetry
        symmetry_defect = PulsePropagation._cp_symmetric_tensor_defect(S)
        symmetry_defect <= max(1e-12, 100eps(RT)) || throw(ArgumentError(
            "symmetric_source=true requires a fully permutation-symmetric " *
            "tensor; measured relative defect $symmetry_defect."))
    end
    checksum = PulsePropagation._resolved_checksum(
        S; source_checksum, checksum_fn)
    metadata = PulsePropagation._cp_source_metadata(
        size(S); layout, nmodes, npolarizations,
        source_checksum=checksum)
    # cuBLAS nrm2 uses a 32-bit vector length. The reduction remains valid
    # for the large symmetric sources admitted by the bounded path.
    source_norm = sqrt(sum(abs2, S))
    if source_norm == 0
        factors = ntuple(index -> zeros(CT, size(S, index), 0), 4)
        return PulsePropagation.MMGNLSECPDecomposition(
            Float64[], factors;
            layout=metadata.layout,
            nmodes=metadata.nmodes,
            npolarizations=metadata.npolarizations,
            source_checksum=checksum,
            relative_error=0.0,
            seed,
            iterations=0,
            converged=true,
            zero_tensor=true)
    end

    source_device = _cuda_complex_source(S, CT)

    best = nothing
    warm = nothing
    for rank in Int(min_rank):Int(rank_step):Int(max_rank)
        rank_best = nothing
        for restart in 1:Int(restarts)
            initial = restart == 1 ? warm : nothing
            local_seed =
                Int(seed) + 104729 * rank + 1009 * (restart - 1)
            weights, factors, history, iterations, als_converged =
                _cp_als_complex_cuda(
                    source_device, rank;
                    seed=local_seed,
                    initial,
                    maxiter,
                    tolerance,
                    ridge,
                    check_every,
                    target_error=resolved_error,
                    device,
                    synchronize,
                    rank_block_size,
                    max_workspace_bytes,
                    workspace_memory_fraction,
                    reclaim_memory,
                    symmetric_source)
            err = isempty(history) ?
                  PulsePropagation._cp_relative_error(
                      CT.(S), weights, factors) :
                  history[end]
            candidate = PulsePropagation.MMGNLSECPDecomposition(
                weights, factors;
                layout=metadata.layout,
                nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum=checksum,
                relative_error=err,
                seed,
                iterations,
                converged=err <= resolved_error,
                zero_tensor=false)
            if rank_best === nothing ||
               candidate.relative_error < rank_best.relative_error
                rank_best = candidate
            end
        end
        warm = rank_best
        if best === nothing ||
           rank_best.relative_error < best.relative_error
            best = rank_best
        end
        rank_best.relative_error <= resolved_error && return rank_best
    end

    throw(PulsePropagation.CPCompressionFailure(
        Float64(resolved_error), Int(max_rank), best))
end

function PulsePropagation._cp_compress_symmetric_experimental_cuda(
    source::AbstractArray{RT,4}, rank::Integer;
    initial=nothing,
    maxiter::Integer,
    check_every::Integer,
    learning_rate::Real,
    ridge::Real,
    tolerance::Real,
    target_error,
    initial_maxiter::Integer,
    initial_tolerance::Real,
    seed::Integer,
    metadata,
    checksum,
    verbose::Bool,
    device=nothing,
    synchronize::Bool=true,
    rank_block_size=nothing,
    max_workspace_bytes=nothing,
    workspace_memory_fraction::Real=0.5,
    reclaim_memory::Bool=true,
) where {RT<:AbstractFloat}
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    r = Int(rank)
    rng = Random.MersenneTwister(Int(seed))
    source_device = source isa CUDA.CuArray{RT,4} ? source :
                    CUDA.CuArray(source)
    warm = initial
    if warm === nothing
        warm, _ = PulsePropagation.cp_als_warm_cuda(
            source_device, r;
            maxiter=Int(initial_maxiter),
            tol=initial_tolerance,
            ridge=RT(ridge),
            check_every=Int(check_every),
            rng,
            verbose,
            device,
            synchronize,
            rank_block_size,
            max_workspace_bytes,
            workspace_memory_fraction,
            reclaim_memory,
            symmetric_source=true,
        )
    end
    factor_cpu = PulsePropagation._cp_symmetric_initial_factor(
        warm, size(source, 1), r, rng, RT)
    factor = CUDA.CuArray(factor_cpu)
    identity = CUDA.CuArray(Matrix{RT}(I, r, r))
    resolved_block, workspace_budget, bytes_per_column =
        _cuda_cp_rank_block_size(
            size(source, 1), r, RT;
            rank_block_size,
            max_workspace_bytes,
            workspace_memory_fraction,
            reclaim_memory)
    if verbose
        @info "CUDA symmetric CP workspace" rank=r rank_block_size=resolved_block workspace_budget_bytes=workspace_budget bytes_per_rank_column=bytes_per_column
    end
    synchronize && CUDA.synchronize()
    weights, best_factor, relative_error, iterations, converged =
        PulsePropagation._cp_symmetric_optimize(
            source_device, factor, identity;
            maxiter=Int(maxiter),
            check_every=Int(check_every),
            learning_rate,
            ridge,
            tolerance,
            target_error,
            rank_block_size=resolved_block,
            verbose,
            synchronize_fn=synchronize ? CUDA.synchronize : () -> nothing,
        )
    synchronize && CUDA.synchronize()
    return PulsePropagation._cp_symmetric_result(
        weights, best_factor, relative_error, iterations, converged,
        metadata, checksum, seed)
end

function _assert_cuda_array(x, name)
    x isa CUDA.CuArray || error("$name must be a CuArray for the in-place CUDA RHS.")
    return x
end

function _full_kerr_kernel!(nonlinear, at, sk, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        i1 = (idx - 1) ÷ nt + 1
        acc = zero(eltype(nonlinear))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
            acc += sk[i1, i2, i3, i4] * at[t, i2] * at[t, i3] * conj(at[t, i4])
        end
        @inbounds nonlinear[t, i1] = acc
    end
    return nothing
end

function _raman_pairs_kernel!(pairs, at, sra, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        rem = (idx - 1) ÷ nt
        i1 = rem % nm + 1
        i2 = rem ÷ nm + 1
        acc = zero(eltype(pairs))
        @inbounds for i4 in 1:nm, i3 in 1:nm
            acc += sra[i1, i2, i3, i4] * at[t, i3] * conj(at[t, i4])
        end
        @inbounds pairs[t, i1, i2] = acc
    end
    return nothing
end

function _raman_combine_kernel!(nonlinear, pairs, at, nt::Int, nm::Int, total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        i1 = (idx - 1) ÷ nt + 1
        acc = zero(eltype(nonlinear))
        @inbounds for i2 in 1:nm
            acc += pairs[t, i1, i2] * at[t, i2]
        end
        @inbounds nonlinear[t, i1] += acc
    end
    return nothing
end

function _launch_1d(kernel, total::Int, args...)
    threads = 256
    blocks = cld(total, threads)
    @cuda threads=threads blocks=blocks kernel(args..., total)
    return nothing
end

function _adjoint_delta_kernel!(delta1, delta2, ut, gamma, nt::Int, nm::Int,
                                total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        rem = (idx - 1) ÷ nt
        i = rem % nm + 1
        j = rem ÷ nm + 1
        acc1 = zero(eltype(delta1))
        acc2 = zero(eltype(delta2))
        @inbounds for k in 1:nm, l in 1:nm
            g = gamma[l, k, i, j]
            uk = ut[t, k]
            ul = ut[t, l]
            acc1 += g * real(uk * conj(ul))
            acc2 += g * uk * ul
        end
        @inbounds begin
            delta1[t, i, j] = acc1
            delta2[t, i, j] = acc2
        end
    end
    return nothing
end

function _adjoint_terms_kernel!(term1_t, term2_t, lambdat, lambdawc_t,
                                delta1, delta2, one_m_fR, nt::Int, nm::Int,
                                total::Int)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if idx <= total
        t = (idx - 1) % nt + 1
        j = (idx - 1) ÷ nt + 1
        acc1 = zero(eltype(term1_t))
        acc2 = zero(eltype(term2_t))
        @inbounds for i in 1:nm
            acc1 += lambdat[t, i] * (2 * one_m_fR * delta1[t, i, j])
            acc2 += lambdawc_t[t, i] * delta2[t, i, j]
        end
        @inbounds begin
            term1_t[t, j] = acc1
            term2_t[t, j] = acc2
        end
    end
    return nothing
end

function _cuda_adjoint_full_rhs(lambda_tilde::CUDA.CuMatrix{Complex{T}},
                                u_tilde::CUDA.CuMatrix{Complex{T}},
                                z::T,
                                d_op::CUDA.CuMatrix{Complex{T}},
                                tau::CUDA.CuVector{T},
                                gamma::CUDA.CuArray{T,4},
                                one_m_fR::T) where {T}
    nt, nm = size(lambda_tilde)
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdat = fft(exp_p .* lambda_tilde .* reshape(tau, :, 1), 1)
    lambdawc_t = conj.(lambdat)
    ut = fft(exp_p .* u_tilde, 1)

    delta1 = CUDA.zeros(T, nt, nm, nm)
    delta2 = CUDA.zeros(Complex{T}, nt, nm, nm)
    _launch_1d(_adjoint_delta_kernel!, nt * nm * nm,
               delta1, delta2, ut, gamma, nt, nm)

    term1_t = CUDA.zeros(Complex{T}, nt, nm)
    term2_t = similar(term1_t)
    _launch_1d(_adjoint_terms_kernel!, nt * nm,
               term1_t, term2_t, lambdat, lambdawc_t, delta1, delta2,
               one_m_fR, nt, nm)

    term1 = 1im .* exp_m .* ifft(term1_t, 1)
    term2 = -1im .* exp_m .* ifft(term2_t, 1)
    return term1 .+ one_m_fR .* term2
end

function _validated_cuda_adjoint_cp(cp, ::Type{T}) where {T}
    cpd = PulsePropagation._as_cp(cp)
    all(isreal, cpd.λ) && all(U -> all(isreal, U), cpd.U) || error(
        "CUDA rank-channel adjoints currently require real CP weights and factors.")
    nm = size(cpd.U[1], 1)
    tensor = PulsePropagation.cp_reconstruct(cpd, (nm, nm, nm, nm))
    tensor_norm = max(norm(tensor), eps(T))
    symmetry_defect = maximum(
        norm(tensor .- permutedims(tensor, permutation)) / tensor_norm
        for permutation in ((2, 1, 3, 4), (1, 3, 2, 4),
                            (1, 2, 4, 3)))
    symmetry_defect <= T(1e-5) || error(
        "CUDA rank-channel adjoints require a permutation-symmetric physical " *
        "overlap tensor; reconstructed CP symmetry defect $symmetry_defect " *
        "exceeds 1e-5.")
    return cpd
end

function _cuda_cp_adjoint_factors(cp, ::Type{T}) where {T}
    cpd = _validated_cuda_adjoint_cp(cp, T)
    λ_cpu = Complex{T}.(cpd.λ)
    U4_cpu = Complex{T}.(cpd.U[4])
    λ = CUDA.CuArray(λ_cpu)
    U1 = CUDA.CuArray(Complex{T}.(cpd.U[1]))
    U2 = CUDA.CuArray(Complex{T}.(cpd.U[2]))
    U3 = CUDA.CuArray(Complex{T}.(cpd.U[3]))
    U4 = CUDA.CuArray(U4_cpu)
    WU3t = CUDA.CuArray(Matrix(transpose(Complex{T}.(cpd.U[3]) .*
                                          reshape(λ_cpu, 1, :))))
    WU4t = CUDA.CuArray(Matrix(transpose(U4_cpu .* reshape(λ_cpu, 1, :))))
    return (; λ, U1, U2, U3, U4, WU3t, WU4t)
end

function _cuda_adjoint_cp_rhs(lambda_tilde::CUDA.CuMatrix{Complex{T}},
                              u_tilde::CUDA.CuMatrix{Complex{T}},
                              z::T,
                              d_op::CUDA.CuMatrix{Complex{T}},
                              tau::CUDA.CuVector{T},
                              cp,
                              one_m_fR::T) where {T}
    exp_p = exp.(d_op .* z)
    exp_m = exp.(-d_op .* z)

    lambdat = fft(exp_p .* lambda_tilde .* reshape(tau, :, 1), 1)
    lambdawc_t = conj.(lambdat)
    ut = fft(exp_p .* u_tilde, 1)

    B1 = ut * cp.U1
    B2 = ut * cp.U2
    C1 = real.(B2 .* conj.(B1))
    C2 = B2 .* B1
    L3 = lambdat * cp.U3
    LC3 = lambdawc_t * cp.U3

    term1_t = (2 * one_m_fR) .* ((L3 .* Complex{T}.(C1)) * cp.WU4t)
    term2_t = (LC3 .* C2) * cp.WU4t
    term1 = 1im .* exp_m .* ifft(term1_t, 1)
    term2 = -1im .* exp_m .* ifft(term2_t, 1)
    return term1 .+ one_m_fR .* term2
end

function _cuda_adjoint_cp_raman_rhs(
    lambda_tilde::CUDA.CuMatrix{Complex{T}},
    u_tilde::CUDA.CuMatrix{Complex{T}},
    z::T,
    d_op::CUDA.CuMatrix{Complex{T}},
    tau::CUDA.CuVector{T},
    cp,
    one_m_fR::T,
    hR_sigma::CUDA.CuVector{Complex{T}},
    cache::CUDACPAdjointRamanCache,
) where {T}
    nt, nm = size(lambda_tilde)
    iseven(nt) || error("CUDA Agarwal Raman adjoints require an even time-grid size.")
    size(u_tilde) == (nt, nm) || error("u_tilde and lambda_tilde must have equal size.")
    length(hR_sigma) == nt || error("The CUDA Raman response must have length $nt.")

    cache.exp_p .= exp.(d_op .* z)
    cache.exp_m .= exp.(-d_op .* z)

    cache.lambdat .= cache.exp_p .* lambda_tilde .* reshape(tau, :, 1)
    cache.fft_lambdat! * cache.lambdat
    cache.lambdawc_t .= conj.(cache.lambdat)
    cache.ut .= cache.exp_p .* u_tilde
    cache.fft_ut! * cache.ut

    B1 = cache.ut * cp.U1
    B2 = cache.ut * cp.U2
    C1 = real.(B2 .* conj.(B1))
    L3 = cache.lambdat * cp.U3

    cache.term_t .= (2 * one_m_fR) .* ((L3 .* Complex{T}.(C1)) * cp.WU4t)

    # fftshift(ifft(X)) == ifft(sigma .* X) for an even-length first axis,
    # where sigma[k] = (-1)^(k-1).  hR_sigma stores hRω .* sigma.
    cache.rankwork .= Complex{T}.(C1)
    cache.fft_rank! * cache.rankwork
    cache.rankwork .*= reshape(hR_sigma, :, 1)
    cache.ifft_rank! * cache.rankwork
    cache.term_t .+= (L3 .* Complex{T}.(real.(cache.rankwork))) * cp.WU4t

    cache.rankwork .= (cache.lambdat * cp.U1) .* (conj.(cache.ut) * cp.U2)
    cache.ifft_rank! * cache.rankwork
    cache.rankwork .*= reshape(hR_sigma, :, 1)
    cache.fft_rank! * cache.rankwork
    U4u = cache.ut * cp.U4
    cache.term_t .+= (cache.rankwork .* U4u) * cp.WU3t

    cache.ifft_term! * cache.term_t
    cache.rhs .= 1im .* cache.exp_m .* cache.term_t

    C2 = B2 .* B1
    LC3 = cache.lambdawc_t * cp.U3
    cache.term_t .= (LC3 .* C2) * cp.WU4t
    cache.ifft_term! * cache.term_t
    cache.rhs .+= (-1im * one_m_fR) .* cache.exp_m .* cache.term_t

    cache.rankwork .= (cache.lambdawc_t * cp.U1) .* (cache.ut * cp.U2)
    cache.ifft_rank! * cache.rankwork
    cache.rankwork .*= reshape(hR_sigma, :, 1)
    cache.fft_rank! * cache.rankwork
    cache.term_t .= (cache.rankwork .* U4u) * cp.WU3t
    cache.ifft_term! * cache.term_t
    cache.rhs .+= -1im .* cache.exp_m .* cache.term_t

    # The RK4 driver retains four RHS stages, so return a distinct device
    # allocation while keeping all FFT work arrays and plans cached.
    return copy(cache.rhs)
end

function _cuda_interpolate_forward(u_tilde::CUDA.CuArray{Complex{T},3},
                                   zgrid::AbstractVector{T}, z::T) where {T}
    nsave = length(zgrid)
    z <= zgrid[1] && return u_tilde[:, :, 1]
    z >= zgrid[end] && return u_tilde[:, :, end]
    q = searchsortedlast(zgrid, z)
    q = clamp(q, 1, nsave - 1)
    z0 = zgrid[q]
    z1 = zgrid[q + 1]
    θ = (z - z0) / (z1 - z0)
    return (one(T) - θ) .* u_tilde[:, :, q] .+ θ .* u_tilde[:, :, q + 1]
end

function _cuda_adjoint_rk4(rhs, y::CUDA.CuMatrix{Complex{T}}, z::T,
                           h::T) where {T}
    k1 = rhs(y, z)
    k2 = rhs(y .+ (h / 2) .* k1, z + h / 2)
    k3 = rhs(y .+ (h / 2) .* k2, z + h / 2)
    k4 = rhs(y .+ h .* k3, z + h)
    return y .+ (h / 6) .* (k1 .+ 2 .* k2 .+ 2 .* k3 .+ k4)
end

function _cuda_adjoint_save_points(zgrid::AbstractVector{T}, zsave_adj,
                                   return_lambdaw_zsave::Bool) where {T}
    if return_lambdaw_zsave
        return zsave_adj === nothing ? reverse(T.(zgrid)) :
               reverse(sort(T.(collect(zsave_adj))))
    end
    return T[zgrid[1]]
end

function _cuda_integrate_adjoint(lambda_terminal::AbstractMatrix{Complex{T}},
                                 forward::PulsePropagation.PropagationOutput{T},
                                 fiber::PulsePropagation.Fiber{T},
                                 sim::PulsePropagation.Simulation{T},
                                 rhs_builder;
                                 dz_adj::T=sim.dz,
                                 zsave_adj=nothing,
                                 return_lambdaw_zsave::Bool=false,
                                 synchronize::Bool=true) where {T}
    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    size(lambda_terminal) == (nt, nm) ||
        error("lambda_terminal must have size $(nt) x $(nm).")
    length(forward.z) == size(forward.fields, 3) ||
        error("forward.z and forward.fields have inconsistent save counts.")
    all(diff(forward.z) .> zero(T)) ||
        error("forward.z must be strictly increasing.")
    dz_adj > zero(T) || error("dz_adj must be positive.")

    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    omega = PulsePropagation.ifft_frequency_grid(nt, forward.dt)
    d_op, _ = PulsePropagation.calc_d_op(fiber2, sim, nt, forward.dt, omega, fields0)
    zgrid = T.(forward.z)
    tau = (omega .+ T(2π) * sim.f0) ./ (T(2π) * sim.f0)
    tau .*= T.(PulsePropagation.damped_freq_window(nt))

    d_op_d = CUDA.CuArray(d_op)
    tau_d = CUDA.CuArray(tau)
    u_tilde = Array{Complex{T},3}(undef, nt, nm, length(zgrid))
    @inbounds for q in eachindex(zgrid)
        aw = PulsePropagation.inverse_fft(forward.fields[:, :, q], dims=1)
        u_tilde[:, :, q] = exp.(-d_op .* zgrid[q]) .* aw
    end
    u_tilde_d = CUDA.CuArray(u_tilde)
    lambda_d = CUDA.CuArray(exp.(-d_op .* zgrid[end]) .*
                            Matrix{Complex{T}}(lambda_terminal))

    rhs = rhs_builder(d_op_d, tau_d, u_tilde_d, zgrid)
    save_desc = _cuda_adjoint_save_points(zgrid, zsave_adj, return_lambdaw_zsave)
    all((zgrid[1] .<= save_desc) .& (save_desc .<= zgrid[end])) ||
        error("zsave_adj points must lie within the forward z interval.")

    saved_z = T[]
    saved_states = Vector{Array{Complex{T},2}}()
    current_z = zgrid[end]
    for target_z in save_desc
        target_z <= current_z + sqrt(eps(T)) ||
            error("zsave_adj points must be sorted within the forward z interval.")
        while current_z - target_z > sqrt(eps(T)) * max(one(T), abs(current_z))
            step = min(T(dz_adj), current_z - target_z)
            h = -step
            lambda_d = _cuda_adjoint_rk4(rhs, lambda_d, current_z, h)
            current_z += h
        end
        current_z = target_z
        synchronize && CUDA.synchronize()
        push!(saved_z, current_z)
        if return_lambdaw_zsave
            push!(saved_states, Array(exp.(d_op_d .* current_z) .* lambda_d))
        else
            push!(saved_states, Array(lambda_d))
        end
    end

    if !return_lambdaw_zsave
        out = Array{Complex{T},3}(undef, nt, nm, 1)
        out[:, :, 1] = saved_states[end]
        return PulsePropagation.AdjointOutput{T}(; z=[zgrid[1]], lambdaw=out,
                                                dt=forward.dt,
                                                frame=:interaction)
    end

    z_asc = reverse(saved_z)
    states_asc = reverse(saved_states)
    out = Array{Complex{T},3}(undef, nt, nm, length(states_asc))
    @inbounds for q in eachindex(states_asc)
        out[:, :, q] = states_asc[q]
    end
    return PulsePropagation.AdjointOutput{T}(; z=collect(z_asc), lambdaw=out,
                                            dt=forward.dt, frame=:lab)
end

function PulsePropagation.n_op_rk4ip_full_cuda!(
    out::CUDA.CuMatrix{Complex{T}},
    aw::CUDA.CuMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDAFullRHSCache{T},
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    isempty(hbw) || error("CUDA full RHS currently supports scalar propagation with no separate hbw path.")
    n2_prefactor = _assert_cuda_array(n2_prefactor, "n2_prefactor")
    haw = _assert_cuda_array(haw, "haw")
    at_noise === nothing || at_noise isa CUDA.CuArray ||
        error("at_noise must be nothing or a CuArray for the in-place CUDA RHS.")

    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")
    cache.nm == nm || error("CUDA full RHS cache was built for $(cache.nm) modes, got $nm.")

    cache.at .= aw
    cache.fft_modes! * cache.at
    at_noise === nothing || (cache.at .+= at_noise)

    total_modes = nt * nm
    _launch_1d(_full_kerr_kernel!, total_modes, cache.nonlinear,
               cache.at, cache.sk, nt, nm)

    if sim.include_Raman && !isempty(haw)
        _launch_1d(_raman_pairs_kernel!, nt * nm * nm, cache.raman_pairs,
                   cache.at, cache.sra, nt, nm)
        cache.ifft_pairs! * cache.raman_pairs
        cache.raman_pairs .*= reshape(haw, :, 1, 1)
        cache.fft_pairs! * cache.raman_pairs
        _launch_1d(_raman_combine_kernel!, total_modes, cache.nonlinear,
                   cache.raman_pairs, cache.at, nt, nm)
    end

    cache.ifft_modes! * cache.nonlinear
    out .= cache.nonlinear .* reshape(n2_prefactor, :, 1)
    return out
end

function PulsePropagation.n_op_rk4ip_full_cuda(
    aw::AbstractMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDAFullRHSCache{T},
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    awd = aw isa CUDA.CuArray ? aw : CUDA.CuArray(aw)
    out = similar(awd)
    n2d = n2_prefactor isa CUDA.CuArray ? n2_prefactor : CUDA.CuArray(n2_prefactor)
    hawd = haw isa CUDA.CuArray ? haw : CUDA.CuArray(haw)
    hbwd = hbw isa CUDA.CuArray ? hbw : CUDA.CuArray(hbw)
    noise_d = at_noise === nothing || at_noise isa CUDA.CuArray ? at_noise : CUDA.CuArray(at_noise)
    return PulsePropagation.n_op_rk4ip_full_cuda!(out, awd, sim, n2d, cache,
                                         hawd, hbwd, noise_d)
end

function PulsePropagation.n_op_rk4ip_cp_cuda!(
    out::CUDA.CuMatrix{Complex{T}},
    aw::CUDA.CuMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDACPFastRHSCache{T},
    f_raman::T,
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    isempty(hbw) || error("CUDA CP RHS currently supports scalar CP propagation with no separate hbw path.")
    n2_prefactor = _assert_cuda_array(n2_prefactor, "n2_prefactor")
    haw = _assert_cuda_array(haw, "haw")
    at_noise === nothing || at_noise isa CUDA.CuArray ||
        error("at_noise must be nothing or a CuArray for the in-place CUDA RHS.")

    nt, nm = size(aw)
    size(out) == (nt, nm) || error("out must have size $(size(aw)).")

    cache.at .= aw
    cache.fft_modes! * cache.at
    at_noise === nothing || (cache.at .+= at_noise)

    mul!(cache.b2, cache.at, cache.u2)
    mul!(cache.b3, cache.at, cache.u3)
    cache.conj_at .= conj.(cache.at)
    mul!(cache.b4, cache.conj_at, cache.u4)

    cache.p .= cache.b2 .* cache.b3 .* cache.b4
    mul!(cache.k, cache.p, cache.wu1t)
    cache.nonlinear .= (one(T) - f_raman) .* cache.k

    if sim.include_Raman && !isempty(haw)
        cache.conv .= cache.b3 .* cache.b4
        cache.ifft_rank! * cache.conv
        cache.conv .*= reshape(haw, :, 1)
        cache.fft_rank! * cache.conv
        cache.p .= cache.b2 .* cache.conv
        mul!(cache.k, cache.p, cache.wu1t)
        cache.nonlinear .+= cache.k
    end

    cache.ifft_modes! * cache.nonlinear
    out .= cache.nonlinear .* reshape(n2_prefactor, :, 1)
    return out
end

function PulsePropagation.n_op_rk4ip_cp_cuda(
    aw::AbstractMatrix{Complex{T}},
    sim::PulsePropagation.Simulation{T},
    n2_prefactor,
    cache::CUDACPFastRHSCache{T},
    f_raman::T,
    haw,
    hbw,
    at_noise=nothing,
) where {T}
    awd = aw isa CUDA.CuArray ? aw : CUDA.CuArray(aw)
    out = similar(awd)
    n2d = n2_prefactor isa CUDA.CuArray ? n2_prefactor : CUDA.CuArray(n2_prefactor)
    hawd = haw isa CUDA.CuArray ? haw : CUDA.CuArray(haw)
    hbwd = hbw isa CUDA.CuArray ? hbw : CUDA.CuArray(hbw)
    noise_d = at_noise === nothing || at_noise isa CUDA.CuArray ? at_noise : CUDA.CuArray(at_noise)
    return PulsePropagation.n_op_rk4ip_cp_cuda!(out, awd, sim, n2d, cache, f_raman,
                                       hawd, hbwd, noise_d)
end

function _rk4ip_full_cuda_step!(a1w::CUDA.CuMatrix{Complex{T}},
                                a0w::CUDA.CuMatrix{Complex{T}},
                                dz::T,
                                half::CUDA.CuMatrix{Complex{T}},
                                a_ip::CUDA.CuMatrix{Complex{T}},
                                stage::CUDA.CuMatrix{Complex{T}},
                                k1::CUDA.CuMatrix{Complex{T}},
                                k2::CUDA.CuMatrix{Complex{T}},
                                k3::CUDA.CuMatrix{Complex{T}},
                                k4::CUDA.CuMatrix{Complex{T}},
                                sim::PulsePropagation.Simulation{T},
                                n2_prefactor::CUDA.CuVector{Complex{T}},
                                cache::CUDAFullRHSCache{T},
                                haw::CUDA.CuVector{Complex{T}},
                                hbw::CUDA.CuVector{Complex{T}},
                                at_noise=nothing) where {T}
    a_ip .= half .* a0w

    PulsePropagation.n_op_rk4ip_full_cuda!(k1, a0w, sim, n2_prefactor, cache, haw, hbw, at_noise)
    k1 .= half .* k1

    stage .= a_ip .+ k1 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_full_cuda!(k2, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    stage .= a_ip .+ k2 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_full_cuda!(k3, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    stage .= half .* (a_ip .+ k3 .* dz)
    PulsePropagation.n_op_rk4ip_full_cuda!(k4, stage, sim, n2_prefactor, cache, haw, hbw, at_noise)

    a1w .= half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (dz / 6)) .+ k4 .* (dz / 6)
    return a1w
end

function _rk4ip_cp_cuda_step!(a1w::CUDA.CuMatrix{Complex{T}},
                              a0w::CUDA.CuMatrix{Complex{T}},
                              dz::T,
                              half::CUDA.CuMatrix{Complex{T}},
                              a_ip::CUDA.CuMatrix{Complex{T}},
                              stage::CUDA.CuMatrix{Complex{T}},
                              k1::CUDA.CuMatrix{Complex{T}},
                              k2::CUDA.CuMatrix{Complex{T}},
                              k3::CUDA.CuMatrix{Complex{T}},
                              k4::CUDA.CuMatrix{Complex{T}},
                              sim::PulsePropagation.Simulation{T},
                              n2_prefactor::CUDA.CuVector{Complex{T}},
                              cache::CUDACPFastRHSCache{T},
                              f_raman::T,
                              haw::CUDA.CuVector{Complex{T}},
                              hbw::CUDA.CuVector{Complex{T}},
                              at_noise=nothing) where {T}
    a_ip .= half .* a0w

    PulsePropagation.n_op_rk4ip_cp_cuda!(k1, a0w, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)
    k1 .= half .* k1

    stage .= a_ip .+ k1 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k2, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    stage .= a_ip .+ k2 .* (dz / 2)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k3, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    stage .= half .* (a_ip .+ k3 .* dz)
    PulsePropagation.n_op_rk4ip_cp_cuda!(k4, stage, sim, n2_prefactor, cache, f_raman, haw, hbw, at_noise)

    a1w .= half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (dz / 6)) .+ k4 .* (dz / 6)
    return a1w
end

function PulsePropagation.propagate_rk4ip_full_cuda(fiber::PulsePropagation.Fiber{T},
                                           initial_condition::PulsePropagation.InitialCondition{T},
                                           sim::PulsePropagation.Simulation{T};
                                           zsave=nothing,
                                           dz=nothing,
                                           device=nothing,
                                           synchronize::Bool=true) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.gain_model == 0 ||
        error("propagate_rk4ip_full_cuda currently supports passive propagation only.")

    fields0 = PulsePropagation.last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    step_dz = dz === nothing ? sim.dz : T(dz)
    step_dz > zero(T) || error("dz must be positive.")
    omega = PulsePropagation.ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = PulsePropagation.calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(PulsePropagation.damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = PulsePropagation.raman_model(fiber2, sim, nt, initial_condition.dt)
    isempty(hbw) || error("propagate_rk4ip_full_cuda currently supports scalar propagation with no separate hbw path.")
    srsk = PulsePropagation.calc_srsk(fiber3, sim, size(fiber3.sr, 1))

    save_z = if zsave === nothing
        save_period = sim.save_period == 0 ? fiber2.L0 : sim.save_period
        collect(T, 0:round(Int, fiber2.L0 / save_period)) .* save_period
    else
        T.(collect(zsave))
    end
    isempty(save_z) && error("zsave must contain at least one point.")
    first(save_z) == zero(T) || error("zsave must start at 0.")
    last(save_z) == fiber2.L0 || error("zsave must end at fiber.L0.")
    all(diff(save_z) .> zero(T)) || error("zsave must be strictly increasing.")

    half = CUDA.CuArray(exp.(d_op .* (step_dz / 2)))
    aw = CUDA.CuArray(PulsePropagation.inverse_fft(fields0, dims=1))
    next_aw = similar(aw)
    a_ip = similar(aw)
    stage = similar(aw)
    k1 = similar(aw)
    k2 = similar(aw)
    k3 = similar(aw)
    k4 = similar(aw)
    n2_prefactor_d = CUDA.CuArray(n2_prefactor)
    haw_d = CUDA.CuArray(haw)
    hbw_d = CUDA.CuArray(hbw)
    cache = PulsePropagation.cuda_full_rhs_cache(srsk, nt, nm; T=T)

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    out[:, :, 1] = fields0

    zprev = zero(T)
    @inbounds for q in 2:length(save_z)
        znext = save_z[q]
        nsteps = PulsePropagation._fixed_step_count(znext - zprev, step_dz; label="save interval")
        for _ in 1:nsteps
            _rk4ip_full_cuda_step!(next_aw, aw, step_dz, half, a_ip, stage,
                                   k1, k2, k3, k4, sim, n2_prefactor_d, cache,
                                   haw_d, hbw_d, nothing)
            aw, next_aw = next_aw, aw
        end
        synchronize && CUDA.synchronize()
        out[:, :, q] = Array(fft(aw, 1))
        zprev = znext
    end

    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PulsePropagation.PropagationOutput{T}(; z=save_z, fields=out,
                                        dt=initial_condition.dt,
                                        betas=sim_betas, dz=save_dz,
                                        t_delay=delays)
end

function PulsePropagation.propagate_rk4ip_cp_cuda(fiber::PulsePropagation.Fiber{T},
                                         initial_condition::PulsePropagation.InitialCondition{T},
                                         sim::PulsePropagation.Simulation{T},
                                         srsk_cp;
                                         zsave=nothing,
                                         dz=nothing,
                                         device=nothing,
                                         synchronize::Bool=true) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.gain_model == 0 ||
        error("propagate_rk4ip_cp_cuda currently supports passive propagation only.")

    fields0 = PulsePropagation.last_fields(initial_condition)
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    step_dz = dz === nothing ? sim.dz : T(dz)
    step_dz > zero(T) || error("dz must be positive.")
    omega = PulsePropagation.ifft_frequency_grid(nt, initial_condition.dt)
    d_op, sim_betas = PulsePropagation.calc_d_op(fiber2, sim, nt, initial_condition.dt, omega, fields0)
    damp = T.(PulsePropagation.damped_freq_window(nt))
    c = T(2.99792458e-4)
    n2_prefactor = Complex{T}.(1im .* fiber2.n2 .* (omega .+ 2π * sim.f0) ./ c .* damp)
    fiber3, haw, hbw = PulsePropagation.raman_model(fiber2, sim, nt, initial_condition.dt)
    isempty(hbw) || error("propagate_rk4ip_cp_cuda currently supports scalar CP propagation with no separate hbw path.")

    save_z = if zsave === nothing
        save_period = sim.save_period == 0 ? fiber2.L0 : sim.save_period
        collect(T, 0:round(Int, fiber2.L0 / save_period)) .* save_period
    else
        T.(collect(zsave))
    end
    isempty(save_z) && error("zsave must contain at least one point.")
    first(save_z) == zero(T) || error("zsave must start at 0.")
    last(save_z) == fiber2.L0 || error("zsave must end at fiber.L0.")
    all(diff(save_z) .> zero(T)) || error("zsave must be strictly increasing.")

    half = CUDA.CuArray(exp.(d_op .* (step_dz / 2)))
    aw = CUDA.CuArray(PulsePropagation.inverse_fft(fields0, dims=1))
    next_aw = similar(aw)
    a_ip = similar(aw)
    stage = similar(aw)
    k1 = similar(aw)
    k2 = similar(aw)
    k3 = similar(aw)
    k4 = similar(aw)
    n2_prefactor_d = CUDA.CuArray(n2_prefactor)
    haw_d = CUDA.CuArray(haw)
    hbw_d = CUDA.CuArray(hbw)
    cache = PulsePropagation.cuda_cp_fast_rhs_cache(srsk_cp, nt; T=T)

    out = Array{Complex{T},3}(undef, nt, nm, length(save_z))
    out[:, :, 1] = fields0

    zprev = zero(T)
    @inbounds for q in 2:length(save_z)
        znext = save_z[q]
        nsteps = PulsePropagation._fixed_step_count(znext - zprev, step_dz; label="save interval")
        for _ in 1:nsteps
            _rk4ip_cp_cuda_step!(next_aw, aw, step_dz, half, a_ip, stage,
                                 k1, k2, k3, k4, sim, n2_prefactor_d, cache,
                                 fiber3.fr, haw_d, hbw_d, nothing)
            aw, next_aw = next_aw, aw
        end
        synchronize && CUDA.synchronize()
        out[:, :, q] = Array(fft(aw, 1))
        zprev = znext
    end

    save_dz = zeros(T, length(save_z))
    length(save_dz) > 1 && (save_dz[2:end] .= diff(save_z))
    delays = zeros(T, length(save_z))
    return PulsePropagation.PropagationOutput{T}(; z=save_z, fields=out,
                                        dt=initial_condition.dt,
                                        betas=sim_betas, dz=save_dz,
                                        t_delay=delays)
end

function PulsePropagation.solve_adjoint_cuda(lambda_terminal::AbstractMatrix{Complex{T}},
                                             forward::PulsePropagation.PropagationOutput{T},
                                             fiber::PulsePropagation.Fiber{T},
                                             sim::PulsePropagation.Simulation{T};
                                             device=nothing,
                                             synchronize::Bool=true,
                                             dz_adj::T=sim.dz,
                                             zsave_adj=nothing,
                                             return_lambdaw_zsave::Bool=false,
                                             linear_gain=nothing,
                                             raman::Symbol=:none,
                                             raman_fraction::T=T(0.18),
                                             adaptive::Bool=false,
                                             abstol::T=T(1e-9),
                                             reltol::T=T(1e-9)) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.scalar || error("solve_adjoint_cuda currently supports scalar modal propagation only.")
    sim.gain_model == 0 || error("solve_adjoint_cuda currently supports passive propagation only.")
    linear_gain === nothing || error("solve_adjoint_cuda does not yet support linear_gain.")
    forward.linear_gain === nothing || error("solve_adjoint_cuda does not yet support linear_gain.")
    raman in (:none, :off) ||
        error("solve_adjoint_cuda currently supports passive Kerr adjoints only; Raman CUDA adjoints are not implemented yet.")
    adaptive == false ||
        error("solve_adjoint_cuda currently uses fixed-step RK4 integration; adaptive=true is not implemented.")
    fields0 = forward.fields[:, :, 1]
    nt, nm = size(fields0)
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim, nm)
    sim_kerr = PulsePropagation.Simulation{T}(; lambda0=sim.lambda0, f0=sim.f0,
        dz=sim.dz, save_period=sim.save_period, midx=sim.midx,
        scalar=sim.scalar, ellipticity=sim.ellipticity,
        include_Raman=false, gain_model=sim.gain_model,
        pulse_centering=sim.pulse_centering,
        progress_bar=sim.progress_bar,
        step_method=sim.step_method, cs=sim.cs, cs_model=sim.cs_model,
        betas=sim.betas, source=sim.source)
    gamma = CUDA.CuArray(PulsePropagation._adjoint_gamma_tensor(fiber2, sim_kerr,
                                                                nm; tensor=:sk))
    rhs_builder = function(d_op_d, tau_d, u_tilde_d, zgrid)
        return (λ, z) -> begin
            u_z = _cuda_interpolate_forward(u_tilde_d, zgrid, T(z))
            _cuda_adjoint_full_rhs(λ, u_z, T(z), d_op_d, tau_d, gamma,
                                   one(T))
        end
    end
    return _cuda_integrate_adjoint(lambda_terminal, forward, fiber, sim,
        rhs_builder; dz_adj=dz_adj, zsave_adj=zsave_adj,
        return_lambdaw_zsave=return_lambdaw_zsave,
        synchronize=synchronize)
end

function PulsePropagation.solve_adjoint_compressed_rankchannels_cuda(
    lambda_terminal::AbstractMatrix{Complex{T}},
    forward::PulsePropagation.PropagationOutput{T},
    fiber::PulsePropagation.Fiber{T},
    sim::PulsePropagation.Simulation{T},
    compressed;
    device=nothing,
    synchronize::Bool=true,
    dz_adj::T=sim.dz,
    zsave_adj=nothing,
    return_lambdaw_zsave::Bool=false,
    raman::Symbol=:none,
    raman_fraction::T=T(0.18),
    adaptive::Bool=false,
    abstol::T=T(1e-9),
    reltol::T=T(1e-9)) where {T}
    CUDA.functional() || error("CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    sim.scalar || error("solve_adjoint_compressed_rankchannels_cuda currently supports scalar modal propagation only.")
    sim.gain_model == 0 ||
        error("solve_adjoint_compressed_rankchannels_cuda currently supports passive propagation only.")
    forward.linear_gain === nothing ||
        error("solve_adjoint_compressed_rankchannels_cuda does not support linear_gain.")
    raman in (:none, :off, :agarwal) ||
        error("raman must be :none, :off, or :agarwal for the CUDA rank-channel adjoint.")
    adaptive == false ||
        error("solve_adjoint_compressed_rankchannels_cuda currently uses fixed-step RK4 integration; adaptive=true is not implemented.")
    fields0 = forward.fields[:, :, 1]
    if sim.include_Raman
        forward.ode_sol === nothing || error(
            "CUDA adjoints of Raman-active forward trajectories do not consume a CPU dense interpolant; use the CPU adjoint for trajectories with ode_sol.")
        return_lambdaw_zsave && error(
            "CUDA adjoints of Raman-active forward trajectories currently require return_lambdaw_zsave=false so RK4 stages remain aligned with saved forward planes.")
        gaps = diff(T.(forward.z))
        isempty(gaps) && error(
            "CUDA adjoints of Raman-active forward trajectories require at least two saved forward planes.")
        forward_dz = gaps[1]
        all(g -> isapprox(g, forward_dz; rtol=T(1e-10), atol=T(1e-12)), gaps) ||
            error("CUDA adjoints of Raman-active forward trajectories require a uniform saved-forward grid.")
        isapprox(forward_dz, sim.dz; rtol=T(1e-10), atol=T(1e-12)) || error(
            "CUDA adjoints of Raman-active forward trajectories require every forward RK4IP step to be saved.")
        iseven(length(gaps)) || error(
            "CUDA stage alignment for Raman-active forward trajectories requires an even number of forward intervals.")
        isapprox(dz_adj, 2forward_dz; rtol=T(1e-10), atol=T(1e-12)) || error(
            "CUDA adjoints of Raman-active forward trajectories require dz_adj=2*forward_dz so every RK4 stage lies on a saved forward plane.")
    end
    if raman === :agarwal
        sim.include_Raman || error(
            "CUDA Agarwal-Raman adjoints require Raman in the forward model.")
        fiber.material == "agarwal" || error(
            "CUDA Agarwal-Raman adjoints require an Agarwal forward material.")
        isfinite(raman_fraction) && zero(T) < raman_fraction < one(T) || error(
            "CUDA Agarwal-Raman adjoints require a finite Raman fraction strictly between zero and one.")
        isapprox(fiber.fr, raman_fraction; rtol=zero(T), atol=eps(T)) || error(
            "Forward and adjoint Raman fractions must match.")
    end
    fiber2 = PulsePropagation.expand_betas_for_polarization(fiber, sim,
                                                            size(fields0, 2))
    c = T(2.99792458e-4)
    omega0 = T(2π) * sim.f0
    scale = fiber2.n2 * omega0 / c
    compressed_cp = _validated_cuda_adjoint_cp(compressed, T)
    cs = PulsePropagation._compressed_bundle(compressed_cp, zero(T))
    gamma_cp = _cuda_cp_adjoint_factors(PulsePropagation._scaled_cp(cs.sk, scale), T)
    one_m_fR = raman === :agarwal ? one(T) - raman_fraction : one(T)
    hR_sigma_d = if raman === :agarwal
        iseven(size(fields0, 1)) ||
            error("CUDA Agarwal Raman adjoints require an even time-grid size.")
        hR = PulsePropagation.agarwal_raman_response(size(fields0, 1),
                                                     forward.dt,
                                                     raman_fraction)
        sigma = Complex{T}.(exp.(1im * T(π) .*
                     repeat(T[zero(T), one(T)], div(size(fields0, 1), 2))))
        CUDA.CuArray(hR .* sigma)
    else
        CUDA.zeros(Complex{T}, 0)
    end
    rhs_builder = function(d_op_d, tau_d, u_tilde_d, zgrid)
        if raman === :agarwal
            cache = _cuda_cp_adjoint_raman_cache(size(fields0, 1),
                                                 size(fields0, 2),
                                                 length(gamma_cp.λ), T)
            return (lambda_state, z) -> begin
                u_z = _cuda_interpolate_forward(u_tilde_d, zgrid, T(z))
                _cuda_adjoint_cp_raman_rhs(lambda_state, u_z, T(z), d_op_d,
                                            tau_d, gamma_cp, one_m_fR,
                                            hR_sigma_d, cache)
            end
        end
        return (lambda_state, z) -> begin
            u_z = _cuda_interpolate_forward(u_tilde_d, zgrid, T(z))
            _cuda_adjoint_cp_rhs(lambda_state, u_z, T(z), d_op_d, tau_d,
                                 gamma_cp, one(T))
        end
    end
    return _cuda_integrate_adjoint(lambda_terminal, forward, fiber, sim,
        rhs_builder; dz_adj=dz_adj, zsave_adj=zsave_adj,
        return_lambdaw_zsave=return_lambdaw_zsave,
        synchronize=synchronize)
end

include("mmgnlse_cuda.jl")
include("full_field_cuda.jl")

end
