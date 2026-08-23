"""
Metadata attached to a CP decomposition of an MMGNLSE overlap tensor.

`layout == :spatial` means the four tensor axes are spatial-mode axes and
polarization is contracted analytically. `layout == :component` means each
axis is an already-flattened `(mode, polarization)` component axis.

`source_checksum` is deliberately opaque. Cache code can supply a SHA-256
function through `cp_compress(...; checksum_fn=...)` without coupling this
numerical core to a particular serialization or hashing package.
"""
struct CPSourceMetadata
    layout::Symbol
    source_size::NTuple{4,Int}
    nmodes::Int
    npolarizations::Int
    source_checksum::Union{Nothing,String}
end

"""
    MMGNLSECPDecomposition

A deterministic CP representation

```math
S_{ijkl} \\approx \\sum_r \\lambda_r
U^{(1)}_{ir}U^{(2)}_{jr}U^{(3)}_{kr}U^{(4)}_{lr}.
```

No factor is conjugated in the reconstruction. The weights produced by the
ALS implementation are real and nonnegative, while the factors may be
complex. A zero source tensor is represented by rank-zero factors and is
identified by `cp_iszero(cp)` so callers can route directly to a linear
solver.
"""
struct MMGNLSECPDecomposition{TW<:Number,TF<:Number}
    λ::Vector{TW}
    U::NTuple{4,Matrix{TF}}
    metadata::CPSourceMetadata
    relative_error::Float64
    seed::Int
    iterations::Int
    converged::Bool
    zero_tensor::Bool
end

struct CPCompressionFailure{C} <: Exception
    target_error::Float64
    max_rank::Int
    best::C
end

"""
    CPMemoryBudget(; rank_block_size=:auto,
                     max_workspace_bytes=nothing,
                     workspace_memory_fraction=0.5,
                     reclaim_memory=true)

Opt-in memory controls for [`cp_compress_bounded`](@ref). This type and the
bounded compression entry point are deliberately separate from `cp_compress`
so existing callers do not need to know about the experimental interface.
"""
struct CPMemoryBudget
    rank_block_size::Union{Symbol,Int}
    max_workspace_bytes::Union{Nothing,Int}
    workspace_memory_fraction::Float64
    reclaim_memory::Bool
    function CPMemoryBudget(
        rank_block_size::Union{Symbol,Integer},
        max_workspace_bytes::Union{Nothing,Integer},
        workspace_memory_fraction::Real,
        reclaim_memory::Bool,
    )
        rank_block_size === :auto ||
            (rank_block_size isa Integer && rank_block_size > 0) ||
            throw(ArgumentError(
                "rank_block_size must be :auto or a positive integer."))
        max_workspace_bytes === nothing || max_workspace_bytes > 0 ||
            throw(ArgumentError(
                "max_workspace_bytes must be positive or nothing."))
        isfinite(workspace_memory_fraction) &&
            0 < workspace_memory_fraction <= 1 || throw(ArgumentError(
                "workspace_memory_fraction must be finite and in (0, 1]."))
        resolved_block = rank_block_size === :auto ? :auto :
                         Int(rank_block_size)
        resolved_maximum = max_workspace_bytes === nothing ? nothing :
                           Int(max_workspace_bytes)
        return new(
            resolved_block,
            resolved_maximum,
            Float64(workspace_memory_fraction),
            reclaim_memory,
        )
    end
end

CPMemoryBudget(;
    rank_block_size::Union{Symbol,Integer}=:auto,
    max_workspace_bytes::Union{Nothing,Integer}=nothing,
    workspace_memory_fraction::Real=0.5,
    reclaim_memory::Bool=true,
) = CPMemoryBudget(
    rank_block_size,
    max_workspace_bytes,
    workspace_memory_fraction,
    reclaim_memory,
)

"""Memory plan returned by [`cp_compression_plan`](@ref)."""
struct CPCompressionPlan
    nmodes::Int
    requested_rank::Int
    backend::Symbol
    fit_precision::Symbol
    source_structure::Symbol
    source_bytes::Int
    unfolding_bytes::Int
    bytes_per_rank_column::Int
    resolved_rank_block_size::Int
    khatri_rao_workspace_bytes::Int
    factor_bytes::Int
    gram_bytes::Int
    estimated_peak_bytes::Int
    free_device_bytes::Union{Nothing,Int}
    estimated_headroom_bytes::Union{Nothing,Int}
    cuda_addressable::Bool
end

function Base.showerror(io::IO, err::CPCompressionFailure)
    print(io, "CP compression did not reach relative error ", err.target_error,
          " by rank ", err.max_rank, "; best error was ",
          err.best.relative_error, ".")
end

cp_rank(cp::MMGNLSECPDecomposition) = length(cp.λ)
cp_iszero(cp::MMGNLSECPDecomposition) = cp.zero_tensor

function _cp_source_metadata(source_size::NTuple{4,Int};
                             layout::Symbol=:spatial,
                             nmodes::Union{Nothing,Integer}=nothing,
                             npolarizations::Integer=1,
                             source_checksum=nothing)
    layout in (:spatial, :component) ||
        throw(ArgumentError("layout must be :spatial or :component."))
    all(>(0), source_size) || throw(ArgumentError("All source dimensions must be positive."))
    length(unique(source_size)) == 1 ||
        throw(ArgumentError("MMGNLSE overlap tensors must have four equal dimensions."))
    npol = Int(npolarizations)
    npol in (1, 2) || throw(ArgumentError("npolarizations must be one or two."))
    axis_size = source_size[1]
    inferred_nmodes = if layout == :spatial
        axis_size
    else
        axis_size % npol == 0 ||
            throw(ArgumentError("Component axis size $axis_size is not divisible by $npol polarizations."))
        axis_size ÷ npol
    end
    nm = nmodes === nothing ? inferred_nmodes : Int(nmodes)
    nm == inferred_nmodes ||
        throw(ArgumentError("nmodes=$nm is inconsistent with the tensor dimensions and layout."))
    checksum = source_checksum === nothing ? nothing : String(source_checksum)
    return CPSourceMetadata(layout, source_size, nm, npol, checksum)
end

"""
    MMGNLSECPDecomposition(λ, U; layout=:spatial, ...)

Construct a decomposition from user-supplied factors. This is useful for
loading an already verified cache and for exact low-rank test fixtures.
"""
function MMGNLSECPDecomposition(λ::AbstractVector,
                                U::NTuple{4,<:AbstractMatrix};
                                layout::Symbol=:spatial,
                                nmodes::Union{Nothing,Integer}=nothing,
                                npolarizations::Integer=1,
                                source_checksum=nothing,
                                relative_error::Real=NaN,
                                seed::Integer=0,
                                iterations::Integer=0,
                                converged::Bool=true,
                                zero_tensor::Bool=isempty(λ))
    rank = length(λ)
    all(size(F, 2) == rank for F in U) ||
        throw(DimensionMismatch("Every factor must have length(λ) columns."))
    source_size = ntuple(n -> size(U[n], 1), 4)
    metadata = _cp_source_metadata(source_size; layout, nmodes, npolarizations,
                                   source_checksum)
    TW = eltype(λ)
    TF = promote_type(map(eltype, U)...)
    factors = ntuple(n -> Matrix{TF}(U[n]), 4)
    return MMGNLSECPDecomposition{TW,TF}(collect(TW, λ), factors, metadata,
                                         Float64(relative_error), Int(seed),
                                         Int(iterations), converged, zero_tensor)
end

function _cp_reconstruct(λ::AbstractVector, U::NTuple{4,<:AbstractMatrix},
                         dims::NTuple{4,Int})
    T = promote_type(eltype(λ), map(eltype, U)...)
    out = zeros(T, dims)
    U1, U2, U3, U4 = U
    @inbounds for r in eachindex(λ)
        wr = λ[r]
        for i4 in 1:dims[4], i3 in 1:dims[3], i2 in 1:dims[2], i1 in 1:dims[1]
            out[i1, i2, i3, i4] +=
                wr * U1[i1, r] * U2[i2, r] * U3[i3, r] * U4[i4, r]
        end
    end
    return out
end

"""Reconstruct the dense tensor represented by `cp`."""
cp_reconstruct_mmgnlse(cp::MMGNLSECPDecomposition) =
    _cp_reconstruct(cp.λ, cp.U, cp.metadata.source_size)

function _cp_require_compatible(left::MMGNLSECPDecomposition,
                                right::MMGNLSECPDecomposition)
    left.metadata.source_size == right.metadata.source_size ||
        throw(DimensionMismatch(
            "CP tensors have source sizes $(left.metadata.source_size) and " *
            "$(right.metadata.source_size)."))
    return nothing
end

"""
    cp_inner_product(left, right; block_size=256)

Compute the Frobenius inner product between two CP tensors without forming
either dense tensor. `block_size` bounds the temporary cross-Gram matrix by
blocking columns of `right`.
"""
function cp_inner_product(left::MMGNLSECPDecomposition,
                          right::MMGNLSECPDecomposition;
                          block_size::Integer=256)
    _cp_require_compatible(left, right)
    block_size > 0 || throw(ArgumentError("block_size must be positive."))
    T = promote_type(eltype(left.λ), eltype(right.λ),
                     map(eltype, left.U)..., map(eltype, right.U)...)
    (isempty(left.λ) || isempty(right.λ)) && return zero(T)
    result = zero(T)
    tied = cp_is_symmetric(left) && cp_is_symmetric(right)
    for first_column in 1:Int(block_size):cp_rank(right)
        last_column = min(first_column + Int(block_size) - 1, cp_rank(right))
        columns = first_column:last_column
        cross = ones(T, cp_rank(left), length(columns))
        if tied
            cross .= adjoint(left.U[1]) * view(right.U[1], :, columns)
            cross .^= 4
        else
            @inbounds for mode in 1:4
                cross .*= adjoint(left.U[mode]) *
                          view(right.U[mode], :, columns)
            end
        end
        result += dot(left.λ, cross * view(right.λ, columns))
    end
    return result
end

"""Return the Frobenius norm of a CP tensor without dense reconstruction."""
function cp_norm(cp::MMGNLSECPDecomposition; block_size::Integer=256)
    norm_squared = real(cp_inner_product(cp, cp; block_size))
    RT = typeof(float(norm_squared))
    scale = max(abs(norm_squared), one(RT))
    norm_squared >= -sqrt(eps(RT)) * scale ||
        throw(DomainError(norm_squared, "CP norm squared is negative."))
    return sqrt(max(norm_squared, zero(norm_squared)))
end

"""
    cp_relative_error(candidate, reference; block_size=256)

Return `norm(candidate-reference) / norm(reference)` using CP Gram matrices
only. This is the authority residual used by factorized CP compression.
"""
function cp_relative_error(candidate::MMGNLSECPDecomposition,
                           reference::MMGNLSECPDecomposition;
                           block_size::Integer=256,
                           backend::Symbol=:cpu,
                           device=nothing,
                           synchronize::Bool=true)
    _cp_require_compatible(candidate, reference)
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    backend === :cuda && return _cp_relative_error_factorized_cuda(
        candidate, reference; block_size, device, synchronize)
    reference_norm_squared =
        real(cp_inner_product(reference, reference; block_size))
    reference_norm_squared > 0 || return cp_iszero(candidate) ? 0.0 : Inf
    candidate_norm_squared =
        real(cp_inner_product(candidate, candidate; block_size))
    cross = cp_inner_product(reference, candidate; block_size)
    residual_squared = reference_norm_squared + candidate_norm_squared -
                       2real(cross)
    return sqrt(max(Float64(residual_squared / reference_norm_squared), 0.0))
end

function _mode_unfold_mmgnlse(X::AbstractArray{<:Any,4}, n::Integer)
    order = (n, (k for k in 1:4 if k != n)...)
    return reshape(permutedims(X, order), size(X, n), :)
end

# Row ordering matches `_mode_unfold_mmgnlse`: the first factor's row index
# varies fastest in the flattened column index.
function _khatri_rao_rows_mmgnlse(factors)
    kr = Matrix(factors[1])
    rank = size(kr, 2)
    for F in factors[2:end]
        size(F, 2) == rank || throw(DimensionMismatch("Khatri-Rao rank mismatch."))
        old_rows = size(kr, 1)
        new_rows = size(F, 1)
        out = similar(kr, old_rows * new_rows, rank)
        @inbounds for j in 1:new_rows, i in 1:old_rows, r in 1:rank
            out[(j - 1) * old_rows + i, r] = kr[i, r] * F[j, r]
        end
        kr = out
    end
    return kr
end

function _normalize_cp_columns!(A::AbstractMatrix, floor_norm::Real)
    RT = typeof(real(float(zero(eltype(A)))))
    weights = zeros(RT, size(A, 2))
    @inbounds for r in axes(A, 2)
        nr = norm(view(A, :, r))
        if nr <= floor_norm
            fill!(view(A, :, r), zero(eltype(A)))
            weights[r] = zero(RT)
        else
            A[:, r] ./= nr
            weights[r] = nr
        end
    end
    return weights
end

function _complex_random_factors(rng, ::Type{CT}, dims::NTuple{4,Int}, rank::Int;
                                 initial=nothing) where {CT<:Complex}
    RT = typeof(real(zero(CT)))
    scale = inv(sqrt(RT(2)))
    U = ntuple(n -> scale .* (randn(rng, RT, dims[n], rank) .+
                              im .* randn(rng, RT, dims[n], rank)), 4)
    if initial !== nothing
        r0 = min(cp_rank(initial), rank)
        @inbounds for n in 1:4
            size(initial.U[n], 1) == dims[n] ||
                throw(DimensionMismatch("Warm-start factor dimensions do not match the source."))
            U[n][:, 1:r0] .= initial.U[n][:, 1:r0]
        end
    end
    floor_norm = sqrt(eps(RT))
    @inbounds for n in 1:4
        _normalize_cp_columns!(U[n], floor_norm)
    end
    return U
end

function _cp_relative_error(X, λ, U)
    Xnorm = norm(X)
    Xnorm == 0 && return 0.0
    return Float64(norm(_cp_reconstruct(λ, U, size(X)) .- X) / Xnorm)
end

"""
    cp_als_complex(X, rank; seed=..., initial=nothing, ...)

Fit a four-way tensor with complex CP-ALS. The least-squares update is for a
nonconjugated CP model, so it uses `conj(K)` and `transpose(K)*conj(K)` rather
than the real-only normal equations used by the legacy compressor.

The return value is `(weights, factors, history, iterations, converged)`.
"""
function cp_als_complex(X::AbstractArray{<:Number,4}, rank::Integer;
                        seed::Integer=0x4d4d474e,
                        initial::Union{Nothing,MMGNLSECPDecomposition}=nothing,
                        maxiter::Integer=300,
                        tolerance::Real=1e-10,
                        ridge::Real=1e-12,
                        check_every::Integer=5,
                        target_error::Real=0.0,
                        rank_block_size::Integer=rank,
                        symmetric_source::Bool=false)
    r = Int(rank)
    r > 0 || throw(ArgumentError("rank must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    check_every > 0 || throw(ArgumentError("check_every must be positive."))
    RT = typeof(float(real(zero(eltype(X)))))
    CT = Complex{RT}
    Xc = X isa AbstractArray{CT,4} ? X : CT.(X)
    Xnorm = norm(Xc)
    Xnorm > zero(RT) || throw(ArgumentError("Use cp_compress for a zero tensor."))
    rng = MersenneTwister(Int(seed))
    U = _complex_random_factors(rng, CT, size(Xc), r; initial)
    λ = ones(RT, r)
    history = Float64[]
    previous_error = Inf
    converged = false
    iterations = 0
    floor_norm = sqrt(eps(RT))
    unfolding_workspace = symmetric_source ? nothing :
                          similar(Xc, length(Xc))

    for iteration in 1:Int(maxiter)
        iterations = iteration
        for n in 1:4
            rest = [k for k in 1:4 if k != n]
            gram = ones(CT, r, r)
            for k in rest
                gram .*= transpose(U[k]) * conj.(U[k])
            end
            ridge_scale = max(maximum(abs, gram), one(RT))
            @inbounds for q in 1:r
                gram[q, q] += RT(ridge) * ridge_scale
            end
            unfolded = symmetric_source || n == 1 ?
                         reshape(Xc, size(Xc, 1), :) :
                         _mode_unfold!(unfolding_workspace, Xc, n)
            mttkrp = _cp_mttkrp_blocked(
                unfolded, U, rest, rank_block_size;
                conjugate_factors=true)
            weighted_factor = mttkrp / gram
            λ = _normalize_cp_columns!(weighted_factor, floor_norm)
            U = Base.setindex(U, weighted_factor, n)
        end

        if iteration == 1 || iteration % check_every == 0 || iteration == maxiter
            err = _cp_relative_error_from_mttkrp(
                Xc, λ, U, rank_block_size; symmetric_source)
            push!(history, err)
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
    return λ, U, history, iterations, converged
end

function _resolved_checksum(S; source_checksum=nothing, checksum_fn=nothing)
    source_checksum !== nothing && checksum_fn !== nothing &&
        throw(ArgumentError("Specify source_checksum or checksum_fn, not both."))
    value = checksum_fn === nothing ? source_checksum : checksum_fn(S)
    return value === nothing ? nothing : String(value)
end

"""
    cp_compress(S; target_error, max_rank, backend=:cpu, ...)

Deterministically search CP ranks from `min_rank` through `max_rank` until the
relative Frobenius error is at most `target_error`. The fixed default seed is
part of the returned provenance. Rank searches are warm-started; optional
additional deterministic fresh starts can be requested with `restarts`.

`backend=:cpu` uses the CPU implementation. `backend=:cuda` requires CUDA.jl
and a functional CUDA device, performs the CP-ALS rank search on that device,
and returns the same host-resident `MMGNLSECPDecomposition`.

`rank_block_size` bounds the number of CP columns in the temporary
Khatri–Rao product. `max_workspace_bytes` imposes a byte ceiling on that
temporary and can reduce the block further. On CUDA,
`workspace_memory_fraction` also limits it to a fraction of currently
available device memory, while `reclaim_memory=true` releases unused CUDA
pool storage before sizing the block. For a fully permutation-symmetric
source, `symmetric_source=true` reuses the mode-1 unfolding for all four ALS
updates and avoids the full unfolding allocation; symmetry is checked before
the fit starts unless `check_symmetry=false` is used for a trusted source.

For example:

```julia
S_compressed = cp_compress(S_dense; error=1e-4, backend=:cuda)
```

A zero tensor returns a rank-zero decomposition with `cp_iszero(cp) == true`.
If the target is missed, `CPCompressionFailure` is thrown and holds
the best decomposition in its `best` field.
"""
function _cp_compress_impl(S::AbstractArray{<:Number,4};
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
                     rank_block_size=nothing,
                     max_workspace_bytes=nothing,
                     symmetric_source::Bool=false)
    resolved_error = error === nothing ? target_error : float(error)
    resolved_error >= 0 || throw(ArgumentError("error must be nonnegative."))
    max_rank > 0 || throw(ArgumentError("max_rank must be positive."))
    min_rank > 0 || throw(ArgumentError("min_rank must be positive."))
    min_rank <= max_rank || throw(ArgumentError("min_rank cannot exceed max_rank."))
    rank_step > 0 || throw(ArgumentError("rank_step must be positive."))
    restarts > 0 || throw(ArgumentError("restarts must be positive."))
    all(isfinite, S) || throw(ArgumentError("The overlap tensor contains non-finite values."))
    checksum = _resolved_checksum(S; source_checksum, checksum_fn)
    metadata = _cp_source_metadata(size(S); layout, nmodes, npolarizations,
                                   source_checksum=checksum)
    RT = typeof(float(real(zero(eltype(S)))))
    CT = Complex{RT}
    source_norm = norm(S)
    if source_norm == 0
        factors = ntuple(n -> zeros(CT, size(S, n), 0), 4)
        return MMGNLSECPDecomposition(Float64[], factors;
            layout=metadata.layout, nmodes=metadata.nmodes,
            npolarizations=metadata.npolarizations, source_checksum=checksum,
            relative_error=0.0, seed, iterations=0, converged=true,
            zero_tensor=true)
    end
    source_complex = S isa AbstractArray{CT,4} ? S : CT.(S)

    best = nothing
    warm = nothing
    for rank in Int(min_rank):Int(rank_step):Int(max_rank)
        resolved_rank_block_size = _cp_workspace_limited_rank_block(
            size(S, 1), rank, CT;
            rank_block_size,
            max_workspace_bytes)
        rank_best = nothing
        for restart in 1:Int(restarts)
            init = restart == 1 ? warm : nothing
            local_seed = Int(seed) + 104729 * rank + 1009 * (restart - 1)
            λ, U, history, iterations, als_converged = cp_als_complex(
                source_complex, rank;
                seed=local_seed, initial=init, maxiter, tolerance, ridge,
                check_every, target_error=resolved_error,
                rank_block_size=resolved_rank_block_size,
                symmetric_source)
            err = history[end]
            candidate = MMGNLSECPDecomposition(λ, U;
                layout=metadata.layout, nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum=checksum, relative_error=err, seed,
                iterations, converged=err <= resolved_error,
                zero_tensor=false)
            if rank_best === nothing || candidate.relative_error < rank_best.relative_error
                rank_best = candidate
            end
        end
        warm = rank_best
        if best === nothing || rank_best.relative_error < best.relative_error
            best = rank_best
        end
        rank_best.relative_error <= resolved_error && return rank_best
    end

    throw(CPCompressionFailure(Float64(resolved_error), Int(max_rank), best))
end

function cp_compress(S::AbstractArray{<:Number,4};
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
                     backend::Symbol=:cpu,
                     device=nothing,
                     synchronize::Bool=true,
                     rank_block_size::Union{Nothing,Integer}=nothing,
                     max_workspace_bytes::Union{Nothing,Integer}=nothing,
                     workspace_memory_fraction::Real=0.5,
                     reclaim_memory::Bool=true,
                     symmetric_source::Bool=false,
                     check_symmetry::Bool=true)
    fftw_threads > 0 || throw(ArgumentError("fftw_threads must be positive."))
    blas_threads > 0 || throw(ArgumentError("blas_threads must be positive."))
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    _cp_validate_memory_controls(
        rank_block_size, max_workspace_bytes, workspace_memory_fraction)
    if backend === :cuda
        return _cp_compress_cuda(S; target_error, error, max_rank, min_rank,
            rank_step, seed, maxiter, tolerance, ridge, check_every, restarts,
            layout, nmodes, npolarizations, source_checksum, checksum_fn,
            fftw_threads, blas_threads, device, synchronize,
            rank_block_size, max_workspace_bytes, workspace_memory_fraction,
            reclaim_memory, symmetric_source, check_symmetry)
    end
    (workspace_memory_fraction == 0.5 && reclaim_memory) ||
        throw(ArgumentError(
            "workspace_memory_fraction and reclaim_memory apply only to " *
            "backend=:cuda; retain their defaults for backend=:cpu."))
    if symmetric_source && check_symmetry
        symmetry_defect = _cp_symmetric_tensor_defect(S)
        RT = typeof(float(real(zero(eltype(S)))))
        symmetry_defect <= max(RT(1e-12), RT(100) * eps(RT)) ||
            throw(ArgumentError(
                "symmetric_source=true requires a fully permutation-symmetric " *
                "tensor; measured relative defect $symmetry_defect."))
    end
    previous_fftw_threads = FFTW.get_num_threads()
    previous_blas_threads = BLAS.get_num_threads()
    try
        FFTW.set_num_threads(Int(fftw_threads))
        BLAS.set_num_threads(Int(blas_threads))
        return _cp_compress_impl(S; target_error, error, max_rank, min_rank,
            rank_step, seed, maxiter, tolerance, ridge, check_every, restarts,
            layout, nmodes, npolarizations, source_checksum, checksum_fn,
            rank_block_size, max_workspace_bytes, symmetric_source)
    finally
        FFTW.set_num_threads(previous_fftw_threads)
        BLAS.set_num_threads(previous_blas_threads)
    end
end

function _cp_real_factorized_source(source::MMGNLSECPDecomposition)
    all(isreal, source.λ) && all(factor -> all(isreal, factor), source.U) ||
        throw(ArgumentError(
            "Factorized CP compression currently requires a real source."))
    RT = promote_type(typeof(float(real(zero(eltype(source.λ))))),
                      map(factor -> typeof(float(real(zero(eltype(factor))))),
                          source.U)...)
    weights = RT.(real.(source.λ))
    factors = ntuple(mode -> RT.(real.(source.U[mode])), 4)
    all(isfinite, weights) && all(factor -> all(isfinite, factor), factors) ||
        throw(ArgumentError("The factorized CP source contains non-finite values."))
    return weights, factors
end

function _cp_real_random_factors(rng, ::Type{RT}, dims::NTuple{4,Int},
                                 rank::Int; initial=nothing) where {RT<:AbstractFloat}
    factors = ntuple(mode -> randn(rng, RT, dims[mode], rank), 4)
    if initial !== nothing
        all(isreal, initial.λ) &&
            all(factor -> all(isreal, factor), initial.U) ||
            throw(ArgumentError("A real factorized fit requires a real initial CP."))
        copied = min(cp_rank(initial), rank)
        @inbounds for mode in 1:4
            size(initial.U[mode], 1) == dims[mode] ||
                throw(DimensionMismatch(
                    "Initial CP factor dimensions do not match the source."))
            factors[mode][:, 1:copied] .=
                RT.(real.(view(initial.U[mode], :, 1:copied)))
        end
    end
    floor_norm = sqrt(eps(RT))
    @inbounds for mode in 1:4
        _normalize_cp_columns!(factors[mode], floor_norm)
    end
    return factors
end

function _cp_factorized_mttkrp(source_weights, source_factors,
                               target_factors, mode::Int)
    source_rank = length(source_weights)
    target_rank = size(target_factors[1], 2)
    T = promote_type(eltype(source_weights),
                     map(eltype, source_factors)...,
                     map(eltype, target_factors)...)
    cross = ones(T, source_rank, target_rank)
    @inbounds for other_mode in 1:4
        other_mode == mode && continue
        cross .*= transpose(source_factors[other_mode]) *
                  target_factors[other_mode]
    end
    cross .*= reshape(source_weights, :, 1)
    return source_factors[mode] * cross
end

function _cp_factorized_inner_product(left_weights, left_factors,
                                      right_weights, right_factors;
                                      block_size::Integer=256)
    block_size > 0 || throw(ArgumentError("block_size must be positive."))
    T = promote_type(eltype(left_weights), eltype(right_weights),
                     map(eltype, left_factors)...,
                     map(eltype, right_factors)...)
    (isempty(left_weights) || isempty(right_weights)) && return zero(T)
    result = zero(T)
    right_rank = length(right_weights)
    tied = all(factor -> factor == left_factors[1], left_factors[2:4]) &&
           all(factor -> factor == right_factors[1], right_factors[2:4])
    for first_column in 1:Int(block_size):right_rank
        last_column = min(first_column + Int(block_size) - 1, right_rank)
        columns = first_column:last_column
        cross = ones(T, length(left_weights), length(columns))
        if tied
            cross .= transpose(left_factors[1]) *
                     view(right_factors[1], :, columns)
            cross .^= 4
        else
            @inbounds for mode in 1:4
                cross .*= transpose(left_factors[mode]) *
                          view(right_factors[mode], :, columns)
            end
        end
        result += dot(left_weights, cross * view(right_weights, columns))
    end
    return result
end

function _cp_factorized_relative_error(source_weights, source_factors,
                                       source_norm_squared,
                                       target_weights, target_factors;
                                       block_size::Integer=256)
    target_norm_squared = real(_cp_factorized_inner_product(
        target_weights, target_factors, target_weights, target_factors;
        block_size))
    cross = _cp_factorized_inner_product(
        source_weights, source_factors, target_weights, target_factors;
        block_size)
    residual_squared = source_norm_squared + target_norm_squared - 2real(cross)
    return sqrt(max(Float64(residual_squared / source_norm_squared), 0.0))
end

function _cp_als_factorized_real(source_weights, source_factors,
                                 rank::Integer;
                                 seed::Integer,
                                 initial=nothing,
                                 maxiter::Integer=300,
                                 tolerance::Real=1e-10,
                                 ridge::Real=1e-12,
                                 check_every::Integer=5,
                                 target_error::Real=0.0,
                                 source_norm_squared=nothing,
                                 error_block_size::Integer=256,
                                 verbose::Bool=false)
    r = Int(rank)
    r > 0 || throw(ArgumentError("rank must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    check_every > 0 || throw(ArgumentError("check_every must be positive."))
    RT = eltype(source_weights)
    dims = ntuple(mode -> size(source_factors[mode], 1), 4)
    rng = MersenneTwister(Int(seed))
    factors = _cp_real_random_factors(rng, RT, dims, r; initial)
    weights = ones(RT, r)
    resolved_source_norm_squared = source_norm_squared === nothing ?
        real(_cp_factorized_inner_product(
            source_weights, source_factors, source_weights, source_factors;
            block_size=error_block_size)) : RT(source_norm_squared)
    resolved_source_norm_squared > zero(RT) || throw(ArgumentError(
        "Use factorized CP compression only for a nonzero source."))
    identity_matrix = Matrix{RT}(I, r, r)
    floor_norm = sqrt(eps(RT))
    history = Float64[]
    previous_error = Inf
    converged = false
    iterations = 0

    for iteration in 1:Int(maxiter)
        iterations = iteration
        for mode in 1:4
            gram = ones(RT, r, r)
            @inbounds for other_mode in 1:4
                other_mode == mode && continue
                gram .*= transpose(factors[other_mode]) *
                         factors[other_mode]
            end
            ridge_scale = max(maximum(abs, gram), one(RT))
            gram .+= RT(ridge) * ridge_scale .* identity_matrix
            updated = _cp_factorized_mttkrp(
                source_weights, source_factors, factors, mode) / gram
            weights = _normalize_cp_columns!(updated, floor_norm)
            factors = Base.setindex(factors, updated, mode)
        end

        if iteration == 1 || iteration % Int(check_every) == 0 ||
           iteration == Int(maxiter)
            err = _cp_factorized_relative_error(
                source_weights, source_factors, resolved_source_norm_squared,
                weights, factors; block_size=error_block_size)
            push!(history, err)
            verbose && @info "factorized CP-ALS" iteration rank=r relative_error=err
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
    return weights, factors, history, iterations, converged
end

function _cp_factorized_rank_sequence(min_rank::Int, rank_step::Int,
                                      max_rank::Int)
    ranks = collect(min_rank:rank_step:max_rank)
    isempty(ranks) && push!(ranks, max_rank)
    last(ranks) == max_rank || push!(ranks, max_rank)
    return ranks
end

"""
    cp_compress(source::MMGNLSECPDecomposition; target_error, backend=:cpu, ...)

Compress an already factorized, real four-way CP source without constructing
its dense tensor. ALS MTTKRP operations and the reported Frobenius residual
are evaluated directly from source/target factor Gram matrices. The CUDA path
keeps both the source and the fit resident on the selected device.

This path is intended for exact quadrature representations whose source rank
can be much larger than the desired solver rank. `source_checksum` should
identify that quadrature authority when the result will be cached.
"""
function cp_compress(source::MMGNLSECPDecomposition;
                     target_error::Real=1e-6,
                     error=nothing,
                     max_rank::Integer=cp_rank(source),
                     min_rank::Integer=1,
                     rank_step::Integer=1,
                     seed::Integer=0x4d4d474e,
                     initial::Union{Nothing,MMGNLSECPDecomposition}=nothing,
                     maxiter::Integer=300,
                     tolerance::Real=1e-10,
                     ridge::Real=1e-12,
                     check_every::Integer=5,
                     restarts::Integer=1,
                     source_checksum=nothing,
                     checksum_fn=nothing,
                     backend::Symbol=:cpu,
                     device=nothing,
                     synchronize::Bool=true,
                     error_block_size::Integer=256,
                     verbose::Bool=false)
    resolved_error = error === nothing ? float(target_error) : float(error)
    0 <= resolved_error < 1 || throw(ArgumentError(
        "error must lie in [0, 1)."))
    max_rank > 0 || throw(ArgumentError("max_rank must be positive."))
    min_rank > 0 || throw(ArgumentError("min_rank must be positive."))
    min_rank <= max_rank || throw(ArgumentError(
        "min_rank cannot exceed max_rank."))
    rank_step > 0 || throw(ArgumentError("rank_step must be positive."))
    restarts > 0 || throw(ArgumentError("restarts must be positive."))
    error_block_size > 0 || throw(ArgumentError(
        "error_block_size must be positive."))
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    initial === nothing || _cp_require_compatible(initial, source)
    checksum = _resolved_checksum(source; source_checksum, checksum_fn)
    metadata = source.metadata

    if cp_iszero(source)
        RT = typeof(float(real(zero(eltype(source.λ)))))
        factors = ntuple(mode -> zeros(RT, metadata.source_size[mode], 0), 4)
        return MMGNLSECPDecomposition(Float64[], factors;
            layout=metadata.layout, nmodes=metadata.nmodes,
            npolarizations=metadata.npolarizations,
            source_checksum=checksum, relative_error=0.0, seed,
            iterations=0, converged=true, zero_tensor=true)
    end

    effective_max_rank = min(Int(max_rank), cp_rank(source))
    effective_min_rank = max(Int(min_rank),
        initial === nothing ? 1 : cp_rank(initial))
    effective_min_rank <= effective_max_rank || throw(ArgumentError(
        "The requested/initial minimum rank exceeds the source rank " *
        "$(cp_rank(source))."))

    if backend === :cuda
        return _cp_compress_factorized_cuda(
            source; target_error=resolved_error,
            max_rank=effective_max_rank, min_rank=effective_min_rank,
            rank_step, seed, initial, maxiter, tolerance, ridge,
            check_every, restarts, source_checksum=checksum,
            device, synchronize, error_block_size, verbose)
    end

    source_weights, source_factors = _cp_real_factorized_source(source)
    source_norm_squared = real(_cp_factorized_inner_product(
        source_weights, source_factors, source_weights, source_factors;
        block_size=error_block_size))
    best = nothing
    warm = initial
    for rank in _cp_factorized_rank_sequence(
        effective_min_rank, Int(rank_step), effective_max_rank)
        if rank == cp_rank(source)
            exact = MMGNLSECPDecomposition(source_weights, source_factors;
                layout=metadata.layout, nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum=checksum, relative_error=0.0, seed,
                iterations=0, converged=true, zero_tensor=false)
            return exact
        end
        rank_best = nothing
        for restart in 1:Int(restarts)
            restart_initial = restart == 1 ? warm : nothing
            local_seed = Int(seed) + 104729 * rank + 1009 * (restart - 1)
            weights, factors, history, iterations, _ =
                _cp_als_factorized_real(
                    source_weights, source_factors, rank;
                    seed=local_seed, initial=restart_initial, maxiter,
                    tolerance, ridge, check_every,
                    target_error=resolved_error, source_norm_squared,
                    error_block_size, verbose)
            err = history[end]
            candidate = MMGNLSECPDecomposition(weights, factors;
                layout=metadata.layout, nmodes=metadata.nmodes,
                npolarizations=metadata.npolarizations,
                source_checksum=checksum, relative_error=err, seed,
                iterations, converged=err <= resolved_error,
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
        rank_best.relative_error <= resolved_error && return rank_best
    end

    throw(CPCompressionFailure(resolved_error, effective_max_rank, best))
end

function _cp_validate_memory_controls(rank_block_size, max_workspace_bytes,
                                      workspace_memory_fraction)
    if rank_block_size !== nothing
        rank_block_size > 0 || throw(ArgumentError(
            "rank_block_size must be positive or nothing."))
    end
    if max_workspace_bytes !== nothing
        max_workspace_bytes > 0 || throw(ArgumentError(
            "max_workspace_bytes must be positive or nothing."))
    end
    isfinite(workspace_memory_fraction) &&
        0 < workspace_memory_fraction <= 1 || throw(ArgumentError(
            "workspace_memory_fraction must be finite and in (0, 1]."))
    return nothing
end

function _cp_workspace_limited_rank_block(nmodes::Int, rank::Int, ::Type{T};
                                          rank_block_size=nothing,
                                          max_workspace_bytes=nothing) where {T}
    requested = rank_block_size === nothing ? rank :
                min(Int(rank_block_size), rank)
    max_workspace_bytes === nothing && return requested
    bytes_per_column = Base.checked_mul(
        Base.checked_mul(Base.checked_mul(nmodes, nmodes), nmodes), sizeof(T))
    affordable = Int(max_workspace_bytes) ÷ bytes_per_column
    affordable > 0 || throw(ArgumentError(
        "max_workspace_bytes=$(Int(max_workspace_bytes)) is too small for " *
        "one Khatri–Rao rank column ($bytes_per_column bytes at $nmodes modes)."))
    return min(requested, affordable)
end

function _cp_bounded_fit_type(S, fit_precision::Symbol)
    source_type = typeof(float(real(zero(eltype(S)))))
    fit_precision === :source && return source_type
    fit_precision === :float32 && return Float32
    fit_precision === :float64 && return Float64
    throw(ArgumentError(
        "fit_precision must be :source, :float32, or :float64."))
end

_cp_precision_symbol(::Type{Float32}) = :float32
_cp_precision_symbol(::Type{Float64}) = :float64
_cp_precision_symbol(::Type{T}) where {T} = Symbol(lowercase(string(T)))

function _cp_resolve_source_structure(S, source_structure::Symbol,
                                      verify_source_structure::Bool)
    source_structure in (:general, :permutation_symmetric, :auto) ||
        throw(ArgumentError(
            "source_structure must be :general, :permutation_symmetric, " *
            "or :auto."))
    source_structure === :general && return :general
    source_structure === :permutation_symmetric &&
        !verify_source_structure && return :permutation_symmetric

    defect = _cp_symmetric_tensor_defect(S)
    RT = typeof(float(real(zero(eltype(S)))))
    threshold = max(RT(1e-12), RT(100) * eps(RT))
    if source_structure === :auto
        return defect <= threshold ? :permutation_symmetric : :general
    end
    if verify_source_structure && defect > threshold
        throw(ArgumentError(
            "source_structure=:permutation_symmetric requires a fully " *
            "permutation-symmetric tensor; measured relative defect " *
            "$defect."))
    end
    return :permutation_symmetric
end

function _cp_checked_product(values::Integer...)
    result = 1
    for value in values
        result = Base.checked_mul(result, Int(value))
    end
    return result
end

"""
    cp_compression_plan(S, requested_rank; backend=:cpu,
                        fit_precision=:source,
                        source_structure=:general,
                        verify_source_structure=true,
                        memory=CPMemoryBudget(), device=nothing)

Resolve an experimental bounded CP-compression configuration without fitting
the tensor. `source_structure=:permutation_symmetric` selects the generic CP
algorithm while reusing the mode-1 source unfolding; it does **not** select
the separate tied-factor symmetric-CP approximation.

With `source_structure=:auto`, the source is scanned to decide whether this
reuse is valid. An explicit `:permutation_symmetric` source is also scanned
unless `verify_source_structure=false` is requested for a trusted source.
CUDA plans report the currently available device memory and conservatively
reserve the fixed allocations before sizing the Khatri--Rao rank block.
"""
function cp_compression_plan(
    S::AbstractArray{<:Number,4},
    requested_rank::Integer;
    backend::Symbol=:cpu,
    fit_precision::Symbol=:source,
    source_structure::Symbol=:general,
    verify_source_structure::Bool=true,
    memory::CPMemoryBudget=CPMemoryBudget(),
    device=nothing,
)
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    rank = Int(requested_rank)
    rank > 0 || throw(ArgumentError("requested_rank must be positive."))
    length(unique(size(S))) == 1 || throw(ArgumentError(
        "MMGNLSE overlap tensors must have four equal dimensions."))
    n = size(S, 1)
    n > 0 || throw(ArgumentError("The overlap tensor axes must be nonempty."))
    resolved_structure = _cp_resolve_source_structure(
        S, source_structure, verify_source_structure)
    RT = _cp_bounded_fit_type(S, fit_precision)
    CT = Complex{RT}
    value_bytes = sizeof(CT)
    n3 = _cp_checked_product(n, n, n)
    n4 = Base.checked_mul(n3, n)
    source_bytes = Base.checked_mul(n4, value_bytes)
    unfolding_bytes = resolved_structure === :general ? source_bytes : 0
    bytes_per_rank_column = Base.checked_mul(n3, value_bytes)
    factor_bytes = _cp_checked_product(4, n, rank, value_bytes)
    gram_bytes = _cp_checked_product(3, rank, rank, value_bytes)
    mttkrp_bytes = _cp_checked_product(n, rank, value_bytes)
    fixed_bytes = Base.checked_add(
        Base.checked_add(source_bytes, unfolding_bytes),
        Base.checked_add(
            factor_bytes, Base.checked_add(gram_bytes, mttkrp_bytes)))

    requested_block = memory.rank_block_size === :auto ? rank :
                      min(rank, memory.rank_block_size)
    free_device_bytes = nothing
    estimated_headroom_bytes = nothing
    workspace_limit = memory.max_workspace_bytes === nothing ?
                      Base.checked_mul(bytes_per_rank_column, requested_block) :
                      memory.max_workspace_bytes

    if backend === :cuda
        status = _cp_cuda_memory_status(
            ; device, reclaim_memory=memory.reclaim_memory)
        free_device_bytes = status.available_bytes
        fixed_headroom = max(free_device_bytes - fixed_bytes, 0)
        # The CUDA fitter resolves this fraction after the resident source
        # and factors have been allocated. Size from the remaining headroom,
        # not from the pre-fit free-memory reading, so a dry run cannot admit
        # a block that the actual fit immediately rejects.
        fraction_limit = floor(
            Int, memory.workspace_memory_fraction * fixed_headroom)
        workspace_limit = min(workspace_limit, fraction_limit, fixed_headroom)
    end

    affordable = workspace_limit ÷ bytes_per_rank_column
    affordable > 0 || throw(ArgumentError(
        "The requested memory budget cannot hold one Khatri--Rao rank " *
        "column ($bytes_per_rank_column bytes at $n modes)."))
    resolved_block = min(requested_block, affordable)
    workspace_bytes = Base.checked_mul(bytes_per_rank_column, resolved_block)
    estimated_peak_bytes = Base.checked_add(fixed_bytes, workspace_bytes)
    if free_device_bytes !== nothing
        estimated_headroom_bytes = free_device_bytes - estimated_peak_bytes
    end
    cuda_addressable = resolved_structure === :permutation_symmetric ||
                       n4 <= typemax(Cint)

    return CPCompressionPlan(
        n,
        rank,
        backend,
        _cp_precision_symbol(RT),
        resolved_structure,
        source_bytes,
        unfolding_bytes,
        bytes_per_rank_column,
        resolved_block,
        workspace_bytes,
        factor_bytes,
        gram_bytes,
        estimated_peak_bytes,
        free_device_bytes,
        estimated_headroom_bytes,
        cuda_addressable,
    )
end

"""
    cp_compress_bounded(S; memory=CPMemoryBudget(),
                        fit_precision=:source,
                        source_structure=:general, ...)

Experimental, opt-in front end for memory-bounded generic CP compression.
It returns the same `MMGNLSECPDecomposition` as `cp_compress`. The legacy
entry point and its default dispatch are not changed.

`fit_precision` controls the CP fit only and may be `:source`, `:float32`, or
`:float64`. `source_structure=:permutation_symmetric` only enables source
unfolding reuse; tied-factor symmetric CP remains available exclusively via
`cp_compress_symmetric_experimental`.
"""
function cp_compress_bounded(
    S::AbstractArray{<:Number,4};
    memory::CPMemoryBudget=CPMemoryBudget(),
    fit_precision::Symbol=:source,
    source_structure::Symbol=:general,
    verify_source_structure::Bool=true,
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
    backend::Symbol=:cpu,
    device=nothing,
    synchronize::Bool=true,
)
    plan = cp_compression_plan(
        S, max_rank;
        backend,
        fit_precision,
        source_structure,
        verify_source_structure,
        memory,
        device,
    )
    backend === :cuda && !plan.cuda_addressable && throw(ArgumentError(
        "The generic CUDA CP path cannot safely permute a tensor with " *
        "$(length(S)) elements. Use a verified permutation-symmetric " *
        "source, backend=:cpu, or fewer modes."))

    checksum = _resolved_checksum(S; source_checksum, checksum_fn)
    RT = _cp_bounded_fit_type(S, fit_precision)
    CT = Complex{RT}
    fit_source = backend === :cuda ?
        _cp_prepare_bounded_cuda_source(S, CT; device) :
        (S isa AbstractArray{CT,4} ? S : CT.(S))
    symmetric_source = plan.source_structure === :permutation_symmetric

    common = (;
        target_error,
        error,
        max_rank,
        min_rank,
        rank_step,
        seed,
        maxiter,
        tolerance,
        ridge,
        check_every,
        restarts,
        layout,
        nmodes,
        npolarizations,
        source_checksum=checksum,
        checksum_fn=nothing,
        fftw_threads,
        blas_threads,
        backend,
        device,
        synchronize,
        rank_block_size=plan.resolved_rank_block_size,
        max_workspace_bytes=memory.max_workspace_bytes,
        symmetric_source,
        check_symmetry=false,
    )
    if backend === :cuda
        return cp_compress(
            fit_source;
            common...,
            workspace_memory_fraction=memory.workspace_memory_fraction,
            reclaim_memory=memory.reclaim_memory,
        )
    end
    return cp_compress(fit_source; common...)
end

"""
    cp_is_symmetric(cp; atol=0, rtol=0)

Return `true` when `cp` has real weights and four real, mutually equal factor
matrices.  Exact equality is the default because the experimental CUDA
symmetric contraction changes the algebra of the nonlinear RHS and must not
be selected for an approximately symmetric representation.
"""
function cp_is_symmetric(cp::MMGNLSECPDecomposition;
                         atol::Real=0, rtol::Real=0)
    atol >= 0 || throw(ArgumentError("atol must be nonnegative."))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative."))
    all(isreal, cp.λ) || return false
    all(factor -> all(isreal, factor), cp.U) || return false
    reference = cp.U[1]
    return all(index -> isapprox(cp.U[index], reference; atol, rtol), 2:4)
end

"""
    cp_compression_memory_estimate(nmodes, rank;
                                   symmetric=false,
                                   precision=:float64,
                                   rank_block_size=rank)

Estimate the principal CUDA allocations for a fourth-order CP fit.  The
estimate is intentionally conservative and is intended for choosing
`rank_block_size` or `max_workspace_bytes`; CUDA allocator and linear-algebra
library overhead are not included.
"""
function cp_compression_memory_estimate(
    nmodes::Integer,
    rank::Integer;
    symmetric::Bool=false,
    precision::Symbol=:float64,
    rank_block_size::Integer=rank,
)
    n = Int(nmodes)
    r = Int(rank)
    block = Int(rank_block_size)
    n > 0 || throw(ArgumentError("nmodes must be positive."))
    r > 0 || throw(ArgumentError("rank must be positive."))
    block > 0 || throw(ArgumentError("rank_block_size must be positive."))
    block = min(block, r)
    precision in (:float32, :float64) || throw(ArgumentError(
        "precision must be :float32 or :float64."))
    real_bytes = precision === :float32 ? sizeof(Float32) : sizeof(Float64)
    value_bytes = symmetric ? real_bytes : 2real_bytes
    source_bytes = n^4 * value_bytes
    factor_bytes = 4n * r * value_bytes
    gram_bytes = 3r^2 * value_bytes
    full_khatri_rao_bytes = n^3 * r * value_bytes
    blocked_khatri_rao_bytes = n^3 * block * value_bytes
    mttkrp_bytes = n * r * value_bytes
    unfolding_bytes = symmetric ? 0 : source_bytes
    estimated_peak_bytes = source_bytes + factor_bytes + gram_bytes +
                           blocked_khatri_rao_bytes + mttkrp_bytes +
                           unfolding_bytes
    return (;
        nmodes=n,
        rank=r,
        rank_block_size=block,
        symmetric,
        precision,
        source_bytes,
        factor_bytes,
        gram_bytes,
        full_khatri_rao_bytes,
        blocked_khatri_rao_bytes,
        mttkrp_bytes,
        unfolding_bytes,
        estimated_peak_bytes,
    )
end

function _cp_symmetric_tensor_defect(S)
    source_norm2 = sum(abs2, S)
    source_norm2 == 0 && return 0.0
    defects = zeros(Float64, 3)
    @inbounds for i4 in axes(S, 4), i3 in axes(S, 3),
                  i2 in axes(S, 2), i1 in axes(S, 1)
        value = S[i1, i2, i3, i4]
        defects[1] += abs2(value - S[i2, i1, i3, i4])
        defects[2] += abs2(value - S[i1, i3, i2, i4])
        defects[3] += abs2(value - S[i1, i2, i4, i3])
    end
    return sqrt(maximum(defects) / Float64(source_norm2))
end

function _cp_symmetric_normalize_columns!(factor)
    RT = eltype(factor)
    norms = vec(sqrt.(sum(abs2, factor; dims=1)))
    factor ./= reshape(max.(norms, eps(RT)), 1, :)
    return factor
end

function _cp_symmetric_khatri_rao_cube!(destination, factor)
    nmodes, rank = size(factor)
    size(destination) == (nmodes^3, rank) || throw(DimensionMismatch(
        "Symmetric Khatri–Rao destination has size $(size(destination)); " *
        "expected $((nmodes^3, rank))."))
    reshape(destination, nmodes, nmodes, nmodes, rank) .=
        reshape(factor, nmodes, 1, 1, rank) .*
        reshape(factor, 1, nmodes, 1, rank) .*
        reshape(factor, 1, 1, nmodes, rank)
    return destination
end

function _cp_symmetric_mttkrp(source_mode1, factor,
                              rank_block_size::Int)
    rank = size(factor, 2)
    block = min(rank, rank_block_size)
    block > 0 || throw(ArgumentError(
        "rank_block_size must be positive."))
    result = similar(factor)
    khatri_rao_workspace = similar(
        factor, size(factor, 1)^3, block)
    for first_column in 1:block:rank
        last_column = min(first_column + block - 1, rank)
        columns = first_column:last_column
        factor_block = @view factor[:, columns]
        khatri = @view khatri_rao_workspace[:, 1:length(columns)]
        _cp_symmetric_khatri_rao_cube!(khatri, factor_block)
        mul!(@view(result[:, columns]), source_mode1, khatri)
    end
    return result
end

function _cp_symmetric_initial_factor(initial, nmodes::Int, rank::Int,
                                      rng, ::Type{RT}) where {RT<:AbstractFloat}
    factor = randn(rng, RT, nmodes, rank)
    if initial !== nothing
        hasproperty(initial, :U) || throw(ArgumentError(
            "initial must provide four CP factor matrices in initial.U."))
        factors = initial.U
        length(factors) == 4 || throw(ArgumentError(
            "initial must provide exactly four CP factor matrices."))
        all(size(source, 1) == nmodes for source in factors) ||
            throw(DimensionMismatch(
                "Initial CP factor dimensions do not match the source tensor."))
        all(source -> all(isreal, source), factors) || throw(ArgumentError(
            "The symmetric real fit requires real initial CP factors."))
        copied = min(rank, minimum(size(source, 2) for source in factors))
        for column in 1:copied
            reference = RT.(real.(view(factors[1], :, column)))
            candidate = zeros(RT, nmodes)
            for mode in 1:4
                source = RT.(real.(view(factors[mode], :, column)))
                alignment = dot(reference, source) < 0 ? -one(RT) : one(RT)
                candidate .+= alignment .* source
            end
            candidate_norm = norm(candidate)
            if candidate_norm > eps(RT)
                factor[:, column] .= candidate ./ candidate_norm
            end
        end
    end
    return _cp_symmetric_normalize_columns!(factor)
end

function _cp_symmetric_optimize(source, factor, identity;
                                maxiter::Int,
                                check_every::Int,
                                learning_rate::Real,
                                ridge::Real,
                                tolerance::Real,
                                target_error,
                                rank_block_size::Int,
                                verbose::Bool,
                                synchronize_fn=()->nothing)
    RT = eltype(factor)
    rank = size(factor, 2)
    source_mode1 = reshape(source, size(factor, 1), :)
    source_norm2 = sum(abs2, source)
    first_moment = similar(factor)
    second_moment = similar(factor)
    fill!(first_moment, zero(RT))
    fill!(second_moment, zero(RT))
    weights = similar(factor, RT, rank)
    fill!(weights, zero(RT))
    beta1 = RT(0.9)
    beta2 = RT(0.999)
    adam_epsilon = RT(1e-8)
    best_error = Inf
    best_factor = Array(factor)
    best_weights = zeros(RT, rank)
    best_iteration = 0
    previous_error = Inf
    converged = false

    for iteration in 1:maxiter
        mttkrp = _cp_symmetric_mttkrp(
            source_mode1, factor, rank_block_size)
        correlation = transpose(factor) * factor
        gram4 = correlation .^ 4
        rhs = vec(sum(factor .* mttkrp; dims=1))
        weights .= (gram4 .+ RT(ridge) .* identity) \ rhs

        coefficient_gram =
            (reshape(weights, :, 1) .* reshape(weights, 1, :)) .*
            (correlation .^ 3)
        gradient = RT(4) .* (
            factor * coefficient_gram .-
            mttkrp .* reshape(weights, 1, :)) ./ source_norm2
        radial = vec(sum(factor .* gradient; dims=1))
        gradient .-= factor .* reshape(radial, 1, :)

        first_moment .= beta1 .* first_moment .+ (one(RT) - beta1) .* gradient
        second_moment .= beta2 .* second_moment .+
                         (one(RT) - beta2) .* abs2.(gradient)
        first_scale = one(RT) - beta1^iteration
        second_scale = one(RT) - beta2^iteration
        factor .-= RT(learning_rate) .* (first_moment ./ first_scale) ./
                   (sqrt.(second_moment ./ second_scale) .+ adam_epsilon)
        _cp_symmetric_normalize_columns!(factor)

        if iteration == 1 || iteration % check_every == 0 ||
           iteration == maxiter
            mttkrp = _cp_symmetric_mttkrp(
                source_mode1, factor, rank_block_size)
            correlation = transpose(factor) * factor
            gram4 = correlation .^ 4
            rhs = vec(sum(factor .* mttkrp; dims=1))
            weights .= (gram4 .+ RT(ridge) .* identity) \ rhs
            residual2 = source_norm2 - RT(2) * dot(weights, rhs) +
                        dot(weights, gram4 * weights)
            synchronize_fn()
            relative_error = sqrt(max(
                Float64(residual2 / source_norm2), 0.0))
            verbose && @info "Symmetric CP fit" iteration relative_error
            if relative_error < best_error
                best_error = relative_error
                best_factor = Array(factor)
                best_weights = Array(weights)
                best_iteration = iteration
            end
            hit_target = target_error !== nothing &&
                         relative_error <= target_error
            stalled = isfinite(previous_error) &&
                      abs(previous_error - relative_error) <=
                      tolerance * max(previous_error, eps(Float64))
            if hit_target || stalled
                converged = true
                break
            end
            previous_error = relative_error
        end
    end
    return best_weights, best_factor, best_error, best_iteration, converged
end

function _cp_symmetric_result(weights, factor, relative_error, iterations,
                              converged, metadata, checksum, seed)
    return MMGNLSECPDecomposition(
        weights, ntuple(_ -> factor, 4);
        layout=metadata.layout,
        nmodes=metadata.nmodes,
        npolarizations=metadata.npolarizations,
        source_checksum=checksum,
        relative_error,
        seed,
        iterations,
        converged,
        zero_tensor=isempty(weights),
    )
end

"""
    cp_compress_symmetric_experimental(S, rank; backend=:cpu, initial=nothing, ...)

Fit a real, fully permutation-symmetric fourth-order tensor as

```math
S_{ijkl} \\approx \\sum_{r=1}^{R} \\lambda_r
u_{ir}u_{jr}u_{kr}u_{lr}.
```

This is an experimental, fixed-rank API.  It validates the symmetry of `S`,
uses a generic real CP fit as the default warm start, solves the weights by
regularized least squares at every iteration, and optimizes one shared,
column-normalized factor matrix.  Supply an existing real CP decomposition as
`initial` to skip the generic warm-start fit.  `backend=:cuda` performs the
expensive fitting operations on a CUDA device while returning the usual
host-resident `MMGNLSECPDecomposition`. `rank_block_size`,
`max_workspace_bytes`, `workspace_memory_fraction`, and `reclaim_memory`
provide the same bounded-workspace controls as `cp_compress`.
`check_symmetry=false` skips the otherwise allocation-free full scan for a
trusted tensor, such as an unmodified real overlap returned by
`compute_fiber_properties`.
"""
function cp_compress_symmetric_experimental(
    S::AbstractArray{<:Number,4}, rank::Integer;
    initial=nothing,
    backend::Symbol=:cpu,
    maxiter::Integer=500,
    check_every::Integer=10,
    learning_rate::Real=0.02,
    ridge::Real=1e-8,
    tolerance::Real=1e-10,
    target_error::Union{Nothing,Real}=nothing,
    symmetry_tolerance=nothing,
    initial_maxiter::Integer=75,
    initial_tolerance::Real=1e-7,
    seed::Integer=0x4d4d474e,
    layout::Symbol=:spatial,
    nmodes::Union{Nothing,Integer}=nothing,
    npolarizations::Integer=1,
    source_checksum=nothing,
    checksum_fn=nothing,
    verbose::Bool=false,
    device=nothing,
    synchronize::Bool=true,
    rank_block_size::Union{Nothing,Integer}=nothing,
    max_workspace_bytes::Union{Nothing,Integer}=nothing,
    workspace_memory_fraction::Real=0.5,
    reclaim_memory::Bool=true,
    check_symmetry::Bool=true,
)
    r = Int(rank)
    r > 0 || throw(ArgumentError("rank must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    check_every > 0 || throw(ArgumentError("check_every must be positive."))
    initial_maxiter > 0 || throw(ArgumentError(
        "initial_maxiter must be positive."))
    learning_rate > 0 || throw(ArgumentError(
        "learning_rate must be positive."))
    ridge >= 0 || throw(ArgumentError("ridge must be nonnegative."))
    tolerance >= 0 || throw(ArgumentError("tolerance must be nonnegative."))
    initial_tolerance >= 0 || throw(ArgumentError(
        "initial_tolerance must be nonnegative."))
    target_error === nothing || target_error >= 0 || throw(ArgumentError(
        "target_error must be nonnegative or nothing."))
    backend in (:cpu, :cuda) || throw(ArgumentError(
        "backend must be :cpu or :cuda; got $(repr(backend))."))
    _cp_validate_memory_controls(
        rank_block_size, max_workspace_bytes, workspace_memory_fraction)
    all(isfinite, S) || throw(ArgumentError(
        "The overlap tensor contains non-finite values."))
    all(isreal, S) || throw(ArgumentError(
        "The experimental symmetric CP fit currently requires a real tensor."))

    RT = typeof(float(real(zero(eltype(S)))))
    source = S isa Array{RT,4} ? S : RT.(real.(S))
    resolved_symmetry_tolerance = symmetry_tolerance === nothing ?
        max(RT(1e-12), RT(100) * eps(RT)) : RT(symmetry_tolerance)
    resolved_symmetry_tolerance >= 0 || throw(ArgumentError(
        "symmetry_tolerance must be nonnegative."))
    if check_symmetry
        symmetry_defect = _cp_symmetric_tensor_defect(source)
        symmetry_defect <= resolved_symmetry_tolerance || throw(ArgumentError(
            "The source tensor is not fully permutation symmetric: relative " *
            "defect $symmetry_defect exceeds $resolved_symmetry_tolerance."))
    end

    checksum = _resolved_checksum(
        S; source_checksum, checksum_fn)
    metadata = _cp_source_metadata(
        size(S); layout, nmodes, npolarizations,
        source_checksum=checksum)
    if iszero(norm(source))
        factor = zeros(RT, size(source, 1), 0)
        return _cp_symmetric_result(
            RT[], factor, 0.0, 0, true, metadata, checksum, seed)
    end

    if backend === :cuda
        return _cp_compress_symmetric_experimental_cuda(
            source, r;
            initial,
            maxiter=Int(maxiter),
            check_every=Int(check_every),
            learning_rate,
            ridge,
            tolerance,
            target_error,
            initial_maxiter=Int(initial_maxiter),
            initial_tolerance,
            seed=Int(seed),
            metadata,
            checksum,
            verbose,
            device,
            synchronize,
            rank_block_size,
            max_workspace_bytes,
            workspace_memory_fraction,
            reclaim_memory,
        )
    end

    resolved_rank_block_size = _cp_workspace_limited_rank_block(
        size(source, 1), r, RT;
        rank_block_size,
        max_workspace_bytes)
    if verbose
        @info "Symmetric CP workspace" backend=:cpu rank=r rank_block_size=resolved_rank_block_size
    end
    rng = MersenneTwister(Int(seed))
    warm = initial
    if warm === nothing
        warm, _ = cp_als_warm(
            source, r;
            maxiter=Int(initial_maxiter),
            tol=initial_tolerance,
            ridge=RT(ridge),
            check_every=Int(check_every),
            rng,
            verbose,
            rank_block_size=resolved_rank_block_size,
            symmetric_source=true,
        )
    end
    factor = _cp_symmetric_initial_factor(
        warm, size(source, 1), r, rng, RT)
    identity = Matrix{RT}(I, r, r)
    weights, best_factor, relative_error, iterations, converged =
        _cp_symmetric_optimize(
            source, factor, identity;
            maxiter=Int(maxiter),
            check_every=Int(check_every),
            learning_rate,
            ridge,
            tolerance,
            target_error,
            rank_block_size=resolved_rank_block_size,
            verbose,
        )
    return _cp_symmetric_result(
        weights, best_factor, relative_error, iterations, converged,
        metadata, checksum, seed)
end

if isdefined(@__MODULE__, :SpatialOverlap)
    @eval begin
        _coerce_overlap(overlap::MMGNLSECPDecomposition) = overlap
        num_spatial_modes(overlap::MMGNLSECPDecomposition) = overlap.metadata.nmodes
        has_component_overlap(overlap::MMGNLSECPDecomposition) =
            overlap.metadata.layout == :component

        function cp_compress(overlap::SpatialOverlap; kwargs...)
            return cp_compress(overlap.values; layout=:spatial,
                               nmodes=num_spatial_modes(overlap), kwargs...)
        end

        function cp_compress(overlap::ComponentOverlap; kwargs...)
            flattened = flatten_component_tensor(overlap.values)
            return cp_compress(flattened; layout=:component,
                               nmodes=num_spatial_modes(overlap),
                               npolarizations=2, kwargs...)
        end

        function cp_compression_plan(overlap::SpatialOverlap,
                                     requested_rank::Integer; kwargs...)
            return cp_compression_plan(
                overlap.values, requested_rank; kwargs...)
        end

        function cp_compression_plan(overlap::ComponentOverlap,
                                     requested_rank::Integer; kwargs...)
            flattened = flatten_component_tensor(overlap.values)
            return cp_compression_plan(flattened, requested_rank; kwargs...)
        end

        function cp_compress_bounded(overlap::SpatialOverlap; kwargs...)
            return cp_compress_bounded(
                overlap.values;
                layout=:spatial,
                nmodes=num_spatial_modes(overlap),
                kwargs...)
        end

        function cp_compress_bounded(overlap::ComponentOverlap; kwargs...)
            flattened = flatten_component_tensor(overlap.values)
            return cp_compress_bounded(
                flattened;
                layout=:component,
                nmodes=num_spatial_modes(overlap),
                npolarizations=2,
                kwargs...)
        end

        function cp_compress_symmetric_experimental(
            overlap::SpatialOverlap, rank::Integer; kwargs...)
            return cp_compress_symmetric_experimental(
                overlap.values, rank;
                layout=:spatial,
                nmodes=num_spatial_modes(overlap),
                kwargs...)
        end

        function cp_compress_symmetric_experimental(
            overlap::ComponentOverlap, rank::Integer; kwargs...)
            flattened = flatten_component_tensor(overlap.values)
            return cp_compress_symmetric_experimental(
                flattened, rank;
                layout=:component,
                nmodes=num_spatial_modes(overlap),
                npolarizations=2,
                kwargs...)
        end
    end
end

"""
    cp_source_matches(cp, S; source_checksum=nothing, checksum_fn=nothing)

Check tensor dimensions and, when the decomposition stores a checksum, source
identity. A checksum must be supplied explicitly or generated with the same
canonical checksum function used at compression time.
"""
function cp_source_matches(cp::MMGNLSECPDecomposition,
                           S::AbstractArray{<:Number,4};
                           source_checksum=nothing,
                           checksum_fn=nothing)
    size(S) == cp.metadata.source_size || return false
    stored = cp.metadata.source_checksum
    stored === nothing && return true
    candidate = _resolved_checksum(S; source_checksum, checksum_fn)
    candidate === nothing && return false
    return candidate == stored
end

function validate_cp_source(cp::MMGNLSECPDecomposition,
                            S::AbstractArray{<:Number,4}; kwargs...)
    cp_source_matches(cp, S; kwargs...) ||
        throw(ArgumentError("The CP decomposition does not match the supplied source tensor."))
    return cp
end

function _validate_spatial_cp(cp::MMGNLSECPDecomposition, nmodes::Int)
    cp.metadata.layout == :spatial ||
        throw(ArgumentError("This contraction requires a spatial CP decomposition."))
    cp.metadata.nmodes == nmodes ||
        throw(DimensionMismatch("Field has $nmodes modes; CP metadata has $(cp.metadata.nmodes)."))
    return cp
end

"""
    dense_spatial_isotropic_cubic(A, S)

Apply an isotropic silica instantaneous contraction to fields with public
shape `time × mode × polarization`. One polarization is the linearly
x-polarized scalar reduction. With two polarizations the x channel is

`xxx* + (2/3) xyy* + (1/3) yyx*`,

and the y channel is obtained by exchanging x and y.
"""
function dense_spatial_isotropic_cubic(A::AbstractArray{<:Number,3},
                                       S::AbstractArray{<:Number,4})
    nt, nm, npol = size(A)
    npol in (1, 2) || throw(DimensionMismatch("Polarization dimension must be one or two."))
    size(S) == (nm, nm, nm, nm) ||
        throw(DimensionMismatch("S must have size ($nm, $nm, $nm, $nm)."))
    T = promote_type(eltype(A), eltype(S), Float64)
    out = zeros(T, nt, nm, npol)
    c23 = T(2) / T(3)
    c13 = T(1) / T(3)
    @inbounds for t in 1:nt, i1 in 1:nm, i4 in 1:nm, i3 in 1:nm, i2 in 1:nm
        s = S[i1, i2, i3, i4]
        ax2 = A[t, i2, 1]
        ax3 = A[t, i3, 1]
        ax4c = conj(A[t, i4, 1])
        out[t, i1, 1] += s * ax2 * ax3 * ax4c
        if npol == 2
            ay2 = A[t, i2, 2]
            ay3 = A[t, i3, 2]
            ay4c = conj(A[t, i4, 2])
            out[t, i1, 1] += s * (c23 * ax2 * ay3 * ay4c +
                                   c13 * ay2 * ay3 * ax4c)
            out[t, i1, 2] += s * (ay2 * ay3 * ay4c +
                                   c23 * ay2 * ax3 * ax4c +
                                   c13 * ax2 * ax3 * ay4c)
        end
    end
    return out
end

dense_spatial_isotropic_cubic(A::AbstractMatrix, S::AbstractArray{<:Number,4}) =
    dropdims(dense_spatial_isotropic_cubic(reshape(A, size(A, 1), size(A, 2), 1), S);
             dims=3)

function _cp_projected_fields(A::AbstractMatrix, cp::MMGNLSECPDecomposition)
    U1, U2, U3, U4 = cp.U
    B2 = A * U2
    B3 = A * U3
    B4 = conj.(A) * U4
    WU1 = U1 .* reshape(cp.λ, 1, :)
    return B2, B3, B4, WU1
end

"""CP implementation of `dense_spatial_isotropic_cubic`."""
function cp_spatial_isotropic_cubic(A::AbstractArray{<:Number,3},
                                    cp::MMGNLSECPDecomposition)
    nt, nm, npol = size(A)
    npol in (1, 2) || throw(DimensionMismatch("Polarization dimension must be one or two."))
    _validate_spatial_cp(cp, nm)
    T = promote_type(eltype(A), eltype(cp.λ), map(eltype, cp.U)...)
    cp_iszero(cp) && return zeros(T, nt, nm, npol)

    Ax = @view A[:, :, 1]
    B2x, B3x, B4x, WU1 = _cp_projected_fields(Ax, cp)
    out = zeros(T, nt, nm, npol)
    if npol == 1
        out[:, :, 1] .= (B2x .* B3x .* B4x) * transpose(WU1)
        return out
    end

    Ay = @view A[:, :, 2]
    _, U2, U3, U4 = cp.U
    B2y = Ay * U2
    B3y = Ay * U3
    B4y = conj.(Ay) * U4
    c23 = T(2) / T(3)
    c13 = T(1) / T(3)
    Px = B2x .* B3x .* B4x .+
         c23 .* B2x .* B3y .* B4y .+
         c13 .* B2y .* B3y .* B4x
    Py = B2y .* B3y .* B4y .+
         c23 .* B2y .* B3x .* B4x .+
         c13 .* B2x .* B3x .* B4y
    out[:, :, 1] .= Px * transpose(WU1)
    out[:, :, 2] .= Py * transpose(WU1)
    return out
end

cp_spatial_isotropic_cubic(A::AbstractMatrix, cp::MMGNLSECPDecomposition) =
    dropdims(cp_spatial_isotropic_cubic(reshape(A, size(A, 1), size(A, 2), 1), cp);
             dims=3)

"""Flatten `time × mode × polarization` using mode-fast component order."""
function flatten_field_components(A::AbstractArray{<:Number,3})
    nt, nm, npol = size(A)
    return reshape(A, nt, nm * npol)
end

function unflatten_field_components(A::AbstractMatrix, nmodes::Integer,
                                    npolarizations::Integer)
    size(A, 2) == nmodes * npolarizations ||
        throw(DimensionMismatch("Flattened field component count is inconsistent."))
    return reshape(A, size(A, 1), Int(nmodes), Int(npolarizations))
end

"""
Flatten a tensor with axes `(m1,p1,m2,p2,m3,p3,m4,p4)` to four mode-fast
component axes.
"""
function flatten_component_tensor(S::AbstractArray{<:Number,8})
    dims = size(S)
    nm = dims[1]
    npol = dims[2]
    all(dims[2q - 1] == nm && dims[2q] == npol for q in 1:4) ||
        throw(DimensionMismatch("All four (mode, polarization) axis pairs must match."))
    return reshape(S, ntuple(_ -> nm * npol, 4))
end

"""Build the fully expanded isotropic component tensor from spatial `S`."""
function isotropic_component_tensor(S::AbstractArray{<:Number,4};
                                    npolarizations::Integer=2)
    nm = size(S, 1)
    size(S) == (nm, nm, nm, nm) ||
        throw(DimensionMismatch("S must have four equal spatial dimensions."))
    npol = Int(npolarizations)
    npol in (1, 2) || throw(ArgumentError("npolarizations must be one or two."))
    npol == 1 && return copy(S)
    T = promote_type(eltype(S), Float64)
    out = zeros(T, 2nm, 2nm, 2nm, 2nm)
    c23 = T(2) / T(3)
    c13 = T(1) / T(3)
    component(m, p) = m + (p - 1) * nm
    @inbounds for i4 in 1:nm, i3 in 1:nm, i2 in 1:nm, i1 in 1:nm
        s = S[i1, i2, i3, i4]
        x1, x2, x3, x4 = component(i1, 1), component(i2, 1), component(i3, 1), component(i4, 1)
        y1, y2, y3, y4 = component(i1, 2), component(i2, 2), component(i3, 2), component(i4, 2)
        out[x1, x2, x3, x4] += s
        out[y1, y2, y3, y4] += s
        out[x1, x2, y3, y4] += c23 * s
        out[y1, y2, x3, x4] += c23 * s
        out[x1, y2, y3, x4] += c13 * s
        out[y1, x2, x3, y4] += c13 * s
    end
    return out
end

"""Dense cubic contraction for an arbitrary flattened component tensor."""
function dense_component_cubic(A::AbstractMatrix,
                               S::AbstractArray{<:Number,4})
    nt, nc = size(A)
    size(S) == (nc, nc, nc, nc) ||
        throw(DimensionMismatch("Component tensor must have four axes of length $nc."))
    T = promote_type(eltype(A), eltype(S))
    out = zeros(T, nt, nc)
    @inbounds for t in 1:nt, i1 in 1:nc, i4 in 1:nc, i3 in 1:nc, i2 in 1:nc
        out[t, i1] += S[i1, i2, i3, i4] * A[t, i2] * A[t, i3] * conj(A[t, i4])
    end
    return out
end

function dense_component_cubic(A::AbstractArray{<:Number,3},
                               S::AbstractArray{<:Number,4})
    nt, nm, npol = size(A)
    flat = dense_component_cubic(flatten_field_components(A), S)
    return reshape(flat, nt, nm, npol)
end

"""CP cubic contraction for an arbitrary flattened component tensor."""
function cp_component_cubic(A::AbstractMatrix, cp::MMGNLSECPDecomposition)
    cp.metadata.layout == :component ||
        throw(ArgumentError("This contraction requires a component CP decomposition."))
    nc = size(A, 2)
    cp.metadata.source_size == (nc, nc, nc, nc) ||
        throw(DimensionMismatch("Field and component CP dimensions do not match."))
    T = promote_type(eltype(A), eltype(cp.λ), map(eltype, cp.U)...)
    cp_iszero(cp) && return zeros(T, size(A))
    B2, B3, B4, WU1 = _cp_projected_fields(A, cp)
    return (B2 .* B3 .* B4) * transpose(WU1)
end

function cp_component_cubic(A::AbstractArray{<:Number,3},
                            cp::MMGNLSECPDecomposition)
    nt, nm, npol = size(A)
    (nm == cp.metadata.nmodes && npol == cp.metadata.npolarizations) ||
        throw(DimensionMismatch("Field shape does not match component CP metadata."))
    flat = cp_component_cubic(flatten_field_components(A), cp)
    return reshape(flat, nt, nm, npol)
end

function _filtered_rank_channels(filter_channels, channels)
    filtered = filter_channels(channels)
    size(filtered) == size(channels) ||
        throw(DimensionMismatch("A Raman rank-channel filter must preserve array shape."))
    return filtered
end

"""
    cp_spatial_raman_cubic(A, cp, filter_channels; part=:isotropic)

Contract generated rank channels for silica's delayed response. The callback
`filter_channels` applies the sampled Raman convolution independently down
each rank column. `part=:isotropic` implements the `h_a` tensor and
`part=:anisotropic` implements the `h_b` tensor. Calling both with their
respective kernels and adding the results gives the anisotropic Raman term.
For the one-polarization reduction, either part has the scalar cubic spatial
contraction; the Raman model controls which kernels are present.
"""
function cp_spatial_raman_cubic(A::AbstractArray{<:Number,3},
                                cp::MMGNLSECPDecomposition,
                                filter_channels;
                                part::Symbol=:isotropic)
    part in (:isotropic, :anisotropic) ||
        throw(ArgumentError("part must be :isotropic or :anisotropic."))
    nt, nm, npol = size(A)
    npol in (1, 2) || throw(DimensionMismatch("Polarization dimension must be one or two."))
    _validate_spatial_cp(cp, nm)
    T = promote_type(eltype(A), eltype(cp.λ), map(eltype, cp.U)...)
    cp_iszero(cp) && return zeros(T, nt, nm, npol)

    Ax = @view A[:, :, 1]
    B2x, B3x, B4x, WU1 = _cp_projected_fields(Ax, cp)
    out = zeros(T, nt, nm, npol)
    if npol == 1
        q = _filtered_rank_channels(filter_channels, B3x .* B4x)
        out[:, :, 1] .= (B2x .* q) * transpose(WU1)
        return out
    end

    Ay = @view A[:, :, 2]
    _, U2, U3, U4 = cp.U
    B2y = Ay * U2
    B3y = Ay * U3
    B4y = conj.(Ay) * U4
    if part == :isotropic
        q = _filtered_rank_channels(filter_channels, B3x .* B4x .+ B3y .* B4y)
        out[:, :, 1] .= (B2x .* q) * transpose(WU1)
        out[:, :, 2] .= (B2y .* q) * transpose(WU1)
    else
        qxx = _filtered_rank_channels(filter_channels, B3x .* B4x)
        qyy = _filtered_rank_channels(filter_channels, B3y .* B4y)
        qxy = _filtered_rank_channels(filter_channels, B3x .* B4y .+ B3y .* B4x)
        half = T(1) / T(2)
        out[:, :, 1] .= (B2x .* qxx .+ half .* B2y .* qxy) * transpose(WU1)
        out[:, :, 2] .= (B2y .* qyy .+ half .* B2x .* qxy) * transpose(WU1)
    end
    return out
end
