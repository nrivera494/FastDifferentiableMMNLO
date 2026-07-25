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
                        target_error::Real=0.0)
    r = Int(rank)
    r > 0 || throw(ArgumentError("rank must be positive."))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive."))
    check_every > 0 || throw(ArgumentError("check_every must be positive."))
    RT = typeof(float(real(zero(eltype(X)))))
    CT = Complex{RT}
    Xc = CT.(X)
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

    for iteration in 1:Int(maxiter)
        iterations = iteration
        for n in 1:4
            rest = [k for k in 1:4 if k != n]
            KR = _khatri_rao_rows_mmgnlse([U[k] for k in rest])
            gram = ones(CT, r, r)
            for k in rest
                gram .*= transpose(U[k]) * conj.(U[k])
            end
            ridge_scale = max(maximum(abs, gram), one(RT))
            @inbounds for q in 1:r
                gram[q, q] += RT(ridge) * ridge_scale
            end
            weighted_factor = (_mode_unfold_mmgnlse(Xc, n) * conj.(KR)) / gram
            λ = _normalize_cp_columns!(weighted_factor, floor_norm)
            U = Base.setindex(U, weighted_factor, n)
        end

        if iteration == 1 || iteration % check_every == 0 || iteration == maxiter
            err = _cp_relative_error(Xc, λ, U)
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
    cp_compress(S; target_error, max_rank, ...)

Deterministically search CP ranks from `min_rank` through `max_rank` until the
relative Frobenius error is at most `target_error`. The fixed default seed is
part of the returned provenance. Rank searches are warm-started; optional
additional deterministic fresh starts can be requested with `restarts`.

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
                     checksum_fn=nothing)
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

    best = nothing
    warm = nothing
    for rank in Int(min_rank):Int(rank_step):Int(max_rank)
        rank_best = nothing
        for restart in 1:Int(restarts)
            init = restart == 1 ? warm : nothing
            local_seed = Int(seed) + 104729 * rank + 1009 * (restart - 1)
            λ, U, history, iterations, als_converged = cp_als_complex(S, rank;
                seed=local_seed, initial=init, maxiter, tolerance, ridge,
                check_every, target_error=resolved_error)
            err = isempty(history) ? _cp_relative_error(CT.(S), λ, U) : history[end]
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
                     blas_threads::Integer=1)
    fftw_threads > 0 || throw(ArgumentError("fftw_threads must be positive."))
    blas_threads > 0 || throw(ArgumentError("blas_threads must be positive."))
    previous_fftw_threads = FFTW.get_num_threads()
    previous_blas_threads = BLAS.get_num_threads()
    try
        FFTW.set_num_threads(Int(fftw_threads))
        BLAS.set_num_threads(Int(blas_threads))
        return _cp_compress_impl(S; target_error, error, max_rank, min_rank,
            rank_step, seed, maxiter, tolerance, ridge, check_every, restarts,
            layout, nmodes, npolarizations, source_checksum, checksum_fn)
    finally
        FFTW.set_num_threads(previous_fftw_threads)
        BLAS.set_num_threads(previous_blas_threads)
    end
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
