"""
    _mmgnlse_vjp_analytic(field_t, lambda_t, parameters, z)

Analytic real-pairing vector--Jacobian product of `_mmgnlse_rhs`.  Its
defining convention is

`2real(sum(conj(lambda) .* (Df[A] * delta_A)))` equals
`2real(sum(conj(vjp) .* delta_A))`.

The implementation makes no permutation-symmetry assumption about the
overlap tensor.  Dense contractions scale as the corresponding dense forward
contraction, while CP contractions use rank channels and scale as
`O(Nt * Nm * rank)` (up to the fixed polarization factor).
"""
function _mmgnlse_vjp_analytic(field_t::AbstractArray{<:Complex,3},
                                lambda_t::AbstractArray{<:Complex,3},
                                parameters::MMGNLSEParameters,
                                z::Real,
                                cache=_mmgnlse_solver_cache(parameters))
    validate_public_field(field_t, parameters.domain,
                          parameter_mode_count(parameters); name=:field)
    size(lambda_t) == size(field_t) || throw(DimensionMismatch(
        "The VJP state and cotangent must have matching public shapes."))
    all(_mmgnlse_isfinite, lambda_t) || throw(ArgumentError(
        "The VJP cotangent contains a non-finite value."))

    field = Array(field_t)
    cotangent = Array(lambda_t)
    # fft(D .* ifft(A)) is a Fourier-similarity transform, so its adjoint has
    # exactly the same transforms with conj(D).  The FFT normalization factors
    # cancel.
    linear_operator = _mmgnlse_linear_operator(parameters, z, cache)
    result = fft(conj.(linear_operator) .* ifft(cotangent, 1), 1)

    result .+= _mmgnlse_vjp_nonlinear(field, cotangent, parameters, cache)
    return result
end

"""Nonlinear-only temporal VJP used by the interaction-picture adjoint."""
function _mmgnlse_vjp_nonlinear(field_t::AbstractArray{<:Complex,3},
                                 lambda_t::AbstractArray{<:Complex,3},
                                 parameters::MMGNLSEParameters,
                                 cache=_mmgnlse_solver_cache(parameters))
    validate_public_field(field_t, parameters.domain,
                          parameter_mode_count(parameters); name=:field)
    size(lambda_t) == size(field_t) || throw(DimensionMismatch(
        "The VJP state and cotangent must have matching public shapes."))
    all(_mmgnlse_isfinite, lambda_t) || throw(ArgumentError(
        "The VJP cotangent contains a non-finite value."))

    field = Array(field_t)
    cotangent = Array(lambda_t)
    result = zeros(ComplexF64, size(field))
    all(iszero, cache.nonlinear_prefactor) && return result

    # Pull the temporal cotangent through
    # fft(prefactor .* ifft(cubic(A))).
    cubic_cotangent = fft(
        conj.(cache.nonlinear_prefactor) .* ifft(cotangent, 1), 1)
    fraction = cache.raman.fraction
    if fraction != 1
        result .+= _mmgnlse_vjp_instantaneous(
            field, (1 - fraction) .* cubic_cotangent, parameters.S)
    end
    if !iszero(fraction)
        result .+= _mmgnlse_vjp_delayed(
            field, fraction .* cubic_cotangent, parameters.S, cache.raman)
    end
    return result
end


# -- Instantaneous dense contractions ---------------------------------------

function _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, S,
                                         output_pol::Int, p2::Int,
                                         p3::Int, p4::Int, coefficient)
    nt, nm, _ = size(field)
    @inbounds for t in 1:nt, i1 in 1:nm, i2 in 1:nm, i3 in 1:nm, i4 in 1:nm
        s = coefficient * S[i1, i2, i3, i4]
        lambda_i = cotangent[t, i1, output_pol]
        a2 = field[t, i2, p2]
        a3 = field[t, i3, p3]
        a4 = field[t, i4, p4]

        # Holomorphic second and third slots.
        out[t, i2, p2] += lambda_i * conj(s) * conj(a3) * a4
        out[t, i3, p3] += lambda_i * conj(s) * conj(a2) * a4
        # Antiholomorphic fourth slot.
        out[t, i4, p4] += conj(lambda_i) * s * a2 * a3
    end
    return out
end

function _mmgnlse_vjp_instantaneous(field, cotangent,
                                     overlap::SpatialOverlap)
    nt, nm, npol = size(field)
    T = promote_type(eltype(field), eltype(cotangent),
                     eltype(overlap.values), ComplexF64)
    out = zeros(T, nt, nm, npol)
    if npol == 1
        _mmgnlse_dense_cubic_term_vjp!(
            out, field, cotangent, overlap.values, 1, 1, 1, 1, one(T))
        return out
    end

    c23 = T(2) / T(3)
    c13 = T(1) / T(3)
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    1, 1, 1, 1, one(T))
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    1, 1, 2, 2, c23)
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    1, 2, 2, 1, c13)
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    2, 2, 2, 2, one(T))
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    2, 2, 1, 1, c23)
    _mmgnlse_dense_cubic_term_vjp!(out, field, cotangent, overlap.values,
                                    2, 1, 1, 2, c13)
    return out
end

function _mmgnlse_dense_component_vjp(field::AbstractMatrix,
                                       cotangent::AbstractMatrix, S)
    nt, nc = size(field)
    size(S) == (nc, nc, nc, nc) || throw(DimensionMismatch(
        "The component tensor and flattened field dimensions do not match."))
    T = promote_type(eltype(field), eltype(cotangent), eltype(S), ComplexF64)
    out = zeros(T, nt, nc)
    @inbounds for t in 1:nt, i1 in 1:nc, i2 in 1:nc, i3 in 1:nc, i4 in 1:nc
        s = S[i1, i2, i3, i4]
        lambda_i = cotangent[t, i1]
        a2 = field[t, i2]
        a3 = field[t, i3]
        a4 = field[t, i4]
        out[t, i2] += lambda_i * conj(s) * conj(a3) * a4
        out[t, i3] += lambda_i * conj(s) * conj(a2) * a4
        out[t, i4] += conj(lambda_i) * s * a2 * a3
    end
    return out
end

function _mmgnlse_vjp_instantaneous(field, cotangent,
                                     overlap::ComponentOverlap)
    nt, nm, npol = size(field)
    flat_field = reshape(field, nt, nm * npol)
    flat_cotangent = reshape(cotangent, nt, nm * npol)
    flat_tensor = flatten_component_tensor(overlap.values)
    return reshape(_mmgnlse_dense_component_vjp(
        flat_field, flat_cotangent, flat_tensor), nt, nm, npol)
end


# -- Instantaneous CP contractions ------------------------------------------

function _mmgnlse_cp_generic_instantaneous_vjp(field::AbstractMatrix,
                                                 cotangent::AbstractMatrix,
                                                 cp::MMGNLSECPDecomposition)
    nt, nc = size(field)
    U1, U2, U3, U4 = cp.U
    T = promote_type(eltype(field), eltype(cotangent), eltype(cp.λ),
                     map(eltype, cp.U)..., ComplexF64)
    cp_iszero(cp) && return zeros(T, nt, nc)

    B2 = field * U2
    B3 = field * U3
    B4 = conj.(field) * U4
    WU1 = U1 .* reshape(cp.λ, 1, :)
    rank_cotangent = cotangent * conj.(WU1)

    vB2 = rank_cotangent .* conj.(B3 .* B4)
    vB3 = rank_cotangent .* conj.(B2 .* B4)
    vB4 = rank_cotangent .* conj.(B2 .* B3)
    return vB2 * adjoint(U2) .+
           vB3 * adjoint(U3) .+
           conj.(vB4 * adjoint(U4))
end

function _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                        cp::MMGNLSECPDecomposition,
                                        output_pol::Int, p2::Int,
                                        p3::Int, p4::Int, coefficient)
    _, U2, U3, U4 = cp.U
    B2 = projections[p2][1]
    B3 = projections[p3][2]
    B4 = projections[p4][3]
    rank_cotangent = conj(coefficient) .* rank_cotangents[output_pol]

    vB2 = rank_cotangent .* conj.(B3 .* B4)
    vB3 = rank_cotangent .* conj.(B2 .* B4)
    vB4 = rank_cotangent .* conj.(B2 .* B3)
    @views out[:, :, p2] .+= vB2 * adjoint(U2)
    @views out[:, :, p3] .+= vB3 * adjoint(U3)
    @views out[:, :, p4] .+= conj.(vB4 * adjoint(U4))
    return out
end

function _mmgnlse_cp_spatial_instantaneous_vjp(field, cotangent, cp)
    nt, nm, npol = size(field)
    _validate_spatial_cp(cp, nm)
    T = promote_type(eltype(field), eltype(cotangent), eltype(cp.λ),
                     map(eltype, cp.U)..., ComplexF64)
    out = zeros(T, nt, nm, npol)
    cp_iszero(cp) && return out

    U1, U2, U3, U4 = cp.U
    WU1 = U1 .* reshape(cp.λ, 1, :)
    projections = ntuple(npol) do polarization
        A = @view field[:, :, polarization]
        (A * U2, A * U3, conj.(A) * U4)
    end
    rank_cotangents = ntuple(npol) do polarization
        (@view cotangent[:, :, polarization]) * conj.(WU1)
    end

    if npol == 1
        _mmgnlse_cp_spatial_term_vjp!(
            out, projections, rank_cotangents, cp, 1, 1, 1, 1, one(T))
        return out
    end

    c23 = T(2) / T(3)
    c13 = T(1) / T(3)
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 1, 1, 1, 1, one(T))
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 1, 1, 2, 2, c23)
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 1, 2, 2, 1, c13)
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 2, 2, 2, 2, one(T))
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 2, 2, 1, 1, c23)
    _mmgnlse_cp_spatial_term_vjp!(out, projections, rank_cotangents,
                                   cp, 2, 1, 1, 2, c13)
    return out
end

function _mmgnlse_vjp_instantaneous(field, cotangent,
                                     cp::MMGNLSECPDecomposition)
    if cp.metadata.layout == :spatial
        return _mmgnlse_cp_spatial_instantaneous_vjp(field, cotangent, cp)
    elseif cp.metadata.layout == :component
        nt, nm, npol = size(field)
        (nm == cp.metadata.nmodes &&
         npol == cp.metadata.npolarizations) || throw(DimensionMismatch(
            "Field shape does not match component CP metadata."))
        flat_field = reshape(field, nt, nm * npol)
        flat_cotangent = reshape(cotangent, nt, nm * npol)
        return reshape(_mmgnlse_cp_generic_instantaneous_vjp(
            flat_field, flat_cotangent, cp), nt, nm, npol)
    end
    throw(ArgumentError("Unsupported CP overlap layout $(cp.metadata.layout)."))
end


# -- Raman contractions -----------------------------------------------------

function _mmgnlse_filter_raman_adjoint(channels, kernel)
    isempty(kernel) && return zeros(ComplexF64, size(channels))
    all(iszero, kernel) && return zeros(ComplexF64, size(channels))
    shaped_kernel = reshape(conj.(kernel), :,
                            ntuple(_ -> 1, ndims(channels) - 1)...)
    return fft(shaped_kernel .* ifft(channels, 1), 1)
end

function _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S,
                                           kernel, output_pol::Int,
                                           p2::Int, p3::Int, p4::Int,
                                           coefficient)
    all(iszero, kernel) && return out
    nt, nm, _ = size(field)
    T = promote_type(eltype(out), eltype(kernel), eltype(S), ComplexF64)

    pair = Array{T}(undef, nt, nm, nm)
    @inbounds for t in 1:nt, i3 in 1:nm, i4 in 1:nm
        pair[t, i3, i4] = field[t, i3, p3] * conj(field[t, i4, p4])
    end
    filtered_pair = _mmgnlse_filter_raman(pair, kernel)
    filtered_cotangent = zeros(T, nt, nm, nm)

    @inbounds for t in 1:nt, i1 in 1:nm, i2 in 1:nm, i3 in 1:nm, i4 in 1:nm
        s = coefficient * S[i1, i2, i3, i4]
        projected = cotangent[t, i1, output_pol] * conj(s)
        out[t, i2, p2] += projected * conj(filtered_pair[t, i3, i4])
        filtered_cotangent[t, i3, i4] += projected * conj(field[t, i2, p2])
    end

    pair_cotangent = _mmgnlse_filter_raman_adjoint(filtered_cotangent, kernel)
    @inbounds for t in 1:nt, i3 in 1:nm, i4 in 1:nm
        v = pair_cotangent[t, i3, i4]
        out[t, i3, p3] += v * field[t, i4, p4]
        out[t, i4, p4] += conj(v) * field[t, i3, p3]
    end
    return out
end

function _mmgnlse_dense_component_delayed_vjp(field::AbstractMatrix,
                                                cotangent::AbstractMatrix,
                                                S, kernel)
    nt, nc = size(field)
    size(cotangent) == size(field) || throw(DimensionMismatch(
        "The flattened component field and cotangent must have matching shapes."))
    size(S) == (nc, nc, nc, nc) || throw(DimensionMismatch(
        "The component tensor and flattened field dimensions do not match."))
    T = promote_type(eltype(field), eltype(cotangent), eltype(S),
                     eltype(kernel), ComplexF64)
    out = zeros(T, nt, nc)
    all(iszero, kernel) && return out

    pair = Array{T}(undef, nt, nc, nc)
    @inbounds for t in 1:nt, i3 in 1:nc, i4 in 1:nc
        pair[t, i3, i4] = field[t, i3] * conj(field[t, i4])
    end
    filtered_pair = _mmgnlse_filter_raman(pair, kernel)
    filtered_cotangent = zeros(T, nt, nc, nc)

    @inbounds for t in 1:nt, i1 in 1:nc, i2 in 1:nc, i3 in 1:nc, i4 in 1:nc
        s = S[i1, i2, i3, i4]
        projected = cotangent[t, i1] * conj(s)
        out[t, i2] += projected * conj(filtered_pair[t, i3, i4])
        filtered_cotangent[t, i3, i4] += projected * conj(field[t, i2])
    end

    pair_cotangent = _mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    @inbounds for t in 1:nt, i3 in 1:nc, i4 in 1:nc
        value = pair_cotangent[t, i3, i4]
        out[t, i3] += value * field[t, i4]
        out[t, i4] += conj(value) * field[t, i3]
    end
    return out
end

function _mmgnlse_vjp_delayed(field, cotangent, overlap::SpatialOverlap,
                               kernels)
    nt, nm, npol = size(field)
    T = promote_type(eltype(field), eltype(cotangent),
                     eltype(overlap.values), ComplexF64)
    out = zeros(T, nt, nm, npol)
    S = overlap.values
    if npol == 1
        _mmgnlse_dense_delayed_term_vjp!(
            out, field, cotangent, S, kernels.ha .+ kernels.hb,
            1, 1, 1, 1, one(T))
        return out
    end

    half = T(1) / T(2)
    # h_a channel.
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.ha,
                                      1, 1, 1, 1, one(T))
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.ha,
                                      1, 1, 2, 2, one(T))
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.ha,
                                      2, 2, 1, 1, one(T))
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.ha,
                                      2, 2, 2, 2, one(T))
    # h_b anisotropic channel.
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      1, 1, 1, 1, one(T))
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      1, 2, 1, 2, half)
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      1, 2, 2, 1, half)
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      2, 2, 2, 2, one(T))
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      2, 1, 1, 2, half)
    _mmgnlse_dense_delayed_term_vjp!(out, field, cotangent, S, kernels.hb,
                                      2, 1, 2, 1, half)
    return out
end

function _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents,
                                        cp::MMGNLSECPDecomposition, kernel,
                                        output_pol::Int, p2::Int,
                                        p3::Int, p4::Int, coefficient)
    all(iszero, kernel) && return out
    _, U2, U3, U4 = cp.U
    B2 = projections[p2][1]
    B3 = projections[p3][2]
    B4 = projections[p4][3]
    rank_cotangent = conj(coefficient) .* rank_cotangents[output_pol]

    rank_pair = B3 .* B4
    filtered_pair = _mmgnlse_filter_raman(rank_pair, kernel)
    vB2 = rank_cotangent .* conj.(filtered_pair)
    filtered_cotangent = rank_cotangent .* conj.(B2)
    pair_cotangent = _mmgnlse_filter_raman_adjoint(filtered_cotangent, kernel)
    vB3 = pair_cotangent .* conj.(B4)
    vB4 = pair_cotangent .* conj.(B3)

    @views out[:, :, p2] .+= vB2 * adjoint(U2)
    @views out[:, :, p3] .+= vB3 * adjoint(U3)
    @views out[:, :, p4] .+= conj.(vB4 * adjoint(U4))
    return out
end

function _mmgnlse_cp_component_delayed_vjp(field::AbstractMatrix,
                                             cotangent::AbstractMatrix,
                                             cp::MMGNLSECPDecomposition,
                                             kernel)
    nt, nc = size(field)
    size(cotangent) == size(field) || throw(DimensionMismatch(
        "The flattened component field and cotangent must have matching shapes."))
    cp.metadata.layout == :component || throw(ArgumentError(
        "This contraction requires a component CP decomposition."))
    cp.metadata.source_size == (nc, nc, nc, nc) || throw(DimensionMismatch(
        "Field and component CP dimensions do not match."))
    T = promote_type(eltype(field), eltype(cotangent), eltype(cp.λ),
                     map(eltype, cp.U)..., eltype(kernel), ComplexF64)
    (cp_iszero(cp) || all(iszero, kernel)) && return zeros(T, nt, nc)

    U1, U2, U3, U4 = cp.U
    B2 = field * U2
    B3 = field * U3
    B4 = conj.(field) * U4
    WU1 = U1 .* reshape(cp.λ, 1, :)
    rank_cotangent = cotangent * conj.(WU1)

    filtered_pair = _mmgnlse_filter_raman(B3 .* B4, kernel)
    vB2 = rank_cotangent .* conj.(filtered_pair)
    filtered_cotangent = rank_cotangent .* conj.(B2)
    pair_cotangent = _mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    vB3 = pair_cotangent .* conj.(B4)
    vB4 = pair_cotangent .* conj.(B3)
    return vB2 * adjoint(U2) .+
           vB3 * adjoint(U3) .+
           conj.(vB4 * adjoint(U4))
end

function _mmgnlse_vjp_delayed(field, cotangent,
                               cp::MMGNLSECPDecomposition, kernels)
    if cp.metadata.layout == :component
        nt, nm, npol = size(field)
        (nm == cp.metadata.nmodes &&
         npol == cp.metadata.npolarizations) || throw(DimensionMismatch(
            "Field shape does not match component CP metadata."))
        flat_field = flatten_field_components(field)
        flat_cotangent = flatten_field_components(cotangent)
        flat_out = _mmgnlse_cp_component_delayed_vjp(
            flat_field, flat_cotangent, cp, kernels.ha .+ kernels.hb)
        return reshape(flat_out, nt, nm, npol)
    elseif cp.metadata.layout != :spatial
        throw(ArgumentError("Unsupported CP overlap layout $(cp.metadata.layout)."))
    end
    nt, nm, npol = size(field)
    _validate_spatial_cp(cp, nm)
    T = promote_type(eltype(field), eltype(cotangent), eltype(cp.λ),
                     map(eltype, cp.U)..., ComplexF64)
    out = zeros(T, nt, nm, npol)
    cp_iszero(cp) && return out

    U1, U2, U3, U4 = cp.U
    WU1 = U1 .* reshape(cp.λ, 1, :)
    projections = ntuple(npol) do polarization
        A = @view field[:, :, polarization]
        (A * U2, A * U3, conj.(A) * U4)
    end
    rank_cotangents = ntuple(npol) do polarization
        (@view cotangent[:, :, polarization]) * conj.(WU1)
    end

    if npol == 1
        _mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, cp,
            kernels.ha .+ kernels.hb, 1, 1, 1, 1, one(T))
        return out
    end

    half = T(1) / T(2)
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.ha, 1, 1, 1, 1, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.ha, 1, 1, 2, 2, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.ha, 2, 2, 1, 1, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.ha, 2, 2, 2, 2, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 1, 1, 1, 1, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 1, 2, 1, 2, half)
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 1, 2, 2, 1, half)
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 2, 2, 2, 2, one(T))
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 2, 1, 1, 2, half)
    _mmgnlse_cp_delayed_term_vjp!(out, projections, rank_cotangents, cp,
                                   kernels.hb, 2, 1, 2, 1, half)
    return out
end

function _mmgnlse_vjp_delayed(field, cotangent,
                               overlap::ComponentOverlap, kernels)
    nt, nm, npol = size(field)
    flat_field = flatten_field_components(field)
    flat_cotangent = flatten_field_components(cotangent)
    flat_tensor = flatten_component_tensor(overlap.values)
    flat_out = _mmgnlse_dense_component_delayed_vjp(
        flat_field, flat_cotangent, flat_tensor, kernels.ha .+ kernels.hb)
    return reshape(flat_out, nt, nm, npol)
end
