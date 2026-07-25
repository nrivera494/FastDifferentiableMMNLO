# Device-resident analytic VJP and interaction-picture adjoint for the public
# MMGNLSE interface. Host/device transfers are confined to the supplied
# trajectory/terminal state and the returned saved trajectory.

# -- Dense instantaneous VJPs -----------------------------------------------

function _cuda_mmgnlse_dense_spatial_term_vjp_kernel!(
    out, field, cotangent, S, output_pol::Int, p2::Int, p3::Int,
    p4::Int, coefficient, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator2 = zero(eltype(out))
        accumulator3 = zero(eltype(out))
        accumulator4 = zero(eltype(out))

        @inbounds for i4 in 1:nm, i3 in 1:nm, i1 in 1:nm
            s = coefficient * S[i1, target, i3, i4]
            accumulator2 += cotangent[t, i1, output_pol] * conj(s) *
                            conj(field[t, i3, p3]) * field[t, i4, p4]
        end
        @inbounds for i4 in 1:nm, i2 in 1:nm, i1 in 1:nm
            s = coefficient * S[i1, i2, target, i4]
            accumulator3 += cotangent[t, i1, output_pol] * conj(s) *
                            conj(field[t, i2, p2]) * field[t, i4, p4]
        end
        @inbounds for i3 in 1:nm, i2 in 1:nm, i1 in 1:nm
            s = coefficient * S[i1, i2, i3, target]
            accumulator4 += conj(cotangent[t, i1, output_pol]) * s *
                            field[t, i2, p2] * field[t, i3, p3]
        end

        @inbounds out[t, target, p2] += accumulator2
        @inbounds out[t, target, p3] += accumulator3
        @inbounds out[t, target, p4] += accumulator4
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_vjp_kernel!(
    out, field, cotangent, S, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))

        @inbounds for i4 in 1:nc, i3 in 1:nc, i1 in 1:nc
            s = S[i1, target, i3, i4]
            accumulator += cotangent[t, i1] * conj(s) *
                           conj(field[t, i3]) * field[t, i4]
        end
        @inbounds for i4 in 1:nc, i2 in 1:nc, i1 in 1:nc
            s = S[i1, i2, target, i4]
            accumulator += cotangent[t, i1] * conj(s) *
                           conj(field[t, i2]) * field[t, i4]
        end
        @inbounds for i3 in 1:nc, i2 in 1:nc, i1 in 1:nc
            s = S[i1, i2, i3, target]
            accumulator += conj(cotangent[t, i1]) * s *
                           field[t, i2] * field[t, i3]
        end
        @inbounds out[t, target] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_term_vjp!(
    out, field, cotangent, overlap::CUDAMMGNLSEDenseSpatial,
    output_pol::Int, p2::Int, p3::Int, p4::Int, coefficient,
)
    nt, nm, _ = size(field)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_term_vjp_kernel!, nt * nm,
        out, field, cotangent, overlap.values, output_pol, p2, p3, p4,
        ComplexF64(coefficient), nt, nm)
    return out
end

function _cuda_mmgnlse_instantaneous_vjp(
    field, cotangent, overlap::CUDAMMGNLSEDenseSpatial,
)
    nt, nm, np = size(field)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        return _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field, cotangent, overlap, 1, 1, 1, 1, 1.0)
    end
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 1, 1, 1, 1, 1.0)
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 1, 1, 2, 2, 2 / 3)
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 1, 2, 2, 1, 1 / 3)
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 2, 2, 2, 2, 1.0)
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 2, 2, 1, 1, 2 / 3)
    _cuda_mmgnlse_dense_spatial_term_vjp!(
        out, field, cotangent, overlap, 2, 1, 1, 2, 1 / 3)
    return out
end

function _cuda_mmgnlse_instantaneous_vjp(
    field, cotangent, overlap::CUDAMMGNLSEDenseComponent,
)
    nt, nm, np = size(field)
    nc = nm * np
    flat_field = reshape(field, nt, nc)
    flat_cotangent = reshape(cotangent, nt, nc)
    flat_out = CUDA.zeros(ComplexF64, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_component_vjp_kernel!, length(flat_out),
        flat_out, flat_field, flat_cotangent, overlap.values, nt, nc)
    return reshape(flat_out, nt, nm, np)
end


# -- CP instantaneous VJPs --------------------------------------------------

function _cuda_mmgnlse_cp_generic_instantaneous_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP,
)
    _, U2, U3, U4 = overlap.U
    B2, B3, B4 = _cuda_mmgnlse_cp_projections(field, overlap)
    rank_cotangent = cotangent * overlap.cwu1
    vB2 = rank_cotangent .* conj.(B3 .* B4)
    vB3 = rank_cotangent .* conj.(B2 .* B4)
    vB4 = rank_cotangent .* conj.(B2 .* B3)
    return vB2 * adjoint(U2) .+
           vB3 * adjoint(U3) .+
           conj.(vB4 * adjoint(U4))
end

function _cuda_mmgnlse_cp_spatial_term_vjp!(
    out, projections, rank_cotangents, overlap::CUDAMMGNLSECP,
    output_pol::Int, p2::Int, p3::Int, p4::Int, coefficient,
)
    _, U2, U3, U4 = overlap.U
    B2 = projections[p2][1]
    B3 = projections[p3][2]
    B4 = projections[p4][3]
    rank_cotangent =
        conj(ComplexF64(coefficient)) .* rank_cotangents[output_pol]
    vB2 = rank_cotangent .* conj.(B3 .* B4)
    vB3 = rank_cotangent .* conj.(B2 .* B4)
    vB4 = rank_cotangent .* conj.(B2 .* B3)
    @views out[:, :, p2] .+= vB2 * adjoint(U2)
    @views out[:, :, p3] .+= vB3 * adjoint(U3)
    @views out[:, :, p4] .+= conj.(vB4 * adjoint(U4))
    return out
end

function _cuda_mmgnlse_cp_spatial_instantaneous_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP,
)
    nt, nm, np = size(field)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    projections = ntuple(np) do polarization
        component = copy(@view field[:, :, polarization])
        _cuda_mmgnlse_cp_projections(component, overlap)
    end
    rank_cotangents = ntuple(np) do polarization
        copy(@view cotangent[:, :, polarization]) * overlap.cwu1
    end
    if np == 1
        return _cuda_mmgnlse_cp_spatial_term_vjp!(
            out, projections, rank_cotangents, overlap,
            1, 1, 1, 1, 1.0)
    end
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 1, 1, 1, 1, 1.0)
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 1, 1, 2, 2, 2 / 3)
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 1, 2, 2, 1, 1 / 3)
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 2, 2, 2, 2, 1.0)
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 2, 2, 1, 1, 2 / 3)
    _cuda_mmgnlse_cp_spatial_term_vjp!(
        out, projections, rank_cotangents, overlap, 2, 1, 1, 2, 1 / 3)
    return out
end

function _cuda_mmgnlse_instantaneous_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP,
)
    if overlap.layout == :spatial
        return _cuda_mmgnlse_cp_spatial_instantaneous_vjp(
            field, cotangent, overlap)
    elseif overlap.layout == :component
        nt, nm, np = size(field)
        flat_field = reshape(field, nt, nm * np)
        flat_cotangent = reshape(cotangent, nt, nm * np)
        return reshape(
            _cuda_mmgnlse_cp_generic_instantaneous_vjp(
                flat_field, flat_cotangent, overlap),
            nt, nm, np)
    end
    throw(ArgumentError("Unsupported CP layout $(overlap.layout)."))
end


# -- Dense delayed VJPs -----------------------------------------------------

function _cuda_mmgnlse_dense_spatial_delayed_direct_kernel!(
    out, field, cotangent, filtered_pair, S, output_pol::Int, p2::Int,
    coefficient, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i1 in 1:nm
            s = coefficient * S[i1, target, i3, i4]
            projected = cotangent[t, i1, output_pol] * conj(s)
            accumulator += projected * conj(filtered_pair[t, i3, i4])
        end
        @inbounds out[t, target, p2] += accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_filtered_cotangent_kernel!(
    filtered_cotangent, field, cotangent, S, output_pol::Int, p2::Int,
    coefficient, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i3 = remainder % nm + 1
        i4 = remainder ÷ nm + 1
        accumulator = zero(eltype(filtered_cotangent))
        @inbounds for i2 in 1:nm, i1 in 1:nm
            s = coefficient * S[i1, i2, i3, i4]
            projected = cotangent[t, i1, output_pol] * conj(s)
            accumulator += projected * conj(field[t, i2, p2])
        end
        @inbounds filtered_cotangent[t, i3, i4] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_spatial_pair_vjp_kernel!(
    out, field, pair_cotangent, p3::Int, p4::Int,
    nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator3 = zero(eltype(out))
        accumulator4 = zero(eltype(out))
        @inbounds for i4 in 1:nm
            accumulator3 += pair_cotangent[t, target, i4] *
                            field[t, i4, p4]
        end
        @inbounds for i3 in 1:nm
            accumulator4 += conj(pair_cotangent[t, i3, target]) *
                            field[t, i3, p3]
        end
        @inbounds out[t, target, p3] += accumulator3
        @inbounds out[t, target, p4] += accumulator4
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_delayed_direct_kernel!(
    out, field, cotangent, filtered_pair, S, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nc, i3 in 1:nc, i1 in 1:nc
            s = S[i1, target, i3, i4]
            projected = cotangent[t, i1] * conj(s)
            accumulator += projected * conj(filtered_pair[t, i3, i4])
        end
        @inbounds out[t, target] += accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_filtered_cotangent_kernel!(
    filtered_cotangent, field, cotangent, S,
    nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i3 = remainder % nc + 1
        i4 = remainder ÷ nc + 1
        accumulator = zero(eltype(filtered_cotangent))
        @inbounds for i2 in 1:nc, i1 in 1:nc
            s = S[i1, i2, i3, i4]
            projected = cotangent[t, i1] * conj(s)
            accumulator += projected * conj(field[t, i2])
        end
        @inbounds filtered_cotangent[t, i3, i4] = accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_component_pair_vjp_kernel!(
    out, field, pair_cotangent, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i4 in 1:nc
            accumulator += pair_cotangent[t, target, i4] * field[t, i4]
        end
        @inbounds for i3 in 1:nc
            accumulator += conj(pair_cotangent[t, i3, target]) *
                           field[t, i3]
        end
        @inbounds out[t, target] += accumulator
    end
    return nothing
end

function _cuda_mmgnlse_filter_raman_adjoint(channels, kernel)
    dimensions = (length(kernel),
                  ntuple(_ -> 1, ndims(channels) - 1)...)
    return fft(reshape(conj.(kernel), dimensions) .* ifft(channels, 1), 1)
end

function _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
    out, field, cotangent, overlap::CUDAMMGNLSEDenseSpatial, kernel,
    output_pol::Int, p2::Int, p3::Int, p4::Int, coefficient,
)
    nt, nm, _ = size(field)
    component3 = copy(@view field[:, :, p3])
    component4 = copy(@view field[:, :, p4])
    pair = reshape(component3, nt, nm, 1) .*
           conj.(reshape(component4, nt, 1, nm))
    filtered_pair = _cuda_mmgnlse_filter_raman(pair, kernel)
    filtered_cotangent = CUDA.zeros(ComplexF64, nt, nm, nm)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_delayed_direct_kernel!, nt * nm,
        out, field, cotangent, filtered_pair, overlap.values, output_pol,
        p2, ComplexF64(coefficient), nt, nm)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_filtered_cotangent_kernel!,
        length(filtered_cotangent), filtered_cotangent, field, cotangent,
        overlap.values, output_pol, p2, ComplexF64(coefficient), nt, nm)
    pair_cotangent = _cuda_mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_pair_vjp_kernel!, nt * nm,
        out, field, pair_cotangent, p3, p4, nt, nm)
    return out
end

function _cuda_mmgnlse_dense_component_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSEDenseComponent, kernel,
)
    nt, nm, np = size(field)
    nc = nm * np
    flat_field = reshape(field, nt, nc)
    flat_cotangent = reshape(cotangent, nt, nc)
    pair = reshape(flat_field, nt, nc, 1) .*
           conj.(reshape(flat_field, nt, 1, nc))
    filtered_pair = _cuda_mmgnlse_filter_raman(pair, kernel)
    filtered_cotangent = CUDA.zeros(ComplexF64, nt, nc, nc)
    flat_out = CUDA.zeros(ComplexF64, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_component_delayed_direct_kernel!,
        length(flat_out), flat_out, flat_field, flat_cotangent,
        filtered_pair, overlap.values, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_component_filtered_cotangent_kernel!,
        length(filtered_cotangent), filtered_cotangent, flat_field,
        flat_cotangent, overlap.values, nt, nc)
    pair_cotangent = _cuda_mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    _launch_1d(
        _cuda_mmgnlse_dense_component_pair_vjp_kernel!, length(flat_out),
        flat_out, flat_field, pair_cotangent, nt, nc)
    return reshape(flat_out, nt, nm, np)
end


# -- CP delayed VJPs --------------------------------------------------------

function _cuda_mmgnlse_cp_delayed_term_vjp!(
    out, projections, rank_cotangents, overlap::CUDAMMGNLSECP, kernel,
    output_pol::Int, p2::Int, p3::Int, p4::Int, coefficient,
)
    _, U2, U3, U4 = overlap.U
    B2 = projections[p2][1]
    B3 = projections[p3][2]
    B4 = projections[p4][3]
    rank_cotangent =
        conj(ComplexF64(coefficient)) .* rank_cotangents[output_pol]
    filtered_pair = _cuda_mmgnlse_filter_raman(B3 .* B4, kernel)
    vB2 = rank_cotangent .* conj.(filtered_pair)
    filtered_cotangent = rank_cotangent .* conj.(B2)
    pair_cotangent = _cuda_mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    vB3 = pair_cotangent .* conj.(B4)
    vB4 = pair_cotangent .* conj.(B3)
    @views out[:, :, p2] .+= vB2 * adjoint(U2)
    @views out[:, :, p3] .+= vB3 * adjoint(U3)
    @views out[:, :, p4] .+= conj.(vB4 * adjoint(U4))
    return out
end

function _cuda_mmgnlse_cp_component_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP, kernel,
)
    _, U2, U3, U4 = overlap.U
    B2, B3, B4 = _cuda_mmgnlse_cp_projections(field, overlap)
    rank_cotangent = cotangent * overlap.cwu1
    filtered_pair = _cuda_mmgnlse_filter_raman(B3 .* B4, kernel)
    vB2 = rank_cotangent .* conj.(filtered_pair)
    filtered_cotangent = rank_cotangent .* conj.(B2)
    pair_cotangent = _cuda_mmgnlse_filter_raman_adjoint(
        filtered_cotangent, kernel)
    vB3 = pair_cotangent .* conj.(B4)
    vB4 = pair_cotangent .* conj.(B3)
    return vB2 * adjoint(U2) .+
           vB3 * adjoint(U3) .+
           conj.(vB4 * adjoint(U4))
end

function _cuda_mmgnlse_cp_spatial_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP, raman,
)
    nt, nm, np = size(field)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    projections = ntuple(np) do polarization
        component = copy(@view field[:, :, polarization])
        _cuda_mmgnlse_cp_projections(component, overlap)
    end
    rank_cotangents = ntuple(np) do polarization
        copy(@view cotangent[:, :, polarization]) * overlap.cwu1
    end
    if np == 1
        return _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.ha .+ raman.hb, 1, 1, 1, 1, 1.0)
    end
    if raman.ha_active
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.ha, 1, 1, 1, 1, 1.0)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.ha, 1, 1, 2, 2, 1.0)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.ha, 2, 2, 1, 1, 1.0)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.ha, 2, 2, 2, 2, 1.0)
    end
    if raman.hb_active
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 1, 1, 1, 1, 1.0)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 1, 2, 1, 2, 0.5)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 1, 2, 2, 1, 0.5)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 2, 2, 2, 2, 1.0)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 2, 1, 1, 2, 0.5)
        _cuda_mmgnlse_cp_delayed_term_vjp!(
            out, projections, rank_cotangents, overlap,
            raman.hb, 2, 1, 2, 1, 0.5)
    end
    return out
end


# -- Delayed dispatch and nonlinear VJP ------------------------------------

function _cuda_mmgnlse_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSEDenseSpatial, raman,
)
    nt, nm, np = size(field)
    out = CUDA.zeros(ComplexF64, nt, nm, np)
    if np == 1
        return _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.ha .+ raman.hb,
            1, 1, 1, 1, 1.0)
    end
    if raman.ha_active
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.ha, 1, 1, 1, 1, 1.0)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.ha, 1, 1, 2, 2, 1.0)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.ha, 2, 2, 1, 1, 1.0)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.ha, 2, 2, 2, 2, 1.0)
    end
    if raman.hb_active
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 1, 1, 1, 1, 1.0)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 1, 2, 1, 2, 0.5)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 1, 2, 2, 1, 0.5)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 2, 2, 2, 2, 1.0)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 2, 1, 1, 2, 0.5)
        _cuda_mmgnlse_dense_spatial_delayed_term_vjp!(
            out, field, cotangent, overlap, raman.hb, 2, 1, 2, 1, 0.5)
    end
    return out
end

function _cuda_mmgnlse_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSEDenseComponent, raman,
)
    return _cuda_mmgnlse_dense_component_delayed_vjp(
        field, cotangent, overlap, raman.ha .+ raman.hb)
end

function _cuda_mmgnlse_delayed_vjp(
    field, cotangent, overlap::CUDAMMGNLSECP, raman,
)
    if overlap.layout == :spatial
        return _cuda_mmgnlse_cp_spatial_delayed_vjp(
            field, cotangent, overlap, raman)
    elseif overlap.layout == :component
        nt, nm, np = size(field)
        flat_field = reshape(field, nt, nm * np)
        flat_cotangent = reshape(cotangent, nt, nm * np)
        return reshape(
            _cuda_mmgnlse_cp_component_delayed_vjp(
                flat_field, flat_cotangent, overlap,
                raman.ha .+ raman.hb),
            nt, nm, np)
    end
    throw(ArgumentError("Unsupported CP layout $(overlap.layout)."))
end

function _cuda_mmgnlse_vjp_nonlinear(
    field_t, cotangent_t, cache::CUDAMMGNLSEForwardCache,
)
    !cache.nonlinear_active &&
        return CUDA.zeros(ComplexF64, size(field_t)...)
    cubic_cotangent = fft(
        conj.(cache.nonlinear_prefactor) .* ifft(cotangent_t, 1), 1)
    fraction = cache.raman.fraction
    result = CUDA.zeros(ComplexF64, size(field_t)...)
    if fraction != 1
        result .+= _cuda_mmgnlse_instantaneous_vjp(
            field_t, (1 - fraction) .* cubic_cotangent, cache.overlap)
    end
    if !iszero(fraction)
        result .+= _cuda_mmgnlse_delayed_vjp(
            field_t, fraction .* cubic_cotangent,
            cache.overlap, cache.raman)
    end
    return result
end


# -- Interaction-picture trajectory helpers --------------------------------

function _cuda_mmgnlse_to_host_solution(device_solution)
    device_cache = device_solution.cache
    host_cache = (
        backend=:cuda,
        integration_z=device_cache.integration_z,
        core=device_cache.core,
        device=string(CUDA.device()),
    )
    return PulsePropagation.MMGNLSESolution(
        z=copy(device_solution.z),
        fields=Array(device_solution.fields),
        parameters=device_solution.parameters,
        initial_field=Array(device_solution.initial_field),
        dz=device_solution.dz,
        method=device_solution.method,
        cache=host_cache,
    )
end

function _cuda_mmgnlse_device_solution(solution)
    return PulsePropagation.MMGNLSESolution(
        z=copy(solution.z),
        fields=CUDA.CuArray(ComplexF64.(solution.fields)),
        parameters=solution.parameters,
        initial_field=CUDA.CuArray(ComplexF64.(solution.initial_field)),
        dz=Float64(solution.dz),
        method=solution.method,
        cache=solution.cache,
    )
end

function _cuda_mmgnlse_prepare_forward(
    parameters; forward_solution, initial_field, dz_forward,
)
    if forward_solution !== nothing
        forward_solution isa PulsePropagation.MMGNLSESolution ||
            throw(ArgumentError(
                "forward_solution must be an MMGNLSESolution."))
        initial_field === nothing || throw(ArgumentError(
            "Supply exactly one of forward_solution or initial_field."))
        dz_forward === nothing || throw(ArgumentError(
            "dz_forward is only valid when initial_field is supplied."))
        forward_solution.parameters === parameters || throw(ArgumentError(
            "forward_solution was constructed with a different " *
            "MMGNLSEParameters object."))
        if PulsePropagation._mmgnlse_forward_is_step_saved(forward_solution)
            return _cuda_mmgnlse_device_solution(forward_solution),
                   forward_solution
        end
        replay = _cuda_mmgnlse_solve_device(
            CUDA.CuArray(ComplexF64.(forward_solution.initial_field)),
            parameters, Float64(forward_solution.dz);
            method=forward_solution.method, saveat=:steps)
        return replay, _cuda_mmgnlse_to_host_solution(replay)
    end

    initial_field === nothing && throw(ArgumentError(
        "Supply exactly one of forward_solution or initial_field."))
    dz_forward === nothing && throw(ArgumentError(
        "dz_forward is required when initial_field is supplied."))
    isfinite(dz_forward) && dz_forward > 0 || throw(ArgumentError(
        "dz_forward must be finite and positive."))
    validated = PulsePropagation._mmgnlse_validate_initial_field(
        initial_field, parameters)
    device_forward = _cuda_mmgnlse_solve_device(
        CUDA.CuArray(validated), parameters, Float64(dz_forward);
        saveat=:steps)
    return device_forward, _cuda_mmgnlse_to_host_solution(device_forward)
end

function _cuda_mmgnlse_forward_interaction_fields(
    solution, cache::CUDAMMGNLSEForwardCache,
)
    interaction = ifft(solution.fields, 1)
    for index in eachindex(solution.z)
        propagator = _cuda_mmgnlse_linear_propagator(
            cache, 0.0, solution.z[index])
        @views interaction[:, :, :, index] ./= propagator
    end
    return interaction
end

function _cuda_mmgnlse_interpolate_interaction_field(
    fields, zgrid::AbstractVector, z::Real,
)
    tolerance = 64eps(float(eltype(zgrid))) *
                max(one(float(eltype(zgrid))),
                    abs(first(zgrid)), abs(last(zgrid)))
    z < first(zgrid) - tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z before the forward trajectory."))
    z > last(zgrid) + tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z after the forward trajectory."))
    z <= first(zgrid) && return @view fields[:, :, :, 1]
    z >= last(zgrid) && return @view fields[:, :, :, end]
    right = searchsortedfirst(zgrid, z)
    zgrid[right] == z && return @view fields[:, :, :, right]
    left = right - 1
    weight = (z - zgrid[left]) / (zgrid[right] - zgrid[left])
    return (1 - weight) .* @view(fields[:, :, :, left]) .+
           weight .* @view(fields[:, :, :, right])
end

function _cuda_mmgnlse_convert_adjoint_units(
    field_t, units::Symbol, photon_scale_raw,
)
    units === :power && return field_t
    scale = reshape(photon_scale_raw, :, 1, 1)
    return fft(scale .* ifft(field_t, 1), 1)
end


# -- Public CUDA adjoint backend -------------------------------------------

function PulsePropagation._mmgnlse_solve_adjoint_cuda(
    lambda_terminal::AbstractArray{<:Number,3},
    parameters::PulsePropagation.MMGNLSEParameters,
    dz_adj::Real;
    forward_solution=nothing,
    initial_field=nothing,
    dz_forward=nothing,
    method=PulsePropagation.Vern9(),
    units::Symbol=:power,
    saveat=nothing,
    device=nothing,
    synchronize::Bool=true,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    (forward_solution === nothing) ⊻ (initial_field === nothing) ||
        throw(ArgumentError(
            "Supply exactly one of forward_solution or initial_field."))
    isfinite(dz_adj) && dz_adj > 0 || throw(ArgumentError(
        "dz_adj must be finite and positive."))
    PulsePropagation._mmgnlse_validate_adjoint_method(method)
    units in (:power, :photon) || throw(ArgumentError(
        "units must be :power or :photon."))

    device_forward, host_forward = _cuda_mmgnlse_prepare_forward(
        parameters; forward_solution, initial_field, dz_forward)
    first(device_forward.z) == 0 || throw(ArgumentError(
        "The forward trajectory must begin at z=0."))
    last(device_forward.z) == parameters.length || throw(ArgumentError(
        "The forward trajectory must end at parameters.length."))
    all(diff(device_forward.z) .> 0) || throw(ArgumentError(
        "The forward trajectory z values must be strictly increasing."))

    lambda_w = PulsePropagation._mmgnlse_validate_adjoint_terminal(
        lambda_terminal, parameters)
    lambda_raw = CUDA.CuArray(ifftshift(lambda_w, 1))
    T = promote_type(eltype(device_forward.z), typeof(float(dz_adj)))
    targets, every_step = PulsePropagation._mmgnlse_save_targets(
        T(parameters.length), saveat)
    integration_z, saved_z = PulsePropagation._mmgnlse_step_grid(
        T(parameters.length), T(dz_adj), targets, every_step)
    descending_saved = reverse(saved_z)
    descending_stops = reverse(integration_z[1:end-1])

    cache = _cuda_mmgnlse_forward_cache(parameters)
    forward_interaction = _cuda_mmgnlse_forward_interaction_fields(
        device_forward, cache)
    terminal_propagator = _cuda_mmgnlse_linear_propagator(
        cache, 0.0, T(parameters.length))
    lambda_interaction = conj.(terminal_propagator) .* lambda_raw

    function rhs!(derivative, lambda_tilde, _, z)
        if !cache.nonlinear_active
            fill!(derivative, zero(eltype(derivative)))
            return nothing
        end
        coordinate = T(z)
        forward_tilde = _cuda_mmgnlse_interpolate_interaction_field(
            forward_interaction, device_forward.z, coordinate)
        propagator = _cuda_mmgnlse_linear_propagator(
            cache, 0.0, coordinate)
        field_t = fft(propagator .* forward_tilde, 1)
        lambda_raw_z = lambda_tilde ./ conj.(propagator)
        lambda_t = fft(lambda_raw_z, 1)
        nonlinear_vjp_t = _cuda_mmgnlse_vjp_nonlinear(
            field_t, lambda_t, cache)
        derivative .=
            -conj.(propagator) .* ifft(nonlinear_vjp_t, 1)
        return nothing
    end

    problem = PulsePropagation.ODEProblem(
        rhs!, lambda_interaction, (T(parameters.length), zero(T)))
    ode_solution = PulsePropagation.DifferentialEquations.solve(
        problem, method; dt=T(dz_adj), adaptive=false,
        saveat=descending_saved, tstops=descending_stops,
        save_everystep=false, dense=false,
        internalnorm=_cuda_mmgnlse_fixed_internalnorm)
    ode_solution.retcode == PulsePropagation.ReturnCode.Success || error(
        "Vern9 adjoint solve failed with retcode $(ode_solution.retcode).")
    length(ode_solution.u) == length(saved_z) || error(
        "Adjoint solver saved $(length(ode_solution.u)) states; " *
        "expected $(length(saved_z)).")

    nt, nm, np = PulsePropagation.parameter_field_shape(parameters)
    device_fields = CUDA.zeros(
        ComplexF64, nt, nm, np, length(saved_z))
    photon_scale_raw = if units === :photon
        centered_weights =
            PulsePropagation._mmgnlse_photon_weights(parameters)
        centered_scale = map(centered_weights) do weight
            weight > 0 ? inv(sqrt(weight)) : 0.0
        end
        CUDA.CuArray(ifftshift(centered_scale, 1))
    else
        CUDA.zeros(Float64, 0)
    end
    for (index, (coordinate, state)) in enumerate(
            zip(saved_z, reverse(ode_solution.u)))
        propagator = _cuda_mmgnlse_linear_propagator(
            cache, 0.0, coordinate)
        lambda_raw_z = state ./ conj.(propagator)
        lambda_t = fft(lambda_raw_z, 1)
        converted = _cuda_mmgnlse_convert_adjoint_units(
            lambda_t, units, photon_scale_raw)
        @views device_fields[:, :, :, index] .= converted
    end
    synchronize && CUDA.synchronize()
    host_cache = (
        backend=:cuda,
        forward_solution=host_forward,
        integration_z=Float64.(integration_z),
        frame=:interaction,
        core=PulsePropagation._mmgnlse_solver_cache(parameters),
        device=string(CUDA.device()),
    )
    return PulsePropagation.MMGNLSEAdjointSolution(
        z=collect(T, saved_z),
        fields=Array(device_fields),
        parameters=parameters,
        units=units,
        cache=host_cache,
    )
end
