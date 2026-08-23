# Device-resident analytic VJP and interaction-picture adjoint for the public
# MMGNLSE interface. Host/device transfers are confined to the supplied
# trajectory/terminal state and the returned saved trajectory.

const _CUDA_MMGNLSE_CP_DUAL_RAMAN_BATCH_LOW_MAX_RANK = 64
const _CUDA_MMGNLSE_CP_DUAL_RAMAN_BATCH_HIGH_MIN_RANK = 256

@inline function _cuda_mmgnlse_cp_use_dual_raman_batch(overlap)
    return _cuda_mmgnlse_cp_is_rank_tuned(overlap) &&
           (overlap.rank <= _CUDA_MMGNLSE_CP_DUAL_RAMAN_BATCH_LOW_MAX_RANK ||
            overlap.rank >= _CUDA_MMGNLSE_CP_DUAL_RAMAN_BATCH_HIGH_MIN_RANK)
end

mutable struct CUDAMMGNLSECPAdjointWorkspace{
    F,A3,A2,P3F,P3I,P2F,P2I,
}
    forward::F
    interpolated::A3
    propagator::A3
    field_t::A3
    lambda_t::A3
    cubic_cotangent::A3
    vjp_t::A3
    spectral_scratch::A3
    rank_cotangent::A3
    raman_channels::A2
    fft_field!::P3F
    fft_lambda!::P3F
    fft_cubic!::P3F
    ifft_cubic!::P3I
    ifft_vjp!::P3I
    fft_raman!::P2F
    ifft_raman!::P2I
end

mutable struct CUDAMMGNLSEDenseAdjointWorkspace{A3,P3F,P3I}
    interpolated::A3
    propagator::A3
    field_t::A3
    lambda_t::A3
    cubic_cotangent::A3
    vjp_t::A3
    spectral_scratch::A3
    fft_field!::P3F
    fft_lambda!::P3F
    fft_cubic!::P3F
    ifft_cubic!::P3I
    ifft_vjp!::P3I
end

mutable struct CUDAMMGNLSEDenseRamanAdjointWorkspace{B,A3,P3F,P3I}
    base::B
    pair::A3
    filtered_cotangent::A3
    pair_cotangent::A3
    fft_pairs!::P3F
    ifft_pairs!::P3I
end

function _cuda_mmgnlse_cp_adjoint_workspace(
    prototype::CUDA.CuArray{ComplexF64,3},
    overlap::CUDAMMGNLSECP,
)
    forward = _cuda_mmgnlse_cp_forward_workspace(prototype, overlap)
    interpolated = similar(prototype)
    propagator = similar(prototype)
    field_t = similar(prototype)
    lambda_t = similar(prototype)
    cubic_cotangent = similar(prototype)
    vjp_t = similar(prototype)
    spectral_scratch = similar(prototype)
    ncomponents = overlap.layout == :spatial ? size(prototype, 3) : 1
    rank_cotangent = CUDA.zeros(
        ComplexF64, size(prototype, 1), overlap.rank, ncomponents)
    raman_channels = if _cuda_mmgnlse_cp_is_rank_tuned(overlap)
        CUDA.zeros(ComplexF64, size(prototype, 1), 2 * overlap.rank)
    else
        CUDA.zeros(ComplexF64, 0, 0)
    end
    fft_field! = plan_fft!(field_t, 1)
    fft_lambda! = plan_fft!(lambda_t, 1)
    fft_cubic! = plan_fft!(cubic_cotangent, 1)
    ifft_cubic! = plan_ifft!(cubic_cotangent, 1)
    ifft_vjp! = plan_ifft!(spectral_scratch, 1)
    fft_raman!, ifft_raman! =
        if _cuda_mmgnlse_cp_is_rank_tuned(overlap)
            plan_fft!(raman_channels, 1), plan_ifft!(raman_channels, 1)
        else
            nothing, nothing
        end
    return CUDAMMGNLSECPAdjointWorkspace(
        forward, interpolated, propagator, field_t, lambda_t,
        cubic_cotangent, vjp_t, spectral_scratch, rank_cotangent,
        raman_channels, fft_field!, fft_lambda!, fft_cubic!,
        ifft_cubic!, ifft_vjp!, fft_raman!, ifft_raman!)
end

function _cuda_mmgnlse_dense_adjoint_workspace(
    prototype::CUDA.CuArray{ComplexF64,3},
)
    arrays = ntuple(_ -> similar(prototype), 7)
    interpolated, propagator, field_t, lambda_t,
        cubic_cotangent, vjp_t, spectral_scratch = arrays
    return CUDAMMGNLSEDenseAdjointWorkspace(
        interpolated, propagator, field_t, lambda_t,
        cubic_cotangent, vjp_t, spectral_scratch,
        plan_fft!(field_t, 1), plan_fft!(lambda_t, 1),
        plan_fft!(cubic_cotangent, 1), plan_ifft!(cubic_cotangent, 1),
        plan_ifft!(spectral_scratch, 1))
end

function _cuda_mmgnlse_dense_raman_adjoint_workspace(
    prototype::CUDA.CuArray{ComplexF64,3},
    overlap,
)
    base = _cuda_mmgnlse_dense_adjoint_workspace(prototype)
    nt, nm, np = size(prototype)
    width = overlap isa CUDAMMGNLSEDenseSpatial ? nm : nm * np
    pair = CUDA.zeros(ComplexF64, nt, width, width)
    filtered_cotangent = similar(pair)
    pair_cotangent = similar(pair)
    return CUDAMMGNLSEDenseRamanAdjointWorkspace(
        base, pair, filtered_cotangent, pair_cotangent,
        plan_fft!(pair, 1), plan_ifft!(pair, 1))
end

_cuda_mmgnlse_dense_adjoint_base(workspace::CUDAMMGNLSEDenseAdjointWorkspace) =
    workspace
_cuda_mmgnlse_dense_adjoint_base(
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
) = workspace.base

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

function _cuda_mmgnlse_dense_spatial_physical_scalar_vjp_kernel!(
    out, field, cotangent, S, coefficient, nt::Int, nm::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        direct = zero(eltype(out))
        conjugated = zero(eltype(out))
        @inbounds for i4 in 1:nm, i3 in 1:nm, i1 in 1:nm
            s = S[i1, target, i3, i4]
            direct += cotangent[t, i1, 1] * s *
                      conj(field[t, i3, 1]) * field[t, i4, 1]
        end
        @inbounds for i3 in 1:nm, i2 in 1:nm, i1 in 1:nm
            s = S[i1, i2, i3, target]
            conjugated += conj(cotangent[t, i1, 1]) * s *
                           field[t, i2, 1] * field[t, i3, 1]
        end
        @inbounds out[t, target, 1] =
            coefficient * (2direct + conjugated)
    end
    return nothing
end

function _cuda_mmgnlse_dense_instantaneous_vjp!(
    out,
    field_t,
    cubic_cotangent,
    overlap,
    coefficient,
    workspace::CUDAMMGNLSEDenseAdjointWorkspace,
)
    fill!(out, zero(eltype(out)))
    nt, nm, np = size(field_t)
    if overlap isa CUDAMMGNLSEDenseSpatial
        if np == 1 && overlap.physical_symmetric
            _launch_1d(
                _cuda_mmgnlse_dense_spatial_physical_scalar_vjp_kernel!,
                nt * nm, out, field_t, cubic_cotangent,
                overlap.values, ComplexF64(coefficient), nt, nm)
            return out
        end
        if np == 1
            _cuda_mmgnlse_dense_spatial_term_vjp!(
                out, field_t, cubic_cotangent,
                overlap, 1, 1, 1, 1, coefficient)
            return out
        end
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 1, 1, 1, 1, coefficient)
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 1, 1, 2, 2, coefficient * (2 / 3))
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 1, 2, 2, 1, coefficient * (1 / 3))
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 2, 2, 2, 2, coefficient)
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 2, 2, 1, 1, coefficient * (2 / 3))
        _cuda_mmgnlse_dense_spatial_term_vjp!(
            out, field_t, cubic_cotangent,
            overlap, 2, 1, 1, 2, coefficient * (1 / 3))
        return out
    end

    nc = nm * np
    _launch_1d(
        _cuda_mmgnlse_dense_component_vjp_kernel!, nt * nc,
        reshape(out, nt, nc), reshape(field_t, nt, nc),
        reshape(cubic_cotangent, nt, nc),
        overlap.values, nt, nc)
    coefficient == 1 || (out .*= coefficient)
    return out
end

function _cuda_mmgnlse_dense_vjp_nonlinear!(
    out,
    field_t,
    cotangent_t,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEDenseAdjointWorkspace,
)
    fill!(out, zero(eltype(out)))
    !cache.nonlinear_active && return out
    workspace.cubic_cotangent .= cotangent_t
    workspace.ifft_cubic! * workspace.cubic_cotangent
    workspace.cubic_cotangent .*= conj.(cache.nonlinear_prefactor)
    workspace.fft_cubic! * workspace.cubic_cotangent
    _cuda_mmgnlse_dense_instantaneous_vjp!(
        out, field_t, workspace.cubic_cotangent, cache.overlap, 1.0,
        workspace)
    return out
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

function _cuda_mmgnlse_cp_rank_cotangents!(
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    cotangent,
    overlap::CUDAMMGNLSECP,
)
    nt, nm, np = size(cotangent)
    if overlap.layout == :spatial
        @inbounds for polarization in 1:np
            mul!(
                @view(workspace.rank_cotangent[:, :, polarization]),
                @view(cotangent[:, :, polarization]),
                overlap.cwu1)
        end
    else
        mul!(
            @view(workspace.rank_cotangent[:, :, 1]),
            reshape(cotangent, nt, nm * np),
            overlap.cwu1)
    end
    return workspace
end

function _cuda_mmgnlse_cp_output_component(out, overlap, polarization)
    if overlap.layout == :spatial
        return @view out[:, :, polarization]
    end
    return reshape(out, size(out, 1), size(out, 2) * size(out, 3))
end

function _cuda_mmgnlse_cp_projection_component(
    projections, overlap, polarization,
)
    index = overlap.layout == :spatial ? polarization : 1
    return @view projections[:, :, index]
end

function _cuda_mmgnlse_cp_instantaneous_term_vjp!(
    out,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    overlap::CUDAMMGNLSECP,
    output_pol::Int,
    p2::Int,
    p3::Int,
    p4::Int,
    coefficient,
)
    iszero(coefficient) && return out
    fw = workspace.forward
    _, U2, U3, U4 = overlap.U
    rank_cotangent = @view workspace.rank_cotangent[
        :, :, overlap.layout == :spatial ? output_pol : 1]
    B2 = _cuda_mmgnlse_cp_projection_component(fw.b2, overlap, p2)
    B3 = _cuda_mmgnlse_cp_projection_component(fw.b3, overlap, p3)
    B4 = _cuda_mmgnlse_cp_projection_component(fw.b4, overlap, p4)
    fw.rank1 .= conj(ComplexF64(coefficient)) .* rank_cotangent

    fw.rank2 .= fw.rank1 .* conj.(B3 .* B4)
    mul!(fw.conjugated, fw.rank2, adjoint(U2))
    _cuda_mmgnlse_cp_output_component(out, overlap, p2) .+= fw.conjugated

    fw.rank2 .= fw.rank1 .* conj.(B2 .* B4)
    mul!(fw.conjugated, fw.rank2, adjoint(U3))
    _cuda_mmgnlse_cp_output_component(out, overlap, p3) .+= fw.conjugated

    fw.rank2 .= fw.rank1 .* conj.(B2 .* B3)
    mul!(fw.conjugated, fw.rank2, adjoint(U4))
    _cuda_mmgnlse_cp_output_component(out, overlap, p4) .+=
        conj.(fw.conjugated)
    return out
end

function _cuda_mmgnlse_filter_rank_adjoint!(
    channels,
    kernel,
    workspace::CUDAMMGNLSECPForwardWorkspace,
)
    workspace.ifft_rank! * channels
    channels .*= conj.(reshape(kernel, :, 1))
    workspace.fft_rank! * channels
    return channels
end

function _cuda_mmgnlse_cp_dual_raman_scale_kernel!(
    channels, kernel, nt::Int, rank::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        channel = (index - 1) ÷ nt + 1
        scale = channel <= rank ? kernel[t] : conj(kernel[t])
        @inbounds channels[index] *= scale
    end
    return nothing
end

function _cuda_mmgnlse_cp_filter_dual_adjoint!(
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    kernel,
)
    channels = workspace.raman_channels
    workspace.ifft_raman! * channels
    nt = size(channels, 1)
    rank = size(channels, 2) ÷ 2
    _launch_1d(
        _cuda_mmgnlse_cp_dual_raman_scale_kernel!,
        length(channels), channels, kernel, nt, rank)
    workspace.fft_raman! * channels
    return channels
end

function _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
    out,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    overlap::CUDAMMGNLSECP,
    kernel,
    output_pol::Int,
    p2::Int,
    p3::Int,
    p4::Int,
    coefficient,
)
    iszero(coefficient) && return out
    fw = workspace.forward
    _, U2, U3, U4 = overlap.U
    rank_cotangent = @view workspace.rank_cotangent[
        :, :, overlap.layout == :spatial ? output_pol : 1]
    B2 = _cuda_mmgnlse_cp_projection_component(fw.b2, overlap, p2)
    B3 = _cuda_mmgnlse_cp_projection_component(fw.b3, overlap, p3)
    B4 = _cuda_mmgnlse_cp_projection_component(fw.b4, overlap, p4)
    fw.rank1 .= conj(ComplexF64(coefficient)) .* rank_cotangent

    fw.rank2 .= B3 .* B4
    _cuda_mmgnlse_filter_rank!(fw.rank2, kernel, fw)
    fw.rank3 .= fw.rank1 .* conj.(fw.rank2)
    mul!(fw.conjugated, fw.rank3, adjoint(U2))
    _cuda_mmgnlse_cp_output_component(out, overlap, p2) .+= fw.conjugated

    fw.rank2 .= fw.rank1 .* conj.(B2)
    _cuda_mmgnlse_filter_rank_adjoint!(fw.rank2, kernel, fw)
    fw.rank3 .= fw.rank2 .* conj.(B4)
    mul!(fw.conjugated, fw.rank3, adjoint(U3))
    _cuda_mmgnlse_cp_output_component(out, overlap, p3) .+= fw.conjugated

    fw.rank3 .= fw.rank2 .* conj.(B3)
    mul!(fw.conjugated, fw.rank3, adjoint(U4))
    _cuda_mmgnlse_cp_output_component(out, overlap, p4) .+=
        conj.(fw.conjugated)
    return out
end

function _cuda_mmgnlse_cp_combined_term_vjp!(
    out,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    overlap::CUDAMMGNLSECP,
    kernel,
    output_pol::Int,
    p2::Int,
    p3::Int,
    p4::Int,
    instantaneous_coefficient,
    delayed_coefficient,
)
    iszero(instantaneous_coefficient) && iszero(delayed_coefficient) &&
        return out
    fw = workspace.forward
    _, U2, U3, U4 = overlap.U
    rank_cotangent = @view workspace.rank_cotangent[
        :, :, overlap.layout == :spatial ? output_pol : 1]
    B2 = _cuda_mmgnlse_cp_projection_component(fw.b2, overlap, p2)
    B3 = _cuda_mmgnlse_cp_projection_component(fw.b3, overlap, p3)
    B4 = _cuda_mmgnlse_cp_projection_component(fw.b4, overlap, p4)
    instantaneous_scale = conj(ComplexF64(instantaneous_coefficient))
    delayed_scale = conj(ComplexF64(delayed_coefficient))

    if iszero(delayed_scale)
        return _cuda_mmgnlse_cp_finish_combined_term_vjp!(
            out, workspace, overlap, output_pol, p2, p3, p4,
            U2, U3, U4, B2, B3, B4, rank_cotangent,
            fw.rank2, fw.rank3, instantaneous_scale, delayed_scale)
    elseif _cuda_mmgnlse_cp_use_dual_raman_batch(overlap)
        filtered_pair = @view workspace.raman_channels[:, 1:overlap.rank]
        filtered_cotangent =
            @view workspace.raman_channels[:, overlap.rank + 1:2 * overlap.rank]
        filtered_pair .= B3 .* B4
        filtered_cotangent .= rank_cotangent .* conj.(B2)
        _cuda_mmgnlse_cp_filter_dual_adjoint!(workspace, kernel)
        return _cuda_mmgnlse_cp_finish_combined_term_vjp!(
            out, workspace, overlap, output_pol, p2, p3, p4,
            U2, U3, U4, B2, B3, B4, rank_cotangent,
            filtered_pair, filtered_cotangent,
            instantaneous_scale, delayed_scale)
    else
        fw.rank2 .= B3 .* B4
        _cuda_mmgnlse_filter_rank!(fw.rank2, kernel, fw)
        fw.rank3 .= rank_cotangent .* conj.(B2)
        _cuda_mmgnlse_filter_rank_adjoint!(fw.rank3, kernel, fw)
        return _cuda_mmgnlse_cp_finish_combined_term_vjp!(
            out, workspace, overlap, output_pol, p2, p3, p4,
            U2, U3, U4, B2, B3, B4, rank_cotangent,
            fw.rank2, fw.rank3, instantaneous_scale, delayed_scale)
    end
end

function _cuda_mmgnlse_cp_finish_combined_term_vjp!(
    out,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    overlap::CUDAMMGNLSECP,
    output_pol::Int,
    p2::Int,
    p3::Int,
    p4::Int,
    U2,
    U3,
    U4,
    B2,
    B3,
    B4,
    rank_cotangent,
    filtered_pair,
    filtered_cotangent,
    instantaneous_scale,
    delayed_scale,
)
    if _cuda_mmgnlse_cp_use_stacked_backprojection(overlap)
        return _cuda_mmgnlse_cp_finish_combined_term_vjp_stacked!(
            out, workspace, overlap, output_pol,
            B2, B3, B4, rank_cotangent,
            filtered_pair, filtered_cotangent,
            instantaneous_scale, delayed_scale)
    end

    fw = workspace.forward
    if iszero(delayed_scale)
        fw.rank1 .= instantaneous_scale .* rank_cotangent .*
                    conj.(B3 .* B4)
    elseif iszero(instantaneous_scale)
        fw.rank1 .=
            delayed_scale .* rank_cotangent .* conj.(filtered_pair)
    else
        fw.rank1 .=
            instantaneous_scale .* rank_cotangent .* conj.(B3 .* B4) .+
            delayed_scale .* rank_cotangent .* conj.(filtered_pair)
    end
    output2 = _cuda_mmgnlse_cp_output_component(out, overlap, p2)
    mul!(
        output2, fw.rank1, adjoint(U2),
        one(eltype(output2)), zero(eltype(output2)))

    if iszero(delayed_scale)
        fw.rank1 .= instantaneous_scale .* rank_cotangent .* conj.(B2)
    elseif iszero(instantaneous_scale)
        fw.rank1 .= delayed_scale .* filtered_cotangent
    else
        fw.rank1 .=
            instantaneous_scale .* rank_cotangent .* conj.(B2) .+
            delayed_scale .* filtered_cotangent
    end

    fw.rank2 .= fw.rank1 .* conj.(B4)
    output3 = _cuda_mmgnlse_cp_output_component(out, overlap, p3)
    mul!(
        output3, fw.rank2, adjoint(U3),
        one(eltype(output3)), one(eltype(output3)))

    # conj(V * U4') == conj(V) * transpose(U4), which lets cuBLAS
    # accumulate the antiholomorphic contribution directly into `out`.
    fw.rank2 .= conj.(fw.rank1 .* conj.(B3))
    output4 = _cuda_mmgnlse_cp_output_component(out, overlap, p4)
    mul!(
        output4, fw.rank2, transpose(U4),
        one(eltype(output4)), one(eltype(output4)))
    return out
end

function _cuda_mmgnlse_cp_finish_combined_term_vjp_stacked!(
    out,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
    overlap::CUDAMMGNLSECP,
    output_pol::Int,
    B2,
    B3,
    B4,
    rank_cotangent,
    filtered_pair,
    filtered_cotangent,
    instantaneous_scale,
    delayed_scale,
)
    fw = workspace.forward

    # Form the shared U3/U4 cotangent before B2 is reused as the first
    # packed backprojection channel.
    if iszero(delayed_scale)
        fw.rank1 .= instantaneous_scale .* rank_cotangent .* conj.(B2)
    elseif iszero(instantaneous_scale)
        fw.rank1 .= delayed_scale .* filtered_cotangent
    else
        fw.rank1 .=
            instantaneous_scale .* rank_cotangent .* conj.(B2) .+
            delayed_scale .* filtered_cotangent
    end

    if iszero(delayed_scale)
        B2 .= instantaneous_scale .* rank_cotangent .* conj.(B3 .* B4)
    elseif iszero(instantaneous_scale)
        B2 .= delayed_scale .* rank_cotangent .* conj.(filtered_pair)
    else
        B2 .=
            instantaneous_scale .* rank_cotangent .* conj.(B3 .* B4) .+
            delayed_scale .* rank_cotangent .* conj.(filtered_pair)
    end

    # B3 must remain live while the antiholomorphic channel is formed.
    # Keep the U3 channel briefly in rank2, then pack all three channels
    # into the already contiguous projection workspace.
    fw.rank2 .= fw.rank1 .* conj.(B4)
    B4 .= conj.(fw.rank1 .* conj.(B3))
    B3 .= fw.rank2

    packed = reshape(fw.projections, size(fw.projections, 1), :)
    output = _cuda_mmgnlse_cp_output_component(out, overlap, output_pol)
    mul!(
        output, packed, overlap.backprojection_factors,
        one(eltype(output)), zero(eltype(output)))
    return out
end

function _cuda_mmgnlse_cp_vjp_nonlinear_baseline!(
    out,
    field_t,
    cotangent_t,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
)
    fill!(out, zero(eltype(out)))
    !cache.nonlinear_active && return out

    workspace.cubic_cotangent .= cotangent_t
    workspace.ifft_cubic! * workspace.cubic_cotangent
    workspace.cubic_cotangent .*= conj.(cache.nonlinear_prefactor)
    workspace.fft_cubic! * workspace.cubic_cotangent

    overlap = cache.overlap
    _cuda_mmgnlse_cp_project!(workspace.forward, field_t, overlap)
    _cuda_mmgnlse_cp_rank_cotangents!(
        workspace, workspace.cubic_cotangent, overlap)
    fraction = cache.raman.fraction
    instant_scale = 1 - fraction
    if overlap.layout == :component
        _cuda_mmgnlse_cp_instantaneous_term_vjp!(
            out, workspace, overlap, 1, 1, 1, 1, instant_scale)
        if !iszero(fraction)
            _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                out, workspace, overlap, cache.raman.combined,
                1, 1, 1, 1, fraction)
        end
        return out
    end

    np = size(field_t, 3)
    if np == 1
        _cuda_mmgnlse_cp_instantaneous_term_vjp!(
            out, workspace, overlap, 1, 1, 1, 1, instant_scale)
        if !iszero(fraction)
            _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                out, workspace, overlap, cache.raman.combined,
                1, 1, 1, 1, fraction)
        end
        return out
    end

    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 1, 1, 1, instant_scale)
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 1, 2, 2, instant_scale * (2 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 2, 2, 1, instant_scale * (1 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 2, 2, 2, instant_scale)
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 2, 1, 1, instant_scale * (2 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 1, 1, 2, instant_scale * (1 / 3))

    if !iszero(fraction)
        if cache.raman.ha_active
            for (output_pol, p2, p3, p4) in (
                (1, 1, 1, 1), (1, 1, 2, 2),
                (2, 2, 1, 1), (2, 2, 2, 2))
                _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                    out, workspace, overlap, cache.raman.ha,
                    output_pol, p2, p3, p4, fraction)
            end
        end
        if cache.raman.hb_active
            for (output_pol, p2, p3, p4, coefficient) in (
                (1, 1, 1, 1, 1.0), (1, 2, 1, 2, 0.5),
                (1, 2, 2, 1, 0.5), (2, 2, 2, 2, 1.0),
                (2, 1, 1, 2, 0.5), (2, 1, 2, 1, 0.5))
                _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                    out, workspace, overlap, cache.raman.hb,
                    output_pol, p2, p3, p4, fraction * coefficient)
            end
        end
    end
    return out
end

function _cuda_mmgnlse_cp_vjp_from_cubic!(
    out,
    field_t,
    cubic_cotangent,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
)
    if !cache.nonlinear_active
        fill!(out, zero(eltype(out)))
        return out
    end

    overlap = cache.overlap
    if overlap.layout == :spatial && size(field_t, 3) > 1
        fill!(out, zero(eltype(out)))
    end
    _cuda_mmgnlse_cp_project!(workspace.forward, field_t, overlap)
    _cuda_mmgnlse_cp_rank_cotangents!(
        workspace, cubic_cotangent, overlap)
    fraction = cache.raman.fraction
    instant_scale = 1 - fraction
    if overlap.layout == :component
        _cuda_mmgnlse_cp_combined_term_vjp!(
            out, workspace, overlap, cache.raman.combined,
            1, 1, 1, 1, instant_scale, fraction)
        return out
    end

    np = size(field_t, 3)
    if np == 1
        _cuda_mmgnlse_cp_combined_term_vjp!(
            out, workspace, overlap, cache.raman.combined,
            1, 1, 1, 1, instant_scale, fraction)
        return out
    end

    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 1, 1, 1, instant_scale)
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 1, 2, 2, instant_scale * (2 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 1, 2, 2, 1, instant_scale * (1 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 2, 2, 2, instant_scale)
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 2, 1, 1, instant_scale * (2 / 3))
    _cuda_mmgnlse_cp_instantaneous_term_vjp!(
        out, workspace, overlap, 2, 1, 1, 2, instant_scale * (1 / 3))

    if !iszero(fraction)
        if cache.raman.ha_active
            for (output_pol, p2, p3, p4) in (
                (1, 1, 1, 1), (1, 1, 2, 2),
                (2, 2, 1, 1), (2, 2, 2, 2))
                _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                    out, workspace, overlap, cache.raman.ha,
                    output_pol, p2, p3, p4, fraction)
            end
        end
        if cache.raman.hb_active
            for (output_pol, p2, p3, p4, coefficient) in (
                (1, 1, 1, 1, 1.0), (1, 2, 1, 2, 0.5),
                (1, 2, 2, 1, 0.5), (2, 2, 2, 2, 1.0),
                (2, 1, 1, 2, 0.5), (2, 1, 2, 1, 0.5))
                _cuda_mmgnlse_cp_delayed_term_vjp_fast!(
                    out, workspace, overlap, cache.raman.hb,
                    output_pol, p2, p3, p4, fraction * coefficient)
            end
        end
    end
    return out
end

function _cuda_mmgnlse_cp_vjp_nonlinear!(
    out,
    field_t,
    cotangent_t,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSECPAdjointWorkspace,
)
    if !_cuda_mmgnlse_cp_is_optimized(cache.overlap)
        return _cuda_mmgnlse_cp_vjp_nonlinear_baseline!(
            out, field_t, cotangent_t, cache, workspace)
    end
    if !cache.nonlinear_active
        fill!(out, zero(eltype(out)))
        return out
    end
    workspace.cubic_cotangent .= cotangent_t
    workspace.ifft_cubic! * workspace.cubic_cotangent
    workspace.cubic_cotangent .*= conj.(cache.nonlinear_prefactor)
    workspace.fft_cubic! * workspace.cubic_cotangent
    return _cuda_mmgnlse_cp_vjp_from_cubic!(
        out, field_t, workspace.cubic_cotangent, cache, workspace)
end


# -- Dense delayed VJPs -----------------------------------------------------

function _cuda_mmgnlse_dense_pair_direct_vjp_kernel!(
    out, cotangent, pairs, coefficient, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        target = (index - 1) ÷ nt + 1
        accumulator = zero(eltype(out))
        @inbounds for i1 in 1:nc
            accumulator += cotangent[t, i1] *
                           conj(pairs[t, i1, target])
        end
        @inbounds out[t, target] += coefficient * accumulator
    end
    return nothing
end

function _cuda_mmgnlse_dense_pair_output_cotangent_kernel!(
    pair_cotangent, field, cotangent, nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i1 = remainder % nc + 1
        i2 = remainder ÷ nc + 1
        @inbounds pair_cotangent[t, i1, i2] =
            cotangent[t, i1] * conj(field[t, i2])
    end
    return nothing
end

function _cuda_mmgnlse_dense_pair_overlap_adjoint_kernel!(
    pair_cotangent, output_cotangent, S,
    nt::Int, nc::Int, total::Int,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        t = (index - 1) % nt + 1
        remainder = (index - 1) ÷ nt
        i3 = remainder % nc + 1
        i4 = remainder ÷ nc + 1
        accumulator = zero(eltype(pair_cotangent))
        @inbounds for i2 in 1:nc, i1 in 1:nc
            accumulator += output_cotangent[t, i1, i2] *
                           conj(S[i1, i2, i3, i4])
        end
        @inbounds pair_cotangent[t, i3, i4] = accumulator
    end
    return nothing
end

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

function _cuda_mmgnlse_filter_pairs_adjoint!(
    channels,
    kernel,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    workspace.ifft_pairs! * channels
    channels .*= conj.(reshape(kernel, :, 1, 1))
    workspace.fft_pairs! * channels
    return channels
end

function _cuda_mmgnlse_filter_pairs!(
    channels,
    kernel,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    workspace.ifft_pairs! * channels
    channels .*= reshape(kernel, :, 1, 1)
    workspace.fft_pairs! * channels
    return channels
end

function _cuda_mmgnlse_dense_spatial_delayed_term_vjp_fast!(
    out,
    field,
    cotangent,
    overlap::CUDAMMGNLSEDenseSpatial,
    kernel,
    output_pol::Int,
    p2::Int,
    p3::Int,
    p4::Int,
    coefficient,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    nt, nm, _ = size(field)
    component3 = @view field[:, :, p3]
    component4 = @view field[:, :, p4]
    workspace.pair .= reshape(component3, nt, nm, 1) .*
                      conj.(reshape(component4, nt, 1, nm))
    _cuda_mmgnlse_filter_pairs!(
        workspace.pair, kernel, workspace)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_delayed_direct_kernel!, nt * nm,
        out, field, cotangent, workspace.pair, overlap.values, output_pol,
        p2, ComplexF64(coefficient), nt, nm)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_filtered_cotangent_kernel!,
        length(workspace.filtered_cotangent),
        workspace.filtered_cotangent, field, cotangent, overlap.values,
        output_pol, p2, ComplexF64(coefficient), nt, nm)
    workspace.pair_cotangent .= workspace.filtered_cotangent
    _cuda_mmgnlse_filter_pairs_adjoint!(
        workspace.pair_cotangent, kernel, workspace)
    _launch_1d(
        _cuda_mmgnlse_dense_spatial_pair_vjp_kernel!, nt * nm,
        out, field, workspace.pair_cotangent, p3, p4, nt, nm)
    return out
end

function _cuda_mmgnlse_dense_component_delayed_vjp_fast!(
    out,
    field,
    cotangent,
    overlap::CUDAMMGNLSEDenseComponent,
    kernel,
    coefficient,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    base = workspace.base
    nt, nm, np = size(field)
    nc = nm * np
    flat_field = reshape(field, nt, nc)
    flat_cotangent = reshape(base.spectral_scratch, nt, nc)
    base.spectral_scratch .= coefficient .* cotangent
    workspace.pair .= reshape(flat_field, nt, nc, 1) .*
                      conj.(reshape(flat_field, nt, 1, nc))
    _cuda_mmgnlse_filter_pairs!(
        workspace.pair, kernel, workspace)
    _launch_1d(
        _cuda_mmgnlse_dense_component_delayed_direct_kernel!, nt * nc,
        reshape(out, nt, nc), flat_field, flat_cotangent,
        workspace.pair, overlap.values, nt, nc)
    _launch_1d(
        _cuda_mmgnlse_dense_component_filtered_cotangent_kernel!,
        length(workspace.filtered_cotangent),
        workspace.filtered_cotangent, flat_field, flat_cotangent,
        overlap.values, nt, nc)
    workspace.pair_cotangent .= workspace.filtered_cotangent
    _cuda_mmgnlse_filter_pairs_adjoint!(
        workspace.pair_cotangent, kernel, workspace)
    _launch_1d(
        _cuda_mmgnlse_dense_component_pair_vjp_kernel!, nt * nc,
        reshape(out, nt, nc), flat_field, workspace.pair_cotangent,
        nt, nc)
    return out
end

function _cuda_mmgnlse_dense_pair_vjp_nonlinear!(
    out,
    field,
    cotangent,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    base = workspace.base
    fill!(out, zero(eltype(out)))
    !cache.nonlinear_active && return out

    base.cubic_cotangent .= cotangent
    base.ifft_cubic! * base.cubic_cotangent
    base.cubic_cotangent .*= conj.(cache.nonlinear_prefactor)
    base.fft_cubic! * base.cubic_cotangent

    nt, nm, np = size(field)
    overlap = cache.overlap
    nc = overlap isa CUDAMMGNLSEDenseSpatial ? nm : nm * np
    flat_field = reshape(field, nt, nc)
    flat_cotangent = reshape(base.cubic_cotangent, nt, nc)
    flat_out = reshape(out, nt, nc)

    # Reverse the same pair-first graph used by the optimized dense forward
    # path.  The expensive forward overlap contraction is shared by the
    # instantaneous and Raman direct terms.
    _launch_1d(
        _cuda_mmgnlse_dense_raman_pairs_kernel!,
        length(workspace.pair), workspace.pair, flat_field,
        overlap.values, nt, nc)

    fraction = cache.raman.fraction
    instantaneous = 1 - fraction
    if !iszero(instantaneous)
        _launch_1d(
            _cuda_mmgnlse_dense_pair_direct_vjp_kernel!,
            nt * nc, flat_out, flat_cotangent, workspace.pair,
            ComplexF64(instantaneous), nt, nc)
    end

    # The cotangent of each contracted output pair is lambda[i1] * conj(A[i2]).
    _launch_1d(
        _cuda_mmgnlse_dense_pair_output_cotangent_kernel!,
        length(workspace.filtered_cotangent),
        workspace.filtered_cotangent, flat_field, flat_cotangent, nt, nc)

    if !iszero(fraction)
        _cuda_mmgnlse_filter_pairs!(
            workspace.pair, cache.raman.combined, workspace)
        _launch_1d(
            _cuda_mmgnlse_dense_pair_direct_vjp_kernel!,
            nt * nc, flat_out, flat_cotangent, workspace.pair,
            ComplexF64(fraction), nt, nc)

        workspace.pair_cotangent .= workspace.filtered_cotangent
        _cuda_mmgnlse_filter_pairs_adjoint!(
            workspace.pair_cotangent, cache.raman.combined, workspace)
        workspace.filtered_cotangent .=
            instantaneous .* workspace.filtered_cotangent .+
            fraction .* workspace.pair_cotangent
    elseif instantaneous != 1
        workspace.filtered_cotangent .*= instantaneous
    end

    # Apply S* once to the combined instantaneous-plus-Raman pair
    # cotangent, then reverse A[i3] * conj(A[i4]).
    _launch_1d(
        _cuda_mmgnlse_dense_pair_overlap_adjoint_kernel!,
        length(workspace.pair_cotangent),
        workspace.pair_cotangent, workspace.filtered_cotangent,
        overlap.values, nt, nc)
    if overlap isa CUDAMMGNLSEDenseSpatial
        _launch_1d(
            _cuda_mmgnlse_dense_spatial_pair_vjp_kernel!,
            nt * nm, out, field, workspace.pair_cotangent,
            1, 1, nt, nm)
    else
        _launch_1d(
            _cuda_mmgnlse_dense_component_pair_vjp_kernel!,
            nt * nc, flat_out, flat_field,
            workspace.pair_cotangent, nt, nc)
    end
    return out
end

function _cuda_mmgnlse_dense_vjp_nonlinear!(
    out,
    field_t,
    cotangent_t,
    cache::CUDAMMGNLSEForwardCache,
    workspace::CUDAMMGNLSEDenseRamanAdjointWorkspace,
)
    base = workspace.base
    overlap = cache.overlap
    _, _, np = size(field_t)
    if overlap isa CUDAMMGNLSEDenseComponent ||
       (overlap isa CUDAMMGNLSEDenseSpatial && np == 1)
        return _cuda_mmgnlse_dense_pair_vjp_nonlinear!(
            out, field_t, cotangent_t, cache, workspace)
    end

    fill!(out, zero(eltype(out)))
    !cache.nonlinear_active && return out
    base.cubic_cotangent .= cotangent_t
    base.ifft_cubic! * base.cubic_cotangent
    base.cubic_cotangent .*= conj.(cache.nonlinear_prefactor)
    base.fft_cubic! * base.cubic_cotangent

    fraction = cache.raman.fraction
    if fraction != 1
        _cuda_mmgnlse_dense_instantaneous_vjp!(
            out, field_t, base.cubic_cotangent, cache.overlap,
            1 - fraction, base)
    end
    iszero(fraction) && return out

    if overlap isa CUDAMMGNLSEDenseSpatial
        if np == 1
            _cuda_mmgnlse_dense_spatial_delayed_term_vjp_fast!(
                out, field_t, base.cubic_cotangent, overlap,
                cache.raman.combined, 1, 1, 1, 1, fraction, workspace)
            return out
        end
        if cache.raman.ha_active
            for (output_pol, p2, p3, p4) in (
                (1, 1, 1, 1), (1, 1, 2, 2),
                (2, 2, 1, 1), (2, 2, 2, 2))
                _cuda_mmgnlse_dense_spatial_delayed_term_vjp_fast!(
                    out, field_t, base.cubic_cotangent, overlap,
                    cache.raman.ha, output_pol, p2, p3, p4,
                    fraction, workspace)
            end
        end
        if cache.raman.hb_active
            for (output_pol, p2, p3, p4, coefficient) in (
                (1, 1, 1, 1, 1.0), (1, 2, 1, 2, 0.5),
                (1, 2, 2, 1, 0.5), (2, 2, 2, 2, 1.0),
                (2, 1, 1, 2, 0.5), (2, 1, 2, 1, 0.5))
                _cuda_mmgnlse_dense_spatial_delayed_term_vjp_fast!(
                    out, field_t, base.cubic_cotangent, overlap,
                    cache.raman.hb, output_pol, p2, p3, p4,
                    fraction * coefficient, workspace)
            end
        end
        return out
    end

    _cuda_mmgnlse_dense_component_delayed_vjp_fast!(
        out, field_t, base.cubic_cotangent, overlap,
        cache.raman.combined, fraction, workspace)
    return out
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

function _cuda_mmgnlse_to_host_solution(device_solution, backend::Symbol)
    device_cache = device_solution.cache
    host_cache = (
        backend,
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
    parameters;
    forward_solution,
    initial_field,
    dz_forward,
    cp_optimization::Symbol,
    backend::Symbol,
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
            method=forward_solution.method, saveat=:steps,
            cp_optimization)
        return replay, _cuda_mmgnlse_to_host_solution(replay, backend)
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
        saveat=:steps, cp_optimization)
    return device_forward, _cuda_mmgnlse_to_host_solution(
        device_forward, backend)
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

function _cuda_mmgnlse_interpolate_interaction_field!(
    out, fields, zgrid::AbstractVector, z::Real,
)
    tolerance = 64eps(float(eltype(zgrid))) *
                max(one(float(eltype(zgrid))),
                    abs(first(zgrid)), abs(last(zgrid)))
    z < first(zgrid) - tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z before the forward trajectory."))
    z > last(zgrid) + tolerance && throw(ArgumentError(
        "Adjoint interpolation requested z=$z after the forward trajectory."))
    if z <= first(zgrid)
        out .= @view fields[:, :, :, 1]
        return out
    elseif z >= last(zgrid)
        out .= @view fields[:, :, :, end]
        return out
    end
    right = searchsortedfirst(zgrid, z)
    if zgrid[right] == z
        out .= @view fields[:, :, :, right]
        return out
    end
    left = right - 1
    weight = (z - zgrid[left]) / (zgrid[right] - zgrid[left])
    out .= (1 - weight) .* @view(fields[:, :, :, left]) .+
           weight .* @view(fields[:, :, :, right])
    return out
end

function _cuda_mmgnlse_adjoint_propagator!(
    out,
    cache::CUDAMMGNLSEForwardCache,
    coordinate::Real,
)
    if cache.constant_passive
        out .= exp.(cache.beta_operator .* coordinate)
    else
        out .= _cuda_mmgnlse_linear_propagator(cache, 0.0, coordinate)
    end
    return out
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
    backend::Symbol=:cuda,
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
    cp_optimization = _cuda_mmgnlse_cp_optimization(backend)

    device_forward, host_forward = _cuda_mmgnlse_prepare_forward(
        parameters;
        forward_solution,
        initial_field,
        dz_forward,
        cp_optimization,
        backend)
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

    cache = _cuda_mmgnlse_forward_cache(
        parameters, cp_optimization;
        dense_contraction=parameters.S isa
                          PulsePropagation.MMGNLSECPDecomposition ?
                          nothing : :custom)
    forward_interaction = _cuda_mmgnlse_forward_interaction_fields(
        device_forward, cache)
    terminal_propagator = _cuda_mmgnlse_linear_propagator(
        cache, 0.0, T(parameters.length))
    lambda_interaction = conj.(terminal_propagator) .* lambda_raw
    cp_workspace = cache.overlap isa CUDAMMGNLSECP ?
        _cuda_mmgnlse_cp_adjoint_workspace(
            similar(lambda_interaction), cache.overlap) :
        nothing
    dense_workspace = if cp_workspace !== nothing
        nothing
    elseif iszero(cache.raman.fraction)
        _cuda_mmgnlse_dense_adjoint_workspace(
            similar(lambda_interaction))
    else
        _cuda_mmgnlse_dense_raman_adjoint_workspace(
            similar(lambda_interaction), cache.overlap)
    end
    dense_base = dense_workspace === nothing ?
        nothing : _cuda_mmgnlse_dense_adjoint_base(dense_workspace)

    function rhs!(derivative, lambda_tilde, _, z)
        if !cache.nonlinear_active
            fill!(derivative, zero(eltype(derivative)))
            return nothing
        end
        coordinate = T(z)
        if cp_workspace === nothing && dense_workspace === nothing
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
        elseif cp_workspace !== nothing
            _cuda_mmgnlse_interpolate_interaction_field!(
                cp_workspace.interpolated, forward_interaction,
                device_forward.z, coordinate)
            _cuda_mmgnlse_adjoint_propagator!(
                cp_workspace.propagator, cache, coordinate)
            cp_workspace.field_t .=
                cp_workspace.propagator .* cp_workspace.interpolated
            cp_workspace.fft_field! * cp_workspace.field_t
            if _cuda_mmgnlse_cp_is_optimized(cache.overlap)
                # The generic temporal VJP applies
                # ifft(fft(lambda_raw)) before multiplying by
                # conj(prefactor). The optimized solver paths cancel those
                # transforms exactly.
                cp_workspace.cubic_cotangent .=
                    lambda_tilde ./ conj.(cp_workspace.propagator) .*
                    conj.(cache.nonlinear_prefactor)
                cp_workspace.fft_cubic! * cp_workspace.cubic_cotangent
                _cuda_mmgnlse_cp_vjp_from_cubic!(
                    cp_workspace.vjp_t, cp_workspace.field_t,
                    cp_workspace.cubic_cotangent, cache, cp_workspace)
            else
                cp_workspace.lambda_t .=
                    lambda_tilde ./ conj.(cp_workspace.propagator)
                cp_workspace.fft_lambda! * cp_workspace.lambda_t
                _cuda_mmgnlse_cp_vjp_nonlinear_baseline!(
                    cp_workspace.vjp_t, cp_workspace.field_t,
                    cp_workspace.lambda_t, cache, cp_workspace)
            end
            cp_workspace.spectral_scratch .= cp_workspace.vjp_t
            cp_workspace.ifft_vjp! * cp_workspace.spectral_scratch
            derivative .= -conj.(cp_workspace.propagator) .*
                          cp_workspace.spectral_scratch
        else
            _cuda_mmgnlse_interpolate_interaction_field!(
                dense_base.interpolated, forward_interaction,
                device_forward.z, coordinate)
            _cuda_mmgnlse_adjoint_propagator!(
                dense_base.propagator, cache, coordinate)
            dense_base.field_t .=
                dense_base.propagator .* dense_base.interpolated
            dense_base.fft_field! * dense_base.field_t
            dense_base.lambda_t .=
                lambda_tilde ./ conj.(dense_base.propagator)
            dense_base.fft_lambda! * dense_base.lambda_t
            _cuda_mmgnlse_dense_vjp_nonlinear!(
                dense_base.vjp_t, dense_base.field_t,
                dense_base.lambda_t, cache, dense_workspace)
            dense_base.spectral_scratch .= dense_base.vjp_t
            dense_base.ifft_vjp! * dense_base.spectral_scratch
            derivative .= -conj.(dense_base.propagator) .*
                          dense_base.spectral_scratch
        end
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
        backend,
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
