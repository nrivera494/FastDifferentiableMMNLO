# Experimental GPU-only scalar 3+1D solvers. This file is included inside
# PulsePropagationCUDAExt so CUDA remains a weak dependency of the main package.

mutable struct CUDAFullField3DWorkspace{
    F,R1,R2,C1,C2,R3,C3,PI,PF,PTI,PTF,PSI,PSF,PR,PC,
}
    field::F
    stage::F
    derivative::F
    accumulator::F
    mixed_field::F
    potential::R2
    temporal_generator::R1
    temporal_loss::R1
    transverse_generator::R2
    temporal_phase::C1
    transverse_phase::C2
    potential_phase::C2
    nonlinear_prefactor::C1
    shock_prefactor::C1
    intensity::R3
    raman_spectrum::C3
    raman_response::C1
    to_spectral!::PI
    to_real!::PF
    temporal_to_spectral!::PTI
    temporal_to_real!::PTF
    spatial_to_spectral!::PSI
    spatial_to_real!::PSF
    rfft_intensity!::PR
    irfft_delayed!::PC
    phase_distance::Float64
    potential_phase_distance::Float64
    nonlinear_coefficient::Float64
    raman_fraction::Float64
    potential_active::Bool
    nonlinear_active::Bool
    phase_active::Bool
    raman_active::Bool
    self_steepening::Bool
end

function _cuda_full_field_phase_kernel!(field, potential, distance,
                                        nonlinear_coefficient, nt, total)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        spatial_index = (index - 1) ÷ nt + 1
        value = field[index]
        intensity = abs2(value)
        angle = distance * (
            potential[spatial_index] + nonlinear_coefficient * intensity)
        sine, cosine = sincos(angle)
        field[index] = value * Complex(cosine, sine)
    end
    return nothing
end

function _cuda_full_field_raman_phase_kernel!(
    field, potential, delayed, distance,
    nonlinear_coefficient, raman_fraction, nt, total,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        spatial_index = (index - 1) ÷ nt + 1
        value = field[index]
        response = (one(raman_fraction) - raman_fraction) * abs2(value) +
                   raman_fraction * delayed[index]
        angle = distance * (
            potential[spatial_index] + nonlinear_coefficient * response)
        sine, cosine = sincos(angle)
        field[index] = value * Complex(cosine, sine)
    end
    return nothing
end

function _cuda_full_field_polarization_kernel!(out, field, total)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        value = field[index]
        out[index] = value * abs2(value)
    end
    return nothing
end

function _cuda_full_field_raman_polarization_kernel!(
    out, field, delayed, raman_fraction, total,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        value = field[index]
        response = (one(raman_fraction) - raman_fraction) * abs2(value) +
                   raman_fraction * delayed[index]
        out[index] = value * response
    end
    return nothing
end

function _cuda_full_field_potential_rhs_kernel!(out, field, potential, nt, total)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        spatial_index = (index - 1) ÷ nt + 1
        value = field[index]
        out[index] = value * Complex(zero(real(value)), potential[spatial_index])
    end
    return nothing
end

function _cuda_full_field_add_potential_mixed_kernel!(
    out, mixed_field, potential, nt, total,
)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if index <= total
        spatial_index = (index - 1) ÷ nt + 1
        value = mixed_field[index]
        out[index] +=
            value * Complex(zero(real(value)), potential[spatial_index])
    end
    return nothing
end

function _cuda_full_field_launch(kernel, count::Integer, args...)
    threads = 256
    blocks = cld(Int(count), threads)
    CUDA.@cuda threads=threads blocks=blocks kernel(args...)
    return nothing
end

function _cuda_full_field_workspace(
    initial_field,
    parameters::PulsePropagation.FullField3DParameters,
    ::Type{T},
    method::Symbol=:strang,
) where {T<:AbstractFloat}
    CT = Complex{T}
    grid = parameters.grid
    nt, nx, ny = PulsePropagation.full_field_shape(grid)
    field = CUDA.CuArray(CT.(initial_field))
    potential = CUDA.CuArray(T.(parameters.potential))
    temporal_generator = CUDA.CuArray(T.(ifftshift(parameters.dispersion)))
    temporal_loss = CUDA.CuArray(T.(ifftshift(parameters.spectral_loss)))
    kx = T.(PulsePropagation._full_field_wavenumbers(nx, grid.dx))
    ky = T.(PulsePropagation._full_field_wavenumbers(ny, grid.dy))
    transverse_generator_host = -(
        reshape(kx .^ 2, nx, 1) .+ reshape(ky .^ 2, 1, ny)) ./
        T(2parameters.beta0)
    transverse_generator = CUDA.CuArray(transverse_generator_host)
    temporal_phase = CUDA.zeros(CT, nt)
    transverse_phase = CUDA.zeros(CT, nx, ny)

    fraction = T(PulsePropagation.raman_fraction(parameters.raman))
    nonlinear_coefficient = T(
        parameters.omega0 * parameters.n2 / 2.99792458e-4)
    potential_active = any(!iszero, parameters.potential)
    nonlinear_active = !iszero(nonlinear_coefficient)
    phase_active = potential_active || nonlinear_active
    raman_active = nonlinear_active && !iszero(fraction)
    split_phase_method = method in (:strang, :composed_strang4)
    potential_phase = split_phase_method && potential_active ?
        CUDA.zeros(CT, nx, ny) : CUDA.zeros(CT, 0, 0)

    if parameters.self_steepening
        omega = PulsePropagation.ifft_frequency_grid(nt, grid.time.dt)
        damp = PulsePropagation.damped_freq_window(nt)
        prefactor_host = Complex{T}.(
            1im .* parameters.n2 .* (omega .+ parameters.omega0) ./
            2.99792458e-4 .* damp)
    else
        prefactor_host = fill(
            Complex{T}(zero(T), nonlinear_coefficient), nt)
    end
    nonlinear_prefactor = CUDA.CuArray(prefactor_host)
    shock_prefactor_host = if parameters.self_steepening
        prefactor_host .- Complex{T}(zero(T), nonlinear_coefficient)
    else
        fill(zero(CT), nt)
    end
    shock_prefactor = CUDA.CuArray(shock_prefactor_host)

    needs_strang_midpoint =
        method === :strang && parameters.self_steepening && nonlinear_active
    needs_composed_midpoint =
        method === :composed_strang4 && parameters.self_steepening &&
        nonlinear_active
    if method === :rk4ip
        stage = similar(field)
        derivative = phase_active ? similar(field) : CUDA.zeros(CT, 0, 0, 0)
        accumulator = phase_active ? similar(field) : CUDA.zeros(CT, 0, 0, 0)
        mixed_field = potential_active && nonlinear_active ?
            similar(field) : CUDA.zeros(CT, 0, 0, 0)
    elseif needs_composed_midpoint
        stage = similar(field)
        derivative = similar(field)
        accumulator = similar(field)
        mixed_field = CUDA.zeros(CT, 0, 0, 0)
    elseif needs_strang_midpoint
        stage = similar(field)
        derivative = similar(field)
        accumulator = CUDA.zeros(CT, 0, 0, 0)
        mixed_field = CUDA.zeros(CT, 0, 0, 0)
    else
        stage = CUDA.zeros(CT, 0, 0, 0)
        derivative = CUDA.zeros(CT, 0, 0, 0)
        accumulator = CUDA.zeros(CT, 0, 0, 0)
        mixed_field = CUDA.zeros(CT, 0, 0, 0)
    end

    to_spectral! = plan_ifft!(field, (1, 2, 3))
    to_real! = plan_fft!(field, (1, 2, 3))
    needs_temporal_plans = nonlinear_active &&
        (method === :rk4ip || needs_strang_midpoint ||
         needs_composed_midpoint)
    temporal_to_spectral! = needs_temporal_plans ? plan_ifft!(field, 1) : nothing
    temporal_to_real! = needs_temporal_plans ? plan_fft!(field, 1) : nothing
    needs_spatial_plans = method === :rk4ip && phase_active
    spatial_to_spectral! = needs_spatial_plans ? plan_ifft!(field, (2, 3)) : nothing
    spatial_to_real! = needs_spatial_plans ? plan_fft!(field, (2, 3)) : nothing

    if raman_active
        intensity = CUDA.zeros(T, nt, nx, ny)
        raman_spectrum = CUDA.zeros(CT, nt ÷ 2 + 1, nx, ny)
        response = PulsePropagation._full_field_raman_spectrum(parameters, T)
        raman_response = CUDA.CuArray(response)
        rfft_intensity! = plan_rfft(intensity, 1)
        irfft_delayed! = plan_irfft(raman_spectrum, nt, 1)
    else
        intensity = CUDA.zeros(T, 0, 0, 0)
        raman_spectrum = CUDA.zeros(CT, 0, 0, 0)
        raman_response = CUDA.zeros(CT, 0)
        rfft_intensity! = nothing
        irfft_delayed! = nothing
    end

    return CUDAFullField3DWorkspace(
        field, stage, derivative, accumulator, mixed_field,
        potential, temporal_generator, temporal_loss, transverse_generator,
        temporal_phase, transverse_phase, potential_phase,
        nonlinear_prefactor, shock_prefactor,
        intensity, raman_spectrum, raman_response,
        to_spectral!, to_real!, temporal_to_spectral!, temporal_to_real!,
        spatial_to_spectral!, spatial_to_real!,
        rfft_intensity!, irfft_delayed!, NaN, NaN,
        Float64(nonlinear_coefficient), Float64(fraction),
        potential_active, nonlinear_active, phase_active, raman_active,
        parameters.self_steepening)
end

function _cuda_full_field_prepare_linear!(workspace::CUDAFullField3DWorkspace,
                                          distance::Real)
    distance_value = Float64(distance)
    if workspace.phase_distance != distance_value
        T = typeof(real(zero(eltype(workspace.field))))
        scale = T(distance_value)
        imaginary_unit = Complex{T}(zero(T), one(T))
        workspace.temporal_phase .= exp.(
            scale .* (imaginary_unit .* workspace.temporal_generator .-
                      T(0.5) .* workspace.temporal_loss))
        workspace.transverse_phase .=
            cis.(scale .* workspace.transverse_generator)
        workspace.phase_distance = distance_value
    end
    return workspace
end

function _cuda_full_field_apply_linear_spectral!(
    array, workspace::CUDAFullField3DWorkspace, distance::Real,
)
    iszero(distance) && return array
    _cuda_full_field_prepare_linear!(workspace, distance)
    nt, nx, ny = size(array)
    array .*= reshape(workspace.temporal_phase, nt, 1, 1) .*
              reshape(workspace.transverse_phase, 1, nx, ny)
    return array
end

function _cuda_full_field_linear!(workspace::CUDAFullField3DWorkspace,
                                  distance::Real)
    iszero(distance) && return workspace.field
    mul!(workspace.field, workspace.to_spectral!, workspace.field)
    _cuda_full_field_apply_linear_spectral!(
        workspace.field, workspace, distance)
    mul!(workspace.field, workspace.to_real!, workspace.field)
    return workspace.field
end

function _cuda_full_field_potential_only!(
    workspace::CUDAFullField3DWorkspace, distance::Real,
)
    workspace.potential_active || return workspace.field
    distance_value = Float64(distance)
    if workspace.potential_phase_distance != distance_value
        T = typeof(real(zero(eltype(workspace.field))))
        workspace.potential_phase .=
            cis.(T(distance_value) .* workspace.potential)
        workspace.potential_phase_distance = distance_value
    end
    nx = size(workspace.field, 2)
    ny = size(workspace.field, 3)
    workspace.field .*= reshape(workspace.potential_phase, 1, nx, ny)
    return workspace.field
end

function _cuda_full_field_raman_delayed!(
    workspace::CUDAFullField3DWorkspace, field,
)
    workspace.intensity .= abs2.(field)
    mul!(workspace.raman_spectrum,
         workspace.rfft_intensity!, workspace.intensity)
    workspace.raman_spectrum .*= reshape(
        workspace.raman_response, :, 1, 1)
    mul!(workspace.intensity,
         workspace.irfft_delayed!, workspace.raman_spectrum)
    return workspace.intensity
end

function _cuda_full_field_polarization!(
    out, field, workspace::CUDAFullField3DWorkspace,
)
    total = length(field)
    if workspace.raman_active
        _cuda_full_field_raman_delayed!(workspace, field)
        T = typeof(real(zero(eltype(field))))
        _cuda_full_field_launch(
            _cuda_full_field_raman_polarization_kernel!, total,
            out, field, workspace.intensity,
            T(workspace.raman_fraction), total)
    else
        _cuda_full_field_launch(
            _cuda_full_field_polarization_kernel!, total,
            out, field, total)
    end
    return out
end

function _cuda_full_field_exact_phase!(workspace::CUDAFullField3DWorkspace,
                                       distance::Real)
    workspace.phase_active || return workspace.field
    !workspace.nonlinear_active &&
        return _cuda_full_field_potential_only!(workspace, distance)
    T = typeof(real(zero(eltype(workspace.field))))
    step = T(distance)
    coefficient = T(workspace.nonlinear_coefficient)
    nt = size(workspace.field, 1)
    total = length(workspace.field)
    if workspace.raman_active
        _cuda_full_field_raman_delayed!(workspace, workspace.field)
        _cuda_full_field_launch(
            _cuda_full_field_raman_phase_kernel!, total,
            workspace.field, workspace.potential, workspace.intensity,
            step, coefficient, T(workspace.raman_fraction), nt, total)
    else
        _cuda_full_field_launch(
            _cuda_full_field_phase_kernel!, total,
            workspace.field, workspace.potential,
            step, coefficient, nt, total)
    end
    return workspace.field
end

function _cuda_full_field_nonlinear_rhs_time!(
    out, field, workspace::CUDAFullField3DWorkspace,
)
    workspace.nonlinear_active || return fill!(out, zero(eltype(out)))
    _cuda_full_field_polarization!(out, field, workspace)
    mul!(out, workspace.temporal_to_spectral!, out)
    nt = size(out, 1)
    out .*= reshape(workspace.nonlinear_prefactor, nt, 1, 1)
    mul!(out, workspace.temporal_to_real!, out)
    return out
end

function _cuda_full_field_shock_rhs_time!(
    out, field, workspace::CUDAFullField3DWorkspace,
)
    workspace.nonlinear_active || return fill!(out, zero(eltype(out)))
    _cuda_full_field_polarization!(out, field, workspace)
    mul!(out, workspace.temporal_to_spectral!, out)
    nt = size(out, 1)
    out .*= reshape(workspace.shock_prefactor, nt, 1, 1)
    mul!(out, workspace.temporal_to_real!, out)
    return out
end

"""
Advance only the self-steepening correction by an iterated midpoint map.

The explicit-midpoint predictor differs from the implicit-midpoint fixed point
by `O(h^3)`. Two Picard corrections raise that difference to `O(h^5)`, which
is sufficient for the surrounding fourth-order composition. The ordinary
Kerr--Raman phase is deliberately excluded and is applied exactly by
`_cuda_full_field_exact_phase!`.
"""
function _cuda_full_field_shock_midpoint!(
    workspace::CUDAFullField3DWorkspace, distance::Real,
)
    (!workspace.self_steepening || !workspace.nonlinear_active) &&
        return workspace.field
    T = typeof(real(zero(eltype(workspace.field))))
    step = T(distance)

    workspace.accumulator .= workspace.field
    _cuda_full_field_shock_rhs_time!(
        workspace.derivative, workspace.accumulator, workspace)
    workspace.stage .=
        workspace.accumulator .+ (step / 2) .* workspace.derivative
    _cuda_full_field_shock_rhs_time!(
        workspace.derivative, workspace.stage, workspace)
    workspace.field .= workspace.accumulator .+ step .* workspace.derivative

    # Two fixed-point corrections make the midpoint solve accurate through
    # the local O(h^5) scale required by the composed fourth-order step.
    for _ in 1:2
        workspace.stage .=
            (workspace.accumulator .+ workspace.field) ./ 2
        _cuda_full_field_shock_rhs_time!(
            workspace.derivative, workspace.stage, workspace)
        workspace.field .=
            workspace.accumulator .+ step .* workspace.derivative
    end
    return workspace.field
end

function _cuda_full_field_symmetric_residual!(
    workspace::CUDAFullField3DWorkspace, distance::Real,
)
    if !workspace.self_steepening || !workspace.nonlinear_active
        return _cuda_full_field_exact_phase!(workspace, distance)
    end
    _cuda_full_field_exact_phase!(workspace, distance / 2)
    _cuda_full_field_shock_midpoint!(workspace, distance)
    _cuda_full_field_exact_phase!(workspace, distance / 2)
    return workspace.field
end

function _cuda_full_field_strang_residual!(
    workspace::CUDAFullField3DWorkspace, distance::Real,
)
    if !workspace.self_steepening || !workspace.nonlinear_active
        return _cuda_full_field_exact_phase!(workspace, distance)
    end
    T = typeof(real(zero(eltype(workspace.field))))
    step = T(distance)
    _cuda_full_field_potential_only!(workspace, distance / 2)
    _cuda_full_field_nonlinear_rhs_time!(
        workspace.derivative, workspace.field, workspace)
    workspace.stage .= workspace.field .+ (step / 2) .* workspace.derivative
    _cuda_full_field_nonlinear_rhs_time!(
        workspace.derivative, workspace.stage, workspace)
    workspace.field .+= step .* workspace.derivative
    _cuda_full_field_potential_only!(workspace, distance / 2)
    return workspace.field
end

# Retain the private profiler entry point used by the existing benchmark.
_cuda_full_field_nonlinear!(workspace::CUDAFullField3DWorkspace, distance::Real) =
    _cuda_full_field_strang_residual!(workspace, distance)

function _cuda_full_field_residual_spectral!(
    out, field, workspace::CUDAFullField3DWorkspace,
)
    workspace.phase_active || return fill!(out, zero(eltype(out)))
    out .= field
    # Joint spectral -> (omega, x, y). Keeping this mixed representation
    # avoids a second joint transform for the frequency-independent potential.
    mul!(out, workspace.spatial_to_real!, out)
    nt = size(out, 1)
    total = length(out)
    if workspace.potential_active && workspace.nonlinear_active
        workspace.mixed_field .= out
        mul!(out, workspace.temporal_to_real!, out)
        _cuda_full_field_polarization!(out, out, workspace)
        mul!(out, workspace.temporal_to_spectral!, out)
        out .*= reshape(workspace.nonlinear_prefactor, nt, 1, 1)
        _cuda_full_field_launch(
            _cuda_full_field_add_potential_mixed_kernel!, total,
            out, workspace.mixed_field, workspace.potential, nt, total)
        mul!(out, workspace.spatial_to_spectral!, out)
    elseif workspace.nonlinear_active
        mul!(out, workspace.temporal_to_real!, out)
        _cuda_full_field_polarization!(out, out, workspace)
        mul!(out, workspace.temporal_to_spectral!, out)
        out .*= reshape(workspace.nonlinear_prefactor, nt, 1, 1)
        mul!(out, workspace.spatial_to_spectral!, out)
    else
        _cuda_full_field_launch(
            _cuda_full_field_potential_rhs_kernel!, total,
            out, out, workspace.potential, nt, total)
        mul!(out, workspace.spatial_to_spectral!, out)
    end
    return out
end

function _cuda_full_field_rk4ip_step!(
    workspace::CUDAFullField3DWorkspace, distance::Real,
)
    if !workspace.phase_active
        return _cuda_full_field_apply_linear_spectral!(
            workspace.field, workspace, distance)
    end
    T = typeof(real(zero(eltype(workspace.field))))
    step = T(distance)
    half = distance / 2

    # k1 is evaluated before field is overwritten by the midpoint linear base.
    _cuda_full_field_residual_spectral!(
        workspace.derivative, workspace.field, workspace)
    _cuda_full_field_apply_linear_spectral!(
        workspace.field, workspace, half)
    _cuda_full_field_apply_linear_spectral!(
        workspace.derivative, workspace, half)
    workspace.accumulator .= workspace.derivative

    workspace.stage .= workspace.field .+ (step / 2) .* workspace.derivative
    _cuda_full_field_residual_spectral!(
        workspace.derivative, workspace.stage, workspace)
    workspace.accumulator .+= T(2) .* workspace.derivative

    workspace.stage .= workspace.field .+ (step / 2) .* workspace.derivative
    _cuda_full_field_residual_spectral!(
        workspace.derivative, workspace.stage, workspace)
    workspace.accumulator .+= T(2) .* workspace.derivative

    workspace.stage .= workspace.field .+ step .* workspace.derivative
    _cuda_full_field_apply_linear_spectral!(
        workspace.stage, workspace, half)
    _cuda_full_field_residual_spectral!(
        workspace.derivative, workspace.stage, workspace)

    workspace.stage .= workspace.field .+ (step / 6) .* workspace.accumulator
    _cuda_full_field_apply_linear_spectral!(
        workspace.stage, workspace, half)
    workspace.field .= workspace.stage .+ (step / 6) .* workspace.derivative
    return workspace.field
end

function _cuda_full_field_copy_to_host!(destination, source)
    length(destination) == length(source) || throw(DimensionMismatch(
        "Full-field device and host save buffers have different lengths."))
    eltype(destination) === eltype(source) || throw(ArgumentError(
        "Full-field device and host save buffers must have the same element type."))
    GC.@preserve destination source begin
        unsafe_copyto!(
            pointer(destination), pointer(source), length(source);
            stream=CUDA.stream(), async=false)
    end
    return destination
end

function _cuda_full_field_copy_rk4ip_to_host!(
    destination, workspace::CUDAFullField3DWorkspace,
)
    workspace.stage .= workspace.field
    mul!(workspace.stage, workspace.to_real!, workspace.stage)
    return _cuda_full_field_copy_to_host!(destination, workspace.stage)
end

function PulsePropagation._solve_full_field_3d_cuda(
    initial_field::AbstractArray{<:Number,3},
    parameters::PulsePropagation.FullField3DParameters,
    dz::Real;
    precision::Symbol=:float32,
    saveat=nothing,
    device=nothing,
    synchronize::Bool=true,
    method::Symbol=:strang,
)
    CUDA.functional() || error(
        "CUDA.jl is available but no functional CUDA device was found.")
    device === nothing || CUDA.device!(device)
    T = precision === :float32 ? Float32 : Float64
    CT = Complex{T}
    step = Float64(dz)
    targets, every_step = PulsePropagation._mmgnlse_save_targets(
        Float64(parameters.length), saveat)
    steps, saved = PulsePropagation._mmgnlse_step_grid(
        Float64(parameters.length), step, targets, every_step)

    CUDA.synchronize()
    memory_before = Int(coalesce(CUDA.used_memory(), 0))
    workspace = _cuda_full_field_workspace(initial_field, parameters, T, method)
    CUDA.synchronize()
    workspace_bytes = max(
        Int(coalesce(CUDA.used_memory(), 0)) - memory_before, 0)

    nt, nx, ny = size(workspace.field)
    fields = Array{CT,4}(undef, nt, nx, ny, length(saved))
    @views fields[:, :, :, 1] .= CT.(initial_field)
    save_index = 2
    tolerance = 32eps(Float64) * max(1.0, Float64(parameters.length))
    linear_fft_pairs = 0
    residual_steps = 0
    residual_rhs_evaluations = 0
    nonlinear_steps = 0
    nonlinear_rhs_evaluations = 0
    raman_fft_pairs = 0
    temporal_nonlinear_fft_pairs = 0
    spatial_fft_pairs = 0
    self_steepening_fft_pairs = 0
    joint_fft_transforms = 0
    mixed_fft_calls = 0
    nsteps = length(steps) - 1

    if method === :strang
        first_step = steps[2] - steps[1]
        _cuda_full_field_linear!(workspace, first_step / 2)
        linear_fft_pairs += 1

        for index in 1:nsteps
            z0, z1 = steps[index], steps[index + 1]
            local_step = z1 - z0
            _cuda_full_field_strang_residual!(workspace, local_step)
            residual_steps += workspace.phase_active
            nonlinear_steps += workspace.nonlinear_active
            if workspace.self_steepening && workspace.nonlinear_active
                residual_rhs_evaluations += 2
                nonlinear_rhs_evaluations += 2
                temporal_nonlinear_fft_pairs += 2
                self_steepening_fft_pairs += 2
                raman_fft_pairs += 2 * workspace.raman_active
            else
                raman_fft_pairs += workspace.raman_active
            end

            is_save = save_index <= length(saved) &&
                      abs(z1 - saved[save_index]) <= tolerance
            if is_save
                _cuda_full_field_linear!(workspace, local_step / 2)
                linear_fft_pairs += 1
                @views _cuda_full_field_copy_to_host!(
                    fields[:, :, :, save_index], workspace.field)
                save_index += 1
                if index < nsteps
                    next_step = steps[index + 2] - steps[index + 1]
                    _cuda_full_field_linear!(workspace, next_step / 2)
                    linear_fft_pairs += 1
                end
            else
                next_step = steps[index + 2] - steps[index + 1]
                _cuda_full_field_linear!(
                    workspace, (local_step + next_step) / 2)
                linear_fft_pairs += 1
            end
        end
        joint_fft_transforms = 2 * linear_fft_pairs
    elseif method === :composed_strang4
        cube_root_four = cbrt(4.0)
        outer = 1 / (4 - cube_root_four)
        middle = 1 - 4outer
        first_step = steps[2] - steps[1]
        _cuda_full_field_linear!(workspace, outer * first_step / 2)
        linear_fft_pairs += 1

        for index in 1:nsteps
            z0, z1 = steps[index], steps[index + 1]
            local_step = z1 - z0

            _cuda_full_field_symmetric_residual!(
                workspace, outer * local_step)
            _cuda_full_field_linear!(
                workspace, outer * local_step)
            _cuda_full_field_symmetric_residual!(
                workspace, outer * local_step)
            _cuda_full_field_linear!(
                workspace, (outer + middle) * local_step / 2)
            _cuda_full_field_symmetric_residual!(
                workspace, middle * local_step)
            _cuda_full_field_linear!(
                workspace, (middle + outer) * local_step / 2)
            _cuda_full_field_symmetric_residual!(
                workspace, outer * local_step)
            _cuda_full_field_linear!(
                workspace, outer * local_step)
            _cuda_full_field_symmetric_residual!(
                workspace, outer * local_step)
            linear_fft_pairs += 4
            residual_steps += 5 * workspace.phase_active
            nonlinear_steps += 5 * workspace.nonlinear_active

            if workspace.self_steepening && workspace.nonlinear_active
                # Four shock RHS evaluations per composed residual.
                residual_rhs_evaluations += 20
                nonlinear_rhs_evaluations += 20
                temporal_nonlinear_fft_pairs += 20
                self_steepening_fft_pairs += 20
                # Each RHS evaluation and both exact Raman half phases require
                # one Raman convolution for every composed residual.
                raman_fft_pairs += 30 * workspace.raman_active
            else
                raman_fft_pairs += 5 * workspace.raman_active
            end

            is_save = save_index <= length(saved) &&
                      abs(z1 - saved[save_index]) <= tolerance
            if is_save
                _cuda_full_field_linear!(
                    workspace, outer * local_step / 2)
                linear_fft_pairs += 1
                @views _cuda_full_field_copy_to_host!(
                    fields[:, :, :, save_index], workspace.field)
                save_index += 1
                if index < nsteps
                    next_step = steps[index + 2] - steps[index + 1]
                    _cuda_full_field_linear!(
                        workspace, outer * next_step / 2)
                    linear_fft_pairs += 1
                end
            else
                next_step = steps[index + 2] - steps[index + 1]
                _cuda_full_field_linear!(
                    workspace, outer * (local_step + next_step) / 2)
                linear_fft_pairs += 1
            end
        end
        joint_fft_transforms = 2 * linear_fft_pairs
    else
        mul!(workspace.field, workspace.to_spectral!, workspace.field)
        joint_fft_transforms += 1
        for index in 1:nsteps
            z0, z1 = steps[index], steps[index + 1]
            local_step = z1 - z0
            _cuda_full_field_rk4ip_step!(workspace, local_step)
            if workspace.phase_active
                residual_steps += 1
                residual_rhs_evaluations += 4
                nonlinear_steps += workspace.nonlinear_active
                nonlinear_rhs_evaluations += 4 * workspace.nonlinear_active
                temporal_nonlinear_fft_pairs +=
                    4 * workspace.nonlinear_active
                spatial_fft_pairs += 4
                mixed_fft_calls += 4 * (
                    workspace.nonlinear_active ? 4 : 2)
                raman_fft_pairs += 4 * workspace.raman_active
            end

            is_save = save_index <= length(saved) &&
                      abs(z1 - saved[save_index]) <= tolerance
            if is_save
                @views _cuda_full_field_copy_rk4ip_to_host!(
                    fields[:, :, :, save_index], workspace)
                joint_fft_transforms += 1
                save_index += 1
            end
        end
    end

    save_index == length(saved) + 1 || error(
        "Not every requested full-field plane was saved.")
    synchronize && CUDA.synchronize()
    return PulsePropagation.FullField3DSolution(
        z=Float64.(saved), fields=fields, parameters=parameters,
        initial_field=CT.(initial_field), dz=step,
        cache=(
            backend=:cuda,
            precision,
            method,
            self_steepening=parameters.self_steepening,
            device=string(CUDA.device()),
            integration_z=Float64.(steps),
            accepted_steps=nsteps,
            linear_fft_pairs,
            joint_fft_transforms,
            mixed_fft_calls,
            residual_steps,
            residual_rhs_evaluations,
            nonlinear_steps,
            nonlinear_rhs_evaluations,
            raman_fft_pairs,
            temporal_nonlinear_fft_pairs,
            spatial_fft_pairs,
            # Additional temporal FFT pairs caused specifically by the
            # explicit shock subflow in the Strang implementation.
            strang_shock_fft_pairs=self_steepening_fft_pairs,
            self_steepening_fft_pairs,
            workspace_bytes,
            field_bytes=sizeof(CT) * nt * nx * ny,
        ),
    )
end
