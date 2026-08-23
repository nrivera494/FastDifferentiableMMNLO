"""
    FullField3DGrid(time, x, y)

Uniform experimental `time x x x y` grid for the GPU-only full-field solver.
Time is stored in the package's usual picosecond units and the transverse axes
are stored in metres. The public field axis order is `(time, x, y)` so every
temporal trace is contiguous in Julia's column-major storage.
"""
struct FullField3DGrid{T<:Real,G<:TimeGrid}
    time::G
    x::Vector{T}
    y::Vector{T}
    dx::T
    dy::T
end

function _full_field_uniform_spacing(values::AbstractVector{<:Real}, name)
    return _mmgnlse_uniform_spacing(values, name)
end

function FullField3DGrid(time::TimeGrid,
                         x::AbstractVector{<:Real},
                         y::AbstractVector{<:Real})
    T = promote_type(float(eltype(x)), float(eltype(y)))
    x_values = T.(x)
    y_values = T.(y)
    dx = T(_full_field_uniform_spacing(x_values, :x))
    dy = T(_full_field_uniform_spacing(y_values, :y))
    return FullField3DGrid{T,typeof(time)}(
        time, x_values, y_values, dx, dy)
end

function FullField3DGrid(time::TimeGrid;
                         Nx::Integer,
                         Ny::Integer=Nx,
                         dx,
                         dy=dx)
    Nx >= 2 || throw(ArgumentError("Nx must be at least two."))
    Ny >= 2 || throw(ArgumentError("Ny must be at least two."))
    dx_value = float(dx)
    dy_value = float(dy)
    isfinite(dx_value) && dx_value > 0 || throw(ArgumentError(
        "dx must be finite and positive."))
    isfinite(dy_value) && dy_value > 0 || throw(ArgumentError(
        "dy must be finite and positive."))
    x = centered_time_grid(Int(Nx), dx_value)
    y = centered_time_grid(Int(Ny), dy_value)
    return FullField3DGrid(time, x, y)
end

FullField3DGrid(time::TimeGrid, transverse::TransverseGrid) =
    FullField3DGrid(time, transverse.x, transverse.y)

full_field_shape(grid::FullField3DGrid) =
    (grid.time.nt, length(grid.x), length(grid.y))

"""
    FullField3DParameters(grid; length, beta0, dispersion, spectral_loss, potential,
                         n2, omega0, raman=NoRaman(), self_steepening=false)

Parameters for the experimental scalar paraxial 3+1D equation. `dispersion`
is the real temporal propagation-constant offset in `m^-1`, supplied in the
public centered frequency order. `spectral_loss` is an optional nonnegative
power-attenuation coefficient in `m^-1` on the same centered grid, so its field
propagator is `exp(-spectral_loss*z/2)`. `potential` is the real
carrier-frequency index potential in `m^-1` with axes `(x, y)`. The nonlinear
coefficient is `omega0*n2/c`. Set `self_steepening=true` to apply the same
frequency-dependent nonlinear prefactor `(omega0 + omega)*n2/c` and edge
damping used by the modal solver. The index potential remains a distinct linear
residual and is never multiplied by the self-steepening prefactor. Transverse
boundaries are periodic, as required by the Fourier split-step method.
"""
struct FullField3DParameters{G,T<:Real,R<:AbstractRamanResponse}
    grid::G
    length::T
    beta0::T
    dispersion::Vector{T}
    spectral_loss::Vector{T}
    potential::Matrix{T}
    n2::T
    omega0::T
    raman::R
    self_steepening::Bool
end

function FullField3DParameters(grid::FullField3DGrid;
                               length,
                               beta0,
                               dispersion=zeros(grid.time.nt),
                               spectral_loss=zeros(grid.time.nt),
                               potential=zeros(Base.length(grid.x),
                                               Base.length(grid.y)),
                               n2=0,
                               omega0,
                               raman=NoRaman(),
                               self_steepening::Bool=false)
    T = promote_type(
        typeof(float(length)), typeof(float(beta0)),
        float(eltype(dispersion)), float(eltype(spectral_loss)),
        float(eltype(potential)),
        typeof(float(n2)), typeof(float(omega0)))
    length_value = T(length)
    beta0_value = T(beta0)
    n2_value = T(n2)
    omega0_value = T(omega0)
    isfinite(length_value) && length_value > 0 || throw(ArgumentError(
        "Full-field length must be finite and positive (metres)."))
    isfinite(beta0_value) && beta0_value > 0 || throw(ArgumentError(
        "beta0 must be finite and positive (m^-1)."))
    isfinite(n2_value) && n2_value >= 0 || throw(ArgumentError(
        "n2 must be finite and nonnegative (m^2/W)."))
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "omega0 must be finite and positive (rad/ps)."))
    raman isa AbstractRamanResponse || throw(ArgumentError(
        "raman must be NoRaman(), AgarwalRaman(...), or AnisotropicRaman(...)."))

    dispersion_values = T.(collect(dispersion))
    Base.length(dispersion_values) == grid.time.nt || throw(DimensionMismatch(
        "dispersion must contain Nt=$(grid.time.nt) centered samples."))
    all(isfinite, dispersion_values) || throw(ArgumentError(
        "dispersion samples must be finite."))
    spectral_loss_values = T.(collect(spectral_loss))
    Base.length(spectral_loss_values) == grid.time.nt ||
        throw(DimensionMismatch(
            "spectral_loss must contain Nt=$(grid.time.nt) centered samples."))
    all(value -> isfinite(value) && value >= 0, spectral_loss_values) ||
        throw(ArgumentError(
            "spectral_loss samples must be finite and nonnegative."))
    potential_values = T.(Array(potential))
    size(potential_values) == (Base.length(grid.x), Base.length(grid.y)) ||
        throw(DimensionMismatch(
            "potential must have axes (x, y) and size " *
            "$((Base.length(grid.x), Base.length(grid.y))); " *
            "got $(size(potential_values))."))
    all(isfinite, potential_values) || throw(ArgumentError(
        "potential samples must be finite."))

    return FullField3DParameters{typeof(grid),T,typeof(raman)}(
        grid, length_value, beta0_value, dispersion_values, spectral_loss_values,
        potential_values, n2_value, omega0_value, raman, self_steepening)
end

"""Second-order symmetric split-step method for the full-field solver."""
struct StrangSplit end

"""
    ComposedStrang4()

Experimental fourth-order symmetric composition of the full-field split-step
map. Diffraction, temporal dispersion, and spectral loss use their diagonal
spectral propagator. The index potential and ordinary Kerr--Raman response use
their exact real-space phase flow. When self-steepening is active, only the
remaining shock correction is advanced by an iterated implicit-midpoint map.

The outer composition is Suzuki's five-stage fourth-order fractal. Its middle
stage is negative, but its magnitude is substantially smaller than the
three-stage triple jump; dissipative problems should still be checked for
transient high-frequency amplification.
"""
struct ComposedStrang4 end

function _full_field_scalar_beta(beta::AbstractBeta)
    values = beta isa TaylorBeta ? beta.coefficients : beta.values
    size(values, 2) == 1 || throw(DimensionMismatch(
        "Full-field beta must have a singleton mode axis."))
    size(values, 3) == 1 || throw(DimensionMismatch(
        "Full-field beta must have a singleton polarization axis."))
    return beta
end

function _full_field_reference_beta(beta::TaylorBeta)
    return float(beta.coefficients[1])
end

function _full_field_reference_beta(beta::SampledBeta)
    values = reshape(beta.values, size(beta.values, 1), :)
    beta0, _ = _mmgnlse_reference_line(beta.omega, @view(values[:, 1]))
    return float(beta0)
end

function _full_field_spectral_profile(value::Real, nt::Integer, name)
    isfinite(value) || throw(ArgumentError("$name must be finite."))
    value >= 0 || throw(ArgumentError("$name must be nonnegative."))
    return fill(float(value), Int(nt))
end

function _full_field_spectral_profile(value::AbstractVector{<:Real},
                                      nt::Integer, name)
    length(value) == nt || throw(DimensionMismatch(
        "$name must be a scalar or contain Nt=$nt centered samples."))
    all(entry -> isfinite(entry) && entry >= 0, value) ||
        throw(ArgumentError("$name samples must be finite and nonnegative."))
    return float.(collect(value))
end

function _full_field_spectral_profile(value, nt::Integer, name)
    throw(ArgumentError(
        "$name must be a nonnegative real scalar or real vector; got $(typeof(value))."))
end

function _full_field_potential(value::Real, nx::Integer, ny::Integer)
    isfinite(value) || throw(ArgumentError("potential must be finite."))
    return fill(float(value), Int(nx), Int(ny))
end

function _full_field_potential(value::AbstractMatrix{<:Real},
                               nx::Integer, ny::Integer)
    size(value) == (nx, ny) || throw(DimensionMismatch(
        "potential must have axes (x, y) and size $((nx, ny)); got $(size(value))."))
    all(isfinite, value) || throw(ArgumentError(
        "potential samples must be finite."))
    return float.(Array(value))
end

function _full_field_potential(value, nx::Integer, ny::Integer)
    throw(ArgumentError(
        "potential must be a real scalar or an (x, y) real matrix; got $(typeof(value))."))
end

"""
    FullFieldMMGNLSEParameters(domain; length, alpha=0, gain=0, beta,
                               potential=0, n2, omega0, raman=NoRaman(),
                               self_steepening=true, reference_index=nothing)

Physical parameters for the scalar Cartesian full-field MMGNLSE. The domain
must activate `:time` and `:space`, supply a `TransverseGrid`, and omit
`:polarization`. `beta` is the propagation constant of the homogeneous
reference medium and must have singleton mode and polarization axes. Its
carrier value supplies the paraxial diffraction reference, while the temporal
operator removes the reference `beta0 + beta1*omega` line exactly as in the
modal solver. `alpha` and `gain` are longitudinally invariant power-rate
scalars or centered frequency vectors in `m^-1`. `potential` has axes `(x, y)`
and units `m^-1`.

By default the reference index is inferred as `beta0*c/omega0`. An explicit
`reference_index` is useful when `beta` contains only relative dispersion; if
`beta` also contains a positive beta0, the two references must agree.
"""
struct FullFieldMMGNLSEParameters{D,T<:Real,A,G,B,P,R}
    domain::D
    length::T
    alpha::A
    gain::G
    beta::B
    reference_beta::T
    reference_index::T
    potential::P
    n2::T
    omega0::T
    raman::R
    self_steepening::Bool
end

function FullFieldMMGNLSEParameters(
    domain::MMGNLSEDomain;
    length,
    alpha=0,
    gain=0,
    beta,
    potential=0,
    n2,
    omega0,
    raman=NoRaman(),
    self_steepening::Bool=true,
    reference_index=nothing,
)
    has_dof(domain.dofs, :time) || throw(ArgumentError(
        "Full-field MMGNLSE parameters require the :time degree of freedom."))
    has_dof(domain.dofs, :space) || throw(ArgumentError(
        "Full-field MMGNLSE parameters require the :space degree of freedom."))
    has_dof(domain.dofs, :polarization) && throw(ArgumentError(
        "The full-field MMGNLSE solver does not yet support polarization."))
    domain.time_grid isa TimeGrid || throw(ArgumentError(
        "Full-field MMGNLSE parameters require a TimeGrid."))
    transverse = domain.transverse_grid
    transverse isa TransverseGrid || throw(ArgumentError(
        "Full-field MMGNLSE parameters require domain.transverse_grid."))

    length_value = float(length)
    n2_value = float(n2)
    omega0_value = float(omega0)
    isfinite(length_value) && length_value > 0 || throw(ArgumentError(
        "Full-field length must be finite and positive (metres)."))
    isfinite(n2_value) && n2_value >= 0 || throw(ArgumentError(
        "n2 must be finite and nonnegative (m^2/W)."))
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "omega0 must be finite and positive (rad/ps)."))
    raman isa AbstractRamanResponse || throw(ArgumentError(
        "raman must be NoRaman(), AgarwalRaman(...), or AnisotropicRaman(...)."))

    beta_value = _full_field_scalar_beta(_coerce_beta(beta))
    _validate_beta(beta_value, domain, 1)
    beta0 = _full_field_reference_beta(beta_value)
    isfinite(beta0) || throw(ArgumentError(
        "The carrier beta0 extracted from beta must be finite."))
    c_m_per_ps = 2.99792458e-4
    reference_beta = if reference_index === nothing
        beta0 > 0 || throw(ArgumentError(
            "Full-field beta must contain a positive beta0 unless reference_index is supplied."))
        beta0
    else
        index_value = float(reference_index)
        isfinite(index_value) && index_value > 0 || throw(ArgumentError(
            "reference_index must be finite and positive."))
        candidate = index_value * omega0_value / c_m_per_ps
        beta0 > 0 && !isapprox(beta0, candidate; rtol=1e-6, atol=0) &&
            throw(ArgumentError(
                "reference_index implies beta0=$candidate m^-1, inconsistent with beta0=$beta0 m^-1 from beta."))
        candidate
    end
    reference_index_value = reference_beta * c_m_per_ps / omega0_value

    nt = domain.time_grid.nt
    nx, ny = Base.length(transverse.x), Base.length(transverse.y)
    alpha_values = _full_field_spectral_profile(alpha, nt, :alpha)
    gain_values = _full_field_spectral_profile(gain, nt, :gain)
    potential_values = _full_field_potential(potential, nx, ny)
    T = promote_type(typeof(length_value), eltype(alpha_values),
                     eltype(gain_values), typeof(reference_beta),
                     eltype(potential_values), typeof(n2_value),
                     typeof(omega0_value))
    return FullFieldMMGNLSEParameters{
        typeof(domain),T,Vector{T},Vector{T},typeof(beta_value),Matrix{T},
        typeof(raman)}(
        domain, T(length_value), T.(alpha_values), T.(gain_values), beta_value,
        T(reference_beta), T(reference_index_value), T.(potential_values),
        T(n2_value), T(omega0_value), raman, self_steepening)
end

FullFieldMMGNLSEParameters(; domain, kwargs...) =
    FullFieldMMGNLSEParameters(domain; kwargs...)

function full_field_shape(domain::MMGNLSEDomain)
    transverse = domain.transverse_grid
    transverse isa TransverseGrid || throw(ArgumentError(
        "A full-field shape requires domain.transverse_grid."))
    nt = has_dof(domain.dofs, :time) ? domain.time_grid.nt : 1
    np = has_dof(domain.dofs, :polarization) ? 2 : 1
    return (nt, length(transverse.x), length(transverse.y), np)
end

full_field_shape(parameters::FullFieldMMGNLSEParameters) =
    full_field_shape(parameters.domain)

"""
    paraxial_index_potential(grid, index_yx; beta0, omega0,
                             reference_index=nothing)

Convert a refractive-index profile with the fiber pipeline's `(y, x)` axis
order into the `(x, y)` carrier-frequency paraxial potential used by
`FullField3DParameters`. A constant reference contribution changes only the
global phase; by default it is chosen consistently with `beta0`.
"""
function paraxial_index_potential(grid::FullField3DGrid,
                                  index_yx::AbstractMatrix{<:Real};
                                  beta0,
                                  omega0,
                                  reference_index=nothing)
    expected = (length(grid.y), length(grid.x))
    size(index_yx) == expected || throw(DimensionMismatch(
        "index_yx must have fiber-profile axes (y, x) and size $expected; " *
        "got $(size(index_yx))."))
    c_m_per_ps = 2.99792458e-4
    k0 = float(omega0) / c_m_per_ps
    beta = float(beta0)
    nref = reference_index === nothing ? beta / k0 : float(reference_index)
    isfinite(nref) && nref > 0 || throw(ArgumentError(
        "reference_index must be finite and positive."))
    values_yx = (k0^2 / (2beta)) .* (float.(index_yx) .^ 2 .- nref^2)
    return permutedims(values_yx, (2, 1))
end

"""
    paraxial_index_potential(domain, index_yx; beta, omega0,
                             reference_index=nothing)

Construct the carrier-frequency paraxial potential for a public full-field
domain. By default the homogeneous reference is inferred from beta0. The input
index profile follows the fiber convention `(y, x)` and the result follows the
public full-field convention `(x, y)`.
"""
function paraxial_index_potential(
    domain::MMGNLSEDomain,
    index_yx::AbstractMatrix{<:Real};
    beta,
    omega0,
    reference_index=nothing,
)
    transverse = domain.transverse_grid
    transverse isa TransverseGrid || throw(ArgumentError(
        "paraxial_index_potential requires domain.transverse_grid."))
    beta_value = _full_field_scalar_beta(_coerce_beta(beta))
    beta0 = _full_field_reference_beta(beta_value)
    omega0_value = float(omega0)
    isfinite(omega0_value) && omega0_value > 0 || throw(ArgumentError(
        "omega0 must be finite and positive (rad/ps)."))
    c_m_per_ps = 2.99792458e-4
    reference_beta = if reference_index === nothing
        isfinite(beta0) && beta0 > 0 || throw(ArgumentError(
            "beta must contain a positive beta0 unless reference_index is supplied."))
        beta0
    else
        index_value = float(reference_index)
        isfinite(index_value) && index_value > 0 || throw(ArgumentError(
            "reference_index must be finite and positive."))
        candidate = index_value * omega0_value / c_m_per_ps
        beta0 > 0 && !isapprox(beta0, candidate; rtol=1e-6, atol=0) &&
            throw(ArgumentError(
                "reference_index is inconsistent with beta0 extracted from beta."))
        candidate
    end
    grid = FullField3DGrid(domain.time_grid, transverse)
    return paraxial_index_potential(
        grid, index_yx; beta0=reference_beta, omega0=omega0_value,
        reference_index=reference_beta * c_m_per_ps / omega0_value)
end

Base.@kwdef struct FullField3DSolution{T,A,P,I,C}
    z::Vector{T}
    fields::A
    parameters::P
    initial_field::I
    dz::T
    cache::C = nothing
end

function _full_field_z_index(zgrid, z)
    z === :initial && return 1
    z === :final && return length(zgrid)
    z isa Integer && return Int(z)
    _, index = findmin(abs.(zgrid .- z))
    return index
end

function get_temporal_field(solution::FullField3DSolution; z=:final)
    z === :all && return solution.fields
    return solution.fields[:, :, :, _full_field_z_index(solution.z, z)]
end

function get_spectral_field(solution::FullField3DSolution; z=:final)
    field_t = get_temporal_field(solution; z)
    return fftshift(ifft(field_t, 1), 1)
end

"""Spatially integrated power on a single `(time, x, y)` field."""
function full_field_power(field::AbstractArray{<:Number,3},
                          grid::FullField3DGrid)
    size(field) == full_field_shape(grid) || throw(DimensionMismatch(
        "field must have full-field shape $(full_field_shape(grid)); got $(size(field))."))
    return vec(sum(abs2, field; dims=(2, 3))) .* (grid.dx * grid.dy)
end

function _full_field_mode_matrix(modes_yxm, grid::FullField3DGrid)
    expected = (length(grid.y), length(grid.x))
    size(modes_yxm, 1) == expected[1] &&
        size(modes_yxm, 2) == expected[2] || throw(DimensionMismatch(
            "modes must have fiber axes (y, x, mode) with spatial size $expected; " *
            "got $(size(modes_yxm))."))
    modes_xym = permutedims(modes_yxm, (2, 1, 3))
    return reshape(modes_xym, length(grid.x) * length(grid.y),
                   size(modes_yxm, 3))
end

"""Reconstruct an experimental `(time, x, y)` field from modal amplitudes."""
function reconstruct_full_field(amplitudes::AbstractMatrix,
                                modes_yxm::AbstractArray{<:Number,3},
                                grid::FullField3DGrid)
    size(amplitudes, 1) == grid.time.nt || throw(DimensionMismatch(
        "amplitudes must have Nt=$(grid.time.nt) rows."))
    mode_matrix = _full_field_mode_matrix(modes_yxm, grid)
    size(amplitudes, 2) == size(mode_matrix, 2) || throw(DimensionMismatch(
        "amplitude and mode counts differ."))
    field = amplitudes * transpose(mode_matrix)
    return reshape(field, full_field_shape(grid))
end

"""Project an experimental `(time, x, y)` field onto sampled carrier modes."""
function project_full_field(field::AbstractArray{<:Number,3},
                            modes_yxm::AbstractArray{<:Number,3},
                            grid::FullField3DGrid)
    size(field) == full_field_shape(grid) || throw(DimensionMismatch(
        "field must have full-field shape $(full_field_shape(grid)); got $(size(field))."))
    mode_matrix = _full_field_mode_matrix(modes_yxm, grid)
    flattened = reshape(field, grid.time.nt, :)
    return (grid.dx * grid.dy) .* (flattened * conj.(mode_matrix))
end

function full_field_subspace_leakage(field::AbstractArray{<:Number,3},
                                     modes_yxm::AbstractArray{<:Number,3},
                                     grid::FullField3DGrid)
    amplitudes = project_full_field(field, modes_yxm, grid)
    projected = reconstruct_full_field(amplitudes, modes_yxm, grid)
    total = sum(abs2, field) * grid.dx * grid.dy
    retained = sum(abs2, projected) * grid.dx * grid.dy
    total > 0 || return 0.0
    return max(0.0, 1 - real(retained / total))
end

"""Global-phase-insensitive relative L2 error."""
function phase_invariant_relative_error(candidate, reference)
    size(candidate) == size(reference) || throw(DimensionMismatch(
        "candidate and reference shapes differ."))
    denominator = norm(vec(reference))
    denominator > 0 || return norm(vec(candidate))
    overlap = dot(vec(reference), vec(candidate))
    phase = iszero(overlap) ? one(overlap) : overlap / abs(overlap)
    return norm(vec(candidate .- phase .* reference)) / denominator
end

function _validate_full_field_initial(initial_field,
                                      parameters::FullField3DParameters)
    size(initial_field) == full_field_shape(parameters.grid) ||
        throw(DimensionMismatch(
            "initial_field must have axes (time, x, y) and size " *
            "$(full_field_shape(parameters.grid)); got $(size(initial_field))."))
    all(isfinite, initial_field) || throw(ArgumentError(
        "initial_field contains a non-finite value."))
    return initial_field
end

function _full_field_raman_spectrum(parameters::FullField3DParameters,
                                    ::Type{T}) where {T<:AbstractFloat}
    nt = parameters.grid.time.nt
    parameters.raman isa NoRaman && return Complex{T}[]
    dt = T(parameters.grid.time.dt)
    lag = dt .* collect(T, 0:nt-1)
    response = raman_response(parameters.raman, lag)
    kernel = T.(response.ha .+ response.hb)
    return Complex{T}.(rfft(kernel)) .* dt
end

function _full_field_wavenumbers(count::Integer, spacing)
    return ifft_frequency_grid(Int(count), spacing)
end

function _solve_full_field_3d_cuda(args...; kwargs...)
    error("The experimental full-field solver requires loading CUDA.jl and a functional CUDA device.")
end

"""
    solve_full_field_3d(initial_field, parameters, dz;
                        backend=:cuda, precision=:float32, saveat=nothing,
                        method=:strang)

Run the experimental GPU-only scalar 3+1D solver. `method=:strang` selects the
second-order split-step method; when self-steepening is active its nonlinear
subflow is advanced with an explicit midpoint rule. `method=:rk4ip` selects a
fourth-order interaction-picture method whose state is stored in joint
`(omega, kx, ky)` space between nonlinear evaluations.
`method=:composed_strang4` selects the five-stage fourth-order symmetric
composition used by `ComposedStrang4()`. FFT plans, propagators, and Raman work
arrays are created before propagation. The returned field axes
are `(time, x, y, z)`. `precision` may be `:float32` or `:float64`.

Because RK4IP advances the index potential explicitly, its pure-potential
imaginary-axis stability condition is checked before GPU allocation:
`dz * maximum(abs, potential) <= 2sqrt(2)`. This is a necessary (not
sufficient) bound; strong nonlinear dynamics can require a smaller step.
"""
function solve_full_field_3d(initial_field,
                             parameters::FullField3DParameters,
                             dz;
                             backend::Symbol=:cuda,
                             precision::Symbol=:float32,
                             saveat=nothing,
                             device=nothing,
                             synchronize::Bool=true,
                             method::Symbol=:strang)
    _validate_full_field_initial(initial_field, parameters)
    step = float(dz)
    isfinite(step) && step > 0 || throw(ArgumentError(
        "dz must be finite and positive."))
    backend === :cuda || throw(ArgumentError(
        "The experimental full-field solver currently supports only backend=:cuda."))
    precision in (:float32, :float64) || throw(ArgumentError(
        "precision must be :float32 or :float64."))
    method in (:strang, :rk4ip, :composed_strang4) || throw(ArgumentError(
        "method must be :strang, :rk4ip, or :composed_strang4."))
    if method === :rk4ip
        maximum_step = min(step, float(parameters.length))
        potential_scale = maximum(abs, parameters.potential)
        stability_argument = maximum_step * potential_scale
        stability_limit = 2sqrt(2.0)
        stability_argument <= stability_limit || throw(ArgumentError(
            "RK4IP treats the index potential explicitly, but " *
            "min(dz, length)*maximum(abs, potential)=" *
            "$(stability_argument) exceeds the classical RK4 imaginary-axis " *
            "stability limit 2sqrt(2)=$(stability_limit). Reduce dz to at " *
            "most $(stability_limit / potential_scale) m; strong nonlinear " *
            "dynamics may require a smaller value."))
    end
    return _solve_full_field_3d_cuda(
        initial_field, parameters, step;
        precision, saveat, device, synchronize, method)
end

"""
Public solution returned by `solve_mmgnlse_full_field`. Fields always have axes
`(time, x, y, polarization, z)`; the currently supported scalar model retains
a singleton polarization axis.
"""
Base.@kwdef struct FullFieldMMGNLSESolution{T,A,P,I,M,C}
    z::Vector{T}
    fields::A
    parameters::P
    initial_field::I
    dz::T
    method::M
    cache::C = nothing
end

function _full_field_public_saved_fields(solution::FullFieldMMGNLSESolution, z)
    z === :all && return solution.fields
    return solution.fields[:, :, :, :, _full_field_z_index(solution.z, z)]
end

function get_temporal_field(solution::FullFieldMMGNLSESolution; z=:final)
    return _full_field_public_saved_fields(solution, z)
end

function get_spectral_field(solution::FullFieldMMGNLSESolution; z=:final)
    return fftshift(ifft(get_temporal_field(solution; z), 1), 1)
end

function get_spatiotemporal_field(solution::FullFieldMMGNLSESolution;
                                  z=:final, domain::Symbol=:time)
    domain === :time && return get_temporal_field(solution; z)
    domain === :frequency && return get_spectral_field(solution; z)
    throw(ArgumentError("domain must be :time or :frequency."))
end

"""
    get_spatial_field(solution; time=nothing, wavelength=nothing,
                      polarization=nothing, z=:final)

Return the complex `(x, y)` full field at the nearest requested time in ps or
absolute wavelength in metres. Exactly one of `time` and `wavelength` must be
provided. `z=:all` retains saved z after y. Polarization may be omitted or set
to `1`; other values are rejected until vector full-field propagation is
implemented.
"""
function get_spatial_field(
    solution::FullFieldMMGNLSESolution;
    time=nothing,
    wavelength=nothing,
    polarization=nothing,
    z=:final,
)
    polarization in (nothing, 1) || throw(ArgumentError(
        "The scalar full-field solution accepts only polarization=nothing or 1."))
    selected_domain, sample = _mmgnlse_spatial_sample(
        solution.parameters; time, wavelength)
    field = selected_domain === :time ?
        get_temporal_field(solution; z) : get_spectral_field(solution; z)
    sampled = selectdim(field, 1, sample)
    return dropdims(sampled; dims=3)
end

function full_field_power(field::AbstractArray{<:Number,4},
                          domain::MMGNLSEDomain)
    size(field) == full_field_shape(domain) || throw(DimensionMismatch(
        "field must have full-field axes (time, x, y, polarization) and size " *
        "$(full_field_shape(domain)); got $(size(field))."))
    transverse = domain.transverse_grid
    return dropdims(sum(abs2, field; dims=(2, 3)); dims=(2, 3)) .*
           (transverse.dx * transverse.dy)
end

function full_field_power(solution::FullFieldMMGNLSESolution; z=:final)
    field = get_temporal_field(solution; z)
    z === :all && throw(ArgumentError(
        "full_field_power currently requires a single saved z plane."))
    return full_field_power(field, solution.parameters.domain)
end

function _full_field_temporal_dispersion(parameters::FullFieldMMGNLSEParameters)
    time = parameters.domain.time_grid
    omega_raw = ifft_frequency_grid(time.nt, time.dt)
    operator_raw = _mmgnlse_beta_array(
        parameters.beta, time.nt, 1, 1, omega_raw)
    return fftshift(vec(imag.(operator_raw)), 1)
end

function _full_field_legacy_parameters(
    parameters::FullFieldMMGNLSEParameters,
)
    transverse = parameters.domain.transverse_grid
    grid = FullField3DGrid(parameters.domain.time_grid, transverse)
    dispersion = _full_field_temporal_dispersion(parameters)
    net_loss = parameters.alpha .- parameters.gain
    T = promote_type(typeof(parameters.length), eltype(dispersion),
                     eltype(net_loss), eltype(parameters.potential))
    # The experimental constructor deliberately accepts attenuation only.
    # The public facade supports gain as well, so construct the already
    # validated internal representation with the signed net power loss.
    return FullField3DParameters{typeof(grid),T,typeof(parameters.raman)}(
        grid, T(parameters.length), T(parameters.reference_beta),
        T.(dispersion), T.(net_loss), T.(parameters.potential),
        T(parameters.n2), T(parameters.omega0), parameters.raman,
        parameters.self_steepening)
end

function _validate_public_full_field_initial(
    initial_field,
    parameters::FullFieldMMGNLSEParameters,
)
    expected = full_field_shape(parameters)
    ndims(initial_field) == 4 || throw(DimensionMismatch(
        "initial_field must retain axes (time, x, y, polarization); got " *
        "$(ndims(initial_field)) dimensions."))
    size(initial_field) == expected || throw(DimensionMismatch(
        "initial_field must have axes (time, x, y, polarization) and size " *
        "$expected; got $(size(initial_field))."))
    all(isfinite, initial_field) || throw(ArgumentError(
        "initial_field contains a non-finite value."))
    return initial_field
end

_full_field_method_symbol(::RK4IP) = :rk4ip
_full_field_method_symbol(::StrangSplit) = :strang
_full_field_method_symbol(::ComposedStrang4) = :composed_strang4
function _full_field_method_symbol(method)
    throw(ArgumentError(
        "Full-field method must be RK4IP(), StrangSplit(), or " *
        "ComposedStrang4(); got $(repr(method))."))
end

"""
    solve_mmgnlse_full_field(initial_field, parameters, dz;
                            method=RK4IP(), saveat=nothing,
                            backend=:cuda, device=nothing,
                            synchronize=true, precision=:float64)

Solve the scalar Cartesian full-field MMGNLSE on a CUDA GPU. The public input
has axes `(time, x, y, polarization)` and must retain a singleton polarization
axis. Its normalization is `abs2(A)` in `W/m^2`, so transverse integration
gives instantaneous power in watts. Ordinary arrays are transferred to the
selected GPU as needed. Saved host fields have axes
`(time, x, y, polarization, z)`. `RK4IP()` is the fourth-order
interaction-picture method, `StrangSplit()` is the second-order split-step
method, and `ComposedStrang4()` is the experimental fourth-order symmetric
composition that keeps the index potential in an exact phase subflow.
"""
function solve_mmgnlse_full_field(
    initial_field,
    parameters::FullFieldMMGNLSEParameters,
    dz;
    method=RK4IP(),
    saveat=nothing,
    backend::Symbol=:cuda,
    device=nothing,
    synchronize::Bool=true,
    precision::Symbol=:float64,
)
    field = _validate_public_full_field_initial(initial_field, parameters)
    method_symbol = _full_field_method_symbol(method)
    internal_parameters = _full_field_legacy_parameters(parameters)
    scalar_field = selectdim(field, 4, 1)
    internal_solution = solve_full_field_3d(
        scalar_field, internal_parameters, dz;
        backend, precision, saveat, device, synchronize, method=method_symbol)
    nt, nx, ny, nz = size(internal_solution.fields)
    fields = reshape(internal_solution.fields, nt, nx, ny, 1, nz)
    initial = reshape(Array(internal_solution.initial_field), nt, nx, ny, 1)
    cache = merge(internal_solution.cache,
                  (public_api=:solve_mmgnlse_full_field,))
    return FullFieldMMGNLSESolution(
        z=internal_solution.z,
        fields=fields,
        parameters=parameters,
        initial_field=initial,
        dz=internal_solution.dz,
        method=method,
        cache=cache,
    )
end
