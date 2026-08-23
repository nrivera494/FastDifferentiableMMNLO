using LinearAlgebra
using PulsePropagation

# Matched ideal-GRIN configuration for the full-field/CP crossover campaign.
# The quadratic profile is deliberately not clipped at the nominal core radius:
# this keeps the full-field transverse operator and the analytic HG quadrature
# representation of the modal overlap mathematically identical.

const GRIN210_C_M_PER_PS = 2.99792458e-4
const GRIN210_WAVELENGTH_M = 1550e-9
const GRIN210_CORE_RADIUS_M = 67.5e-6
const GRIN210_NA = 0.15
const GRIN210_GROUP_COUNT = 20
const GRIN210_MAXIMUM_HG_ORDER = GRIN210_GROUP_COUNT - 1
const GRIN210_MODE_COUNT =
    GRIN210_GROUP_COUNT * (GRIN210_GROUP_COUNT + 1) ÷ 2
const GRIN210_OMEGA0 =
    2pi * GRIN210_C_M_PER_PS / GRIN210_WAVELENGTH_M
const GRIN210_MATERIAL = Silica()
const GRIN210_N2_M2_PER_W = GRIN210_MATERIAL.n2
const GRIN210_REFERENCE_INDEX =
    silica_refractive_index(GRIN210_WAVELENGTH_M)
const GRIN210_K0 = 2pi / GRIN210_WAVELENGTH_M
const GRIN210_BETA_REFERENCE = GRIN210_K0 * GRIN210_REFERENCE_INDEX
const GRIN210_OSCILLATOR_Q_M2 =
    GRIN210_K0 * GRIN210_NA / GRIN210_CORE_RADIUS_M
const GRIN210_TRANSVERSE_SPACING_M1 =
    GRIN210_OSCILLATOR_Q_M2 / GRIN210_BETA_REFERENCE
const GRIN210_NORMALIZED_FREQUENCY =
    2pi * GRIN210_CORE_RADIUS_M * GRIN210_NA / GRIN210_WAVELENGTH_M

const GRIN210_PEAK_POWER_W = 751_549.7920240875
const GRIN210_FWHM_PS = 0.2499694427126389
const GRIN210_PULSE_ENERGY_NJ =
    GRIN210_PEAK_POWER_W * GRIN210_FWHM_PS * sqrt(pi) /
    (2sqrt(log(2))) / 1e3

const GRIN210_TORCH_PHASES = ComplexF64[
     0.042233154627023345 + 0.9991077822989120im,
    -0.9548483373690423   + 0.29709367651226704im,
     0.8131419320624027   - 0.5820654587946469im,
     0.18628501477914675  + 0.9824957472013469im,
     0.6319768780431223   + 0.7749872422297470im,
    -0.9492435495998746   + 0.31454202190332287im,
    -0.06792170771520316  - 0.9976906542716788im,
     0.3907153673065814   - 0.9205115435183218im,
     0.9992441657975882   - 0.03887283266604082im,
     0.6421650166589653   - 0.7665664298541848im,
]

# Nine points cover the twenty complete groups without turning the initial
# cache build into a needlessly dense sweep. A driver may add every complete
# group inside a measured crossover bracket.
const GRIN210_DEFAULT_MODE_COUNTS =
    (15, 28, 45, 66, 91, 120, 153, 190, 210)
const GRIN210_DEFAULT_TARGET_ERRORS = (1e-2, 1e-3, 1e-4)
const GRIN210_WORKING_QUADRATURE_ORDER = 48
const GRIN210_AUTHORITY_QUADRATURE_ORDER = 64

function grin210_fiber()
    return GRIN(
        lambda0=GRIN210_WAVELENGTH_M,
        core_radius=GRIN210_CORE_RADIUS_M,
        NA=GRIN210_NA,
    )
end

grin210_mode_count(group_count::Integer) = begin
    groups = Int(group_count)
    1 <= groups <= GRIN210_GROUP_COUNT || throw(ArgumentError(
        "group_count must lie in 1:$(GRIN210_GROUP_COUNT)."))
    groups * (groups + 1) ÷ 2
end

function grin210_complete_group_count(mode_count::Integer)
    modes = Int(mode_count)
    for groups in 1:GRIN210_GROUP_COUNT
        grin210_mode_count(groups) == modes && return groups
    end
    throw(ArgumentError(
        "$modes is not a complete-group scalar HG mode count up to " *
        "$GRIN210_MODE_COUNT."))
end

function grin210_mode_labels(mode_count::Integer=GRIN210_MODE_COUNT)
    groups = grin210_complete_group_count(mode_count)
    labels = PulsePropagation._grin_hg_labels(Int(mode_count))
    maximum(label.order for label in labels) == groups - 1 || error(
        "HG enumeration did not end at complete order $(groups - 1).")
    return labels
end

function grin210_torch_coefficients(mode_count::Integer=GRIN210_MODE_COUNT)
    modes = Int(mode_count)
    modes >= 15 || throw(ArgumentError(
        "The modes 6:15 Torch launch requires at least 15 retained modes."))
    coefficients = zeros(ComplexF64, modes)
    coefficients[6:15] .= GRIN210_TORCH_PHASES
    coefficients ./= norm(coefficients)
    return coefficients
end

function grin210_modal_initial(time::TimeGrid,
                               mode_count::Integer=GRIN210_MODE_COUNT)
    return gaussian_pulse(
        time, Int(mode_count);
        peak_power=GRIN210_PEAK_POWER_W,
        fwhm=GRIN210_FWHM_PS,
        coefficients=grin210_torch_coefficients(mode_count),
        time_offset=0.0,
    )
end

function grin210_sampled_modes(grid::FullField3DGrid,
                               mode_count::Integer=GRIN210_MODE_COUNT)
    labels = grin210_mode_labels(mode_count)
    return PulsePropagation._grin_fields(
        grin210_fiber(), nothing, labels, :HG, grid.x, grid.y)
end

function grin210_full_field_initial(grid::FullField3DGrid;
                                    mode_count::Integer=GRIN210_MODE_COUNT)
    modal = grin210_modal_initial(grid.time, mode_count)
    modes = grin210_sampled_modes(grid, mode_count)
    return reconstruct_full_field(@view(modal[:, :, 1]), modes, grid), modes
end

function grin210_ideal_potential(grid::FullField3DGrid)
    rate = GRIN210_TRANSVERSE_SPACING_M1
    return [
        -0.5 * GRIN210_BETA_REFERENCE * rate^2 * (x^2 + y^2)
        for x in grid.x, y in grid.y
    ]
end

function grin210_bulk_taylor_coefficients(;
                                           beta_order::Integer=5,
                                           dispersion_span::Real=35.0,
                                           dispersion_samples::Integer=max(
                                               13, 2Int(beta_order) + 3),
                                           dispersion_fit_order::Integer=min(
                                               12, Int(dispersion_samples) - 1))
    offsets, omega0 = PulsePropagation._dispersion_offsets(
        GRIN210_WAVELENGTH_M, Int(beta_order), dispersion_span,
        Int(dispersion_samples))
    values = Matrix{Float64}(undef, length(offsets), 1)
    for (index, offset) in pairs(offsets)
        omega = omega0 + offset
        wavelength = 2pi * GRIN210_C_M_PER_PS / omega
        values[index, 1] = omega / GRIN210_C_M_PER_PS *
                           silica_refractive_index(wavelength)
    end
    beta = PulsePropagation._fit_taylor_beta(
        offsets, values, Int(beta_order), Int(dispersion_fit_order))
    return vec(beta.coefficients)
end

function grin210_modal_beta(mode_count::Integer=GRIN210_MODE_COUNT; kwargs...)
    labels = grin210_mode_labels(mode_count)
    bulk = grin210_bulk_taylor_coefficients(; kwargs...)
    coefficients = repeat(reshape(bulk, :, 1), 1, Int(mode_count))
    # The transverse eigenvalues are carrier-frequency constants in the
    # full-field paraxial model. All higher dispersion orders remain common.
    for (mode, label) in pairs(labels)
        coefficients[1, mode] +=
            -GRIN210_TRANSVERSE_SPACING_M1 * (label.order + 1)
    end
    return TaylorBeta(coefficients)
end

function grin210_temporal_dispersion(time::TimeGrid; kwargs...)
    coefficients = grin210_bulk_taylor_coefficients(; kwargs...)
    omega = frequency_axis(time)
    dispersion = zeros(Float64, time.nt)
    for order in 2:length(coefficients)-1
        dispersion .+= coefficients[order + 1] .* omega .^ order ./
                      factorial(order)
    end
    return dispersion
end

function grin210_full_field_parameters(grid::FullField3DGrid;
                                       length,
                                       self_steepening::Bool=true,
                                       raman=GRIN210_MATERIAL.raman,
                                       n2::Real=GRIN210_N2_M2_PER_W,
                                       kwargs...)
    return FullField3DParameters(
        grid;
        length,
        beta0=GRIN210_BETA_REFERENCE,
        dispersion=grin210_temporal_dispersion(grid.time; kwargs...),
        potential=grin210_ideal_potential(grid),
        n2,
        omega0=GRIN210_OMEGA0,
        raman,
        self_steepening,
    )
end

function grin210_modal_parameters(time::TimeGrid, overlap;
                                  length,
                                  raman=GRIN210_MATERIAL.raman,
                                  n2::Real=GRIN210_N2_M2_PER_W,
                                  kwargs...)
    modes = overlap.metadata.nmodes
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), time)
    return MMGNLSEParameters(
        domain;
        length,
        beta=grin210_modal_beta(modes; kwargs...),
        S=overlap,
        n2,
        omega0=GRIN210_OMEGA0,
        raman,
    )
end

function grin210_quadrature_source(mode_count::Integer;
                                   quadrature_order::Integer=
                                       GRIN210_WORKING_QUADRATURE_ORDER,
                                   precision_bits::Integer=256)
    grin210_complete_group_count(mode_count)
    return grin_hg_overlap_quadrature_cp(
        grin210_fiber(), Int(mode_count);
        quadrature_order=Int(quadrature_order),
        precision_bits=Int(precision_bits),
    )
end

function grin210_target_tag(target::Real)
    value = Float64(target)
    exponent = round(Int, -log10(value))
    isapprox(value, 10.0^-exponent; rtol=1e-12, atol=0) &&
        return "1em$(exponent)"
    return replace(string(value), '.' => 'p', '-' => 'm')
end

function grin210_cp_cache_paths(directory::AbstractString,
                                mode_count::Integer,
                                target::Real)
    prefix = "grin_$(lpad(Int(mode_count), 3, '0'))_cp_" *
             grin210_target_tag(target)
    return (
        data=joinpath(directory, prefix * ".jld2"),
        metadata=joinpath(directory, prefix * ".toml"),
    )
end

function grin210_configuration()
    return (;
        wavelength_m=GRIN210_WAVELENGTH_M,
        core_radius_m=GRIN210_CORE_RADIUS_M,
        numerical_aperture=GRIN210_NA,
        normalized_frequency=GRIN210_NORMALIZED_FREQUENCY,
        group_count=GRIN210_GROUP_COUNT,
        maximum_hg_order=GRIN210_MAXIMUM_HG_ORDER,
        mode_count=GRIN210_MODE_COUNT,
        beta_reference_m_inv=GRIN210_BETA_REFERENCE,
        transverse_spacing_m_inv=GRIN210_TRANSVERSE_SPACING_M1,
        peak_power_w=GRIN210_PEAK_POWER_W,
        pulse_fwhm_ps=GRIN210_FWHM_PS,
        pulse_energy_nj=GRIN210_PULSE_ENERGY_NJ,
        working_quadrature_order=GRIN210_WORKING_QUADRATURE_ORDER,
        authority_quadrature_order=GRIN210_AUTHORITY_QUADRATURE_ORDER,
    )
end

GRIN210_MODE_COUNT == 210 || error("Expected twenty HG groups to contain 210 modes.")
GRIN210_NORMALIZED_FREQUENCY / 2 > GRIN210_GROUP_COUNT || error(
    "The configured GRIN fiber does not guide twenty complete groups.")
GRIN210_NORMALIZED_FREQUENCY / 2 <= GRIN210_GROUP_COUNT + 1 || error(
    "The configured GRIN fiber also guides the twenty-first group.")
