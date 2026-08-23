using PulsePropagation
using Random

const ESLAMI_C_M_PER_PS = 2.99792458e-4
const ESLAMI_WAVELENGTH_M = 1700e-9
const ESLAMI_NCORE = 1.885
const ESLAMI_NCLAD = 1.866
const ESLAMI_CORE_RADIUS_M = 40e-6
const ESLAMI_NA = sqrt(ESLAMI_NCORE^2 - ESLAMI_NCLAD^2)
const ESLAMI_MODE_COUNT = 190
const ESLAMI_MAXIMUM_HG_ORDER = 18
const ESLAMI_OMEGA0 = 2pi * ESLAMI_C_M_PER_PS / ESLAMI_WAVELENGTH_M
const ESLAMI_N2_M2_PER_W = 1.92e-19

function eslami_modal_fiber()
    return GRIN(
        lambda0=ESLAMI_WAVELENGTH_M,
        core_radius=ESLAMI_CORE_RADIUS_M,
        NA=ESLAMI_NA,
    )
end

function eslami_mode_labels()
    labels = PulsePropagation._grin_hg_labels(ESLAMI_MODE_COUNT)
    maximum(label.order for label in labels) == ESLAMI_MAXIMUM_HG_ORDER ||
        error("The 190-mode basis does not end at complete HG order 18.")
    return labels
end

function eslami_modal_beta()
    labels = eslami_mode_labels()
    k0 = 2pi / ESLAMI_WAVELENGTH_M
    beta0 = [
        sqrt((k0 * ESLAMI_NCORE)^2 -
             2k0 * ESLAMI_NA / ESLAMI_CORE_RADIUS_M * (label.order + 1))
        for label in labels
    ]
    coefficients = zeros(Float64, 9, ESLAMI_MODE_COUNT)
    coefficients[1, :] .= beta0
    coefficients_si = (
        5.93e-26,
        3.10e-40,
        -7.35e-55,
        2.15e-69,
        -3.60e-84,
        2.84e-99,
        1.35e-115,
    )
    for (index, coefficient_si) in enumerate(coefficients_si)
        order = index + 1
        coefficients[order + 1, :] .= coefficient_si * 1e12^order
    end
    return TaylorBeta(coefficients)
end

function eslami_spectral_loss_modal(time)
    cutoff = 2pi * ESLAMI_C_M_PER_PS / 2800e-9
    transition = 2pi * 10.0
    normalized_shortfall = max.(
        (cutoff .- (ESLAMI_OMEGA0 .+ frequency_axis(time))) ./ transition,
        0.0,
    )
    return 5.0e4 .* min.(normalized_shortfall .^ 8, 1.0)
end

function _eslami_gaussian_hg_1d_coefficients(fiber, maximum_order;
                                              beam_radius=25e-6,
                                              quadrature_order=32)
    quadrature_order >= maximum_order + 1 || throw(ArgumentError(
        "quadrature_order must exceed the maximum projected HG order."))
    nodes, weights = PulsePropagation._gauss_hermite_rule(quadrature_order)
    mode_q = 2pi / fiber.lambda0 * fiber.NA / fiber.core_radius
    input_q = 2 / beam_radius^2
    argument_scale = sqrt(2mode_q / (mode_q + input_q))
    polynomials = zeros(Float64, maximum_order + 1, quadrature_order)
    polynomials[1, :] .= 1
    if maximum_order >= 1
        polynomials[2, :] .= sqrt(2) .* argument_scale .* nodes
    end
    for order in 1:maximum_order-1
        polynomials[order + 2, :] .=
            sqrt(2 / (order + 1)) .* argument_scale .* nodes .*
            polynomials[order + 1, :] .-
            sqrt(order / (order + 1)) .* polynomials[order, :]
    end
    prefactor = (mode_q * input_q)^0.25 / sqrt(pi) *
                sqrt(2 / (mode_q + input_q))
    return prefactor .* vec(polynomials * weights)
end

function eslami_launch_coefficients(; beam_radius=25e-6,
                                    renormalize=true)
    fiber = eslami_modal_fiber()
    labels = eslami_mode_labels()
    one_dimensional = _eslami_gaussian_hg_1d_coefficients(
        fiber, ESLAMI_MAXIMUM_HG_ORDER; beam_radius)
    coefficients = ComplexF64[
        one_dimensional[label.nx + 1] * one_dimensional[label.ny + 1]
        for label in labels
    ]
    capture = sum(abs2, coefficients)
    renormalize && (coefficients ./= sqrt(capture))
    return coefficients, capture
end

function eslami_modal_initial(time;
                              peak_power=5.85e5,
                              seed=20220819,
                              renormalize_spatial=true)
    rng = MersenneTwister(seed)
    pulse_fwhm = 0.350
    temporal_intensity = exp.(
        -4log(2) .* (time_axis(time) ./ pulse_fwhm) .^ 2)
    temporal_intensity .*= max.(0.0, 1 .+ 0.002 .* randn(rng, time.nt))
    coefficients, capture = eslami_launch_coefficients(
        renormalize=renormalize_spatial)
    initial = ComplexF64.(
        sqrt(peak_power) .*
        reshape(sqrt.(temporal_intensity), time.nt, 1, 1) .*
        reshape(coefficients, 1, ESLAMI_MODE_COUNT, 1))
    return initial, capture
end

function eslami_modal_parameters(time, overlap;
                                 propagation_length=0.20,
                                 spectral_filter=true)
    domain = MMGNLSEDomain(
        degrees_of_freedom(:time, :space), time)
    alpha = spectral_filter ? eslami_spectral_loss_modal(time) : 0.0
    return MMGNLSEParameters(
        domain;
        length=propagation_length,
        alpha,
        beta=eslami_modal_beta(),
        S=overlap,
        n2=ESLAMI_N2_M2_PER_W,
        omega0=ESLAMI_OMEGA0,
        raman=NoRaman(),
    )
end

function eslami_target_tag(target)
    exponent = round(Int, -log10(target))
    isapprox(target, 10.0^-exponent; rtol=1e-12) &&
        return "1em$(exponent)"
    return replace(string(target), '.' => 'p', '-' => 'm')
end

