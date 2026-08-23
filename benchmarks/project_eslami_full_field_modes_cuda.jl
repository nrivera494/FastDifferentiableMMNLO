using CUDA
using LinearAlgebra
using NPZ
using PulsePropagation

# Run the same Eslami-like case, leaving `initial`, `final_field`, `geometry`,
# `initial_energy`, and `final_energy` available for projection below.
include(joinpath(@__DIR__, "eslami_like_full_field_cuda.jl"))

"""All scalar guided HG modes of the carrier-frequency parabolic GRIN model."""
function eslami_guided_hg_modes(geometry)
    wavelength = 2pi * C_M_PER_PS / geometry.omega0
    numerical_aperture = sqrt(geometry.ncore^2 - geometry.nclad^2)
    normalized_frequency =
        2pi * geometry.core_radius * numerical_aperture / wavelength

    # For the analytic parabolic GRIN model, total order g is guided when
    # g + 1 < V / 2. Each scalar HG order contains g + 1 modes.
    maximum_order = ceil(Int, normalized_frequency / 2) - 2
    maximum_order >= 0 || error("The carrier supports no guided scalar modes.")
    mode_count = (maximum_order + 1) * (maximum_order + 2) ÷ 2
    labels = PulsePropagation._grin_hg_labels(mode_count)
    fiber = GRIN(
        lambda0=wavelength,
        core_radius=geometry.core_radius,
        NA=numerical_aperture,
    )
    modes = PulsePropagation._grin_fields(
        fiber, nothing, labels, :HG, geometry.grid.x, geometry.grid.y)
    return (; modes, labels, normalized_frequency, maximum_order)
end

function modal_energy_gpu(field, modes_yxm, grid)
    mode_matrix = ComplexF32.(
        PulsePropagation._full_field_mode_matrix(modes_yxm, grid))
    device_modes = CuArray(mode_matrix)
    device_field = CuArray(field)
    flattened = reshape(device_field, grid.time.nt, :)
    amplitudes = flattened * device_modes
    amplitudes .*= Float32(grid.dx * grid.dy)
    energy = Float64.(Array(vec(sum(abs2, amplitudes; dims=1)))) .* grid.time.dt

    amplitudes = nothing
    flattened = nothing
    device_field = nothing
    device_modes = nothing
    CUDA.reclaim()
    return energy, mode_matrix
end

basis = eslami_guided_hg_modes(geometry)
input_modal_energy, mode_matrix =
    modal_energy_gpu(initial, basis.modes, geometry.grid)
output_modal_energy, _ =
    modal_energy_gpu(final_field, basis.modes, geometry.grid)

weight = geometry.grid.dx * geometry.grid.dy
gram = real.(adjoint(mode_matrix) * mode_matrix) .* weight
gram_error = maximum(abs, gram - I)

orders = Int[label.order for label in basis.labels]
nx_labels = Int[label.nx for label in basis.labels]
ny_labels = Int[label.ny for label in basis.labels]
input_fraction_total = input_modal_energy ./ initial_energy
output_fraction_total = output_modal_energy ./ final_energy
input_capture = sum(input_fraction_total)
output_capture = sum(output_fraction_total)

order_axis = collect(0:basis.maximum_order)
input_fraction_by_order = [
    sum(input_fraction_total[orders .== order]) for order in order_axis
]
output_fraction_by_order = [
    sum(output_fraction_total[orders .== order]) for order in order_axis
]

output_guided_distribution = output_modal_energy ./ sum(output_modal_energy)
effective_mode_count = inv(sum(abs2, output_guided_distribution))
ranking = sortperm(output_modal_energy; rev=true)
cumulative_guided = cumsum(output_guided_distribution[ranking])
modes_for_90_percent = findfirst(>=(0.90), cumulative_guided)
modes_for_95_percent = findfirst(>=(0.95), cumulative_guided)
modes_for_99_percent = findfirst(>=(0.99), cumulative_guided)

top_count = min(20, length(ranking))
top_modes = [
    (
        rank=rank,
        index=index,
        nx=nx_labels[index],
        ny=ny_labels[index],
        order=orders[index],
        fraction_total=output_fraction_total[index],
        fraction_guided=output_guided_distribution[index],
    )
    for (rank, index) in enumerate(ranking[1:top_count])
]

projection_diagnostics = (;
    grid=transverse_count,
    mode_count=length(orders),
    maximum_order=basis.maximum_order,
    normalized_frequency=basis.normalized_frequency,
    gram_error,
    input_capture,
    output_capture,
    effective_mode_count,
    modes_for_90_percent,
    modes_for_95_percent,
    modes_for_99_percent,
)
@info "Eslami output modal projection" projection_diagnostics
@info "Eslami modal power by total HG order" order_axis input_fraction_by_order output_fraction_by_order
@info "Eslami leading output modes" top_modes

modal_output_path = get(ENV, "ESLAMI_MODAL_OUTPUT", "")
if !isempty(modal_output_path)
    npzwrite(modal_output_path, Dict(
        "mode_nx" => nx_labels,
        "mode_ny" => ny_labels,
        "mode_order" => orders,
        "input_fraction_total" => input_fraction_total,
        "output_fraction_total" => output_fraction_total,
        "order" => order_axis,
        "input_fraction_by_order" => input_fraction_by_order,
        "output_fraction_by_order" => output_fraction_by_order,
        "input_capture" => [input_capture],
        "output_capture" => [output_capture],
        "effective_mode_count" => [effective_mode_count],
        "modes_for_90_percent_guided" => [modes_for_90_percent],
        "modes_for_95_percent_guided" => [modes_for_95_percent],
        "modes_for_99_percent_guided" => [modes_for_99_percent],
        "gram_max_error" => [gram_error],
        "normalized_frequency" => [basis.normalized_frequency],
    ))
    @info "wrote Eslami modal projection" modal_output_path
end
