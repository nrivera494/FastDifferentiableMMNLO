using LinearAlgebra
using NPZ
using Printf
using PyPlot

# Static artifact generator for the matched GRIN-210 benchmark.  This script
# intentionally consumes only benchmark outputs: it never synthesizes or fills
# missing measurements.

const TARGET_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
const TARGET_MARKERS = ("o", "s", "^")
const TARGET_STYLES = ("-", "--", "-.")
const DB_FLOOR = -70.0

flat_array(value) = ndims(Array(value)) == 0 ? [Array(value)[]] : vec(Array(value))

function require_file(path::AbstractString, purpose::AbstractString)
    isfile(path) || error(
        "Missing $purpose at $path. Run the corresponding CUDA benchmark " *
        "with artifact output enabled before plotting.")
    return path
end

function read_npz_checked(path::AbstractString, required_keys)
    data = NPZ.npzread(path)
    missing_keys = filter(key -> !haskey(data, key), collect(required_keys))
    isempty(missing_keys) || error(
        "$path is missing required NPZ keys: $(join(missing_keys, ", ")).")
    return data
end

function finite_nonnegative(values, label)
    array = Float64.(Array(values))
    all(isfinite, array) || error("$label contains non-finite values.")
    all(>=(0), array) || error("$label contains negative values.")
    return array
end

function orient_table(values, nrows, ncols, label)
    table = Array(values)
    size(table) == (nrows, ncols) && return table
    size(table) == (ncols, nrows) && return permutedims(table)
    error("$label has shape $(size(table)); expected ($nrows, $ncols).")
end

function target_tag(target::Real)
    exponent = round(Int, -log10(Float64(target)))
    isapprox(target, 10.0^-exponent; rtol=1e-12, atol=0) &&
        return "1em$(exponent)"
    return replace(string(Float64(target)), '.' => 'p', '-' => 'm')
end

function target_text(target::Real)
    exponent = round(Int, log10(Float64(target)))
    isapprox(target, 10.0^exponent; rtol=1e-12, atol=0) &&
        return "CP target 10^$exponent"
    return @sprintf("CP target %.3g", target)
end

function positive_scale(values, label)
    scale = maximum(values)
    isfinite(scale) && scale > 0 || error("$label has no positive finite scale.")
    return scale
end

function relative_db(values, scale; floor_db=DB_FLOOR)
    floor_ratio = 10.0^(floor_db / 10)
    return 10 .* log10.(max.(Float64.(values) ./ scale, floor_ratio))
end

function spectral_limits(frequency, series; floor_db=DB_FLOOR)
    length(frequency) > 1 || return (first(frequency) - 0.5,
                                    first(frequency) + 0.5)
    scale = maximum(maximum.(series))
    threshold = scale * 10.0^(floor_db / 10)
    active = findall(index -> any(values[index] >= threshold for values in series),
                     eachindex(frequency))
    isempty(active) && return extrema(frequency)
    left = frequency[first(active)]
    right = frequency[last(active)]
    padding = max(0.08 * (right - left), 2abs(frequency[2] - frequency[1]))
    return (max(minimum(frequency), left - padding),
            min(maximum(frequency), right + padding))
end

function style_axis!(axis)
    axis.tick_params(direction="in", top=true, right=true)
    axis.grid(true; color="0.86", linewidth=0.55, alpha=0.8)
    return axis
end

function save_pair(figure, output_dir, stem, artifacts)
    figure.tight_layout()
    for extension in ("png", "pdf")
        path = joinpath(output_dir, "$stem.$extension")
        figure.savefig(path; dpi=240, bbox_inches="tight")
        push!(artifacts, path)
    end
    PyPlot.close(figure)
    return nothing
end

function load_crossover(input_dir)
    path = require_file(joinpath(input_dir, "crossover_timings.npz"),
                        "crossover timing data")
    data = read_npz_checked(path, (
        "mode_counts", "target_errors", "cp_median_ms", "cp_ranks",
        "cp_reported_errors", "full_field_median_ms", "dz_m"))
    modes = Int.(flat_array(data["mode_counts"]))
    targets = Float64.(flat_array(data["target_errors"]))
    issorted(modes) || error("mode_counts in $path must be increasing.")
    all(>(0), modes) || error("mode_counts in $path must be positive.")
    all(target -> 0 < target < 1, targets) || error(
        "target_errors in $path must lie in (0, 1).")
    times = finite_nonnegative(
        orient_table(data["cp_median_ms"], length(modes), length(targets),
                     "cp_median_ms"), "cp_median_ms")
    ranks = Int.(orient_table(data["cp_ranks"], length(modes),
                              length(targets), "cp_ranks"))
    errors = finite_nonnegative(
        orient_table(data["cp_reported_errors"], length(modes),
                     length(targets), "cp_reported_errors"),
        "cp_reported_errors")
    all(>(0), ranks) || error("cp_ranks contains non-positive ranks.")
    full_ms = only(Float64.(flat_array(data["full_field_median_ms"])))
    isfinite(full_ms) && full_ms > 0 || error(
        "full_field_median_ms must be positive and finite.")
    dz_m = only(Float64.(flat_array(data["dz_m"])))
    return (; path, data, modes, targets, times, ranks, errors, full_ms, dz_m)
end

function plot_crossover(crossover, output_dir, artifacts)
    fig, axis = PyPlot.subplots(figsize=(7.1, 4.6))
    for (column, target) in pairs(crossover.targets)
        color = TARGET_COLORS[mod1(column, length(TARGET_COLORS))]
        axis.plot(crossover.modes, crossover.times[:, column];
                  color, marker=TARGET_MARKERS[mod1(column, 3)],
                  linestyle=TARGET_STYLES[mod1(column, 3)], linewidth=1.8,
                  markersize=5.0, label=target_text(target))
    end
    axis.axhline(crossover.full_ms; color="0.15", linestyle=":",
                 linewidth=2.0,
                 label=@sprintf("Full field RK4IP: %.3g ms", crossover.full_ms))
    axis.set_xlabel("Retained scalar modes")
    axis.set_ylabel("Time per dz increment (ms)")
    axis.set_yscale("log")
    axis.set_xlim(first(crossover.modes), last(crossover.modes))
    axis.legend(frameon=false, fontsize=8.5, ncol=2)
    style_axis!(axis)
    save_pair(fig, output_dir, "crossover-time-per-step", artifacts)

    fig, axis = PyPlot.subplots(figsize=(7.1, 4.6))
    for (column, target) in pairs(crossover.targets)
        color = TARGET_COLORS[mod1(column, length(TARGET_COLORS))]
        axis.plot(crossover.modes, crossover.ranks[:, column]; color,
                  marker=TARGET_MARKERS[mod1(column, 3)],
                  linestyle=TARGET_STYLES[mod1(column, 3)], linewidth=1.8,
                  markersize=5.0, label=target_text(target))
    end
    axis.set_xlabel("Retained scalar modes")
    axis.set_ylabel("Compressed CP rank")
    axis.set_xlim(first(crossover.modes), last(crossover.modes))
    axis.legend(frameon=false, fontsize=8.5)
    style_axis!(axis)
    save_pair(fig, output_dir, "cp-rank-vs-modes", artifacts)
end

function load_solution_set(input_dir, targets)
    full_path = require_file(joinpath(input_dir, "full_field_solution.npz"),
                             "full-field GRIN solution")
    full = read_npz_checked(full_path, (
        "frequency_detuning_THz", "rk4ip_projected_spectrum",
        "rk4ip_projected_modal_energy", "rk4ip_fluence",
        "rk4ip_peak_time_intensity", "rk4ip_output_spectrum",
        "strang_output_spectrum", "strang_fluence",
        "strang_peak_time_intensity", "x_m", "y_m",
        "strang_rk4ip_field_error", "strang_rk4ip_spectrum_error",
        "rk4ip_subspace_leakage"))
    modal = NamedTuple[]
    for target in targets
        path = require_file(
            joinpath(input_dir, "modal_solution_$(target_tag(target)).npz"),
            "modal solution for CP target $(target)")
        data = read_npz_checked(path, (
            "frequency_detuning_THz", "output_spectrum",
            "output_modal_energy", "field_relative_error",
            "spectrum_relative_error", "modal_energy_relative_error",
            "target_error", "reported_cp_error", "cp_rank"))
        measured_target = only(Float64.(flat_array(data["target_error"])))
        isapprox(measured_target, target; rtol=1e-12, atol=0) || error(
            "$path reports target $measured_target, expected $target.")
        push!(modal, (; target, path, data))
    end
    return (; full_path, full, modal)
end

function plot_grin_integrator_spatial(solutions, output_dir, artifacts)
    data = solutions.full
    x = Float64.(flat_array(data["x_m"])) .* 1e6
    y = Float64.(flat_array(data["y_m"])) .* 1e6
    nx, ny = length(x), length(y)
    fields = (("Fluence", "strang_fluence", "rk4ip_fluence"),
              ("Peak-time intensity", "strang_peak_time_intensity",
               "rk4ip_peak_time_intensity"))
    fig, axes = PyPlot.subplots(2, 3; figsize=(10.4, 7.0))
    extent = (first(x), last(x), first(y), last(y))
    for (row, (row_label, strang_key, rk_key)) in pairs(fields)
        strang = spatial_matrix(data, strang_key, nx, ny)
        rk4ip = spatial_matrix(data, rk_key, nx, ny)
        scale = positive_scale(rk4ip, rk_key)
        disagreement = abs.(strang .- rk4ip) ./ scale
        for (column, (title, values)) in pairs((
                ("Strang", strang ./ scale), ("RK4IP", rk4ip ./ scale),
                ("|Strang - RK4IP|", disagreement)))
            axis = axes[row, column]
            image = axis.imshow(permutedims(values); origin="lower",
                                extent=extent, aspect="equal",
                                cmap=column == 3 ? "cividis" : "magma",
                                vmin=0.0,
                                vmax=column == 3 ?
                                     max(maximum(disagreement), eps(Float64)) :
                                     1.0)
            axis.set_title(title)
            axis.set_xlabel("x (um)")
            column == 1 && axis.set_ylabel("$row_label\ny (um)")
            axis.tick_params(direction="in")
            fig.colorbar(image; ax=axis, fraction=0.046, pad=0.04)
        end
    end
    save_pair(fig, output_dir, "grin-strang-vs-rk4ip-spatial", artifacts)
end

function plot_grin_integrator_spectral(solutions, output_dir, artifacts)
    data = solutions.full
    frequency = Float64.(flat_array(data["frequency_detuning_THz"]))
    strang = finite_nonnegative(flat_array(data["strang_output_spectrum"]),
                                "GRIN strang_output_spectrum")
    rk4ip = finite_nonnegative(flat_array(data["rk4ip_output_spectrum"]),
                               "GRIN rk4ip_output_spectrum")
    length(frequency) == length(strang) == length(rk4ip) || error(
        "GRIN integrator spectral axes and series have inconsistent lengths.")
    scale = positive_scale(rk4ip, "GRIN rk4ip_output_spectrum")
    xlimits = spectral_limits(frequency, [strang, rk4ip])
    fig, axes = PyPlot.subplots(2, 1; figsize=(7.2, 6.3), sharex=true)
    axes[1].plot(frequency, relative_db(strang, scale); color="#D55E00",
                 linestyle="--", linewidth=1.7, label="Strang")
    axes[1].plot(frequency, relative_db(rk4ip, scale); color="#0072B2",
                 linewidth=1.8, label="RK4IP")
    axes[2].plot(frequency, relative_db(abs.(strang .- rk4ip), scale);
                 color="#009E73", linewidth=1.6,
                 label="|Strang - RK4IP|")
    axes[1].set_ylabel("Spectrum / RK4IP peak (dB)")
    axes[2].set_ylabel("Absolute difference / peak (dB)")
    axes[2].set_xlabel("Frequency detuning (THz)")
    axes[1].set_ylim(DB_FLOOR, 2); axes[2].set_ylim(DB_FLOOR, 2)
    axes[1].set_xlim(xlimits...)
    axes[1].legend(frameon=false); axes[2].legend(frameon=false)
    style_axis!(axes[1]); style_axis!(axes[2])
    save_pair(fig, output_dir, "grin-strang-vs-rk4ip-spectral", artifacts)
end

function load_grid_refinement(directory)
    path = require_file(joinpath(directory, "full_field_solution.npz"),
                        "128x128 full-field grid refinement")
    data = read_npz_checked(path, (
        "frequency_detuning_THz", "rk4ip_projected_spectrum",
        "rk4ip_projected_modal_energy", "rk4ip_subspace_leakage",
        "x_m", "y_m"))
    return (; directory, path, data)
end

function plot_grid_convergence(solutions, refinement, output_dir, artifacts)
    coarse = solutions.full
    fine = refinement.data
    frequency = Float64.(flat_array(coarse["frequency_detuning_THz"]))
    fine_frequency = Float64.(flat_array(fine["frequency_detuning_THz"]))
    validate_same_axis(frequency, fine_frequency, "96-grid frequency",
                       "128-grid frequency")
    coarse_spectrum = finite_nonnegative(
        flat_array(coarse["rk4ip_projected_spectrum"]),
        "96-grid projected spectrum")
    fine_spectrum = finite_nonnegative(
        flat_array(fine["rk4ip_projected_spectrum"]),
        "128-grid projected spectrum")
    coarse_energy = finite_nonnegative(
        flat_array(coarse["rk4ip_projected_modal_energy"]),
        "96-grid modal energy")
    fine_energy = finite_nonnegative(
        flat_array(fine["rk4ip_projected_modal_energy"]),
        "128-grid modal energy")
    length(coarse_energy) == length(fine_energy) || error(
        "Grid-refinement modal-energy lengths differ.")
    spectrum_scale = positive_scale(fine_spectrum, "128-grid spectrum")
    energy_scale = sum(fine_energy)
    modes = collect(eachindex(fine_energy))
    xlimits = spectral_limits(frequency, [coarse_spectrum, fine_spectrum])
    fig, axes = PyPlot.subplots(2, 2; figsize=(10.0, 6.8))
    axes[1, 1].plot(frequency, relative_db(coarse_spectrum, spectrum_scale);
                    color="#D55E00", linestyle="--", label="96x96")
    axes[1, 1].plot(frequency, relative_db(fine_spectrum, spectrum_scale);
                    color="#0072B2", label="128x128")
    axes[1, 1].set_xlim(xlimits...); axes[1, 1].set_ylim(DB_FLOOR, 2)
    axes[1, 1].set_xlabel("Frequency detuning (THz)")
    axes[1, 1].set_ylabel("Projected spectrum (dB)")
    axes[1, 1].legend(frameon=false)
    axes[1, 2].plot(frequency,
                    relative_db(abs.(coarse_spectrum .- fine_spectrum),
                                spectrum_scale; floor_db=-160.0);
                    color="#009E73")
    axes[1, 2].set_xlim(xlimits...); axes[1, 2].set_ylim(-150, -90)
    axes[1, 2].set_xlabel("Frequency detuning (THz)")
    axes[1, 2].set_ylabel("Spectrum difference / peak (dB)")
    floor_fraction = 1e-16
    axes[2, 1].semilogy(modes,
                        max.(coarse_energy ./ energy_scale, floor_fraction);
                        color="#D55E00", linestyle="--", label="96x96")
    axes[2, 1].semilogy(modes,
                        max.(fine_energy ./ energy_scale, floor_fraction);
                        color="#0072B2", label="128x128")
    axes[2, 1].set_xlabel("Scalar HG mode index")
    axes[2, 1].set_ylabel("Energy / retained energy")
    axes[2, 1].legend(frameon=false)
    axes[2, 2].semilogy(
        modes, max.(abs.(coarse_energy .- fine_energy) ./ energy_scale,
                    1e-18); color="#009E73")
    axes[2, 2].set_xlabel("Scalar HG mode index")
    axes[2, 2].set_ylabel("Modal-energy difference / total")
    for axis in axes
        style_axis!(axis)
    end
    save_pair(fig, output_dir, "spatial-grid-convergence", artifacts)
    return (;
        spectrum_relative_error=norm(coarse_spectrum - fine_spectrum) /
                                norm(fine_spectrum),
        modal_energy_relative_error=norm(coarse_energy - fine_energy) /
                                    norm(fine_energy),
        coarse_leakage=only(flat_array(coarse["rk4ip_subspace_leakage"])),
        fine_leakage=only(flat_array(fine["rk4ip_subspace_leakage"])),
    )
end

function validate_same_axis(reference, candidate, reference_name, candidate_name)
    length(reference) == length(candidate) || error(
        "$candidate_name has $(length(candidate)) samples; " *
        "$reference_name has $(length(reference)).")
    all(isapprox.(reference, candidate; rtol=1e-10, atol=1e-12)) || error(
        "$candidate_name does not match $reference_name.")
end

function plot_full_vs_cp_spectra(solutions, output_dir, artifacts)
    frequency = Float64.(flat_array(solutions.full["frequency_detuning_THz"]))
    reference = finite_nonnegative(
        flat_array(solutions.full["rk4ip_projected_spectrum"]),
        "rk4ip_projected_spectrum")
    length(frequency) == length(reference) || error(
        "Full-field frequency and projected spectrum lengths differ.")
    spectra = Vector{Vector{Float64}}()
    for item in solutions.modal
        modal_frequency = Float64.(flat_array(
            item.data["frequency_detuning_THz"]))
        validate_same_axis(frequency, modal_frequency, "full-field frequency",
                           "modal frequency for target $(item.target)")
        push!(spectra, vec(finite_nonnegative(
            flat_array(item.data["output_spectrum"]),
            "modal output spectrum for target $(item.target)")))
    end
    scale = positive_scale(reference, "full-field projected spectrum")
    xlimits = spectral_limits(frequency, [reference, spectra...])
    fig, axes = PyPlot.subplots(2, 1; figsize=(7.2, 6.4), sharex=true)
    axes[1].plot(frequency, relative_db(reference, scale); color="0.1",
                 linewidth=2.2, label="Projected full field RK4IP")
    for (index, item) in pairs(solutions.modal)
        axes[1].plot(frequency, relative_db(spectra[index], scale);
                     color=TARGET_COLORS[mod1(index, length(TARGET_COLORS))],
                     linestyle=TARGET_STYLES[mod1(index, 3)], linewidth=1.55,
                     label=target_text(item.target))
        axes[2].plot(frequency,
                     relative_db(abs.(spectra[index] .- reference), scale);
                     color=TARGET_COLORS[mod1(index, length(TARGET_COLORS))],
                     linestyle=TARGET_STYLES[mod1(index, 3)], linewidth=1.55,
                     label=target_text(item.target))
    end
    axes[1].set_ylabel("Spectrum / full-field peak (dB)")
    axes[2].set_ylabel("Absolute difference / peak (dB)")
    axes[2].set_xlabel("Frequency detuning (THz)")
    axes[1].set_ylim(DB_FLOOR, 2)
    axes[2].set_ylim(DB_FLOOR, 2)
    axes[1].set_xlim(xlimits...)
    axes[1].legend(frameon=false, fontsize=8.3, ncol=2)
    style_axis!(axes[1]); style_axis!(axes[2])
    save_pair(fig, output_dir, "full-field-vs-cp-spectra", artifacts)
end

function plot_full_vs_cp_modal_energy(solutions, output_dir, artifacts)
    reference = finite_nonnegative(
        flat_array(solutions.full["rk4ip_projected_modal_energy"]),
        "rk4ip_projected_modal_energy")
    total = sum(reference)
    isfinite(total) && total > 0 || error(
        "Projected full-field modal energy has zero total.")
    modes = collect(eachindex(reference))
    floor_fraction = 1e-14
    fig, axes = PyPlot.subplots(2, 1; figsize=(7.2, 6.4), sharex=true)
    axes[1].plot(modes, max.(reference ./ total, floor_fraction);
                 color="0.1", linewidth=2.0, label="Projected full field RK4IP")
    for (index, item) in pairs(solutions.modal)
        energy = finite_nonnegative(flat_array(item.data["output_modal_energy"]),
                                    "modal energy for target $(item.target)")
        length(energy) == length(reference) || error(
            "Modal-energy length for target $(item.target) is " *
            "$(length(energy)); expected $(length(reference)).")
        color = TARGET_COLORS[mod1(index, length(TARGET_COLORS))]
        axes[1].plot(modes, max.(energy ./ total, floor_fraction); color,
                     linestyle=TARGET_STYLES[mod1(index, 3)], linewidth=1.4,
                     label=target_text(item.target))
        axes[2].plot(modes,
                     max.(abs.(energy .- reference) ./ total, floor_fraction);
                     color, linestyle=TARGET_STYLES[mod1(index, 3)],
                     linewidth=1.4, label=target_text(item.target))
    end
    axes[1].set_ylabel("Energy / full-field retained energy")
    axes[2].set_ylabel("Absolute difference / retained energy")
    axes[2].set_xlabel("Scalar HG mode index")
    axes[1].set_yscale("log"); axes[2].set_yscale("log")
    axes[1].set_xlim(first(modes), last(modes))
    axes[1].legend(frameon=false, fontsize=8.3, ncol=2)
    style_axis!(axes[1]); style_axis!(axes[2])
    save_pair(fig, output_dir, "full-field-vs-cp-modal-energy", artifacts)
end

function load_integrator_comparison(input_dir)
    path = require_file(joinpath(input_dir, "integrator_comparison.npz"),
                        "Strang/RK4IP integrator comparison")
    data = read_npz_checked(path, (
        "frequency_detuning_THz", "x_m", "y_m", "strang_spectrum",
        "rk4ip_spectrum", "reference_spectrum", "strang_fluence",
        "rk4ip_fluence", "reference_fluence", "strang_peak_intensity",
        "rk4ip_peak_intensity", "reference_peak_intensity",
        "strang_order", "rk4ip_order", "cross_order", "reference_gap"))
    return (; path, data)
end

function spatial_matrix(data, key, nx, ny)
    values = finite_nonnegative(data[key], key)
    size(values) == (nx, ny) && return values
    size(values) == (ny, nx) && return permutedims(values)
    error("$key has shape $(size(values)); expected ($nx, $ny).")
end

function plot_integrator_spatial(comparison, output_dir, artifacts)
    data = comparison.data
    x = Float64.(flat_array(data["x_m"])) .* 1e6
    y = Float64.(flat_array(data["y_m"])) .* 1e6
    nx, ny = length(x), length(y)
    fields = (("Fluence", "strang_fluence", "rk4ip_fluence",
               "reference_fluence"),
              ("Peak-time intensity", "strang_peak_intensity",
               "rk4ip_peak_intensity", "reference_peak_intensity"))
    fig, axes = PyPlot.subplots(2, 3; figsize=(10.2, 6.8))
    extent = (first(x), last(x), first(y), last(y))
    for (row, (row_label, strang_key, rk_key, reference_key)) in pairs(fields)
        strang = spatial_matrix(data, strang_key, nx, ny)
        rk4ip = spatial_matrix(data, rk_key, nx, ny)
        reference = spatial_matrix(data, reference_key, nx, ny)
        scale = positive_scale(reference, reference_key)
        disagreement = abs.(strang .- rk4ip) ./ scale
        for (column, (title, values)) in pairs((
                ("Strang", strang ./ scale), ("RK4IP", rk4ip ./ scale),
                ("|Strang - RK4IP|", disagreement)))
            axis = axes[row, column]
            image = axis.imshow(permutedims(values); origin="lower",
                                extent=extent, aspect="equal",
                                cmap=column == 3 ? "cividis" : "magma",
                                vmin=0.0,
                                vmax=column == 3 ?
                                     max(maximum(disagreement), eps(Float64)) :
                                     1.0)
            axis.set_title(title)
            axis.set_xlabel("x (um)")
            column == 1 && axis.set_ylabel("$row_label\ny (um)")
            axis.tick_params(direction="in")
            fig.colorbar(image; ax=axis, fraction=0.046, pad=0.04)
        end
    end
    save_pair(fig, output_dir, "strang-vs-rk4ip-spatial", artifacts)
end

function plot_integrator_spectral(comparison, output_dir, artifacts)
    data = comparison.data
    frequency = Float64.(flat_array(data["frequency_detuning_THz"]))
    strang = finite_nonnegative(flat_array(data["strang_spectrum"]),
                                "strang_spectrum")
    rk4ip = finite_nonnegative(flat_array(data["rk4ip_spectrum"]),
                               "rk4ip_spectrum")
    reference = finite_nonnegative(flat_array(data["reference_spectrum"]),
                                   "reference_spectrum")
    length(frequency) == length(strang) == length(rk4ip) == length(reference) ||
        error("Integrator spectral axes and series have inconsistent lengths.")
    scale = positive_scale(reference, "reference_spectrum")
    xlimits = spectral_limits(frequency, [reference, strang, rk4ip])
    fig, axes = PyPlot.subplots(2, 1; figsize=(7.2, 6.3), sharex=true)
    axes[1].plot(frequency, relative_db(reference, scale); color="0.1",
                 linewidth=2.0, label="RK4IP reference")
    axes[1].plot(frequency, relative_db(strang, scale); color="#D55E00",
                 linestyle="--", linewidth=1.6, label="Strang")
    axes[1].plot(frequency, relative_db(rk4ip, scale); color="#0072B2",
                 linestyle="-.", linewidth=1.6, label="RK4IP")
    axes[2].plot(frequency, relative_db(abs.(strang .- reference), scale);
                 color="#D55E00", linestyle="--", linewidth=1.6,
                 label="Strang error")
    axes[2].plot(frequency, relative_db(abs.(rk4ip .- reference), scale);
                 color="#0072B2", linestyle="-.", linewidth=1.6,
                 label="RK4IP error")
    axes[1].set_ylabel("Spectrum / reference peak (dB)")
    axes[2].set_ylabel("Absolute error / peak (dB)")
    axes[2].set_xlabel("Frequency detuning (THz)")
    axes[1].set_ylim(DB_FLOOR, 2); axes[2].set_ylim(DB_FLOOR, 2)
    axes[1].set_xlim(xlimits...)
    axes[1].legend(frameon=false, fontsize=8.5)
    axes[2].legend(frameon=false, fontsize=8.5)
    style_axis!(axes[1]); style_axis!(axes[2])
    save_pair(fig, output_dir, "strang-vs-rk4ip-spectral", artifacts)
end

function read_simple_numeric_csv(path)
    lines = filter(line -> !isempty(strip(line)), readlines(path))
    length(lines) >= 2 || error("$path has no data rows.")
    header = strip.(split(first(lines), ','; keepempty=true))
    columns = Dict(name => Float64[] for name in header)
    for (offset, line) in enumerate(Iterators.drop(lines, 1))
        line_number = offset + 1
        cells = strip.(split(line, ','; keepempty=true))
        length(cells) == length(header) || error(
            "$path line $line_number has $(length(cells)) columns; " *
            "expected $(length(header)).")
        for (name, cell) in zip(header, cells)
            isempty(cell) && error("$path line $line_number has an empty $name.")
            push!(columns[name], parse(Float64, cell))
        end
    end
    return columns
end

function plot_integrator_order(csv_path, comparison, output_dir, artifacts)
    columns = read_simple_numeric_csv(csv_path)
    required = ("steps", "dz_m", "strang_error", "rk4ip_error", "cross_error")
    missing_keys = filter(key -> !haskey(columns, key), collect(required))
    isempty(missing_keys) || error(
        "$csv_path is missing columns: $(join(missing_keys, ", ")).")
    dz = columns["dz_m"]
    permutation = sortperm(dz)
    dz = dz[permutation]
    strang = columns["strang_error"][permutation]
    rk4ip = columns["rk4ip_error"][permutation]
    cross = columns["cross_error"][permutation]
    all(>(0), vcat(dz, strang, rk4ip, cross)) || error(
        "$csv_path must contain positive dz and error values.")
    fig, axis = PyPlot.subplots(figsize=(7.1, 4.8))
    axis.loglog(dz, strang; color="#D55E00", marker="s", linewidth=1.7,
                label="Strang vs reference")
    axis.loglog(dz, rk4ip; color="#0072B2", marker="o", linewidth=1.7,
                label="RK4IP vs reference")
    axis.loglog(dz, cross; color="#009E73", marker="^", linestyle="--",
                linewidth=1.5, label="Strang vs RK4IP")
    anchor = max(1, length(dz) - 1)
    second_order = strang[anchor] .* (dz ./ dz[anchor]) .^ 2
    fourth_order = rk4ip[anchor] .* (dz ./ dz[anchor]) .^ 4
    axis.loglog(dz, second_order; color="0.5", linestyle=":", linewidth=1.1,
                label="O(dz^2)")
    axis.loglog(dz, fourth_order; color="0.25", linestyle=":", linewidth=1.1,
                label="O(dz^4)")
    strang_order = only(Float64.(flat_array(
        comparison.data["strang_order"])))
    rk4ip_order = only(Float64.(flat_array(
        comparison.data["rk4ip_order"])))
    cross_order = only(Float64.(flat_array(
        comparison.data["cross_order"])))
    axis.text(0.02, 0.03,
              @sprintf("observed: Strang %.2f, RK4IP %.2f, cross %.2f",
                       strang_order, rk4ip_order, cross_order);
              transform=axis.transAxes, fontsize=8.5, va="bottom")
    axis.set_xlabel("dz (m)")
    axis.set_ylabel("Relative L2 field error")
    axis.legend(frameon=false, fontsize=8.2, ncol=2)
    style_axis!(axis)
    save_pair(fig, output_dir, "integrator-convergence-order", artifacts)
end

csv_cell(value) = begin
    text = string(value)
    occursin(r"[\",\n]", text) ? "\"$(replace(text, '\"' => "\"\""))\"" : text
end

function write_summaries(output_dir, crossover, solutions, comparison,
                         grid_metrics, refinement, artifacts)
    rows = NamedTuple[]
    push!(rows, (; section="timing", target="", metric="full_field_median",
                  value=crossover.full_ms, unit="ms_per_dz",
                  source=crossover.path))
    for (column, target) in pairs(crossover.targets)
        cp_fast = findall(crossover.times[:, column] .< crossover.full_ms)
        full_fast = findall(crossover.times[:, column] .>= crossover.full_ms)
        push!(rows, (; section="crossover", target=target,
              metric="largest_measured_mode_cp_faster",
              value=isempty(cp_fast) ? "not_observed" :
                    string(maximum(crossover.modes[cp_fast])), unit="modes",
              source=crossover.path))
        push!(rows, (; section="crossover", target=target,
              metric="smallest_measured_mode_full_field_faster",
              value=isempty(full_fast) ? "not_observed" :
                    string(minimum(crossover.modes[full_fast])), unit="modes",
              source=crossover.path))
    end
    for item in solutions.modal
        for (metric, key, unit) in (
                ("cp_rank", "cp_rank", "rank"),
                ("reported_cp_error", "reported_cp_error", "relative"),
                ("field_relative_error", "field_relative_error", "relative"),
                ("spectrum_relative_error", "spectrum_relative_error", "relative"),
                ("modal_energy_relative_error", "modal_energy_relative_error",
                 "relative"))
            value = only(flat_array(item.data[key]))
            push!(rows, (; section="solution", target=item.target, metric,
                          value, unit, source=item.path))
        end
    end
    for (metric, key) in (("strang_order", "strang_order"),
                          ("rk4ip_order", "rk4ip_order"),
                          ("cross_order", "cross_order"),
                          ("reference_gap", "reference_gap"))
        value = only(flat_array(comparison.data[key]))
        push!(rows, (; section="integrator", target="", metric, value,
                      unit="relative", source=comparison.path))
    end
    for (metric, key) in (
            ("grin_strang_rk4ip_field_error",
             "strang_rk4ip_field_error"),
            ("grin_strang_rk4ip_spectrum_error",
             "strang_rk4ip_spectrum_error"),
            ("grin_rk4ip_subspace_leakage", "rk4ip_subspace_leakage"))
        value = only(flat_array(solutions.full[key]))
        push!(rows, (; section="grin_integrator", target="", metric, value,
                      unit="relative", source=solutions.full_path))
    end
    for metric in propertynames(grid_metrics)
        push!(rows, (; section="grid_convergence", target="",
                      metric=string(metric), value=getproperty(grid_metrics, metric),
                      unit="relative", source=refinement.path))
    end
    csv_path = joinpath(output_dir, "artifact-summary.csv")
    open(csv_path, "w") do io
        println(io, "section,target,metric,value,unit,source")
        for row in rows
            println(io, join(csv_cell.((row.section, row.target, row.metric,
                                       row.value, row.unit, row.source)), ','))
        end
    end
    push!(artifacts, csv_path)

    markdown_path = joinpath(output_dir, "artifact-summary.md")
    open(markdown_path, "w") do io
        println(io, "# GRIN-210 benchmark artifact summary\n")
        println(io, "All values below are read from benchmark outputs; no " *
                    "synthetic points are used.\n")
        println(io, @sprintf("Full-field RK4IP median: **%.6g ms per dz** (dz = %.6g m).\n", crossover.full_ms,
                            crossover.dz_m))
        println(io, "| CP target | Largest measured mode count with CP faster | " *
                    "Smallest measured mode count with full field faster |")
        println(io, "|---:|---:|---:|")
        for (column, target) in pairs(crossover.targets)
            cp_fast = findall(crossover.times[:, column] .< crossover.full_ms)
            full_fast = findall(crossover.times[:, column] .>= crossover.full_ms)
            lower = isempty(cp_fast) ? "not observed" :
                    string(maximum(crossover.modes[cp_fast]))
            upper = isempty(full_fast) ? "not observed" :
                    string(minimum(crossover.modes[full_fast]))
            println(io, "| $(target_text(target)) | $lower | $upper |")
        end
        println(io, "\nThese are measured bounds, not interpolated crossover estimates.\n")
        println(io, "| CP target | Rank | Field error | Spectrum error | Modal-energy error |")
        println(io, "|---:|---:|---:|---:|---:|")
        for item in solutions.modal
            rank = round(Int, only(flat_array(item.data["cp_rank"])))
            field_error = only(flat_array(item.data["field_relative_error"]))
            spectrum_error = only(flat_array(item.data["spectrum_relative_error"]))
            energy_error = only(flat_array(item.data["modal_energy_relative_error"]))
            println(io, @sprintf("| %s | %d | %.6g | %.6g | %.6g |",
                                 target_text(item.target), rank, field_error,
                                 spectrum_error, energy_error))
        end
        strang_order = only(flat_array(comparison.data["strang_order"]))
        rk4ip_order = only(flat_array(comparison.data["rk4ip_order"]))
        cross_order = only(flat_array(comparison.data["cross_order"]))
        reference_gap = only(flat_array(comparison.data["reference_gap"]))
        println(io, @sprintf("\nObserved longitudinal orders: Strang **%.4g**, RK4IP **%.4g**, method-to-method **%.4g**. Reference gap: **%.4g**.\n", strang_order,
                            rk4ip_order, cross_order, reference_gap))
        grin_field_error = only(flat_array(
            solutions.full["strang_rk4ip_field_error"]))
        grin_spectrum_error = only(flat_array(
            solutions.full["strang_rk4ip_spectrum_error"]))
        grin_leakage = only(flat_array(
            solutions.full["rk4ip_subspace_leakage"]))
        println(io, @sprintf("GRIN 25 mm Strang/RK4IP errors: field **%.6g**, spectrum **%.6g**; RK4IP energy outside the 210-mode subspace: **%.6g**.\n", grin_field_error,
                            grin_spectrum_error, grin_leakage))
        println(io, @sprintf("Spatial-grid refinement 96x96 -> 128x128: spectrum change **%.6g**, projected modal-energy change **%.6g**; leakage %.6g -> %.6g.\n", grid_metrics.spectrum_relative_error,
                            grid_metrics.modal_energy_relative_error,
                            grid_metrics.coarse_leakage,
                            grid_metrics.fine_leakage))
        println(io, "## Generated files\n")
        for path in artifacts
            println(io, "- `$(basename(path))`")
        end
    end
    push!(artifacts, markdown_path)
    return rows
end

function main()
    crossover_dir = abspath(get(
        ENV, "GRIN210_PLOT_INPUT_DIR",
        joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_crossover")))
    integrator_dir = abspath(get(
        ENV, "FULL_FIELD_INTEGRATOR_PLOT_INPUT_DIR",
        joinpath(@__DIR__, "..", "benchmark_cache",
                 "full_field_integrator_validation")))
    grid_refinement_dir = abspath(get(
        ENV, "GRIN210_GRID_REFINEMENT_DIR",
        joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_grid128")))
    output_dir = abspath(get(
        ENV, "GRIN210_PLOT_OUTPUT_DIR", joinpath(crossover_dir, "plots")))
    mkpath(output_dir)

    PyPlot.rc("font", family="sans-serif", size=9.5)
    PyPlot.rc("axes", linewidth=0.8)
    PyPlot.rc("lines", solid_capstyle="round")

    artifacts = String[]
    crossover = load_crossover(crossover_dir)
    solutions = load_solution_set(crossover_dir, crossover.targets)
    comparison = load_integrator_comparison(integrator_dir)
    refinement = load_grid_refinement(grid_refinement_dir)

    plot_crossover(crossover, output_dir, artifacts)
    plot_full_vs_cp_spectra(solutions, output_dir, artifacts)
    plot_full_vs_cp_modal_energy(solutions, output_dir, artifacts)
    plot_grin_integrator_spatial(solutions, output_dir, artifacts)
    plot_grin_integrator_spectral(solutions, output_dir, artifacts)
    plot_integrator_spatial(comparison, output_dir, artifacts)
    plot_integrator_spectral(comparison, output_dir, artifacts)
    grid_metrics = plot_grid_convergence(
        solutions, refinement, output_dir, artifacts)

    convergence_csv = joinpath(integrator_dir, "convergence.csv")
    if isfile(convergence_csv)
        plot_integrator_order(convergence_csv, comparison, output_dir, artifacts)
    else
        @warn "No convergence.csv found; integrator-order plot was not generated." path=convergence_csv
    end
    write_summaries(output_dir, crossover, solutions, comparison,
                    grid_metrics, refinement, artifacts)

    println("plot_grin_210_artifacts: PASS")
    println("input_grin_dir=", crossover_dir)
    println("input_integrator_dir=", integrator_dir)
    println("input_grid_refinement_dir=", grid_refinement_dir)
    println("output_dir=", output_dir)
    println("artifact_count=", length(artifacts))
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
