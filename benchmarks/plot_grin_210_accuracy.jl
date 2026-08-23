using FFTW
using LinearAlgebra
using NPZ
using Printf
using PyPlot

const INPUT_DIR = get(
    ENV, "GRIN210_ACCURACY_INPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_accuracy"))
const TIMING_DIR = get(
    ENV, "GRIN210_TIMING_INPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "grin_210_crossover"))
const OUTPUT_DIR = get(
    ENV, "GRIN210_ACCURACY_PLOT_DIR", joinpath(INPUT_DIR, "plots"))

mkpath(OUTPUT_DIR)

function require_file(path)
    isfile(path) || error("Required benchmark artifact is missing: $path")
    return path
end

function read_key_value_csv(path)
    rows = Dict{String,Float64}()
    for line in Iterators.drop(readlines(require_file(path)), 1)
        name, value = split(line, ','; limit=2)
        rows[name] = parse(Float64, value)
    end
    return rows
end

function read_sweep(path)
    lines = readlines(require_file(path))
    header = split(first(lines), ',')
    index = Dict(name => position for (position, name) in pairs(header))
    rows = NamedTuple[]
    for line in Iterators.drop(lines, 1)
        values = split(line, ',')
        push!(rows, (;
            modes=parse(Int, values[index["modes"]]),
            target=parse(Float64, values[index["target"]]),
            rank=parse(Int, values[index["rank"]]),
            median_ms=parse(Float64, values[index["median_step_ms"]]),
            projected_error=parse(Float64,
                values[index["projected_field_error"]]),
            truncation_floor=parse(Float64,
                values[index["truncation_field_floor"]]),
            total_error=parse(Float64, values[index["total_field_error"]]),
            spectrum_error=parse(Float64, values[index["spectrum_error"]]),
            modal_energy_error=parse(Float64,
                values[index["modal_energy_error"]]),
        ))
    end
    return rows
end

function phase_error(candidate, reference)
    overlap = dot(vec(reference), vec(candidate))
    phase = iszero(overlap) ? one(overlap) : overlap / abs(overlap)
    return norm(vec(candidate .- phase .* reference)) / norm(vec(reference))
end

function save_figure(name)
    tight_layout()
    savefig(joinpath(OUTPUT_DIR, name * ".png"), dpi=220,
            bbox_inches="tight")
    savefig(joinpath(OUTPUT_DIR, name * ".pdf"), bbox_inches="tight")
    close()
end

const TARGETS = (1e-2, 1e-3, 1e-4)
const TARGET_LABELS = Dict(
    1e-2 => "CP tensor 10^-2",
    1e-3 => "CP tensor 10^-3",
    1e-4 => "CP tensor 10^-4",
)
const COLORS = Dict(1e-2 => "C0", 1e-3 => "C1", 1e-4 => "C2")
const MARKERS = Dict(1e-2 => "o", 1e-3 => "s", 1e-4 => "^")

sweep = read_sweep(joinpath(INPUT_DIR, "accuracy_sweep.csv"))
full_metrics = read_key_value_csv(joinpath(INPUT_DIR, "full_convergence.csv"))
modal_metrics = read_key_value_csv(joinpath(INPUT_DIR, "modal_convergence.csv"))
linear_metrics = read_key_value_csv(joinpath(INPUT_DIR, "linear_consistency.csv"))
bridge_lines = readlines(require_file(joinpath(INPUT_DIR, "nonlinear_bridge.csv")))
bridge_errors = [
    parse(Float64, split(line, ',')[2])
    for line in Iterators.drop(bridge_lines, 1)
]

timing_lines = readlines(require_file(joinpath(TIMING_DIR,
                                               "crossover_timings.csv")))
timing_header = split(first(timing_lines), ',')
timing_index = Dict(name => position for (position, name) in pairs(timing_header))
full_time_ms = only([
    parse(Float64, split(line, ',')[timing_index["median_ms"]])
    for line in Iterators.drop(timing_lines, 1)
    if split(line, ',')[timing_index["method"]] == "full_field_rk4ip"
])

full_fine = npzread(require_file(joinpath(INPUT_DIR, "full_dz_5um.npz")))
tight_fine = npzread(require_file(joinpath(
    INPUT_DIR, "modal_tight_dz_5um.npz")))
authority_fine = npzread(require_file(joinpath(
    INPUT_DIR, "modal_authority_dz_5um.npz")))
full_reference = ComplexF64.(full_fine["projected_final"])
tight_reference = ComplexF64.(tight_fine["final"])
authority_reference = ComplexF64.(authority_fine["final"])
full_authority_field_error = phase_error(authority_reference, full_reference)
full_tight_field_error = phase_error(tight_reference, full_reference)

# Accuracy-qualified timing: open markers fail the physical-field tolerance.
figure(figsize=(13.5, 4.4))
for (panel, solution_tolerance) in pairs(TARGETS)
    subplot(1, 3, panel)
    axhline(full_time_ms; color="0.15", linestyle=":", linewidth=2,
            label=@sprintf("Full field %.2f ms", full_time_ms))
    for target in TARGETS
        rows = sort(filter(row -> row.target == target, sweep); by=row -> row.modes)
        modes = [row.modes for row in rows]
        times = [row.median_ms for row in rows]
        plot(modes, times; color=COLORS[target], linewidth=1.4,
             alpha=0.65)
        valid = [row.total_error <= solution_tolerance for row in rows]
        scatter(modes[valid], times[valid]; color=COLORS[target],
                marker=MARKERS[target], s=42,
                label=TARGET_LABELS[target])
        scatter(modes[.!valid], times[.!valid]; facecolors="none",
                edgecolors=COLORS[target], marker=MARKERS[target], s=42)
    end
    yscale("log")
    xlabel("Retained scalar modes")
    panel == 1 && ylabel("Median time per dz increment (ms)")
    title(@sprintf("Physical-field tolerance 10^%d",
                   round(Int, log10(solution_tolerance))))
    grid(true; which="both", alpha=0.25)
    panel == 1 && legend(fontsize=8, loc="lower right")
end
save_figure("accuracy-qualified-crossover")

# Error budget: tensor compression is visibly below modal truncation.
figure(figsize=(8.2, 5.7))
floor_rows = sort(filter(row -> row.target == first(TARGETS), sweep);
                  by=row -> row.modes)
plot([row.modes for row in floor_rows],
     [row.truncation_floor for row in floor_rows];
     color="0.1", linestyle="--", linewidth=2.2,
     label="Modal truncation floor")
for target in TARGETS
    rows = sort(filter(row -> row.target == target, sweep); by=row -> row.modes)
    plot([row.modes for row in rows], [row.total_error for row in rows];
         color=COLORS[target], marker=MARKERS[target], linewidth=1.8,
         label=TARGET_LABELS[target])
end
for tolerance in TARGETS
    axhline(tolerance; color="0.65", linewidth=0.8, linestyle=":")
end
yscale("log")
xlabel("Retained scalar modes")
ylabel("Relative complex-field error")
grid(true; which="both", alpha=0.25)
legend(fontsize=9)
save_figure("physical-field-error-budget")

# Time-to-accuracy Pareto view, including the coarse full-field candidate.
figure(figsize=(8.2, 5.7))
for target in TARGETS
    rows = sort(filter(row -> row.target == target, sweep);
                by=row -> row.modes)
    errors = [row.total_error for row in rows]
    times = [row.median_ms for row in rows]
    plot(errors, times; color=COLORS[target], marker=MARKERS[target],
         linewidth=1.8, label=TARGET_LABELS[target])
    if target == 1e-4
        for row in rows[[1, 4, 7, 9]]
            annotate(string(row.modes), (row.total_error, row.median_ms),
                     xytext=(5, 3), textcoords="offset points", fontsize=8)
        end
    end
end
scatter([full_metrics["field_coarse_fine"]], [full_time_ms];
        color="0.1", marker="D", s=58, label="Full field dz=10 um")
xscale("log")
yscale("log")
xlabel("Error versus converged full field")
ylabel("Median time per dz increment (ms)")
grid(true; which="both", alpha=0.25)
legend(fontsize=9)
save_figure("time-to-physical-accuracy")

# The consistency hierarchy explains the residual disagreement.
consistency_labels = [
    "Basis projection",
    "Nonlinear RHS bridge",
    "Linear fine/modal",
    "Modal dz estimate",
    "CP/authority field",
    "Full dz estimate",
    "Authority/full projected",
]
consistency_values = [
    only(npzread(joinpath(INPUT_DIR, "basis_consistency.npz"))[
        "initial_projection_error"]),
    maximum(bridge_errors),
    linear_metrics["full_fine_vs_modal"],
    modal_metrics["tight_richardson_fine_field"],
    modal_metrics["tight_authority_field"],
    full_metrics["richardson_fine_field"],
    full_authority_field_error,
]
figure(figsize=(9.0, 5.6))
barh(1:length(consistency_values), consistency_values; color="C0")
xscale("log")
yticks(1:length(consistency_labels), consistency_labels)
xlabel("Relative field/RHS error")
grid(true; axis="x", which="both", alpha=0.25)
gca().invert_yaxis()
save_figure("solver-consistency-hierarchy")

# Recomputed fine-reference spectral overlay.
full_spectrum = vec(full_fine["projected_spectrum"])
tight_spectrum = vec(sum(abs2,
    fftshift(ifft(tight_reference, 1), 1); dims=2))
authority_spectrum = vec(sum(abs2,
    fftshift(ifft(authority_reference, 1), 1); dims=2))
scale = maximum(full_spectrum)
time_step_ps = 10.0 / length(full_spectrum)
frequency = collect(range(-1 / (2time_step_ps),
                          1 / (2time_step_ps);
                          length=length(full_spectrum) + 1))[1:end-1]
figure(figsize=(9.0, 7.2))
subplot(2, 1, 1)
plot(frequency, 10log10.(max.(full_spectrum ./ scale, 1e-8));
     color="0.1", linewidth=2.2, label="Projected full field, dz=5 um")
plot(frequency, 10log10.(max.(authority_spectrum ./ scale, 1e-8));
     color="C1", linestyle="--", label="Rank-4096 quadrature authority")
plot(frequency, 10log10.(max.(tight_spectrum ./ scale, 1e-8));
     color="C2", linestyle=":", label="Rank-1600 CP")
ylabel("Spectrum / full peak (dB)")
xlim(-12, 12)
ylim(-70, 1)
grid(true; alpha=0.25)
legend(fontsize=9)
subplot(2, 1, 2)
plot(frequency, 10log10.(max.(abs.(authority_spectrum - full_spectrum) ./
                                scale, 1e-9));
     color="C1", linestyle="--", label="Authority - full")
plot(frequency, 10log10.(max.(abs.(tight_spectrum - full_spectrum) ./
                                scale, 1e-9));
     color="C2", linestyle=":", label="Tight CP - full")
xlabel("Frequency detuning (THz)")
ylabel("Absolute difference / full peak (dB)")
xlim(-12, 12)
ylim(-80, -20)
grid(true; alpha=0.25)
legend(fontsize=9)
save_figure("converged-full-vs-modal-spectra")

# Compact machine-readable and human-readable conclusions.
open(joinpath(OUTPUT_DIR, "accuracy-summary.csv"), "w") do io
    println(io, "metric,value")
    println(io, "full_field_median_ms,$full_time_ms")
    println(io, "full_coarse_fine_field,$(full_metrics["field_coarse_fine"])")
    println(io, "full_fine_richardson,$(full_metrics["richardson_fine_field"])")
    println(io, "modal_fine_richardson,$(modal_metrics["tight_richardson_fine_field"])")
    println(io, "tight_authority_field,$(modal_metrics["tight_authority_field"])")
    println(io, "linear_fine_modal,$(linear_metrics["full_fine_vs_modal"])")
    println(io, "nonlinear_bridge_max,$(maximum(bridge_errors))")
    println(io, "full_authority_projected_field,$full_authority_field_error")
    println(io, "full_tight_projected_field,$full_tight_field_error")
    for tolerance in TARGETS
        valid = filter(row -> row.total_error <= tolerance, sweep)
        minimum_modes = isempty(valid) ? "not_observed" :
                        string(minimum(row.modes for row in valid))
        println(io, "minimum_modes_physical_$(tolerance),$minimum_modes")
    end
end

metric_specs = (
    (name="complex_field", property=:total_error),
    (name="spectrum", property=:spectrum_error),
    (name="modal_energy", property=:modal_energy_error),
)
open(joinpath(OUTPUT_DIR, "qualification-table.csv"), "w") do io
    println(io, "metric,solution_tolerance,cp_tensor_target,minimum_modes")
    for metric in metric_specs, tolerance in TARGETS, target in TARGETS
        valid = filter(
            row -> row.target == target &&
                   getproperty(row, metric.property) <= tolerance,
            sweep)
        minimum_modes = isempty(valid) ? "not_observed" :
                        string(minimum(row.modes for row in valid))
        println(io, join((metric.name, tolerance, target, minimum_modes), ','))
    end
end

open(joinpath(OUTPUT_DIR, "accuracy-summary.md"), "w") do io
    println(io, "# GRIN-210 matched-accuracy summary\n")
    @printf(io, "Full-field dz=10 -> 5 um difference: **%.6g**; fine-grid Richardson estimate: **%.6g**.\n\n",
            full_metrics["field_coarse_fine"],
            full_metrics["richardson_fine_field"])
    @printf(io, "Tight CP dz error estimate: **%.6g**; rank-1600 versus rank-4096 field difference: **%.6g**.\n\n",
            modal_metrics["tight_richardson_fine_field"],
            modal_metrics["tight_authority_field"])
    @printf(io, "Linear fine/full versus modal error: **%.6g**; maximum nonlinear RHS bridge error: **%.6g**.\n\n",
            linear_metrics["full_fine_vs_modal"], maximum(bridge_errors))
    @printf(io, "Rank-4096 authority versus projected fine full field: **%.6g**.\n\n",
            full_authority_field_error)
    println(io, "Physical-field qualification:\n")
    for tolerance in TARGETS
        valid = filter(row -> row.total_error <= tolerance, sweep)
        if isempty(valid)
            @printf(io, "- %.0e: no measured modal point qualifies.\n", tolerance)
        else
            @printf(io, "- %.0e: qualifying mode counts begin at %d.\n",
                    tolerance, minimum(row.modes for row in valid))
        end
    end
    println(io, "\n## Minimum qualifying mode count by metric\n")
    println(io, "| Metric tolerance | CP 1e-2 | CP 1e-3 | CP 1e-4 |")
    println(io, "|---|---:|---:|---:|")
    for metric in metric_specs, tolerance in TARGETS
        entries = String[]
        for target in TARGETS
            valid = filter(
                row -> row.target == target &&
                       getproperty(row, metric.property) <= tolerance,
                sweep)
            push!(entries, isempty(valid) ? "not observed" :
                  string(minimum(row.modes for row in valid)))
        end
        println(io, "| ", metric.name, " ", @sprintf("%.0e", tolerance),
                " | ", join(entries, " | "), " |")
    end
end

println("plot_grin_210_accuracy: PASS output_dir=", OUTPUT_DIR)
