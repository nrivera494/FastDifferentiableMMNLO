using LinearAlgebra
using NPZ
using PulsePropagation
using Serialization

length(ARGS) == 1 || error(
    "Usage: julia --project=. benchmarks/summarize_step30_results.jl RESULTS.jls")

relative_l2(candidate, reference) =
    norm(candidate .- reference) / max(norm(reference), eps(Float64))
intensity_relative_l2(candidate, reference) =
    relative_l2(abs2.(candidate), abs2.(reference))
modal_energy(field) = vec(sum(abs2, field; dims=(1, 3)))

payload = deserialize(ARGS[1])
configuration = payload.configuration
results = payload.results
timings = payload.timings
reference_label = haskey(results, "fine_fp64") ?
                  "fine_fp64" : "fixed_fp64"
reference = results[reference_label].fields[:, :, :, end]
initial = results[reference_label].initial_field
initial_energy = sum(abs2, initial)

root = normpath(joinpath(@__DIR__, ".."))
data_dir = get(
    ENV,
    "STEP30_DATA_DIR",
    joinpath(root, "example_notebooks", "benchmark_data",
             "step_index_soliton"),
)
torch_output = reshape(
    permutedims(npzread(joinpath(
        data_dir, "output_fields_30_5e-05.npy"))),
    8192, 30, 1,
)

println("configuration=", configuration)
for label in sort!(collect(keys(results)))
    solution = results[label]
    final = solution.fields[:, :, :, end]
    println(
        "result variant=", label,
        " seconds=", timings[label],
        " speedup_vs_fixed_fp64=",
        timings["fixed_fp64"] / timings[label],
        " reference=", reference_label,
        " field_relative_l2=", relative_l2(final, reference),
        " intensity_relative_l2=", intensity_relative_l2(final, reference),
        " modal_energy_relative_l2=",
        relative_l2(modal_energy(final), modal_energy(reference)),
        " energy_relative_drift=", (sum(abs2, final) - initial_energy) /
                                    initial_energy,
        " torch_field_relative_l2=", relative_l2(final, torch_output),
        " torch_intensity_relative_l2=",
        intensity_relative_l2(final, torch_output),
        " accepted_steps=", solution.cache.accepted_steps,
        " rejected_steps=", solution.cache.rejected_steps,
        " rhs_evaluations=", solution.cache.rhs_evaluations,
        " step_statistics=", solution.cache.step_statistics,
    )
end
