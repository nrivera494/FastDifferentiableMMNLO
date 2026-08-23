using CUDA
using FFTW
using LinearAlgebra
using NPZ
using PulsePropagation

include(joinpath(@__DIR__, "eslami_190_modal_common.jl"))

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

environment_value(name, default, ::Type{T}) where {T} =
    parse(T, get(ENV, name, string(default)))

const TARGET_ERROR = environment_value(
    "ESLAMI_MODAL_CP_TARGET", 1e-4, Float64)
const NT = environment_value("ESLAMI_MODAL_NT", 16384, Int)
const TIME_WINDOW_PS = environment_value(
    "ESLAMI_MODAL_TIME_WINDOW_PS", 20.0, Float64)
const LENGTH_M = environment_value("ESLAMI_MODAL_LENGTH", 0.20, Float64)
const DZ_M = environment_value("ESLAMI_MODAL_DZ", 37e-6, Float64)
const PEAK_POWER_W = environment_value(
    "ESLAMI_MODAL_PEAK_POWER", 5.85e5, Float64)
const PRECISION = Symbol(lowercase(get(
    ENV, "ESLAMI_MODAL_PRECISION", "mixed")))
const BACKEND = Symbol(lowercase(get(
    ENV, "ESLAMI_MODAL_BACKEND", "cuda_cp_optimized")))
const SPECTRAL_FILTER = get(
    ENV, "ESLAMI_MODAL_SPECTRAL_FILTER", "true") == "true"
const CP_DIR = get(
    ENV,
    "ESLAMI_CP_OUTPUT_DIR",
    joinpath(@__DIR__, "..", "benchmark_cache", "eslami_190_cp"),
)
const OUTPUT_PATH = get(
    ENV,
    "ESLAMI_MODAL_OUTPUT",
    joinpath(
        CP_DIR,
        "eslami_190_modal_$(eslami_target_tag(TARGET_ERROR)).npz",
    ),
)

tag = eslami_target_tag(TARGET_ERROR)
data_path = joinpath(CP_DIR, "eslami_190_cp_$(tag).jld2")
metadata_path = joinpath(CP_DIR, "eslami_190_cp_$(tag).toml")
cp = load_cp_decomposition(data_path, metadata_path)
cp.metadata.nmodes == ESLAMI_MODE_COUNT || error(
    "The cached CP does not contain 190 spatial modes.")
cp.relative_error <= TARGET_ERROR || error(
    "The cached CP misses requested target $TARGET_ERROR.")

time = TimeGrid(NT, TIME_WINDOW_PS)
initial, launch_capture = eslami_modal_initial(time)
parameters = eslami_modal_parameters(
    time, cp; propagation_length=LENGTH_M,
    spectral_filter=SPECTRAL_FILTER)
initial_energy_nj = sum(abs2, initial) * time.dt / 1e3

println(
    "stage=modal_start",
    " modes=", ESLAMI_MODE_COUNT,
    " cp_rank=", cp_rank(cp),
    " cp_error=", cp.relative_error,
    " nt=", NT,
    " length_m=", LENGTH_M,
    " dz_m=", DZ_M,
    " backend=", BACKEND,
    " precision=", PRECISION,
    " launch_capture_before_renormalization=", launch_capture,
    " initial_energy_nj=", initial_energy_nj,
)

wall_seconds = @elapsed solution = solve_mmgnlse(
    initial, parameters, DZ_M;
    backend=BACKEND,
    precision=PRECISION,
    adaptive=false,
    saveat=LENGTH_M,
)
final_field = Array(get_temporal_field(solution; z=:final))
final_energy_nj = sum(abs2, final_field) * time.dt / 1e3
power_drift = abs(final_energy_nj / initial_energy_nj - 1)
initial_spectral = fftshift(ifft(initial, 1), 1)
final_spectral = Array(get_spectral_field(solution; z=:final))
initial_spectrum = vec(sum(abs2, initial_spectral; dims=(2, 3)))
final_spectrum = vec(sum(abs2, final_spectral; dims=(2, 3)))
initial_spectrum ./= maximum(initial_spectrum)
final_spectrum ./= maximum(final_spectrum)
wavelength_nm = wavelength_axis(time, ESLAMI_OMEGA0) .* 1e9
frequency_detuning_thz = frequency_axis(time) ./ (2pi)
mode_energy_nj = vec(sum(abs2, final_field; dims=(1, 3))) .* time.dt ./ 1e3

mkpath(dirname(OUTPUT_PATH))
npzwrite(OUTPUT_PATH, Dict(
    "frequency_detuning_THz" => frequency_detuning_thz,
    "wavelength_nm" => wavelength_nm,
    "input_spectrum_normalized" => initial_spectrum,
    "output_spectrum_normalized" => final_spectrum,
    "output_modal_energy_nJ" => mode_energy_nj,
    "initial_energy_nJ" => [initial_energy_nj],
    "final_energy_nJ" => [final_energy_nj],
    "power_drift" => [power_drift],
    "wall_seconds" => [wall_seconds],
    "cp_rank" => [cp_rank(cp)],
    "cp_relative_error" => [cp.relative_error],
    "launch_capture_before_renormalization" => [launch_capture],
))

println(
    "stage=modal_complete",
    " wall_seconds=", wall_seconds,
    " final_energy_nj=", final_energy_nj,
    " power_drift=", power_drift,
    " output_path=", OUTPUT_PATH,
)
