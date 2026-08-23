using CUDA
using PulsePropagation
using Statistics

include(joinpath(@__DIR__, "eslami_190_modal_common.jl"))

const EXT = Base.get_extension(PulsePropagation, :PulsePropagationCUDAExt)
EXT === nothing && error("PulsePropagation CUDA extension did not load.")
CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

environment_value(name, default, ::Type{T}) where {T} =
    parse(T, get(ENV, name, string(default)))

function measure_cuda(f; warmup::Int, samples::Int)
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative."))
    samples > 0 || throw(ArgumentError("samples must be positive."))
    for _ in 1:warmup
        f()
        CUDA.synchronize()
    end
    milliseconds = Vector{Float64}(undef, samples)
    for index in eachindex(milliseconds)
        CUDA.synchronize()
        start = time_ns()
        f()
        CUDA.synchronize()
        milliseconds[index] = (time_ns() - start) / 1e6
    end
    return (;
        median_ms=median(milliseconds),
        mean_ms=mean(milliseconds),
        minimum_ms=minimum(milliseconds),
        maximum_ms=maximum(milliseconds),
        std_ms=std(milliseconds; corrected=false),
        samples,
    )
end

function write_profile(path, step!; trace::Bool)
    result = if trace
        CUDA.@profile trace=true begin
            step!()
            CUDA.synchronize()
        end
    else
        CUDA.@profile begin
            step!()
            CUDA.synchronize()
        end
    end
    open(path, "w") do io
        show(io, MIME("text/plain"), result)
        println(io)
    end
    return path
end

function print_timing(io, rank, stage, timing)
    println(
        io,
        rank, ',', stage, ',',
        timing.median_ms, ',', timing.mean_ms, ',',
        timing.minimum_ms, ',', timing.maximum_ms, ',',
        timing.std_ms, ',', timing.samples,
    )
    println(
        "rank=", rank,
        " stage=", stage,
        " median_ms=", timing.median_ms,
        " mean_ms=", timing.mean_ms,
        " minimum_ms=", timing.minimum_ms,
        " maximum_ms=", timing.maximum_ms,
        " std_ms=", timing.std_ms,
        " samples=", timing.samples,
    )
end

function profile_rank!(io, rank, cp, time, initial, dz,
                       warmup, samples, component_samples, output_dir)
    cp_rank(cp) == rank || error(
        "Expected cached rank $rank, received $(cp_rank(cp)).")
    parameters = eslami_modal_parameters(
        time, cp; propagation_length=dz, spectral_filter=true)
    policy = EXT._cuda_mmgnlse_cp_optimization(:cuda_cp_optimized)
    state_cache = EXT._cuda_mmgnlse_forward_cache(parameters, policy)
    rhs_cache = EXT._cuda_mmgnlse_forward_cache(parameters, policy, Float32)
    initial_device = CUDA.CuArray(ComplexF64.(initial))
    field_w = CUDA.ifft(initial_device, 1)
    workspace = EXT._cuda_mmgnlse_cp_precision_workspace(
        initial_device, state_cache, rhs_cache, :mixed)
    state = EXT._cuda_mmgnlse_cp_state_workspace(workspace)
    rhs = workspace.rhs
    step!() = EXT._cuda_mmgnlse_cp_rk4ip_step!(
        state.next_field, field_w, state_cache, rhs_cache,
        workspace, 0.0, dz)
    complete_rhs!() = EXT._cuda_mmgnlse_cp_integrator_rhs!(
        state.k1, field_w, rhs_cache, workspace)

    # Compile and establish the constant-step propagator before measuring.
    step!()
    CUDA.synchronize()
    allocated_bytes = Int(CUDA.used_memory())
    println(
        "rank=", rank,
        " device_used_bytes=", allocated_bytes,
        " symmetric_factors=", rhs_cache.overlap.symmetric_factors,
        " real_factors=", rhs_cache.overlap.real_factors,
        " optimization=", rhs_cache.overlap.optimization,
    )

    print_timing(io, rank, "rk4ip_step",
        measure_cuda(step!; warmup, samples))
    print_timing(io, rank, "complete_rhs",
        measure_cuda(complete_rhs!;
            warmup=max(2, warmup ÷ 2), samples=component_samples))
    print_timing(io, rank, "mixed_cast_in", measure_cuda(
        () -> (rhs.next_field .= field_w);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "modal_fft", measure_cuda(
        () -> begin
            rhs.field_t .= rhs.next_field
            rhs.fft_field! * rhs.field_t
        end;
        warmup=2, samples=component_samples))
    print_timing(io, rank, "cp_projection", measure_cuda(
        () -> EXT._cuda_mmgnlse_cp_project!(
            rhs, rhs.field_t, rhs_cache.overlap);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "cp_mix_backprojection", measure_cuda(
        () -> EXT._cuda_mmgnlse_cp_combined!(
            rhs.nonlinear, rhs, rhs_cache.overlap, rhs_cache.raman);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "modal_ifft", measure_cuda(
        () -> (rhs.ifft_field! * rhs.nonlinear);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "nonlinear_scale", measure_cuda(
        () -> (rhs.k1 .= rhs_cache.nonlinear_prefactor .* rhs.nonlinear);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "mixed_cast_out", measure_cuda(
        () -> (state.k1 .= rhs.k1);
        warmup=2, samples=component_samples))
    print_timing(io, rank, "rk_algebra", measure_cuda(
        () -> begin
            state.midpoint_base .= state.to_midpoint .* field_w
            state.k1 .*= state.to_midpoint
            state.stage .= state.midpoint_base .+ (dz / 2) .* state.k1
            state.stage .= state.midpoint_base .+ (dz / 2) .* state.k2
            state.stage .= state.from_midpoint .* (
                state.midpoint_base .+ dz .* state.k3)
            state.next_field .= state.from_midpoint .* (
                state.midpoint_base .+
                (dz / 6) .* (state.k1 .+ 2 .* state.k2 .+ 2 .* state.k3)) .+
                (dz / 6) .* state.k4
        end;
        warmup=2, samples=component_samples))
    flush(io)

    write_profile(
        joinpath(output_dir, "rank_$(rank)_summary.txt"), step!;
        trace=false)
    write_profile(
        joinpath(output_dir, "rank_$(rank)_trace.txt"), step!;
        trace=true)
    CUDA.reclaim()
    return nothing
end

function main()
    ranks = parse.(Int, split(get(
        ENV, "ESLAMI_PROFILE_RANKS", "1024,1280,1664"), ','))
    tags = Dict(1024 => "1em2", 1280 => "1em3", 1664 => "1em4")
    nt = environment_value("ESLAMI_PROFILE_NT", 16384, Int)
    window_ps = environment_value(
        "ESLAMI_PROFILE_TIME_WINDOW_PS", 20.0, Float64)
    dz = environment_value("ESLAMI_PROFILE_DZ", 37e-6, Float64)
    warmup = environment_value("ESLAMI_PROFILE_WARMUP", 5, Int)
    samples = environment_value("ESLAMI_PROFILE_SAMPLES", 30, Int)
    component_samples = environment_value(
        "ESLAMI_PROFILE_COMPONENT_SAMPLES", 20, Int)
    cp_dir = get(
        ENV, "ESLAMI_CP_OUTPUT_DIR",
        joinpath(@__DIR__, "..", "benchmark_cache", "eslami_190_cp"))
    output_dir = get(
        ENV, "ESLAMI_PROFILE_OUTPUT_DIR",
        joinpath(cp_dir, "rk4ip_profile_h100"))
    mkpath(output_dir)

    time = TimeGrid(nt, window_ps)
    initial, capture = eslami_modal_initial(time)
    timing_path = joinpath(output_dir, "timings.csv")
    open(timing_path, "w") do io
        println(io,
            "rank,stage,median_ms,mean_ms,minimum_ms,maximum_ms,std_ms,samples")
        println(
            "gpu=", CUDA.name(CUDA.device()),
            " nt=", nt,
            " time_window_ps=", window_ps,
            " dz_m=", dz,
            " precision=mixed backend=cuda_cp_optimized",
            " launch_capture=", capture,
            " warmup=", warmup,
            " samples=", samples,
            " component_samples=", component_samples,
        )
        for rank in ranks
            tag = get(tags, rank, nothing)
            tag === nothing && error("No cache tag is registered for rank $rank.")
            cp = load_cp_decomposition(
                joinpath(cp_dir, "eslami_190_cp_$(tag).jld2"),
                joinpath(cp_dir, "eslami_190_cp_$(tag).toml"))
            profile_rank!(
                io, rank, cp, time, initial, dz,
                warmup, samples, component_samples, output_dir)
        end
    end
    open(joinpath(output_dir, "context.txt"), "w") do io
        println(io, "gpu=", CUDA.name(CUDA.device()))
        println(io, "nt=", nt)
        println(io, "time_window_ps=", window_ps)
        println(io, "dz_m=", dz)
        println(io, "precision=mixed")
        println(io, "backend=cuda_cp_optimized")
        println(io, "warmup=", warmup)
        println(io, "samples=", samples)
        println(io, "component_samples=", component_samples)
        println(io, "ranks=", join(ranks, ','))
        println(io, "launch_capture=", capture)
    end
    println("stage=complete output_dir=", output_dir)
end

main()
