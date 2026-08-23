using CUDA
using PulsePropagation
using Statistics

CUDA.functional() || error("CUDA is not functional.")
CUDA.allowscalar(false)

const C_M_PER_PS = 2.99792458e-4
const OMEGA0 = 2pi * C_M_PER_PS / 1.55e-6
const CUDA_EXTENSION = Base.get_extension(
    PulsePropagation, :PulsePropagationCUDAExt)

function median_device_seconds(operation; samples=7)
    operation()
    CUDA.synchronize()
    measurements = Float64[]
    for _ in 1:samples
        push!(measurements, CUDA.@elapsed operation())
    end
    return median(measurements)
end

function profile_case(nt, transverse_count, raman)
    time = TimeGrid(nt, 10.0)
    grid = FullField3DGrid(
        time; Nx=transverse_count, Ny=transverse_count, dx=1.5e-6)
    beta0 = 8.0e6
    potential = Float32[
        -0.5f0 * Float32(beta0) * (2.0f3)^2 * Float32(x^2 + y^2)
        for x in grid.x, y in grid.y
    ]
    initial = fill(ComplexF32(2.0f7, -0.5f7), full_field_shape(grid))
    parameters = FullField3DParameters(
        grid;
        length=1.0e-3,
        beta0,
        potential,
        n2=2.3e-20,
        omega0=OMEGA0,
        raman,
    )

    CUDA.reclaim()
    memory_before = Int(CUDA.free_memory())
    workspace = CUDA_EXTENSION._cuda_full_field_workspace(
        initial, parameters, Float32)
    CUDA.synchronize()
    device_bytes = max(memory_before - Int(CUDA.free_memory()), 0)

    linear = () -> CUDA_EXTENSION._cuda_full_field_linear!(workspace, 1.0e-5)
    nonlinear = () -> CUDA_EXTENSION._cuda_full_field_nonlinear!(workspace, 1.0e-5)
    full_step = () -> begin
        CUDA_EXTENSION._cuda_full_field_nonlinear!(workspace, 1.0e-5)
        CUDA_EXTENSION._cuda_full_field_linear!(workspace, 1.0e-5)
    end
    linear_seconds = median_device_seconds(linear)
    nonlinear_seconds = median_device_seconds(nonlinear)
    step_seconds = median_device_seconds(full_step)
    return (;
        nt,
        transverse_count,
        physics=raman isa NoRaman ? "kerr" : "kerr_raman",
        linear_ms=1e3linear_seconds,
        nonlinear_ms=1e3nonlinear_seconds,
        step_ms=1e3step_seconds,
        device_mib=device_bytes / 2.0^20,
    )
end

println("nt,nx,physics,linear_ms,nonlinear_ms,step_ms,device_mib")
for (nt, transverse_count) in ((512, 32), (2048, 64), (8192, 64))
    for raman in (NoRaman(), AgarwalRaman())
        result = profile_case(nt, transverse_count, raman)
        println(join((
            result.nt,
            result.transverse_count,
            result.physics,
            result.linear_ms,
            result.nonlinear_ms,
            result.step_ms,
            result.device_mib,
        ), ','))
    end
end
