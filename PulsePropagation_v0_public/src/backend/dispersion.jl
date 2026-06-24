function expand_betas_for_polarization(fiber::Fiber{T}, sim::Simulation{T}, num_modes::Int) where {T}
    if sim.scalar || size(fiber.betas, 2) == num_modes
        return fiber
    end
    if size(fiber.betas, 2) * 2 != num_modes
        error("Cannot expand betas with $(size(fiber.betas, 2)) columns to $num_modes modes.")
    end
    betas = Array{T}(undef, size(fiber.betas, 1), num_modes)
    for j in axes(fiber.betas, 2)
        betas[:, 2j - 1] = fiber.betas[:, j]
        betas[:, 2j] = fiber.betas[:, j]
    end
    return Fiber{T}(; betas=betas, sr=fiber.sr, L0=fiber.L0, n2=fiber.n2,
                    material=fiber.material, mfd=fiber.mfd, mm_folder=fiber.mm_folder, fr=fiber.fr)
end

function calc_d_op(fiber::Fiber{T}, sim::Simulation{T}, nt::Integer, dt::T,
                   omega::AbstractVector{T}, fields::AbstractMatrix{Complex{T}}) where {T}
    betas = fiber.betas
    sim_betas = sim.betas
    if any(size(betas) .== nt)
        β = size(betas, 2) == nt ? permutedims(betas) : betas
        if sim_betas === nothing
            spectrum = sum(abs2, fftshift(inverse_fft(fields, dims=1), 1), dims=2)[:, 1]
            shifted_omega = fftshift(omega)
            omega0 = sum(shifted_omega .* spectrum) / sum(spectrum)
            omega_range = 2π / dt
            idx = findall((shifted_omega .> omega0 - omega_range / 5) .& (shifted_omega .< omega0 + omega_range / 5))
            fit_order = max(2, min(7, length(idx) - 1))
            coeff = reverse(polyfit_vander(shifted_omega[idx], real.(β[idx, 1]), fit_order))
            sim_betas = [coeff[1], coeff[2]]
        end
        dop = Complex{T}.(1im .* (ifftshift(β, 1) .- (sim_betas[1] .+ sim_betas[2] .* omega)))
    elseif size(betas, 1) < 20
        sim_betas = sim_betas === nothing ? collect(real.(betas[1:2, 1])) : sim_betas
        β = copy(betas)
        β[1, :] .-= sim_betas[1]
        β[2, :] .-= sim_betas[2]
        dop = zeros(Complex{T}, nt, size(β, 2))
        for n in 0:size(β, 1)-1
            dop .+= (1im / factorial(n)) .* (omega .^ n) .* permutedims(β[n + 1, :])
        end
    else
        error("fiber.betas must be either Nt-by-modes or coefficient-by-modes.")
    end
    sim.cs > 1 && (dop ./= sim.cs)
    return dop, collect(T, sim_betas)
end

function polyfit_vander(x, y, degree::Integer)
    v = reduce(hcat, (x .^ k for k in 0:degree))
    return v \ y
end
