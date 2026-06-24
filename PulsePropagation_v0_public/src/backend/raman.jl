function raman_model(fiber::Fiber{T}, sim::Simulation{T}, nt::Integer, dt::T) where {T}
    if !sim.include_Raman
        return fiber, Complex{T}[], Complex{T}[]
    end
    t = (dt / sim.cs) .* collect(T, 0:nt-1)
    if fiber.material == "agarwal"
        fr = fiber.fr == 0 ? T(0.18) : fiber.fr
        centered_t = (collect(T, 0:nt-1) .- floor(T, nt / 2)) .* (dt / sim.cs) .* T(1e-12)
        tau1 = T(12.2)
        tau2 = T(32.0)
        hrt = zeros(T, nt)
        idx = centered_t .>= zero(T)
        tpos = centered_t[idx]
        hrt[idx] .= fr .* (dt / sim.cs) .* T(1e3) .* (tau1^2 + tau2^2) ./ (tau1 * tau2^2) .*
                    exp.(-tpos .* T(1e15) ./ tau2) .* sin.(tpos .* T(1e15) ./ tau1)
        haw = inverse_fft(ifftshift(hrt), dims=1) .* nt
        fiber2 = Fiber{T}(; betas=fiber.betas, sr=fiber.sr, L0=fiber.L0, n2=fiber.n2,
                          material=fiber.material, mfd=fiber.mfd, mm_folder=fiber.mm_folder, fr=fr)
        return fiber2, Complex{T}.(haw), Complex{T}[]
    elseif fiber.material == "silica"
        fr = T(0.245)
        fa = T(0.75)
        fb = T(0.21)
        fc = T(0.04)
        t1 = T(12.2e-3)
        t2 = T(32e-3)
        tb = T(96e-3)
        ha_model = ((t1^2 + t2^2) / (t1 * t2^2)) .* exp.(-t ./ t2) .* sin.(t ./ t1)
        hb_model = ((2tb .- t) ./ tb^2) .* exp.(-t ./ tb)
        ha = fa .* ha_model
        hb = fc .* ha_model .+ fb .* hb_model
        if sim.scalar
            ha .+= sim.ellipticity == zero(T) ? hb : hb ./ T(2)
            include_anisotropic = false
        else
            include_anisotropic = true
        end
    elseif fiber.material == "chalcogenide"
        fr = T(0.115)
        t1 = T(15.2e-3)
        t2 = T(230.5e-3)
        ha = ((t1^2 + t2^2) / (t1 * t2^2)) .* exp.(-t ./ t2) .* sin.(t ./ t1)
        hb = zeros(T, nt)
        include_anisotropic = !sim.scalar && any(!iszero, hb)
    else
        error("Raman model currently supports silica and chalcogenide in Julia parity layer.")
    end
    ha ./= sim.cs
    hb ./= sim.cs
    haw = fr .* inverse_fft(ha, dims=1) .* nt .* dt
    hbw = include_anisotropic ? fr .* inverse_fft(hb, dims=1) .* nt .* dt : Complex{T}[]
    fiber2 = Fiber{T}(; betas=fiber.betas, sr=fiber.sr, L0=fiber.L0, n2=fiber.n2,
                      material=fiber.material, mfd=fiber.mfd, mm_folder=fiber.mm_folder, fr=fr)
    return fiber2, Complex{T}.(haw), Complex{T}.(hbw)
end

function raman_adjoint_model_kernels(fiber::Fiber{T}, sim::Simulation{T},
                                     nt::Integer, dt::T) where {T}
    sim.include_Raman || return fiber, Complex{T}[], Complex{T}[]

    t = (dt / sim.cs) .* collect(T, 0:nt-1)
    if fiber.material == "silica"
        fr = T(0.245)
        fa = T(0.75)
        fb = T(0.21)
        fc = T(0.04)
        t1 = T(12.2e-3)
        t2 = T(32e-3)
        tb = T(96e-3)
        ha_model = ((t1^2 + t2^2) / (t1 * t2^2)) .* exp.(-t ./ t2) .* sin.(t ./ t1)
        hb_model = ((2tb .- t) ./ tb^2) .* exp.(-t ./ tb)
        ha = fa .* ha_model
        hb = fc .* ha_model .+ fb .* hb_model
        if sim.scalar
            ha .+= sim.ellipticity == zero(T) ? hb : hb ./ T(2)
            include_anisotropic = false
        else
            include_anisotropic = true
        end
    elseif fiber.material == "chalcogenide"
        fr = T(0.115)
        t1 = T(15.2e-3)
        t2 = T(230.5e-3)
        ha = ((t1^2 + t2^2) / (t1 * t2^2)) .* exp.(-t ./ t2) .* sin.(t ./ t1)
        hb = zeros(T, nt)
        include_anisotropic = !sim.scalar && any(!iszero, hb)
    else
        error("Model Raman adjoint kernels support silica and chalcogenide.")
    end

    ha ./= sim.cs
    hb ./= sim.cs
    haw = forward_fft(ifftshift(fr .* dt .* ha), dims=1)
    hbw = include_anisotropic ? forward_fft(ifftshift(fr .* dt .* hb), dims=1) : Complex{T}[]
    fiber2 = Fiber{T}(; betas=fiber.betas, sr=fiber.sr, L0=fiber.L0, n2=fiber.n2,
                      material=fiber.material, mfd=fiber.mfd, mm_folder=fiber.mm_folder, fr=fr)
    return fiber2, Complex{T}.(haw), Complex{T}.(hbw)
end
