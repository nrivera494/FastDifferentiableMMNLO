ad_real_type(::Type{<:Complex{T}}) where {T} = T
ad_real_type(::Type{T}) where {T} = T
ad_primal(x) = x
ad_primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)

function dense_transform_dim1(x::AbstractArray{T,N}, inverse::Bool) where {T,N}
    nt = size(x, 1)
    sign = inverse ? one(real(one(T))) : -one(real(one(T)))
    scale = inverse ? inv(real(one(T)) * nt) : one(real(one(T)))
    return map(CartesianIndices(size(x))) do I
        n = I[1] - 1
        tail = Base.tail(Tuple(I))
        sum(map(0:nt-1) do k
            phase = sign * 2π * n * k / nt
            cis(phase) * x[(k + 1, tail...)...]
        end) * scale
    end
end

function forward_fft_ad(x; dims=1)
    dims == 1 || error("AD fallback FFT currently supports dims=1 only.")
    R = ad_real_type(eltype(x))
    return R <: Union{Float32,Float64} ? forward_fft(x, dims=dims) :
           dense_transform_dim1(x, false)
end

function inverse_fft_ad(x; dims=1)
    dims == 1 || error("AD fallback FFT currently supports dims=1 only.")
    R = ad_real_type(eltype(x))
    return R <: Union{Float32,Float64} ? inverse_fft(x, dims=dims) :
           dense_transform_dim1(x, true)
end

function nonlinear_term_ad(at::Array{Complex{T},3}, n2_prefactor::AbstractVector,
                           srsk::SRSKInfo, haw::AbstractVector,
                           hbw::AbstractVector, include_raman::Bool) where {T}
    nt, planes, nm = size(at)
    kerr_modes = map(1:nm) do i1
        js = findall(j -> srsk.sk_indices[1, j] == i1, axes(srsk.sk_indices, 2))
        terms = map(js) do j
            i2, i3, i4 = srsk.sk_indices[2, j], srsk.sk_indices[3, j], srsk.sk_indices[4, j]
            srsk.sk[j] .* at[:, :, i2] .* at[:, :, i3] .* conj.(at[:, :, i4])
        end
        isempty(terms) ? zeros(Complex{T}, nt, planes) : reduce(+, terms)
    end
    nonlinear = cat(kerr_modes...; dims=3)

    if include_raman && !isempty(haw)
        raman_modes = map(1:nm) do i1
            i2_terms = map(1:nm) do i2
                js = findall(j -> srsk.sra_indices[1, j] == i1 &&
                                  srsk.sra_indices[2, j] == i2,
                             axes(srsk.sra_indices, 2))
                ra_raw_terms = map(js) do j
                    i3, i4 = srsk.sra_indices[3, j], srsk.sra_indices[4, j]
                    srsk.sra[j] .* at[:, :, i3] .* conj.(at[:, :, i4])
                end
                ra_raw = isempty(ra_raw_terms) ? zeros(Complex{T}, nt, planes) :
                         reduce(+, ra_raw_terms)
                ra = forward_fft_ad(haw .* inverse_fft_ad(ra_raw, dims=1), dims=1)
                ra .* at[:, :, i2]
            end
            reduce(+, i2_terms)
        end
        nonlinear = nonlinear .+ cat(raman_modes...; dims=3)

        if !isempty(hbw)
            anisotropic_modes = map(1:nm) do i1
                i2_terms = map(1:nm) do i2
                    js = findall(j -> srsk.srb_indices[1, j] == i1 &&
                                      srsk.srb_indices[2, j] == i2,
                                 axes(srsk.srb_indices, 2))
                    rb_raw_terms = map(js) do j
                        i3, i4 = srsk.srb_indices[3, j], srsk.srb_indices[4, j]
                        srsk.srb[j] .* at[:, :, i3] .* conj.(at[:, :, i4])
                    end
                    rb_raw = isempty(rb_raw_terms) ? zeros(Complex{T}, nt, planes) :
                             reduce(+, rb_raw_terms)
                    rb = forward_fft_ad(hbw .* inverse_fft_ad(rb_raw, dims=1), dims=1)
                    rb .* at[:, :, i2]
                end
                reduce(+, i2_terms)
            end
            nonlinear = nonlinear .+ cat(anisotropic_modes...; dims=3)
        end
    end

    nw = inverse_fft_ad(nonlinear, dims=1)
    return nw .* reshape(n2_prefactor, :, 1, 1)
end

function n_op_rk4ip_ad(aw::Matrix{Complex{T}}, sim::Simulation,
                       n2_prefactor::AbstractVector, srsk::SRSKInfo,
                       haw, hbw, at_noise::AbstractMatrix) where {T}
    at = forward_fft_ad(aw, dims=1) .+ at_noise
    nl = nonlinear_term_ad(reshape(at, size(at, 1), 1, size(at, 2)),
                           n2_prefactor, srsk, haw, hbw, sim.include_Raman)
    return reshape(nl, size(aw, 1), size(aw, 2))
end

function step_rk4ip_nogain_ad(a0w::Matrix{Complex{T}}, sim::Simulation,
                              n2_prefactor::AbstractVector, srsk::SRSKInfo,
                              d_op::AbstractMatrix, haw, hbw,
                              at_noise::AbstractMatrix, a5_1=nothing) where {T}
    half = exp.(d_op .* (sim.dz / 2))
    a_ip = half .* a0w
    k5prev = a5_1 === nothing ?
             n_op_rk4ip_ad(a0w, sim, n2_prefactor, srsk, haw, hbw, at_noise) :
             a5_1
    k1 = half .* k5prev
    k2 = n_op_rk4ip_ad(a_ip .+ k1 .* (sim.dz / 2), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    k3 = n_op_rk4ip_ad(a_ip .+ k2 .* (sim.dz / 2), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    k4 = n_op_rk4ip_ad(half .* (a_ip .+ k3 .* sim.dz), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    a1w = half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (sim.dz / 6)) .+
           k4 .* (sim.dz / 6)
    k5 = n_op_rk4ip_ad(a1w, sim, n2_prefactor, srsk, haw, hbw, at_noise)
    return a1w, k5
end

function propagate_ad(fiber::Fiber, initial_condition::InitialCondition{T},
                      sim::Simulation) where {T}
    sim.gain_model == 0 || error("Automatic differentiation supports passive or linear-gain-free propagation only.")
    lowercase(sim.step_method) == "rk4ip" ||
        error("Automatic differentiation currently supports fixed-step RK4IP only.")
    fields0 = last_fields(initial_condition)
    nt, nm = size(fields0)
    CT = T <: AbstractFloat ? T : Float64
    fiber_t = Fiber{CT}(; betas=CT.(fiber.betas), sr=CT.(fiber.sr),
                        L0=CT(fiber.L0), n2=CT(fiber.n2),
                        material=fiber.material,
                        mfd=fiber.mfd === nothing ? nothing : CT(fiber.mfd),
                        mm_folder=fiber.mm_folder, fr=CT(fiber.fr))
    sim_t = Simulation{CT}(; lambda0=CT(sim.lambda0), f0=CT(sim.f0),
                           dz=CT(sim.dz), save_period=CT(sim.save_period),
                           midx=sim.midx, scalar=sim.scalar,
                           ellipticity=CT(sim.ellipticity),
                           include_Raman=sim.include_Raman,
                           gain_model=0,
                           pulse_centering=sim.pulse_centering,
                           progress_bar=sim.progress_bar,
                           step_method="RK4IP",
                           cs=sim.cs, cs_model=sim.cs_model,
                           betas=sim.betas === nothing ? nothing : CT.(sim.betas))
    fiber2, save_period, nsteps, d_op, sim_betas, n2_prefactor, haw, hbw, srsk, at_noise =
        ChainRulesCore.ignore_derivatives() do
            f2 = expand_betas_for_polarization(fiber_t, sim_t, nm)
            sp = sim_t.save_period == 0 ? f2.L0 : sim_t.save_period
            ns = round(Int, f2.L0 / sim_t.dz)
            dt_const = CT(ad_primal(initial_condition.dt))
            omega = ifft_frequency_grid(nt, dt_const)
            zero_fields = zeros(Complex{CT}, nt, nm)
            dop, sb = calc_d_op(f2, sim_t, nt, dt_const, omega, zero_fields)
            damp = CT.(damped_freq_window(nt))
            c = CT(2.99792458e-4)
            prefactor = Complex{T}.(1im .* f2.n2 .* (omega .+ 2π * sim_t.f0) ./ c .* damp)
            f3, ha, hb = raman_model(f2, sim_t, nt, dt_const)
            s = calc_srsk(f3, sim_t, size(f3.sr, 1))
            noise = zeros(Complex{CT}, nt, nm)
            f2, sp, ns, dop, sb, prefactor, ha, hb, s, noise
        end
    aw = inverse_fft_ad(fields0, dims=1)
    a5 = nothing
    for _ in 1:nsteps
        aw, a5 = step_rk4ip_nogain_ad(aw, sim_t, n2_prefactor, srsk, d_op,
                                      haw, hbw, at_noise, a5)
    end
    out = forward_fft_ad(aw, dims=1)
    return PropagationOutput{T}(; z=[zero(initial_condition.dt),
                                     zero(initial_condition.dt) + save_period],
                                fields=cat(fields0, out; dims=3),
                                dt=initial_condition.dt, betas=sim_betas,
                                dz=[zero(initial_condition.dt),
                                    zero(initial_condition.dt) + sim_t.dz],
                                t_delay=zeros(T, 2))
end

function _filtered_photon_number_ad(fields, dt, f0, filter; shifted::Bool=true)
    a = _field_matrix(fields)
    aw = inverse_fft_ad(a, dims=1)
    shifted && (aw = fftshift(aw, 1))
    scale = ChainRulesCore.ignore_derivatives() do
        weights = photon_bin_weights(size(a, 1), dt, f0; shifted=shifted)
        filt = _filter_matrix(filter, size(a))
        reshape(weights, :, 1) .* filt
    end
    return real(sum(scale .* abs2.(aw)))
end

function _fourier_wirtinger_from_time(dA, dA_conj; shifted::Bool=true)
    nt = size(dA, 1)
    dAw = forward_fft_ad(dA, dims=1)
    dAw_conj = nt .* inverse_fft_ad(dA_conj, dims=1)
    if shifted
        dAw = fftshift(dAw, 1)
        dAw_conj = fftshift(dAw_conj, 1)
    end
    return dAw, dAw_conj
end

function _photon_fourier_wirtinger_from_fourier(dAw, dAw_conj, dt, f0; shifted::Bool=true)
    weights = ChainRulesCore.ignore_derivatives() do
        photon_bin_weights(size(dAw, 1), dt, f0; shifted=shifted)
    end
    scale = reshape(sqrt.(weights), :, 1)
    return dAw ./ scale, dAw_conj ./ scale
end

function output_photon_number(fiber::Fiber, initial_field, dt, sim::Simulation,
                              filter; shifted::Bool=true)
    fields0 = _field_matrix(initial_field)
    T = ad_real_type(eltype(fields0))
    ic = InitialCondition{T}(; dt=T(dt),
                             fields=reshape(fields0, size(fields0, 1),
                                            size(fields0, 2), 1))
    out = propagate_ad(fiber, ic, sim)
    return _filtered_photon_number_ad(out.fields[:, :, end], dt, sim.f0, filter;
                                      shifted=shifted)
end

function output_photon_number_gradient(fiber::Fiber, initial_field, dt,
                                       sim::Simulation, filter;
                                       return_fourier::Bool=false,
                                       return_photon_fourier::Bool=false,
                                       shifted::Bool=true)
    objective(a) = output_photon_number(fiber, a, dt, sim, filter; shifted=shifted)
    n = objective(initial_field)
    g = Zygote.gradient(objective, initial_field)[1]
    dA = 0.5 .* conj.(g)
    dA_conj = 0.5 .* g
    fourier = nothing
    photon_fourier = nothing
    if return_fourier || return_photon_fourier
        dAw, dAw_conj = _fourier_wirtinger_from_time(dA, dA_conj; shifted=shifted)
        fourier = (; dAw=dAw, dAw_conj=dAw_conj)
        if return_photon_fourier
            dB, dB_conj = _photon_fourier_wirtinger_from_fourier(dAw, dAw_conj,
                                                                 dt, sim.f0;
                                                                 shifted=shifted)
            photon_fourier = (; dB=dB, dB_conj=dB_conj)
        end
    end
    return (; photon_number=n, dA=dA, dA_conj=dA_conj,
            zygote_gradient=g, fourier=fourier, photon_fourier=photon_fourier)
end
