function n_op_rk4ip(aw::Matrix{Complex{T}}, sim::Simulation{T},
                    n2_prefactor::Vector{Complex{T}}, srsk::SRSKInfo{T},
                    haw, hbw, at_noise::Union{Nothing,Matrix{Complex{T}}}) where {T}
    at = forward_fft(aw, dims=1)
    at_noise === nothing || (at .+= at_noise)
    nl = nonlinear_term(reshape(at, size(at, 1), 1, size(at, 2)),
                        n2_prefactor, srsk, haw, hbw, sim.include_Raman)
    return reshape(nl, size(aw, 1), size(aw, 2))
end

function n_op_rk4ip_cp(aw::Matrix{Complex{T}}, sim::Simulation{T},
                       n2_prefactor::Vector{Complex{T}}, srsk_cp, f_raman,
                       haw, hbw,
                       at_noise::Union{Nothing,Matrix{Complex{T}}}) where {T}
    at = forward_fft(aw, dims=1)
    at_noise === nothing || (at .+= at_noise)
    nl = nonlinear_term_cp(reshape(at, size(at, 1), 1, size(at, 2)),
                           n2_prefactor, srsk_cp, f_raman, haw, hbw,
                           sim.include_Raman)
    return reshape(nl, size(aw, 1), size(aw, 2))
end

function step_rk4ip_nogain(a0w::Matrix{Complex{T}}, sim::Simulation{T},
                           n2_prefactor::Vector{Complex{T}}, srsk::SRSKInfo{T},
                           d_op::Matrix{Complex{T}}, haw, hbw,
                           at_noise::Union{Nothing,Matrix{Complex{T}}},
                           a5_1=nothing) where {T}
    half = exp.(d_op .* (sim.dz / 2))
    a_ip = half .* a0w
    k5prev = a5_1 === nothing ?
             n_op_rk4ip(a0w, sim, n2_prefactor, srsk, haw, hbw, at_noise) :
             a5_1
    k1 = half .* k5prev
    k2 = n_op_rk4ip(a_ip .+ k1 .* (sim.dz / 2), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    k3 = n_op_rk4ip(a_ip .+ k2 .* (sim.dz / 2), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    k4 = n_op_rk4ip(half .* (a_ip .+ k3 .* sim.dz), sim, n2_prefactor, srsk, haw, hbw, at_noise)
    a1w = half .* (a_ip .+ (k1 .+ 2 .* k2 .+ 2 .* k3) .* (sim.dz / 6)) .+
           k4 .* (sim.dz / 6)
    k5 = n_op_rk4ip(a1w, sim, n2_prefactor, srsk, haw, hbw, at_noise)
    return a1w, k5, true
end
