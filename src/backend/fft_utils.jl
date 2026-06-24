forward_fft(x; dims=1) = fft(x, dims)
inverse_fft(x; dims=1) = ifft(x, dims)

function ifft_frequency_grid(nt::Integer, dt)
    return 2π .* ifftshift(collect(range(-floor(Int, nt / 2), floor(Int, (nt - 1) / 2), length=nt))) ./ (nt * dt)
end

function frequency_grid_THz(nt::Integer, dt; shifted::Bool=true)
    f = collect(range(-floor(Int, nt / 2), floor(Int, (nt - 1) / 2), length=nt)) ./ (nt * dt)
    return shifted ? f : ifftshift(f)
end

frequency_grid_THz(fields::AbstractArray, dt; shifted::Bool=true) =
    frequency_grid_THz(size(fields, 1), dt; shifted=shifted)

function absolute_frequency_grid_THz(nt::Integer, dt, f0; shifted::Bool=true)
    return f0 .+ frequency_grid_THz(nt, dt; shifted=shifted)
end

absolute_frequency_grid_THz(fields::AbstractArray, dt, f0; shifted::Bool=true) =
    absolute_frequency_grid_THz(size(fields, 1), dt, f0; shifted=shifted)

function nonpositive_frequency_bins(nt::Integer, dt, f0; shifted::Bool=true)
    return absolute_frequency_grid_THz(nt, dt, f0; shifted=shifted) .<= 0
end

nonpositive_frequency_bins(fields::AbstractArray, dt, f0; shifted::Bool=true) =
    nonpositive_frequency_bins(size(fields, 1), dt, f0; shifted=shifted)

function frequency_grid_diagnostics(nt::Integer, dt, f0; shifted::Bool=true)
    f_abs = absolute_frequency_grid_THz(nt, dt, f0; shifted=shifted)
    bad = f_abs .<= 0
    return (; has_nonpositive=any(bad),
            nonpositive_count=count(bad),
            nonpositive_indices=findall(bad),
            min_frequency_THz=minimum(f_abs),
            max_frequency_THz=maximum(f_abs))
end

frequency_grid_diagnostics(fields::AbstractArray, dt, f0; shifted::Bool=true) =
    frequency_grid_diagnostics(size(fields, 1), dt, f0; shifted=shifted)

function wavelength_grid_nm(nt::Integer, dt, f0; shifted::Bool=true)
    c_m_per_ps = 2.99792458e-4
    f_abs = absolute_frequency_grid_THz(nt, dt, f0; shifted=shifted)
    return c_m_per_ps .* 1e9 ./ f_abs
end

wavelength_grid_nm(fields::AbstractArray, dt, f0; shifted::Bool=true) =
    wavelength_grid_nm(size(fields, 1), dt, f0; shifted=shifted)

function centered_time_grid(nt::Integer, dt)
    return collect(range(-floor(Int, nt / 2), floor(Int, (nt - 1) / 2), length=nt)) .* dt
end

function damped_freq_window(nt::Integer)
    f = Float64.(ifftshift(collect(1:nt)))
    fc_idx = floor(Int, nt / 2) + 1
    fc = Float64(fc_idx)
    ffwhm = nt * 0.85
    f0 = ffwhm / (2 * sqrt(log(2)))
    gexpo = 40
    window = exp.(-((f .- fc) .^ gexpo) ./ (2 * f0^gexpo)) .^ 20
    window[fc_idx:end] .= 1
    return window
end

function _field_matrix(fields::AbstractVector)
    return reshape(fields, :, 1)
end

function _field_matrix(fields::AbstractMatrix)
    return fields
end

function _field_matrix(fields::AbstractArray{<:Complex,3})
    return fields[:, :, end]
end

function spectral_energy_density(fields, dt; shifted::Bool=true, sum_modes::Bool=true)
    a = _field_matrix(fields)
    nt = size(a, 1)
    aw = inverse_fft(a, dims=1)
    shifted && (aw = fftshift(aw, 1))
    density = abs2.(aw) .* (nt * dt)^2
    return sum_modes ? vec(sum(density; dims=2)) : density
end

function photon_bin_weights(nt::Integer, dt, f0; shifted::Bool=true)
    f_abs = absolute_frequency_grid_THz(nt, dt, f0; shifted=shifted)
    hbar = 1.054571817e-34
    omega_abs = 2π .* f_abs .* 1e12
    return ifelse.(f_abs .> 0, (nt * dt) * 1e-12 ./ (hbar .* omega_abs), 0)
end

photon_bin_weights(fields::AbstractArray, dt, f0; shifted::Bool=true) =
    photon_bin_weights(size(fields, 1), dt, f0; shifted=shifted)

function photon_normalized_field(aw, dt, f0; shifted::Bool=true)
    a = _field_matrix(aw)
    weights = photon_bin_weights(size(a, 1), dt, f0; shifted=shifted)
    return sqrt.(weights) .* a
end

function field_from_photon_normalized(bw, dt, f0; shifted::Bool=true)
    b = _field_matrix(bw)
    weights = photon_bin_weights(size(b, 1), dt, f0; shifted=shifted)
    return b ./ sqrt.(weights)
end

function photon_spectrum(fields, dt, f0; shifted::Bool=true, sum_modes::Bool=true, normalization::Symbol=:bin)
    normalization in (:bin, :density) || error("normalization must be :bin or :density")
    energy_density_pJ_per_THz = spectral_energy_density(fields, dt; shifted=shifted, sum_modes=sum_modes)
    energy = normalization === :bin ? energy_density_pJ_per_THz ./ (size(_field_matrix(fields), 1) * dt) :
             energy_density_pJ_per_THz
    f_abs = absolute_frequency_grid_THz(size(_field_matrix(fields), 1), dt, f0; shifted=shifted)
    hbar = 1.054571817e-34
    omega_abs = 2π .* f_abs .* 1e12
    photon_energy = hbar .* omega_abs
    if sum_modes
        return ifelse.(f_abs .> 0, energy .* 1e-12 ./ photon_energy, 0)
    else
        denom = reshape(photon_energy, :, 1)
        valid = reshape(f_abs .> 0, :, 1)
        return ifelse.(valid, energy .* 1e-12 ./ denom, 0)
    end
end

function _filter_matrix(filter, field_size)
    nt, nm = field_size
    if filter isa AbstractVector
        length(filter) == nt || error("filter length must match the number of frequency bins")
        return reshape(filter, nt, 1)
    elseif filter isa AbstractMatrix
        size(filter) == (nt, nm) || error("matrix filter size must match the Fourier field size")
        return filter
    else
        error("filter must be a vector of length Nt or a matrix of size Nt x Nm")
    end
end

function photon_number_gradient(aw, dt, f0, filter; shifted::Bool=true, normalized::Bool=false)
    a = _field_matrix(aw)
    filt = _filter_matrix(filter, size(a))
    weights = normalized ? ones(real(eltype(a)), size(a, 1)) :
              photon_bin_weights(size(a, 1), dt, f0; shifted=shifted)
    scale = reshape(weights, :, 1) .* filt
    dA = scale .* conj.(a)
    dA_conj = scale .* a
    return dA, dA_conj
end

function filtered_photon_number(fields, dt, f0, filter; shifted::Bool=true)
    photons = photon_spectrum(fields, dt, f0; shifted=shifted, sum_modes=false)
    filt = _filter_matrix(filter, size(photons))
    return real(sum(filt .* photons))
end

l2_relative_error(a, b) = norm(vec(a .- b)) / max(norm(vec(b)), eps(real(eltype(b))))
