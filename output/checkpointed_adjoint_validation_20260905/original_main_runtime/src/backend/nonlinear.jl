function nonlinear_term(at::Array{Complex{T},3}, n2_prefactor::AbstractVector{Complex{T}},
                        srsk::SRSKInfo{T}, haw::AbstractVector{Complex{T}},
                        hbw::AbstractVector{Complex{T}}, include_raman::Bool) where {T}
    nt, planes, nm = size(at)
    kerr = zeros(Complex{T}, nt, planes, nm)
    for j in axes(srsk.sk_indices, 2)
        i1, i2, i3, i4 = srsk.sk_indices[:, j]
        kerr[:, :, i1] .+= srsk.sk[j] .* at[:, :, i2] .* at[:, :, i3] .* conj.(at[:, :, i4])
    end
    nonlinear = kerr
    if include_raman && !isempty(haw)
        ra = zeros(Complex{T}, nt, planes, nm, nm)
        for j in axes(srsk.sra_indices, 2)
            i1, i2, i3, i4 = srsk.sra_indices[:, j]
            ra[:, :, i1, i2] .+= srsk.sra[j] .* at[:, :, i3] .* conj.(at[:, :, i4])
        end
        for i1 in 1:nm, i2 in 1:nm
            ra[:, :, i1, i2] .= forward_fft(haw .* inverse_fft(ra[:, :, i1, i2], dims=1), dims=1)
        end
        for i1 in 1:nm, i2 in 1:nm
            nonlinear[:, :, i1] .+= ra[:, :, i1, i2] .* at[:, :, i2]
        end
        if !isempty(hbw)
            rb = zeros(Complex{T}, nt, planes, nm, nm)
            for j in axes(srsk.srb_indices, 2)
                i1, i2, i3, i4 = srsk.srb_indices[:, j]
                rb[:, :, i1, i2] .+= srsk.srb[j] .* at[:, :, i3] .* conj.(at[:, :, i4])
            end
            for i1 in 1:nm, i2 in 1:nm
                rb[:, :, i1, i2] .= forward_fft(hbw .* inverse_fft(rb[:, :, i1, i2], dims=1), dims=1)
            end
            for i1 in 1:nm, i2 in 1:nm
                nonlinear[:, :, i1] .+= rb[:, :, i1, i2] .* at[:, :, i2]
            end
        end
    end
    nw = inverse_fft(nonlinear, dims=1)
    for m in 1:nm
        nw[:, :, m] .*= n2_prefactor
    end
    return nw
end


function nonlinear_term_cp(at::Array{Complex{T},3}, n2_prefactor::AbstractVector{Complex{T}},
                        srsk_cp, f_raman, haw::AbstractVector{Complex{T}},
                        hbw::AbstractVector{Complex{T}}, include_raman::Bool) where {T}
    ## srsk_cp is the output of a gcp decomposition
    r_cp = length(srsk_cp.λ)
    nt, planes, nm = size(at)
    kerr_cp = zeros(Complex{T}, nt, planes, nm)
    raman_cp = zeros(Complex{T}, nt, planes, nm)
    nonlinear = zeros(Complex{T}, nt, planes, nm)

    λ_cp = srsk_cp.λ;
    U1 = srsk_cp.U[1];
    U2 = srsk_cp.U[2];
    U3 = srsk_cp.U[3];
    U4 = srsk_cp.U[4];
    A = at[:,1,:];
    Ac = conj(A);

    # for ii = 1:r_cp
    #     λr = λ_cp[ii]
    #     B2r = A * U2[:,ii]#sum(U2[:,ii] .* at[:,1,:], dims=2)
    #     B3r = A * U3[:,ii]#sum(U3[:,ii] .* at[:,1,:], dims=2)
    #     B4r = Ac * U4[:,ii]#sum(U4[:,ii] .* conj.(at[:,1,:]), dims=2)
    #     for i1 = 1:nm
    #         kerr_cp[:, 1, i1] .+= λr * U1[i1,ii] * (B2r[:] .* B3r[:] .* B4r[:]);
    #         nonlinear[:,1,i1] .= (1-f_raman) * kerr_cp[:,1,i1]
    #     end
    # end

    A = @view at[:, 1, :]          # Nt x Nm
    B2 = A * U2                    # Nt x R
    B3 = A * U3                    # Nt x R
    B4 = conj.(A) * U4             # Nt x R

    P = B2 .* B3 .* B4             # Nt x R
    WU1 = U1 .* reshape(λ_cp, 1, :) # Nm x R

    K = P * transpose(WU1)         # Nt x Nm
    nonlinear[:, 1, :] .= (1 - f_raman) .* K


    
    if include_raman && !isempty(haw)

    P = (B2) .* forward_fft(haw .* inverse_fft(B3 .* B4, dims=1),dims=1)  
    WU1 = U1 .* reshape(λ_cp, 1, :);
    K = P * transpose(WU1);   
    nonlinear[:,1,:] .+= K;

    # for ii = 1:r_cp
    #     λr = λ_cp[ii]
    #     B2r = A * U2[:,ii]#sum(U2[:,ii] .* at[:,1,:], dims=2)
    #     B3r = A * U3[:,ii]#sum(U3[:,ii] .* at[:,1,:], dims=2)
    #     B4r = Ac * U4[:,ii]#sum(U4[:,ii] .* conj.(at[:,1,:]), dims=2)
    #     for i1 = 1:nm
    #         raman_cp[:,1,i1] .+= λr * U1[i1,ii] * (B2r) .* forward_fft(haw .* inverse_fft(B3r .* B4r,dims=1),dims=1);
    #         #nonlinear[:,1,i1] .+= raman_cp[:,1,i1]; #no f_raman b/c its in haw
    #     end
    # end


        
        if !isempty(hbw)
            rb = zeros(Complex{T}, nt, planes, nm, nm)
            for j in axes(srsk.srb_indices, 2)
                i1, i2, i3, i4 = srsk.srb_indices[:, j]
                rb[:, :, i1, i2] .+= srsk.srb[j] .* at[:, :, i3] .* conj.(at[:, :, i4])
            end
            for i1 in 1:nm, i2 in 1:nm
                rb[:, :, i1, i2] .= forward_fft(hbw .* inverse_fft(rb[:, :, i1, i2], dims=1), dims=1)
            end
            for i1 in 1:nm, i2 in 1:nm
                nonlinear[:, :, i1] .+= rb[:, :, i1, i2] .* at[:, :, i2]
            end
        end
    end

    nw = inverse_fft(nonlinear, dims=1)
    for m in 1:nm
        nw[:, :, m] .*= n2_prefactor
    end

    return nw
end
