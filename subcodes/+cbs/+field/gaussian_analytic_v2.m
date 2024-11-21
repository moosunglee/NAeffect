function E_in = gaussian_analytic_v2(solver, varargin)

    p = inputParser;
    p.addParameter('polarization', [1 1i]); % default: x polarization
    p.addParameter('NA', solver.NA);
    p.addParameter('Ntheta', 101)
    p.addParameter('Nstep', 20)
    p.addParameter('dz', 0)
    p.addParameter('w0_NA', [])
    p.parse(varargin{:}); 

    polarization = p.Results.polarization;
    NA_test = p.Results.NA;
    Ntheta  = p.Results.Ntheta;
    Nstep   = p.Results.Nstep; 
    dz = p.Results.dz;
    w0_NA = p.Results.w0_NA;

%%
    coor = solver.utility.image_space.coor;
    dxs = solver.utility.image_space.dxs;

    if solver.dimension < 3
        for idx = solver.dimension+1:3
            coor{idx} = 0;
            dxs(idx) = 1;
        end
    end
    coor{3} = coor{3} - dz;
    [II,JJ] = ndgrid(coor{1}, coor{2});

    rho = sqrt(II.^2 + JJ.^2);

    rho_inv = 1 ./ rho; 
    rho_inv(isnan(rho_inv)) = 1/sqrt(2);
    rho_inv(isinf(rho_inv)) = 1/sqrt(2);

    cos_IJ = II .* rho_inv;
    sin_IJ = JJ .* rho_inv;

    cos_2IJ = 2 * cos_IJ.^2 -1;
    sin_2IJ = 2 * sin_IJ .* cos_IJ;

    sinthetas = 0:NA_test/Ntheta:NA_test;
    sinthetas = reshape(sinthetas,1,1,1,[]);
    costhetas = sqrt(1 - sinthetas.^2);

    k = 2*pi * solver.utility.k0_nm;
    
    besselj_0 = besselj(0, k*rho.*sinthetas);
    besselj_1 = besselj(1, k*rho.*sinthetas);
    besselj_2 = besselj(2, k*rho.*sinthetas);
    step_section = Nstep;
    num_section = ceil(length(sinthetas)/step_section);

    if ~isempty(w0_NA)
        fw_sintheta = exp(-sinthetas.^2 ./ w0_NA^2/2); % Division by 2 because it is for amplitude.
    else
        fw_sintheta = sinthetas*0+1;
    end

    for j2 = 1:num_section
        ind_div = step_section * (j2-1) + (1:step_section);
        ind_div(ind_div>length(sinthetas)) = [];

        sintheta_intrange = sinthetas(ind_div);
        costhetas_intrange = costhetas(ind_div);
        fw_sintheta_intrange = fw_sintheta(ind_div);

        I00_0 = fw_sintheta_intrange .* sintheta_intrange./sqrt(costhetas_intrange) .* (1 + costhetas_intrange) .*...
        besselj_0(:,:,:,ind_div) .* exp(1i.*k.*coor{3}.*costhetas_intrange);
        I01_0 = fw_sintheta_intrange .* sintheta_intrange.^2 ./ sqrt(costhetas_intrange) .*...
        besselj_1(:,:,:,ind_div).* exp(1i.*k.*coor{3}.*costhetas_intrange);
        I02_0 = fw_sintheta_intrange .* sintheta_intrange./sqrt(costhetas_intrange) .* (1 - costhetas_intrange) .*...
            besselj_2(:,:,:,ind_div).* exp(1i.*k.*coor{3}.*costhetas_intrange);

        I00_0(isnan(I00_0)) = 0;
        I01_0(isnan(I01_0)) = 0;
        I02_0(isnan(I02_0)) = 0;

        if j2 == 1
            I00 = sum(I00_0,4);
            I01 = sum(I01_0,4);
            I02 = sum(I02_0,4);
        else
            I00 = I00 + sum(I00_0,4);
            I01 = I01 + sum(I01_0,4);
            I02 = I02 + sum(I02_0,4);
        end
    end
    E1 = repmat(I00*0, 1,1,1,3);

    E1(:,:,:,1) = I00 + I02 .* cos_2IJ;
    E1(:,:,:,2) = I02 .* sin_2IJ;
    E1(:,:,:,3) = -2i.* I01 .* cos_IJ;
    E2 = E1;
    E2(:,:,:,1) = -rot90(E1(:,:,:,2));
    E2(:,:,:,2) =  rot90(E1(:,:,:,1));
    E2(:,:,:,3) =  rot90(E1(:,:,:,3));

    E_in = gather(squeeze(E1 * polarization(1) + E2 * polarization(2)));

end