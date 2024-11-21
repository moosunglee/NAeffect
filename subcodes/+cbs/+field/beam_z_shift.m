function E_in = beam_z_shift(solver, E_in_2D, shift0, varargin)

    p = inputParser;
    p.addParameter('pad', []);
    p.parse(varargin{:});

    padding = p.Results.pad;

    if solver.dimension == 3
        E_in_2D = reshape(E_in_2D, [size(E_in_2D,1), size(E_in_2D,2), 1, 3]);
    elseif solver.dimension == 2
        E_in_2D = reshape(E_in_2D, [size(E_in_2D,1), 1, 1, 3]);
    else
        error('There is no need to shift the beam for 1D.')
    end

    if solver.use_GPU
        E_in_2D = gpuArray(E_in_2D);
    end

    refocusing_kernel = solver.utility.refocusing_kernel;
    NA_circle = solver.utility.NA_circle;
    if ~isempty(padding)
        E_in_2D = padarray(E_in_2D,[padding,padding,0,0],0);
        Nsize = size(E_in_2D,1:2);
        dim = length(Nsize);
        dks = 1./(solver.utility.image_space.dxs(1:2).*Nsize);
        for dim0 = 1:dim
            coor_axis = single(ceil(-Nsize(dim0)/2):ceil(Nsize(dim0)/2-1));
            coor_axis = coor_axis*dks(dim0);
            coor_axis = reshape(coor_axis, circshift([1 1 Nsize(dim0)],dim0));
            if solver.use_GPU
                coor_axis = gpuArray(coor_axis);
            end
            coor{dim0} = coor_axis;
        end
        coorxy=sqrt( (coor{1}).^2 + (coor{2}).^2 );
        NA_circle = coorxy < 1/solver.wavelength0;

        k3=(solver.utility.k0_nm).^2 - (coorxy).^2;
        k3(k3<0)=0;
        k3=sqrt(k3);
        refocusing_kernel=2i * pi * k3;
    end

    shift0 = reshape(shift0, 1,1,[]);

    E_in_2D = cbs.util.fftshift2(fft2(cbs.util.ifftshift2(E_in_2D)));
   

    z_shifter = exp(refocusing_kernel .* (shift0+solver.utility.image_space.coor{end})).*NA_circle;
    E_in =E_in_2D.* z_shifter;
    if solver.dimension == 3
        E_in = cbs.util.fftshift2(ifft2(cbs.util.ifftshift2(E_in)));
    else
        E_in = fftshift(ifft(ifftshift(E_in)));
    end

    if ~isempty(padding)
        E_in = E_in(padding+1:end-padding,padding+1:end-padding,:,:);
    end

    E_in = gather(squeeze(E_in));

end