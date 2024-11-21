function utility = gen_utility(solver)

    if length(solver.FOV) ~= solver.dimension
        FOV = ones(1, solver.dimension) * solver.FOV;
    else
        FOV = solver.FOV;
    end

    if length(solver.pitch) ~= solver.dimension
        pitch = ones(1, solver.dimension) * solver.pitch;
    else
        pitch = solver.pitch;
    end

    Nsize = floor(FOV ./ pitch / 2) * 2 + 1;
    dim = length(Nsize);

    %image space / fourier space coordinate information
    utility=struct('image_space',[],'fourier_space',[]);
    utility.size = Nsize;
    utility.image_space = struct;
    utility.image_space.res = num2cell(pitch);
    utility.image_space.coor = cell(1,dim);
    
    utility.fourier_space = struct;
    utility.fourier_space.res = num2cell(1./(pitch(1,dim).*Nsize));
    utility.fourier_space.coor = cell(1,dim);

    space_type_list = {'image_space','fourier_space'};
    dxs = [];
    dkxs = [];
    for space_type_idx = 1:2
        space_type = space_type_list{space_type_idx};
        space_res = utility.(space_type).res;
        for dim0 = 1:dim
            coor_axis = single(ceil(-utility.size(dim0)/2):ceil(utility.size(dim0)/2-1));
            coor_axis = coor_axis*space_res{dim0};
            coor_axis = reshape(coor_axis, circshift([1 1 utility.size(dim0)],dim0));
            if solver.use_GPU
                coor_axis = gpuArray(coor_axis);
            end
            if strcmp(solver.dataformat, 'double')
                coor_axis = double(coor_axis);
            else
                coor_axis = single(coor_axis);
            end
            utility.(space_type).coor{dim0} = coor_axis;
            if space_type_idx == 1
                dxs = [dxs space_res{dim0}];
            else
                dkxs = [dkxs space_res{dim0}];
            end
        end
    end
    utility.image_space.dxs = dxs;
    utility.image_space.dkxs = dkxs;
    
    if dim == 3
        utility.fourier_space.coorxy=sqrt(...
            (utility.fourier_space.coor{1}).^2+...
            (utility.fourier_space.coor{2}).^2);
    elseif dim == 2
        utility.fourier_space.coorxy=sqrt(...
            (utility.fourier_space.coor{1}).^2);
    elseif dim == 1
        utility.fourier_space.coorxy=sqrt(...
            (utility.fourier_space.coor{1}).^2)*0;
    end

    %other
    utility.Nsize = Nsize;
    utility.lambda=solver.wavelength0;
    utility.nm=solver.RI_bg;
    utility.k0=1/utility.lambda;
    utility.k0_nm=utility.nm*utility.k0;
    utility.kmax=solver.NA*utility.k0;
    utility.NA_circle=utility.fourier_space.coorxy<utility.kmax;
    utility.k3=(utility.k0_nm).^2 - (utility.fourier_space.coorxy).^2;
    utility.k3(utility.k3<0)=0;
    utility.k3=sqrt(utility.k3);
    utility.dV=prod([utility.image_space.res{1:dim}]);
    utility.dVk=1/utility.dV;
    utility.refocusing_kernel=2i * pi * utility.k3;
    utility.cos_theta=utility.k3 / utility.k0_nm;



    


end