classdef calc_psf_trapfreq < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                matlab.ui.Figure
        TabGroup                matlab.ui.container.TabGroup
        ForwardmodeTab          matlab.ui.container.Tab
        rho_f_editfield         matlab.ui.control.NumericEditField
        wavelengthumLabel_28    matlab.ui.control.Label
        GPUSwitch               matlab.ui.control.Switch
        GPUSwitchLabel          matlab.ui.control.Label
        RI_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_20    matlab.ui.control.Label
        Dz_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_12    matlab.ui.control.Label
        Dy_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_11    matlab.ui.control.Label
        Dx_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_10    matlab.ui.control.Label
        FWHMnmLabel             matlab.ui.control.Label
        Wz_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_9     matlab.ui.control.Label
        Wy_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_8     matlab.ui.control.Label
        TrapfreqkHzLabel        matlab.ui.control.Label
        Wx_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_7     matlab.ui.control.Label
        zvoxel_f_editfield      matlab.ui.control.NumericEditField
        wavelengthumLabel_6     matlab.ui.control.Label
        xyvoxel_f_editfield     matlab.ui.control.NumericEditField
        wavelengthumLabel_5     matlab.ui.control.Label
        power_f_editfield       matlab.ui.control.NumericEditField
        wavelengthumLabel_4     matlab.ui.control.Label
        circpol_f_editfield     matlab.ui.control.NumericEditField
        wavelengthumLabel_3     matlab.ui.control.Label
        NA_f_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_2     matlab.ui.control.Label
        wavelength_f_editfield  matlab.ui.control.NumericEditField
        wavelengthumLabel       matlab.ui.control.Label
        InversemodeTab          matlab.ui.container.Tab
        rho_i_editfield         matlab.ui.control.NumericEditField
        wavelengthumLabel_29    matlab.ui.control.Label
        Dz_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_27    matlab.ui.control.Label
        Dy_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_26    matlab.ui.control.Label
        Dx_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_25    matlab.ui.control.Label
        FWHMumLabel_2           matlab.ui.control.Label
        SelectWavelength        matlab.ui.control.DropDown
        SelectLabel             matlab.ui.control.Label
        RI_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_24    matlab.ui.control.Label
        power_i_editfield       matlab.ui.control.NumericEditField
        wavelengthumLabel_19    matlab.ui.control.Label
        circpol_i_editfield     matlab.ui.control.NumericEditField
        wavelengthumLabel_18    matlab.ui.control.Label
        NA_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_17    matlab.ui.control.Label
        Wz_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_15    matlab.ui.control.Label
        Wy_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_14    matlab.ui.control.Label
        Wx_i_editfield          matlab.ui.control.NumericEditField
        wavelengthumLabel_13    matlab.ui.control.Label
        TrapfreqkHzLabel_2      matlab.ui.control.Label
    end

    
    properties (Access = private)
        NAs_theory; % Description
        circpol_ratios;
        Dxs_1064;
        Dys_1064;
        Dzs_1064;
        Dxs_1550;
        Dys_1550;
        Dzs_1550;
    end
    
    methods (Access = private)
        
        function run_inverse_psf(app)
		    % % % if isdeployed 
			% % %     % User is running an executable in standalone mode. 
			% % %     [~ , result] = system('set PATH');
			% % %     executableFolder = char(regexpi(result, 'Path=(.*?);', 'tokens', 'once'));
		    % % % else
			% % %     % User is running an m-file from the MATLAB integrated development environment (regular MATLAB).
            % % %     tmp_path = mfilename('fullpath');
            % % %     [executableFolder , ~ , ~] = fileparts(tmp_path);
            % % % end 
            % % % executableFolder
            if strcmp(app.SelectWavelength.Value, '1064 nm')
                Dxs = app.Dxs_1064;
                Dys = app.Dys_1064;
                Dzs = app.Dzs_1064;
                wavelength = 1.064;
            else
                Dxs = app.Dxs_1550;
                Dys = app.Dys_1550;
                Dzs = app.Dzs_1550;
                wavelength = 1.55;
            end
            RI_bg = 1; % Refractive index of the background
            c = 299792458;              % [m/s] Speed of light

            rho = app.rho_i_editfield.Value;                 % Particle density [kg/m^3]
            RI_sp = app.RI_i_editfield.Value;             % Refractive index of the particle
            
            WXs = 1e-6 * Dxs / sqrt(2 * log(2)); 
            WYs = 1e-6 * Dys / sqrt(2 * log(2)); 
            WZs = 1e-6 * Dzs / sqrt(2 * log(2)); 

            Omega_Xs = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(WXs .* WYs) ./ WXs / (2 * pi * 1000);
            Omega_Ys = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(WXs .* WYs) ./ WYs / (2 * pi * 1000);
            Omega_Zs = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(WXs .* WYs) ./ WZs / (2 * pi * 1000);

            Ratio_XY = Omega_Xs ./ Omega_Ys;
            Ratio_XZ = Omega_Xs ./ Omega_Zs;
            Ratio_YZ = Omega_Ys ./ Omega_Zs;

            Ratio_3D = cat(3, Omega_Xs ./ Omega_Ys, Omega_Xs ./ Omega_Zs,  Omega_Ys ./ Omega_Zs);
            Ratio_3D = Ratio_3D ./ sqrt(Ratio_3D(:,:,1).^2 + Ratio_3D(:,:,2).^2 + Ratio_3D(:,:,3).^2);

            Wy = app.Wy_i_editfield.Value;
            Wx = app.Wx_i_editfield.Value;
            Wz = app.Wz_i_editfield.Value;

            if Wx >= Wy
                rxy = reshape([Wx/Wy, Wx/Wz, Wy/Wz], [1,1,3]);
            else
                 rxy = reshape([Wy/Wx, Wy/Wz, Wx/Wz], [1,1,3]);
            end

            Inner_product_ratio = sum(Ratio_3D .* rxy,3);
            [i,j] = find(Inner_product_ratio == max(Inner_product_ratio(:)));

            NA =         app.NAs_theory(i); % Description
            circpol = app.circpol_ratios(j)*100;

            app.circpol_i_editfield.Value = circpol;
            app.NA_i_editfield.Value = NA;

            % Power & FWHMs estimation
            w0_NA = 0.7835; % Beam waist parameter for Gaussian beam
            Nx = app.xyvoxel_f_editfield.Value;
            Nz = app.zvoxel_f_editfield.Value;
            pitch = wavelength ./ (1 + 15 * NA);             % Pitch values corresponding to NAs
            FOV = [Nx, Nx, Nz] * pitch;
            

            if strcmp(app.GPUSwitch.Value, 'On')
                use_GPU = true;
            else
                use_GPU = false;
            end

            % Gaussian function for fitting intensity profiles
            gauss_fun = @(a, b, c, x) a * exp(-(x - b).^2 ./ (2 * c^2)); 
            % Initialize parameters for the current NA
            pol = [1i*circpol/100 1];
            pol = pol / norm(pol);
        
            % Initialize the solver for the computation
            solver = app.init_solver('use_GPU', use_GPU, 'NA', 1, 'wavelength0', wavelength, ...
                                     'RI_bg', RI_bg, 'pitch', pitch, 'FOV', FOV, 'dimension', 3);
    
            % Compute the analytic field using Novotny's formulation
            E_analytic = app.gaussian_analytic_v2(solver, ...
                         'polarization', pol, 'NA', NA, 'w0_NA', w0_NA);
            I_analytic = sum(abs(E_analytic).^2, 4); % Compute intensity from field
            I_analytic = I_analytic ./ max(I_analytic(:)); % Normalize intensity
    
            % Analyze the intensity profile
            [fwhm_x, fwhm_y, fwhm_z] = app.process_lines(I_analytic, gauss_fun, pitch, NA);
            app.Dx_i_editfield.Value = fwhm_x * 1000;
            app.Dy_i_editfield.Value = fwhm_y * 1000;
            app.Dz_i_editfield.Value = fwhm_z * 1000;

            wx = 1e-6 * fwhm_x / sqrt(2 * log(2)); % Convert to meters
            wy = 1e-6 * fwhm_y / sqrt(2 * log(2)); 
            wz = 1e-6 * fwhm_z / sqrt(2 * log(2)); 
            Omega_Z = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(wx * wy) / wz / (2 * pi * 1000);

            Power = Wz^2 /Omega_Z^2 * 1000;
            app.power_i_editfield.Value = Power;

        end

        function run_psf(app)
            RI_bg = 1; % Refractive index of the background
            wavelength = app.wavelength_f_editfield.Value;
            NA = app.NA_f_editfield.Value;          % Numerical apertures to evaluate
            pitch = wavelength ./ (1 + 15 * NA);             % Pitch values corresponding to NAs
            circpol_ratio = app.circpol_f_editfield.Value;
            w0_NA = 0.7835; % Beam waist parameter for Gaussian beam
            Nx = app.xyvoxel_f_editfield.Value;
            Nz = app.zvoxel_f_editfield.Value;
            FOV = [Nx, Nx, Nz] * pitch;

            if strcmp(app.GPUSwitch.Value, 'On')
                use_GPU = true;
            else
                use_GPU = false;
            end

            % Gaussian function for fitting intensity profiles
            gauss_fun = @(a, b, c, x) a * exp(-(x - b).^2 ./ (2 * c^2)); 
            % Initialize parameters for the current NA
            pol = [1i*circpol_ratio/100 1];
            pol = pol / norm(pol);
        
            % Initialize the solver for the computation
            solver = app.init_solver('use_GPU', use_GPU, 'NA', 1, 'wavelength0', wavelength, ...
                                     'RI_bg', RI_bg, 'pitch', pitch, 'FOV', FOV, 'dimension', 3);
    
            % Compute the analytic field using Novotny's formulation
            E_analytic = app.gaussian_analytic_v2(solver, ...
                         'polarization', pol, 'NA', NA, 'w0_NA', w0_NA);
            I_analytic = sum(abs(E_analytic).^2, 4); % Compute intensity from field
            I_analytic = I_analytic ./ max(I_analytic(:)); % Normalize intensity
    
            % Analyze the intensity profile
            [fwhm_x, fwhm_y, fwhm_z] = app.process_lines(I_analytic, gauss_fun, pitch, NA);
            app.Dx_f_editfield.Value = fwhm_x * 1000;
            app.Dy_f_editfield.Value = fwhm_y * 1000;
            app.Dz_f_editfield.Value = fwhm_z * 1000;


            % epsilon0 = 8.8541878128e-12; % [F/m] Vacuum permittivity
            c = 299792458;              % [m/s] Speed of light
            rho =  app.rho_f_editfield.Value;                 % Particle density [kg/m^3]
            RI_sp = app.RI_f_editfield.Value;             % Refractive index of the particle
            % radius = 142 / 2 * 1e-9;    % Particle radius [m]
            Power = app.power_f_editfield.Value;
           
            % alpha = 3 * (4 * pi / 3 * radius^3) * epsilon0 * (RI_sp^2 - 1) / (RI_sp^2 + 2); 
            % mass = (4 * pi / 3 * radius^3) * rho; % Particle mass [kg]

            wx = 1e-6 * fwhm_x / sqrt(2 * log(2)); % Convert to meters
            wy = 1e-6 * fwhm_y / sqrt(2 * log(2)); 
            wz = 1e-6 * fwhm_z / sqrt(2 * log(2)); 
            Omega_X = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(wx * wy) / wx / (2 * pi * 1000)  * sqrt(Power/1000);
            Omega_Y = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(wx * wy) / wy / (2 * pi * 1000)  * sqrt(Power/1000);
            Omega_Z = sqrt(12 * (RI_sp^2 - 1) ./ (pi * c * rho * (RI_sp^2 + 2))) ./ sqrt(wx * wy) / wz / (2 * pi * 1000)  * sqrt(Power/1000);

            app.Wx_f_editfield.Value = Omega_X;
            app.Wy_f_editfield.Value = Omega_Y;
            app.Wz_f_editfield.Value = Omega_Z;


        end
        
        function solver = init_solver(app,varargin)
            % cbs construct a new solver object
            %
            %
            % Basic arguments
            %    use_GPU            GPU boolean         default: false
            %    NA                 NA of the system    default: 1
            %    wavelength0        wavelength          default: 1
            %    RI_bg              background RI       default: 1
            %    pitch              pitch               default: 1/4
            %    FOV                FOV                 default: 11
            %    dimension          dimension           default: 3
            %    dataformat         dataformat          default: 'single'

            p = inputParser;
            p.addParameter('use_GPU', false);
            p.addParameter('NA', 1);
            p.addParameter('wavelength0', 1);
            p.addParameter('RI_bg', 1);
            p.addParameter('pitch', 1);
            p.addParameter('FOV', 1);
            p.addParameter('dimension', 3);
            p.addParameter('dataformat', 'single');

            p.parse(varargin{:});
            
            solver.use_GPU = p.Results.use_GPU;
            solver.NA = p.Results.NA;
            solver.wavelength0 = p.Results.wavelength0;
            solver.RI_bg = p.Results.RI_bg;
            solver.pitch = p.Results.pitch;
            solver.FOV = p.Results.FOV;
            solver.dimension = p.Results.dimension;
            solver.dataformat = p.Results.dataformat;
            if solver.dimension >= 4 || solver.dimension <=0
                error(['dimension = ' num2str(solver.dimension) '; Dimension should be 1~3.']);
            end
            if ~(length(solver.FOV) == solver.dimension || length(solver.FOV) == 1)
                error(['dimension = ' num2str(solver.FOV) '; FOV should be scalar or has the same dimension with FOV.']);
            end
            if ~(length(solver.pitch) == solver.dimension || length(solver.pitch) == 1)
                error(['pitch = ' num2str(solver.pitch) '; pitch should be scalar or has the same dimension with pitch.']);
            end
            solver.utility = app.gen_utility(solver);
        end
    
        function E_in = gaussian_analytic_v2(app,solver, varargin)
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
        function utility = gen_utility(app,solver)
        
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
    end
    
    methods (Access = public)
        
        function [fwhm_x, fwhm_y, fwhm_z] = process_lines(app, I_analytic, gauss_fun, pitch, NA)
            figure(1),
            % Extract lines
            line_z = app.extract_line(I_analytic, 3);
            line_x = app.extract_line(I_analytic, 2);
            line_y = app.extract_line(I_analytic, 1);
            
            % Process lines
            line_x = app.process_line(line_x, 1/3);
            line_y = app.process_line(line_y, 1/3)';
            line_z = app.process_line(line_z, 1/3);
        
            % Fit and calculate FWHM for X and Z
            fwhm_x = app.fit_and_plot(line_x, gauss_fun, pitch, 231); title('X lineplot');
            fwhm_y = app.fit_and_plot(line_y, gauss_fun, pitch, 232); title('Y lineplot');
            fwhm_z = app.fit_and_plot(line_z, gauss_fun, pitch, 233); title('Z lineplot');
            fprintf('NA: %.2f, FWHM_x: %.4f, FWHM_z: %.4f\n', NA, fwhm_x, fwhm_z);
        
            % Define the middle indices for slicing
            mid_x = floor(size(I_analytic, 2) / 2) + 1; % Middle index along X
            mid_y = floor(size(I_analytic, 1) / 2) + 1; % Middle index along Y
            mid_z = floor(size(I_analytic, 3) / 2) + 1; % Middle index along Z
            
            % Extract XY and XZ cross-sections
            xy_slice = squeeze(I_analytic(:, :, mid_z)); % XY plane at mid-Z
            yz_slice = squeeze(I_analytic(:, mid_x, :)); % XY plane at mid-Z
            xz_slice = squeeze(I_analytic(mid_y, :, :)); % XZ plane at mid-Y
        
        
            % Display XY cross-section
            subplot(234);
            imagesc(xy_slice);
            axis image; % Ensure correct aspect ratio
            colorbar;  % Add color bar for intensity scale
            title('XY Cross-Section');
            xlabel('X');
            ylabel('Y');
            
            subplot(235);
            imagesc(yz_slice');
            axis image; % Ensure correct aspect ratio
            colorbar;  % Add color bar for intensity scale
            title('YZ Cross-Section');
            xlabel('Y');
            ylabel('Z');
            
        
            % Display XZ cross-section
            subplot(236);
            imagesc(xz_slice');
            axis image; % Ensure correct aspect ratio
            colorbar;  % Add color bar for intensity scale
            title('XZ Cross-Section');
            xlabel('X');
            ylabel('Z');
            % sgtitle(fprintf('NA: %.2f, FWHM_x: %.4f, FWHM_z: %.4f\n', NA, fwhm_x, fwhm_z))
            colormap hot
            set(gcf, 'color', 'w')
            drawnow
        
        end
        
        function line = extract_line(app, I_analytic, dim)
            % Extract a line from I_analytic along the specified dimension
            mid = floor(size(I_analytic) / 2) + 1;
            if dim == 2
                line = squeeze(I_analytic(:, mid(2), mid(3)));
            elseif dim == 1
                line = squeeze(I_analytic(mid(1), :, mid(3)));
            elseif dim == 3
                line = squeeze(I_analytic(mid(1), mid(2), :));
            else
                error('Unsupported dimension for line extraction.');
            end
        end
        
        function processed_line = process_line(app, line, target_value)
            % Find indices around the target value and truncate the line
            idx = app.find_closest_indices(line, target_value);
            processed_line = line(idx(1):idx(end));
        end
        
        function idx = find_closest_indices(app, line, target_value)
            % Find indices closest to the target value
            [~, idx(1)] = min(abs(line - target_value));
            idx(2) = round(length(line) / 2 + abs(length(line) / 2 - idx(1)));
            idx(3) = round(length(line) / 2 - abs(length(line) / 2 - idx(1)));
            idx = sort(idx);
            idx = min(max(idx, 1), length(line));
        end
        
        function fwhm = fit_and_plot(app, line, gauss_fun, pitch, subplot_position)
            % Fit a Gaussian to the line and calculate FWHM
            x0 = (1:length(line)) - mean(1:length(line));
            fit_result = fit(x0', line, gauss_fun, ...
                'StartPoint', [max(line), mean(x0), std(x0)]);
            fwhm = 2 * sqrt(2 * log(2)) * fit_result.c * pitch;
        
            % Plot the result
            subplot(subplot_position);
            plot(fit_result, x0, line);
        end

        function load_FWHMs(app)
            app.Dxs_1064 = [5.43782457116493	5.43785777025804	5.43795712180386	5.43811995348428	5.43834111478167	5.43861474274108	5.43893310326302	5.43929069772942	5.43967848581426	5.44008727180191	5.44051260415898	5.44094666566733	5.44138312600438	5.44181735651430	5.44224577567693	5.44266514133592	5.44307154100003	5.44346542420891	5.44384225699158	5.44420483253301	5.44454929717686
4.95955654132047	4.95959281789718	4.95970115583391	4.95987831314201	4.96011868428968	4.96041564780017	4.96076328853818	4.96115185688059	4.96157331486959	4.96201821827894	4.96248061177854	4.96295218566329	4.96342781016408	4.96390063195249	4.96436668363812	4.96482251274857	4.96526574346784	4.96569253730040	4.96610431969247	4.96649753205715	4.96687339359835
4.48809893448046	4.48814066776230	4.48826417195454	4.48846606686052	4.48874130477495	4.48908038536102	4.48947682074526	4.48992047987798	4.49040107245979	4.49090996542219	4.49143762936603	4.49197637156815	4.49251854952576	4.49305830500330	4.49359062790402	4.49411111546784	4.49461664308364	4.49510540246778	4.49557434497807	4.49602343885214	4.49645238623289
4.15322131432897	4.15326584744959	4.15339842257231	4.15361587840519	4.15391120281935	4.15427677848090	4.15470344178289	4.15518036812131	4.15569777607412	4.15624428144204	4.15681250170029	4.15739164474658	4.15797508799007	4.15855597623605	4.15912850623828	4.15968870242394	4.16023271193802	4.16075785572397	4.16126311441658	4.16174636302138	4.16220747083519
3.86444345103424	3.86449113137036	3.86463323301373	3.86486599542697	3.86518234622556	3.86557339407490	3.86602987474208	3.86654024903144	3.86709419917693	3.86768039490181	3.86828793654259	3.86890836401844	3.86953304383619	3.87015493789913	3.87076816808975	3.87136800842660	3.87195065783832	3.87251311585760	3.87305404785322	3.87357162488443	3.87406591272835
3.61296624765344	3.61301734930452	3.61316938364202	3.61341718978477	3.61375416053255	3.61417121647060	3.61465769683840	3.61520191216112	3.61579253365888	3.61641699841576	3.61706513837693	3.61772674499224	3.61839299769704	3.61905633508854	3.61971022537523	3.62034968154158	3.62097110694330	3.62157146058078	3.62214827718024	3.62270070529146	3.62322735044065
3.39201216227356	3.39206628686715	3.39222743362333	3.39249075101301	3.39284909481860	3.39329186538855	3.39380915315401	3.39438735741052	3.39501472415389	3.39567848541830	3.39636731038904	3.39707039205255	3.39777801844735	3.39848287973216	3.39917794529742	3.39985755033708	3.40051797861823	3.40115612892457	3.40176901967546	3.40235600345489	3.40291629308698
3.19641072570806	3.19646823797342	3.19663895562179	3.19691745435787	3.19729669781129	3.19776594003958	3.19831338338289	3.19892604233388	3.19959041630257	3.20029326897106	3.20102246883041	3.20176738141459	3.20251727595723	3.20326396249174	3.20400004840769	3.20472047561296	3.20542013010998	3.20609599414153	3.20674552655286	3.20736784761202	3.20796140251388
3.02206319818316	3.02212366993746	3.02230425291119	3.02259816170303	3.02299844512229	3.02349433181434	3.02407229442120	3.02471900790238	3.02542072331993	3.02616329903591	3.02693351559993	3.02772029502825	3.02851224989526	3.02930058593671	3.03007864380478	3.03083936794231	3.03157856484535	3.03229261092482	3.03297929865298	3.03363684883987	3.03426387966165
2.86570884523288	2.86577285531846	2.86596273743892	2.86627257430947	2.86669451563686	2.86721601403477	2.86782477431431	2.86850641861385	2.86924541605142	2.87002743176023	2.87083920584712	2.87166761144306	2.87250203792468	2.87333313046846	2.87415244454220	2.87495430497238	2.87573353364140	2.87648596384072	2.87720944053384	2.87790233951318	2.87856347926903
2.72471366455977	2.72478067216789	2.72498052244282	2.72530593645802	2.72574903101087	2.72629741493638	2.72693724054110	2.72765306751010	2.72842955853639	2.72925141773943	2.73010464712683	2.73097537681834	2.73185266861405	2.73272585767952	2.73358752863603	2.73443055958644	2.73524916279985	2.73604097987750	2.73680154402705	2.73753012052280	2.73822519106058
2.59693214745850	2.59700232245383	2.59721116597313	2.59755306780682	2.59801719389076	2.59859205027438	2.59926282568064	2.60001372607327	2.60082783661433	2.60168956264971	2.60258443964987	2.60349813235698	2.60441776043116	2.60533417569260	2.60623780486982	2.60712226393993	2.60798131992930	2.60881133152920	2.60960985951713	2.61037402397037	2.61110362061342
2.48059944988380	2.48067306325530	2.48089184472301	2.48124921303159	2.48173495590277	2.48233675476645	2.48303857462216	2.48382415273914	2.48467656311682	2.48557842443677	2.48651488443406	2.48747060131461	2.48843404673344	2.48939296254364	2.49033894657034	2.49126522420481	2.49217339955677	2.49304234464129	2.49387829248790	2.49467897311575	2.49544314854271
2.37425719315629	2.37433386049403	2.37456222905356	2.37493528507873	2.37544317457438	2.37607167730347	2.37680477385275	2.37762552521187	2.37851589443378	2.37945848233634	2.38043685940669	2.38143594358343	2.38244241550450	2.38344473213125	2.38443357669155	2.38540125354663	2.38634178138081	2.38725062071918	2.38812458469963	2.38896175735464	2.38976053726755
2.27666286617394	2.27674296799177	2.27698106532624	2.27737051490731	2.27789956934915	2.27855472476493	2.27931966549242	2.28017540792931	2.28110385613317	2.28208697015778	2.28310753979530	2.28414967467853	2.28519981721542	2.28624561207408	2.28727727240431	2.28828721608040	2.28926893122084	2.29021697477926	2.29112930699096	2.29200272739772	2.29283649549144
2.15985059398480	2.15993770308155	2.16019720020373	2.16062083094399	2.16119695151912	2.16191036494771	2.16274254986423	2.16367402428000	2.16468478262589	2.16575462399718	2.16686504200657	2.16799911194263	2.16914148269278	2.17027936022450	2.17140164600718	2.17250009640020	2.17356744507460	2.17459910669007	2.17559086556033	2.17654097019895	2.17744769867730
2.07806602412203	2.07815685171125	2.07842598150600	2.07886648664815	2.07946574268732	2.08020703942311	2.08107226192161	2.08204075369730	2.08309154962840	2.08420407574977	2.08535869990403	2.08653808066357	2.08772588917450	2.08890945675436	2.09007692314283	2.09121948255428	2.09232993944791	2.09340280429918	2.09443485173354	2.09542334904490	2.09636691041971
2.00226926414448	2.00236338712306	2.00264309180568	2.00310038223430	2.00372219964481	2.00449210710874	2.00539026941878	2.00639586314944	2.00748707296498	2.00864219642503	2.00984142450933	2.01106619196078	2.01230018597964	2.01352933462155	2.01474211500050	2.01592911129074	2.01708272184607	2.01819782445394	2.01927003847966	2.02029727902068	2.02127748965019
1.93182816678488	1.93192588348484	1.93221577875895	1.93269004327313	1.93333477605861	1.93413294645951	1.93506447819091	1.93610745489581	1.93723883732205	1.93843720213317	1.93968090696528	1.94095140439722	1.94223139166980	1.94350681996169	1.94476516284602	1.94599673434544	1.94719392706223	1.94835107128409	1.94946400049836	1.95053002048745	1.95154765036443
1.86619601147876	1.86629698038003	1.86659754093655	1.86708853840144	1.86775634134823	1.86858324747053	1.86954802780166	1.87062830644865	1.87180042788590	1.87304179886029	1.87433076731986	1.87564713755691	1.87697357061130	1.87829506650079	1.87959928450333	1.88087581346455	1.88211680298332	1.88331631621933	1.88447012712542	1.88557545718223	1.88663056636709
1.80489785470638	1.80500248302646	1.80531326328812	1.80582119300511	1.80651218384074	1.80736760934626	1.80836608364831	1.80948388767687	1.81069706581945	1.81198172538194	1.81331570085391	1.81467820548256	1.81605140607611	1.81741969731014	1.81876982883083	1.82009171220541	1.82137680172803	1.82261904980560	1.82381380793833	1.82495863893103	1.82605173423503
1.74751934955509	1.74762744647969	1.74794866609556	1.74847359867792	1.74918795863883	1.75007215483261	1.75110420800915	1.75225981025472	1.75351399454476	1.75484230033059	1.75622156618674	1.75763071723203	1.75905099708544	1.76046595913927	1.76186268817840	1.76323005089309	1.76455945971742	1.76584467119902	1.76708085681568	1.76826538757882	1.76939652830766
1.69369736332622	1.69380893087729	1.69414065182267	1.69468285884564	1.69542034472491	1.69633335835720	1.69739940382446	1.69859301918233	1.69988835033776	1.70126053675092	1.70268548684582	1.70414133307255	1.70560873837530	1.70707100325089	1.70851424494245	1.70992763235551	1.71130161366810	1.71263009799774	1.71390824800824	1.71513292415175	1.71630239653235
1.64311193585600	1.64322703141295	1.64356922124223	1.64412865124267	1.64488951047737	1.64583181237512	1.64693159289873	1.64816345798745	1.64950044171961	1.65091659485807	1.65238738246718	1.65389010896422	1.65540517049544	1.65691494385656	1.65840540343791	1.65986473049987	1.66128380096275	1.66265597916210	1.66397616134882	1.66524123414886	1.66644927144780
1.59548006475865	1.59559884261478	1.59595161784483	1.59652821405746	1.59731261930856	1.59828413660194	1.59941819089032	1.60068813994079	1.60206690536645	1.60352735954842	1.60504420553856	1.60659442843697	1.60815723252889	1.60971495469582	1.61125275052723	1.61275841597669	1.61422303254487	1.61563901634815	1.61700183029243	1.61830760406208	1.61955463560585
1.55055171575150	1.55067385866170	1.55103728953396	1.55163135242536	1.55243948061353	1.55344038011455	1.55460854376996	1.55591719572986	1.55733786938279	1.55884294669987	1.56040644067930	1.56200414232411	1.56361517842600	1.56522084007403	1.56680643558383	1.56835904129596	1.56986920728002	1.57132957933228	1.57273504114371	1.57408190088426	1.57536851502097
1.50810278642764	1.50822887709691	1.50860277982571	1.50921396645395	1.51004585346427	1.51107632745408	1.51227906505615	1.51362661864963	1.51508932691894	1.51663933153413	1.51824959440950	1.51989518078745	1.52155467844017	1.52320894991841	1.52484245091966	1.52644232665274	1.52799861344852	1.52950371162497	1.53095215430446	1.53234066944626	1.53366691528813
1.46793506771390	1.46806452462603	1.46844921109685	1.46907823881842	1.46993393126764	1.47099395754756	1.47223158287306	1.47361798183577	1.47512323737168	1.47671838860670	1.47837555947182	1.48006941772375	1.48177768581621	1.48348100032211	1.48516275841658	1.48681034857362	1.48841294944859	1.48996324125995	1.49145528262102	1.49288548359214	1.49425187063703
1.42986856310223	1.43000172663168	1.43039703339990	1.43104365427631	1.43192358150289	1.43301332684058	1.43428566301252	1.43571128487441	1.43725949524515	1.43889975662601	1.44060441634886	1.44234709907197	1.44410438120433	1.44585693581841	1.44758750160781	1.44928305348373	1.45093263484846	1.45252822654571	1.45406434628668	1.45553684427672	1.45694365641055
1.39374413461212	1.39388084616465	1.39428730288477	1.39495143627125	1.39585544029373	1.39697528582025	1.39828280802268	1.39974779911590	1.40133874116372	1.40302514231729	1.40477729651722	1.40656873950535	1.40837575899321	1.41017782013984	1.41195779329908	1.41370160053660	1.41539856311445	1.41704010716922	1.41862028288291	1.42013542648070	1.42158315955056
1.35941601518728	1.35955636470759	1.35997367656320	1.36065565313951	1.36158391905583	1.36273388393973	1.36407670041759	1.36558123317134	1.36721574281606	1.36894795397603	1.37074830313714	1.37258902414793	1.37444597878469	1.37629809341433	1.37812767202474	1.37992034777129	1.38166480985041	1.38335278986399	1.38497785396324	1.38653598685367	1.38802501092083
1.32675432247847	1.32689850879718	1.32732628632601	1.32802645270153	1.32897925555547	1.33015938901430	1.33153781319342	1.33308249446018	1.33476026083525	1.33653900742228	1.33838759151481	1.34027824861773	1.34218545966044	1.34408812938659	1.34596750902638	1.34780955567261	1.34960225525791	1.35133705031324	1.35300717577149	1.35460885079653	1.35613963181401
1.29564070393821	1.29578854469710	1.29622757696064	1.29694560271910	1.29792293727940	1.29913359786477	1.30054778154899	1.30213271333373	1.30385438677234	1.30567964643494	1.30757701481966	1.30951786602175	1.31147583709624	1.31342936438420	1.31535941173224	1.31725115825893	1.31909233975127	1.32087439672728	1.32259028997772	1.32423598563801	1.32580875122156
1.26596726637524	1.26611873846359	1.26656892133896	1.26730505459281	1.26830698084066	1.26954857359477	1.27099866999128	1.27262398896966	1.27438980585641	1.27626226836590	1.27820881945139	1.28019995365905	1.28220931189303	1.28421397096721	1.28619508450698	1.28813706680695	1.29002741454340	1.29185700780097	1.29361902095654	1.29530918672501	1.29692472841093
1.23763673702879	1.23779202277943	1.23825330281749	1.23900775319809	1.24003452999772	1.24130693668934	1.24279333730776	1.24445944357134	1.24626977213312	1.24818961196566	1.25018573475620	1.25222772331808	1.25428875660074	1.25634533490162	1.25837789879718	1.26037047762417	1.26231041682372	1.26418824490172	1.26599708304434	1.26773200515652	1.26939073592741
1.21055986151738	1.21071888700563	1.21119132496620	1.21196402717194	1.21301607760844	1.21431937420023	1.21584241195543	1.21754951828296	1.21940473218909	1.22137236256047	1.22341833322860	1.22551186576950	1.22762509539455	1.22973373487856	1.23181837750673	1.23386217734207	1.23585220188399	1.23777870686322	1.23963462833887	1.24141517472024	1.24311748555542
1.18465476285408	1.18481754069642	1.18530146557834	1.18609258363105	1.18716976882408	1.18850462131983	1.19006417732697	1.19181282958607	1.19371314742982	1.19572887416570	1.19782542364551	1.19997066755697	1.20213641502232	1.20429804350684	1.20643497157467	1.20853049321437	1.21057119289248	1.21254708652435	1.21445073911921	1.21627723470642	1.21802367751549
1.15984758190484	1.16001428775758	1.16050931791635	1.16131922895745	1.16242160689392	1.16378804059827	1.16538477863944	1.16717509940475	1.16912097326288	1.17118537039285	1.17333255936190	1.17553012236583	1.17774899133591	1.17996394676059	1.18215398429343	1.18430171043839	1.18639373698491	1.18841940419945	1.19037140508616	1.19224436145616	1.19403562779174
1.13606901543878	1.13623950610489	1.13674600581165	1.13757461406993	1.13870274019709	1.14010087776575	1.14173489272474	1.14356715568419	1.14555907956657	1.14767245136123	1.14987087833975	1.15212127196167	1.15439382467483	1.15666268683388	1.15890616351932	1.16110690419241	1.16325064014111	1.16532669879575	1.16732742057573	1.16924778397507	1.17108421336261
1.11325699673694	1.11343121948949	1.11394940624687	1.11479675104363	1.11595058723420	1.11738084481746	1.11905248964248	1.12092724138921	1.12296534450588	1.12512808141495	1.12737843343341	1.12968200346130	1.13200875156687	1.13433196035041	1.13662987634003	1.13888386483843	1.14108005277536	1.14320720649462	1.14525744822424	1.14722535794730	1.14910779986718
1.09135274132910	1.09153100146125	1.09206059260141	1.09292704501098	1.09410678856700	1.09556930966111	1.09727890201985	1.09919621370362	1.10128119880832	1.10349392415472	1.10579648946211	1.10815396821326	1.11053537819133	1.11291369053189	1.11526614512606	1.11757444040041	1.11982357827114	1.12200242516726	1.12410288629327	1.12611916808906	1.12804813799125
1.07030332312796	1.07048541844712	1.07102676656726	1.07191225039548	1.07311826423025	1.07461344069582	1.07636107329355	1.07832157543511	1.08045366559998	1.08271676221165	1.08507212268840	1.08748405137612	1.08992068845059	1.09235459920178	1.09476262022286	1.09712564539367	1.09942852077601	1.10165964621941	1.10381080874344	1.10587619550786	1.10785224733008
1.05005897146754	1.05024506042301	1.05079829269735	1.05170316836182	1.05293537412344	1.05446330370761	1.05624960809349	1.05825359440448	1.06043336899847	1.06274732059344	1.06515602347355	1.06762268840930	1.07011550932303	1.07260568850586	1.07506979906834	1.07748814354325	1.07984533283003	1.08212956816822	1.08433225152826	1.08644725490106	1.08847112211998
1.03057458435213	1.03076461361649	1.03132953655497	1.03225394354077	1.03351277445922	1.03507386739834	1.03689902983334	1.03894698227196	1.04117479654174	1.04354002522623	1.04600246543093	1.04852494845768	1.05107417381224	1.05362140262248	1.05614219567390	1.05861668516790	1.06102894175434	1.06336703723556	1.06562174110960	1.06778727369401	1.06985969276121
1.01180730391153	1.01200153133010	1.01257842586642	1.01352238597511	1.01480802837786	1.01640238186439	1.01826686964881	1.02035917455084	1.02263556803907	1.02505270326549	1.02756958070660	1.03014816124281	1.03275463262159	1.03535958565834	1.03793776762071	1.04046915241044	1.04293725639059	1.04532972723755	1.04763736492201	1.04985404307702	1.05197575512931
0.993717563793877	0.993915788579357	0.994504740493577	0.995468526047739	0.996781264793459	0.998409227052107	1.00031343488514	1.00245040524514	1.00477581656343	1.00724530636693	1.00981727891155	1.01245286062796	1.01511713440795	1.01778034084431	1.02041687197852	1.02300580959646	1.02553065014878	1.02797836436321	1.03033970037701	1.03260834156191	1.03477999836665
0.976269436715041	0.976471739570996	0.977072972557224	0.978056490491509	0.979396698150141	0.981058761240836	0.983002806209049	0.985184970581151	0.987559811146466	0.990082595550308	0.992710110863150	0.995403134826179	0.998126076155767	1.00084842143526	1.00354394988888	1.00619129366307	1.00877337792251	1.01127731052257	1.01369316826486	1.01601456519406	1.01823701222055
0.959428499323691	0.959634816069550	0.960248344042260	0.961252230151681	0.962619818994563	0.964316219190880	0.966300813306532	0.968528604377538	0.970953545803053	0.973529749369343	0.976213742434609	0.978964981848198	0.981747202268124	0.984529452946632	0.987284797721087	0.989991414952251	0.992631836441812	0.995192547209386	0.997663740020254	1.00003869526299	1.00231303037383
0.943162962417100	0.943373517267262	0.943999444735433	0.945023561015877	0.946418938743961	0.948150069888972	0.950175406531053	0.952449396090116	0.954925048198069	0.957555430557022	0.960296460779469	0.963106585504014	0.965949079935346	0.968791877157748	0.971607869004999	0.974374668962334	0.977074195986877	0.979692713985306	0.982220247953506	0.984649478732235	0.986976314773271
0.927443089673336	0.927657863683204	0.928296249613138	0.929340967414123	0.930764371751994	0.932530562706426	0.934597132876281	0.936917975477098	0.939444613876521	0.942130015451739	0.944928651213374	0.947798524174998	0.950702067062231	0.953606434244452	0.956483942608155	0.959311496836863	0.962071027790648	0.964748557386944	0.967332990728566	0.969817749890084	0.972197962545623
0.912241683989708	0.912460671814132	0.913111710337699	0.914177058702271	0.915628958588463	0.917430512709069	0.919538924124532	0.921906793580755	0.924485359551687	0.927226389170998	0.930083350734420	0.933013707276717	0.935978865703789	0.938945736946189	0.941885672072819	0.944775235655279	0.947595830398639	0.950332844468263	0.952975587392979	0.955516568165165	0.957951241453296
0.897532159424256	0.897755449925513	0.898419208591946	0.899505678756886	0.900986198001447	0.902823665250708	0.904974072951255	0.907389730343141	0.910020773034435	0.912818034202921	0.915734210085892	0.918725960130028	0.921753914952970	0.924783988819631	0.927787279868485	0.930739772523012	0.933622366211410	0.936420099456191	0.939121969344863	0.941720354363714	0.944210218071571
0.883290535396723	0.883518226382573	0.884194825509909	0.885302432970978	0.886811928148587	0.888685567224263	0.890878728425779	0.893342767529792	0.896026936325200	0.898881073477552	0.901857383510709	0.904911181638478	0.908002760271654	0.911097311989880	0.914164961789910	0.917181416388958	0.920126916102186	0.922986320688556	0.925748447119814	0.928405156987540	0.930951494934165
0.869493842360019	0.869725731760045	0.870415559332815	0.871544506668593	0.873083509234614	0.874993683691749	0.877230055388644	0.879742974959407	0.882480901004188	0.885392941250658	0.888429996287787	0.891546946803041	0.894703020921693	0.897862859107387	0.900995915992925	0.904077320692044	0.907087057995833	0.910009375203105	0.912832701026703	0.915548878283165	0.918152637087719
0.856120584919266	0.856356999856132	0.857059953806062	0.858210556038543	0.859779152373886	0.861726455715629	0.864006536228276	0.866569008539850	0.869361470695487	0.872332035466310	0.875430840746901	0.878611756585933	0.881833426102003	0.885059580246868	0.888259170437921	0.891406707944078	0.894481670376059	0.897468012199880	0.900353847099390	0.903130568333198	0.905793117205407
0.843150439957753	0.843391401967004	0.844107750416439	0.845280286823583	0.846878902930394	0.848863669961934	0.851188072358365	0.853800729390527	0.856648510162634	0.859678453497128	0.862839825033011	0.866085872028260	0.869374158867011	0.872667730231693	0.875935113551501	0.879149943198160	0.882291212343809	0.885342795865407	0.888292277394651	0.891130818936378	0.893853153664756
0.830564835038008	0.830810263132459	0.831540073882334	0.832734862889831	0.834363925914610	0.836386726764519	0.838756090722605	0.841419553060544	0.844323436695076	0.847413555192615	0.850638617557698	0.853950634204244	0.857306683320445	0.860668830216461	0.864004920341719	0.867288289297114	0.870497382467889	0.873615416516264	0.876629740151462	0.879531440890784	0.882314878214623
0.818345425838881	0.818595447573085	0.819339038054222	0.820556181966517	0.822216015292426	0.824277466254312	0.826692199950271	0.829407366916930	0.832367996510856	0.835519390636693	0.838809039721307	0.842188299290586	0.845613229818467	0.849045118473536	0.852451332225786	0.855804572458615	0.859082689324372	0.862268528340270	0.865349091012797	0.868315268492498	0.871161111628256
0.806475064563200	0.806729810239137	0.807487104631858	0.808727121036917	0.810418194855212	0.812518680363836	0.814979490796035	0.817747049443386	0.820765478250175	0.823979012109893	0.827334328112604	0.830781814606517	0.834276740191790	0.837779827468950	0.841257510204672	0.844681922185292	0.848030375043664	0.851285394842146	0.854433627051124	0.857465579606291	0.860375204949291
0.794938010488305	0.795197436557503	0.795968804510951	0.797231974868587	0.798954626595742	0.801094661947446	0.803602404489307	0.806423156574484	0.809500189433261	0.812776861862784	0.816198994081732	0.819715868135368	0.823282204881885	0.826857711211743	0.830408176341808	0.833905171579186	0.837325579193929	0.840651214504693	0.843868586569100	0.846967909740810	0.849942670506444
0.783718536183848	0.783982772363964	0.784768389864605	0.786054867420208	0.787809710613146	0.789989924516373	0.792545176294755	0.795420008130846	0.798556661801290	0.801897606893483	0.805387587516731	0.808975265537360	0.812614190969409	0.816263598800020	0.819888326796698	0.823459352767305	0.826953176206293	0.830350984441887	0.833639063634611	0.836807143229071	0.839848861309267
0.772802654784636	0.773071657251649	0.773871726730403	0.775181946940643	0.776969459612094	0.779190505549695	0.781794078978119	0.784723757893985	0.787920933486330	0.791327297421613	0.794886486645840	0.798546130077423	0.802259221749541	0.805983892493498	0.809684405267663	0.813331199107121	0.816899892044341	0.820371578146938	0.823732057161858	0.826970624113673	0.830080753955263
0.762176218666989	0.762450199530240	0.763264914350162	0.764599305354375	0.766419838433073	0.768682360132956	0.771335002423561	0.774320500572615	0.777579361920119	0.781052212900133	0.784681933826397	0.788415108593258	0.792203672817155	0.796005259910113	0.799783193405611	0.803507317821542	0.807152723429131	0.810700060718814	0.814134506619383	0.817445400893655	0.814239557017517
0.751826044804603	0.752105005072581	0.752934667484194	0.754293613209667	0.756147707118188	0.758452439118542	0.761155005001939	0.764197249678904	0.767518958390634	0.771059565305621	0.774761100777469	0.778569173267445	0.782434926174626	0.786314921327404	0.790172124835677	0.793975339186045	0.797699199877665	0.801323981537939	0.804834231402589	0.808219193627150	0.805228765273453
0.741739693509606	0.742023789847508	0.742868532055224	0.744252287451600	0.746140736208668	0.748488322668156	0.751241623434703	0.754341823373648	0.757727456359025	0.761337166197031	0.765111893187782	0.768996582724295	0.772941022289963	0.776901360213398	0.780839435140602	0.784723603637718	0.788527946844495	0.792231966237623	0.795820142664270	0.793114699067288	0.796504093888445
0.731905200265098	0.732194394815788	0.733054516631040	0.734463739453675	0.736386957474083	0.738778181973811	0.741583250373124	0.744742276321046	0.748193146271340	0.751873431487218	0.755722939318678	0.759685653409274	0.763710742383353	0.767753157012155	0.771774142366851	0.775741245996788	0.779628113475105	0.783413512480926	0.780981598294930	0.784590915758553	0.788055360959616
0.722310521021616	0.722604943586949	0.723480825172610	0.724915751743843	0.726874363098962	0.729310161795613	0.732167910139813	0.735387192330746	0.738904537860933	0.742656700875459	0.746582703102377	0.750625284474093	0.754732733804988	0.758859330169333	0.762965167503282	0.767017240711240	0.770988720651488	0.774857696579331	0.772642236641004	0.776330979601536	0.779872521526015
0.712944847803475	0.713244622656246	0.714136460156951	0.715597758083063	0.717592516819008	0.720073592873455	0.722985208464452	0.726265683634162	0.729850984581484	0.733676753493598	0.737680840941320	0.741805308495650	0.745997167036140	0.750209761448212	0.754402745273092	0.758542202065714	0.762600501548877	0.760643221775345	0.764555324179186	0.768325704896366	0.771946478981714
0.703797177605486	0.704102465693305	0.705010549918817	0.706498590065977	0.708530205587085	0.711057599959526	0.714024052048811	0.717367302427026	0.721022054467423	0.724922983768316	0.729007173564036	0.733215209291036	0.737493620465569	0.741794574673086	0.746077067949146	0.750306220055755	0.748584061127863	0.752713638746340	0.756711817628062	0.760566286620819	0.764268613729244
0.694857140166240	0.695168021130332	0.696092718377943	0.697608080647379	0.699677369596624	0.702251994091775	0.705274515861225	0.708681835428458	0.712407630773381	0.716385690113886	0.720551732745469	0.724845704328474	0.729212907049424	0.733604560590347	0.737978978027329	0.742300318870999	0.740796497562361	0.745016513815068	0.749103465352390	0.753044452836000	0.756830876493405
0.686114228602534	0.686430742371640	0.687372352201959	0.688915762605825	0.691023490109842	0.693646444784144	0.696726373850014	0.700199326868900	0.703997909856915	0.708054872573322	0.712304998388861	0.716687123412191	0.721145504928680	0.725630455425364	0.730099350345454	0.728800383863218	0.733230378600659	0.737543663818407	0.741722059141237	0.745752254187337	0.749625432259306
0.677558527808091	0.677880876914728	0.678839879816348	0.680411793815986	0.682558907627603	0.685231312771889	0.688370076129042	0.691910215644672	0.695783398260032	0.699921421732616	0.704257856566731	0.708730490922791	0.713282619709345	0.717863801358959	0.722430005461297	0.721350518308601	0.725877914037791	0.730287367160526	0.734560114332089	0.738682498419815	0.742645327549782
0.669180059246269	0.669508386026142	0.670485190352057	0.672086382615801	0.674273728925786	0.676996885681024	0.680195883820204	0.683804915992716	0.687754853235301	0.691976086783343	0.696401217026931	0.700967173416678	0.705615840661133	0.710295959656339	0.714962540736639	0.714103531227971	0.718731532928998	0.723240269255031	0.727610511860885	0.731828264954029	0.735883824352291
0.660968600322739	0.661302993378698	0.662297950217973	0.663929234863512	0.666157986072604	0.668933106155400	0.672194061583421	0.675873996863469	0.679902668535202	0.684209423104316	0.688725990166072	0.693387959992800	0.698136314211742	0.702918598003546	0.702224989738594	0.707051738454892	0.711783752512707	0.716395223306065	0.720866413819195	0.725182827347362	0.729334407683041
0.652914617433391	0.653255227832489	0.654268987297665	0.655931013397718	0.658202304524819	0.661030882137501	0.664355518984169	0.668108363210345	0.672218037548576	0.676613127487863	0.681224022699496	0.685985063989029	0.690836287344611	0.695724256066743	0.695253305615347	0.700188170385865	0.705027778979181	0.709745730662683	0.714321553125431	0.718740382335448	0.722991846970624
0.645007475748398	0.645354530127665	0.646387530127882	0.648081340776789	0.650396293000443	0.653279965141745	0.656670105857256	0.660498061352908	0.664691489638986	0.669177670953748	0.673885929050680	0.678749538467451	0.683707234441876	0.683349898346722	0.688458538289580	0.693505534241773	0.698456836782736	0.703285189480535	0.707969795590008	0.712495033730895	0.716850276613146
0.637236732835228	0.637590507279308	0.638643154986101	0.640369717681164	0.642729726609348	0.645670249625374	0.649128129923278	0.653033580626811	0.657313571133378	0.661893921075758	0.666702906786295	0.671672713575242	0.676740957899060	0.676608610901327	0.681833282117091	0.686996834801600	0.692064181558206	0.697007510265109	0.701805271737459	0.706441449993669	0.710904802526757
0.629591680037108	0.629952282609799	0.631025468248548	0.632785792113236	0.635192336428470	0.638191594034376	0.641719430028659	0.645705255139833	0.650074821870127	0.654752959900360	0.659666404548984	0.664746600916930	0.669929458334476	0.670025446848907	0.675370803872907	0.680655597622332	0.685843915093349	0.690907103616881	0.695822976387300	0.700574927212382	0.705151417767675
0.622060414488480	0.622428133134733	0.623522454431124	0.625317717238615	0.627772574622948	0.630832582037004	0.634433036488084	0.638502221636722	0.642964737845383	0.647744319059312	0.652766685972936	0.657961565372342	0.663264091887705	0.663592017798475	0.669063328921240	0.674474738146779	0.679789392744341	0.684977956869413	0.690017357718248	0.694890550629231	0.699585459687522
0.614630006451880	0.615005085489231	0.616121469979920	0.617952972678842	0.620457971423750	0.623581100274370	0.627257051613150	0.631413015862957	0.635972417967827	0.640857705144036	0.645993593459243	0.651308232888696	0.651683855718795	0.657300380569527	0.662903433736392	0.668447368330577	0.673894526192218	0.679214476974480	0.684383566577357	0.689384091441813	0.694203474848522
0.607286325538433	0.607669066198885	0.608808358448903	0.610677762457174	0.613234844624022	0.616424058118720	0.620178665420438	0.624425139123055	0.629085704445601	0.634081735953101	0.639336249932613	0.644776542816185	0.645389212542535	0.651141967313690	0.656883343595408	0.662566646704642	0.668153031463623	0.673611321069047	0.678917010906212	0.684051769184929	0.689002480825095
0.600012950470494	0.600403706854516	0.601566915351742	0.603475792783393	0.606087676011498	0.609345845779629	0.613182929594411	0.617524358486411	0.622291105718115	0.627403225376089	0.632782610444148	0.638354930034884	0.639211177929622	0.645107405646472	0.650994665693310	0.656824995849432	0.662558514480905	0.668162971687975	0.673613213753682	0.678890106540353	0.683979941082837
0.592790294293660	0.593189479154232	0.594377893723296	0.596328314926894	0.598997612703616	0.602328322409393	0.606252321615655	0.610693625733247	0.615572310211377	0.620807008782273	0.626318248402086	0.632030222395147	0.633137655096646	0.639185846018152	0.645227616364663	0.651213908543895	0.657103550685256	0.662863365433254	0.668467269938002	0.673895316092694	0.679133442215753
0.585594103657728	0.586002166758385	0.587217142835913	0.589211667730016	0.591941565957070	0.595349223719832	0.599365172129061	0.603912418675716	0.608909841251193	0.614274672016931	0.619925949695885	0.625786622577296	0.627154740706681	0.633364813713876	0.639571477599387	0.645724164457397	0.651780634176706	0.657706508295569	0.663474847581366	0.669064992417849	0.674461956480324
0.578392657444871	0.578810280681566	0.580053451331243	0.582094817500065	0.584889629578157	0.588379093716058	0.592493059899697	0.597153664443970	0.602277991666288	0.607781986992368	0.613583335051811	0.614921057123464	0.621244215434184	0.627628181635262	0.634012073308744	0.640343795584482	0.646579844395371	0.652684776399029	0.658630472691159	0.664395498887757	0.669964031115762
0.571142535964774	0.571570315829801	0.572844049602354	0.574935796012317	0.577800433935629	0.581378202317067	0.585598104572056	0.590380759612508	0.595642107074600	0.601296722060288	0.607260435080836	0.608875557573942	0.615381984059890	0.621954575619569	0.628530961195881	0.635057422784186	0.641489020309726	0.647788893535246	0.653928053801859	0.659883825438843	0.665639936058290
0.563777754656512	0.564216712823781	0.565523778707521	0.567670563107162	0.570611437399259	0.574285875158347	0.578621533107714	0.583537897678431	0.588949600647877	0.594769337253135	0.600911448302464	0.602823405279887	0.609531390295471	0.616311942873200	0.623100560918199	0.629841898610286	0.636489386297891	0.643004959035343	0.649358132168523	0.655525360397480	0.661489393137448
0.556188677469469	0.556640051533147	0.557984308201735	0.560192706303393	0.563218783570569	0.567001233714728	0.571466567513314	0.576532724920963	0.582112886823544	0.588118017216333	0.594460417104555	0.596696512693300	0.603631590480720	0.610646324026419	0.617674315508126	0.624658193129837	0.631549742407731	0.638309146042959	0.644904624841430	0.651311471372763	0.657511320266903
0.548157240034567	0.548623073148533	0.550010559752309	0.552290447691064	0.555415684434924	0.559323813533114	0.563940029304512	0.569180808912368	0.574957372336496	0.581178820989988	0.587755443118751	0.590358895136233	0.597559781392890	0.604849160438317	0.612158036348055	0.619426816550148	0.626605203830346	0.633651520388465	0.640532469316941	0.647221874899103	0.653699954451314
0.539069600132031	0.539553824003477	0.540996453348435	0.543367464600669	0.546619117379854	0.550687387554406	0.555496140056816	0.560959896972697	0.566987505158319	0.573485666849741	0.580362119144840	0.583418856480671	0.590960999360567	0.598603029054652	0.606272907700673	0.613908211616868	0.621455857987677	0.628872045662370	0.636121078945707	0.643175086880548	0.650012907824571];

            app.Dys_1064 = [5.45128826789901	5.45125479842749	5.45115465230027	5.45099109933812	5.45076994760679	5.45049419370941	5.45017398287843	5.44981594110713	5.44942818076295	5.44901692601175	5.44859129304610	5.44815666912330	5.44771959345984	5.44728394120482	5.44685425101201	5.44643502299492	5.44602721778246	5.44563410536231	5.44525609655907	5.44489458556579	5.44454929717686
4.97420897414646	4.97417223577344	4.97406403694735	4.97388560947753	4.97364404090438	4.97334447716484	4.97299607663024	4.97260575313599	4.97218348613243	4.97173646132136	4.97127216272765	4.97079823542814	4.97032272684659	4.96984876961542	4.96938212864491	4.96892510066638	4.96848215487740	4.96805342160675	4.96764256486572	4.96724872893562	4.96687339359835
4.50482703225147	4.50478568514888	4.50466131515188	4.50445785946852	4.50418208045687	4.50384041853100	4.50344263766875	4.50299724877740	4.50251442030569	4.50200375903703	4.50147434050428	4.50093416638291	4.50039049199068	4.49984956945297	4.49931651325608	4.49879510727743	4.49828893497005	4.49780020193730	4.49733013139663	4.49688100136235	4.49645238623289
4.17122245039587	4.17117711513100	4.17104321641668	4.17082471175952	4.17052804025490	4.17016022005877	4.16973162227381	4.16925260194277	4.16873266277650	4.16818338894785	4.16761326813117	4.16703171638944	4.16644627421839	4.16586402988596	4.16528998681884	4.16472884181318	4.16418395694511	4.16365797586333	4.16315246210811	4.16266906259045	4.16220747083519
3.88372465423853	3.88367633116814	3.88353281188540	3.88329876981700	3.88298029744651	3.88258639830831	3.88212713776221	3.88161350731506	3.88105652864780	3.88046765644440	3.87985679040981	3.87923367281383	3.87860655647463	3.87798268050915	3.87736786389864	3.87676662651964	3.87618278128166	3.87561921574630	3.87507813441740	3.87455990049567	3.87406591272835
3.63353439278151	3.63348302249800	3.63332994753308	3.63307989771424	3.63273966964462	3.63231945300231	3.63182910582578	3.63128066293460	3.63068661949574	3.63005800991698	3.62940607132619	3.62874136175469	3.62807212482341	3.62740625751764	3.62675019419589	3.62610880754154	3.62548566818483	3.62488472173333	3.62430695208933	3.62375463175846	3.62322735044065
3.41387701593058	3.41382266080597	3.41365920716749	3.41339361098936	3.41303182771605	3.41258481858916	3.41206305286809	3.41147983083578	3.41084760045201	3.41017928538482	3.40948598448205	3.40877886937457	3.40806714817200	3.40735916327587	3.40666188904608	3.40597969321341	3.40531746305663	3.40467856218408	3.40406437561928	3.40347668137286	3.40291629308698
3.21958116915717	3.21952293616326	3.21935065922657	3.21906814743208	3.21868475103226	3.21821084618462	3.21765767528477	3.21703891306977	3.21636878986969	3.21565986157334	3.21492478566948	3.21417510534977	3.21342087895823	3.21267042103136	3.21193093622510	3.21120812287474	3.21050631897651	3.20982887329203	3.20917809107446	3.20855564316543	3.20796140251388
3.04654769532883	3.04648591405140	3.04630332160101	3.04600501011976	3.04559975863162	3.04509784051897	3.04451358512676	3.04385935416962	3.04315079737235	3.04240133664505	3.04162424713597	3.04083153583290	3.04003397700465	3.03924066966362	3.03845894020064	3.03769493360425	3.03695314419329	3.03623734543342	3.03554953619713	3.03489177987318	3.03426387966165
2.89151666583910	2.89145162299892	2.89125910964181	2.89094444345267	2.89051707777798	2.88998759961692	2.88937103696751	2.88868112591730	2.88793350948836	2.88714302892109	2.88632371221220	2.88548790522886	2.88464662600011	2.88381008620708	2.88298624933227	2.88218017334187	2.88139862918704	2.88064352934888	2.87991873038636	2.87922509975081	2.87856347926903
2.75185358842187	2.75178472524596	2.75158249038892	2.75125104911920	2.75080142128665	2.75024419638152	2.74959491904625	2.74886882542713	2.74808243984236	2.74725039503650	2.74638814610505	2.74550882158941	2.74462396881910	2.74374402104991	2.74287678183813	2.74202936422318	2.74120680385694	2.74041296203992	2.73965029965538	2.73892071657006	2.73822519106058
2.62541169974800	2.62533962453294	2.62512687780937	2.62477946654866	2.62430599042054	2.62372126301055	2.62303943250745	2.62227723737442	2.62145077584629	2.62057724239814	2.61967184782697	2.61874884636584	2.61781974231060	2.61689578869606	2.61598531760990	2.61509627721668	2.61423267503428	2.61339930073354	2.61259930594748	2.61183343909568	2.61110362061342
2.51043761442643	2.51036253033935	2.51013954461974	2.50977446799891	2.50927856000928	2.50866524820778	2.50795073571208	2.50715129643600	2.50628522289729	2.50536966825133	2.50442091454957	2.50345291183615	2.50247969996604	2.50151112803230	2.50055775833220	2.49962559743530	2.49872114759158	2.49784815365048	2.49700955753410	2.49620762993381	2.49544314854271
2.40544654737471	2.40536764430016	2.40513420754394	2.40475229917161	2.40423340234026	2.40359162196260	2.40284316849396	2.40200691575754	2.40110085905892	2.40014245185026	2.39914973829597	2.39813735961313	2.39711927467051	2.39610622130226	2.39510899949037	2.39413371710849	2.39318788191489	2.39227535334288	2.39139812075014	2.39056026945339	2.38976053726755
2.28233164190793	2.28224599075524	2.28199168463421	2.28157643889580	2.28101169393357	2.28031332323928	2.27949936590930	2.27858926481860	2.27760310926730	2.27656039109520	2.27547960094149	2.27437777576376	2.27326894563999	2.27216652748853	2.27108031929296	2.27001904458977	2.26898874448841	2.26799464528758	2.29454637748210	2.29367138953499	2.29283649549144
2.19524933603529	2.19516017127276	2.19489488649557	2.19446155132562	2.19387269579885	2.19314414509265	2.19229522025969	2.19134632030992	2.19031794388789	2.18923080107323	2.18810391381538	2.18695493732402	2.18579946276499	2.18464996327979	2.18351777642047	2.18241144331391	2.18133773585815	2.18030158179226	2.17930671454593	2.17835510566985	2.17744769867730
2.11489998568615	2.11480704331901	2.11453072952344	2.11407923447359	2.11346613135783	2.11270744961479	2.11182306339812	2.11083445774587	2.10976384552198	2.10863173166609	2.10745857726511	2.10626230824359	2.10505898798944	2.10386261673301	2.10268390684350	2.10153234856439	2.10041499273878	2.09933676424583	2.09830129050903	2.09731106984513	2.09636691041971
2.04054876744525	2.04045176845600	2.04016455396269	2.03969489002517	2.03905690097821	2.03826772302070	2.03734801020650	2.03631994037204	2.03520597712812	2.03402854148336	2.03280834816920	2.03156443185167	2.03031329048060	2.02906913849508	2.02784381897684	2.02664676254933	2.02548513501054	2.02436430849206	2.02328810866806	2.02225881847065	2.02127748965019
1.97156302322214	1.97146229207336	1.97116378338812	1.97067597180495	1.97001278393672	1.96919266531805	1.96823700319080	1.96716886251618	1.96601159082209	1.96478851027748	1.96352102550208	1.96222889902372	1.96092918927469	1.95963731922387	1.95836483852236	1.95712205737140	1.95591576168018	1.95475204792225	1.95363493052646	1.95256622748026	1.95154765036443
1.90739660589854	1.90729232797100	1.90698212365492	1.90647560966996	1.90578737940910	1.90493609435510	1.90394419390594	1.90283543616881	1.90163442695163	1.90036507333624	1.89904979254762	1.89770931083069	1.89636102588395	1.89502058861334	1.89370043193949	1.89241130326745	1.89116024869265	1.88995352403164	1.88879476611134	1.88768680855100	1.88663056636709
1.84757577383155	1.84746752015149	1.84714583369413	1.84662047044427	1.84590672187876	1.84502392720409	1.84399534866837	1.84284567841966	1.84160045292860	1.84028442074663	1.83892100802562	1.83753116084948	1.83613384701006	1.83474464056521	1.83337670288791	1.83204066358344	1.83074433045396	1.82949390842923	1.82829344382152	1.82714571402299	1.82605173423503
1.79168593510308	1.79157370241479	1.79124046740146	1.79069607839353	1.78995640722087	1.78904181668114	1.78797618814522	1.78678527364148	1.78549534414606	1.78413205827236	1.78271993664670	1.78128070138414	1.77983365831275	1.77839514648399	1.77697894787652	1.77559545963998	1.77425354804981	1.77295928631692	1.77171672596186	1.77052877600665	1.76939652830766
1.73936464340185	1.73924854763188	1.73890354384951	1.73833992058907	1.73757407386726	1.73662712382944	1.73552392036462	1.73429123741035	1.73295621753100	1.73154523250049	1.73008390704565	1.72859455557174	1.72709719869153	1.72560896906948	1.72414394887266	1.72271295871154	1.72132511851532	1.71998649684259	1.71870144356543	1.71747303217188	1.71630239653235
1.69029277182077	1.69017255785874	1.68981566087690	1.68923253230185	1.68844029315306	1.68746081698297	1.68631956235656	1.68504452457213	1.68366367953584	1.68220474490527	1.68069353864381	1.67915362785000	1.67760568232332	1.67606704178747	1.67455265610492	1.67307365487241	1.67163931470683	1.67025593601034	1.66892813219511	1.66765883398990	1.66644927144780
1.64418752019670	1.64406326982629	1.64369418108427	1.64309139315792	1.64227238210239	1.64125977179529	1.64008030878152	1.63876232870622	1.63733529074766	1.63582748265350	1.63426592072782	1.63267496119938	1.63107565352104	1.62948641514360	1.62792212223751	1.62639461344526	1.62491319851582	1.62348481939178	1.62211391745238	1.62080347942300	1.61955463560585
1.60079932042461	1.60067079112671	1.60028975532755	1.59966681546073	1.59882070546214	1.59777460579156	1.59655621833381	1.59519508395227	1.59372104350808	1.59216397864991	1.59055154266584	1.58890895490068	1.58725781108666	1.58561724377106	1.58400257250627	1.58242618442945	1.58089739686565	1.57942335712037	1.57800875361402	1.57665662851986	1.57536851502097
1.55990495781954	1.55977246345779	1.55937871954909	1.55873554497452	1.55786206643221	1.55678212992533	1.55552433106665	1.55411925040137	1.55259775145295	1.55099078031589	1.54932690284670	1.54763174673621	1.54592834880475	1.54423597580040	1.54257024445071	1.54094429888495	1.53936761191923	1.53784767740884	1.53638900862235	1.53499493833779	1.53366691528813
1.52130678347006	1.52116995063510	1.52076371629939	1.52010019050814	1.51919846281917	1.51808439478052	1.51678660913436	1.51533698584880	1.51376764994670	1.51211006556107	1.51039397997562	1.50864595257550	1.50688938931988	1.50514450758885	1.50342737502879	1.50175137734024	1.50012639544012	1.49855976542875	1.49705659599235	1.49562017747720	1.49425187063703
1.48482524416075	1.48468420021949	1.48426489674534	1.48358053637175	1.48265098178507	1.48150192338444	1.48016360275333	1.47866877073975	1.47705070564251	1.47534210850396	1.47357304781721	1.47177151249828	1.46996131583967	1.46816329607107	1.46639405213628	1.46466739408148	1.46299339071436	1.46137979675370	1.45983172800812	1.45835258402130	1.45694365641055
1.45030193460022	1.45015632993273	1.44972427546486	1.44901883102331	1.44806042100981	1.44687606227579	1.44549647656183	1.44395616467840	1.44228890540347	1.44052825597244	1.43870572198336	1.43684975395961	1.43498531512075	1.43313346507500	1.43131161036888	1.42953369594329	1.42781033442027	1.42614913393850	1.42455572324058	1.42303326263605	1.42158315955056
1.41759158014813	1.41744166602002	1.41699646630680	1.41626943426503	1.41528190313049	1.41406166038575	1.41264060429538	1.41105372165671	1.40933644517039	1.40752324739035	1.40564654837608	1.40373545485083	1.40181595647538	1.39990968857951	1.39803448043351	1.39620474194953	1.39443127901987	1.39272217683673	1.39108269829296	1.38951647519915	1.38802501092083
1.38656534027141	1.38641098604227	1.38595231601056	1.38520351155300	1.38418640554930	1.38292973808825	1.38146637587572	1.37983244325863	1.37806434545087	1.37619774809039	1.37426601004201	1.37229941926639	1.37032410938261	1.36836270539113	1.36643343633482	1.36455112985127	1.36272692482316	1.36096936860310	1.35928351737747	1.35767292113197	1.35613963181401
1.35710560024875	1.35694660169340	1.35647438504564	1.35570335765890	1.35465628072066	1.35336259375582	1.35185628974933	1.35017452203932	1.34835495980419	1.34643413558338	1.34444668211013	1.34242331181154	1.34039149024928	1.33837434364664	1.33639035092167	1.33445487761682	1.33257956624434	1.33077273386538	1.32903981216637	1.32738462507358	1.32580875122156
1.32910470590285	1.32894112234631	1.32845511799539	1.32766145286117	1.32658389695653	1.32525252521993	1.32370266753480	1.32197230219148	1.32010041121161	1.31812454674513	1.31608060762683	1.31399995886966	1.31191079841496	1.30983667896245	1.30779744877946	1.30580822480027	1.30388091822030	1.30202416154078	1.30024394122789	1.29854345830527	1.29692472841093
1.30246666642524	1.30229842854468	1.30179824060381	1.30098194046239	1.29987325208475	1.29850367566248	1.29690945666529	1.29512966859232	1.29320454543470	1.29117303936855	1.28907158762384	1.28693259737422	1.28478529516986	1.28265382583923	1.28055829638945	1.27851454033460	1.27653477377022	1.27462758504248	1.27279904799416	1.27105284882068	1.26939073592741
1.27710281787469	1.27692965504614	1.27641519844914	1.27557557319846	1.27443536011431	1.27302672104366	1.27138744338894	1.26955774862483	1.26757853933774	1.26549023632572	1.26333017363941	1.26113224128327	1.25892612312258	1.25673630525153	1.25458383846066	1.25248470252468	1.25045159844314	1.24849324897899	1.24661607081442	1.24482355186435	1.24311748555542
1.25293243872820	1.25275421321169	1.25222549412690	1.25136197569866	1.25018960674660	1.24874155044761	1.24705613131248	1.24517523946713	1.24314132186867	1.24099532185206	1.23877602773330	1.23651807651703	1.23425192296490	1.23200303281572	1.22979267638818	1.22763755896637	1.22555040081720	1.22354027434424	1.22161365367199	1.21977422166123	1.21802367751549
1.22988223198850	1.22969902720083	1.22915521857603	1.22826763503525	1.22706261185541	1.22557417918794	1.22384212372547	1.22190948847577	1.21981953789935	1.21761487334036	1.21533531760420	1.21301644116166	1.21068921091009	1.20838037901384	1.20611136291079	1.20389943423638	1.20175735953119	1.19969486959233	1.19771824175482	1.19583122044737	1.19403562779174
1.20788390526212	1.20769576373320	1.20713686461666	1.20622465391611	1.20498609362086	1.20345657017225	1.20167700332873	1.19969142988503	1.19754467548477	1.19528052013474	1.19293944091382	1.19055850134268	1.18816960910476	1.18579966427524	1.18347102080850	1.18120127281797	1.17900380589442	1.17688795624225	1.17486057161029	1.17292536970920	1.17108421336261
1.18687643777368	1.18668313193993	1.18610886601920	1.18517137811068	1.18389890438345	1.18232742424590	1.18049932763670	1.17846007077056	1.17625531107126	1.17393050938038	1.17152703294934	1.16908301842755	1.16663122924245	1.16419935098842	1.16181030401491	1.15948190626143	1.15722792534853	1.15505821286343	1.15297924051605	1.15099522193343	1.14910779986718
1.16680224492384	1.16660353186055	1.16601335121554	1.16505055066178	1.16374331104055	1.16212918057472	1.16025174258678	1.15815738462860	1.15589370882080	1.15350714145238	1.15104021209240	1.14853196081452	1.14601619039911	1.14352140736732	1.14107086208127	1.13868286699909	1.13637150851845	1.13414691163684	1.13201604224424	1.12998237069485	1.12804813799125
1.14760930017661	1.14740518401869	1.14679897328356	1.14581000779535	1.14446738843375	1.14280966338706	1.14088181465006	1.13873143507756	1.13640775915348	1.13395802993157	1.13142655810541	1.12885293045455	1.12627221049126	1.12371325597362	1.12120014546704	1.11875175532771	1.11638207045577	1.11410209831361	1.11191795721778	1.10983410426433	1.10785224733008
1.12924948230781	1.12903972047818	1.12841725989817	1.12740125895985	1.12602242246948	1.12432045759564	1.12234123980611	1.12013379549626	1.11774878454015	1.11523493804042	1.11263743470075	1.10999728626688	1.10735016988477	1.10472613360988	1.10214936290042	1.09963943461688	1.09721085220312	1.09487403226253	1.09263620741694	1.09050117200515	1.08847112211998
1.11167792517813	1.11146258995687	1.11082324997041	1.10978008636272	1.10836448490744	1.10661694955729	1.10458485446326	1.10231939971867	1.09987167477853	1.09729236673627	1.09462771860788	1.09191993859587	1.08920537726672	1.08651496440000	1.08387353054927	1.08130095771755	1.07881210358296	1.07641799618202	1.07412548408719	1.07193878196651	1.06985969276121
1.09485369911350	1.09463257855339	1.09397626004996	1.09290509017957	1.09145158379760	1.08965759394004	1.08757205706358	1.08524697280147	1.08273566329676	1.08008936771556	1.07735624311918	1.07457926436448	1.07179607043542	1.06903797126548	1.06633067955226	1.06369446908449	1.06114461373903	1.05869194361992	1.05634389552557	1.05410454210603	1.05197575512931
1.07873851939469	1.07851150471616	1.07783757525201	1.07673792503970	1.07524571536489	1.07340440075344	1.07126393931568	1.06887809046720	1.06630142684379	1.06358684403001	1.06078384299302	1.05793650114610	1.05508323105517	1.05225628060008	1.04948194638692	1.04678088824028	1.04416870038838	1.04165670048588	1.03925206671605	1.03695943243578	1.03477999836665
1.06329783236440	1.06306478406764	1.06237293787375	1.06124394112123	1.05971233411052	1.05782238379134	1.05562586662305	1.05317773550562	1.05053445064774	1.04775028213027	1.04487593942334	1.04195666431907	1.03903182463033	1.03613452595611	1.03329181885464	1.03052483247545	1.02784942758563	1.02527692438517	1.02281493133746	1.02046768300492	1.01823701222055
1.04849881285693	1.04825951254250	1.04754912887156	1.04639020256915	1.04481812561296	1.04287843633491	1.04062443224954	1.03811297365924	1.03540145219329	1.03254605895110	1.02959876269794	1.02660593597490	1.02360821395038	1.02063932386217	1.01772705312865	1.01489267283711	1.01215280960347	1.00951868220352	1.00699820754785	1.00459585091065	1.00231303037383
1.02635096864827	1.02610251673722	1.03333612705820	1.03214664020382	1.03053311214599	1.02854257219521	1.02622965651762	1.02365296258659	1.02087187285678	1.01794364955458	1.01492183072942	1.01185398479325	1.00878176923823	1.00573973249823	1.00275629417590	0.999853380654722	0.997047702755469	0.994350834949173	0.991771083929030	0.989312187555348	0.986976314773271
1.01294480947696	1.01269001824900	1.01193386540125	1.01069977214758	1.00902508941509	1.00695821405347	1.01241289510112	1.00976966697172	1.00691696573040	1.00391429453647	1.00081629529582	0.997671735498254	0.994523422344510	0.991406809137887	0.988350703662945	0.985377940925066	0.982505242808516	0.979744718353189	0.977104160055466	0.974588003841222	0.972197962545623
1.00008689125461	0.999825629850248	0.999050276156609	0.997785073285537	0.996068357248590	0.993949701029684	0.991486717875669	0.988741286150240	0.993511025278916	0.990432109134716	0.987256022290913	0.984033075521588	0.980806857100384	0.977613998562150	0.974484007970578	0.971439727197002	0.968498693472262	0.965672976969815	0.962970665330651	0.960396143805364	0.957951241453296
0.987753208562708	0.987485409840342	0.986690401226229	0.985393289392120	0.983633433740864	0.981461721974533	0.978937460722171	0.976124215802861	0.973086322440951	0.977472239646258	0.974216235483198	0.970913145469212	0.967607340828078	0.964336467996264	0.961130682174855	0.958013557955112	0.955002762014332	0.952110593883194	0.949345338007007	0.946711418521520	0.944210218071571
0.975921412916963	0.975646803898728	0.974831633024154	0.973501931415921	0.971697775154974	0.969471763460476	0.966884864853548	0.964002075350915	0.960889631007053	0.957611770913950	0.961673693066107	0.958288296273431	0.954901137103409	0.951550586622053	0.948267457202674	0.945075862612952	0.941993760769161	0.939033795633787	0.936204301466360	0.933509783327638	0.930951494934165
0.964571143813923	0.964289507556277	0.963453846729065	0.962090482987047	0.960241016561904	0.957959453644986	0.955308157322889	0.952354178940119	0.949165462891865	0.945807933991527	0.942342563926174	0.938824231654538	0.942667177928014	0.939234905039978	0.935872719001401	0.932604808101712	0.929449974641823	0.926420802067055	0.923525699280454	0.920769217871612	0.918152637087719
0.953683046737885	0.953394421703623	0.952537453002974	0.951139687774391	0.949243684356818	0.946904956045911	0.944187777381972	0.941160793148161	0.937894020864228	0.934454873629272	0.930905903611712	0.927303481579528	0.923696302702728	0.927368902082868	0.923925573640310	0.920579952957416	0.917350638408434	0.914250717994127	0.911288674473799	0.908468991389944	0.905793117205407
0.943239612253556	0.942943537910496	0.942064993253693	0.940631715239042	0.938688001091440	0.936290738111758	0.933505849868459	0.930404038415609	0.927057031576886	0.923534196533804	0.919899753459924	0.916211325603404	0.912518679399133	0.908863708249407	0.912407523089897	0.908982171562463	0.905676748807146	0.902504487504055	0.899474094603812	0.896589723239179	0.893853153664756
0.933224296192723	0.932920696743923	0.932019812067576	0.930550224988864	0.928557479098576	0.926099900009862	0.923245525313200	0.920066880051885	0.916637759706488	0.913029035279067	0.909306942477992	0.905530313562882	0.901750252390543	0.898009559234546	0.894342759705444	0.897793657400506	0.894410278427651	0.891164118177914	0.888063570811115	0.885113401667569	0.882314878214623
0.923622219980011	0.923310838655258	0.922387026190088	0.920879987169713	0.918836649343960	0.916317256378801	0.913391472151734	0.910133931435798	0.906620315448524	0.902923601324953	0.899111482543114	0.895244512502630	0.891374897191752	0.887546353246406	0.883794367703404	0.880146458198480	0.883534687787535	0.880212668407715	0.877040526819079	0.874022953697838	0.871161111628256
0.914418618884347	0.914099243252370	0.913151637924663	0.911606254711603	0.909510950976262	0.906927899868752	0.903928419754197	0.900589843837070	0.896989440372421	0.893202338815022	0.889297928924819	0.885338222573860	0.881376774563921	0.877458268890541	0.873619016791047	0.869887054501295	0.866283458223300	0.869633851853787	0.866388471854289	0.863301811687640	0.860375204949291
0.905601417430587	0.905273826264157	0.904301912381284	0.902716690034878	0.900567893455043	0.897919158473333	0.894844245990201	0.891422078645450	0.887732468003618	0.883852422563135	0.879853114438547	0.875798162018120	0.871742390321040	0.867731737576676	0.863802949140643	0.859984904629739	0.856299044118352	0.852760261616912	0.856092952625758	0.852935719377570	0.849942670506444
0.897158007345889	0.896821825216917	0.895824779629712	0.894198681542295	0.891994678181226	0.889278276483733	0.886125409720849	0.882617307672828	0.878835869406551	0.874860127360861	0.870763378882689	0.866610590404326	0.862458026050932	0.858352674452782	0.854332143800936	0.850425869083221	0.846655590756982	0.843036763885469	0.839578978060496	0.842909780096400	0.839848861309267
0.889078070311162	0.888733140916238	0.887709983422414	0.886041702110192	0.883780790932526	0.880994592538166	0.877761372559649	0.874164551905689	0.870288535443138	0.866214468491633	0.862017286358388	0.857763824554288	0.853511803823068	0.849309208691833	0.845194442603911	0.841197601746208	0.837340921057629	0.833639853852630	0.830104369219035	0.826739702243711	0.830080753955263
0.881351547453116	0.880997604283483	0.879947556804938	0.878235606051983	0.875915714765799	0.873057536461024	0.869741374072874	0.866053154881902	0.862079499587981	0.857903944358629	0.853603350469810	0.849246410789552	0.844892077655207	0.840589406734853	0.836377882941153	0.832288050084333	0.828342624856258	0.824557358847921	0.820942163036997	0.817502428915819	0.814239557017517
0.873969528281051	0.873606178690381	0.872528341161634	0.870771239682089	0.868390512015908	0.865457760354225	0.862055888084536	0.858273158174933	0.854198901049244	0.849918491860647	0.845511481337138	0.841047756654617	0.836588065704464	0.832182407974410	0.827871392414665	0.823686071368235	0.819649513487087	0.815777693012847	0.812080694152143	0.808563935419469	0.805228765273453
0.866924061833716	0.866551079067302	0.865444293254127	0.863640417890397	0.861196669525625	0.858186874023422	0.854696298563418	0.850815999860650	0.846637538785380	0.842249176317715	0.837732037224294	0.833158363024201	0.828589990045322	0.824078489432226	0.819664929512296	0.815381329949595	0.811251039203395	0.807290358061301	0.803509410023492	0.799913592320837	0.796504093888445
0.860208123951598	0.859824958294891	0.858688506448124	0.856836021695181	0.854326883087757	0.851237180960065	0.847654906126504	0.843673397238547	0.839387341479828	0.834887301038227	0.830256543780763	0.825569239502514	0.820888778496381	0.816267980306028	0.811748944237830	0.807364110820669	0.803137413632035	0.799085267429849	0.795218067701359	0.791540988681769	0.788055360959616
0.853815550366913	0.853421918647666	0.852254385763699	0.850351565126533	0.847774633969578	0.844602070474376	0.840924460875641	0.836838332540953	0.832440846637222	0.827825079025509	0.823076834408486	0.818272042147372	0.813475966250283	0.808742353175921	0.804114410761511	0.799625302570639	0.795299234693883	0.791153010529631	0.787197036338874	0.783436496789405	0.779872521526015
0.847741469592734	0.847336841892448	0.846137104870321	0.844181972829614	0.841534526282517	0.838275900949466	0.834499443925063	0.830304594875826	0.825791465590168	0.821055893464952	0.816185926236253	0.811259733855825	0.806343977257771	0.801493814773741	0.796753481187091	0.792156665411886	0.787728230873430	0.783485006500278	0.779437588415529	0.775591087025854	0.771946478981714
0.841981384234409	0.841565480781045	0.840332120967856	0.838322472943409	0.835601648701984	0.832253514932718	0.828374268213679	0.824066596483538	0.819433371372671	0.814573488846808	0.809577384827763	0.804525259977300	0.799485598830676	0.794515014094450	0.789658423334053	0.784950426361942	0.780416287586539	0.776072996324738	0.771931288809723	0.767996249195984	0.764268613729244
0.836532503216816	0.836104809116381	0.834836314245830	0.832769903264003	0.829972678849148	0.826531254033752	0.822545008519764	0.818119781991565	0.813361760567729	0.808372676880336	0.803245714893158	0.798062981222743	0.792894895803320	0.787799465875305	0.782822641621056	0.777999617399770	0.773356108598342	0.768909595803069	0.764670489689670	0.760644082485737	0.756830876493405
0.831392639770520	0.830952568111113	0.829647515262248	0.827521762778967	0.824644727436989	0.821105895335003	0.817008084336188	0.812460461311332	0.807572569084274	0.802449038661278	0.797185783346922	0.791867472092663	0.786566117123365	0.781341116468519	0.776239639926158	0.771297552715631	0.766541000296098	0.761987484004929	0.757647902399574	0.753526989782293	0.749625432259306
0.826560917557266	0.826108010100199	0.824764553840798	0.822576709422930	0.819616260662592	0.815975745234061	0.811761420457202	0.807085958355649	0.802062508856020	0.796798924560795	0.791393881771032	0.785934384973490	0.780494446315805	0.775134991759217	0.769904132957705	0.764838448603107	0.759964557080768	0.755300438874953	0.750856735906275	0.746638204237256	0.742645327549782
0.822037928298534	0.821571352955099	0.820187903070525	0.817934945443637	0.814886962656596	0.811140056549728	0.806803676701014	0.801994594293466	0.796829522314768	0.791419748729615	0.785866762946490	0.780260167730278	0.774675995476729	0.769176496369821	0.763811080868174	0.758617088275207	0.753621590039868	0.748842603154613	0.744290872256746	0.739971277934632	0.735883824352291
0.817825205304116	0.817344418745409	0.815918696771289	0.813597418836248	0.810457752625972	0.806598959388036	0.802134825017810	0.797185699649469	0.791872407526511	0.786309547048655	0.780601983849433	0.774841821002039	0.769107088763044	0.763461780703982	0.757956180129655	0.752628703837942	0.747506626424624	0.742608303727636	0.737944528384636	0.733519977092105	0.729334407683041
0.813926535917677	0.813430561808122	0.811960614901967	0.809567281777440	0.806331146939609	0.802354842887625	0.797756169272385	0.792660193692323	0.787191344052770	0.781468139046248	0.775598763545577	0.769677851817602	0.763785842875261	0.757988095342337	0.752336360529591	0.746869611346578	0.741615817090661	0.736593249583232	0.731812927143359	0.727279229659551	0.722991846970624
0.810346714477843	0.809835061799926	0.808318023239763	0.805848973862102	0.802510849346438	0.798410690982070	0.793670615845629	0.788419852001484	0.782787429638358	0.776895873723646	0.770856521401451	0.764767182188812	0.758710258742583	0.752753056069628	0.746948571227082	0.741336318789013	0.735944868762947	0.730792887621924	0.725891043606402	0.721243730866663	0.716850276613146
0.807093123031496	0.806564541174246	0.804998077451905	0.802448584857087	0.799002812289293	0.794771751044696	0.789882209283986	0.784468227375817	0.778663457729923	0.772594548096166	0.766376400027571	0.760109974779913	0.753879980139624	0.747755565831788	0.741790782930754	0.736026257063990	0.730490909248095	0.725203475716514	0.720174729030246	0.715408822137340	0.710904802526757
0.797266399739430	0.796748666628384	0.795212972392088	0.792711085724551	0.795816090984222	0.791446185506383	0.786398366885378	0.780811666534559	0.774824513587136	0.768568226861196	0.762161608875929	0.755708436462215	0.749296267641287	0.742995875069454	0.736862722704318	0.730938375848169	0.725252038902564	0.719822648070280	0.714661051565114	0.709771083111084	0.705151417767675
0.794466227411393	0.793932445993376	0.792349512339104	0.789771043418227	0.786281288672362	0.781988390640939	0.783227256808359	0.777457154909445	0.771276715815102	0.764821913759018	0.758215733650181	0.751565254984860	0.744960569212496	0.738474598860778	0.732164068623509	0.726071363775940	0.720226223265352	0.714647750666537	0.709346538635319	0.704326352994240	0.699585459687522
0.791994925926101	0.791444214478766	0.789811259940383	0.787151467344562	0.783552290615915	0.779126106470315	0.774001856649078	0.774415550046330	0.768029408955714	0.761363459162294	0.754545315441991	0.747685491832107	0.740876910269372	0.734194491668029	0.727696395443990	0.721425859067972	0.715413089179621	0.709677395328803	0.704229162374627	0.699071908173067	0.694203474848522
0.789869130161514	0.789300435301441	0.787613951399637	0.784867566983045	0.781152088235622	0.776584069768192	0.771297249431385	0.765433695388866	0.765095163426291	0.758203913059995	0.751159609548166	0.744076927097531	0.737051354970652	0.730160133407092	0.723462846328162	0.717003785422066	0.710813470160382	0.704911312539992	0.699307701225468	0.694005579409091	0.689002480825095
0.788110255182543	0.787522122515490	0.785778419247394	0.782939271299645	0.779099364227336	0.774379659909667	0.768919224437362	0.762865513717263	0.756365800535866	0.755357888117554	0.748071324968672	0.740749830340240	0.733492374466385	0.726378234038730	0.719468630225578	0.712808693726600	0.706429375500973	0.700350295206637	0.694581607826558	0.689125825181848	0.683979941082837
0.786745618917944	0.786136721334631	0.784331429117351	0.781392565994254	0.777418600284604	0.772535810100388	0.766888871627884	0.760630949578310	0.753914988493593	0.746886508612201	0.745296814227730	0.737718519139505	0.730211704663957	0.722858212118051	0.715720876216112	0.708845778737029	0.702264369217569	0.695996236168552	0.690051248742737	0.684431509997492	0.679133442215753
0.785813476670250	0.785181949313285	0.783309974737884	0.780262892603673	0.776143802309835	0.771084569701639	0.765235878076598	0.758757271053862	0.751807898998231	0.744538834035138	0.742859785198321	0.735003077686360	0.727226402076133	0.719614248733865	0.712231176277260	0.705124095046069	0.698324937773142	0.691853424538162	0.685718926409330	0.679923148754575	0.674461956480324
0.785364566637801	0.784708384293701	0.782763347165535	0.779598136553162	0.775320629525827	0.770068596021333	0.763999841900487	0.757280820300010	0.750077369750266	0.742546630960929	0.734831240705702	0.732631080451429	0.724559897834425	0.716665722426081	0.709015046405626	0.701655694135857	0.694620171180204	0.687927910888077	0.681588195819263	0.675601869638439	0.669964031115762
0.785472052209725	0.784788573613454	0.782762724816464	0.779466966853637	0.775014373759364	0.769549709324152	0.763238252852334	0.756254337716547	0.748771207318270	0.740952788997091	0.732947535996249	0.724885156890404	0.722246506078976	0.714041419129857	0.706096134716761	0.698459632950574	0.691164520310117	0.684230316918410	0.677665609006388	0.671470824609468	0.665639936058290
0.786244700163696	0.785530427190611	0.783414111501686	0.779971677279515	0.775322854872191	0.769619906041093	0.763036715331519	0.755756576475132	0.747961028054450	0.739821476400479	0.731493174681129	0.723110955488366	0.714788107369874	0.711784593467619	0.703509880452554	0.695563861688942	0.687979407169358	0.680775790081012	0.673960987316737	0.667534451598810	0.661489393137448
0.787859467583773	0.787110051311167	0.784889587066287	0.781279063059927	0.776404948587346	0.770428930564657	0.763534671514766	0.755915608933972	0.747763084664093	0.739257155845422	0.730560512901934	0.721814169390393	0.713136109174465	0.709965202729079	0.701313738656107	0.693014018747872	0.685099426706989	0.677588842248459	0.670489484131929	0.663799701176666	0.657511320266903
0.790651636675700	0.789860120249754	0.787515458519085	0.783704089231405	0.778561569042053	0.772260047496945	0.764995470159310	0.756973511278682	0.748396931604645	0.739456484045869	0.730323476135273	0.721146226294747	0.712048256773466	0.703128814999409	0.699614328380683	0.690894135650381	0.682587685743694	0.674713167252921	0.667276822450889	0.660275659540373	0.653699954451314
0.795476366555382	0.794630071979826	0.792123500259240	0.788050561170830	0.782558351430959	0.775833612511804	0.768088003044370	0.759543135641884	0.750417025748864	0.740913843557284	0.731216460864816	0.721482409245895	0.711842466990803	0.702401115533768	0.693238600739400	0.689400738738614	0.680584804066918	0.672237781051208	0.664364650308072	0.656960332260897	0.650012907824571];

            app.Dzs_1064 = [186.471110254385	186.471105323818	186.471111923790	186.471108441951	186.471110711360	186.471105561450	186.471100954801	186.471105219319	186.471107694427	186.471108831618	186.471110982463	186.471114660823	186.471106653921	186.471114848362	186.471114002738	186.471107491819	186.471112069169	186.471109832757	186.471106685337	186.471113398325	186.471108452008
153.674538287455	153.674539872736	153.674539584593	153.674541088199	153.674539394299	153.674538518796	153.674542665171	153.674542336829	153.674540734359	153.674538898637	153.674538873215	153.674539338365	153.674538587741	153.674538062602	153.674540579579	153.674540832024	153.674538185458	153.674540478145	153.674539652992	153.674537106951	153.674541640786
128.703807363964	128.703803978788	128.703800169630	128.703808399165	128.703807637035	128.703809206208	128.703804285254	128.703807691916	128.703807261417	128.703809253099	128.703803013737	128.703804874393	128.703805891679	128.703808717809	128.703806016143	128.703808440902	128.703807768097	128.703805494140	128.703810232311	128.703807616397	128.703803688314
109.586141154267	109.586141863032	109.586140908621	109.586139964337	109.586143966972	109.586140846603	109.586139848682	109.586139788227	109.586140482079	109.586142326020	109.586141809060	109.586141826863	109.586142265892	109.586140304804	109.586142371600	109.586141881142	109.586144424444	109.586144315482	109.586141941336	109.586144332085	109.586142199001
94.4236767629950	94.4236794579809	94.4236778237838	94.4236785263733	94.4236802314728	94.4236775716866	94.4236784897040	94.4236760590420	94.4236787986021	94.4236780904834	94.4236767167321	94.4236799949479	94.4236783596342	94.4236783576315	94.4236798935172	94.4236769396628	94.4236789028666	94.4236781599829	94.4236774820480	94.4236777033340	94.4236764471062
82.2023025307858	82.2023042128818	82.2023032427573	82.2023016620601	82.2023017567514	82.2023043438803	82.2023035892756	82.2023025595242	82.2023010912630	82.2023016103298	82.2022999572585	82.2023014554152	82.2023041862992	82.2023021172145	82.2023024687803	82.2023006908463	82.2023025839842	82.2023025758714	82.2023029491523	82.2023026807567	82.2023041343422
72.2121333318852	72.2121336121310	72.2121343739826	72.2121317880557	72.2121328866438	72.2121358084806	72.2121339705358	72.2121352915672	72.2121341258032	72.2121330364034	72.2121361858233	72.2121331126919	72.2121354012839	72.2121349409891	72.2121319488439	72.2121336397442	72.2121328453506	72.2121335781045	72.2121348410743	72.2121340049993	72.2121330285764
63.8891974685194	63.8891991613413	63.8892023980313	63.8891998240457	63.8892005476656	63.8891993265956	63.8892021519844	63.8891999409761	63.8891997190153	63.8891984522300	63.8891982559242	63.8891987228002	63.8891989880683	63.8891985709942	63.8891997464685	63.8891986024047	63.8891984792216	63.8892002765464	63.8891984173053	63.8892001090274	63.8892000859731
56.9247057410144	56.9247047823586	56.9247029044929	56.9247031111620	56.9247019727290	56.9247046562075	56.9247063755869	56.9247049412799	56.9247032535636	56.9247047276467	56.9247048287266	56.9247034563579	56.9247037312173	56.9247043555791	56.9247047418586	56.9247031360998	56.9247034185426	56.9247041016749	56.9247037679099	56.9247053473543	56.9247022980763
51.0400279763472	51.0400259692044	51.0400257295029	51.0400272701007	51.0400268077811	51.0400263180717	51.0400290312861	51.0400262349518	51.0400266049351	51.0400265097789	51.0400285228748	51.0400264371889	51.0400259584394	51.0400258115511	51.0400261179192	51.0400266991772	51.0400284944011	51.0400256245560	51.0400275186104	51.0400265202200	51.0400268955915
46.0242907221244	46.0242912566613	46.0242925773028	46.0242920523675	46.0242921875593	46.0242939720630	46.0242914694561	46.0242924016352	46.0242917996402	46.0242921272989	46.0242919389060	46.0242918110828	46.0242918928616	46.0242912683543	46.0242909475397	46.0242918861468	46.0242917634572	46.0242919761974	46.0242918852686	46.0242918003866	46.0242904712304
41.7155064628521	41.7155064704494	41.7155069637820	41.7155064848159	41.7155061039398	41.7155051455595	41.7155069472173	41.7155069141826	41.7155080976330	41.7155073780760	41.7155068847987	41.7155064402474	41.7155062270461	41.7155059495433	41.7155065629683	41.7155072156998	41.7155053981924	41.7155059476887	41.7155054254040	41.7155057733804	41.7155057577363
37.9478564538625	37.9478572502326	37.9478555687365	37.9478571952153	37.9478559204606	37.9478569608792	37.9478564471128	37.9478576127703	37.9478571239773	37.9478556082305	37.9478571318751	37.9478568348545	37.9478570996347	37.9478565366834	37.9478560365993	37.9478572476972	37.9478571141006	37.9478572264928	37.9478563079576	37.9478571001986	37.9478570064217
34.7038773558433	34.7038778207330	34.7038770040922	34.7038772857919	34.7038788878099	34.7038777336723	34.7038785990135	34.7038769737651	34.7038782107606	34.7038783876174	34.7038791164446	34.7038789198767	34.7038783450985	34.7038767927213	34.7038776594842	34.7038776263195	34.7038781521692	34.7038786900479	34.7038781530309	34.7038776098891	34.7038795133199
31.8266420573471	31.8266416215619	31.8266416178923	31.8266427983544	31.8266414890066	31.8266429125129	31.8266409750230	31.8266417273837	31.8266411818900	31.8266424295705	31.8266417461829	31.8266417701608	31.8266412338134	31.8266414181613	31.8266409828377	31.8266416828938	31.8266415726507	31.8266416381200	31.8266417358360	31.8266418116429	31.8266414006152
29.2919404360281	29.2919400613469	29.2919404599994	29.2919405045465	29.2919397872581	29.2919402157380	29.2919398506701	29.2919405912613	29.2919401031569	29.2919401800731	29.2919400881783	29.2919405341906	29.2919403225576	29.2919401872804	29.2919401689693	29.2919407030502	29.2919400476181	29.2919402240632	29.2919403896724	29.2919397376736	29.2919402549740
27.0478408662004	27.0478399673685	27.0478414067454	27.0478399700872	27.0478396048723	27.0478401238369	27.0478410769173	27.0478414154014	27.0478411781567	27.0478407934518	27.0478394741913	27.0478411268592	27.0478402194859	27.0478405608100	27.0478401294007	27.0478403426402	27.0478404454206	27.0478401448588	27.0478408829941	27.0478405007063	27.0478402410599
25.0518616124228	25.0518604288445	25.0518611853890	25.0518615160099	25.0518616411481	25.0518607829437	25.0518606415117	25.0518613798017	25.0518606581444	25.0518608367168	25.0518613389107	25.0518614138160	25.0518608723302	25.0518600105901	25.0518610786905	25.0518606694068	25.0518613286848	25.0518607517395	25.0518608692069	25.0518611470209	25.0518614361309
23.2391848231282	23.2391852996833	23.2391849416960	23.2391858246910	23.2391853875594	23.2391854830883	23.2391843384563	23.2391852659196	23.2391841839446	23.2391849697807	23.2391851461812	23.2391845851565	23.2391853433956	23.2391852408932	23.2391846926598	23.2391845941792	23.2391854722893	23.2391841460937	23.2391851283191	23.2391854008520	23.2391848018004
21.6415701521973	21.6415700013867	21.6415701394340	21.6415704504896	21.6415707524358	21.6415693723054	21.6415699644028	21.6415706916433	21.6415702794230	21.6415701176564	21.6415695741873	21.6415694810306	21.6415707918552	21.6415711808376	21.6415700144407	21.6415701985649	21.6415705842430	21.6415701902816	21.6415707667036	21.6415703536350	21.6415702971906
20.2035945618756	20.2035950518740	20.2035941903650	20.2035952351353	20.2035947693183	20.2035945673989	20.2035949583192	20.2035953661408	20.2035944697234	20.2035945725055	20.2035955497442	20.2035950013166	20.2035949610186	20.2035948618715	20.2035946443117	20.2035949004453	20.2035951383917	20.2035951060505	20.2035944421639	20.2035947401353	20.2035945061368
18.8784100334416	18.8784102827402	18.8784106222076	18.8784105843655	18.8784103086454	18.8784104288332	18.8784099798952	18.8784105563628	18.8784101640329	18.8784099731338	18.8784102549066	18.8784097928780	18.8784101932649	18.8784099781613	18.8784103727641	18.8784104655692	18.8784094417378	18.8784096541072	18.8784099077768	18.8784091789697	18.8784104933772
17.7025875406880	17.7025875583575	17.7025873971696	17.7025878918403	17.7025872919523	17.7025882896193	17.7025874640328	17.7025874078758	17.7025881814236	17.7025878950635	17.7025883856892	17.7025878168246	17.7025874840549	17.7025873981633	17.7025870736866	17.7025879651515	17.7025876438272	17.7025875194888	17.7025870352498	17.7025874997463	17.7025863738883
16.6094099343046	16.6094091374232	16.6094086464035	16.6094083618660	16.6094093819579	16.6094086866248	16.6094090749523	16.6094092871436	16.6094089669793	16.6094092582559	16.6094096529889	16.6094087416489	16.6094091431392	16.6094094744318	16.6094091207450	16.6094092490489	16.6094093237901	16.6094085328618	16.6094087786981	16.6094091401511	16.6094085076267
15.6125691558063	15.6125695887410	15.6125692393181	15.6125689136698	15.6125696445801	15.6125692066860	15.6125691794657	15.6125697577720	15.6125690614552	15.6125695496670	15.6125694721113	15.6125689415365	15.6125696326798	15.6125691339840	15.6125690781752	15.6125685853546	15.6125694469158	15.6125695832575	15.6125692668151	15.6125688736556	15.6125692784399
14.7011488237526	14.7011493536014	14.7011495076020	14.7011491586747	14.7011486426010	14.7011490252787	14.7011492651859	14.7011491594224	14.7011489314505	14.7011492350138	14.7011488265320	14.7011489997100	14.7011491332231	14.7011494213655	14.7011492880645	14.7011494571055	14.7011489410982	14.7011490520500	14.7011484414697	14.7011485993806	14.7011493095763
13.8879771758233	13.8879775746064	13.8879776223496	13.8879778528735	13.8879776008395	13.8879785472675	13.8879778555946	13.8879773225767	13.8879772772872	13.8879772037571	13.8879775777674	13.8879777756926	13.8879775662828	13.8879776467906	13.8879771392325	13.8879772311110	13.8879776369199	13.8879772681977	13.8879778498198	13.8879776623340	13.8879772281690
13.1197142901966	13.1197134102606	13.1197140742365	13.1197133477766	13.1197138045940	13.1197138726661	13.1197134808902	13.1197144055620	13.1197143361454	13.1197136259252	13.1197136426452	13.1197137598268	13.1197140949841	13.1197136717207	13.1197137042074	13.1197139384178	13.1197135684034	13.1197137801579	13.1197138359017	13.1197136131289	13.1197139025577
12.4122199134644	12.4122200167901	12.4122200013395	12.4122200560007	12.4122193791725	12.4122197078404	12.4122202201531	12.4122199280021	12.4122204861206	12.4122198820834	12.4122196964149	12.4122199814080	12.4122196459058	12.4122196660498	12.4122200688792	12.4122198702253	12.4122200721920	12.4122199168878	12.4122198474349	12.4122198399969	12.4122197763172
11.7592946212318	11.7592945199938	11.7592944127694	11.7592946688991	11.7592945021369	11.7592948465355	11.7592942804201	11.7592942757152	11.7592944723909	11.7592944646100	11.7592946916858	11.7592948999475	11.7592944747136	11.7592946625859	11.7592947858930	11.7592943896530	11.7592946291365	11.7592946806190	11.7592947497919	11.7592947420540	11.7592948726132
11.1555052007466	11.1555048458531	11.1555049786096	11.1555054628070	11.1555056162673	11.1555048366758	11.1555054054526	11.1555049812340	11.1555054468219	11.1555048359867	11.1555053965987	11.1555049657650	11.1555051010401	11.1555054486767	11.1555051893355	11.1555051121781	11.1555052483763	11.1555054428263	11.1555051764069	11.1555049348031	11.1555049349085
10.5960730023525	10.5960730977270	10.5960729296046	10.5960735883895	10.5960735665348	10.5960734602193	10.5960731029489	10.5960732453772	10.5960731328212	10.5960732056480	10.5960735639750	10.5960733002838	10.5960732830532	10.5960730602176	10.5960733935063	10.5960729381025	10.5960735355500	10.5960732161678	10.5960733342123	10.5960733023487	10.5960733970962
10.0767848484975	10.0767848124577	10.0767845808474	10.0767845689361	10.0767847726024	10.0767844980549	10.0767847693023	10.0767845174694	10.0767849854347	10.0767844347741	10.0767847163809	10.0767841729703	10.0767849219682	10.0767847165194	10.0767846822150	10.0767846284280	10.0767844017378	10.0767851558425	10.0767848998510	10.0767846643464	10.0767847043961
9.57580210653786	9.57580177769290	9.57580207884861	9.57580219023329	9.57580242065926	9.57580217385397	9.57580197400236	9.57580200409405	9.57580206434180	9.57580185524641	9.57580183931906	9.57580220164959	9.57580214963199	9.57580228912622	9.57580205734334	9.57580234865965	9.57580197331252	9.57580220858086	9.57580261058553	9.57580227771651	9.57580194932644
9.12653101700897	9.12653127846283	9.12653107451003	9.12653123856379	9.12653099674473	9.12653081294650	9.12653114051551	9.12653065831244	9.12653096607116	9.12653122879988	9.12653112044690	9.12653105021443	9.12653096685253	9.12653092385220	9.12653138686592	9.12653120924913	9.12653087581365	9.12653086134165	9.12653102140576	9.12653108863641	9.12653074364342
8.70740394181809	8.70740365397762	8.70740416409851	8.70740429478803	8.70740429853622	8.70740435270990	8.70740422646381	8.70740401010475	8.70740423145867	8.70740407865753	8.70740427768586	8.70740416227907	8.70740418932915	8.70740402116050	8.70740418453029	8.70740414499290	8.70740453726461	8.70740426106907	8.70740415759159	8.70740419386332	8.70740390300660
8.31579579579534	8.31579560332867	8.31579576094329	8.31579589255199	8.31579595422319	8.31579572056884	8.31579566480148	8.31579594705327	8.31579581220329	8.31579571610093	8.31579585474530	8.31579579523458	8.31579555202844	8.31579535200734	8.31579615238647	8.31579575348938	8.31579544922782	8.31579566702555	8.31579567693840	8.31579558268130	8.31579559063386
7.94935753010549	7.94935760822269	7.94935774808286	7.94935769115751	7.94935756522259	7.94935794827203	7.94935749124357	7.94935781901488	7.94935737681322	7.94935814498055	7.94935759954096	7.94935773458799	7.94935755029457	7.94935780710053	7.94935767285141	7.94935766193118	7.94935791802207	7.94935755300646	7.94935774043870	7.94935751066720	7.94935753732011
7.59007853314785	7.59007838489830	7.59007824214748	7.59007844502600	7.59007862642529	7.59007818806859	7.59007867210097	7.59007854375713	7.59007858891806	7.59007843382173	7.59007852219368	7.59007864661334	7.59007884706809	7.59007852025500	7.59007848261091	7.59007858192686	7.59007822294112	7.59007844388638	7.59007867502632	7.59007839857013	7.59007852415113
7.26829602299316	7.26829620713204	7.26829620068789	7.26829595757451	7.26829617481212	7.26829633448734	7.26829620293860	7.26829621882089	7.26829627393336	7.26829615563728	7.26829603121262	7.26829605511434	7.26829601750098	7.26829605804945	7.26829611498172	7.26829598054440	7.26829600735271	7.26829619221950	7.26829591445319	7.26829594863293	7.26829602759696
6.96596846208548	6.96596853769513	6.96596916693185	6.96596906228926	6.96596881919845	6.96596864933101	6.96596890664087	6.96596854792316	6.96596887418176	6.96596869327199	6.96596877449431	6.96596849681805	6.96596874607719	6.96596891754355	6.96596893892859	6.96596882676366	6.96596880294372	6.96596866820837	6.96596861776945	6.96596868144276	6.96596897389248
6.66676237875156	6.66676231540436	6.66676209797482	6.66676199658150	6.66676258037992	6.66676259650105	6.66676221020485	6.66676211208416	6.66676229428935	6.66676232710057	6.66676223229947	6.66676215884616	6.66676211401529	6.66676230776584	6.66676207044475	6.66676213682101	6.66676223782561	6.66676214736675	6.66676214178939	6.66676224484835	6.66676212708048
6.39926119775250	6.39926102507812	6.39926088883144	6.39926101388577	6.39926111662107	6.39926108747493	6.39926089698441	6.39926120338979	6.39926123436244	6.39926101922710	6.39926094982286	6.39926120327793	6.39926090637175	6.39926106479602	6.39926095809761	6.39926098646735	6.39926105781472	6.39926119882837	6.39926102278632	6.39926122932169	6.39926099941325
6.14701464572489	6.14701473068141	6.14701478135252	6.14701435475375	6.14701475178511	6.14701456140973	6.14701475405571	6.14701467318769	6.14701459673511	6.14701459702536	6.14701466422259	6.14701461839285	6.14701461816052	6.14701448866626	6.14701480421233	6.14701470480492	6.14701464428467	6.14701450022510	6.14701462366809	6.14701477177129	6.14701494242224
5.89509707082953	5.89509709690741	5.89509716621639	5.89509706509246	5.89509695197946	5.89509724546655	5.89509671732815	5.89509703325698	5.89509699885315	5.89509683676162	5.89509691456793	5.89509682630247	5.89509699431233	5.89509686955300	5.89509665274914	5.89509687627472	5.89509686921751	5.89509679272440	5.89509709898306	5.89509668196224	5.89509658056949
5.67038406299041	5.67038400173095	5.67038382713460	5.67038385455594	5.67038410577898	5.67038370630521	5.67038383660835	5.67038414057183	5.67038385104915	5.67038401018655	5.67038420443800	5.67038409091606	5.67038388165159	5.67038416497176	5.67038410132252	5.67038404266972	5.67038412584157	5.67038410571355	5.67038403489290	5.67038424819214	5.67038415521960
5.44459919654954	5.44459933792094	5.44459943733057	5.44459936200782	5.44459918745337	5.44459946961426	5.44459950628785	5.44459936168459	5.44459935909501	5.44459944310864	5.44459922943147	5.44459952875772	5.44459925266122	5.44459923133771	5.44459925709420	5.44459931680226	5.44459902524560	5.44459921126720	5.44459945409288	5.44459926049294	5.44459940545297
5.24358018207163	5.24358022413289	5.24358038927564	5.24358019664176	5.24358021672028	5.24358021789470	5.24358016248734	5.24358048082517	5.24358023582645	5.24358038282803	5.24358023062111	5.24358026275506	5.24358025362326	5.24358024217549	5.24358047029310	5.24358026585859	5.24358054110318	5.24358036854636	5.24358042155788	5.24358046547692	5.24358019414130
5.04039548910851	5.04039515618875	5.04039528002913	5.04039532372988	5.04039521658613	5.04039545678975	5.04039535301690	5.04039522831348	5.04039542027661	5.04039515112965	5.04039502392272	5.04039521086153	5.04039531459194	5.04039531959699	5.04039510752295	5.04039541269588	5.04039549430639	5.04039528598353	5.04039496486619	5.04039509775645	5.04039539623161
4.85986506216941	4.85986506764876	4.85986492676526	4.85986503440807	4.85986492379668	4.85986508577083	4.85986494440999	4.85986493400085	4.85986490564722	4.85986493932506	4.85986514629397	4.85986492896250	4.85986496034019	4.85986510264101	4.85986487381934	4.85986505977992	4.85986503594519	4.85986497377032	4.85986488040173	4.85986490591829	4.85986498030998
4.67631759492378	4.67631771095769	4.67631742753261	4.67631769296363	4.67631760749584	4.67631744584637	4.67631766142789	4.67631767061521	4.67631764713808	4.67631763386386	4.67631759658090	4.67631750520064	4.67631745840690	4.67631765324654	4.67631755167402	4.67631757497588	4.67631755441306	4.67631755622444	4.67631763642795	4.67631757384408	4.67631763633625
4.51358156821246	4.51358127981094	4.51358148786051	4.51358142357421	4.51358158649195	4.51358166707574	4.51358145429303	4.51358147150329	4.51358149268433	4.51358123711683	4.51358149497831	4.51358117294111	4.51358125246512	4.51358161707111	4.51358124610315	4.51358138222841	4.51358134867993	4.51358145325377	4.51358138666330	4.51358132763717	4.51358165610621
4.34717385449027	4.34717413554535	4.34717401067197	4.34717423714542	4.34717381359494	4.34717390976208	4.34717391703580	4.34717391905650	4.34717393031776	4.34717412538787	4.34717385456702	4.34717407424464	4.34717405794384	4.34717416915456	4.34717400170927	4.34717405124432	4.34717396606921	4.34717408404115	4.34717396036807	4.34717415876709	4.34717416496504
4.19995405531759	4.19995430131743	4.19995400058891	4.19995414867627	4.19995422010635	4.19995408935926	4.19995411614031	4.19995405524002	4.19995422010805	4.19995408492358	4.19995421362438	4.19995403577210	4.19995422239359	4.19995412441767	4.19995410165798	4.19995414669901	4.19995408135983	4.19995426812903	4.19995413828859	4.19995401145327	4.19995391419710
4.04856643785284	4.04856655236661	4.04856663980007	4.04856662731650	4.04856650427415	4.04856675421990	4.04856675531695	4.04856659656437	4.04856651459842	4.04856669010877	4.04856659276176	4.04856649019881	4.04856670111186	4.04856662295718	4.04856672878943	4.04856664089722	4.04856651376961	4.04856655807189	4.04856646942683	4.04856628504061	4.04856641680922
3.91492358014188	3.91492367868554	3.91492358721753	3.91492365712941	3.91492352911919	3.91492366637507	3.91492351999689	3.91492352012708	3.91492361952157	3.91492375510819	3.91492356022135	3.91492346306149	3.91492344696022	3.91492346777238	3.91492379267844	3.91492371443574	3.91492359933039	3.91492372020426	3.91492376768273	3.91492365648407	3.91492385577794
3.77674793045241	3.77674816687048	3.77674790804724	3.77674813868553	3.77674784237228	3.77674795219779	3.77674812970970	3.77674786538438	3.77674802176185	3.77674817505257	3.77674810443719	3.77674792842699	3.77674796177970	3.77674787223511	3.77674795544383	3.77674794184258	3.77674793705219	3.77674808176298	3.77674798398432	3.77674804068377	3.77674812499981
3.64468565941954	3.64468559863379	3.64468547260424	3.64468572638377	3.64468563556522	3.64468552637024	3.64468540727949	3.64468565539627	3.64468545712809	3.64468545666323	3.64468549940583	3.64468544690508	3.64468557728964	3.64468561755172	3.64468539803999	3.64468545303841	3.64468548609769	3.64468551553428	3.64468556984068	3.64468537287361	3.64468566108730
3.52850543234034	3.52850547425802	3.52850537236882	3.52850552396040	3.52850533669032	3.52850530694075	3.52850544517367	3.52850544728379	3.52850541002175	3.52850553623675	3.52850557533024	3.52850547197758	3.52850550323382	3.52850548684028	3.52850549042513	3.52850542155286	3.52850547872608	3.52850539552429	3.52850527096338	3.52850532549180	3.52850537655178
3.40740439029174	3.40740440411046	3.40740421348172	3.40740420175336	3.40740435627011	3.40740459062834	3.40740453505616	3.40740437054161	3.40740435312294	3.40740444086248	3.40740456911579	3.40740432784832	3.40740435532837	3.40740432492736	3.40740442972765	3.40740457106734	3.40740438193542	3.40740437565423	3.40740445120188	3.40740436591109	3.40740428518118
3.30106885162009	3.30106900902417	3.30106904568817	3.30106894551904	3.30106884428114	3.30106900342580	3.30106921186682	3.30106913432802	3.30106900606276	3.30106904036320	3.30106899591099	3.30106893523684	3.30106887644386	3.30106908371685	3.30106880341371	3.30106892561141	3.30106894592823	3.30106905284312	3.30106897203075	3.30106903193421	3.30106907892419
3.18968175627749	3.18968152761160	3.18968171988496	3.18968186924639	3.18968155556781	3.18968166698068	3.18968173491531	3.18968149450919	3.18968178384913	3.18968169000581	3.18968162035913	3.18968170757539	3.18968181223618	3.18968183263869	3.18968178243213	3.18968162064062	3.18968170405366	3.18968163653244	3.18968170352151	3.18968177894578	3.18968166398009
3.08283720229756	3.08283731132097	3.08283716755845	3.08283722818928	3.08283742769120	3.08283702999260	3.08283712984567	3.08283714681743	3.08283721961116	3.08283717942028	3.08283714703966	3.08283723053208	3.08283717231709	3.08283726485856	3.08283704938087	3.08283711802897	3.08283718922072	3.08283716543165	3.08283721936001	3.08283721951605	3.08283721885239
2.98927593376521	2.98927607434378	2.98927595252618	2.98927596913917	2.98927602602051	2.98927602588352	2.98927594276139	2.98927592298373	2.98927592093245	2.98927600776868	2.98927595305029	2.98927588933350	2.98927600941708	2.98927584600491	2.98927598539872	2.98927619556781	2.98927601217798	2.98927608985906	2.98927598872429	2.98927595196731	2.98927587807046
2.89056487556962	2.89056495977991	2.89056498050264	2.89056479092587	2.89056481210901	2.89056496547422	2.89056501234071	2.89056491313078	2.89056484029517	2.89056498271806	2.89056487059457	2.89056496402240	2.89056495648712	2.89056483567787	2.89056486963948	2.89056493636117	2.89056480504307	2.89056490986859	2.89056490603846	2.89056496790051	2.89056479042170
2.79568499423713	2.79568488408678	2.79568516540519	2.79568509790870	2.79568502361389	2.79568499177948	2.79568497388200	2.79568495646046	2.79568490930421	2.79568502630215	2.79568504638412	2.79568494681874	2.79568497400155	2.79568500962857	2.79568507578919	2.79568500521808	2.79568491539748	2.79568483180146	2.79568491635022	2.79568506169386	2.79568494578415
2.71275231420685	2.71275215954743	2.71275228957094	2.71275226319174	2.71275232267773	2.71275206862145	2.71275227165967	2.71275229259365	2.71275240154828	2.71275217605493	2.71275225769307	2.71275228395812	2.71275214088233	2.71275228685085	2.71275218105332	2.71275233845348	2.71275237107881	2.71275228686728	2.71275224249836	2.71275215414941	2.71275225876944
2.62470289596938	2.62470277083689	2.62470275146604	2.62470280719279	2.62470280051280	2.62470267120321	2.62470280990330	2.62470281095578	2.62470293459673	2.62470281430710	2.62470275027922	2.62470277006525	2.62470280790863	2.62470280536642	2.62470277482460	2.62470286261508	2.62470293610440	2.62470275078710	2.62470279946460	2.62470273120590	2.62470282769245
2.54776750137464	2.54776760172929	2.54776753137591	2.54776746946673	2.54776768420988	2.54776753352514	2.54776763729556	2.54776743133924	2.54776757367982	2.54776768380283	2.54776756838299	2.54776742142053	2.54776749136778	2.54776748658835	2.54776751646602	2.54776743422649	2.54776758345077	2.54776750081644	2.54776743139822	2.54776752009853	2.54776742709346
2.46580056882397	2.46580053698677	2.46580053562318	2.46580056327494	2.46580055650644	2.46580059354166	2.46580053601062	2.46580055380001	2.46580045463969	2.46580052292875	2.46580051792229	2.46580054694681	2.46580060877320	2.46580057118710	2.46580056059431	2.46580053842977	2.46580050991748	2.46580057758631	2.46580052676838	2.46580049788162	2.46580055293530
2.38672818265641	2.38672811789624	2.38672823352228	2.38672827858281	2.38672831217483	2.38672815438194	2.38672801422818	2.38672818430345	2.38672820775482	2.38672819595525	2.38672803308458	2.38672817596974	2.38672822536104	2.38672817055143	2.38672816152206	2.38672806320560	2.38672800676756	2.38672819883843	2.38672820233155	2.38672814030827	2.38672806006918
2.31039789144938	2.31039780542620	2.31039784805332	2.31039786344861	2.31039782211321	2.31039786902065	2.31039790221940	2.31039780028364	2.31039795419608	2.31039787699907	2.31039787891672	2.31039792968102	2.31039781831123	2.31039786363685	2.31039783940145	2.31039786195931	2.31039774999406	2.31039786094261	2.31039783412522	2.31039777221789	2.31039787108032
2.24361302955812	2.24361314090897	2.24361289393521	2.24361313080149	2.24361309263196	2.24361305679429	2.24361301331753	2.24361298748334	2.24361299449780	2.24361313362773	2.24361307006336	2.24361306219048	2.24361300571526	2.24361296863053	2.24361307961506	2.24361305382111	2.24361314546623	2.24361314921977	2.24361300664897	2.24361312630667	2.24361305232807
2.17207545699973	2.17207541048036	2.17207537343960	2.17207538594786	2.17207535347726	2.17207539316913	2.17207532297875	2.17207530874596	2.17207535740398	2.17207549099158	2.17207537106401	2.17207544850695	2.17207542718725	2.17207550267586	2.17207546384442	2.17207532046440	2.17207539248225	2.17207528860677	2.17207532143254	2.17207544610192	2.17207543829829
2.10284621050069	2.10284637803569	2.10284633874755	2.10284628583059	2.10284638016493	2.10284635945423	2.10284627419748	2.10284626237671	2.10284641792472	2.10284630410045	2.10284617287915	2.10284624699592	2.10284620672144	2.10284646795501	2.10284615246598	2.10284613215238	2.10284630653992	2.10284622322311	2.10284632175599	2.10284630964364	2.10284633431784
2.04201178348743	2.04201185408260	2.04201163991752	2.04201181820043	2.04201176070614	2.04201174770548	2.04201170964828	2.04201186041396	2.04201176319090	2.04201181854603	2.04201185775201	2.04201176367323	2.04201164983706	2.04201177410107	2.04201175785308	2.04201179782373	2.04201166316726	2.04201169636431	2.04201168350412	2.04201159315839	2.04201174599651
1.97672606226091	1.97672600409074	1.97672604189849	1.97672604365451	1.97672611643476	1.97672598066314	1.97672603400132	1.97672611661273	1.97672607356923	1.97672598474792	1.97672587901474	1.97672608830095	1.97672618714880	1.97672605874228	1.97672606698463	1.97672595213562	1.97672606067495	1.97672603489276	1.97672599430531	1.97672595810593	1.97672595702502
1.91335055345399	1.91335057557598	1.91335061945264	1.91335052922325	1.91335058764294	1.91335066864652	1.91335047722704	1.91335044096783	1.91335052350079	1.91335041088722	1.91335056053209	1.91335057906483	1.91335040089137	1.91335052424090	1.91335056622660	1.91335049833262	1.91335056008623	1.91335047657681	1.91335056876530	1.91335056219984	1.91335061667939
1.85718071114262	1.85718068982689	1.85718057179628	1.85718075146821	1.85718064975042	1.85718061222242	1.85718068364062	1.85718059205577	1.85718060743905	1.85718068681451	1.85718062659883	1.85718074057921	1.85718061223493	1.85718061283794	1.85718073621676	1.85718056593801	1.85718066563535	1.85718066785216	1.85718060190450	1.85718061667643	1.85718087265297
1.79692246060034	1.79692243503013	1.79692262802028	1.79692259960641	1.79692241011622	1.79692245453550	1.79692256578234	1.79692240915133	1.79692249063857	1.79692238444543	1.79692260675127	1.79692249267163	1.79692253315243	1.79692259498900	1.79692252333586	1.79692245885981	1.79692244592248	1.79692249583190	1.79692245168708	1.79692246828220	1.79692249054032
1.73817058326830	1.73817066549744	1.73817058320453	1.73817055987816	1.73817066821887	1.73817067944326	1.73817070086636	1.73817079906357	1.73817068357036	1.73817071244022	1.73817065913546	1.73817075099642	1.73817074608701	1.73817077484736	1.73817066675551	1.73817069406300	1.73817071730344	1.73817058164344	1.73817071123461	1.73817064344455	1.73817066517493
1.68529644291248	1.68529644546800	1.68529649937399	1.68529647771164	1.68529644312830	1.68529641390670	1.68529644221776	1.68529628722136	1.68529634969365	1.68529639852144	1.68529649516616	1.68529638954435	1.68529646666608	1.68529640443771	1.68529642791724	1.68529644262148	1.68529638735216	1.68529643566968	1.68529637651487	1.68529648078376	1.68529647156396
1.62872498945775	1.62872493802404	1.62872496211132	1.62872493036047	1.62872504396535	1.62872493074808	1.62872494328709	1.62872490957354	1.62872485797893	1.62872496960742	1.62872488553491	1.62872490465808	1.62872486914167	1.62872491624565	1.62872487093774	1.62872493833396	1.62872492790453	1.62872486895937	1.62872492993259	1.62872492912926	1.62872492957284
1.57315328624892	1.57315314917273	1.57315306689316	1.57315317076920	1.57315314833246	1.57315328396823	1.57315310955769	1.57315315193455	1.57315320153016	1.57315325153394	1.57315313031447	1.57315307746577	1.57315326946263	1.57315323215518	1.57315316862634	1.57315320143097	1.57315323351065	1.57315325069445	1.57315322745344	1.57315312585537	1.57315316734241
1.51836966408519	1.51836959068204	1.51836954670817	1.51836951645617	1.51836964810260	1.51836954860441	1.51836955976664	1.51836963151144	1.51836951371350	1.51836960555443	1.51836969261541	1.51836959341366	1.51836956859711	1.51836959226399	1.51836957589495	1.51836961622524	1.51836962403220	1.51836963342036	1.51836957837695	1.51836958834620	1.51836965412317
1.46692036515195	1.46692034240120	1.46692034021782	1.46692033608347	1.46692038882172	1.46692047087721	1.46692034569766	1.46692039127084	1.46692032105726	1.46692036478975	1.46692036216607	1.46692029713330	1.46692026944425	1.46692035062549	1.46692029147806	1.46692024674797	1.46692031049485	1.46692020515023	1.46692032649301	1.46692033531931	1.46692029591537
1.41210135835660	1.41210141444735	1.41210135101268	1.41210132721230	1.41210143962537	1.41210126628193	1.41210114862434	1.41210137176725	1.41210129731689	1.41210136260958	1.41210133091224	1.41210142424057	1.41210136138562	1.41210135910707	1.41210142172280	1.41210128699702	1.41210128764056	1.41210132619167	1.41210137256488	1.41210131634035	1.41210128979153
1.35663189400456	1.35663192357127	1.35663181301670	1.35663193503084	1.35663192959606	1.35663186097568	1.35663192804913	1.35663196828707	1.35663186747499	1.35663198863672	1.35663191063810	1.35663197114885	1.35663190566693	1.35663186133949	1.35663191402152	1.35663189512275	1.35663184215486	1.35663188297458	1.35663186038277	1.35663189804675	1.35663192714444
1.29996728871934	1.29996737073434	1.29996738960052	1.29996732000888	1.29996736872588	1.29996743971342	1.29996734203549	1.29996748898010	1.29996740034454	1.29996727669590	1.29996738698022	1.29996739978866	1.29996735077287	1.29996741327333	1.29996739217552	1.29996728749591	1.29996728196374	1.29996727154128	1.29996739758174	1.29996744290993	1.29996732353509
1.23553877357658	1.23553863980228	1.23553868599629	1.23553862511969	1.23553866163280	1.23553871301035	1.23553875868794	1.23553877379306	1.23553876793519	1.23553876802699	1.23553877814419	1.23553871271465	1.23553863035595	1.23553877401116	1.23553868964194	1.23553873137954	1.23553868826051	1.23553875084517	1.23553871047824	1.23553873060776	1.23553876108331];

            app.Dxs_1550 = [7.92164293731733	7.92169130065786	7.92183603270299	7.92207324050812	7.92239542096954	7.92279403312845	7.92325781020459	7.92377874199305	7.92434365884596	7.92493916474902	7.92555877485566	7.92619110130109	7.92682692228082	7.92745949492215	7.92808360178500	7.92869451980326	7.92928654938914	7.92986034541711	7.93040930294826	7.93093749100203	7.93143929569937
7.22491722671600	7.22497007358559	7.22512789678619	7.22538614119880	7.22573614002177	7.22616874500966	7.22667534207850	7.22724106024706	7.22785502865204	7.22850348249077	7.22917691494529	7.22986405500002	7.23055676230691	7.23124555357658	7.23192448170742	7.23258835207193	7.23323420266124	7.23385627659459	7.23445597897394	7.23502863117006	7.23557650785632
6.53811392672546	6.53817468033256	6.53835467915766	6.53864871070185	6.53904966705552	6.53954379446390	6.54012122554700	6.54076745172051	6.54146764512753	6.54220898423956	6.54297774954637	6.54376240651215	6.54455227340314	6.54533852985577	6.54611408357611	6.54687235423431	6.54760874905730	6.54832075743669	6.54900381387964	6.54965812154048	6.55028304105849
6.05027541091157	6.05034028528841	6.05053341634124	6.05085019880455	6.05128041764097	6.05181297617048	6.05243452515364	6.05312929566543	6.05388303845384	6.05467916939394	6.05550693386791	6.05635061029812	6.05720055111335	6.05804676989274	6.05888081265915	6.05969688792961	6.06048938299242	6.06125439508660	6.06199043923468	6.06269441981498	6.06336614642345
5.62959279532886	5.62966254420690	5.62986926420785	5.63020863396511	5.63066919436520	5.63123885979816	5.63190384463167	5.63264763062171	5.63345431773494	5.63430855670636	5.63519360399740	5.63609713274355	5.63700743585489	5.63791310027752	5.63880672299024	5.63968083934326	5.64052904516732	5.64134870575876	5.64213671633632	5.64289041563667	5.64361076628996
5.26325023379121	5.26332452828153	5.26354615686120	5.26390700480571	5.26439796762438	5.26500566865976	5.26571420750206	5.26650715304896	5.26736740290718	5.26827709925580	5.26922136121975	5.27018532056188	5.27115589447594	5.27212199739649	5.27307448372294	5.27400618044206	5.27491137720081	5.27578602625165	5.27662616012222	5.27743084460188	5.27819827242299
4.94137095304641	4.94144979847413	4.94168455197352	4.94206814446012	4.94259009274653	4.94323518104754	4.94398874889450	4.94483098301080	4.94574498429294	4.94671200374364	4.94771538698005	4.94873961326574	4.94977045934019	4.95079720318988	4.95180990242008	4.95279985322009	4.95376186831777	4.95469157904082	4.95558434357807	4.95643944018987	4.95725580314439
4.65642566131587	4.65650931092533	4.65675821157058	4.65716384892146	4.65771633017243	4.65839990700848	4.65919733170738	4.66008987746999	4.66105767145959	4.66208155020098	4.66314396817472	4.66422920602226	4.66532148173741	4.66640911340173	4.66748160906348	4.66853110330785	4.66955014703582	4.67053472409058	4.67148092644367	4.67238763563916	4.67325224703267
4.40244168908261	4.40252978233371	4.40279284963566	4.40322100624031	4.40380412588304	4.40452651721074	4.40536847401584	4.40631058482020	4.40733282062584	4.40841458036246	4.40953660637207	4.41068276061446	4.41183645426471	4.41298487612960	4.41411832509155	4.41522652284828	4.41630336044200	4.41734355914800	4.41834390311289	4.41930180047162	4.42021523822891
4.17466938771629	4.17476263619358	4.17503924962484	4.17549038422423	4.17610505383406	4.17686498103276	4.17775202992991	4.17874480119618	4.17982112063357	4.18096033506535	4.18214335416896	4.18334969400495	4.18456548432655	4.18577596611663	4.18696996928723	4.18813786612809	4.18927256595756	4.19036890798594	4.19142284499872	4.19243246465038	4.19339536261967
3.96927272562748	3.96937034009420	3.96966147536313	3.97013552773490	3.97078101322072	3.97157988078139	3.97251195755518	3.97355475060213	3.97468591704079	3.97588317433845	3.97712613068289	3.97839458089138	3.97967259055618	3.98094462349930	3.98219987724234	3.98342797684115	3.98462049092083	3.98577398384410	3.98688194853565	3.98794331467138	3.98895587043600
3.78312445602621	3.78322674814973	3.78353091957654	3.78402899111748	3.78470517875467	3.78554267403310	3.78651970901820	3.78761353128859	3.78879962919066	3.79005496258020	3.79135846133399	3.79268943335968	3.79402924587559	3.79536418420693	3.79668056139060	3.79796901138753	3.79922058480172	3.80042978263624	3.80159285683726	3.80270606642899	3.80376911264188
3.61365521364651	3.61376245117079	3.61408116477506	3.61460176710429	3.61530938124934	3.61618606192480	3.61720844987251	3.61835285408427	3.61959461732244	3.62090861986848	3.62227262300075	3.62366487973464	3.62506839514739	3.62646531197617	3.62784339021055	3.62919276082467	3.63051576063251	3.63178161108459	3.63299939225211	3.63416579730208	3.63527902278308
3.45873915177465	3.45885083764646	3.45918345707714	3.45972691237884	3.46046690975982	3.46138243258385	3.46245038268501	3.46364599989617	3.46494308618490	3.46631609566664	3.46774142356476	3.46919691604730	3.47066305070515	3.47212325206946	3.47356370744366	3.47497338632411	3.47634351615715	3.47766760511886	3.47894070573639	3.48016021020417	3.48132384614772
3.31656714527218	3.31668383495041	3.31703068727036	3.31759802453603	3.31836873354435	3.31932314227974	3.32043748262524	3.32168409989702	3.32303663252482	3.32446880051180	3.32595553259653	3.32747368021779	3.32900349312397	3.33052697247634	3.33202986111530	3.33350111365097	3.33493124378976	3.33631232228182	3.33764137766541	3.33891374761885	3.34012835339448
3.14639889161319	3.14652578926353	3.14690381608627	3.14752094733382	3.14836022072804	3.14939949780917	3.15061179726462	3.15196873837781	3.15344117769749	3.15499968721393	3.15661730743439	3.15826938299914	3.15993355091523	3.16159117325937	3.16322608205933	3.16482626825217	3.16638114649026	3.16788403700150	3.16932879851365	3.17071287951915	3.17203377156938
3.02725783589206	3.02739015051921	3.02778220990066	3.02842392321863	3.02929689959149	3.03037679615209	3.03163722366400	3.03304809044250	3.03457885519175	3.03619954643998	3.03788156470982	3.03959964758321	3.04133000772601	3.04305418982073	3.04475491623250	3.04641943685712	3.04803703584987	3.04959994987193	3.05110340243139	3.05254341261240	3.05391796160766
2.91683995939767	2.91697707514457	2.91738453914826	2.91805070481066	2.91895653585209	2.92007812419016	2.92138659971938	2.92285131606408	2.92444116569871	2.92612390003107	2.92787084520375	2.92965497097584	2.93145275079640	2.93324339720899	2.93501013586449	2.93673904928037	2.93841990619615	2.94004408738614	2.94160619136043	2.94310270486710	2.94453070702905
2.81422335707715	2.81436570765938	2.81478801727764	2.81547891223148	2.81641813964476	2.81758088393978	2.81893791266999	2.82045727957430	2.82210544488469	2.82385118494828	2.82566297057716	2.82751379184779	2.82937843423402	2.83123643365448	2.83306954560848	2.83486366068216	2.83660768630906	2.83829337830738	2.83991453084900	2.84146759576942	2.84295004574138
2.71861261070684	2.71875969886188	2.71919754553726	2.71991281440059	2.72088564764075	2.72209025712342	2.72349571719227	2.72506943138666	2.72677693911949	2.72858532728708	2.73046305389641	2.73238069850866	2.73431300230029	2.73623811379345	2.73813805543248	2.73999766059215	2.74180549306780	2.74355290426689	2.74523373782368	2.74684394608314	2.74838099423777
2.62931559751086	2.62946813041528	2.62992080803772	2.63066068662710	2.63166741341464	2.63291345482818	2.63436799883092	2.63599637971565	2.63776375473604	2.63963514692724	2.64157849576573	2.64356328984093	2.64556383832073	2.64755706231937	2.64952388915142	2.65144950857792	2.65332169991610	2.65513130785470	2.65687185018881	2.65853948636817	2.66013187152810
2.54572831947511	2.54588576368814	2.54635373340670	2.54711843839075	2.54815909371722	2.54944716174599	2.55095062236359	2.55263406571944	2.55446111949013	2.55639612285359	2.55840541998806	2.56045819618573	2.56252721250562	2.56458851071821	2.56662319086048	2.56861517693963	2.57055181590985	2.57242401149847	2.57422487476179	2.57595048826849	2.57759826746849
2.46732244796931	2.46748492235194	2.46796821632030	2.46875803219074	2.46983237707496	2.47116242484879	2.47271545892563	2.47445422427277	2.47634132854630	2.47834017616077	2.48041605066303	2.48253687957392	2.48467454753909	2.48680467184230	2.48890719306044	2.49096619603359	2.49296774000437	2.49490303266154	2.49676499958528	2.49854901211034	2.50025266127551
2.39363110956467	2.39379877696435	2.39429726778708	2.39511222690427	2.39622062146610	2.39759333569684	2.39919545957992	2.40098999988774	2.40293767355770	2.40500067859964	2.40714327333095	2.40933239557758	2.41153948709393	2.41373887497901	2.41591012718869	2.41803602657406	2.42010328147770	2.42210222528313	2.42402542301755	2.42586833922061	2.42762816799257
2.32424269099003	2.32441566487445	2.32492963362815	2.32576954227994	2.32691223827790	2.32832751216212	2.32997956516754	2.33182964240742	2.33383818300970	2.33596572382672	2.33817535689168	2.34043367055085	2.34271031210216	2.34497960795282	2.34721981827887	2.34941316363852	2.35154671094413	2.35360958542845	2.35559477157263	2.35749709536874	2.35931367193744
2.25879244305905	2.25897037680980	2.25949981088123	2.26036522204822	2.26154247645768	2.26300055373830	2.26470229590549	2.26660869678692	2.26867828716478	2.27087083400827	2.27314848031289	2.27547595921276	2.27782286330855	2.28016193807777	2.28247178116065	2.28473356579768	2.28693352564288	2.28906094733556	2.29110837760596	2.29307043831824	2.29494473522792
2.19695427239367	2.19713798237489	2.19768267127216	2.19857302826255	2.19978489327986	2.20128605385619	2.20303816215681	2.20500123363084	2.20713208455066	2.20939002935593	2.21173585794488	2.21413309405405	2.21655056980462	2.21896045755624	2.22134008816273	2.22367073404534	2.22593793364817	2.22813045908135	2.23024050272821	2.23226321955470	2.23419525043000
2.13843924338021	2.13862783192702	2.13918823045124	2.14010457722609	2.14135112167748	2.14289533289353	2.14469826452373	2.14671792466677	2.14891073113356	2.15123449468081	2.15364860637342	2.15611616303742	2.15860471148039	2.16108604370232	2.16353597325724	2.16593612809127	2.16827074402755	2.17052915785049	2.17270271434453	2.17478620920787	2.17677669124755
2.08298526679662	2.08317925512783	2.08375510109336	2.08469710089396	2.08597894993587	2.08756645474580	2.08941997635062	2.09149675086241	2.09375210845047	2.09614161200490	2.09862487893949	2.10116358621504	2.10372351225808	2.10627657581776	2.10879763191354	2.11126765556831	2.11367073661210	2.11599511690267	2.11823288537120	2.12037797200482	2.12242734448328
2.03036029950385	2.03055947960182	2.03115156898064	2.03211905637089	2.03343597925161	2.03506733335039	2.03697206420962	2.03910628630606	2.04142391804865	2.04388061145846	2.04643304368722	2.04904275845849	2.05167516445733	2.05430037168914	2.05689335315912	2.05943367367202	2.06190575254794	2.06429707498335	2.06659909630499	2.06880623338547	2.07091524288709
1.98035241631549	1.98055684985560	1.98116479906297	1.98215828035847	1.98351054727167	1.98518577905323	1.98714194983753	1.98933365685091	1.99171480358216	1.99423818501464	1.99686089775007	1.99954235253067	2.00224755016632	2.00494562578734	2.00761091966473	2.01022243112464	2.01276373012592	2.01522269987429	2.01759008835039	2.01985987643254	2.02202906223159
1.93277180436243	1.93298185022145	1.93360502237342	1.93462500158588	1.93601301326220	1.93773219264301	1.93974023538516	1.94199047595233	1.94443459050247	1.94702580968471	1.94971876583454	1.95247301255403	1.95525137450534	1.95802312081692	1.96076093890121	1.96344437151555	1.96605591696407	1.96858310900894	1.97101609252426	1.97334935971299	1.97557935085688
1.88744644744163	1.88766181693770	1.88830138479064	1.88934735866878	1.89077110702102	1.89253475671901	1.89459491525571	1.89690376858372	1.89941184511240	1.90207082293447	1.90483486886612	1.90766223661021	1.91051452052924	1.91336040044410	1.91617202922011	1.91892779316715	1.92161001165520	1.92420602977356	1.92670566248678	1.92910310415339	1.93139425673945
1.84421923203159	1.84443989155880	1.84509570307837	1.84616807764930	1.84762765066074	1.84943636190967	1.85154881436700	1.85391652528476	1.85648890890736	1.85921667268983	1.86205232156923	1.86495294001084	1.86788012917503	1.87080045006237	1.87368647071507	1.87651550957236	1.87926925990815	1.88193455083788	1.88450139331075	1.88696361687562	1.88931715286211
1.80294831842489	1.80317449034762	1.80384651083804	1.80494554808984	1.80644130103906	1.80829494444260	1.81046024039649	1.81288740987071	1.81552459552085	1.81832133322066	1.82122921886476	1.82420396451632	1.82720638740074	1.83020236484666	1.83316335947687	1.83606603771926	1.83889207737868	1.84162765756508	1.84426269060564	1.84679011308515	1.84920649764081
1.76350363683138	1.76373533980939	1.76442357200294	1.76554917926897	1.76708177150407	1.76898037123956	1.77119908234551	1.77368585840092	1.77638863336398	1.77925489159079	1.78223551505206	1.78528530487331	1.78836374737916	1.79143558498735	1.79447234230194	1.79744960045133	1.80034860236860	1.80315523591094	1.80585879886138	1.80845272290487	1.81093251026010
1.72576590815639	1.72600303785879	1.72670800328936	1.72786045749444	1.72942968613345	1.73137423440594	1.73364616466450	1.73619354368099	1.73896186395406	1.74189830818524	1.74495251330042	1.74807757021926	1.75123258086059	1.75438157001880	1.75749459742429	1.76054732778580	1.76352008675806	1.76639850337893	1.76917170316711	1.77183248187813	1.77437659788440
1.68962758642152	1.68987043799272	1.69059158155107	1.69177143316170	1.69337734086991	1.69536791628507	1.69769399143903	1.70030207150128	1.70313675616302	1.70614410160613	1.70927205546141	1.71247339254421	1.71570576745362	1.71893244123958	1.72212281546506	1.72525155186043	1.72829914692350	1.73125007190709	1.73409368222137	1.73682214309873	1.73943160063647
1.65498774094298	1.65523611157508	1.65597396364588	1.65718104681701	1.65882446371115	1.66086122372958	1.66324160261801	1.66591078247316	1.66881255083100	1.67189123435049	1.67509383589073	1.67837213456819	1.68168271359789	1.68498791677843	1.68825614607602	1.69146210480163	1.69458502811377	1.69760937570531	1.70052395336881	1.70332148265272	1.70599670942184
1.62175596329160	1.62200976523376	1.62276464255888	1.62399902642634	1.62567989681674	1.62776344874724	1.63019864562579	1.63292972194857	1.63589876314296	1.63904936672290	1.64232760509567	1.64568336970397	1.64907289936904	1.65245727306685	1.65580480105925	1.65908833693569	1.66228767086636	1.66538643803257	1.66837316235674	1.67123994813751	1.67398222725012
1.58984656866552	1.59010625212870	1.59087774298138	1.59213996218705	1.59385857357034	1.59598912591608	1.59847960350635	1.60127267973742	1.60431001706099	1.60753344214268	1.61088774310740	1.61432204016030	1.61779119943286	1.62125584616958	1.62468282419680	1.62804547238782	1.63132194203032	1.63449601410644	1.63755589638587	1.64049314900192	1.64330320853988
1.55918247260181	1.55944774303856	1.56023636107072	1.56152630461747	1.56328318567376	1.56546130928432	1.56800720263628	1.57086319729739	1.57396915571426	1.57726595998877	1.58069717120961	1.58421078912875	1.58776040140829	1.59130604207027	1.59481396742992	1.59825634432349	1.60161109699513	1.60486132672940	1.60799506912813	1.61100385623796	1.61388250315942
1.52969115231795	1.52996222174358	1.53076817073392	1.53208636352313	1.53388138156724	1.53610723743076	1.53870944765162	1.54162878845888	1.54480423083992	1.54817511905724	1.55168401827848	1.55527735418995	1.55890887166419	1.56253642346555	1.56612611706384	1.56964904157639	1.57308293677372	1.57641055513228	1.57961931133512	1.58270037771438	1.58564868027578
1.50130679793076	1.50158374811504	1.50240666819057	1.50375331351578	1.50558713655037	1.50786128464014	1.51052016173966	1.51350350955934	1.51674883202181	1.52019450199147	1.52378170290554	1.52745642027452	1.53116999892523	1.53488075989119	1.53855288845698	1.54215776504723	1.54567177380053	1.54907774575412	1.55236241093058	1.55551717502417	1.55853607341532
1.47396740701398	1.47425035109178	1.47509075196706	1.47646588182464	1.47833876314444	1.48066136455809	1.48337748868013	1.48642548924229	1.48974166396668	1.49326286659916	1.49692937039025	1.50068576120898	1.50448278248446	1.50827759188951	1.51203340207905	1.51572103969566	1.51931649192238	1.52280174447091	1.52616345453864	1.52939263794114	1.53248347786695
1.44761483745015	1.44790353541745	1.44876150101866	1.45016551186464	1.45207786644220	1.45444946237100	1.45722344931112	1.46033648216261	1.46372406446408	1.46732153539001	1.47106840442942	1.47490772783124	1.47878895372559	1.48266862464502	1.48650940051182	1.49028091404870	1.49395905338633	1.49752483779217	1.50096471857857	1.50426959967259	1.50743323311111
1.42219694969738	1.42249165775799	1.42336754800351	1.42480023650033	1.42675264070491	1.42917384469873	1.43200589950624	1.43518480418628	1.43864435898680	1.44231953655084	1.44614717941645	1.45007021899788	1.45403701257859	1.45800283526628	1.46192948811724	1.46578612027855	1.46954761461606	1.47319522649694	1.47671460264662	1.48009637161124	1.48333395871371
1.39766369732305	1.39796425273290	1.39885801998638	1.40032044805931	1.40231270624208	1.40478396592656	1.40767505697850	1.41092042930938	1.41445300375445	1.41820593188203	1.42211588418575	1.42612379874503	1.43017684540939	1.43422993615346	1.43824383126662	1.44218674170676	1.44603322038046	1.44976357911142	1.45336353104454	1.45682328727221	1.46013646342052
1.37396844788728	1.37427511136868	1.37518697448288	1.37667890540145	1.37871157938739	1.38123346603560	1.38418390859000	1.38749664437463	1.39110309155423	1.39493497767967	1.39892785588632	1.40302168177237	1.40716250011963	1.41130385770100	1.41540606750109	1.41943668278871	1.42336912960599	1.42718376611003	1.43086569108395	1.43440464874912	1.43779420389827
1.35106849441138	1.35138133660706	1.35231134975699	1.35383324335392	1.35590682899284	1.35847977494389	1.36149023528093	1.36487116229373	1.36855188614287	1.37246392537878	1.37654086850394	1.38072158736099	1.38495137303807	1.38918239514174	1.39337423695034	1.39749332473849	1.40151331863077	1.40541383740304	1.40917879170227	1.41279848854398	1.41626590413251
1.32892350581207	1.32924252002998	1.33019093141300	1.33174289566590	1.33385797538733	1.33648241982994	1.33955388382803	1.34300331771632	1.34675968731684	1.35075272858557	1.35491465567514	1.35918350214183	1.36350304684293	1.36782508671672	1.37210788694819	1.37631730758053	1.38042625668975	1.38441344823854	1.38826330870218	1.39196492542858	1.39551167692914
1.30749515705601	1.30782043927119	1.30878738093752	1.31037011473043	1.31252688618632	1.31520364768665	1.31833629048350	1.32185534025552	1.32568815620618	1.32976311373546	1.33401130228678	1.33836958477589	1.34278060918901	1.34719472055491	1.35156981559789	1.35587090922055	1.36007017634181	1.36414582157622	1.36808181624487	1.37186705757872	1.37549420865689
1.28674834725902	1.28708011121158	1.28806575854386	1.28967940337014	1.29187832661219	1.29460772251132	1.29780267095638	1.30139217804426	1.30530234970939	1.30946013807582	1.31379598953807	1.31824454886962	1.32274831626643	1.32725621738360	1.33172516133036	1.33611935454077	1.34041041620791	1.34457592701673	1.34859951817136	1.35246987260812	1.35617932987439
1.26664986434025	1.26698767314668	1.26799259113333	1.26963720426346	1.27187917228727	1.27466185124268	1.27791972354549	1.28158046164199	1.28556898172603	1.28981114561891	1.29423542692300	1.29877609731646	1.30337376168104	1.30797690941396	1.31254104303481	1.31702993145927	1.32141441719318	1.32567155222257	1.32978447987913	1.33374131704784	1.33753438673493
1.24716814532412	1.24751254678290	1.24853658684154	1.25021274610878	1.25249782535669	1.25533459244288	1.25865613830247	1.26238906319245	1.26645702967858	1.27078445016239	1.27529868717829	1.27993251018778	1.28462576170874	1.28932551633707	1.29398657347629	1.29857180198620	1.30305130552903	1.30740170950171	1.31160569831208	1.31565073394404	1.31952944705675
1.22827375161118	1.22862471854463	1.22966827162305	1.23137635622429	1.23370525382427	1.23659656805286	1.23998265158723	1.24378874279688	1.24793720873449	1.25235111413407	1.25695654160648	1.26168527635512	1.26647554445938	1.27127357021737	1.27603335724367	1.28071658378718	1.28529268123262	1.28973812942653	1.29403478725509	1.29816992869582	1.30213573900414
1.20993926264676	1.21029679413893	1.21135995801265	1.21310048674948	1.21547365140207	1.21842045677696	1.22187198186922	1.22575211108657	1.22998233493873	1.23448397466746	1.23918213805044	1.24400691509775	1.24889586668137	1.25379381907615	1.25865360409024	1.26343688760388	1.26811166655386	1.27265397599483	1.27704508096140	1.28127224388930	1.28532706277419
1.19213843318897	1.19250272671451	1.19358593450668	1.19535904405915	1.19777696333528	1.20078004663829	1.20429779055295	1.20825312439982	1.21256610177051	1.21715694768996	1.22194915707285	1.22687199124182	1.23186124129591	1.23686076063490	1.24182285214951	1.24670778389790	1.25148309834846	1.25612412164104	1.26061182586422	1.26493287122758	1.26907852126652
1.17484619367759	1.17521729875062	1.17632050016859	1.17812691504438	1.18059041543757	1.18365033323679	1.18723516046415	1.19126684834328	1.19566399557121	1.20034536538565	1.20523327873547	1.21025546300761	1.21534675497864	1.22044993663240	1.22551610979064	1.23050467987519	1.23538259522338	1.24012440038095	1.24471064091094	1.24912748908811	1.25336613502951
1.15803936747143	1.15841723618059	1.15954099413646	1.16138108420678	1.16389064067581	1.16700814620155	1.17066136993174	1.17477054843735	1.17925304037783	1.18402644350312	1.18901166471400	1.19413495827991	1.19933025169260	1.20453887501531	1.20971110304463	1.21480540981041	1.21978805916853	1.22463282176808	1.22931975448750	1.23383474582609	1.23816830715911
1.14169526481963	1.14208022439653	1.14322467145224	1.14509876979809	1.14765516744843	1.15083124590213	1.15455362291700	1.15874158325784	1.16331095679720	1.16817793528162	1.17326203941915	1.17848845073740	1.18378951741833	1.18910583847921	1.19438620656999	1.19958840823227	1.20467807854773	1.20962787728380	1.21441785937653	1.21903299642122	1.22346408792660
1.12579334108664	1.12618521498126	1.12735072972944	1.12925941518609	1.13186340451010	1.13509895075378	1.13889175039106	1.14315960971398	1.14781713443960	1.15277942763487	1.15796433674911	1.16329558422933	1.16870469333815	1.17413067045575	1.17952145504218	1.18483395130872	1.19003273747061	1.19509017493210	1.19998560958729	1.20470344678214	1.20923414714134
1.11031307246335	1.11071217031010	1.11189904798281	1.11384293022182	1.11649503676661	1.11979100340073	1.12365528461871	1.12800444460351	1.13275186051338	1.13781099460744	1.14309865161964	1.14853703193858	1.15405608679759	1.15959409733869	1.16509764952288	1.17052284619206	1.17583330357515	1.18100099549045	1.18600416578140	1.19082734970816	1.18615718380648
1.09523531986650	1.09564169993464	1.09685032472733	1.09882999742950	1.10153098022544	1.10488843259026	1.10882543586215	1.11325728900797	1.11809622697884	1.12325407838252	1.12864635943621	1.13419382092153	1.13982532914313	1.14547757112736	1.15109661773528	1.15663703001767	1.16206182757249	1.16734229036111	1.17245589030109	1.17738699992941	1.17303061822434
1.08054204772785	1.08095585970689	1.08218645265838	1.08420226234185	1.08695329245934	1.09037317960399	1.09438409800689	1.09890036790214	1.10383244749892	1.10909095481556	1.11458985360422	1.12024894219404	1.12599503390643	1.13176425423466	1.13750116715745	1.14315944062999	1.14870148255510	1.15409746495767	1.15932457025704	1.15538333881893	1.16032086608737
1.06621530959427	1.06663659868318	1.06788960926852	1.06994250520884	1.07274418564569	1.07622764257724	1.08031398778481	1.08491595263317	1.08994305205709	1.09530438444459	1.10091223529829	1.10668497218274	1.11254858913833	1.11843746088128	1.12429510018240	1.13007421490427	1.13573647517832	1.14125091967041	1.13770819745700	1.14296615313123	1.14801304298620
1.05223804429818	1.05266694924678	1.05394290445888	1.05603325782975	1.05888649793685	1.06243487483468	1.06659796171234	1.07128767695257	1.07641168579365	1.08187768395451	1.08759693929484	1.09348606692625	1.09946963324536	1.10548112160545	1.11146238436055	1.11736531013310	1.12315083042789	1.12878702606676	1.12555954494231	1.13093320389459	1.13609242543235
1.03859451644240	1.03903126709843	1.04033041832364	1.04245918877067	1.04536513656369	1.04897948660201	1.05322098317714	1.05799992123958	1.06322286996077	1.06879612428431	1.07462914746438	1.08063752194325	1.08674399333272	1.09288086796956	1.09898901528770	1.10501923969085	1.11093127563811	1.10807991123911	1.11377891199858	1.11927155900448	1.12454612283295
1.02526844482002	1.02571317840660	1.02703604546444	1.02920377312243	1.03216336340224	1.03584518791096	1.04016661717637	1.04503695372358	1.05036107558694	1.05604382033918	1.06199353291753	1.06812366015141	1.07435630800905	1.08062179581136	1.08686039033945	1.09302127921656	1.09051249506409	1.09652831710762	1.10235274184539	1.10796779343901	1.11336122199058
1.01224498849603	1.01269782232247	1.01404489066924	1.01625242076347	1.01926688751174	1.02301752067524	1.02742067939766	1.03238430160539	1.03781196692816	1.04360702342328	1.04967602738883	1.05593134414459	1.06229334168460	1.06869095862623	1.07506346628460	1.08135865372844	1.07916781050227	1.08531544891031	1.09126913582209	1.09701022564120	1.10252616290238
0.999508509712338	0.999969596500039	1.00134130254985	1.00358969176600	1.00666015946453	1.01048119305961	1.01496793183038	1.02002721489361	1.02556086492314	1.03147090021660	1.03766235667550	1.04404609143693	1.05054089971472	1.05707444164409	1.06358457992054	1.06169226898136	1.06814573842612	1.07442918946483	1.08051613733866	1.08638721239697	1.09202953007700
0.987044847840734	0.987514435355102	0.988911479055770	0.991201391367273	0.994329235735700	0.998222307139500	1.00279475375941	1.00795191188838	1.01359423618708	1.01962237188492	1.02593954669026	1.03245515629761	1.03908652307282	1.04576023694209	1.05241213201599	1.05083957084430	1.05743494089387	1.06385847659663	1.07008287332212	1.07608822608150	1.08186114445692
0.974839372022290	0.975317667613270	0.976740643839932	0.979073207757981	0.982259661499030	0.986226666170665	0.990886860828305	0.996144379500668	1.00189856530033	1.00804787078401	1.01449425412758	1.02114578834196	1.02791781299319	1.03473565551440	1.04153377644905	1.04028239981518	1.04702431958642	1.05359249750498	1.05995892235373	1.06610323419231	1.07201121028764
0.962877235401776	0.963364368015672	0.964813788421783	0.967190235373281	0.970437006982978	0.974479710986119	0.979230117253955	0.984590972536197	0.990459761873155	0.996733747734852	1.00331333448437	1.01010473909638	1.01702198586046	1.02398860944243	1.02297818861923	1.03000963695140	1.03690305088563	1.04362091712877	1.05013435087988	1.05642230246047	1.06247029380685
0.951144414494132	0.951640604455224	0.953117415706185	0.955538600344420	0.958847342117923	0.962967920407074	0.967811141377313	0.973278160691762	0.979264998308546	0.985667619930627	0.992384619534040	0.999320346976499	1.00638744866931	1.01350807979648	1.01282201475920	1.02001096249821	1.02706114418960	1.03393410011951	1.04060000690265	1.04703721110897	1.05323060413954
0.939625552077083	0.940131129415302	0.941635969641182	0.944103456958668	0.947475802773201	0.951676640948970	0.956615285788295	0.962191724715233	0.968300572312432	0.974835892836756	0.981694727470446	0.988779872767433	0.996002080248973	0.995481524847198	1.00292362250832	1.01027591924318	1.01748881298237	1.02452259745755	1.03134697665838	1.03793919387489	1.04428376762253
0.928305339961600	0.928820706750783	0.930354200637282	0.932869400030453	0.936307338685286	0.940591063928284	0.945628332564589	0.951317741356177	0.957552611468104	0.964225146641701	0.971230657305643	0.978470629072794	0.985853755970074	0.985661060042334	0.993272170390820	1.00079424490942	1.00817623407402	1.01537743875398	1.02236668632908	1.02912053731056	1.03562260982425
0.917168309367948	0.917693623296822	0.919257004403655	0.921821364551748	0.925327138472508	0.929696335892939	0.934835613987757	0.940642032139201	0.947007472391434	0.953822384986512	0.960980194596734	0.968380812077298	0.975931049432810	0.976070856150685	0.983857816752252	0.991556508069012	0.999114658396579	1.00649058193380	1.01365185327618	1.02057435749927	1.02724119820849
0.906197032384533	0.906732712743267	0.908326883804739	0.910942163270539	0.914518318294708	0.918976035862177	0.924221058793731	0.930148913098608	0.936649759079271	0.943612494870238	0.950928912836513	0.958496641284896	0.966221186490548	0.966698898108681	0.974669323146543	0.982552485082244	0.990294698076812	0.997853235649392	1.00519445908203	1.01229356529634	1.01913295349216
0.895372659774826	0.895919062507808	0.897545374500823	0.900213447041546	0.903862646340989	0.908412317129016	0.913767321428931	0.919821592657503	0.926463578806515	0.933580303546294	0.941062089563065	0.948804286632969	0.949351481545238	0.957533440342455	0.965695791627263	0.973772038333150	0.981707241870037	0.989457170222183	0.996987328791470	1.00427193772069	1.01129268312496
0.884674670087194	0.885232233968926	0.886891917334763	0.889615182226637	0.893340276004823	0.897986216866759	0.903455804709924	0.909641927382620	0.916431263626693	0.923709338431184	0.931363940931173	0.939289179824288	0.940181689117281	0.948562131065548	0.956925986825836	0.965205227573559	0.973343278737826	0.981294734868436	0.989023879877951	0.996504034914973	1.00371605913386
0.874079016192918	0.874648257165883	0.876342780822556	0.879123570314154	0.882928475392690	0.887674869321827	0.893264606082083	0.899589056065730	0.906533095735975	0.913980262530956	0.921816772733487	0.929934343565856	0.931181671624749	0.939771083130573	0.948347465590562	0.956840878782068	0.965193287521691	0.973357665046971	0.981297406086904	0.988984608042968	0.996399310037299
0.863557324251253	0.864138858874816	0.865870091328160	0.868711412232076	0.872599935119248	0.877452022073077	0.883168375707153	0.889638327194492	0.896745417653968	0.904371148202431	0.912399741234215	0.920720777173821	0.922334054092198	0.931144852774776	0.939946288232903	0.948666956335741	0.957246794326785	0.965637476928309	0.973801070189327	0.981708527615085	0.989339181066157
0.853074117170563	0.853668569995768	0.855438506950813	0.858344064832260	0.862320890257011	0.867285053351258	0.873135354135380	0.879759632469323	0.887039712349013	0.894855020325416	0.903087614688554	0.911625249055271	0.913618278285109	0.922664882385222	0.931706569811137	0.940669600478351	0.949492465201028	0.958125082573433	0.966528208412705	0.974671746473370	0.982533865173405
0.842583223957177	0.843191614422791	0.845002634640911	0.847976418176959	0.852047818397476	0.857131153640991	0.863124242818505	0.869913667980439	0.877378605704325	0.885396642996333	0.893847878646554	0.895796603175484	0.905007987011911	0.914307902566462	0.923607761147920	0.932831621614926	0.941916072327841	0.950809534396220	0.959471045816835	0.967869344556419	0.975981397554303
0.832021472992259	0.832644668070590	0.834500220972118	0.837547388074027	0.841720497766546	0.846932472205632	0.853079905837185	0.860047079311409	0.867711635768829	0.875949111584920	0.884636810427471	0.886989732768581	0.896468046374305	0.906042823690466	0.915623066139027	0.925130601441610	0.934499947299785	0.943677393544865	0.952620720099493	0.961296889903357	0.969682211225524
0.821292708602397	0.821932178617049	0.823836289781026	0.826963632508027	0.831247800032807	0.836600598502441	0.842916651456688	0.850078624845737	0.857962224432376	0.866440234392567	0.875387844829367	0.878173112576650	0.887945114669212	0.897822747716379	0.907712162563625	0.917532717887069	0.927216537643019	0.936708224040894	0.945963314255148	0.954947563797143	0.963635748525843
0.810237265110599	0.810894811913889	0.812853080557038	0.816070201851747	0.820478491103743	0.825988639340065	0.832493589892515	0.839873800401779	0.848002795654600	0.856750870944846	0.865990269278252	0.869247771008361	0.879350555761019	0.889569388083380	0.899807551764608	0.909981408383870	0.920020798717534	0.929867675046496	0.939475751081549	0.948809033860664	0.957840779329107
0.798537332757122	0.799215943026529	0.801237187609097	0.804558452933411	0.809111213448251	0.814804427609330	0.821529178028189	0.829163772381740	0.837578878873656	0.846642079449701	0.856222684994420	0.860015292163159	0.870505296453356	0.881124221909243	0.891771558397571	0.902360467567793	0.912817712895435	0.923082553340066	0.933106489859483	0.942851385988213	0.952288437249317
0.785298779173923	0.786004190263747	0.788105755276040	0.791559766365715	0.796296666957567	0.802223188684710	0.809228420740746	0.817187819838046	0.825968651114764	0.835434981216220	0.845452347032038	0.849905304784361	0.860892433279021	0.872025122246772	0.883198348440926	0.894321207471287	0.905316387352810	0.916120063414992	0.926680198411053	0.936956226152947	0.946917357462047];


            app.Dys_1550 = [7.94125640530400	7.94120764808516	7.94106175852013	7.94082349997564	7.94050133344974	7.94009962429472	7.93963315174959	7.93911156834216	7.93854669190091	7.93794758958479	7.93732754156152	7.93669439580931	7.93605767844243	7.93542303464987	7.93479707619231	7.93418635868620	7.93359228154399	7.93301960837555	7.93246893765654	7.93194230040129	7.93143929569937
7.24626256888515	7.24620888369530	7.24605126303519	7.24579133576657	7.24543925909112	7.24500320012374	7.24449549336402	7.24392688454491	7.24331174014632	7.24266069662243	7.24198415388026	7.24129375082411	7.24060104657088	7.23991076974696	7.23923081574429	7.23856503201919	7.23791976272756	7.23729503269042	7.23669684275635	7.23612295122518	7.23557650785632
6.56248294700418	6.56242267239125	6.56224145262162	6.56194506523810	6.56154336064564	6.56104559698569	6.56046612414354	6.55981742068137	6.55911388540129	6.55837001200490	6.55759889829101	6.55681199050142	6.55601985988958	6.55523190305849	6.55445544694259	6.55369575587756	6.55295846409507	6.55224636989240	6.55156166740380	6.55090738907618	6.55028304105849
6.07649887040751	6.07643282749346	6.07623776827617	6.07591945792035	6.07548727668712	6.07495144839388	6.07432708131992	6.07362926034896	6.07287183017253	6.07207166623042	6.07124113308582	6.07039394774777	6.06954109496100	6.06869290067974	6.06785665373045	6.06703919625041	6.06624542600086	6.06547919416180	6.06474277844696	6.06403857802181	6.06336614642345
5.65768130153694	5.65761090604970	5.65740154148650	5.65706059624478	5.65659694676323	5.65602283752978	5.65535409202462	5.65460556285140	5.65379446485878	5.65293661403462	5.65204643576917	5.65113898881200	5.65022542711153	5.64931629568450	5.64842094176915	5.64754507760057	5.64669455134975	5.64587356710227	5.64508504878596	5.64433010228650	5.64361076628996
5.29321315913012	5.29313839844629	5.29291532744132	5.29255098861133	5.29205550697268	5.29144370184080	5.29072903082960	5.28992999623385	5.28906476506127	5.28814895361039	5.28719900001389	5.28623082549696	5.28525590376322	5.28428596732634	5.28333016557317	5.28239580765365	5.28148796465520	5.28061260585813	5.27977107523491	5.27896624368517	5.27819827242299
4.97322294469002	4.97314376247508	4.97290572316126	4.97251866117980	4.97199170312950	4.97134051525345	4.97058042434724	4.96973080525485	4.96880994426749	4.96783621411426	4.96682623617835	4.96579613466105	4.96475932151271	4.96372795473612	4.96271218695587	4.96171838675714	4.96075352152191	4.95982294363923	4.95892814102797	4.95807208169160	4.95725580314439
4.69017959801368	4.69009484056969	4.68984374129748	4.68943231875790	4.68887372638612	4.68818329863916	4.68737739992039	4.68647618170526	4.68549990989170	4.68446710885780	4.68339640580906	4.68230422263939	4.68120563635749	4.68011219094589	4.67903487370658	4.67798202066768	4.67695965601108	4.67597271652558	4.67502482518813	4.67411785875202	4.67325224703267
4.43810989451098	4.43801989358992	4.43775389894883	4.43731932865190	4.43672897169080	4.43599779398910	4.43514667006248	4.43419360804785	4.43316140594656	4.43206961635323	4.43093757806462	4.42978278246334	4.42862092514775	4.42746526125809	4.42632646363816	4.42521348410394	4.42413286983046	4.42309011787763	4.42208814013680	4.42112994248442	4.42021523822891
4.21226536087304	4.21217060891328	4.21189016126859	4.21143153726928	4.21080919341982	4.21003786756842	4.20913967996696	4.20813464093274	4.20704553931376	4.20589399284626	4.20470066783113	4.20348263532489	4.20225731625603	4.20103844473895	4.19983830691000	4.19866426931548	4.19752574337895	4.19642551058152	4.19536964791761	4.19435918903999	4.19339536261967
4.00880926884765	4.00870895125117	4.00841434220191	4.00793150952515	4.00727650657360	4.00646475976631	4.00551891402414	4.00446116486095	4.00331558435683	4.00210348900994	4.00084739329213	3.99956642242818	3.99827739818571	3.99699551938661	3.99573215399352	3.99449766404693	3.99329938531792	3.99214294282131	3.99103192149045	3.98996908898833	3.98895587043600
3.82461251454848	3.82450751847770	3.82419753002418	3.82369149911858	3.82300168905774	3.82214987725658	3.82115674021159	3.82004626871048	3.81884243794491	3.81756996931156	3.81625082534148	3.81490629231637	3.81355280462347	3.81220688504704	3.81088041155640	3.80958535210346	3.80832728530564	3.80711331806666	3.80594791319806	3.80483215781898	3.80376911264188
3.65712246462497	3.65701308461090	3.65668824639154	3.65615641484803	3.65543399249473	3.65454054015231	3.65349966198659	3.65233506529681	3.65107339801767	3.64973964829846	3.64835753529307	3.64694738096431	3.64552963810842	3.64411865455834	3.64272981711927	3.64137187596308	3.64005430335240	3.63878255466000	3.63756091558069	3.63639269398253	3.63527902278308
3.50417470311971	3.50405975927488	3.50371969642939	3.50316328317669	3.50240737118028	3.50147257173012	3.50038224911869	3.49916390067948	3.49784404838203	3.49644799645770	3.49500172107341	3.49352704518755	3.49204387096491	3.49056802686141	3.48911530754138	3.48769448732251	3.48631668798369	3.48498734626186	3.48370936215522	3.48248874698465	3.48132384614772
3.32482528314043	3.32470052612789	3.32433000926599	3.32372511097171	3.32290244321952	3.32188506297928	3.32069929947929	3.31937352898055	3.31793693041036	3.31641793256585	3.31484347288592	3.31323831854396	3.31162302869071	3.31001707974178	3.30843471053133	3.30688871545508	3.30538778992828	3.30393958489080	3.34261925291095	3.34134459941657	3.34012835339448
3.19796660794614	3.19783657855710	3.19745025758283	3.19681898924314	3.19596116399268	3.19489983542632	3.19366314981440	3.19228082375975	3.19078271900962	3.18919900532284	3.18755739324609	3.18588360230473	3.18420034519336	3.18252579237187	3.18087646001102	3.17926479054188	3.17770064904148	3.17619121407707	3.17474192438551	3.17335565205664	3.17203377156938
3.08091633253151	3.08078093716585	3.08037841236967	3.07972068931772	3.07882754098181	3.07772231851779	3.07643397393523	3.07499380592678	3.07343417345777	3.07178494744591	3.07007593492567	3.06833324979094	3.06658029265380	3.06483745858661	3.06312035301450	3.06144280101016	3.05981507400856	3.05824434641075	3.05673590252726	3.05529338182327	3.05391796160766
2.97260428043657	2.97246297518563	2.97204462458676	2.97136044455968	2.97043092468812	2.96928125531354	2.96794158641518	2.96644374654673	2.96482115406613	2.96310583012936	2.96132815380017	2.95951613210546	2.95769364941103	2.95588120909697	2.95409612967103	2.95235228578758	2.95065999056503	2.94902741979935	2.94745945731831	2.94596015912198	2.94453070702905
2.87210777540838	2.87196104175353	2.87152618067093	2.87081548949954	2.86984943817416	2.86865472058692	2.86726254001040	2.86570651586309	2.86402064067982	2.86223889424159	2.86039246386851	2.85851007140136	2.85661675966383	2.85473481304331	2.85288109848527	2.85107066295725	2.84931337176822	2.84761809935501	2.84599073247224	2.84443387458775	2.84295004574138
2.77863227363039	2.77848036499535	2.77802853875445	2.77729059679364	2.77628800571815	2.77504788181428	2.77360291405470	2.77198771246396	2.77023812196901	2.76838896961576	2.76647291207595	2.76452014265749	2.76255600575199	2.76060330108146	2.75868014051336	2.75680218051179	2.75497968559549	2.75322176903106	2.75153372882761	2.74991969290795	2.74838099423777
2.69148738305305	2.69132974184495	2.69086106065978	2.69009584709932	2.68905602190402	2.68776999594389	2.68627153770566	2.68459679515119	2.68278279117824	2.68086557989950	2.67887952296468	2.67685478071436	2.67481922023383	2.67279546953953	2.67080276160096	2.66885628761741	2.66696789094807	2.66514637586177	2.66339757837110	2.66172548772503	2.66013187152810
2.61006873686568	2.60990523944642	2.60941976404060	2.60862677352084	2.60754921740482	2.60621684272226	2.60466447031749	2.60292961543736	2.60105045923438	2.59906449970405	2.59700739721472	2.59491070754668	2.59280273264595	2.59070715656513	2.58864408508571	2.58662863622238	2.58467375492665	2.58278843401431	2.58097831176210	2.57924768895327	2.57759826746849
2.53384903786877	2.53367985725201	2.53317726747373	2.53235622677415	2.53124053938604	2.52986111011257	2.52825405555628	2.52645826745285	2.52451345429399	2.52245800516942	2.52032911133234	2.51815941744157	2.51597822878466	2.51381022484178	2.51167603192214	2.50959141183102	2.50756948655898	2.50561964747227	2.50374745895540	2.50195805965727	2.50025266127551
2.46236259052838	2.46218746680550	2.46166755108947	2.46081806867280	2.45966396089026	2.45823709240940	2.45657455042544	2.45471711756278	2.45270554819601	2.45058022049170	2.44837871706327	2.44613545410479	2.44388045827175	2.44163901764152	2.43943291067916	2.43727835061300	2.43518885131164	2.43317359099251	2.43123929032183	2.42939021868830	2.42762816799257
2.39519810204398	2.39501709808297	2.39447942186287	2.39360130055438	2.39240813331207	2.39093305513869	2.38921479386531	2.38729480375903	2.38521591396347	2.38301941758312	2.38074470210669	2.37842698635795	2.37609716808137	2.37378196010018	2.37150314995924	2.36927804176386	2.36711990750286	2.36503903383189	2.36304200710387	2.36113288825719	2.35931367193744
2.33199149122006	2.33180425399098	2.33124917364446	2.33034169545502	2.32910911040068	2.32758518700837	2.32581028046748	2.32382742493047	2.32168009157661	2.31941181100316	2.31706286760531	2.31467000009027	2.31226466840632	2.30987474421536	2.30752254453451	2.30522611453538	2.30299902738888	2.30085169505317	2.29879094746404	2.29682121635882	2.29494473522792
2.27241796390598	2.27222497810281	2.27165138478147	2.27071442937144	2.26944194753858	2.26786875897930	2.26603641184734	2.26398959044332	2.26177309408086	2.25943211166736	2.25700825640533	2.25453880923121	2.25205732803889	2.24959193522666	2.24716535281540	2.24479675568002	2.24249983732108	2.24028569978373	2.23816073217385	2.23612987075675	2.23419525043000
2.21618939321297	2.21599005966580	2.21539827092487	2.21443169544359	2.21311806143770	2.21149512397538	2.20960455278032	2.20749278953538	2.20520663291108	2.20279191881547	2.20029198210734	2.19774551362033	2.19518663660277	2.19264472440104	2.19014326249495	2.18770172450880	2.18533450463551	2.18305228986330	2.18086252235727	2.17876999538501	2.17677669124755
2.16304434648918	2.16283887840973	2.16222805099104	2.16123109717323	2.15987695255790	2.15820304261902	2.15625342138142	2.15407579917484	2.15171862918965	2.14922962748201	2.14665251775811	2.14402807524615	2.14139106619928	2.13877177005117	2.13619436860817	2.13367905562978	2.13124042238194	2.12888979247893	2.12663461683385	2.12447984862649	2.12242734448328
2.11275182643820	2.11253976634937	2.11191031168256	2.11088264377030	2.10948646371468	2.10776115515330	2.10575139423982	2.10350751928270	2.10107868566909	2.09851390672677	2.09585887385137	2.09315513587728	2.09043908302306	2.08774136936722	2.08508732677006	2.08249734538502	2.07998685935284	2.07756682744368	2.07524559964349	2.07302775427024	2.07091524288709
2.06510067010328	2.06488227994614	2.06423370212499	2.06317461094318	2.06173600834327	2.05995837345126	2.05788825077543	2.05557650679501	2.05307486014355	2.05043340291148	2.04769948939927	2.04491552135190	2.04211928433317	2.03934227051797	2.03661055383559	2.03394497657195	2.03136147857266	2.02887169045301	2.02648342533212	2.02420175412385	2.02202906223159
2.01990251637283	2.01967765823828	2.01900948291012	2.01791869968288	2.01643696297125	2.01460629138796	2.01247451372872	2.01009425474706	2.00751854835418	2.00479935107152	2.00198525898976	1.99912039460800	1.99624282851789	1.99338552007166	1.99057502473588	1.98783294292244	1.98517550138712	1.98261515163045	1.98015925933748	1.97781299600991	1.97557935085688
1.97698644207380	1.97675484342926	1.97606690881953	1.97494370153282	1.97341835470010	1.97153380397954	1.96933942152853	1.96688947775115	1.96423882481378	1.96144058271898	1.95854535127822	1.95559777272714	1.95263795491971	1.94969934636917	1.94680915579792	1.94398962271825	1.94125775540805	1.93862559832399	1.93610113525473	1.93368991401833	1.93139425673945
1.93619578965186	1.93595751109979	1.93524954073615	1.93409330619853	1.93252358299440	1.93058408487114	1.92832627892723	1.92580553948469	1.92307865496929	1.92020028108315	1.91722273778379	1.91419169334456	1.91114824957067	1.90812676498094	1.90515608158431	1.90225822221843	1.89945065614241	1.89674572404907	1.89415235799176	1.89167515072666	1.88931715286211
1.89739037481261	1.89714529141110	1.89641668243681	1.89522747485490	1.89361242263040	1.89161719668876	1.88929481527512	1.88670212675134	1.88389764675126	1.88093821580990	1.87787686654793	1.87476087986026	1.87163276092323	1.86852765982219	1.86547496184553	1.86249773204108	1.85961369641996	1.85683529775925	1.85417161425885	1.85162778607327	1.84920649764081
1.86044122201309	1.86018896412924	1.85943952070039	1.85821638241991	1.85655535791547	1.85450338552505	1.85211525296263	1.84944981382499	1.84656656682221	1.84352439605009	1.84037768789357	1.83717581254794	1.83396201256275	1.83077187325176	1.82763623084024	1.82457836319068	1.82161652028841	1.81876374532970	1.81602913377991	1.81341785158553	1.81093251026010
1.82523057370879	1.82497094071891	1.82420069674828	1.82294275139315	1.82123493026218	1.81912542591586	1.81667016289310	1.81393014196889	1.81096719770431	1.80784095250152	1.80460795832901	1.80131867203169	1.79801741458612	1.79474126021088	1.79152133144492	1.78838178232883	1.78534130189187	1.78241306036458	1.77960635638307	1.77692677736411	1.77437659788440
1.79165170575308	1.79138486105385	1.79059265863989	1.78929965630135	1.78754421839838	1.78537591892980	1.78285271783315	1.78003731873819	1.77699274787970	1.77378106548643	1.77046028410386	1.76708224449189	1.76369201205396	1.76032860433719	1.75702313205989	1.75380086754359	1.75068036397870	1.74767579686852	1.74479631082704	1.74204736061411	1.73943160063647
1.75960529405815	1.75933120587302	1.75851703008816	1.75718815178844	1.75538386764720	1.75315568981999	1.75056327625494	1.74767078601673	1.74454342913245	1.74124508317822	1.73783469440280	1.73436622047733	1.73088616182521	1.72743370467366	1.72404140605642	1.72073491274245	1.71753372354968	1.71445141655089	1.71149799806734	1.70867885975041	1.70599670942184
1.72900232946354	1.72872072792001	1.72788415632496	1.72651845495447	1.72466475732551	1.72237552713541	1.71971236638805	1.71674164444959	1.71352982345908	1.71014312926653	1.70664182431530	1.70308146481457	1.69950977944154	1.69596701995703	1.69248681505931	1.68909488224175	1.68581135741562	1.68265059204729	1.67962201390966	1.67673176127521	1.67398222725012
1.69975890942852	1.69946943081189	1.69860967517301	1.69720709917834	1.69530275574516	1.69295134388235	1.69021635433224	1.68716536294579	1.68386771491751	1.68039104252932	1.67679730145039	1.67314336396853	1.66947847285584	1.66584415546932	1.66227428216726	1.65879552993289	1.65542841936429	1.65218770022285	1.64908352018663	1.64612093475284	1.64330320853988
1.67179926247532	1.67150191280918	1.67061884381194	1.66917811285977	1.66722222939127	1.66480731038529	1.66199888412368	1.65886628230283	1.65548122809012	1.65191246523775	1.64822477919491	1.64447560357571	1.64071609611038	1.63698829582623	1.63332727958074	1.62976054582514	1.62630846729929	1.62298707931025	1.61980521928028	1.61676960677605	1.61388250315942
1.64505324260720	1.64474766832838	1.64384091077731	1.64236083440819	1.64035216882449	1.63787282502935	1.63498956615685	1.63177381233043	1.62829940764609	1.62463731946439	1.62085336264263	1.61700728236672	1.61315107197097	1.60932844198786	1.60557466674546	1.60191830743780	1.59838042946578	1.59497624831533	1.59171625940821	1.58860596926050	1.58564868027578
1.61945542991335	1.61914173682948	1.61821046462668	1.61669081843756	1.61462853975979	1.61208291398204	1.60912257747001	1.60582238601389	1.60225662170247	1.59849907724271	1.59461735346498	1.59067265666027	1.58671826721815	1.58279892580032	1.57895101020519	1.57520328086266	1.57157764649293	1.56809003579877	1.56475038062665	1.56156495493242	1.55853607341532
1.59494664814467	1.59462452702797	1.59366842394497	1.59210797911498	1.58999053744479	1.58737711240331	1.58433899290277	1.58095188706981	1.57729347292184	1.57343843748118	1.56945693311535	1.56541150145304	1.56135705749521	1.55733914986983	1.55339525686654	1.54955491267007	1.54584036775893	1.54226739907036	1.53884684028632	1.53558462430860	1.53248347786695
1.57147047884032	1.57113984385439	1.57015808587914	1.56855611545786	1.56638227772182	1.56369994770846	1.56058179431904	1.55710613281813	1.55335267856546	1.54939809230201	1.54531473707630	1.54116689546660	1.53701027445376	1.53289206691910	1.52885047281032	1.52491569643160	1.52111034774734	1.51745094962947	1.51394799400132	1.51060816051055	1.50743323311111
1.54897703422343	1.54863753722266	1.54762971404480	1.54598499340365	1.54375376378391	1.54100061768201	1.53780076857234	1.53423441246364	1.53038372720127	1.52632791334609	1.52214070122761	1.51788792735218	1.51362711809501	1.50940646420989	1.50526523266875	1.50123444863770	1.49733693722877	1.49358950450848	1.49000288487643	1.48658349131505	1.48333395871371
1.52741838339120	1.52706977860985	1.52603491517943	1.52434662968250	1.52205647998129	1.51923080481119	1.51594724622819	1.51228863643968	1.50833858167255	1.50417897408718	1.49988541558441	1.49552556462509	1.49115858235253	1.48683360149095	1.48259110183215	1.47846207039241	1.47447072827574	1.47063341862355	1.46696167452930	1.46346200085668	1.46013646342052
1.49515406125457	1.49479212461364	1.50532976480618	1.50359692566022	1.50124628363931	1.49834656613132	1.49497718825654	1.49122347860857	1.48717207621112	1.48290638135873	1.47850423793180	1.47403515484730	1.46955972887054	1.46512822553798	1.46078198181200	1.45655308120205	1.45246603661425	1.44853729388942	1.44477916518294	1.44119712706953	1.43779420389827
1.47562463690042	1.47525349282048	1.47415194062583	1.47235418124205	1.46991450150165	1.46690360184084	1.47484969536079	1.47099912743227	1.46684344699245	1.46246921343659	1.45795612991547	1.45337527878728	1.44878891967883	1.44424870035722	1.43979672543378	1.43546603010710	1.43128121102196	1.42725975076460	1.42341312928049	1.41974763753172	1.41626590413251
1.45689351095040	1.45651290062771	1.45538341779989	1.45354028533137	1.45103944351272	1.44795305515702	1.44436504953692	1.44036559542563	1.44731399359241	1.44282873041241	1.43820191217191	1.43350682994216	1.42880698167819	1.42415573098809	1.41959606424285	1.41516125672496	1.41087685609211	1.40676044577370	1.40282380757755	1.39907332979165	1.39551167692914
1.43892619668440	1.43853607636516	1.43737793411716	1.43548834450920	1.43292464501724	1.42976096716215	1.42608370687910	1.42198546475041	1.41755996220251	1.42394922128919	1.41920598214188	1.41439414988466	1.40957836304842	1.40481346371636	1.40014338098781	1.39560245754739	1.39121642962614	1.38700321477345	1.38297488149517	1.37913787472590	1.37549420865689
1.42168991425515	1.42128994411775	1.42010243030276	1.41816536467769	1.41553713506158	1.41229432289259	1.40852573848435	1.40432615316166	1.39979218828092	1.39501709724627	1.40093438764255	1.39600256726137	1.39106844549414	1.38618729605299	1.38140456500321	1.37675520014290	1.37226514735247	1.36795329207145	1.36383133830367	1.35990597304843	1.35617932987439
1.40515533168382	1.40474505330097	1.40352770356695	1.40154158705820	1.39884734555541	1.39552364017832	1.39166131940834	1.38735806142593	1.38271284537819	1.37782170835232	1.37277347188493	1.36764808182757	1.37324635882370	1.36824633722929	1.36334841583851	1.35858783135118	1.35399197433724	1.34957917594355	1.34536168598186	1.34134613505733	1.33753438673493
1.38929389841895	1.38887345266975	1.38762504901749	1.38558883087435	1.38282675532899	1.37941981378869	1.37546151780268	1.37105189466958	1.36629298152214	1.36128294560655	1.35611292349451	1.35086503425589	1.34561021540341	1.35096033668087	1.34594420972038	1.34107042019172	1.33636606159123	1.33185020008543	1.32753519307743	1.32342757204362	1.31952944705675
1.37408025640883	1.37364893195212	1.37236911054052	1.37028119524332	1.36744963323132	1.36395738996596	1.35990039718756	1.35538180418846	1.35050605681250	1.34537404585831	1.34007953410911	1.33470636965545	1.32932703428650	1.32400255419553	1.32916516485140	1.32417524116030	1.31935989258316	1.31473872973832	1.31032407725216	1.30612231381821	1.30213573900414
1.35949024469631	1.35904793462912	1.35773557906249	1.35559470994873	1.35269173079027	1.34911168338633	1.34495345275232	1.34032292066537	1.33532744327096	1.33007042169666	1.32464818790873	1.31914653220979	1.31363995853621	1.30819054667824	1.30284894505962	1.30787596975734	1.30294714616626	1.29821834159033	1.29370163041093	1.28940385888710	1.28532706277419
1.34550197857180	1.34504842033350	1.34370258980368	1.34150718633891	1.33853053313748	1.33486035419324	1.33059809925559	1.32585265969201	1.32073420306808	1.31534894408647	1.30979554272886	1.30416230628417	1.29852521615909	1.29294794602348	1.28748212101575	1.28216791136340	1.28710399483178	1.28226444407367	1.27764349134126	1.27324763638689	1.26907852126652
1.33209479254769	1.33162953669283	1.33024909660078	1.32799783346145	1.32494546429813	1.32118256089903	1.31681301749906	1.31194949055212	1.30670454189591	1.30118761763467	1.29549980247507	1.28973143326079	1.28396052685534	1.27825217742513	1.27265928197944	1.26722268277914	1.26197308293808	1.26685382553888	1.26212606332157	1.25762951890587	1.25336613502951
1.31925009557216	1.31877287164094	1.31735701987359	1.31504773280606	1.31191745741460	1.30805882318521	1.30357940561484	1.29859410578472	1.29321922839160	1.28756690628470	1.28174083565488	1.27583372787906	1.26992542689233	1.26408282951315	1.25835948824316	1.25279753345816	1.24742805701895	1.24227283025024	1.24712789151309	1.24252848160119	1.23816830715911
1.30695015571932	1.30646044190661	1.30500795342000	1.30263913286461	1.29942838805460	1.29547129891130	1.29087823146373	1.28576776858701	1.28025902028210	1.27446738816559	1.26849935539907	1.26244973412060	1.25640042037041	1.25041984989763	1.24456285004639	1.23887234115167	1.23337994665423	1.22810813604482	1.22307092485108	1.22792311638294	1.22346408792660
1.29517949350430	1.29467701255080	1.29318653600070	1.29075616260372	1.28746261836975	1.28340373902697	1.27869372882280	1.27345399948667	1.26780753450984	1.26187258097935	1.25575824081816	1.24956196246160	1.24336775932872	1.23724555777476	1.23125130266547	1.22542880071563	1.21981055229260	1.21441896002968	1.20926858297886	1.20436704744150	1.20923414714134
1.28392367568435	1.28340808824962	1.28187846452168	1.27938450119974	1.27600493550939	1.27184128630963	1.26701038624297	1.26163753571825	1.25584884648145	1.24976598807305	1.24350109763554	1.23715400499892	1.23081075716747	1.22454274592187	1.21840756131538	1.21244965153020	1.20670208693506	1.20118779219334	1.19592131732843	1.19091040514057	1.18615718380648
1.27316989591915	1.27264056693782	1.27107043518731	1.26851073465916	1.26504255631740	1.26077024622379	1.25581449824876	1.25030395488599	1.24436869946121	1.23813312882036	1.23171315324766	1.22521061406628	1.21871380570131	1.21229581118057	1.20601566607486	1.19991860760384	1.19403830407106	1.18839795298262	1.18301230886935	1.17788919993584	1.17303061822434
1.26290629308483	1.26236294413000	1.26075061517284	1.25812277622144	1.25456281744804	1.25017827166484	1.24509331660368	1.23944060130076	1.23335356975901	1.22696076890335	1.22038042984797	1.21371758455946	1.20706254101107	1.20049026901251	1.19406078467342	1.18782058940608	1.18180378358992	1.17603392896193	1.17052598926630	1.16528770013503	1.16032086608737
1.25312278705071	1.25256462921752	1.25090910847619	1.24821041972780	1.24455521409919	1.24005426243540	1.23483567341638	1.22903559094506	1.22279175786896	1.21623624735813	1.20949033633965	1.20266197936848	1.19584370734811	1.18911225760279	1.18252905235212	1.17614137153559	1.16998404814768	1.16408102541036	1.15844741654028	1.15309080256154	1.14801304298620
1.24381014705940	1.24323676754702	1.24153591946709	1.23876397845838	1.23500999089897	1.23038828088188	1.22503093454628	1.21907834969533	1.21267221421710	1.20594814120715	1.19903105589470	1.19203159395364	1.18504480690353	1.17814910472056	1.17140722644288	1.16486764257662	1.15856557055783	1.15252548882422	1.14676255669487	1.14128434893463	1.13609242543235
1.23496173518688	1.23437231448009	1.23262459640379	1.22977643567644	1.22591969833464	1.22117269767313	1.21567124714776	1.20956027338396	1.20298575179584	1.19608703910398	1.18899266418653	1.18181640482418	1.17465527003550	1.16758968240715	1.16068415246151	1.15398771627810	1.14753645736115	1.14135509895554	1.13545895390360	1.12985546743691	1.12454612283295
1.22657061232377	1.22596473796178	1.22416802014522	1.22124044460741	1.21727682495215	1.21239937439172	1.20674821631596	1.20047295540365	1.19372342634177	1.18664369808797	1.17936555120586	1.17200577098135	1.16466415962166	1.15742317629907	1.15034826707498	1.14348981283930	1.13688462947287	1.13055746645051	1.12452396396153	1.11879152843400	1.11336122199058
1.21863288183410	1.21800986358350	1.21616196218139	1.21315165086286	1.20907673329557	1.20406342534362	1.19825636292238	1.19180980768931	1.18487855542601	1.17761061087780	1.17014182230172	1.16259179918374	1.15506307233021	1.14764022631466	1.14039016490707	1.13336412337470	1.12659966295807	1.12012203985243	1.11394677036521	1.10808114394757	1.10252616290238
1.21114529289878	1.21050416114295	1.20860302840154	1.20550630256339	1.20131513282120	1.19615990391847	1.19019032426539	1.18356549759763	1.17644497834663	1.16898118888241	1.16131385291352	1.15356631329134	1.14584347514980	1.13823188959230	1.13080022733604	1.12360075818537	1.11667154861445	1.11003815808989	1.10371639917231	1.09771316917572	1.09202953007700
1.20410659982496	1.20344681922491	1.20148972980122	1.19830253722325	1.19398985340885	1.18868648401383	1.18254718205701	1.17573612354441	1.16841814083516	1.16075032526352	1.15287643503198	1.14492321119258	1.13699850007474	1.12919102397766	1.12157087038011	1.11419136005422	1.10709122507067	1.10029669197009	1.09382325249504	1.08767783512006	1.08186114445692
1.19751765870557	1.19683799908256	1.19482263488290	1.19154055462447	1.18710036853170	1.18164199967301	1.17532490496858	1.16831919281473	1.16079491780194	1.15291410764183	1.14482473216345	1.13665722959592	1.12852236183170	1.12051087347107	1.11269471367074	1.10512827709264	1.09785100052800	1.09088913053538	1.08425831954695	1.07796567744237	1.07201121028764
1.19138076672015	1.19068037284781	1.18860342798641	1.18522183532852	1.18064810476609	1.17502671072728	1.16852350760359	1.16131378512711	1.15357352600197	1.14546976539606	1.13715520622195	1.12876402130172	1.12040979261925	1.11218595372318	1.10416557647560	1.09640462610570	1.08894301110390	1.08180723834575	1.07501323337387	1.06856769297537	1.06247029380685
1.18570125063195	1.18497873195732	1.18283736193426	1.17935083341638	1.17463653924473	1.16884399104870	1.16214479546259	1.15472114682622	1.14675430759567	1.13841693188128	1.12986661982673	1.12124123150121	1.11265794779761	1.10421198099683	1.09597872069630	1.08801494134135	1.08036138767906	1.07304467749437	1.06608086190997	1.05947632140254	1.05323060413954
1.18048628518859	1.17974092649425	1.17753095490755	1.17393412545701	1.16907125609678	1.16309828103591	1.15619309639166	1.14854395733299	1.14033883077016	1.13175620702223	1.12295827835738	1.11408753044423	1.10526400474718	1.09658574897361	1.08812996748306	1.07995422379978	1.07210013776557	1.06459490208081	1.05745405788527	1.05068400643170	1.04428376762253
1.17574663073157	1.17497650893263	1.17269450525432	1.16898062027312	1.16396088229426	1.15779721200514	1.15067425831567	1.14278737883581	1.13433115176944	1.12549010585349	1.11643174916893	1.10730305092987	1.09822742279991	1.08930551549461	1.08061628641096	1.07221878662651	1.06415500356021	1.05645242419938	1.04912680156965	1.04218390916317	1.03562260982425
1.16143130167472	1.16067708721748	1.15843994789009	1.15479524649833	1.15931852485322	1.15295258537628	1.14559909059068	1.13746056282196	1.12873867988336	1.11962474600750	1.11029177636304	1.10089102899629	1.09154999685656	1.08237178151097	1.07343720809824	1.06480680246469	1.05652313707175	1.04861381081468	1.04109455719801	1.03397099342992	1.02724119820849
1.15735207137505	1.15657452037900	1.15426852835875	1.15051230039622	1.14542853322019	1.13917478091545	1.14097952612209	1.13257387751322	1.12357040367802	1.11416726158504	1.10454361018841	1.09485539964900	1.08523391191670	1.07578534828882	1.06659239320154	1.05771671412590	1.04920173502001	1.04107520068903	1.03335257038040	1.02603933001980	1.01913295349216
1.15375200675325	1.15294972932223	1.15057089554633	1.14669619753607	1.14145305328719	1.13500513630544	1.12754031581064	1.12814298361034	1.11883986435408	1.10912908675121	1.09919665313448	1.08920347697853	1.07928497266685	1.06955024632091	1.06008401493999	1.05094934169706	1.04219012051543	1.03383455146583	1.02589774594048	1.01838481841291	1.01129268312496
1.15065526813299	1.14982675343368	1.14736993119445	1.14336914546282	1.13795656036631	1.13130204070848	1.12360038281683	1.11505852567529	1.11456536892689	1.10452644061548	1.09426452620375	1.08394672330356	1.07371206580094	1.06367318931601	1.05391681369917	1.04450745530819	1.03548960591894	1.02689153970171	1.01872835544762	1.01100443611887	1.00371605913386
1.14809281132668	1.14723603931830	1.14469588980097	1.14055993288357	1.13496608313435	1.12809057617965	1.12013597711574	1.11131713334075	1.10184854974339	1.10038030539708	1.08976548297670	1.07909981016977	1.06852735369898	1.05816372262049	1.04809804168497	1.03839606772795	1.02910288918322	1.02024710853322	1.01184345548509	1.00389566293500	0.996399310037299
1.14610508262419	1.14521802866652	1.14258813057563	1.13830687502898	1.13251780200216	1.12540470919488	1.11717847699611	1.10806211192590	1.09827848209657	1.08803968022854	1.08572383115794	1.07468401920816	1.06374827798331	1.05303599828226	1.04263861355941	1.03262311816193	1.02303556453300	1.01390432449159	1.00524386246696	0.997057272039337	0.989339181066157
1.14474707597640	1.14382707156920	1.14110004106153	1.13666115234515	1.13066061426715	1.12329049157664	1.11477031110783	1.10533249072696	1.09520885908943	1.08461952562770	1.08217355926447	1.07072816768220	1.05939936392670	1.04831023076832	1.03755481506556	1.02720142685354	1.01729666686877	1.00786915362784	0.998932646554945	0.990489522962123	0.982533865173405
1.14409313885195	1.14313722403428	1.14030375835219	1.13569278383153	1.12946145323436	1.12181047404567	1.11296969449789	1.10318165529885	1.09268790663541	1.08171734724420	1.07047784125361	1.06727267481213	1.05551484401800	1.04401482735828	1.03286962043480	1.02214876027614	1.01189966476720	1.00215058138468	0.992915094900819	0.984194433257449	0.975981397554303
1.14424968151824	1.14325405252536	1.14030278297827	1.13550168555991	1.12901528097802	1.12105455127838	1.11186021456400	1.10168631162431	1.09078510482620	1.07939546360728	1.06773373545227	1.05598875280091	1.05214474264541	1.04019187162432	1.02861742561144	1.01749285261954	1.00686557702201	0.996764073903752	0.987200787468697	0.978176448794831	0.969682211225524
1.14537514754418	1.14433464847619	1.14125167026111	1.13623681639153	1.12946453705818	1.12115667097348	1.11156655651460	1.10096109166815	1.08960482418085	1.07774733897291	1.06561494583901	1.05340406265585	1.04127958632995	1.03690407106125	1.02484979009917	1.01327429561144	1.00222547427519	0.991731506941552	0.981803930299416	0.972442007311182	0.963635748525843
1.14772765912701	1.14663593409856	1.14340126472210	1.13814157254106	1.13104113298074	1.12233545622401	1.11229212673887	1.10119289838400	1.08931657862562	1.07692541219278	1.06425646645702	1.05151507236593	1.03887312563726	1.03425387363039	1.02165068849738	1.00955992999592	0.998030213240320	0.987089041493263	0.976746941092511	0.967001488025441	0.957840779329107
1.15179508493319	1.15064205622129	1.14722640485231	1.14167416941973	1.13418267547684	1.12500282904022	1.11442003377061	1.10273393386227	1.09023981585123	1.07721568730291	1.06391100715048	1.05054191218652	1.03728825983383	1.02429473259633	1.01917497286004	1.00647169252264	0.994371136467025	0.982899795320616	0.972066778436326	0.961867706097688	0.952288437249317
1.15882366283923	1.15759080853345	1.15393929565429	1.14800602547687	1.14000514564692	1.13020873813090	1.11892520261115	1.10647737931515	1.09318267750786	1.07933882768082	1.06521196817793	1.05103175113974	1.03698859319819	1.02323472614746	1.00988704653775	1.00429623173852	0.991453426977183	0.979293841595617	0.967824483523226	0.957038136676803	0.946917357462047];

            app.Dzs_1550 = [271.644965694255	271.644963033681	271.644967495471	271.644969001305	271.644966733985	271.644958100099	271.644961823566	271.644964600002	271.644965187289	271.644966554436	271.644962625026	271.644974974645	271.644964220779	271.644964673739	271.644973052140	271.644964402868	271.644968326017	271.644961419972	271.644970031778	271.644972183847	271.644973943902
223.867973556363	223.867973197521	223.867974172413	223.867975368641	223.867971353897	223.867974430359	223.867980335786	223.867973548414	223.867972724746	223.867975578376	223.867974581227	223.867972511002	223.867973752723	223.867972715071	223.867974770391	223.867969939296	223.867971301786	223.867976658525	223.867972454672	223.867972351204	223.867971854533
187.491452489790	187.491450363494	187.491443630687	187.491455002925	187.491456409419	187.491458386765	187.491448853668	187.491454504985	187.491454271300	187.491459064058	187.491453911512	187.491449544783	187.491453531000	187.491457640867	187.491453991922	187.491455438903	187.491455047444	187.491452774573	187.491456982384	187.491454903425	187.491450287515
159.641467310816	159.641466095239	159.641464394596	159.641465682222	159.641469567893	159.641465983523	159.641467376720	159.641464965566	159.641465390728	159.641469563940	159.641467175155	159.641466864639	159.641467153560	159.641466993485	159.641469834915	159.641467439677	159.641472026658	159.641470309509	159.641468299803	159.641470975130	159.641468818515
137.553287966998	137.553289867263	137.553286489724	137.553290746737	137.553290369751	137.553288691016	137.553288429358	137.553284110173	137.553289763036	137.553287956272	137.553288400517	137.553290441821	137.553289691549	137.553289354190	137.553290602914	137.553289211216	137.553291049416	137.553288198222	137.553288254380	137.553287565731	137.553287639677
119.749593774102	119.749596990258	119.749595610395	119.749593921578	119.749596216328	119.749594842014	119.749596130386	119.749595074106	119.749594196088	119.749594872488	119.749592096266	119.749594899199	119.749596895072	119.749595315590	119.749597134447	119.749593591292	119.749595254376	119.749596577803	119.749593767833	119.749596812234	119.749595902888
105.196248673988	105.196249361029	105.196251085344	105.196246800569	105.196249915670	105.196252378107	105.196249039801	105.196250031268	105.196249389664	105.196248891102	105.196252505110	105.196247791727	105.196250172376	105.196249375555	105.196246991221	105.196249056763	105.196247788831	105.196247783255	105.196250470482	105.196248314642	105.196245185111
93.0716717175779	93.0716733700143	93.0716756937500	93.0716721945518	93.0716726306351	93.0716736669779	93.0716748791611	93.0716731462282	93.0716731515605	93.0716734350619	93.0716719495195	93.0716731307033	93.0716728989785	93.0716718095451	93.0716737230483	93.0716701045557	93.0716720687325	93.0716730401434	93.0716721943314	93.0716749856419	93.0716731752214
82.9260255312693	82.9260238807175	82.9260208660774	82.9260223955639	82.9260203725144	82.9260230768677	82.9260271739769	82.9260239613825	82.9260210819522	82.9260231318481	82.9260243447329	82.9260209153573	82.9260223657275	82.9260239603474	82.9260243657107	82.9260224420598	82.9260213866727	82.9260230730700	82.9260230488972	82.9260250991696	82.9260202859694
74.3534201494163	74.3534182318768	74.3534193776706	74.3534184667939	74.3534193634104	74.3534177736610	74.3534214786282	74.3534197749117	74.3534178203878	74.3534187527134	74.3534207719317	74.3534197042007	74.3534182659770	74.3534173091733	74.3534187422322	74.3534191159619	74.3534213908243	74.3534183535017	74.3534198739380	74.3534176110439	74.3534187902840
67.0466649397151	67.0466658954232	67.0466688932809	67.0466679795344	67.0466651411255	67.0466671598467	67.0466651766757	67.0466686981667	67.0466669772378	67.0466666713557	67.0466667032326	67.0466682175470	67.0466673215753	67.0466669573680	67.0466659053689	67.0466676485888	67.0466670835190	67.0466674323045	67.0466665296869	67.0466664826121	67.0466658786718
60.7697690595867	60.7697677747281	60.7697696187807	60.7697684014810	60.7697688521630	60.7697681109469	60.7697693051490	60.7697685539245	60.7697699855071	60.7697685618632	60.7697679599576	60.7697682286286	60.7697689840206	60.7697679615714	60.7697678707329	60.7697686780742	60.7697667735080	60.7697675753590	60.7697679597814	60.7697680422634	60.7697674496147
55.2811824409270	55.2811834267840	55.2811822540604	55.2811828831986	55.2811826391589	55.2811842340155	55.2811836209603	55.2811845526960	55.2811837089519	55.2811819939063	55.2811841708920	55.2811836687976	55.2811845433344	55.2811829046502	55.2811827379659	55.2811836145614	55.2811835393658	55.2811841200385	55.2811843597434	55.2811839368062	55.2811844020306
50.5554584416896	50.5554607536286	50.5554592728232	50.5554587760616	50.5554600718600	50.5554608488707	50.5554604894932	50.5554606086033	50.5554609261858	50.5554613307681	50.5554605999340	50.5554617133653	50.5554604319597	50.5554594090388	50.5554605495542	50.5554603063417	50.5554607261196	50.5554603954743	50.5554598814265	50.5554592734100	50.5554612519961
46.3639987085842	46.3639986855703	46.3639993063607	46.3639996230135	46.3639986675129	46.3640004525718	46.3639976230479	46.3639991906669	46.3639983455535	46.3639990882087	46.3639988265936	46.3639998693032	46.3639981450139	46.3639986257475	46.3639975836435	46.3639983167123	46.3639984167493	46.3639992593942	46.3639999208901	46.3639998350785	46.3639986310163
42.6715298586246	42.6715298462213	42.6715303924543	42.6715296072371	42.6715295846799	42.6715296920001	42.6715297514048	42.6715303045540	42.6715299359120	42.6715305783269	42.6715297483825	42.6715310511669	42.6715300731876	42.6715295480376	42.6715299224250	42.6715310396018	42.6715297072646	42.6715298097191	42.6715299685281	42.6715291961510	42.6715308523452
39.4023992206666	39.4023988415618	39.4024006860522	39.4023993286854	39.4023980812365	39.4023996775911	39.4023996948122	39.4023996054223	39.4024004786287	39.4023995776472	39.4023988777277	39.4024001869750	39.4023995030898	39.4023994915986	39.4023991125434	39.4023993530011	39.4023998195199	39.4023992296450	39.4024001716420	39.4023995150528	39.4023993706498
36.4947228981305	36.4947228148470	36.4947227857164	36.4947238787953	36.4947230271046	36.4947225261419	36.4947233742335	36.4947230478288	36.4947219359200	36.4947222805796	36.4947222061352	36.4947228969894	36.4947230494757	36.4947220395796	36.4947228090005	36.4947222780415	36.4947231400026	36.4947227915798	36.4947217321375	36.4947229749026	36.4947232618482
33.8540749362169	33.8540763389688	33.8540751038300	33.8540763119194	33.8540749279127	33.8540758241835	33.8540740630465	33.8540752239774	33.8540755121366	33.8540755927969	33.8540756189136	33.8540748136221	33.8540754971545	33.8540758536228	33.8540742569205	33.8540744019009	33.8540757651371	33.8540739640562	33.8540751209551	33.8540754770969	33.8540747926471
31.5267236158963	31.5267233508811	31.5267238353000	31.5267238299630	31.5267244616748	31.5267221937044	31.5267237123293	31.5267244563580	31.5267236681653	31.5267236525592	31.5267222027606	31.5267224683467	31.5267241747978	31.5267246930375	31.5267232846489	31.5267236799355	31.5267243633982	31.5267234481591	31.5267242394979	31.5267235856736	31.5267237456557
29.4319275336408	29.4319276680996	29.4319272393425	29.4319289980731	29.4319274296539	29.4319273478229	29.4319277033743	29.4319277457797	29.4319271651626	29.4319275526398	29.4319292883147	29.4319282656588	29.4319279220852	29.4319283779689	29.4319274848947	29.4319280095168	29.4319273886651	29.4319279242820	29.4319274395466	29.4319275001263	29.4319270241654
27.5014423781562	27.5014426874707	27.5014425640270	27.5014436407740	27.5014422676873	27.5014429451505	27.5014429828475	27.5014430033766	27.5014422053800	27.5014425185545	27.5014427135333	27.5014423805773	27.5014425872825	27.5014422370373	27.5014424670587	27.5014428496487	27.5014420266410	27.5014425600263	27.5014423573530	27.5014415760592	27.5014430812006
25.7885427235980	25.7885429895258	25.7885437278866	25.7885433163531	25.7885431153354	25.7885438504641	25.7885423698192	25.7885432264752	25.7885434137763	25.7885428179672	25.7885443182046	25.7885434719320	25.7885432147548	25.7885430516062	25.7885424671934	25.7885437008851	25.7885426251151	25.7885430965822	25.7885418564628	25.7885432361781	25.7885422814077
24.1960375989059	24.1960368196410	24.1960359246701	24.1960355795299	24.1960371556871	24.1960358375786	24.1960362964512	24.1960368268234	24.1960359284862	24.1960367628034	24.1960371118014	24.1960358779267	24.1960371367942	24.1960371485649	24.1960363847785	24.1960370532244	24.1960370905907	24.1960361360898	24.1960363811285	24.1960365522974	24.1960360324602
22.7438739004111	22.7438745869135	22.7438740276690	22.7438736663201	22.7438747867092	22.7438740159481	22.7438739054196	22.7438746770346	22.7438745253292	22.7438750788530	22.7438741180094	22.7438736317468	22.7438745439003	22.7438741534329	22.7438740585391	22.7438734663384	22.7438744788175	22.7438745395447	22.7438743176609	22.7438736664296	22.7438746395551
21.4161474794625	21.4161479857160	21.4161483997471	21.4161480605743	21.4161469739296	21.4161476197122	21.4161481152451	21.4161479965692	21.4161475511312	21.4161478861161	21.4161475121930	21.4161477451136	21.4161475878341	21.4161480547029	21.4161483951558	21.4161484495273	21.4161474965979	21.4161479552639	21.4161471873859	21.4161471804349	21.4161478714755
20.2315456969350	20.2315461329592	20.2315463457164	20.2315463422327	20.2315470514048	20.2315475309183	20.2315471198955	20.2315459979963	20.2315457126899	20.2315461469150	20.2315462848191	20.2315463099668	20.2315460803564	20.2315463204201	20.2315460905443	20.2315459407919	20.2315466636005	20.2315459913567	20.2315466051778	20.2315464719916	20.2315461762977
19.1123655603040	19.1123645304309	19.1123656356887	19.1123645670343	19.1123648447519	19.1123650956249	19.1123647068547	19.1123658035529	19.1123657792584	19.1123645750319	19.1123649227184	19.1123651253595	19.1123656255589	19.1123653870673	19.1123650756971	19.1123652290615	19.1123645598510	19.1123650669625	19.1123651848068	19.1123646903310	19.1123654741176
18.0817115041169	18.0817115790536	18.0817114230729	18.0817119820150	18.0817110935752	18.0817110334410	18.0817113319616	18.0817113404131	18.0817118390741	18.0817111364588	18.0817114910332	18.0817113687279	18.0817111595239	18.0817112524401	18.0817114065716	18.0817115061960	18.0817116009716	18.0817114158446	18.0817117559698	18.0817110244715	18.0817113161610
17.1305511717537	17.1305514949765	17.1305515843460	17.1305515295359	17.1305513108464	17.1305519002979	17.1305510582681	17.1305506105809	17.1305513976195	17.1305518234800	17.1305517743887	17.1305517292108	17.1305509415493	17.1305513945728	17.1305512309825	17.1305510179180	17.1305516162050	17.1305514193597	17.1305515977526	17.1305516944817	17.1305516326892
16.2509704469195	16.2509707411000	16.2509703466219	16.2509710119843	16.2509708782413	16.2509705969732	16.2509712045327	16.2509704545639	16.2509707332621	16.2509702829351	16.2509708365974	16.2509702655564	16.2509704404268	16.2509709939065	16.2509708104610	16.2509709768853	16.2509705485125	16.2509706991208	16.2509706116372	16.2509706324358	16.2509707447845
15.4360086622267	15.4360083692527	15.4360084641224	15.4360087412604	15.4360088157916	15.4360088033715	15.4360086271852	15.4360086040576	15.4360085466381	15.4360083758589	15.4360090732076	15.4360086697901	15.4360086708375	15.4360081086068	15.4360090021573	15.4360083868826	15.4360090728692	15.4360085692594	15.4360088158595	15.4360089338081	15.4360086201375
14.6795261938000	14.6795262671789	14.6795262231542	14.6795258412438	14.6795261901154	14.6795262779178	14.6795261077081	14.6795261978811	14.6795267386853	14.6795260145209	14.6795262384387	14.6795257882087	14.6795262231223	14.6795261122733	14.6795265262089	14.6795260177405	14.6795256281566	14.6795266957295	14.6795264812484	14.6795262436982	14.6795262075328
13.9497120121191	13.9497115657977	13.9497116926299	13.9497122103866	13.9497124076721	13.9497118032999	13.9497118200349	13.9497118852048	13.9497117041917	13.9497116873947	13.9497117278892	13.9497121167555	13.9497118413397	13.9497125239673	13.9497118412554	13.9497122198703	13.9497115304137	13.9497118468994	13.9497122894775	13.9497120470965	13.9497118290187
13.2952287346831	13.2952289508721	13.2952285364346	13.2952291137962	13.2952286975088	13.2952286146397	13.2952287613842	13.2952282846138	13.2952285696088	13.2952292282877	13.2952289238466	13.2952288089513	13.2952288196957	13.2952285131500	13.2952292515376	13.2952291417385	13.2952285836602	13.2952284652849	13.2952287489968	13.2952285149479	13.2952285047901
12.6846577750925	12.6846572943948	12.6846576839234	12.6846580216071	12.6846577522285	12.6846580180990	12.6846576685780	12.6846576596291	12.6846580481481	12.6846575675398	12.6846576549837	12.6846574916955	12.6846577653123	12.6846576666615	12.6846577615822	12.6846576281205	12.6846579126471	12.6846580478232	12.6846577691527	12.6846577737299	12.6846576250171
12.1141759582467	12.1141755559867	12.1141759079208	12.1141763250401	12.1141761252804	12.1141757134039	12.1141760060945	12.1141761145364	12.1141757887758	12.1141759778601	12.1141758861350	12.1141759444058	12.1141757879356	12.1141754027072	12.1141763537804	12.1141757706895	12.1141757059333	12.1141757310759	12.1141758772876	12.1141756907012	12.1141754976514
11.5803609954653	11.5803611232519	11.5803612269702	11.5803610704053	11.5803613095705	11.5803614964746	11.5803607650762	11.5803611456561	11.5803608805779	11.5803615382385	11.5803608963050	11.5803608738804	11.5803610458776	11.5803615176733	11.5803610585869	11.5803609558136	11.5803610856695	11.5803610489029	11.5803610159242	11.5803609116804	11.5803610597169
11.0569756251660	11.0569753418763	11.0569753513722	11.0569754985416	11.0569756859403	11.0569754530142	11.0569756956023	11.0569757044209	11.0569758315592	11.0569760004151	11.0569757524052	11.0569757993958	11.0569758712187	11.0569754080421	11.0569756416140	11.0569757729292	11.0569754899458	11.0569755592860	11.0569759730327	11.0569754794453	11.0569756060578
10.5882132206225	10.5882133205144	10.5882133095574	10.5882127869698	10.5882132294103	10.5882134789575	10.5882134900075	10.5882133059699	10.5882134605849	10.5882133805506	10.5882130839147	10.5882130825798	10.5882130446559	10.5882131218794	10.5882133303126	10.5882130440066	10.5882129231536	10.5882134563755	10.5882128833898	10.5882129973851	10.5882131154970
10.1477926907062	10.1477929407915	10.1477937101615	10.1477935534585	10.1477930682133	10.1477929302410	10.1477934001817	10.1477931432544	10.1477933844430	10.1477931718115	10.1477931808251	10.1477930676429	10.1477929257878	10.1477932984910	10.1477934012800	10.1477930948715	10.1477933085933	10.1477929398602	10.1477929532677	10.1477930268118	10.1477933216866
9.71191914582637	9.71191892059081	9.71191846466249	9.71191822395323	9.71191926109350	9.71191924058570	9.71191848701420	9.71191860026526	9.71191875063473	9.71191889331845	9.71191856835579	9.71191846000814	9.71191843931407	9.71191877111438	9.71191849747078	9.71191868275184	9.71191864403981	9.71191874264506	9.71191868966731	9.71191861055550	9.71191851219430
9.32223192949643	9.32223146485483	9.32223143279884	9.32223166810552	9.32223158518734	9.32223160789222	9.32223148492542	9.32223177660156	9.32223198112166	9.32223162091073	9.32223153865724	9.32223197497632	9.32223167110962	9.32223169498137	9.32223153989486	9.32223176261896	9.32223174231511	9.32223198027214	9.32223147395328	9.32223204638192	9.32223173604569
8.95476755762951	8.95476783525872	8.95476781611667	8.95476738054484	8.95476751271391	8.95476739202515	8.95476786509763	8.95476768392343	8.95476739130870	8.95476762022437	8.95476749708864	8.95476784317095	8.95476786939683	8.95476758273890	8.95476776156626	8.95476766797288	8.95476746587485	8.95476760274807	8.95476749445077	8.95476806602731	8.95476776503095
8.58778241823689	8.58778248663906	8.58778251403588	8.58778236578066	8.58778230140803	8.58778255240801	8.58778204915476	8.58778240235888	8.58778236329919	8.58778247782320	8.58778220169157	8.58778204052118	8.58778234783860	8.58778209037587	8.58778204531428	8.58778229927358	8.58778210016657	8.58778220065133	8.58778238379738	8.58778203365925	8.58778193062330
8.26042781230175	8.26042778059038	8.26042766751943	8.26042751438139	8.26042782404371	8.26042729149993	8.26042773578945	8.26042800502421	8.26042770087591	8.26042787867577	8.26042776505501	8.26042780264271	8.26042789096417	8.26042790709083	8.26042802132262	8.26042796948876	8.26042772879287	8.26042796176230	8.26042789476651	8.26042809243143	8.26042807331278
7.93151273271131	7.93151260364598	7.93151311053248	7.93151271911621	7.93151243417179	7.93151260502466	7.93151283315365	7.93151257190636	7.93151261640974	7.93151277424576	7.93151272535934	7.93151269007284	7.93151276591667	7.93151269398763	7.93151250279383	7.93151255064667	7.93151253263524	7.93151269325036	7.93151287120479	7.93151239614528	7.93151259910110
7.63867445430935	7.63867461976847	7.63867512030814	7.63867493530104	7.63867458520645	7.63867480740901	7.63867454770442	7.63867488750259	7.63867482900654	7.63867472720052	7.63867479634269	7.63867465086809	7.63867461439867	7.63867472645660	7.63867481555008	7.63867474369587	7.63867513139335	7.63867463079883	7.63867477227288	7.63867499276420	7.63867470238338
7.34268138704865	7.34268101806509	7.34268122532608	7.34268121212660	7.34268116714085	7.34268148142746	7.34268154892489	7.34268101814851	7.34268134609405	7.34268116838331	7.34268082365028	7.34268115502388	7.34268124459419	7.34268141520728	7.34268116075839	7.34268130652843	7.34268144654699	7.34268109460039	7.34268101455073	7.34268077310955	7.34268132860151
7.07969068298438	7.07969082787602	7.07969072198798	7.07969058879646	7.07969056697780	7.07969095559960	7.07969052380478	7.07969035616330	7.07969067225154	7.07969041312987	7.07969102108344	7.07969064908760	7.07969073325230	7.07969072882156	7.07969067889568	7.07969077099111	7.07969067657727	7.07969078790402	7.07969076710145	7.07969068001771	7.07969079463582
6.81230481886369	6.81230520995293	6.81230495790740	6.81230518702671	6.81230504003489	6.81230480604049	6.81230506818284	6.81230519233490	6.81230500562651	6.81230527728427	6.81230508344328	6.81230476525844	6.81230492002188	6.81230499017809	6.81230497990180	6.81230499277230	6.81230515281991	6.81230511841664	6.81230500214423	6.81230485142546	6.81230517146403
6.57523601973291	6.57523582940375	6.57523586512865	6.57523610886484	6.57523604108228	6.57523638572435	6.57523592357053	6.57523581049770	6.57523600012978	6.57523581258306	6.57523584809554	6.57523584622842	6.57523587185946	6.57523615913953	6.57523601181575	6.57523584212934	6.57523595968512	6.57523612114102	6.57523590777831	6.57523570948878	6.57523605366238
6.33281904713278	6.33281916507606	6.33281908784352	6.33281950405355	6.33281905451105	6.33281919440439	6.33281912038065	6.33281907443600	6.33281871074505	6.33281915667519	6.33281903583788	6.33281926211286	6.33281918118830	6.33281914901534	6.33281901076600	6.33281908248086	6.33281899072811	6.33281928646269	6.33281886663425	6.33281926865596	6.33281906766733
6.11835403044661	6.11835428476194	6.11835384749335	6.11835419125796	6.11835409830054	6.11835412556391	6.11835421054275	6.11835392971882	6.11835428157771	6.11835393519616	6.11835427924450	6.11835400974856	6.11835417009000	6.11835397052130	6.11835411891603	6.11835417012876	6.11835398698035	6.11835423395968	6.11835416054333	6.11835397739843	6.11835376362480
5.89781790661709	5.89781814799530	5.89781827562685	5.89781832212400	5.89781805695545	5.89781838867911	5.89781834767326	5.89781826538117	5.89781824351774	5.89781827414677	5.89781822786779	5.89781797173453	5.89781829027721	5.89781831606044	5.89781845865689	5.89781834152722	5.89781807048205	5.89781822123427	5.89781808029192	5.89781782329725	5.89781803647004
5.70313125951805	5.70313134114204	5.70313147661734	5.70313151803121	5.70313148828662	5.70313159580986	5.70313125911083	5.70313144481078	5.70313149996914	5.70313159030728	5.70313140108163	5.70313127543803	5.70313147940923	5.70313152422393	5.70313128261232	5.70313156613098	5.70313151028241	5.70313157236284	5.70313161784497	5.70313152111230	5.70313159069361
5.50184147503914	5.50184178444478	5.50184140739965	5.50184177839194	5.50184131172654	5.50184150698240	5.50184173031018	5.50184145044512	5.50184157305533	5.50184183152578	5.50184169349403	5.50184154257179	5.50184155566867	5.50184149584297	5.50184140585234	5.50184149123488	5.50184148499202	5.50184162525966	5.50184158871506	5.50184160062015	5.50184172344897
5.30945745667089	5.30945712443299	5.30945742519878	5.30945770827685	5.30945749560237	5.30945746327201	5.30945724858858	5.30945740168888	5.30945722990173	5.30945755306030	5.30945738569458	5.30945727270360	5.30945728466511	5.30945774239979	5.30945719327888	5.30945742767581	5.30945741052997	5.30945737051983	5.30945735592130	5.30945740595633	5.30945747309794
5.14021033531053	5.14021034447608	5.14021021411409	5.14021054608027	5.14021015268783	5.14021008466580	5.14021022572332	5.14021035251303	5.14021015876115	5.14021046917183	5.14021071475033	5.14021053869622	5.14021049140023	5.14021046758837	5.14021040738595	5.14021030637994	5.14021052580018	5.14021028266345	5.14021016288916	5.14021027064363	5.14021031380501
4.96379398656937	4.96379400644061	4.96379372909703	4.96379370895158	4.96379393416093	4.96379427537346	4.96379419148870	4.96379402554389	4.96379400005041	4.96379405117796	4.96379423802003	4.96379394058231	4.96379393255753	4.96379398773354	4.96379403796806	4.96379421793351	4.96379402556957	4.96379390762080	4.96379401782287	4.96379389072945	4.96379382731695
4.80888797938089	4.80888824159093	4.80888840810970	4.80888824533403	4.80888809026266	4.80888821630610	4.80888838237887	4.80888816223789	4.80888833895197	4.80888822235399	4.80888819722599	4.80888825231482	4.80888801202390	4.80888817866800	4.80888804868993	4.80888815401613	4.80888826085936	4.80888816792777	4.80888826883466	4.80888828008350	4.80888828946845
4.64662263710541	4.64662242825678	4.64662269280043	4.64662284157680	4.64662252592229	4.64662261986565	4.64662274266103	4.64662250538109	4.64662264946493	4.64662269367582	4.64662263237250	4.64662259437895	4.64662263472748	4.64662267990665	4.64662273141512	4.64662266095586	4.64662260099330	4.64662253099441	4.64662261642287	4.64662282308924	4.64662250231609
4.49097523863998	4.49097540871083	4.49097526191712	4.49097534959233	4.49097559895013	4.49097499421856	4.49097520741043	4.49097515567157	4.49097534352704	4.49097522882016	4.49097503640693	4.49097528253829	4.49097518094099	4.49097521076238	4.49097505514155	4.49097502847968	4.49097514546006	4.49097515311028	4.49097534945502	4.49097534823045	4.49097521938416
4.35467785623029	4.35467805346222	4.35467798247275	4.35467803886988	4.35467813721206	4.35467813786311	4.35467791308162	4.35467804790284	4.35467805964838	4.35467823269767	4.35467808547531	4.35467803055248	4.35467812526134	4.35467803073275	4.35467795729349	4.35467826152732	4.35467813163000	4.35467808392428	4.35467801420757	4.35467815041351	4.35467783280982
4.21087941490558	4.21087966328440	4.21087968385236	4.21087947306726	4.21087944433400	4.21087969814496	4.21087972732700	4.21087954375926	4.21087935813675	4.21087954888475	4.21087932012388	4.21087950234721	4.21087969222886	4.21087932845436	4.21087957010606	4.21087956728758	4.21087935842862	4.21087963675123	4.21087954942674	4.21087962290762	4.21087943125988
4.07266134323085	4.07266121312882	4.07266161205135	4.07266148125536	4.07266151183811	4.07266145422170	4.07266128836530	4.07266116176419	4.07266145043421	4.07266141392517	4.07266153385423	4.07266133230368	4.07266140371514	4.07266136993137	4.07266133060418	4.07266141010981	4.07266114972512	4.07266114866178	4.07266139995417	4.07266143811061	4.07266129189409
3.95184800792637	3.95184799029355	3.95184806158018	3.95184801616148	3.95184811090011	3.95184769856810	3.95184804652121	3.95184800045866	3.95184819192497	3.95184800635189	3.95184797337699	3.95184800469140	3.95184792552399	3.95184810935006	3.95184790407849	3.95184810486053	3.95184813353335	3.95184814142356	3.95184809202232	3.95184785280762	3.95184809373978
3.82358024571291	3.82358007098017	3.82357996171217	3.82358014267339	3.82358011802227	3.82357996409996	3.82358008417065	3.82357998897297	3.82358022712792	3.82358008363324	3.82358004687088	3.82358007541305	3.82358003688523	3.82357998625472	3.82358002518421	3.82358018012364	3.82358018944622	3.82358012416616	3.82358010336806	3.82358005264397	3.82358012266770
3.71150335923888	3.71150345591728	3.71150345266227	3.71150333765249	3.71150365049495	3.71150340619806	3.71150358230962	3.71150325715105	3.71150351428921	3.71150364982049	3.71150340756448	3.71150324277881	3.71150334489936	3.71150336298474	3.71150343094204	3.71150326160246	3.71150345417785	3.71150340814425	3.71150330701809	3.71150341145922	3.71150325132493
3.59209673707828	3.59209674628391	3.59209664492749	3.59209684816574	3.59209668476639	3.59209686199734	3.59209669023513	3.59209686612404	3.59209661390357	3.59209671356619	3.59209673112417	3.59209682820821	3.59209671690373	3.59209673363339	3.59209689268382	3.59209684757381	3.59209672459119	3.59209678525793	3.59209672321453	3.59209655816413	3.59209673316003
3.47690663338197	3.47690655330831	3.47690673024730	3.47690682355055	3.47690684715804	3.47690666301975	3.47690649928425	3.47690668630742	3.47690667940452	3.47690665641583	3.47690652044104	3.47690669998903	3.47690676937109	3.47690668370304	3.47690658339283	3.47690656310433	3.47690650400503	3.47690673934532	3.47690677821090	3.47690662404883	3.47690647416350
3.36571110050566	3.36571098705410	3.36571101345029	3.36571103855023	3.36571103487651	3.36571111048161	3.36571112174093	3.36571097727380	3.36571118445863	3.36571106981158	3.36571114656429	3.36571111567513	3.36571104443885	3.36571111467472	3.36571105580602	3.36571101468852	3.36571091789543	3.36571104883205	3.36571096231566	3.36571093440775	3.36571108267850
3.26842146416868	3.26842151854032	3.26842124061363	3.26842147019805	3.26842152905202	3.26842140270531	3.26842144473692	3.26842132616033	3.26842127994009	3.26842140899352	3.26842149827829	3.26842126000611	3.26842135299144	3.26842118997037	3.26842146438636	3.26842145998233	3.26842150884655	3.26842159623495	3.26842135181163	3.26842142187302	3.26842153265382
3.16420793407224	3.16420779887336	3.16420796448018	3.16420779197520	3.16420779110720	3.16420776856823	3.16420762046215	3.16420771869317	3.16420792602659	3.16420793930899	3.16420785380030	3.16420792173877	3.16420784747626	3.16420783540706	3.16420788432279	3.16420772919824	3.16420769085874	3.16420784603955	3.16420768831081	3.16420784374197	3.16420786600717
3.06335674555343	3.06335687421627	3.06335702594119	3.06335695246081	3.06335703650745	3.06335697320225	3.06335683845843	3.06335683206345	3.06335704861227	3.06335695278240	3.06335678781740	3.06335680662101	3.06335672162649	3.06335711721114	3.06335665813484	3.06335669170427	3.06335693466646	3.06335674925140	3.06335690409914	3.06335691326906	3.06335693811777
2.97473527354254	2.97473537649550	2.97473509589454	2.97473530944895	2.97473524043433	2.97473528425930	2.97473513470477	2.97473535419355	2.97473530817368	2.97473532465959	2.97473544423737	2.97473527574892	2.97473508036562	2.97473529128136	2.97473526732574	2.97473526291741	2.97473516062889	2.97473514660469	2.97473512781588	2.97473502769384	2.97473521717852
2.87962914906307	2.87962905886707	2.87962932384028	2.87962931732940	2.87962927040342	2.87962914226298	2.87962925493771	2.87962923055846	2.87962921900398	2.87962904556008	2.87962920200765	2.87962931631356	2.87962933640824	2.87962924114514	2.87962917426902	2.87962912249723	2.87962917326280	2.87962921477780	2.87962902118260	2.87962918687310	2.87962905772324
2.78730583248102	2.78730579289529	2.78730591049921	2.78730582137901	2.78730588253667	2.78730606082865	2.78730569330451	2.78730571376093	2.78730578868450	2.78730568325773	2.78730579929926	2.78730586975036	2.78730572455278	2.78730582026484	2.78730589152754	2.78730570716936	2.78730580228471	2.78730580155865	2.78730575582230	2.78730569110476	2.78730603090565
2.70547935436737	2.70547920072415	2.70547903235455	2.70547933382150	2.70547923669605	2.70547913662942	2.70547909894360	2.70547917312003	2.70547907869067	2.70547926714388	2.70547909628800	2.70547922409208	2.70547908833343	2.70547915297967	2.70547931319102	2.70547911298927	2.70547934673193	2.70547917428567	2.70547915710766	2.70547911537117	2.70547948540811
2.61769719354373	2.61769715629390	2.61769743743555	2.61769739604317	2.61769712000013	2.61769718470867	2.61769734676938	2.61769711859451	2.61769723730242	2.61769708260378	2.61769740645156	2.61769724026412	2.61769729923521	2.61769738931668	2.61769728493476	2.61769719100818	2.61769717216152	2.61769724486790	2.61769718055919	2.61769720473441	2.61769723715931
2.53210949162334	2.53210951407180	2.53210955092674	2.53210940864768	2.53210961782933	2.53210957306203	2.53210964931702	2.53210974289709	2.53210961307965	2.53210966235579	2.53210959228651	2.53210967541934	2.53210973212840	2.53210974764068	2.53210950754080	2.53210956845891	2.53210967258117	2.53210952896422	2.53210968274333	2.53210953725851	2.53210954086861
2.45508419051749	2.45508425150295	2.45508425581341	2.45508425590722	2.45508415985169	2.45508411702545	2.45508419989041	2.45508401942925	2.45508406686122	2.45508417018425	2.45508436533554	2.45508415572915	2.45508432814138	2.45508416057058	2.45508417720567	2.45508424520114	2.45508428735253	2.45508415739287	2.45508414613037	2.45508417570000	2.45508429266009
2.37267279321239	2.37267280006414	2.37267269613320	2.37267271341786	2.37267289464936	2.37267283243905	2.37267274070386	2.37267260146918	2.37267250199875	2.37267275662754	2.37267269689073	2.37267266239967	2.37267257513759	2.37267273921497	2.37267262269170	2.37267255996089	2.37267262873261	2.37267267978266	2.37267275410880	2.37267270632451	2.37267278571451
2.29171766323855	2.29171746355051	2.29171734368835	2.29171749501152	2.29171746232642	2.29171765991613	2.29171740584062	2.29171746757382	2.29171753982308	2.29171761266692	2.29171743607841	2.29171735909017	2.29171763878485	2.29171758443659	2.29171749188987	2.29171753967858	2.29171758641120	2.29171761144399	2.29171757758725	2.29171742958254	2.29171749001948
2.21191051634290	2.21191040073770	2.21191042120814	2.21191036363088	2.21191056738039	2.21191042259925	2.21191044801950	2.21191044176759	2.21191041000014	2.21191048219486	2.21191043507920	2.21191041295827	2.21191040691924	2.21191043966800	2.21191044436446	2.21191044517000	2.21191047693594	2.21191045326068	2.21191043893574	2.21191036059983	2.21191051616477
2.13696109787327	2.13696110493924	2.13696110109983	2.13696103691303	2.13696110182234	2.13696128994803	2.13696110775591	2.13696119855138	2.13696107346267	2.13696107339664	2.13696101165620	2.13696106008279	2.13696095398561	2.13696106598561	2.13696098534809	2.13696100445404	2.13696108868079	2.13696095805462	2.13696106247608	2.13696111645132	2.13696101544903
2.05710250967461	2.05710259140820	2.05710252811934	2.05710245129206	2.05710254890689	2.05710234692568	2.05710221488515	2.05710258819418	2.05710246741544	2.05710244950364	2.05710246681675	2.05710253949553	2.05710249103097	2.05710246930887	2.05710258897845	2.05710235968578	2.05710239369723	2.05710243040677	2.05710250732817	2.05710240238877	2.05710237701172
1.97629657480618	1.97629663695525	1.97629648022881	1.97629661011695	1.97629656791760	1.97629650419110	1.97629660773951	1.97629656420761	1.97629658526319	1.97629666770744	1.97629650555589	1.97629659110931	1.97629653707455	1.97629646828038	1.97629655534323	1.97629657109392	1.97629642327230	1.97629657038652	1.97629657937527	1.97629650201306	1.97629654551062
1.89374921171401	1.89374936311318	1.89374941612812	1.89374937889857	1.89374928588259	1.89374952500801	1.89374935077630	1.89374952510744	1.89374940627690	1.89374919807329	1.89374937107876	1.89374940938878	1.89374941056956	1.89374937632102	1.89374934728620	1.89374927127335	1.89374928489662	1.89374924414299	1.89374937508685	1.89374949771607	1.89374924454614
1.79989219222322	1.79989194208899	1.79989205640175	1.79989206347425	1.79989205321795	1.79989212982303	1.79989212743673	1.79989211822993	1.79989210109170	1.79989213802171	1.79989230875165	1.79989204541124	1.79989195794276	1.79989227110180	1.79989215238716	1.79989217776805	1.79989208217745	1.79989213229862	1.79989216936957	1.79989222975430	1.79989209824665];
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            if ~canUseGPU
                app.GPUSwitch.Value = 'Off';
                app.GPUSwitch.Enable = 'Off';
            else
                app.GPUSwitch.Value = 'On';
            end

            app.NAs_theory = 0.1:0.01:0.99;
            app.circpol_ratios = 0:0.05:1;
            app.load_FWHMs;

            app.run_psf();
        end

        % Value changed function: wavelength_f_editfield
        function wavelength_f_editfieldValueChanged(app, event)
            value = app.wavelength_f_editfield.Value;
            if value == 1.064
                 app.SelectWavelength.Value =1.064;
                app.SelectWavelength.Items = '1064 nm';
            elseif value == 1.55
                 app.SelectWavelength.Value =1.55;
                app.SelectWavelength.Items = '1550 nm';
            end
            app.run_psf();
        end

        % Callback function
        function SelectWavelengthValueChanged(app, event)

        end

        % Clicked callback: SelectWavelength
        function SelectWavelengthClicked(app, event)
            item = event.InteractionInformation.Item;
            if strcmp(item, '1064 nm')
                app.SelectWavelength.Value = 1.064;
            else
                app.SelectWavelength.Value = 1.55;
            end
            app.inverse_psf();
        end

        % Value changed function: NA_f_editfield
        function NA_f_editfieldValueChanged(app, event)
            app.run_psf();
        end

        % Value changed function: power_f_editfield
        function power_f_editfieldValueChanged(app, event)
            app.run_psf();           
        end

        % Value changed function: circpol_f_editfield
        function circpol_f_editfieldValueChanged(app, event)
            app.run_psf();      
        end

        % Value changed function: RI_f_editfield
        function RI_f_editfieldValueChanged(app, event)
            app.RI_i_editfield.Value = app.RI_f_editfield.Value; 
            app.run_psf();
        end

        % Value changed function: rho_f_editfield
        function rho_f_editfieldValueChanged(app, event)
            app.rho_i_editfield.Value = app.rho_f_editfield.Value; 
            app.run_psf();
        end

        % Button down function: InversemodeTab
        function InversemodeTabButtonDown(app, event)
            app.run_inverse_psf();
        end

        % Value changed function: Wx_i_editfield
        function Wx_i_editfieldValueChanged(app, event)
            app.run_inverse_psf();
        end

        % Value changed function: Wy_i_editfield
        function Wy_i_editfieldValueChanged(app, event)
            app.run_inverse_psf();
        end

        % Value changed function: Wz_i_editfield
        function Wz_i_editfieldValueChanged(app, event)
            app.run_inverse_psf();
        end

        % Value changed function: RI_i_editfield
        function RI_i_editfieldValueChanged(app, event)
            app.run_inverse_psf();
        end

        % Value changed function: rho_i_editfield
        function rho_i_editfieldValueChanged(app, event)
            app.run_inverse_psf();
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 424 258];
            app.UIFigure.Name = 'MATLAB App';

            % Create TabGroup
            app.TabGroup = uitabgroup(app.UIFigure);
            app.TabGroup.Position = [2 1 423 257];

            % Create ForwardmodeTab
            app.ForwardmodeTab = uitab(app.TabGroup);
            app.ForwardmodeTab.Title = 'Forward mode';

            % Create wavelengthumLabel
            app.wavelengthumLabel = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel.HorizontalAlignment = 'right';
            app.wavelengthumLabel.Position = [28 182 94 22];
            app.wavelengthumLabel.Text = 'wavelength (um)';

            % Create wavelength_f_editfield
            app.wavelength_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.wavelength_f_editfield.Limits = [0 Inf];
            app.wavelength_f_editfield.ValueChangedFcn = createCallbackFcn(app, @wavelength_f_editfieldValueChanged, true);
            app.wavelength_f_editfield.Position = [129 182 43 22];
            app.wavelength_f_editfield.Value = 1.064;

            % Create wavelengthumLabel_2
            app.wavelengthumLabel_2 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_2.HorizontalAlignment = 'right';
            app.wavelengthumLabel_2.Position = [182 182 25 22];
            app.wavelengthumLabel_2.Text = 'NA';

            % Create NA_f_editfield
            app.NA_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.NA_f_editfield.Limits = [0 1];
            app.NA_f_editfield.ValueChangedFcn = createCallbackFcn(app, @NA_f_editfieldValueChanged, true);
            app.NA_f_editfield.Position = [211 182 38 22];
            app.NA_f_editfield.Value = 0.84;

            % Create wavelengthumLabel_3
            app.wavelengthumLabel_3 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_3.HorizontalAlignment = 'right';
            app.wavelengthumLabel_3.Position = [263 182 91 22];
            app.wavelengthumLabel_3.Text = 'Cir. pol ratio (%)';

            % Create circpol_f_editfield
            app.circpol_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.circpol_f_editfield.Limits = [0 100];
            app.circpol_f_editfield.ValueChangedFcn = createCallbackFcn(app, @circpol_f_editfieldValueChanged, true);
            app.circpol_f_editfield.Position = [362 182 34 22];
            app.circpol_f_editfield.Value = 100;

            % Create wavelengthumLabel_4
            app.wavelengthumLabel_4 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_4.HorizontalAlignment = 'right';
            app.wavelengthumLabel_4.Position = [49 126 72 22];
            app.wavelengthumLabel_4.Text = 'Power (mW)';

            % Create power_f_editfield
            app.power_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.power_f_editfield.Limits = [0 Inf];
            app.power_f_editfield.ValueChangedFcn = createCallbackFcn(app, @power_f_editfieldValueChanged, true);
            app.power_f_editfield.Position = [123 126 37 22];
            app.power_f_editfield.Value = 250;

            % Create wavelengthumLabel_5
            app.wavelengthumLabel_5 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_5.HorizontalAlignment = 'right';
            app.wavelengthumLabel_5.Position = [45 34 70 22];
            app.wavelengthumLabel_5.Text = '# voxel (XY)';

            % Create xyvoxel_f_editfield
            app.xyvoxel_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.xyvoxel_f_editfield.Limits = [3 Inf];
            app.xyvoxel_f_editfield.Position = [119 34 31 22];
            app.xyvoxel_f_editfield.Value = 51;

            % Create wavelengthumLabel_6
            app.wavelengthumLabel_6 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_6.HorizontalAlignment = 'right';
            app.wavelengthumLabel_6.Position = [155 34 62 22];
            app.wavelengthumLabel_6.Text = '# voxel (Z)';

            % Create zvoxel_f_editfield
            app.zvoxel_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.zvoxel_f_editfield.Limits = [3 Inf];
            app.zvoxel_f_editfield.Position = [221 34 31 22];
            app.zvoxel_f_editfield.Value = 101;

            % Create wavelengthumLabel_7
            app.wavelengthumLabel_7 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_7.HorizontalAlignment = 'right';
            app.wavelengthumLabel_7.Position = [297 126 53 22];
            app.wavelengthumLabel_7.Text = 'Ωx / (2π)';

            % Create Wx_f_editfield
            app.Wx_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Wx_f_editfield.Editable = 'off';
            app.Wx_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Wx_f_editfield.Position = [357 126 43 22];

            % Create TrapfreqkHzLabel
            app.TrapfreqkHzLabel = uilabel(app.ForwardmodeTab);
            app.TrapfreqkHzLabel.Position = [311 153 89 22];
            app.TrapfreqkHzLabel.Text = 'Trap freq. (kHz)';

            % Create wavelengthumLabel_8
            app.wavelengthumLabel_8 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_8.HorizontalAlignment = 'right';
            app.wavelengthumLabel_8.Position = [297 97 53 22];
            app.wavelengthumLabel_8.Text = 'Ωy / (2π)';

            % Create Wy_f_editfield
            app.Wy_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Wy_f_editfield.Editable = 'off';
            app.Wy_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Wy_f_editfield.Position = [357 97 43 22];

            % Create wavelengthumLabel_9
            app.wavelengthumLabel_9 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_9.HorizontalAlignment = 'right';
            app.wavelengthumLabel_9.Position = [297 69 53 22];
            app.wavelengthumLabel_9.Text = 'Ωz / (2π)';

            % Create Wz_f_editfield
            app.Wz_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Wz_f_editfield.Editable = 'off';
            app.Wz_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Wz_f_editfield.Position = [357 69 43 22];

            % Create FWHMnmLabel
            app.FWHMnmLabel = uilabel(app.ForwardmodeTab);
            app.FWHMnmLabel.Position = [206 153 70 22];
            app.FWHMnmLabel.Text = 'FWHM (nm)';

            % Create wavelengthumLabel_10
            app.wavelengthumLabel_10 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_10.HorizontalAlignment = 'right';
            app.wavelengthumLabel_10.Position = [189 126 25 22];
            app.wavelengthumLabel_10.Text = 'Δx';

            % Create Dx_f_editfield
            app.Dx_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Dx_f_editfield.Editable = 'off';
            app.Dx_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dx_f_editfield.Position = [225 126 51 22];

            % Create wavelengthumLabel_11
            app.wavelengthumLabel_11 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_11.HorizontalAlignment = 'right';
            app.wavelengthumLabel_11.Position = [189 97 25 22];
            app.wavelengthumLabel_11.Text = 'Δy';

            % Create Dy_f_editfield
            app.Dy_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Dy_f_editfield.Editable = 'off';
            app.Dy_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dy_f_editfield.Position = [225 97 51 22];

            % Create wavelengthumLabel_12
            app.wavelengthumLabel_12 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_12.HorizontalAlignment = 'right';
            app.wavelengthumLabel_12.Position = [188 69 25 22];
            app.wavelengthumLabel_12.Text = 'Δz';

            % Create Dz_f_editfield
            app.Dz_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.Dz_f_editfield.Editable = 'off';
            app.Dz_f_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dz_f_editfield.Position = [225 69 51 22];

            % Create wavelengthumLabel_20
            app.wavelengthumLabel_20 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_20.HorizontalAlignment = 'right';
            app.wavelengthumLabel_20.Position = [50 69 60 22];
            app.wavelengthumLabel_20.Text = 'Particle RI';

            % Create RI_f_editfield
            app.RI_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.RI_f_editfield.ValueChangedFcn = createCallbackFcn(app, @RI_f_editfieldValueChanged, true);
            app.RI_f_editfield.Position = [117 69 43 22];
            app.RI_f_editfield.Value = 1.4496;

            % Create GPUSwitchLabel
            app.GPUSwitchLabel = uilabel(app.ForwardmodeTab);
            app.GPUSwitchLabel.HorizontalAlignment = 'center';
            app.GPUSwitchLabel.Position = [360 34 31 22];
            app.GPUSwitchLabel.Text = 'GPU';

            % Create GPUSwitch
            app.GPUSwitch = uiswitch(app.ForwardmodeTab, 'slider');
            app.GPUSwitch.Position = [284 36 45 20];
            app.GPUSwitch.Value = 'On';

            % Create wavelengthumLabel_28
            app.wavelengthumLabel_28 = uilabel(app.ForwardmodeTab);
            app.wavelengthumLabel_28.HorizontalAlignment = 'right';
            app.wavelengthumLabel_28.Position = [7 97 103 22];
            app.wavelengthumLabel_28.Text = 'particle ρ (kg/m^3)';

            % Create rho_f_editfield
            app.rho_f_editfield = uieditfield(app.ForwardmodeTab, 'numeric');
            app.rho_f_editfield.Limits = [0 Inf];
            app.rho_f_editfield.ValueChangedFcn = createCallbackFcn(app, @rho_f_editfieldValueChanged, true);
            app.rho_f_editfield.Position = [117 97 43 22];
            app.rho_f_editfield.Value = 1850;

            % Create InversemodeTab
            app.InversemodeTab = uitab(app.TabGroup);
            app.InversemodeTab.Title = 'Inverse mode';
            app.InversemodeTab.ButtonDownFcn = createCallbackFcn(app, @InversemodeTabButtonDown, true);

            % Create TrapfreqkHzLabel_2
            app.TrapfreqkHzLabel_2 = uilabel(app.InversemodeTab);
            app.TrapfreqkHzLabel_2.Position = [73 177 89 22];
            app.TrapfreqkHzLabel_2.Text = 'Trap freq. (kHz)';

            % Create wavelengthumLabel_13
            app.wavelengthumLabel_13 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_13.BackgroundColor = [1 1 1];
            app.wavelengthumLabel_13.HorizontalAlignment = 'right';
            app.wavelengthumLabel_13.Position = [59 156 53 22];
            app.wavelengthumLabel_13.Text = 'Ωx / (2π)';

            % Create Wx_i_editfield
            app.Wx_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Wx_i_editfield.Limits = [0 Inf];
            app.Wx_i_editfield.ValueChangedFcn = createCallbackFcn(app, @Wx_i_editfieldValueChanged, true);
            app.Wx_i_editfield.Position = [119 156 43 22];
            app.Wx_i_editfield.Value = 280;

            % Create wavelengthumLabel_14
            app.wavelengthumLabel_14 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_14.BackgroundColor = [1 1 1];
            app.wavelengthumLabel_14.HorizontalAlignment = 'right';
            app.wavelengthumLabel_14.Position = [59 126 53 22];
            app.wavelengthumLabel_14.Text = 'Ωy / (2π)';

            % Create Wy_i_editfield
            app.Wy_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Wy_i_editfield.Limits = [0 Inf];
            app.Wy_i_editfield.ValueChangedFcn = createCallbackFcn(app, @Wy_i_editfieldValueChanged, true);
            app.Wy_i_editfield.Position = [119 126 43 22];
            app.Wy_i_editfield.Value = 270;

            % Create wavelengthumLabel_15
            app.wavelengthumLabel_15 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_15.HorizontalAlignment = 'right';
            app.wavelengthumLabel_15.Position = [59 96 53 22];
            app.wavelengthumLabel_15.Text = 'Ωz / (2π)';

            % Create Wz_i_editfield
            app.Wz_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Wz_i_editfield.Limits = [0 Inf];
            app.Wz_i_editfield.ValueChangedFcn = createCallbackFcn(app, @Wz_i_editfieldValueChanged, true);
            app.Wz_i_editfield.Position = [119 96 43 22];
            app.Wz_i_editfield.Value = 94;

            % Create wavelengthumLabel_17
            app.wavelengthumLabel_17 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_17.HorizontalAlignment = 'right';
            app.wavelengthumLabel_17.Position = [251 96 25 22];
            app.wavelengthumLabel_17.Text = 'NA';

            % Create NA_i_editfield
            app.NA_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.NA_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.NA_i_editfield.Position = [280 96 38 22];
            app.NA_i_editfield.Value = 0.84;

            % Create wavelengthumLabel_18
            app.wavelengthumLabel_18 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_18.HorizontalAlignment = 'right';
            app.wavelengthumLabel_18.Position = [185 35 91 22];
            app.wavelengthumLabel_18.Text = 'Cir. pol ratio (%)';

            % Create circpol_i_editfield
            app.circpol_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.circpol_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.circpol_i_editfield.Position = [284 35 34 22];
            app.circpol_i_editfield.Value = 100;

            % Create wavelengthumLabel_19
            app.wavelengthumLabel_19 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_19.HorizontalAlignment = 'right';
            app.wavelengthumLabel_19.Position = [207 67 72 22];
            app.wavelengthumLabel_19.Text = 'Power (mW)';

            % Create power_i_editfield
            app.power_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.power_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.power_i_editfield.Position = [281 67 37 22];
            app.power_i_editfield.Value = 250;

            % Create wavelengthumLabel_24
            app.wavelengthumLabel_24 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_24.HorizontalAlignment = 'right';
            app.wavelengthumLabel_24.Position = [52 65 60 22];
            app.wavelengthumLabel_24.Text = 'Particle RI';

            % Create RI_i_editfield
            app.RI_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.RI_i_editfield.ValueChangedFcn = createCallbackFcn(app, @RI_i_editfieldValueChanged, true);
            app.RI_i_editfield.Position = [119 65 43 22];
            app.RI_i_editfield.Value = 1.4496;

            % Create SelectLabel
            app.SelectLabel = uilabel(app.InversemodeTab);
            app.SelectLabel.HorizontalAlignment = 'right';
            app.SelectLabel.Position = [205 161 51 22];
            app.SelectLabel.Text = 'Select  λ';

            % Create SelectWavelength
            app.SelectWavelength = uidropdown(app.InversemodeTab);
            app.SelectWavelength.Items = {'1064 nm', '1550 nm'};
            app.SelectWavelength.ClickedFcn = createCallbackFcn(app, @SelectWavelengthClicked, true);
            app.SelectWavelength.Position = [177 140 82 22];
            app.SelectWavelength.Value = '1064 nm';

            % Create FWHMumLabel_2
            app.FWHMumLabel_2 = uilabel(app.InversemodeTab);
            app.FWHMumLabel_2.Position = [335 117 70 22];
            app.FWHMumLabel_2.Text = 'FWHM (um)';

            % Create wavelengthumLabel_25
            app.wavelengthumLabel_25 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_25.HorizontalAlignment = 'right';
            app.wavelengthumLabel_25.Position = [329 95 25 22];
            app.wavelengthumLabel_25.Text = 'Δx';

            % Create Dx_i_editfield
            app.Dx_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Dx_i_editfield.Editable = 'off';
            app.Dx_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dx_i_editfield.Position = [361 95 43 22];

            % Create wavelengthumLabel_26
            app.wavelengthumLabel_26 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_26.HorizontalAlignment = 'right';
            app.wavelengthumLabel_26.Position = [329 65 25 22];
            app.wavelengthumLabel_26.Text = 'Δy';

            % Create Dy_i_editfield
            app.Dy_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Dy_i_editfield.Editable = 'off';
            app.Dy_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dy_i_editfield.Position = [361 65 43 22];

            % Create wavelengthumLabel_27
            app.wavelengthumLabel_27 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_27.HorizontalAlignment = 'right';
            app.wavelengthumLabel_27.Position = [329 35 25 22];
            app.wavelengthumLabel_27.Text = 'Δz';

            % Create Dz_i_editfield
            app.Dz_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.Dz_i_editfield.Editable = 'off';
            app.Dz_i_editfield.BackgroundColor = [0.9412 0.9412 0.9412];
            app.Dz_i_editfield.Position = [361 35 43 22];

            % Create wavelengthumLabel_29
            app.wavelengthumLabel_29 = uilabel(app.InversemodeTab);
            app.wavelengthumLabel_29.HorizontalAlignment = 'right';
            app.wavelengthumLabel_29.Position = [9 34 103 22];
            app.wavelengthumLabel_29.Text = 'particle ρ (kg/m^3)';

            % Create rho_i_editfield
            app.rho_i_editfield = uieditfield(app.InversemodeTab, 'numeric');
            app.rho_i_editfield.Limits = [0 Inf];
            app.rho_i_editfield.ValueChangedFcn = createCallbackFcn(app, @rho_i_editfieldValueChanged, true);
            app.rho_i_editfield.Position = [119 34 43 22];
            app.rho_i_editfield.Value = 1850;

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = calc_psf_trapfreq

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end