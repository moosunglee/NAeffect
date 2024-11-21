function solver_options = init_solver_options(varargin)
    p = inputParser;

    % 1. run_EH
    p.addParameter('boundary_thickness', [4, 4, 4]);
    p.addParameter('attenuation_width', [1, 1, 1]);
    p.addParameter('boundary_sharpness', 1);
    p.addParameter('radius_img', []);
    p.addParameter('radius_fft', []);
    p.addParameter('acyclic', true);
    p.addParameter('iterations', -1);
    p.addParameter('cut_bool', true)
    p.addParameter('xtol', 1e-6)
    p.addParameter('verbose', true);
    p.addParameter('verbose_num', 10);
    p.addParameter('crange', []);

    p.parse(varargin{:});
            
    solver_options = struct;

    % run_EH parameters
    solver_options.boundary_thickness = p.Results.boundary_thickness;
    solver_options.attenuation_width = p.Results.attenuation_width;
    solver_options.boundary_sharpness = p.Results.boundary_sharpness;
    solver_options.radius_img = p.Results.radius_img;
    solver_options.radius_fft = p.Results.radius_fft;
    solver_options.acyclic = p.Results.acyclic;
    solver_options.iterations = p.Results.iterations;
    solver_options.cut_bool = p.Results.cut_bool;
    solver_options.xtol = p.Results.xtol;
    solver_options.verbose = p.Results.verbose;
    solver_options.verbose_num = p.Results.verbose_num;
    solver_options.crange = p.Results.crange;


end