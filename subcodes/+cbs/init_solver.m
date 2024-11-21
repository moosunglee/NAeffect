classdef init_solver
% cbs abstract class representing Convergent Born Series
%
% Most quantities have SI dimensions.  Any SI units can be used for
% these quantities (for example m or microns) as long as the units
% are consistent.

    properties (SetAccess=protected)
        use_GPU
        NA
        wavelength0
        RI_bg
        pitch
        FOV
        dimension
        dataformat
        utility
    end

    methods
        function solver = init_solver(varargin)
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
            solver.utility = cbs.util.gen_utility(solver);
        end
    end

end