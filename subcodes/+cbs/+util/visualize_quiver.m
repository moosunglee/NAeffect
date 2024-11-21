function visualize_quiver(solver, S, varargin)
    p = inputParser;
    p.addParameter('threshold',0.1);
    p.addParameter('title', []);
    p.addParameter('mag', 2);
    p.parse(varargin{:});
            
    threshold0 = p.Results.threshold;
    title0 = p.Results.title;
    mag0 =  p.Results.mag;

    S_int = sum(abs(S).^2, solver.dimension+1);
    S_int = S_int / max(S_int(:));

    list_threshold = S_int > threshold0;

    figure
    if solver.dimension == 3
        [II,JJ,KK] = ndgrid(solver.utility.image_space.coor{1},...
            solver.utility.image_space.coor{2}, ...
            solver.utility.image_space.coor{3});
        Y = II(list_threshold);
        X = JJ(list_threshold);
        Z = KK(list_threshold);
        SY = S(:,:,:,1);
        SX = S(:,:,:,2);
        SZ = S(:,:,:,3);
        SY = SY(list_threshold);
        SX = SX(list_threshold);
        SZ = SZ(list_threshold);
        quiver3(X,Y,Z,SX,SY,SZ,mag0);

    elseif solver.dimension == 2
        [II,JJ] = ndgrid(solver.utility.image_space.coor{1},...
            solver.utility.image_space.coor{2});
        Y = II(list_threshold);
        X = JJ(list_threshold);
        SY = S(:,:,1);
        SX = S(:,:,2);
        SY = SY(list_threshold);
        SX = SX(list_threshold);
        quiver(X,Y,SX,SY,mag0);
    else
        error('1D does not have a quiver plot.')
    end

    if ~isempty(title0)
        sgtitle(['Flux (' title0 ')']);
    end
    set(gcf,'color', 'w')

end