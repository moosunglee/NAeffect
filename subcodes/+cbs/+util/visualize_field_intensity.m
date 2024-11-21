function visualize_field_intensity(solver, E, varargin)
    % 'crange',[0 1], 'cmap', gray, 'center0', 0
    p = inputParser;
    p.addParameter('crange',[]);
    p.addParameter('cmap', hot);
    p.addParameter('center0', 0);
    p.addParameter('title', []);
    p.parse(varargin{:});
            
    h = struct;
    h.crange = p.Results.crange;
    h.cmap = p.Results.cmap;
    h.center0 = p.Results.center0;
    title0 = p.Results.title;


    figure,

    img = abs(E).^2;
    if solver.dimension == 1
        plot(solver.utility.image_space.coor{1},img), ylim(h.crange), 
        xlabel('x'), legend('Ey', 'Ex', 'Ez')
    elseif solver.dimension == 2
        imagesc(img(:,:)),axis image, colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end

        xlabel('x'), ylabel('y')
    else
        if length(h.center0) == 1
            h.center0 = h.center0.* ones(1,3);
        end
        img1 = squeeze(img(floor(end/2)+1 + h.center0(1),:,:));
        img2 = squeeze(img(:,floor(end/2)+1 + h.center0(2),:,:));
        img3 = squeeze(img(:,:,floor(end/2)+1 + h.center0(3),:));
        subplot(311), ...
            imagesc(img1(:,:)),...
            axis image, colormap(h.cmap),xlabel('z'),ylabel('x'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
        subplot(312), ...
            imagesc(img2(:,:)),...
            axis image, colormap(h.cmap),xlabel('z'),ylabel('y'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
        subplot(313), imagesc(img3(:,:)),...
            axis image, colormap(h.cmap),xlabel('x'),ylabel('y'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
    end
    set(gcf,'color','w')
    if ~isempty(title0)
        sgtitle(['xyz field (' title0 ')']);
    end

    figure,
    img = sum(abs(E).^2,solver.dimension+1);
    if solver.dimension == 1
        plot(solver.utility.image_space.coor{1},img), ylim(h.crange), 
        xlabel('x'), legend('Intensity')
    elseif solver.dimension == 2
        imagesc(img(:,:)),axis image, colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end

        xlabel('x'), ylabel('y')
    else
        if length(h.center0) == 1
            h.center0 = h.center0.* ones(1,3);
        end
        img1 = squeeze(img(floor(end/2)+1 + h.center0(1),:,:));
        img2 = squeeze(img(:,floor(end/2)+1 + h.center0(2),:,:));
        img3 = squeeze(img(:,:,floor(end/2)+1 + h.center0(3),:));
        subplot(131), ...
            imagesc(img1(:,:)),...
            axis image, colormap(h.cmap),xlabel('z'),ylabel('x'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
        subplot(132), ...
            imagesc(img2(:,:)),...
            axis image, colormap(h.cmap),xlabel('z'),ylabel('y'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
        subplot(133), imagesc(img3(:,:)),...
            axis image, colormap(h.cmap),xlabel('x'),ylabel('y'), colormap(h.cmap)
        if ~isempty(h.crange)
            clim(h.crange)
        end
    end
    set(gcf,'color','w')
    if ~isempty(title0)
        sgtitle(['Sum intensity (' title0 ')']);
    end
end