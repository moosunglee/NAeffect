function [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = pol_parallel_transport(utility)

    dim = length(utility.size);
    if dim == 3
        % utility to convert (r, theta) -> (x,y,z)
        K_1=utility.fourier_space.coor{1};
        K_2=utility.fourier_space.coor{2};
        K_3=utility.k3;
        K_mask = K_3 > 0;
        Nsize = utility.size;
        % need to consider k0_nm coefficient 
        Radial_2D =    zeros(Nsize(1),Nsize(2),2,'single');
        Perp_2D =      zeros(Nsize(1),Nsize(2),2,'single');
        Radial_3D =    zeros(Nsize(1),Nsize(2),3,'single');
        ewald_TanVec = zeros(Nsize(1),Nsize(2),3,'single');
        norm_rad=utility.fourier_space.coorxy;
    
        Radial_2D(:,:,1)=K_1./norm_rad;
        Radial_2D(:,:,2)=K_2./norm_rad;
        Radial_2D(K_1 == 0,K_2 == 0,:) = [1 0]; % define the center of radial
    
        Perp_2D(:,:,1)=Radial_2D(:,:,2);
        Perp_2D(:,:,2)=-Radial_2D(:,:,1);
    
        Radial_3D(:,:,1)=K_1/utility.k0_nm.*K_mask;
        Radial_3D(:,:,2)=K_2/utility.k0_nm.*K_mask;
        Radial_3D(:,:,3)=K_3/utility.k0_nm.*K_mask;
    
        ewald_TanProj=sum(Radial_3D(:,:,1:2).*Radial_2D,3);
        ewald_TanVec(:,:,1:2)=Radial_2D(:,:,:);
        ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
        ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,3));
        ewald_TanVec_norm(~K_mask)=1;
        ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;
    else
        % utility to convert (r, theta) -> (x,y,z)
        K_1=utility.fourier_space.coor{1};
        K_3=utility.k3;
        K_mask = K_3 > 0;
        Nsize = utility.size;
        % need to consider k0_nm coefficient 
        Radial_2D =    zeros(Nsize(1),2,'single');
        Perp_2D =      zeros(Nsize(1),2,'single');
        Radial_3D =    zeros(Nsize(1),3,'single');
        ewald_TanVec = zeros(Nsize(1),3,'single');
        norm_rad=utility.fourier_space.coorxy;
    
        Radial_2D(:,1)=K_1./norm_rad;
        Radial_2D(K_1 == 0,:) = [1 0]; % define the center of radial
    
        Perp_2D(:,1)=Radial_2D(:,2);
        Perp_2D(:,2)=-Radial_2D(:,1);
    
        Radial_3D(:,1)=K_1/utility.k0_nm.*K_mask;
        Radial_3D(:,3)=K_3/utility.k0_nm.*K_mask;
    
        ewald_TanProj=sum(Radial_3D(:,1:2).*Radial_2D,2);
        ewald_TanVec(:,1:2)=Radial_2D(:,:);
        ewald_TanVec=ewald_TanVec-ewald_TanProj.*Radial_3D;
        ewald_TanVec_norm=sqrt(sum(ewald_TanVec.^2,2));
        ewald_TanVec_norm(~K_mask)=1;
        ewald_TanVec=ewald_TanVec./ewald_TanVec_norm;
%%
    end
end