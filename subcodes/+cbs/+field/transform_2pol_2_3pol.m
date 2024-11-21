function fft_Field_3pol=transform_2pol_2_3pol(utility, fft_Field_2pol, use_abbe_sine)
%             Nsize = size(fft_Field_2pol); % MS LEE: Nsize -> obj.size
%%
    Nsize = utility.Nsize;
    fft_Field_2pol = fft_Field_2pol.*utility.NA_circle;
    %%
    if use_abbe_sine
        %abbe sine condition is due to the magnification
        filter=(utility.NA_circle);
        filter(utility.NA_circle)=filter(utility.NA_circle)./sqrt(utility.cos_theta(utility.NA_circle));
        fft_Field_2pol=fft_Field_2pol.*filter;
    end

    [Radial_2D,Perp_2D,ewald_TanVec,K_mask] = cbs.field.pol_parallel_transport(utility);
    %%
    fft_Field_2pol=fft_Field_2pol.*K_mask;

    dim = length(utility.Nsize);
    %%
    if dim == 3
        Field_new_basis=zeros(utility.Nsize(1),Nsize(2),2,'single');%the field in the polar basis
        Field_new_basis(:,:,1,:)=sum(fft_Field_2pol.*Radial_2D,3);
        Field_new_basis(:,:,2,:)=sum(fft_Field_2pol.*Perp_2D,3);
        
        fft_Field_3pol=zeros(Nsize(1),Nsize(2),3,'single');%the field in the 3D
        fft_Field_3pol         =fft_Field_3pol          + Field_new_basis(:,:,1,:).*ewald_TanVec;
        fft_Field_3pol(:,:,1:2,:)=fft_Field_3pol(:,:,1:2,:) + Field_new_basis(:,:,2,:).*Perp_2D;
    else
        Field_new_basis=zeros(Nsize(1),2,'single');%the field in the polar basis
        Field_new_basis(:,1,:)=sum(fft_Field_2pol.*Radial_2D,2);
        Field_new_basis(:,2,:)=sum(fft_Field_2pol.*Perp_2D,2);
        
        fft_Field_3pol=zeros(Nsize(1),3,'single');%the field in the 3D
        fft_Field_3pol         =fft_Field_3pol          + Field_new_basis(:,1,:).*ewald_TanVec;
        fft_Field_3pol(:,1:2,:)=fft_Field_3pol(:,1:2,:) + Field_new_basis(:,2,:).*Perp_2D;
    end
end