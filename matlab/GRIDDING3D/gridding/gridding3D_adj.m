function ress = gridding3D_adj(a,bb)
kspace_data_dim = size(bb,2);

if (kspace_data_dim > 1)
    kspace = bb(a.data_ind,:);
    data = [real(kspace(:))'; imag(kspace(:))'];

    disp('multiple coil data passed');
    kspace = reshape(data,[2 a.params.trajectory_length kspace_data_dim]);

    disp('call gridding mex kernel');
    if a.atomic == true
        disp('using atomic operations');
        m = mex_gridding3D_adj_atomic_f(single(kspace),single(a.coords),int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
    else
        m = mex_gridding3D_adj_f(single(kspace),single(a.coords),int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
    end;
    size(m)
    m = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
    ress = m;
else
    %prepare data
    kspace = bb(a.data_ind);
    data = [real(kspace(:))'; imag(kspace(:))'];

    % preweight, DCF
    %dw = d.*w;

    % performs the normal nufft
    disp('call gridding mex kernel')
    if a.atomic == true
        m = mex_gridding3D_adj_atomic_f(single(data),single(a.coords),int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
    else
        m = mex_gridding3D_adj_f(single(data),single(a.coords),int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
    end
    size(m)
    m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));
    ress = m;
end
