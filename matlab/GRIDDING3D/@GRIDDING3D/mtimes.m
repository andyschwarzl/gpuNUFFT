function ress = mtimes(a,bb)
kspace_data_dim = size(bb,2);
if (kspace_data_dim > 1)
    data = [real(bb(:))'; imag(bb(:))'];
    'multiple coil data passed'
    kspace = reshape(data,[2 a.params.trajectory_length kspace_data_dim]);
    'call gridding mex kernel'
    
    data_ind = repmat(a.op.data_ind,[1 1 kspace_data_dim]);
    tic
    m = cuda_mex_kernel(single(kspace(data_ind)),single(a.op.coords(a.op.coord_ind)),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
    toc
    size(m)
    m = squeeze(m(1,:,:,:,:) + 1j*(m(2,:,:,:,:)));
    
    %crop data 
    ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    ind_start = ind_off;
    ind_end = ind_start + a.params.im_width -1;
    ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end,:);
else
    %prepare data
    data = [real(bb(:))'; imag(bb(:))'];
    % preweight, DCF
    %dw = d.*w;

    % performs the normal nufft

    'call gridding mex kernel'
    tic
    m = cuda_mex_kernel(single(data(a.op.data_ind)),single(a.op.coords(a.op.coord_ind)),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
    toc

    size(m)
    m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));

    %crop data 
    ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    ind_start = ind_off;
    ind_end = ind_start + a.params.im_width -1;
    ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end);
end
