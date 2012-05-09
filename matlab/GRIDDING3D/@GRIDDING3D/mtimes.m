function ress = mtimes(a,bb)
kspace_data_dim = size(bb,2);
if (kspace_data_dim > 1)
    kspace = bb(a.op.data_ind,:);
    data = [real(kspace(:))'; imag(kspace(:))'];
    'multiple coil data passed'
    kspace = reshape(data,[2 a.params.trajectory_length kspace_data_dim]);
    'call gridding mex kernel'
    
    %data_ind = repmat(a.op.data_ind,[1 1 kspace_data_dim]);
    m = cuda_mex_kernel(single(kspace),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
    size(m)
    m = squeeze(m(1,:,:,:,:) + 1j*(m(2,:,:,:,:)));
    
    %crop data 
    %ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    %ind_start = ind_off;
    %ind_end = ind_start + a.params.im_width -1;
    %ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end,:);
    ress = m;
else
    %prepare data
    kspace = bb(a.op.data_ind);
    data = [real(kspace(:))'; imag(kspace(:))'];
    
    % preweight, DCF
    %dw = d.*w;

    % performs the normal nufft
    'call gridding mex kernel'
    m = cuda_mex_kernel(single(data),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);

    size(m)
    m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));

    %crop data 
    ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
    ind_start = ind_off;
    ind_end = ind_start + a.params.im_width -1;
    ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end);
end
