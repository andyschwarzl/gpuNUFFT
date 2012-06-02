function ress = mtimes(a,bb)
if (a.adjoint)
    kspace_data_dim = size(bb,2);
    if (kspace_data_dim > 1)
        kspace = bb(a.op.data_ind,:);
        data = [real(kspace(:))'; imag(kspace(:))'];
        
        disp('multiple coil data passed');
        kspace = reshape(data,[2 a.params.trajectory_length kspace_data_dim]);

        disp('call gridding mex kernel');
        m = cuda_mex_kernel(single(kspace),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
        size(m)
        m = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
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
        ress = m;
    end
else
    imspace = bb;
    imdata = [real(imspace(:))'; imag(imspace(:))'];
    n_coils = size(bb,4);
    size(imdata)
    disp('multiple coil data passed');
    imdata = reshape(imdata,[2 a.params.im_width*a.params.im_width*a.params.im_width n_coils]);
    disp('call forward gridding mex kernel');

    %data_ind = repmat(a.op.data_ind,[1 1 kspace_data_dim]);
    data = mex_gridding3D_forw(single(imdata),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
    size(data)
    data = squeeze(data(1,:) + 1j*(data(2,:)));
    
    %put data in correct order
    data_test = zeros(1,length(a.op.data_ind));
    data_test(a.op.data_ind) = data;
    ress = transpose(data_test);
    return;
end
