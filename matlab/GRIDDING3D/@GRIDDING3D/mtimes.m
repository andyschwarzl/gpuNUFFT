function ress = mtimes(a,bb)
if (a.adjoint)
    kspace_data_dim = size(bb,2);
    if (kspace_data_dim > 1)
        kspace = bb(a.op.data_ind,:);
        data = [real(kspace(:))'; imag(kspace(:))'];
        
        disp('multiple coil data passed');
        kspace = reshape(data,[2 a.params.trajectory_length kspace_data_dim]);

        disp('call gridding mex kernel');
        m = mex_gridding3D_adj_f(single(kspace),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
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
        m = mex_gridding3D_adj_f(single(data),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);

        size(m)
        m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));
        ress = m;
    end
else
    imspace = bb;
    imdata = [real(imspace(:))'; imag(imspace(:))'];
   
    n_chnls = size(bb,4);
    size(imdata)
    imdata = reshape(imdata,[2 a.params.im_width*a.params.im_width*a.params.im_width n_chnls]);
        
    if (n_chnls > 1)
        disp('multiple channel image data passed');
    end    
    disp('call forward gridding mex kernel');

    data = mex_gridding3D_forw_f(single(imdata),single(a.op.coords),int32(a.op.sector_data_cnt),int32(a.op.sector_centers),a.params);
    
    %put data in correct order
    %data_test = zeros(1,length(a.op.data_ind));
    disp(['returned data dimensions:' size(data)]);
    if (n_chnls > 1)
        data = squeeze(data(1,:,:) + 1j*(data(2,:,:)));
        data_test = zeros([length(a.op.data_ind),n_chnls]);
        data_test(a.op.data_ind,:) = data;
        ress = data_test;
    else
        data = squeeze(data(1,:) + 1j*(data(2,:)));
        data_test = zeros(1,length(a.op.data_ind));
        data_test(a.op.data_ind) = data;
        ress = transpose(data_test);
    end
    return;
end
