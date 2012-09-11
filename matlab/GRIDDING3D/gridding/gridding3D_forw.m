function ress = gridding3D_forw(a,bb)
imspace = bb;
imdata = [real(imspace(:))'; imag(imspace(:))'];

n_chnls = size(bb,4);
size(imdata);
imdata = reshape(imdata,[2 a.params.im_width*a.params.im_width*a.params.im_width n_chnls]);

if (n_chnls > 1)
    if a.verbose
        disp('multiple channel image data passed');
    end
end    
if a.verbose
    disp('call forward gridding mex kernel');
end
if a.atomic == true
    data = mex_gridding3D_forw_atomic_f(single(imdata),single(a.coords)',int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
else
    data = mex_gridding3D_forw_f(single(imdata),single(a.coords)',int32(a.sector_data_cnt),int32(a.sector_centers),a.params);
end
%put data in correct order
%data_test = zeros(1,length(a.op.data_ind));
if a.verbose
    disp(['returned data dimensions:' num2str(size(data))]);
end
if (n_chnls > 1)
    data = squeeze(data(1,:,:) + 1j*(data(2,:,:)));
    data_test = zeros([length(a.data_ind),n_chnls]);
    data_test(a.data_ind,:) = data;
    ress = data_test;
else
    data = squeeze(data(1,:) + 1j*(data(2,:)));
    data_test = zeros(1,length(a.data_ind));
    data_test(a.data_ind) = data;
    ress = transpose(data_test);
end
