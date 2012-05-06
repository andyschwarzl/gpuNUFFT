function ress = mtimes(a,bb)
%prepare data
data = [real(bb(:))'; imag(bb(:))'];
% preweight, DCF
%dw = d.*w;

% performs the normal nufft

'call gridding mex kernel'
tic
m = cuda_mex_kernel(single(data(a.op.data_ind)),single(a.op.coords(a.op.coord_ind)),int32(a.op.sectors_test),int32(a.op.test_sector_centers),a.params);
toc

size(m)
m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));

% stop here, if we just want the k-space data
%if strcmp(a.opt,'k-space') return; end;

%crop data 
ind_off = (a.params.im_width * (double(a.params.osr)-1) / 2) + 1;
ind_start = ind_off;
ind_end = ind_start + a.params.im_width -1;
ress = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end);