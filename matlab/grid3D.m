function [m, p] = grid3D(d,k,w,n,osf,wg,sw,opt)
% function m = grid3D(d,k,w,n,osf,kw,sw,opt)
%
%     d -- k-space data
%     k -- k-trajectory, scaled -0.5 to 0.5
%     w -- k-space weighting
%     n -- image size (m will be osf*n X osf*n)
%     osf -- oversampling factor (usually between 1 and 2)
%     wg -- full kernel width in oversampled grid samples (usually 3 to 7)
%     sw -- sector width to use
%     opt -- 'k-space', 'image', defaults it 'image' if not specified
%
%     m -- gridded k-space data
%     p -- gridding kernel, optional
%
%  Uses optimum Kaiser-Bessel window for a given
%    oversampling factor and kernel size
%  Now uses Phil's numbers
%
%  extend from John Pauly, 2003, 2005, 2007, 2011
%  (c)Board of Trustees, Leland Stanford Junior University
%
%  A. Schwarzl, Graz University of Technology

if nargin < 7,
    opt = 'image';
end
    
% convert to single column
d = d(:);
%k = k(:);
w = w(:);

% preweight, DCF
dw = d.*w;
p = 0;
m = zeros(osf*n,osf*n);

%START precomputation
data = [real(dw(:))'; imag(dw(:))'];
coords = k;%[real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];

[test_coords, test_sector_centers,sector_dim] = assign_sectors(osf*n,sw,coords);

[v i] = sort(test_coords);
v = v +1;
sectors_test = zeros(1,sector_dim+1);
cnt = 0;
for b=1:sector_dim+1
    while (cnt < length(v) && b == int32(v(cnt+1)))
        cnt = cnt +1;
    end
    sectors_test(b)=cnt;
end
%sectors_test
%% calculate indices of data elements in order to sort them
data_ind = i-1;
data_ind=[2*data_ind+1;2*data_ind+2];

%% calculate indices of coord elements in order to sort them
coord_ind = i-1;
coord_ind = [3*coord_ind+1;
             3*coord_ind+2;
             3*coord_ind+3];
%%
test_sector_centers = int32(reshape(test_sector_centers,[3,sector_dim]));

%% init params
params.im_width = uint32(n);
params.osr = single(osf);
params.kernel_width = uint32(wg);
params.sector_width = uint32(sw);

%%
'call gridding mex kernel'
tic
m = cuda_mex_kernel(single(data(data_ind)),single(coords(coord_ind)),int32(sectors_test),int32(test_sector_centers),params);
toc

size(m)
m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));

% zero out data at edges, which is probably due to data outside mtx
%m(:,1) = 0; m(:,osf*n) = 0;
%m(1,:) = 0; m(osf*n,:) = 0;
%flipud(m(:,:,ceil(n/2)+1))

% stop here, if we just want the k-space data
if strcmp(opt,'k-space') return; end;

im = fftshift(m);
m = im;

ind_off = (n * (osf-1) / 2) + 1;
ind_start = ind_off;
ind_end = ind_start + n -1;
m = m(ind_start:ind_end,ind_start:ind_end,ind_start:ind_end);

