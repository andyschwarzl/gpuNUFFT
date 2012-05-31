function [m, p] = gridkb(d,k,w,n,osf,wg,sw,opt)
% function m = grid3D(d,k,w,n,osf,kw,opt)
%
%     d -- k-space data
%     k -- k-trajectory, scaled -0.5 to 0.5
%     w -- k-space weighting
%     n -- image size (m will be osf*n X osf*n)
%     osf -- oversampling factor (usually between 1 and 2)
%     wg -- full kernel width in oversampled grid samples (usually 3 to 7)
%     sw -- sector width
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
k = k(:);
w = w(:);

% width of the kernel on the original grid
kw = wg/osf;

% preweight
dw = d.*w;

% compute kernel, assume e1 is 0.001, assuming nearest neighbor
kosf = floor(0.91/(osf*1e-3));

% half width in oversampled grid units
kwidth = osf*kw/2;

% beta from the Beatty paper
beta = pi*sqrt((kw*(osf-0.5)).^2-0.8);

% compute kernel
om = [0:kosf*kwidth]/(kosf*kwidth);
p = besseli(0,beta*sqrt(1-om.*om));
p = p./p(1);
% last sample is zero so we can use min() below for samples bigger than kwidth
p(end) = 0;

% convert k-space samples to matrix indices
nx = (n*osf/2+1) + osf*n*real(k);
ny = (n*osf/2+1) + osf*n*imag(k);

m = zeros(osf*n,osf*n);

'START precomputation'
data = [real(dw(:))'; imag(dw(:))'];
coords = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];

[test_coords, test_sector_centers,sector_dim] = assign_sectors(n,sw,coords);

[v i] = sort(test_coords);
v = v +1;
sectors_test = zeros(1,sector_dim+1);
cnt = 0;
for b=1:sector_dim+1
    while (cnt < length(v) && b == v(cnt+1))
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

size(m);
m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));

% zero out data at edges, which is probably due to data outside mtx
%m(:,1) = 0; m(:,osf*n) = 0;
%m(1,:) = 0; m(osf*n,:) = 0;
flipud(m(:,:,ceil(n/2)+1))
% stop here, if we just want the k-space data
if strcmp(opt,'k-space') return; end;

%im = fftshift(fft2(fftshift(m)));
%im = fftshift(m);
%m = im;
if strcmp(opt,'deappo') return; end;
im = m(:,:,ceil(n/2)+1);
% compute deappodization function
x = [-osf*n/2:osf*n/2-1]/(n);
sqa = sqrt(pi*pi*kw*kw*x.*x-beta*beta);
dax = sin(sqa)./(sqa);
% normalize by DC value
dax(osf*n/2)
dax(osf*n/2+1)
dax = dax/dax(osf*n/2+1);
% make it a 2D array
da = dax'*dax;
% deappodize
im = im./da
figure, imshow(abs(flipud(da)),[]);
%return the result
m = im;

