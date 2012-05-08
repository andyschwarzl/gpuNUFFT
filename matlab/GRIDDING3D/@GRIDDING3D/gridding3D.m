function [res] = gridding3D(k,w,n,osf,wg,sw,opt)
% function m = GRIDDING3D(d,k,w,n,osf,kw,sw,opt)
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
w = w(:);

p = 0;
m = zeros(osf*n,osf*n);


res.op = gridding3D_init(k,n,osf,sw);

res.params.im_width = uint32(n);
res.params.osr = single(osf);
res.params.kernel_width = uint32(wg);
res.params.sector_width = uint32(sw);
res.params.trajectory_length = length(k);
res.adjoint = 0;
%res.opt = opt;

res = class(res,'GRIDDING3D');
