function [res] = gridding3D(k,w,n,osf,wg,sw,imageDim,varargin)
% function m = GRIDDING3D(d,k,w,n,osf,kw,sw,opt)
%
%     k -- k-trajectory, scaled -0.5 to 0.5
%     w -- k-space weighting
%     n -- image size (m will be osf*n X osf*n)
%     osf -- oversampling factor (usually between 1 and 2)
%     wg -- kernel width (usually 3 to 7)
%     sw -- sector width to use
%     opt -- 'sparse', E operator
%         -- 'atomic' (true/false)
%     res -- gridding operator
%
%  Uses optimum Kaiser-Bessel window for a given
%    oversampling factor and kernel size
%  Now uses Phil's numbers
%
%  extend from John Pauly, 2003, 2005, 2007, 2011
%  (c)Board of Trustees, Leland Stanford Junior University
%
%  A. Schwarzl, Graz University of Technology

if nargin <= 8,
    method = 'gridding';
    E = 0;
    atomic = eval(varargin{1});
elseif nargin > 8
    method = varargin{1};
    E = varargin{2};    
end

res.method = method;
res.adjoint = 0;
res.imageDim = imageDim;

if strcmp(method,'gridding')
    % convert to single column
    w = w(:);

    p = 0;
    m = zeros(osf*n,osf*n);

    res.op = gridding3D_init(k,n,osf,sw);

    res.op.params.im_width = uint32(n);
    res.op.params.osr = single(osf);
    res.op.params.kernel_width = uint32(wg);
    res.op.params.sector_width = uint32(sw);
    res.op.params.trajectory_length = uint32(length(k));
    res.op.atomic = atomic;
    %res.opt = opt;
elseif strcmp(method,'sparse')
    res.op = E;
end

res = class(res,'GRIDDING3D');
