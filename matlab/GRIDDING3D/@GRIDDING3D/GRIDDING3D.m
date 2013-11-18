function [res] = gridding3D(k,w,n,osf,wg,sw,imageDim,varargin)
% function m = GRIDDING3D(d,k,w,n,osf,kw,sw,imageDim,varargin)
%
%     k -- k-trajectory, scaled -0.5 to 0.5
%          dims: 3 ... x, y and z
%                N ... # sample points
%                nCh ... # channels / coils
%     w -- k-space weighting, density compensation
%     n -- image size (m will be osf*n)
%     osf -- oversampling factor (usually between 1 and 2)
%     wg -- kernel width (usually 3 to 7)
%     sw -- sector width to use
%     imageDim -- image dimensions [n n n] 
%     varargin 
%        opt  -- 'sparse' + E operator or
%             -- 'true'/'false' for atomic operation
%     res -- gridding operator
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
    % adapt k space data dimension
    % transpose to 3 x N x nCh    
    if size(k,1) > size(k,2)
        warning('GRIDDING3D:init:kspace','k space data passed in wrong dimensions. Expected dimensions are 3 x N x nCh - automatic transposing is applied');
        k = k';
    end
    
    % convert to single col
    w = w(:);    
    if size(w,1) ~= size(k,2)
        warning('GRIDDING3D:init:density','density compensation dim does not match k space data dim. k: %s w: %s',num2str(size(k)),num2str(size(w)));
    end
    
    %res.op = gridding3D_init(k,n,osf,sw,w);

    
    res.op.params.im_width = uint32(n);
    res.op.params.osr = single(osf);
    res.op.params.kernel_width = uint32(wg);
    res.op.params.sector_width = uint32(sw);
    res.op.params.trajectory_length = uint32(length(k));
    
    [a,b,c,d,e] = mex_griddingND_precomp_f(single(k)',single(w)',[],res.op.params);
    a
    b
    c
    d
    e
    res.op.atomic = atomic;
    res.op.verbose = false;
elseif strcmp(method,'sparse')
    res.op = E;
end

res = class(res,'GRIDDING3D');
