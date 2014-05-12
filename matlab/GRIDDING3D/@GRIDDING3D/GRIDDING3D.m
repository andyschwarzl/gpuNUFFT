function [res] = gridding3D(k,w,n,osf,wg,sw,imageDim,sens,varargin)
% function m = GRIDDING3D(d,k,w,n,osf,kw,sw,imageDim,sens,varargin)
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
%             -- interpolationType  0,1,2,3 for interpolation type
%                        0 ... const kernel
%                        1 ... 1d texture lookup
%                        2 ... 2d texture lookup
%                        3 ... 3d texture lookup
%
%  res -- gridding operator
%
%  A. Schwarzl, Graz University of Technology

interpolation_type = 0;
if nargin <= 9,
    method = 'gridding';
    E = 0;
    atomic = eval(varargin{1});
elseif nargin > 9 && strcmp(varargin{1},'sparse') == 0
    method = 'gridding';
    E = 0;    
    atomic = eval(varargin{1});
    interpolation_type = varargin{2}
elseif nargin > 9 && strcmp(varargin{1},'sparse') == 1
    method = varargin{1};
    E = varargin{2};    
end

%check input size of imageDims
if (length(imageDim) > 3)
    error('GRIDDING3D:init:imageDims','Image dimensions too large. Currently supported: 2d, 3d');
end

if (length(imageDim) < 3) 
    imageDim(3) = 0;
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

%     res.op.params.im_width = uint32(imageDim(1));
%     res.op.params.im_height = uint32(imageDim(2));
%     res.op.params.im_depth = uint32(imageDim(3));
    res.op.params.img_dims = uint32(imageDim);
    res.op.params.osr = single(osf);
    res.op.params.kernel_width = uint32(wg);
    res.op.params.sector_width = uint32(sw);
    res.op.params.trajectory_length = uint32(length(k));
    res.op.params.interpolation_type = uint32(interpolation_type);
    res.op.params.balance_workload = true;
    res.op.params.is2d_processing = imageDim(3) == 0;
    
    [res.op.dataIndices,res.op.sectorDataCount,res.op.densSorted,res.op.coords,res.op.sectorCenters,res.op.sectorProcessingOrder] = mex_griddingND_precomp_f(single(k)',single(w)',res.op.params);
    res.op.atomic = atomic;
    res.op.verbose = false;
    res.op.sens = [real(sens(:))'; imag(sens(:))'];
    test = res.op.sectorDataCount;
    test_cnt = test(2:end)-test(1:end-1);
    test_order = res.op.sectorProcessingOrder;
    figure;
    bar(test_cnt,'DisplayName','Workload per Sector');
    figure;
    bar(test_cnt(test_order(1,:)+1),'DisplayName','Workload per Sector ordered');figure(gcf)   
elseif strcmp(method,'sparse')
    res.op = E;
end

res = class(res,'GRIDDING3D');
