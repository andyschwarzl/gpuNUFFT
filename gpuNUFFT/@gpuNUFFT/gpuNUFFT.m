function [res] = gpuNUFFT(k,w,osf,wg,sw,imageDim,sens,varargin)
% function m = gpuNUFFT(k,w,osf,wg,sw,imageDim,sens,varargin)
%
%     k -- k-trajectory, scaled -0.5 to 0.5
%          dims: 3 ... x, y and z
%                N ... # sample points
%                nCh ... # channels / coils
%     w -- k-space weighting, density compensation
%     osf -- oversampling factor (usually between 1 and 2)
%     wg -- kernel width (usually 3 to 7)
%     sw -- sector width to use
%     imageDim -- image dimensions [n n n] 
%     sens -- coil sensitivity data
%     varargin 
%        opt  -- true/false for atomic operation (default true)
%             -- true/false for using textures on gpu (default true)
%             -- true/false for balanced operation (default true)
%
%  res -- gpuNUFFT operator
%
%  A. Schwarzl, Graz University of Technology
%  F. Knoll, NYU School of Medicine
%

%t = gpuDevice;
%if eval(t.ComputeCapability) < 2.0
%    atomic = false;
%else
%    atomic = true;
%end

atomic = true;
use_textures = true;    
balance_workload = true;

if nargin < 7
    sens = [];
end
if nargin >= 8
    atomic = varargin{1};
    if nargin >= 9
        use_textures = varargin{2};
        if nargin >= 10
            balance_workload = varargin{3};
        end
    end
end

%check types of 
if ~islogical(atomic)
    error('gpuNUFFT:usage:atomic','Argument 8 (atomic) has to be of logical type.');
end

if ~islogical(use_textures)
    error('gpuNUFFT:usage:use_textures','Argument 9 (textures) has to be of logical type.');
end

if ~islogical(balance_workload)
    error('gpuNUFFT:usage:balance_workload','Argument 10 (balance_workload) has to be of logical type.');
end

%check input size of imageDims
if (length(imageDim) > 3)
    error('gpuNUFFT:init:imageDims','Image dimensions too large. Currently supported: 2d, 3d');
end

if (length(imageDim) < 3) 
    imageDim(3) = 0;
end

res.adjoint = 0;
res.imageDim = imageDim;

% adapt k space data dimension
% transpose to 3 x N x nCh    
if size(k,1) > size(k,2)
	warning('gpuNUFFT:init:kspace','k space data passed in wrong dimensions. Expected dimensions are 3 x N x nCh - automatic transposing is applied');
	k = k';
end

% convert to single col
w = w(:);    
if size(w,1) ~= size(k,2)
    warning('gpuNUFFT:init:density','density compensation dim does not match k space data dim. k: %s w: %s',num2str(size(k)),num2str(size(w)));
end

% check that sector width fits inside oversampled grid
if (sum(mod(imageDim*osf,sw))~=0)
    warning('gpuNUFFT:init:oversampling','GRID width [%.1f,%.1f,%.1f] (image width * OSR) is no integer multiple of sector width %d.\nTry to use integer multiples for best performance.',imageDim(1)*osf,imageDim(2)*osf,imageDim(3)*osf,sw)
end

res.op.params.img_dims = uint32(imageDim);
res.op.params.osr = single(osf);
res.op.params.kernel_width = uint32(wg);
res.op.params.sector_width = uint32(sw);
res.op.params.trajectory_length = uint32(length(k));
res.op.params.use_textures = use_textures;
res.op.params.balance_workload = balance_workload;
res.op.params.is2d_processing = imageDim(3) == 0;

[res.op.dataIndices,res.op.sectorDataCount,res.op.densSorted,res.op.coords,res.op.sectorCenters,res.op.sectorProcessingOrder] = mex_gpuNUFFT_precomp_f(single(k)',single(w)',res.op.params);
res.op.atomic = atomic;
res.op.verbose = false;

if ~isempty(sens) 
	if (res.op.params.is2d_processing)
			res.op.sensChn = size(sens,3);
	else
			res.op.sensChn = size(sens,4);
	end
	res.op.sens = [real(sens(:))'; imag(sens(:))'];
    res.op.sens = reshape(res.op.sens,[2 imageDim(1)*imageDim(2)*max(1,imageDim(3)) res.op.sensChn]);
else
	res.op.sens = sens;
	res.op.sensChn = 0;
end

if res.op.verbose
	test = res.op.sectorDataCount;
	test_cnt = test(2:end)-test(1:end-1);
	test_order = res.op.sectorProcessingOrder;
	figure;
	bar(test_cnt,'DisplayName','Workload per Sector');
	figure;
	bar(test_cnt(test_order(1,:)+1),'DisplayName','Workload per Sector ordered');figure(gcf)   
end

res = class(res,'gpuNUFFT');
