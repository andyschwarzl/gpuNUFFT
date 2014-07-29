function ress = gpuNUFFT_forw(a,bb)
% ress = gpuNUFFT_forw(a,bb)
% Performs forward gpuNUFFT 
% from image to k-space
%
% supports multi-channel data
%
% a  ... GpuNUFFT Operator
% bb ... image data
%        W x H x D x (nChn) for 3d 
%        W x H x (nChn)     for 2d  
%

%check if imgDims are 2d or 3d 
if (a.params.is2d_processing)
    nChn = size(bb,3);
else
    nChn = size(bb,4);
end

% split and prepare data
bb = [real(bb(:))'; imag(bb(:))'];
bb = reshape(bb,[2 a.params.img_dims(1)*a.params.img_dims(2)*max(1,a.params.img_dims(3)) nChn]);

if a.verbose
    if (nChn > 1)
        disp('multiple channel image data passed');
    end
 disp('call forward gpuNUFFT mex kernel');
end

% check if sens data (multichannel) is present and show
% warning if only single channel data is passed
% do not pass sens data in this case
sens = a.sens;
%if a.sensChn ~= 0 && ...
%   a.sensChn ~= nChn
%    warning('gpuNUFFT:forw:sens',['Image data dimensions (', num2str(size(bb)), ') do not fit sense data dimensions (', num2str(size(a.sens)), '). Sens data will not be applied. Please pass image data in correct dimensions.']);
%   sens = [];
%end
%TODO
if a.sensChn ~= 0 
    nChn = a.sensChn;
end

if a.atomic == true
    m = mex_gpuNUFFT_forw_atomic_f(single(bb),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(sens),a.params);
else
    m = mex_gpuNUFFT_forw_f(single(bb),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(sens),a.params);
end

if a.verbose
    disp(['returned data dimensions:' num2str(size(m))]);
end

if (nChn > 1)
    ress(:,:) = squeeze(m(1,:,:) + 1i*(m(2,:,:)));
else
    ress = squeeze(transpose(m(1,:) + 1i*(m(2,:))));
end

