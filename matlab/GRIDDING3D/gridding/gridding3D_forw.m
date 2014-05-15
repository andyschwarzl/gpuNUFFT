function ress = gridding3D_forw(a,bb)
% ress = gridding3D_forw(a,bb)
% Performs forward gridding 
% from image to k-space
%
% supports multi-channel data
%
% a  ... Gridding Operator
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
 disp('call forward gridding mex kernel');
end

% check if sens data (multichannel) is present and show
% warning if only single channel data is passed
% do not pass sens data in this case
sens = a.sens;
if a.sensChn ~= 0 && ...
   a.sensChn ~= nChn
    warning('GRIDDING3D:forw:sens',['Image data dimensions (', num2str(size(bb)), ') do not fit sense data dimensions (', num2str(size(a.sens)), '). Sens data will not be applied. Please pass image data in correct dimensions.']);
   sens = [];
end

if a.atomic == true
    data = mex_gridding3D_forw_atomic_f(single(bb),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(sens),a.params);
else
    data = mex_gridding3D_forw_f(single(bb),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(sens),a.params);
end

if a.verbose
    disp(['returned data dimensions:' num2str(size(data))]);
end

if (nChn > 1)
    data = squeeze(data(1,:,:) + 1j*(data(2,:,:)));
    ress(:,:) = data;
else
    data = transpose(squeeze(data(1,:) + 1j*(data(2,:))));
    ress = data;
end
