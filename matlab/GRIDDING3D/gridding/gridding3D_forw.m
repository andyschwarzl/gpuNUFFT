function ress = gridding3D_forw(a,bb)
% ress = gridding3D_forw(a,bb)
% Performs forward gridding 
% from image space to k-space
%
% supports multi-channel data
%
% a  ... Gridding Operator
% bb ... image data
%

%check if imgDims are 2d or 3d 
%size(bb)
if (a.params.is2d_processing)
    n_chnls = size(bb,3);
else
    n_chnls = size(bb,4);
end

bb = [real(bb(:))'; imag(bb(:))'];

%bb = reshape(bb,[2 prod(a.params.img_dims) n_chnls]);
bb = reshape(bb,[2 a.params.img_dims(1)*a.params.img_dims(2)*max(1,a.params.img_dims(3)) n_chnls]);
if (n_chnls > 1)
    if a.verbose
        disp('multiple channel image data passed');
    end
end    
if a.verbose
    disp('call forward gridding mex kernel');
end
if a.atomic == true
    data = mex_gridding3D_forw_atomic_f(single(bb),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorCenters(:)),a.params);
else
    data = mex_gridding3D_forw_f(single(bb),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorProcessingOrder),uint64(a.sectorCenters(:)),a.params);
end
%put data in correct order
%data_test = zeros(1,length(a.op.data_ind));
if a.verbose
    disp(['returned data dimensions:' num2str(size(data))]);
end
if (n_chnls > 1)
    data = squeeze(data(1,:,:) + 1j*(data(2,:,:)));
    ress(:,:) = data;
else
    data = transpose(squeeze(data(1,:) + 1j*(data(2,:))));
    ress = data;
end
