function img = regrid_multicoil_gpu(data, FT)
% 
% RegpuNUFFT wrapper 3D multicoil data
% 

[nIntl,nRO,nCh]=size(data);
data = reshape(data,[nIntl*nRO, nCh]);
img = FT'*data;
% for ii = 1:nCh
%     img(:,:,:,ii) = FT'*data(:,ii);
% end
