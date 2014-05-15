function data = inversegrid_multicoil(img, FT, nIntl,nRO)
% 
% Inverse gpuNUFFT wrapper for multicoil data

[nx,ny,nz,nCh]=size(img);

for ii = 1:nCh
    data(:,ii) = FT*img(:,:,:,ii);
end
data=reshape(data,nIntl,nRO,nCh);