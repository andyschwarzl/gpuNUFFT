function img = regrid_multicoil_gpu(data, FT)
% 
% gpuNUFFT gridding wrapper 3D multicoil data
% 

[nIntl,nRO,nCh]=size(data);
data = reshape(data,[nIntl*nRO, nCh]);
img = FT'*data;
