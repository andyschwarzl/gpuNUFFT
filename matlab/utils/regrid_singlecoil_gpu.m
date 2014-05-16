function img = regrid_singlecoil_gpu(data, FT)
% 
% gpuNUFFT wrapper 3D singlecoil data
% 

[nIntl,nRO]=size(data);
data = reshape(data,[nIntl*nRO,1]);
img = FT'*data;
