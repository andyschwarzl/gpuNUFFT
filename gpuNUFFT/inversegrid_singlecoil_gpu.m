function data = inversegrid_singlecoil_gpu(img, FT, nIntl,nRO)
% 
% Inverse gpuNUFFT gridding wrapper for  single coil data

data = FT*img;
data=reshape(data,nIntl,nRO);