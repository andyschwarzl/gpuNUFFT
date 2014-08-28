function img = regrid_singlecoil_gpu(data, FT)
% 
% gpuNUFFT regridding wrapper 3D singlecoil data
% 

img = FT'*data(:);
