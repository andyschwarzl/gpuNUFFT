function data = inversegrid_singlecoil(img, FT, nIntl,nRO)
% 
% Inverse gpuNUFFT wrapper for  single coil data

[nx,ny,nz]=size(img);

data = FT*img;
data=reshape(data,nIntl,nRO);