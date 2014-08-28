function data = inversegrid_multicoil_gpu(img, FT, nIntl,nRO)
% 
% Inverse gpuNUFFT gridding wrapper for multicoil data

if (length(size(img))>3)
    [nx,ny,nz,nCh]=size(img);
else
    [nx,ny,nCh]=size(img);
end

data = FT * img;
data=reshape(data,nIntl,nRO,nCh);