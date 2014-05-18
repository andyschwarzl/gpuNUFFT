function data = inversegrid_multicoil_gpu(img, FT, nIntl,nRO)
% 
% Inverse gpuNUFFT wrapper for multicoil data

if (length(size(img))>3)
    [nx,ny,nz,nCh]=size(img);
else
    [nx,ny,nCh]=size(img);
end

%for ii = 1:nCh
%    data(:,ii) = FT*img(:,:,:,ii);
%end
data = FT * img;
data=reshape(data,nIntl,nRO,nCh);