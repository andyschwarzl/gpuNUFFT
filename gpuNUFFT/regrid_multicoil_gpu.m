function img = regrid_multicoil_gpu(data, FT, N)
% 
% Regridding wrapper 3D multicoil data
% 

[nIntl,nRO,nCh]=size(data);
img=single(zeros(N,N,N,nCh));
for ii = 1:nCh
    img(:,:,:,ii)=FT'*(col(data(:,:,ii)));
    fprintf('.');
end
