function data = inversegrid_multicoil_gpu(img, FT, nIntl, nRO)
% 
% Inverse gridding wrapper for multicoil data
% 

[~,~,~,nCh]=size(img);
data=single(zeros(nIntl*nRO,nCh));
for ii = 1:nCh
    data(:,ii) = FT*img(:,:,:,ii);
    fprintf('.');
end
data=reshape(data,nIntl,nRO,nCh);