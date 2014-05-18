function img = regrid_singlecoil_gpu(data, FT)
% 
% Regridding wrapper 3D singlecoil data
% 

% [nIntl,nRO]=size(data);
% data = reshape(data,[nIntl*nRO,1]);
img = FT'*data(:);
