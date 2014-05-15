function ress = gridding3D_adj(a,bb)
% ress = gridding3D_adj(a,bb)
% Performs adjoint gridding 
% from k-space to image space 
%
% supports multi-channel data
%
% a  ... Gridding Operator
% bb ... k-space data
%        k x nChn
%
nChn = size(bb,2);
 
% check if sens data (multichannel) is present and show
% warning if only single channel data is passed
% do not pass sens data in this case
sens = a.sens;
if a.sensChn ~= 0 && ...
   a.sensChn ~= nChn
    warning('GRIDDING3D:adj:sens',['k-Space data dimensions (', num2str(size(bb)), ') do not fit sense data dimensions (', num2str(size(a.sens)), '). Sens data will not be applied after gridding. Please pass k-space data in correct dimensions.']);
   sens = [];
end

%prepare data
if (nChn > 1)
    if a.verbose
        disp('multiple coil data passed');
    end
    kspace = bb(:,:);
    kspace = [real(kspace(:))'; imag(kspace(:))'];
    kspace = reshape(kspace,[2 a.params.trajectory_length nChn]);
else    
    kspace = bb;
    kspace = [real(kspace(:))'; imag(kspace(:))'];
end    
if a.verbose
    disp('call gridding mex kernel');
end

if a.atomic == true
    m = mex_gridding3D_adj_atomic_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(sens),a.params);
else
    m = mex_gridding3D_adj_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(sens),a.params);
end;

% generate complex output from split vector
if (a.params.is2d_processing)
    ress = squeeze(m(1,:,:,:) + 1i*(m(2,:,:,:)));
else
    ress = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
end

