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
    m = mex_gridding3D_adj_atomic_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(a.sens),a.params);
else
    m = mex_gridding3D_adj_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(a.sens),a.params);
end;

% generate complex output from split vector
if (a.params.is2d_processing)
    ress = squeeze(m(1,:,:,:) + 1i*(m(2,:,:,:)));
else
    ress = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
end

