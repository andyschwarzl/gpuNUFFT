function ress = gridding3D_adj(a,bb)
% ress = gridding3D_forw(a,bb)
% Performs forward gridding 
% from k-space to image space 
%
% supports multi-channel data
%
% a  ... Gridding Operator
% bb ... k-space data
%
kspace_data_dim = size(bb,2);

if (kspace_data_dim > 1)
    kspace = bb(:,:);
    kspace = [real(kspace(:))'; imag(kspace(:))'];

    if a.verbose
        disp('multiple coil data passed');
    end
    kspace = reshape(kspace,[2 a.params.trajectory_length kspace_data_dim]);
    
    if a.verbose
        disp('call gridding mex kernel');
    end
    if a.atomic == true
        if a.verbose
            disp('using atomic operations');
        end
        m = mex_gridding3D_adj_atomic_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorCenters(:)),single(a.densSorted),a.params);
    else
        m = mex_gridding3D_adj_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),a.params);
    end;
    %size(m);
    if (a.params.is2d_processing)
        ress = squeeze(m(1,:,:,:) + 1i*(m(2,:,:,:)));
    else
        ress = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
    end
else
    %prepare data
    kspace = bb;
    kspace = [real(kspace(:))'; imag(kspace(:))'];

    % performs the normal nufft
    if a.verbose
        disp('call gridding mex kernel')
    end
    if a.atomic == true
        m = mex_gridding3D_adj_atomic_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorCenters(:)),single(a.densSorted),a.params);
    else
        m = mex_gridding3D_adj_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),a.params);
    end
    %size(m);
    if (a.params.is2d_processing)
        ress = squeeze(m(1,:,:,:) + 1i*(m(2,:,:,:)));
    else
        ress = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
    end
end
