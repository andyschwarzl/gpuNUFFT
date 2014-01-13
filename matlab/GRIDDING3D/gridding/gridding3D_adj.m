function ress = gridding3D_adj(a,bb)
%init gpu device
%gpuDevice();

kspace_data_dim = size(bb,2);

if (kspace_data_dim > 1)
    kspace = bb(:,:);%bb(a.data_ind,:);%v2
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
        m = mex_gridding3D_adj_atomic_f(single(kspace),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorCenters),single(a.densSorted),a.params);
    else
        m = mex_gridding3D_adj_f(single(kspace),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorCenters),single(a.densSorted),a.params);
    end;
    size(m);
    m = squeeze(m(1,:,:,:,:) + 1i*(m(2,:,:,:,:)));
    ress = m;
else
    %prepare data
    kspace = bb;%bb(a.data_ind);%v2
    kspace = [real(kspace(:))'; imag(kspace(:))'];

    % preweight, DCF
    %dw = d.*w;
    %data = data .* repmat(single(a.density_comp)',[2 1]);
    % performs the normal nufft
    if a.verbose
        disp('call gridding mex kernel')
    end
    if a.atomic == true
        m = mex_gridding3D_adj_atomic_f(single(kspace),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorCenters),single(a.densSorted),a.params);
    else
        m = mex_gridding3D_adj_f(single(kspace),uint64(a.dataIndices),single(a.coords),uint64(a.sectorDataCount),uint64(a.sectorCenters),single(a.densSorted),a.params);
    end
    size(m);
    m = squeeze(m(1,:,:,:) + 1j*(m(2,:,:,:)));
    ress = m;
end
