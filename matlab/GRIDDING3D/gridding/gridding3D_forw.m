function ress = gridding3D_forw(a,bb)
%init gpu device
%gpuDevice();

%bb ... image data
n_chnls = size(bb,4);
size(bb);

bb = [real(bb(:))'; imag(bb(:))'];

bb = reshape(bb,[2 a.params.im_width*a.params.im_width*a.params.im_width n_chnls]);

if (n_chnls > 1)
    if a.verbose
        disp('multiple channel image data passed');
    end
end    
if a.verbose
    disp('call forward gridding mex kernel');
end
if a.atomic == true
    data = mex_gridding3D_forw_atomic_f(single(bb),uint64(a.dataIndices),single(a.coords)',uint64(a.sectorDataCount),uint64(a.sectorCenters),a.params);
else
    data = mex_gridding3D_forw_f(single(bb),uint64(a.dataIndices),single(a.coords)',uint64(a.sectorDataCount),uint64(a.sectorCenters),a.params);
end
%put data in correct order
%data_test = zeros(1,length(a.op.data_ind));
if a.verbose
    disp(['returned data dimensions:' num2str(size(data))]);
end
if (n_chnls > 1)
    data = squeeze(data(1,:,:) + 1j*(data(2,:,:)));
    %data_test = zeros([length(a.data_ind),n_chnls]);
    %data_test(a.data_ind,:) = data;
    %ress(a.data_ind,:) = data;%v2
    ress(:,:) = data;
else
    data = transpose(squeeze(data(1,:) + 1j*(data(2,:))));
    %data_test = zeros(1,length(a.data_ind));
    %data_test(a.data_ind) = data;
    %ress(a.data_ind) = data;%v2
    ress = data;
    ress = transpose(ress);
end
size(find(ress))