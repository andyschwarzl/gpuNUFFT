function ress = gpuNUFFT_adj(a,bb)
% ress = gpuNUFFT_adj(a,bb)
% Performs adjoint gpuNUFFT 
% from k-space to image space 
%
% supports multi-channel data
%
% a  ... GpuNUFFT Operator
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
    warning('gpuNUFFT:adj:sens',['Only single channel data passed? Channel count of k-space data (', num2str(nChn), ') does not fit sense data channel count (', num2str(a.sensChn), '). Sens data will not be applied automatically. Please pass k-space data in correct dimensions.']);
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
    disp('call gpuNUFFT mex kernel');
end

if a.atomic == true
    ress = mex_gpuNUFFT_adj_atomic_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(sens),a.params);
else
    ress = mex_gpuNUFFT_adj_f(single(kspace),(a.dataIndices),single(a.coords),(a.sectorDataCount),(a.sectorProcessingOrder),(a.sectorCenters(:)),single(a.densSorted),single(sens),a.params);
end;

% generate complex output from split vector
if (a.params.is2d_processing)
  if (nChn > 1)
    ress = squeeze(ress(1,:,:,:) + 1i*(ress(2,:,:,:)));
  else
    ress = squeeze(ress(1,:,:) + 1i*(ress(2,:,:)));
  end
else
  if (nChn > 1)
    ress = squeeze(ress(1,:,:,:,:) + 1i*(ress(2,:,:,:,:)));
  else
    ress = squeeze(ress(1,:,:,:) + 1i.*(ress(2,:,:,:)));
  end
end
