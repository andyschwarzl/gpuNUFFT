function res = mtimes(a,bb)
if strcmp(a.method,'sparse')
    if (a.adjoint)
        bb = bb(:);
        res = Tikreg_gpuNUFFT(a.op,bb,'machine','gpu_float','adjoint');%, ,'single_coil''verbose',1);

        res = res .* conj(a.smaps);%TODO check
        res = res / sqrt(prod(a.imageDim));%TODO check
    else
        bb = bb .* a.smaps;%TODO check
        res = Tikreg_gpuNUFFT(a.op,bb,'machine','gpu_float','verbose','1');%, 'verbose',1);
        
        res = res / sqrt(prod(a.imageDim));%TODO check
    end
elseif strcmp(a.method,'gpuNUFFT')
    if (a.adjoint)
        bb = reshape(bb,[a.op.params.trajectory_length size(a.smaps,4)]);
        res = gpuNUFFT_adj(a.op,bb);
        %TODO check
        %offset = (a.op.params.im_width - size(a.smaps,3))/2;
        %res = res(:,:,offset+1:(offset+size(a.smaps,3)),:) .* conj(a.smaps(:,:,:,:));
        res = res(:,:,11:54,:);%TODO
        res = res .* conj(a.smaps);
        res = res / sqrt(prod(a.imageDim));
    else
        bb = bb .* a.smaps;
        bb = padarray(bb,[0 0 double((a.op.params.im_width-a.imageDim(3))/2) 0]);
        res = gpuNUFFT_forw(a.op,bb);
        res = res / sqrt(prod(a.imageDim));        %TODO check
    end
end