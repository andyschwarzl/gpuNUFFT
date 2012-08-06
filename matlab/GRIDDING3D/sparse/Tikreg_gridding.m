function res = Tikreg_gridding(A,b,varargin)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  res = Tikreg_gridding(A,b,varargin)
%
%  solves (A'*A + lambda^2*I)*x = A'*x
%
%  A - forward nuFTOperator
%  b - measurement as a column
%  varargin - variable number parameter value pairs
%      lambda - tikhonov reguoarization parameter
%      maxit - number of iterations
%      machine - 'cpu_double','cpu_float' or 'gpu_double','gpu_float'
%      tol - stopping tolerance
%      verbose - verbose output for gpu-implementation
%      single_coil - compute each coil data separately 
%  res - result
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fft([1 2 3 4]); % dummy fft to obtain fft license

% default parameters
working_precision = 1; % gpu_float
tol =  1e-5; 
it = 20;
lambda = 0.2;
verbose = 0;
single_coil = false;
adjoint = false;

for k=1:length(varargin)
    if ischar(varargin{k})
        keyword = varargin{k};
        switch keyword
            case 'tol'
                tol = varargin{k+1};
            case 'maxit'
                it = varargin{k+1};
            case 'lambda'
                lambda = varargin{k+1};
            case 'verbose'
                verbose = varargin{k+1};  
            case 'machine'
                if strcmp(varargin{k+1},'cpu_double'),
                    working_precision = 3;
                elseif strcmp(varargin{k+1},'cpu_float'),
                    working_precision = 4;
                elseif strcmp(varargin{k+1},'gpu_float'),
                    working_precision = 1;
                elseif strcmp(varargin{k+1},'gpu_double'),
                    working_precision = 2;   
                else
                    error('invalid machine type');
                end;
            case 'single_coil'
                single_coil = true;
            case 'adjoint'
                adjoint = true;
        end
    end
end
A = struct(A);
ipk = getfield(A,'nufftStruct');
ipk = struct(ipk);
if working_precision == 1,
    %devprops = gpuDevice;
    %devnum = devprops.Index;
    devnum = 0;
    %if not(devprops.CanMapHostMemory)
    %    error('CanMapHostMemory is false');
    %    return;
    %end;
    [idx weight  bp_vxidx bp_midx bp_weight] = tikreg_gridding_init(ipk,1);
   
    if (adjoint == true)
        display('starting gridding on GPU (single precision)');
    
        %b -> daten
        b_il(1,:) = real(b);
        b_il(2,:) = imag(b);
        
        b_coils = A.numCoils;
        if (single_coil == true)
            %single call per coil
            res = zeros(A.imageDim);
            for coil = 1 : A.numCoils,
                coil_start = 2 * (coil-1) * A.trajectory_length +1;
                coil_end = coil_start + 2  * A.trajectory_length - 1;
                %tmp = tikreg_gridding_gpu_f(single(b_il(coil_start:coil_end)),single(ipk.sn),single(A.imageDim),single(1),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
                tmp = mex_gridding3D_adj_sparse_f(single(b_il(coil_start:coil_end)),single(ipk.sn),single(A.imageDim),single(1),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
                tmp = squeeze(tmp(1,:,:,:) + 1i*tmp(2,:,:,:));

                res(:,:,:,coil) = tmp;
            end
        else
            %res = tikreg_gridding_gpu_f(single(b_il),single(A.imageDim),single(ipk.sn),single(b_coils),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
            res = mex_gridding3D_adj_sparse_f(single(b_il),single(A.imageDim),single(ipk.sn),single(b_coils),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
            
            res = squeeze(res(1,:,:,:,:) + 1i*res(2,:,:,:,:));
        end;
    else
        display('starting forward gridding on GPU (single precision)');
    
        %forward gridding single precision
        img = b;
       
        img_il(1,:,:,:,:) = real(img);
        img_il(2,:,:,:,:) = imag(img);
        
        %res = tikreg_gridding_forward_gpu_f(single(img_il),single(A.imageDim),single(ipk.sn),single(1),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
        res = mex_gridding3D_forw_sparse_f(single(img_il),single(A.imageDim),single(ipk.sn),single(A.numCoils),single(idx),single(weight),single(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,single([it lambda^2 devnum tol verbose]));
        
        res = squeeze(res(1,:) + 1i*res(2,:));
    end;
elseif working_precision == 2,
    %devprops = gpuDevice;
    %devnum = devprops.Index;
    devnum =0;
    %if not(devprops.CanMapHostMemory)
    %    error('CanMapHostMemory is false');
    %    return;
    %end;
    [idx weight  bp_vxidx bp_midx bp_weight] = tikreg_gridding_init(ipk,2);
    display('starting gridding on GPU (double precision)');
    res = tikreg_cg_reco_gpu_d(double(b_il),double(idx),double(weight),double(ipk.Kd), uint32(bp_vxidx), bp_midx, bp_weight,double([it lambda^2 devnum tol verbose]));
end;
fprintf('\n');









