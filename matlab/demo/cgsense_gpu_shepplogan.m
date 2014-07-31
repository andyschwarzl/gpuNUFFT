%% CG-SENSE demo using 3D GPU Regridding Operators
% Last change: 10/07/2013
% Florian Knoll (florian.knoll@nyumc.org), NYU School of Medicine

clear all; close all; clc;

addpath(genpath('../../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));
addpath('data');

%% Data parameters
N=160;
% nSl=N/2;
nSl=N;
nFE=207;
nCh=8;
disp_slice=nSl/2;
useGPU = true;
%% Reconstruction parameters
maxit = 5;
alpha = 1e-6;
tol = 1e-6;
display = 0;

%% Load data
load rawdata_phantom_regridding.mat;
[nPE,nFE,nCh]=size(rawdata);

%% Regridding operator GPU
osf = 2; wg = 3; sw = 8;
imwidth = N;

if (useGPU)
    FT = gpuNUFFT(k',w,osf,wg,sw,[N,N,nSl]);
else
    FT = NUFFT3D(k, w, 1, 0, [N,N,nSl], 2,1);
end
F = @(y) inversegrid_singlecoil_gpu(y,FT,nPE,nFE);
Fh = @(x) regrid_singlecoil_gpu(x,FT);
            
%% Regrid
tic
for ii=1:nCh
    img(:,:,:,ii) = Fh(rawdata(:,:,ii));
    fprintf('.');
end
disp(['Time: ', num2str(toc), ' s']);

% Inversegrid
tic
for ii=1:nCh
    rawdata_inversegrid(:,:,ii) = F(img(:,:,:,ii));
    fprintf('.');
end
disp(['Time: ', num2str(toc), ' s']);

%% Estimate Sensitivities
disp('-------------------------');
disp('Estimating Sensitivities Walsh method');
disp('-------------------------');
tic
senseEst = zeros(N,N,nCh,nSl); % Will be permuted later
for ii = 1:nSl
    [dummy,senseEst(:,:,:,ii)] = adapt_array_2d(squeeze(img(:,:,ii,:)));
end
senseEst=permute(senseEst,[1,2,4,3]);
disp(['Time: ', num2str(toc), ' s']);

useMulticoil = false;

if (useGPU && useMulticoil)
    % Operator has to be rebuilt with sensitivities
    FT = gpuNUFFT(k',w,osf,wg,sw,[N,N,nSl],senseEst);
    F  = @(y) FT*y;
    Fh = @(x) FT'*x;
    rawdata = reshape(rawdata,[nPE*nFE,nCh]);
end
%% Image reconstruction with CG SENSE
disp('-------------------------');
disp('Starting CG SENSE');
disp('-------------------------');
mask = 1;
tic

img_cgsense = cg_sense_3d(rawdata,F,Fh,senseEst,mask,alpha,tol,maxit,display,disp_slice,useMulticoil);
compTimeCGSENSE=toc;
disp(['Time: ', num2str(compTimeCGSENSE), ' s']);  

%% Display
if (display)
  img_sos=sqrt(sum(abs(img).^2,4));
  figure;
  subplot(1,2,1); imshow(img_sos(:,:,disp_slice),[]); title('Regridding');
  subplot(1,2,2); imshow(abs(img_cgsense(:,:,disp_slice)),[]); title('CGSENSE');
end
