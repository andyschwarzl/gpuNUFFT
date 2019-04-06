%% Reconstruction of 3D radial vibe data
% Last change: Jan 18
% By: Florian Knoll (florian.knoll@nyumc.org)

clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));

%% Data parameters
N=160;
nSl=N;
nFE=207;
nCh=8;
disp_slice=nSl/2;
useGPU = true;
useMultiCoil = 1;

%% Reconstruction parameters
maxitCG = 10;
alpha = 1e-6;
tol = 1e-6;
display = 1;

%% Load data
load ./data/rawdata_phantom_regridding.mat;
[nPE,nFE,nCh]=size(rawdata);
rawdata = reshape(rawdata,[nPE*nFE,nCh]);

%% Regridding operator GPU without coil sensitivities for now
disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; wg = 3; sw = 8;
imwidth = N;
FT = gpuNUFFT(k',col(w(:,:,1)),osf,wg,sw,[N,N,nSl],[]);

for ii=1:nCh
    img_sens(:,:,:,ii) = FT'*(rawdata(:,ii) .* sqrt(col(w)));
end

%% Estimate sensitivities
disp('Estimate coil sensitivities.');
% Terribly crude, but fast
img_sens_sos = sqrt(sum(abs(img_sens).^2,4));
senseEst = img_sens./repmat(img_sens_sos,[1,1,1,nCh]);

% Use this instead for more reasonable sensitivitities, but takes some time
% for ii=1:nSl
%     disp(['Slice ', num2str(ii), '/', num2str(nSl)]);
%     [~,senseEst(:,:,ii,:)]=adapt_array_2d(squeeze(img_sens(:,:,ii,:)));
% end

%% Redefine regridding operator GPU including coil sensitivities
disp('Generate NUFFT Operator with coil sensitivities');
FT = gpuNUFFT(k',col(w(:,:,1)),osf,wg,sw,[N,N,nSl],senseEst);

%% Forward and adjoint transform
tic
img_comb = FT'*(rawdata .* sqrt(repmat(col(w),[1 nCh])));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);
% figure,imshow(abs(img_comb(:,:,disp_slice)),[]); title('Regridding');
% figure,kshow(abs(fft2c(img_comb(:,:,disp_slice)))); title('Regridding k-space');

tic
test = FT*img_comb;
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);

%% CGSENSE Reconstruction
mask = w;
tic
img_cgsense = cg_sense_3d(rawdata,FT,senseEst,mask,alpha,tol,maxitCG,display,disp_slice,useMultiCoil);
timeCG = toc;
disp(['Time CG SENSE: ', num2str(timeCG), ' s']);

%% Display
figure;
subplot(1,2,1); imshow(abs(img_comb(:,:,disp_slice)),[]); title('Regridding');
subplot(1,2,2); imshow(abs(img_cgsense(:,:,disp_slice)),[]); title('CGSENSE');
