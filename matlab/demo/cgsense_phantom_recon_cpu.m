%% Reconstruction of 3D radial vibe data
% Last change: Jan 2018
% By: Florian Knoll (florian.knoll@nyumc.org)
%
% Note: the useMultiCoil flag that includes coil sensitivities in the
% operator is only implemented for the gpuNUFFT, not for the CPU reference


clear all; close all; clc; clear classes;

addpath(genpath('./utils'));

%% Data parameters
N=160;
nSl=N;
nFE=207;
nCh=8;
disp_slice=nSl/2;
useMultiCoil = 0;

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
FT = NUFFT3D(k, col(sqrt(w)), 1, 0, [N,N,N], 2);

%% Forward and adjoint transform
tic
for cc=1:nCh
    img(:,:,:,cc) = FT'*rawdata(:,cc);
end
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);

tic
for cc=1:nCh
    test(:,cc) = FT*img(:,:,:,cc);
end
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);

%% Estimate sensitivities
disp('Estimate coil sensitivities.');
% Terribly crude, but fast
img_sos = sqrt(sum(abs(img).^2,4));
senseEst = img./repmat(img_sos,[1,1,1,nCh]);

% Use this instead for more reasonable sensitivitities, but takes some time
% for ii=1:nSl
%     disp(['Slice ', num2str(ii), '/', num2str(nSl)]);
%     [~,senseEst(:,:,ii,:)]=adapt_array_2d(squeeze(img_sens(:,:,ii,:)));
% end

%% CGSENSE Reconstruction
mask = 1;
tic
img_cgsense = cg_sense_3d(rawdata,FT,senseEst,mask,alpha,tol,maxitCG,display,disp_slice,useMultiCoil);
timeCG = toc;
disp(['Time CG SENSE: ', num2str(timeCG), ' s']);

%% Display
figure;
subplot(1,2,1); imshow(abs(img_sos(:,:,disp_slice)),[]); title('Regridding');
subplot(1,2,2); imshow(abs(img_cgsense(:,:,disp_slice)),[]); title('CGSENSE');
