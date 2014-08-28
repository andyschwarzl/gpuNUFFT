%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath data
addpath(genpath('../../gpuNUFFT'));
addpath(genpath('./utils'));
addpath(genpath('../../../fessler/NUFFT'));
%% Load data
load sl2d

verbose = true;

%% generate Fourier sampling operator
osf =2;
wg = 3;
sw = 8;
imgDim = [256,256];
%
disp('init gpu op');
tic
FT = gpuNUFFT(k_traj',dens,osf,wg,sw,imgDim);
toc
disp('init cpu op');
tic
FTCPU = NUFFT(k_traj(:,1)+1i*k_traj(:,2), dens, 1, 0, imgDim, 2);
toc
%% recon
disp('GPU...');
imgRecon = FT'*dataRadial(:);
disp('GPU fw')
tic
data_recon = FT*imgRecon(:);
toc
disp('GPU adj')
tic
imgRecon2 = FT'*data_recon;
toc
%%
disp('CPU...');
imgReconCPU = FTCPU'*dataRadial(:);
disp('CPU fw')
tic
data_reconCPU = FTCPU*imgReconCPU(:);
toc
disp('CPU adj')
tic
imgRecon2CPU = FTCPU'*data_reconCPU;
toc

%% show results
if verbose
  figure, h1 = subplot(121); imshow(imrotate(abs(imgRecon(:,:)),90),[]), title('Recon');
  h2 = subplot(122); imshow(imrotate(abs(imgReconCPU(:,:)),90),[]), title('CPU Recon');
  linkaxes([h1 h2]);
  figure, h1 = subplot(121); imshow(imrotate(abs(imgRecon2(:,:)),90),[]), title('Recon2');
  h2 = subplot(122); imshow(imrotate(abs(imgRecon2CPU(:,:)),90),[]), title('CPU Recon2');
  linkaxes([h1 h2]);
end
