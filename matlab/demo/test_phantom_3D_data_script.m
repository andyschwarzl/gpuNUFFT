%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath ../../bin  
addpath ..
addpath data
addpath(genpath('../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));
addpath(genpath('../utils'));
%% Load d12
load sl3d128
%load sl3d64
%load sl3d64
N = imgDim(1);
N3D = imgDim(3);
useGPU = true;
%% generate Fourier sampling operator
osf = 1.5;
wg = 3;
sw = 8;
imwidth = N;
R = 1;
k_traj = k_traj(1:R:end,:);
dataRadial=dataRadial(1:R:end,:);
dens = dens(1:R:end);
%% debug
display('GPU init')
tic
FT = gpuNUFFT(k_traj',dens',imwidth,osf,wg,sw,imgDim,[],'true');
toc
display('CPU init')
tic
FTCPU = NUFFT3D(k_traj, dens, 1, 0, imgDim, 2,1);
toc

%% recon
display('GPU forward')
tic
imgRecon = FT'*dataRadial(:);
toc
display('CPU forward')
tic
imgReconCPU = FTCPU'*dataRadial(:);
toc
%% show results
%% central 8 slices
save3DImage([2,8],abs(imgRecon(:,:,N3D/2-7:N3D/2+8)),'GPU','slice');
save3DImage([2,8],abs(imgReconCPU(:,:,N3D/2-7:N3D/2+8)),'CPU','slice');
%% central 32
save3DImage([4,8],abs(imgRecon(:,:,N3D/2-15:N3D/2+16)),'GPU','slice');
save3DImage([4,8],abs(imgReconCPU(:,:,N3D/2-15:N3D/2+16)),'CPU','slice');

%% apply forward operation
display('CPU forward and inverse')
tic
data_reconCPU = FTCPU*imgReconCPU;
imgRecon2CPU = FTCPU'*data_reconCPU;
toc

display('GPU forward and inverse')
tic
data_recon = FT*imgRecon;
imgRecon2 = FT'*data_recon;
toc

save3DImage([2,8],abs(imgRecon2CPU(:,:,N3D/2-7:N3D/2+8)),'CPU2','slice');
save3DImage([2,8],abs(imgRecon2(:,:,N3D/2-7:N3D/2+8)),'GPU2','slice');
