%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath ../../bin  
addpath ..
addpath data
addpath(genpath('../GRIDDING3D'));
addpath(genpath('../../../fessler/NUFFT'));
addpath(genpath('../utils'));
%% Load data
load sl3d64
N = imgDim(1);
N3D = imgDim(3);
useGPU = true;
%% generate Fourier sampling operator
osf = 2;
wg = 3;
sw = 8;
imwidth = N;
R = 2;
k_traj = k_traj(1:R:end,:);
dataRadial=dataRadial(1:R:end,:);
dens = dens(1:R:end);
%% debug
%imwidth = N;
imgDim = [64,64,32];
N3D = imgDim(3);
imwidth = imgDim(1);
%%
tic
FT = GRIDDING3D(k_traj',dens',imwidth,osf,wg,sw,imgDim,'false');
toc
tic
FTCPU = NUFFT3D(k_traj, dens, 1, 0, imgDim, 2,1);
toc

%% recon
tic
imgRecon = FT'*dataRadial(:);
toc
tic
imgReconCPU = FTCPU'*dataRadial(:);
toc
%% show results
 figure,h1 = subplot(121);
 imshow((abs(imgRecon(:,:,N3D/2))),[]), title('Recon GPU');
 h2=subplot(122);
 imshow((abs(imgReconCPU(:,:,N3D/2))),[]), title('Recon CPU');
 linkaxes([h1 h2]);
%% central 8 slices
show3DImage([2,8],abs(imgRecon(:,:,N3D/2-7:N3D/2+8)),'GPU','slice');
show3DImage([2,8],abs(imgReconCPU(:,:,N3D/2-7:N3D/2+8)),'CPU','slice');
%% central 32
show3DImage([4,8],abs(imgRecon(:,:,N3D/2-15:N3D/2+16)),'GPU','slice');
show3DImage([4,8],abs(imgReconCPU(:,:,N3D/2-15:N3D/2+16)),'CPU','slice');
%%
show3DImage([8,8],abs(imgRecon(:,:,N3D/2-31:N3D/2+32)),'GPU','slice');
show3DImage([8,8],abs(imgReconCPU(:,:,N3D/2-31:N3D/2+32)),'CPU','slice');

