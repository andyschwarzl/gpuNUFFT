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
useGPU = false;
%% generate Fourier sampling operator
osf = 1.25;
wg = 3;
sw = 8;
imwidth = N;
%%
tic
if useGPU
    FT = GRIDDING3D(k_traj',dens',imwidth,osf,wg,sw,imgDim,'false');
else
    FT = NUFFT3D(k_traj, dens, 1, 0, imgDim, 2,1);
end
toc

%% recon
tic
imgRecon = FT'*dataRadial(:);
toc
%% show results
figure, imshow((abs(imgRecon(:,:,N/2))),[]), title('Recon');
%%
show3DImage([4,8],abs(imgRecon(:,:,N3D/2-15:N3D/2+16)),'test','slice');