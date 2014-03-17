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

useGPU = true;
%% generate Fourier sampling operator
osf = 1.25;
wg = 3;
sw = 8;
imwidth = N;
%%
tic
if useGPU
    FT = GRIDDING3D(k_traj',dens',imwidth,osf,wg,sw,[imwidth imwidth imwidth],'false');
else
    FT = NUFFT3D(k_traj, dens, 1, 0, [imwidth imwidth imwidth], 2,1);
end
toc

%% recon
tic
imgRecon = FT'*dataRadial(:);
toc
%% show results
figure, imshow((abs(imgRecon(:,:,N/2))),[]), title('Recon');
%%
show3DImage([6,6],imgRecon(:,:,N/2-18:N/2+18),'test','slice');