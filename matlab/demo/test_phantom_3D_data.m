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
disp('init GPU')
tic
FT = GRIDDING3D(k_traj',dens',imwidth,osf,wg,sw,imgDim,'false',0);
toc
disp('init CPU')
tic
FTCPU = NUFFT3D(k_traj, dens, 1, 0, imgDim, 2,1);
toc

%% recon
disp('GPU')
tic
imgRecon = FT'*dataRadial(:);
toc
disp('CPU')
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

%% apply forward operation
disp('CPU fw')
tic
data_reconCPU = FTCPU*imgReconCPU;
toc

disp('CPU adj')
tic
imgRecon2CPU = FTCPU'*data_reconCPU;
toc
%%
disp('GPU fw')
tic
data_recon = FT*imgRecon;
toc
disp('GPU adj')
tic
imgRecon2 = FT'*data_recon;
toc
%%
show3DImage([2,8],abs(imgRecon2CPU(:,:,N3D/2-7:N3D/2+8)),'CPU result of generated data','slice');
show3DImage([2,8],abs(imgRecon2(:,:,N3D/2-7:N3D/2+8)),'GPU result of generated data','slice');