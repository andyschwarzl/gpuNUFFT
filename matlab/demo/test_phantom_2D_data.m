%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath ../../bin  
addpath data
addpath(genpath('../gpuNUFFT'));
addpath(genpath('../utils'));
%% Load data
load sl2d

%% generate Fourier sampling operator
osf =1.25;
wg = 3;
sw = 8;
imwidth = imgDim(1);
%%
tic
FT = gpuNUFFT(k_traj',dens,imwidth,osf,wg,sw,imgDim,[],'false');
toc

%% recon
tic
imgRecon = FT'*dataRadial(:);
toc
%% show results
figure, imshow(imrotate(abs(imgRecon(:,:)),90),[]), title('Recon');
