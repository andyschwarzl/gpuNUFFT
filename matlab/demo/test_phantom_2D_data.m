%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath ../../bin  
addpath data
addpath(genpath('../GRIDDING3D'));
addpath(genpath('../utils'));
%% Load data
load sl2d

%% generate Fourier sampling operator
osf = 1.25;
wg = 3;
sw = 8;
imwidth = N;
%%
tic
FT = GRIDDING3D(k_traj',dens,imwidth,osf,wg,sw,[imwidth imwidth],'false');
toc

%% recon
imgRecon = FT'*dataRadial(:);

%% show results
figure, imshow(imrotate(abs(imgRecon(:,:)),90),[]), title('Recon');
