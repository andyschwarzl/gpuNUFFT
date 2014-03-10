% 2d data test case and 2d reconstruction
%
clear all; 
close all; clc;

%% add bin to path
addpath ../bin  
addpath ../../cuda-test/gridding3D/daten
addpath(genpath('./GRIDDING3D'));
addpath(genpath('./utils'));
%% Load data
load img_brain_4ch;

img = img(:,:,:);
n_chn = 4;
size(img);

[nPE,nFE,nCh]=size(img);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-1j*theta);

%% generate Fourier sampling operator
%FT = GRIDDING3D(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))'];
imwidth = nPE;
osf = 1.5;
wg = 3;
sw = 8;
% density compensation
w = repmat(abs(rho), [1, numSpokes]);
tic
FT = GRIDDING3D(k_traj,w(:),imwidth,osf,wg,sw,[nPE nPE],'false');
toc
%% generate radial data
display(['inverse gridding: ']);
tic
dataRadial = inversegrid_multicoil_gpu(img,FT,2*nPE,numSpokes);
toc

%% recon
display(['gridding: ']);
tic
imgRegrid_kb_dc = regrid_multicoil_gpu(reshape(dataRadial,[size(k),n_chn]),FT);
toc

%% show results
%  merge channels
recon_sos_dc = sqrt(sum(abs(imgRegrid_kb_dc).^2,3));
recon_sos_res = recon_sos_dc(:,:);
figure, imshow(abs(recon_sos_res(:,:)),[]);
title('gridding dc sos');
    