% 2d data test case and 2d reconstruction
%
clear all; 
close all; clc;

%% add bin to path
addpath(genpath('../gpuNUFFT/'));
addpath(genpath('./utils'));
addpath(genpath('../../fessler/NUFFT'));
%% Load data
load img_brain_4ch;

img = img(:,:,:);
n_chn = 4;
size(img);

[nPE,nFE,nCh]=size(img);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Radial Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-1j*theta);
imwidth = nPE;
osf = 1.25;
wg = 3;
sw = 8;

% cartesian Trajectory
[x,y] = meshgrid(rho,rho);
k_traj_cart = [x(:)'; y(:)'];
w = ones(size(x));
w = w(:);
FT = gpuNUFFT(k_traj_cart,w(:),osf,wg,sw,[nPE nPE],[],false);
%% generate Fourier sampling operator
%FT = gpuNUFFT(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))'];

%%
% density compensation
w = repmat(abs(rho), [1, numSpokes]);
%%
tic
FT = gpuNUFFT(k_traj,w(:),osf,wg,sw,[nPE nPE],[],false);
toc
FTCPU = NUFFT(k_traj(1,:)+1i*k_traj(2,:), w(:), 1, 0, [nPE nPE], 2);
%% generate radial data
display(['inverse gpuNUFFT: ']);
tic
dataRadial = inversegrid_multicoil_gpu(img,FT,2*nPE,numSpokes);
toc
dataRadialCPU = inversegrid_multicoil(img,FTCPU,2*nPE,numSpokes);
%% recon
display(['gpuNUFFT: ']);
tic
imgRegrid_kb_dc = regrid_multicoil_gpu(reshape(dataRadial,[size(k),n_chn]),FT);
toc

imgRegridCPU = regrid_multicoil(reshape(dataRadial,[size(k),n_chn]),FTCPU);
%% show results
%  merge channels
recon_sos_dc = sqrt(sum(abs(imgRegrid_kb_dc).^2,3));
recon_sos_res = recon_sos_dc(:,:);

recon_sos_dcCPU = sqrt(sum(abs(imgRegridCPU).^2,3));
recon_sos_resCPU = recon_sos_dcCPU(:,:);

figure;
subplot(121);
imshow(abs(recon_sos_res(:,:)),[]);
title('gpuNUFFT dc sos');

subplot(122);
imshow(abs(recon_sos_resCPU(:,:)),[]);
title('gpuNUFFT CPU');
    