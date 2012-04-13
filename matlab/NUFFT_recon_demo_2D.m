%% Script: Demo 2D NUFFT, 4 channel head data
clear all; close all; clc;

% 4 channel head coil image
addpath ../bin ../bin/Debug;

addpath(genpath('../../NUFFT_recon_demo_2D/NUFFT'));
% load data
load img_brain_4ch;
[nPE,nFE,nCh]=size(img);

% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-j*theta);

%% generate Fourier sampling operator
FT = NUFFT(k, 1, 1, 0, [nPE,nFE], 2);

% generate radial data
dataRadial = FT*img;

% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,nCh]);
dataRadial_dc = dataRadial.*w;

%% inverse NUFFT
% inverse NUFFT
%recon = FT'*dataRadial;
osf = 1;
wg = 3;

tic
[imgRegrid_kb,kernel] = gridkb(dataRadial,k,w,256,osf,wg,'image');
toc

figure, imshow(abs(flipud(imgRegrid_kb)),[]);
%%
recon_dc = FT'*(dataRadial_dc);

%% SOS combination
recon_orig = sqrt(sum(abs(img).^2,3));
recon_sos = sqrt(sum(abs(recon).^2,3));
recon_sos_dc = sqrt(sum(abs(recon_dc).^2,3));

% Display results
figure, imshow(abs(recon_orig),[]); title('Original Fully Sampled');
figure, imshow(abs(recon_sos),[]); title('NUFFT recon no density compensation');
figure, imshow(abs(recon_sos_dc),[]); title('NUFFT recon with density compensation');