%% testscript with operator usage
clear all; close all; clc;

%% add bin to path
addpath ../bin
addpath ../../daten
addpath(genpath('./GRIDDING3D'));

%% Load data
load img_brain_4ch;
trimmed_size = 64;
img = img(128-trimmed_size/2+1:128+trimmed_size/2,128-trimmed_size/2+1:128+trimmed_size/2,:);
img_a = repmat(img(:,:,1),[1 1 trimmed_size]);
%img_a(1:10,1:10,:) = 0;
%img_a = flipdim(img_a,1);
%img_a = padarray(img(:,:,1),[0 0 trimmed_size/2]);
%%
size(img_a)
figure, imshow(imresize(abs(img_a(:,:,1)),4),[]), title('gridding input');

[nPE,nFE,nCh]=size(img_a);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-j*theta);

%% generate Fourier sampling operator
%FT = GRIDDING3D(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
imwidth = nPE;
osf = 2;
wg = 3;
sw = 8;
w = ones(1,length(k(:)));
FT = gridding3D(k_traj,w,imwidth,osf,wg,sw,'deappo');

%% generate radial data
dataRadial = FT*img_a;
%% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
dataRadial_dc = dataRadial.*w(:);
%% recon
%no density compnesation
imgRegrid_kb = FT'*dataRadial;
%density compensated
imgRegrid_kb_dc = FT'*dataRadial_dc;

%% show results
figure, imshow(imresize((abs(imgRegrid_kb_dc(:,:,1))),4),[]), title('gridding dc');
