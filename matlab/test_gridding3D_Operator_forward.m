%simple testfile to test the behavior of the gridding operator 
%generates a simple 16x16x16 image with 3 spots in the middle layer
clear all; close all; clc;

%% add bin to path
addpath ../bin
addpath ../../daten
addpath(genpath('./GRIDDING3D'));

%% create image data
im_size = 16;
img = zeros([im_size im_size im_size]);
img(9,9,9) = 1;
img(2,15,9) = 1;
img(4,4,9) = 1;

img(5,5,3) = 1;
img(7,7,3) = 1;
img(13,13,3) = 1;

size(img)
figure, imshow(imresize(abs(img(:,:,9)),4),[]), title('gridding input');

[nPE,nFE,nCh]=size(img);

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
dataRadial = FT*img;

%density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
dataRadial_dc = dataRadial.*w(:);
%% recon
imgRegrid_kb = FT'*dataRadial_dc;
figure, imshow(imresize(abs((imgRegrid_kb(:,:,1))),4),[]), title('gridding');
