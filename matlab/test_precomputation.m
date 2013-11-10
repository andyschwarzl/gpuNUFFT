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
size(img)

show3DImageasArray([4 4],img,'input image','slice ');
[nPE,nFE,nCh]=size(img);

%% Generate k space trajectory
% 1D test
X=linspace(-0.5,0.5,6);
Y=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5];
Z=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5];

% 2D test
X=linspace(-0.5,0.5,6);
Y=[-0.5 -0.5 0 0 0.5 0.45];
Z=[-0.5 -0.5 -0.5 -0.5 -0.5 -0.5];

% 3D test
X=linspace(-0.5,0.5,6);
Y=[-0.5 -0.5 0 0 0.5 0.45];
Z=[-0.33, -0.16666,   0,   0, -0.23, 0.45];

k_traj = [X(:),Y(:),Z(:)];
%%
imwidth = nPE;
osf = 1.5;
wg = 3;
sw = 8;
w = ones(1,length(k_traj(:)));
global temp
FT = GRIDDING3D(k_traj,w,imwidth,osf,wg,sw,[imwidth imwidth imwidth],'false');

%% generate radial data
dataRadial = (FT*img);
dataRadial_x = (1/sqrt(prod([16 16 16]))) * dataRadial;
%% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
w = w(:);
w = repmat(w, [1 im_size]);
%%
dataRadial_dc = dataRadial;

%% recon
imgRegrid_kb = FT'*dataRadial_dc;
show3DImageasArray([4 4],imgRegrid_kb,'gridding reconstruction','fft slice ');