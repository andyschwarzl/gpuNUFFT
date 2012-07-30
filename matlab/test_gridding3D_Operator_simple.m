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

show3DImageasArray([4 4],img,'input image','slice ');
[nPE,nFE,nCh]=size(img);

%% check fft output
Xk = fftn(img);
%Xk(:,:,1:8) = 0;
%Xk(:,:,10:16) = 0;

img_fft = ifftn(Xk);

show3DImageasArray([4 4],img_fft,'fft reconstruction','fft slice ');
%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-j*theta);

%% generate Fourier sampling operator
%FT = GRIDDING3D(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
z_crds = linspace(-0.5,0.5,im_size);
z_crds_rep = repmat(z_crds,[length(k(:)) 1]);
test = repmat(k_traj,[1 1 im_size]);
%%
for slice = 1:im_size
    test(3,:,slice) = z_crds_rep(:,slice);
end

k_traj_3d = test(:,:);
%%
imwidth = nPE;
osf = 1.5;
wg = 3;
sw = 8;
w = ones(1,length(k_traj_3d(:)));
FT = gridding3D(k_traj_3d,w,imwidth,osf,wg,sw,'false');

%% generate radial data
dataRadial = (FT*img);
dataRadial_x = (1/sqrt(prod([16 16 16]))) * dataRadial;
%% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
w = w(:);
w = repmat(w, [1 im_size]);
dataRadial_dc = dataRadial.*w(:);

%% recon
imgRegrid_kb = FT'*dataRadial_dc;
imgRegrid_kb_x = (1/sqrt(prod([16 16 16]))) * imgRegrid_kb;

show3DImageasArray([4 4],imgRegrid_kb,'gridding reconstruction','fft slice ');