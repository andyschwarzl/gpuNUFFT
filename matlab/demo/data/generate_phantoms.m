clear all; close all; clc;

%% generate 2d phantom kSpace Data
N = 128;
[dataRadial,k_traj,dens] = create_phantom(N,false,2);

save('./sl2d','dataRadial','k_traj');
%% simple recon (cartesian sampling)
dataRadialR = reshape(dataRadial,[N,N]);
img = flipud(fftshift(ifftn((dataRadialR))));
figure, imshow(abs(img),[]);