clear all; close all; clc;

%% generate 2d phantom kSpace Data
N = 256;
R = 1;
radial = true;
[k_traj,dataRadial,dens] = create_phantom(N,radial,2);

save('./sl2d','dataRadial','k_traj','dens','N','R');
%% simple recon (cartesian sampling)
dataRadialR = reshape(dataRadial,[N,N]);
img = flipud(ifftshift(ifftn((dataRadialR))));
figure, imshow(abs(img),[]);

%% generate 3d phantom kSpace Data
N = 128;
R = 1;
radial = true;
[k_traj,dataRadial,dens] = create_phantom(N,radial,3,R);

save(['./sl3d' num2str(N)],'dataRadial','k_traj','dens','N','R');
%% simple recon (cartesian sampling)
dataRadialR = reshape(dataRadial,[N,N,N]);
img = (ifftshift(ifftn(fftshift(dataRadialR))));
%%
figure, imshow(abs(img(:,:,N/2)),[]);