clear all; close all; clc;
addpath '..'
%% generate 2d phantom kSpace Data
N = 256;
R = 1;
imgDim = [N N];
radial = true;
[k_traj,dataRadial,dens] = create_phantom(imgDim,radial);

save('./sl2d','dataRadial','k_traj','dens','R','imgDim');
%% simple recon (cartesian sampling)
dataRadialR = reshape(dataRadial,[N,N]);
img = flipud(ifftshift(ifftn((dataRadialR))));
figure, imshow(abs(img),[]);

%% generate 3d phantom kSpace Data
N = 64;
R = 1;
N3D = 32;
imgDim = [N N N3D];
radial = false;
[k_traj,dataRadial,dens] = create_phantom(imgDim,radial);
save(['./sl3d' num2str(N)],'dataRadial','k_traj','dens','R','imgDim');
%% simple recon (cartesian sampling)
dataRadialR = reshape(dataRadial,[N,N,N3D]);
img = (ifftshift(ifftn(fftshift(dataRadialR))));
%%
figure, imshow(abs(img(:,:,16)),[]);
%%
show3DImage([4,8],img(:,:,N3D/2-15:N3D/2+16),'test','t');