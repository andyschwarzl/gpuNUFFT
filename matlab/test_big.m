%% Radial Regridding

clear all; close all; clc;

%% add bin to path
addpath ../bin ../bin/Debug

%% Load data
load kspaceRadial.mat;

kspace = kspace(:,:);
%% Radial Sampling Pattern
[numSamplesOnSpoke, numSpokes] = size(kspace);
rho=linspace(-0.5,0.5,numSamplesOnSpoke)';
theta = [0:pi/numSpokes:pi-pi/numSpokes];
for ii = 1:length(theta)
    k(:,ii) = rho*exp(-j*theta(ii));
end
%figure, plot(k), title(['k-space trajectory: ', num2str(numSpokes), ' projections']);

% Calculate density compensation function
w = abs(rho);
w = repmat(w, 1, numSpokes);

%% Perform Regridding with Kaiser Besser Kernel
%osf = 1; % 1 or 2
%wg = 3;  % 3 to 7
%tic
%[imgRegrid_kb,kernel] = gridkb(kspace,k,w,64,osf,wg,'image');
%toc
%%
%figure, imshow(abs(flipud(imgRegrid_kb)),[]);

%% kleiner test
%kspace2 = ([-0.2+0.8i,-0.2+0.8i,0.5,1,0.7]);
kspace2 = kspace;
osf = 1;
wg = 3;
%k = ([0+0i, 0+0i, -0.3+0.2i, 0.5, -0.1]);
%w = ([1,1,1,1,1]);

tic
[imgRegrid_kb,kernel] = gridkb(kspace2,k,w,32,osf,wg,'image');
toc
%figure, imshow(abs(flipud(imgRegrid_kb)),[]);
          
