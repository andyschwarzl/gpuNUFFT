%% Radial Regridding

clear all; close all; clc;

%% add bin to path
addpath ../bin

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
figure, plot(k), title(['k-space trajectory: ', num2str(numSpokes), ' projections']);

% Calculate density compensation function
w = abs(rho);
w = repmat(w, 1, numSpokes);

%% Perform Regridding with Kaiser Besser Kernel 64
osf = 1.25; % 1 or 2
wg = 5;  % 3 to 7
sw = 8;
imwidth = 64;
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
tic
[imgRegrid_kb,kernel] = grid3D(kspace,k_traj,w,imwidth,osf,wg,sw,'deappo');
toc
figure, imshow(imresize(abs((imgRegrid_kb(:,:,imwidth/2 +1))),4),[]);

%% MatlabTest_8SK3w32
osf=1
 kspace_test = ([0.0046-0.0021i]);
 wg = 3;
 k_test = ([0.25-0.4330i]);
 w_test = ([1]);
 
[imgRegrid_kb,kernel] = gridkb(kspace_test,k_test,w_test,32,osf,wg,'image');
 figure, imshow(abs(flipud(imgRegrid_kb)),[]);
%% Perform Regridding with Kaiser Besser Kernel 128
kspace2 = kspace;
osf = 1.5;
wg = 3;
sw = 8;
im_width = 64;
tic
[imgRegrid_kb,kernel] = gridkb(kspace2,k,w,im_width,osf,wg,sw,'deappo');
toc
figure, imshow(imresize(abs(imgRegrid_kb(:,:,im_width*osf/2+1)),4),[]);
%% Perform Regridding with Kaiser Besser Kernel 128, own deapo
kspace2 = kspace;
osf = 1;
wg = 3;
tic
[imgRegrid_kb,kernel] = gridkb(kspace2,k,w,128,osf,wg,'deappo');
toc

% alternative deappo
kspace_de = [1];
k_de = [0];
w_de = [1];
[deapo,kernel_deapo] = gridkb(kspace_de,k_de,w_de,128,osf,wg,'deappo');
figure, imshow(abs(flipud((deapo(:,:,65)))),[]);

imgRegrid_kb = imgRegrid_kb ./ abs(deapo);

figure, imshow(abs(fliplr((imgRegrid_kb(:,:,65)))),[]);

%% Test FFT output
 osf=2;
 kspace_test = ([1]);
 wg = 3;
 sw = 8;
 k_test = ([0]);
 w_test = ([1]);
 
[imgRegrid_kb,kernel] = gridkb(kspace_test,k_test,w_test,64,osf,wg,sw,'deappo');
figure, imshow(abs(flipud(imgRegrid_kb)),[]);

%% Kernel of Size 5 test
osf=1;
kspace_test = ([0.5+0.5i,0.7+1i,1+1i,1+1i,1+1i]);
wg = 5;
sw = 5;
k_test = ([-0.3+0.2i,-0.05001,0.02,0.5,0.3+0.3i]);
w_test = ([0,1,0,0,0]);
 
[imgRegrid_kb,kernel] = gridkb(kspace_test,k_test,w_test,10,osf,wg,sw,'deappo');
figure, imshow(log(abs(flipud(imgRegrid_kb(:,:,5)))),[]);