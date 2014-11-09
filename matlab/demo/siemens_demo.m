%% Reconstruction of 3D radial vibe data
clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));

%% Data parameters
N=256;

[x,y] = meshgrid(linspace(-0.5,0.5,N),linspace(-0.5,0.5,N));
v = linspace(0,1,N);
v = v'*v;
figure,imshow(v);
figure,surf(x,y,v);

k = [col(x)'; col(y)'];
w = col(ones(size(v)));
%%

disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; wg = 3; sw = 10;
imwidth = N;
FT = gpuNUFFT(k',w,osf,wg,sw,[N,N],[],false,false,false);

%% Forward and adjoint transform
tic
test = FT*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb = (FT'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);

% Display
figure;
subplot(1,2,1); imshow(abs(img_comb(:,:)),[]); title('Regridding');
subplot(1,2,2); imshow(abs(v(:,:)),[]); title('orig');

%%
figure, plot(abs(img_comb(6,:)))

%% fft referenz
ref_k = fft2(v);
ref_img = ifftshift(ifft2(reshape(ifftshift(test),[N N])));
figure, imshow(abs(ref_img),[])
figure, plot(abs(ref_img(6,:)))
