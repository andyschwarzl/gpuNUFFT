%% Reconstruction of 3D radial vibe data
clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));

%% Data parameters
N=256;

%smooth version
[x,y] = meshgrid(linspace(-0.5,0.5-1/N,N),linspace(-0.5,0.5-1/N,N));
k = [col(x)'; col(y)'];

%non-smooth version
[x,y] = meshgrid(linspace(-0.5,0.5,N),linspace(-0.5,0.5,N));
k_nonsmooth = [col(x)'; col(y)'];

v = linspace(0,1,N);
v = 1i*(v'*v);
figure,imshow(imag(v));title('test input');
figure,surf(x,y,imag(v));title('test input');

w = col(ones(size(v)));
%%
disp('Generate NUFFT Operator without coil sensitivities');
osf = 2;
kw = 1; 
sw = 8;
imwidth = N;

%k = k_nonsmooth;
%kw = 3; %nonsmooth

FT = gpuNUFFT(k',w,osf,kw,sw,[N,N],[],false);
FTCPU = NUFFT(k(1,:) + 1i*k(2,:),w,1,0,[N,N],2);

%% Forward and adjoint transform
tic
test = FT*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb = (FT'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);

%% Forward and adjoint transform NUFFT
tic
test = FTCPU*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb_CPU = (FTCPU'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);
%% fft reference
ref_k = fft2(v);
ref_img = (ifft2(reshape((ref_k),[N N])));

%% Compare results
figure;
h1=subplot(1,3,1); imshow(imag(img_comb),[]); title('Regridding');
h2=subplot(1,3,2); imshow(imag(img_comb_CPU),[]); title('Regridding NUFFT');
h3=subplot(1,3,3); imshow(imag(ref_img),[]);title('pure FFT Ref');
linkaxes([h1 h2 h3]);

figure;
h1=subplot(3,1,1);plot(imag(img_comb(6,:)));title('Line 6 - gpuNUFFT');
h2=subplot(3,1,2);plot(imag(img_comb_CPU(6,:)));title('Line 6 - NUFFT');
%h3=subplot(1,3,3);plot(imag(img_comb_CPU(6,:)./max(imag(col(img_comb_CPU(6,:))))-img_comb(6,:)./max(imag(col(img_comb(6,:)))))); title('Line 6 DIFF - NUFFT - gpuNUFFT');
h3=subplot(3,1,3);plot(imag(ref_img(6,:)));title('pure FFT Ref Line 6');

linkaxes([h1 h2 h3]);

figure,surf(x,y,double(imag(img_comb)));title('gpuNUFFT imag');
figure,surf(x,y,double(imag(img_comb_CPU)));title('NUFFT imag');
