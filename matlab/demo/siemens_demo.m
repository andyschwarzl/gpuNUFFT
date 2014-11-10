%% Reconstruction of 3D radial vibe data
clear all; close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));

%% Data parameters
N=256;

[x,y] = meshgrid(linspace(-0.5,0.5-1/N,N),linspace(-0.5,0.5-1/N,N));
k = [col(x)'; col(y)'];

[x,y] = meshgrid(linspace(-0.5,0.5,N),linspace(-0.5,0.5,N));
k_nonsmooth = [col(x)'; col(y)'];

v = linspace(0,1,N);
v = 1i*(v'*v);
figure,imshow(imag(v));
figure,surf(x,y,imag(v));

w = col(ones(size(v)));
%%

disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; wg = 5; sw = 8;
imwidth = N;
FT = gpuNUFFT(k',w,osf,wg,sw,[N,N],[]);
FT_nonsmooth = gpuNUFFT(k_nonsmooth', w, osf, wg, sw, [N,N],[]);
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

% Display
figure;
subplot(1,2,1); imshow(imag(img_comb(:,:)),[]); title('Regridding');
subplot(1,2,2); imshow(abs(v(:,:)),[]); title('orig');
figure, plot(imag(img_comb(6,:)));title('Line 6 - gpuNUFFT');
figure,surf(x,y,double(imag(img_comb)));title('gpuNUFFT imag');
%% Forward and adjoint transform NUFFT
tic
test = FTCPU*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb_CPU = (FTCPU'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);

% Display
figure;
subplot(1,2,1); imshow(imag(img_comb_CPU(:,:)),[]); title('Regridding NUFFT');
subplot(1,2,2); imshow(abs(v(:,:)),[]); title('orig');
figure, plot(imag(img_comb_CPU(6,:)));title('Line 6 - NUFFT');
figure,surf(x,y,double(imag(img_comb)));title('NUFFT imag');

figure, plot(imag(img_comb_CPU(6,:)./max(imag(col(img_comb_CPU(6,:))))-img_comb(6,:)./max(imag(col(img_comb(6,:)))))); title('Line 6 DIFF - NUFFT - gpuNUFFT');
%% fft referenz
ref_k = fft2(v);
ref_img = (ifft2(reshape((ref_k),[N N])));
figure, imshow(imag(ref_img),[]);title('pure FFT Ref');
figure, plot(imag(ref_img(6,:)));title('pure FFT Ref Line 6');

figure, plot(imag(img_comb(6,:))./max(imag(img_comb(6,:)))-imag(ref_img(6,:))./max(imag(ref_img(6,:)))); title('Line 6 DIFF - FFT - gpuNUFFT');