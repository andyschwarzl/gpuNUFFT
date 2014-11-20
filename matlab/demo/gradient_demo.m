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
%% smooth recon
disp('Generate NUFFT Operator');
osf = 2;
kw = 3; %1 also possible and ideal for uniform cartesian 
sw = 8;
imwidth = N;

FT = gpuNUFFT(k',w,osf,kw,sw,[N,N],[]);
FTCPU = NUFFT(k(1,:) + 1i*k(2,:),w,1,0,[N,N],2);

run_demo(v,FT,FTCPU,x,y,N,'Smooth ');
%% non smooth recon
kw = 3; %nonsmooth
FT = gpuNUFFT(k_nonsmooth',w,osf,kw,sw,[N,N],[]);
FTCPU = NUFFT(k_nonsmooth(1,:) + 1i*k_nonsmooth(2,:),w,1,0,[N,N],2);

run_demo(v,FT,FTCPU,x,y,N,'Non-smooth ');
