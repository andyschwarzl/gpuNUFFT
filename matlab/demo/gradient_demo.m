%% Forward-Backward gradient test
clear all; 
close all; clc; clear classes;

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));

%% Data parameters
N=512;

%smooth version
[x,y] = meshgrid(linspace(-0.5,0.5-1/N,N),linspace(-0.5,0.5-1/N,N));
offset = 1/(2*(N-1));
k = [col(x)'; col(y)'];

%non-smooth version
[x_ns,y_ns] = meshgrid(linspace(-0.5,0.5,N),linspace(-0.5,0.5,N));
k_nonsmooth = [col(x_ns)'; col(y_ns)'];

v = linspace(0,1,N);
v = 1i*(v'*v);

% test psf
% v = zeros(size(v));
% %v(9,9) = 1i;
%  v(N/2,N/2) = 1i;

figure,imshow(imag(v));title('test input');
figure,surf(x,y,imag(v));title('test input');

w = col(ones(size(v)));
%% smooth recon
disp('Generate NUFFT Operator');
osf = 2;
kw = 3; %1 also possible (nearest neighbor) 
sw = 8;
imwidth = N;
atomic = true;
textures = false;
loadbalancing = false;

FT = gpuNUFFT(k,w,osf,kw,sw,[N,N],[],atomic,textures,loadbalancing);
FTCPU = NUFFT(k(1,:) + 1i*k(2,:),w,1,0,[N,N],2);

run_demo(v,FT,FTCPU,x,y,N,'Smooth ');
%% non smooth recon
FT = gpuNUFFT(k_nonsmooth,w,osf,kw,sw,[N,N],[],atomic,textures,loadbalancing);
FTCPU = NUFFT(k_nonsmooth(1,:) + 1i*k_nonsmooth(2,:),w,1,0,[N,N],2);

run_demo(v,FT,FTCPU,x_ns,y_ns,N,'Non-smooth ');

%% smooth shifted
% k_shift = k + offset;
% x_shift = x + offset;
% y_shift = y + offset;
% 
% FT = gpuNUFFT(k_shift',w,osf,kw,sw,[N,N],[],atomic,textures,loadbalancing);
% FTCPU = NUFFT(k_shift(1,:) + 1i*k_shift(2,:),w,1,0,[N,N],2);
% 
% run_demo(v,FT,FTCPU,x,y,N,'Smooth shifted ');
% run_demo(v,FT,FTCPU,x_shift,y_shift,N,'Smooth shifted ');
% %% non smooth shifted
% k_nonsmooth_shift = k_nonsmooth + offset;
% x_ns_shift = x_ns + offset;
% y_ns_shift = y_ns + offset;
% 
% FT = gpuNUFFT(k_nonsmooth_shift',w,osf,kw,sw,[N,N],[],atomic,textures,loadbalancing);
% FTCPU = NUFFT(k_nonsmooth_shift(1,:) + 1i*k_nonsmooth_shift(2,:),w,1,0,[N,N],2);
% 
% run_demo(v,FT,FTCPU,x_ns,y_ns,N,'Non-smooth shifted ');
% run_demo(v,FT,FTCPU,x_ns_shift,y_ns_shift,N,'Non-smooth shifted ');