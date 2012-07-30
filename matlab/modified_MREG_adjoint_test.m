% MREG Test

clear all; close all; clc;
addpath(genpath('GRIDDING3D'));
addpath(genpath('../bin'));
%% test for adjoint prop
imwidth = 64;
z_pad = [imwidth imwidth imwidth 2];
x1 = randn((z_pad));
y2 = randn([10000 2]);

osf = 1.5;%1,1.25,1.5,1.75,2
wg = 3;%3-7
sw = 8;
%% Generate 96 radial projections rawdata
nPE = 5000;
numSpokes = 1;
% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-1j*theta);
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];

w = ones(1,length(k(:)));
%%
G3D = GRIDDING3D(k_traj,w,imwidth,osf,wg,sw,'false');

%%
y1 = G3D * x1;
x2 = G3D' * y2;
%%
x1(:)'*x2(:) - y1(:)'*y2(:)
