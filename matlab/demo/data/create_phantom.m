%create numerical shepp-logan phantom
clc; clear all; close all;

javaaddpath ./STBBFolder/JarFile/STBB.jar
import STBB.*
sl2d = SheppLogan2D(); 

%% build kspace trajectory
nPE = 128;
numSpokes = 128;
% phantom kspace has to be scaled
% from -32 - 32
%
pR = 32;
% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-pR,pR,nPE)';
k=rho*exp(-1j*theta);

%uniform sampling
[X,Y]=meshgrid(rho,rho);
k_traj = [X(:), Y(:)];
%radial sampling
%k_traj = [real(k(:))'; imag(k(:))'];

%% test kspace
dataRadial = sl2d.FourierDomainSignal([k_traj(:,1),k_traj(:,2)]);
dataRadial = reshape(dataRadial(:,1)+1i*dataRadial(:,2),[nPE,nPE]);

%% scale to -0.5 0.5
k_traj = -0.5 + (k_traj+pR)/(2*pR);
%%
img = fftshift(ifft2((dataRadial)));
figure, imshow(abs(img),[]);

save('sl2d','dataRadial','k_traj');