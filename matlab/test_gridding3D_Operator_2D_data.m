%% testscript with operator usage
clear all; close all; clc;

%% add bin to path
addpath ../bin
addpath ../../daten
addpath(genpath('./GRIDDING3D'));

%% Load data
load img_brain_4ch;
img = img(97:160,97:160,:);
img_a = repmat(img(:,:,1),[1 1 64]);
size(img_a)
figure, imshow(imresize(abs(img_a(:,:,1)),4),[]), title('gridding');

[nPE,nFE,nCh]=size(img_a);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-j*theta);

%% generate Fourier sampling operator
%FT = GRIDDING3D(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
imwidth = nPE;
osf = 1;
wg = 3;
sw = 8;
w = ones(1,length(k(:)));
FT = gridding3D(k_traj,w,imwidth,osf,wg,sw,'deappo');

%% generate radial data
dataRadial = FT*img_a;


%% Perform Regridding with Kaiser Besser Kernel 64
osf = 2;%1,1.25,1.5,1.75,2
wg = 3;%3-7
sw = 8;
imwidth = 64;
k = E.nufftStruct.om'./(2*pi);
w = ones(E.trajectory_length,1);

G3D = GRIDDING3D(k,w,imwidth,osf,wg,sw,'deappo');

%% one call for all coils
res = zeros(E.imageDim);
kspace = reshape(data,[E.trajectory_length E.numCoils]);
%[imgRegrid_kb,kernel] = grid3D(kspace,k,w,imwidth,osf,wg,sw,'deappo');
tic
imgRegrid_kb = G3D'*kspace;
toc
%SENS corr
imgRegrid_kb = imgRegrid_kb(:,:,:,:) .* conj(smaps(:,:,:,:));

%% res = SoS of coil data
res = sqrt(sum(abs(imgRegrid_kb).^2,4));
figure, imshow(imresize(abs(res(:,:,25)),4),[]), title('gridding all coils at once');
figure, imshow(imresize(abs(z(:,:,25)),4),[]), title('reference (TGV)');

%% single call per coil 
res = zeros(E.imageDim);
tic
for coil = 1 : E.numCoils,
        disp(['iteration ',num2str(coil)]);
        coil_start =  (coil-1) * E.trajectory_length +1;
        coil_end = coil_start +  E.trajectory_length - 1;
        % get kspace data and k trajectories
        kspace = data(coil_start:coil_end);
        %[imgRegrid_kb,kernel] = grid3D(kspace,k,w,imwidth,osf,wg,sw,'deappo');
        imgRegrid_kb = G3D'*kspace;

        %SENS corr
        imgRegrid_kb = imgRegrid_kb(:,:,:) .* conj(smaps(:,:,:,coil));
        
        %res = res + imgRegrid_kb; 
        res = sqrt(abs(res).^2 + abs(imgRegrid_kb).^2);
end
toc
%%
figure, imshow(imresize(abs(res(:,:,25)),4),[]), title('gridding');

%%
%for slice = 1:64
%    figure, imshow(imresize(abs(z(:,:,slice)),4),[]);
%end