%% testscript with operator usage
clear all; 
close all; clc; 

%% add bin to path
addpath ../bin  
addpath ../../cuda-test/gridding3D/daten
addpath(genpath('./GRIDDING3D'));
addpath(genpath('./utils'));
%% Load data
load img_brain_4ch;
load kspaceRadial;
%load noisy_phantom;
%load calf_data_cs;
%%
slice=32;
trimmed_size = 256;
img = img(128-trimmed_size/2+1:128+trimmed_size/2,128-trimmed_size/2+1:128+trimmed_size/2,:);
n_chn = 4;
img_a = zeros([trimmed_size,trimmed_size,trimmed_size,n_chn]);
for chn = 1:n_chn,
    img_a(:,:,:,chn) = repmat(img(:,:,chn),[1 1 trimmed_size]);
end

[nPE,nFE,nCh]=size(img_a);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-1j*theta);

%% generate Fourier sampling operator
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
imwidth = nPE;
osf = 1.25;
wg = 3;
sw = 8;
% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
w_mc = reshape(repmat(w(:),[1 n_chn]),[size(w), n_chn]);
%%
FT = GRIDDING3D(k_traj,w,imwidth,osf,wg,sw,[trimmed_size trimmed_size trimmed_size],'false');

%% generate radial data
tic
dataRadial = inversegrid_multicoil_gpu(img_a,FT,2*nPE,numSpokes);
toc
%dataRadial = reshape(dataRadial, [2*nPE*numSpokes n_chn]);
dataRadial_dc = dataRadial;%.*w_mc;%v2
%% recon
tic
imgRegrid_kb_dc = regrid_multicoil_gpu(reshape(dataRadial_dc,[size(k),chn]),FT);
toc

%% merge channels
recon_sos_dc = sqrt(sum(abs(imgRegrid_kb_dc).^2,4));
recon_sos_res = recon_sos_dc(:,:,slice);

disp('finished');
out_file = ['../tmp/results/2D_',num2str(trimmed_size),'_',strrep(num2str(osf), '.', '_'),'_',num2str(wg),'_',num2str(slice),'.png'];
imwrite(abs(recon_sos_res)/max(recon_sos_res(:)),out_file,'png');
disp(['output written to ',out_file]);
exit;
