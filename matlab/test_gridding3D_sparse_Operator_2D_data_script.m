for (counter = 1:10)
%% testscript with operator usage
clear all; 
close all; clc;

%% add bin to path
addpath ../bin  
addpath ../../daten
addpath(genpath('./GRIDDING3D'));
addpath(genpath('./utils'));
addpath(genpath('../../tgv/NUFFT'));
%% Load data
load img_brain_4ch;
%load noisy_phantom;
%load calf_data_cs;
%%
slice=32;
trimmed_size = 128;
img = img(128-trimmed_size/2+1:128+trimmed_size/2,128-trimmed_size/2+1:128+trimmed_size/2,:);
%%
n_chn = 4;
img_a = zeros([trimmed_size,trimmed_size,trimmed_size,n_chn]);
%%
for chn = 1:n_chn,
    img_a(:,:,:,chn) = repmat(img(:,:,chn),[1 1 trimmed_size]);
end
%img_a(1:10,1:10,:) = 0;
%img_a = flipdim(img_a,1);
%img_a = padarray(img(:,:,1),[0 0 trimmed_size/2]);
%%
size(img_a)
%figure, imshow(imresize(abs(img_a(:,:,1,n_chn)),4),[]), title('gridding input');

[nPE,nFE,nCh]=size(img_a);

%% Generate 96 radial projections rawdata
numSpokes = 96;

% Trajectory
theta=linspace(0,pi-pi/numSpokes,numSpokes);
rho=linspace(-0.5,0.5,nPE*2)';
k=rho*exp(-1j*theta);

%% generate Fourier sampling operator
%FT = GRIDDING3D(k, 1, 1, 0, [nPE,nFE], 2);
k_traj = [real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
imwidth = nPE;
osf = 1.25;
wg = 3;
sw = 8;
w = ones(1,length(k(:)));

E = NUFFT3D(k_traj', 1, 1, 0, [imwidth,imwidth,imwidth], 2);

FT = GRIDDING3D(k_traj,w,imwidth,osf,wg,sw,[trimmed_size trimmed_size trimmed_size],'sparse',E);

%% generate radial data
tic
dataRadial = inversegrid_multicoil_gpu(img_a,FT,2*nPE,numSpokes);
toc
dataRadial = reshape(dataRadial, [2*nPE*numSpokes n_chn]);
%% density compensation
w = abs(rho);
w = repmat(w, [1, numSpokes,1]);
w_mc = repmat(w(:),[1 n_chn]);
dataRadial_dc = dataRadial.*w_mc;
%% recon
%no density compnesation
%imgRegrid_kb = FT'*dataRadial;
%tic
%imgRegrid_kb = regrid_multicoil_gpu(reshape(dataRadial,[size(k),chn]),FT);
%toc
%% density compensated
%imgRegrid_kb_dc = FT'*dataRadial_dc;
%pause;
tic
imgRegrid_kb_dc = regrid_multicoil_gpu(reshape(dataRadial_dc,[size(k),chn]),FT);
toc
%% show results
%figure, imshow(imresize(((abs(imgRegrid_kb_dc(:,:,32,n_chn)))),4),[]), title('gridding dc');

%% merge channels
recon_sos_dc = sqrt(sum(abs(imgRegrid_kb_dc).^2,4));
recon_sos_res = recon_sos_dc(:,:,slice);
%figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gridding dc sos');
disp('finished iteration');
end;
out_file = ['../../daten/results/2D_',num2str(trimmed_size),'_',strrep(num2str(osf),'.','_'),'_',num2str(wg),'_',num2str(slice)];
save(out_file, 'recon_sos_res');
disp(['output written to ',out_file]);
exit;
