%% iterative Parallel Imaging for non-Cartesian k-space data
%  Conjugate-Gradient (CG)-SENSE  Reconstruction for radial k-space data
% 
%
% UE Inverse Probleme in der medizinischen Bildgebung
% WS 2013
%
% Matthias Schloegl
% matthias.schloegl@tugraz.at
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
clear all; close all; clear classes; clc
addpath('./data');
addpath ../../bin;
addpath(genpath('../GRIDDING3D'));
addpath(genpath('../../../fessler/NUFFT'));
addpath(genpath('./utils'));
%% (1) simulate data
load im1.mat
load smaps_phantom.mat
%% generate noisy data
acc_factor = 1;
refL = 0;
noise_scale = 0.02;

[data, sp, smaps_true, Rn, ground_truth] = generate_data(acc_factor,refL,'normal',noise_scale);
noisy_im= ifft2c(data);

%%
[nx,ny] = size(im1);
nc = size(smaps,3);

acc_factor = 4;

nRO = nx; % readout-length
nProjections = floor(nx/acc_factor); % number of projections
fprintf(['Number of Projections : ' num2str(nProjections) ' equals R = ' num2str(acc_factor) '\n']);
N = nx*ny;

% simulate trajectory
[k,w] = ismrm_generate_radial_trajectory(nRO, nProjections);
w = w./max(w(:));
figure; plot(k(:,1),k(:,2),'*b')
%%
% build sampling operators
F = NUFFT(k(:,1)+1i.*k(:,2), w, 1, 0, [nx,ny], 2);
%%
%gpuNUFFT
osf =1.25;
wg = 3;
sw = 8;

FT = GRIDDING3D(k',w,nx,osf,wg,sw,[nx,ny],smaps,'false');

noisy_data = squeeze(F * (noisy_im));
%%
% define operators
ift = @(x) F'*x;
ft = @(x) F*x;

% simulate data
%data = squeeze(F*(repmat(im1,[1 1 nc]).*smaps));
data = noisy_data;

% normalize data
data = data./max(abs(data(:)));

% trivial reconstruction
alias_image = zeros(nx,ny,nc);
for i = 1:nc
     alias_image(:,:,i) = F'*(data(:,i));
end

ismrm_imshow(abs(alias_image),[],[2 4]);
 
triv_recon = sqrt(sum(abs(alias_image).^2,3));

%% (2) estimate coil sensitivities
% smaps_est = zeros(nx, ny, nc);   
%     for ii = 1:nc
%         smaps_est(:,:,ii) = h1_l2_2D_pd(alias_image(:,:,ii), sqrt(w).*data(:,ii), F, 1, 1000, 100, 0.001);
% 
%         % smooth sensitivity estimate
%         mask = [0 1 0; 1 4 1; 0 1 0]/8;
%         for j=1:100
%             smaps_est(:,:,ii) = conv2(smaps_est(:,:,ii),mask,'same');
%         end
%     end
% imgSensSOS = sqrt(sum(abs(smaps_est).^2,3));
% smaps_est = smaps_est./repmat(imgSensSOS,[1 1 nc]);
% smaps_est(find(isnan(smaps_est)))=0;
%  
% ismrm_imshow(abs(smaps_est),[],[2 4]);
    
%% (3) Image Reconstruction with CG SENSE CPU
display('CPU recon...');
tic
alpha = .1;    % penalty
tol = 1e-8;     % CG tolerance
maxitCG = 40;   % maximal CG iterations
cgsense_recon = pmri_cgsense_arbtra(data,F,smaps,zeros(nx,ny),alpha,tol,maxitCG);
toc
   
figure;
subplot(1,3,1);imshow(abs(cgsense_recon),[]); title('CG-Sense Recon');
subplot(1,3,2);imshow(abs(im1),[]); title('Ground Truth');
subplot(1,3,3);imshow(abs(triv_recon),[]); title('trivial Recon');   

%% (4) Image Reconstruction with CG SENSE GPU
display('GPU recon...');
tic
alpha = .1;    % penalty
tol = 1e-8;     % CG tolerance
maxitCG = 40;   % maximal CG iterations
cgsense_recon = pmri_cgsense_arbtra(data,FT,smaps,zeros(nx,ny),alpha,tol,maxitCG,true);
toc

figure;
subplot(1,3,1);imshow(abs(cgsense_recon),[]); title('CG-Sense Recon GPU');
subplot(1,3,2);imshow(abs(im1),[]); title('Ground Truth');
subplot(1,3,3);imshow(abs(triv_recon),[]); title('trivial Recon');   
