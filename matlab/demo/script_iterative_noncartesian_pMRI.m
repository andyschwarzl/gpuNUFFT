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
addpath('./data')
addpath(genpath('../GRIDDING3D'));
addpath(genpath('../../../fessler/NUFFT'));
addpath(genpath('./utils'));
%% (1) simulate data
load im1.mat
load smaps_phantom.mat
%% generate noisy data
acc_factor = 1;
refL = 0;
noise_scale = 0.2;

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

% build sampling operators
F = NUFFT(k(:,1)+1i.*k(:,2), w, 1, 0, [nx,ny], 2);
noisy_data = squeeze(F * (noisy_im));

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
    
%% (3) Image Reconstruction with CG SENSE
tic
alpha = .01;    % penalty
tol = 1e-8;     % CG tolerance
maxitCG = 40;   % maximal CG iterations
cgsense_recon = pmri_cgsense_arbtra(data,F,smaps,zeros(nx,ny),alpha,tol,maxitCG);
toc
   
figure;
subplot(1,3,1);imshow(abs(cgsense_recon),[]); title('CG-Sense Recon');
subplot(1,3,2);imshow(abs(im1),[]); title('Ground Truth');
subplot(1,3,3);imshow(abs(triv_recon),[]); title('trivial Recon');   

%% my NUFFT CG SENSE
NFT = NUFFT(k(:,1)+1i.*k(:,2), w, 1, 0, [nx,ny], 2);

ground_truth = im1;
smaps_true = smaps;
triv_recon = sum(NFT' * reshape(data,[nRO nProjections nc]),3);

FT = @(x) NFT * x;
IFT = @(x) NFT' * x;

%sampling pattern given by NUFFT Operator

% image -> k-space
F = @(x,s) FT(x.*s);
% k-space -> image
FH = @(x,s) conj(s).*IFT(x);

% build righthand-side, K^T g
for chn_cnt = 1:nc
    KHg(:,:,chn_cnt) = FH(noisy_data(:,chn_cnt),smaps_true(:,:,chn_cnt));
end
KHg = sum(KHg,3);
myalpha = 0.01;

% left side (K^T K + alpha I) u_k
KHK = @(u_k) myop(F,FH,u_k,smaps_true) + myalpha*u_k;
% solve system
[u,flag,relres,iter] = pcg(KHK,KHg(:),1E-8,100);
[relres,iter]
% reshape solution vector
u = reshape(u,nx,ny);
%%
figure('name','my CG-sense recon');
h1 = subplot(1,3,1);imshow(abs(u),[]); title(['my CG-Sense Recon, noise scale ']);
h2 = subplot(1,3,2);imshow(abs(ground_truth),[]); title('Ground Truth');
h3 = subplot(1,3,3);imshow(abs(triv_recon),[]); title('trivial Recon');  
linkaxes([h1,h2,h3]);
