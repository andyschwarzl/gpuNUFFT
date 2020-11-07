%% Test CG SENSE recon of 2D brain 32 channel data
% Last change: Jan 2018
% By: Florian Knoll (florian.knoll@nyumc.org)
%
% Note: the useMultiCoil flag that includes coil sensitivities in the
% operator is only implemented for the gpuNUFFT, not for the CPU reference

clear all; clc; close all; 

addpath('../../CUDA/bin');
addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));

%% Reconstruction parameters
maxitCG = 20;
alpha = 1e-6;
tol = 1e-6;
display = 1;

% useGPU = false;
useGPU = true;
useMultiCoil = 1; % flag to compute all coils with one gpu call

%% Load data
load ./data/brain_64spokes.mat;
[nFE,nSpokes,nCh]=size(rawdata);
rawdata = reshape(rawdata,[nFE*nSpokes,nCh]);

%% Generate NUFFT Operator without coil sensitivities
% simple density compensation for radial
w = sqrt(abs(k));

disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
imwidth = nFE/2;

if (useGPU)
    FT = gpuNUFFT([real(col(k)), imag(col(k))]',col(w),osf,wg,sw,[imwidth,imwidth],[]);
else
    FT = NUFFT(col(k),col(w),1,0,[imwidth,imwidth], 2);
    useMultiCoil = 0; 
end

%% Estimate sensitivities
disp('Estimate coil sensitivities.');
for ii=1:nCh
    img_sens(:,:,ii) = FT'*(rawdata(:,ii));
end

img_sens_sos = sqrt(sum(abs(img_sens).^2,3));
senseEst = img_sens./repmat(img_sens_sos,[1,1,nCh]);

%% Redefine regridding operator GPU including coil sensitivities
disp('Generate NUFFT Operator with coil sensitivities');
if (useGPU)
    FT = gpuNUFFT([real(col(k)), imag(col(k))]',col(w),osf,wg,sw,[imwidth, imwidth],senseEst);
end

%% Forward and adjoint transform
tic
img_comb = zeros(imwidth,imwidth);
if (useGPU)
    img_comb = FT'*rawdata;
else
    for ii=1:nCh
        img_comb = img_comb + (FT'*(rawdata(:,ii))) .* conj(senseEst(:,:,ii));
    end
end
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);
figure,imshow(abs(img_comb(:,:)),[]); title('Regridding');

tic
if (useGPU)
    test = FT * img_comb;
else
    for ii=1:nCh
        test(:,ii) = FT * (img_comb .* senseEst(:,:,ii));
    end
end
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);

%% CGSENSE Reconstruction
tic
img_cgsense = cg_sense_2d(rawdata,FT,senseEst,1,alpha,tol,maxitCG,display,useMultiCoil);
timeCG = toc;
disp(['Time CG SENSE: ', num2str(timeCG), ' s']);

%% Display
figure;
subplot(1,2,1); imshow(abs(img_comb),[]); title('Regridding');
subplot(1,2,2); imshow(abs(img_cgsense),[]); title('CGSENSE');
