%% Test CG SENSE recon of 2D brain 32 channel data
clear all; clc; close all; 

addpath(genpath('./utils'));
addpath(genpath('../../gpuNUFFT'));
addpath(genpath('../../../fessler/NUFFT'));

load ./data/brain_64spokes.mat;

%% Reconstruction parameters
maxitCG = 10;
alpha = 1e-6;
tol = 1e-6;
display = 1;

% useGPU =false;
useGPU = true;
useMultiCoil = 1; % flag to compute all coils with one gpu call

%%
[nFE,nSpokes,nCh]=size(rawdata);
rawdata = reshape(rawdata,[nFE*nSpokes,nCh]);

% construct density compensation data
w = abs(k);
%%
disp('Generate NUFFT Operator without coil sensitivities');
osf = 2; % oversampling: 1.5 1.25
wg = 3; % kernel width: 5 7
sw = 8; % parallel sectors' width: 12 16
imwidth = nFE/2;

if (useGPU)
    FT = gpuNUFFT([real(col(k)), imag(col(k))]',col(w),osf,wg,sw,[imwidth,imwidth],[],true);
else
    FT = NUFFT(col(k),col(w),1,0,[imwidth,imwidth], 2);
    useMultiCoil = 0; 
end
%%
for ii=1:nCh
    img_sens(:,:,ii) = FT'*rawdata(:,ii);
end

%% Estimate sensitivities
disp('Estimate coil sensitivities.');
img_sens_sos = sqrt(sum(abs(img_sens).^2,3));
[~,senseEst]=adapt_array_2d(img_sens);

%% Redefine regridding operator GPU including coil sensitivities
disp('Generate NUFFT Operator with coil sensitivities');
if (useGPU)
    FT = gpuNUFFT([real(col(k)), imag(col(k))]',col(w),osf,wg,sw,[imwidth, imwidth],senseEst,true);
end

%% Forward and adjoint transform
tic
img_comb = zeros(imwidth,imwidth);
if (useGPU)
    img_comb = FT'*rawdata;
else
    for ii=1:nCh
        img_comb = img_comb + (FT'*rawdata(:,ii)) .* conj(senseEst(:,:,ii));
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
mask = 1;
tic
img_cgsense = cg_sense_2d(rawdata,FT,senseEst,mask,alpha,tol,maxitCG,display,useMultiCoil);
timeCG = toc;
disp(['Time CG SENSE: ', num2str(timeCG), ' s']);

%% Display
figure;
subplot(1,2,1); imshow(abs(img_comb),[]); title('Regridding');
subplot(1,2,2); imshow(abs(img_cgsense),[]); title('CGSENSE');
