%% Radial Regridding
clear all; close all; clc;

%% add bin to path
addpath ../bin
addpath ../../daten
%% Load data
%load 20111017_Daten_MREG;
load MREG_data_Graz;

%% sensmaps
smaps = getfield(E,'sensmaps');
smaps_il = zeros([2,size(smaps{1}),length(smaps)]);
for k = 1:length(smaps),
    smaps_il(1,:,:,:,k) = real(smaps{k});%.*E.nufftStruct.sn;
    smaps_il(2,:,:,:,k) = imag(smaps{k});%.*E.nufftStruct.sn;
end;
smaps = squeeze(smaps_il(1,:,:,:,:) + 1i*smaps_il(2,:,:,:,:));

%% Perform Regridding with Kaiser Besser Kernel 64
osf = 2;
wg = 5;
sw = 8;
imwidth = 64;
k = E.nufftStruct.om'./(2*pi);
w = ones(11685,1);
res = zeros(E.imageDim);
for coil = 1 : E.numCoils,
        disp(['iteration ',num2str(coil)]);
        coil_start =  (coil-1) * E.trajectory_length +1;
        coil_end = coil_start +  E.trajectory_length - 1;
        % get kspace data and k trajectories
        kspace = data(coil_start:coil_end);
        tic
        [imgRegrid_kb,kernel] = grid3D(kspace,k,w,imwidth,osf,wg,sw,'deappo');
        toc
        %SENS corr
        imgRegrid_kb = imgRegrid_kb(:,:,[11:54]) .* conj(smaps(:,:,:,coil));
        
        %res = res + imgRegrid_kb; 
        res = sqrt(abs(res).^2 + abs(imgRegrid_kb).^2);
end
%%
res_test = res .* conj(E.nufftStruct.sn) / sqrt(prod(size(res)));
figure, imshow(imresize(abs(((res_test(:,:,25)))),4),[]), title('gridding');
%%
%figure, imshow(abs(fliplr((res(:,:,25)))),[]);
figure, imshow(imresize(abs(((res(:,:,25)))),4),[]), title('gridding');
%figure, imshow(abs(((z(:,:,25)))),[]);

%%
for slice = 1:44
    figure, imshow(imresize(abs(res(:,:,slice)),4),[]);
end