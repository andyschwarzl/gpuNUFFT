%% Radial Regridding

clear all; close all; clc;

%% add bin to path
addpath ../bin

addpath ../../daten

%% Load data
load MREG_data_Graz;

%% sensmaps
smaps = getfield(E,'sensmaps');
smaps_il = zeros([2,size(smaps{1}),length(smaps)]);
for k = 1:length(smaps),
    smaps_il(1,:,:,:,k) = real(smaps{k});%.*E.nufftStruct.sn;
    smaps_il(2,:,:,:,k) = imag(smaps{k});%.*E.nufftStruct.sn;
end;
smaps = squeeze(smaps_il(1,:,:,:,:) + 1i*smaps_il(2,:,:,:,:));

%% calc deappo func: alternative deappo
osf = 1; % 1 or 2
wg = 3;  % 3 to 7

kspace_de = [1];
k_de = [0;0;0];
w_de = [1];
[deapo,kernel_deapo] = grid3D(kspace_de,k_de,w_de,64,osf,wg,'deappo');
figure, imshow(abs(flipud((deapo(:,:,25)))),[]);

%% Perform Regridding with Kaiser Besser Kernel 64
k = E.nufftStruct.om'/(2*pi);
w = ones(11685,1);
res = zeros(E.imageDim);
for coil = 1 : E.numCoils,
        coil_start =  (coil-1) * E.trajectory_length +1;
        coil_end = coil_start +  E.trajectory_length - 1;
        % get kspace data and k trajectories
        kspace = data(coil_start:coil_end);
        tic
        [imgRegrid_kb,kernel] = grid3D(kspace,k,w,64,osf,wg,'deappo');
        toc
        'SENS corr'
        imgRegrid_kb = imgRegrid_kb(:,:,[11:54]) .* conj(smaps(:,:,:,coil));
        imgRegrid_kb = imgRegrid_kb ./ deapo(:,:,[11:54]);
        res = sqrt(abs(res).^2 + abs(imgRegrid_kb).^2);
end
%%
figure, imshow(abs(fliplr((res(:,:,35)))),[]);
figure, imshow(abs(((z(:,:,35)))),[]);

%%
for slice = 1:44
    figure, imshow(abs(fliplr((res(:,:,slice)))),[]);
end