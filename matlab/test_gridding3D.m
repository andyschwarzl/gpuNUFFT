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
wg = 5;  % 3 to 7
sw = 8;
im_width = 64;
kspace_de = [1];
k_de = [0;0;0];
w_de = [1];
[deapo,kernel_deapo] = grid3D(kspace_de,k_de,w_de,im_width,osf,wg,sw,'deappo');
figure, imshow(abs(fliplr((deapo(:,:,im_width/2)))),[]);

deapo = abs(deapo(:,:,11:54));
%% Perform Regridding with Kaiser Besser Kernel 64
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
        [imgRegrid_kb,kernel] = grid3D(kspace,k,w,64,osf,wg,sw,'deappo');
        toc
        %SENS corr
        imgRegrid_kb = imgRegrid_kb(:,:,[11:54]) .* smaps(:,:,:,coil);
        imgRegrid_kb = imgRegrid_kb ./ deapo;
        
        res = sqrt(abs(res).^2 + abs(imgRegrid_kb).^2);
end
%%
figure, imshow(abs(fliplr((res(:,:,25)))),[]);
figure, imshow(abs(((z(:,:,25)))),[]);

%%
for slice = 1:44
    figure, imshow(abs(fliplr((res(:,:,slice)))),[]);
end

%% Kernel of Size 5 test
%% calc deappo func: alternative deappo
osf = 1; % 1 or 2
wg = 5;  % 3 to 7
sw = 5;
kspace_de = [1];
k_de = [0;0;0];
w_de = [1];
[deapo,kernel_deapo] = grid3D(kspace_de,k_de,w_de,10,osf,wg,sw,'deappo');
figure, imshow(abs(flipud((deapo(:,:,10)))),[]);

osf=1;
kspace_test = ([0.5+0.5i,0.7+1i,1+1i,1+1i,1+1i]);
wg = 5;
sw = 5;
k_test = ([-0.3,0.2,0;
           -0.1,0,0;
           0,0,0;
           0.5,0,0;
           0.3,0.3,0]');
w_test = ([1,1,1,1,1]);

[imgRegrid_kb,kernel] = grid3D(kspace_test,k_test,w_test,10,osf,wg,sw,'image');
imgRegrid_kb = imgRegrid_kb ./ deapo;
figure, imshow(log(abs(fliplr((imgRegrid_kb(:,:,6))))),[]);
