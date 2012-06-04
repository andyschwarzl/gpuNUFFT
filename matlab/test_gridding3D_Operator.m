%% testscript with operator usage
clear all; close all; clc;

%% add bin to path
addpath ../bin
addpath ../../daten
addpath(genpath('./GRIDDING3D'));
%% Load data
%load 20111017_Daten_MREG;
%load MREG_data_Graz;
load 20111013_MREG_data_Graz_SoS;
%load 20111024_MREG_Data_MID_65_2mm_Full_Brain;

%% sensmaps
smaps = getfield(E,'sensmaps');
%%
smaps_il = zeros([2,size(smaps{1}),length(smaps)]);
for k = 1:length(smaps),
    smaps_il(1,:,:,:,k) = real(smaps{k});%.*E.nufftStruct.sn ./ max(E.nufftStruct.sn(:));
    smaps_il(2,:,:,:,k) = imag(smaps{k});%.*E.nufftStruct.sn./ max(E.nufftStruct.sn(:));
end;
smaps = squeeze(smaps_il(1,:,:,:,:) + 1i*smaps_il(2,:,:,:,:));
clear smaps_il;
%% Perform Regridding with Kaiser Besser Kernel 64
osf = 1.25;%1,1.25,1.5,1.75,2
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
size(imgRegrid_kb);
toc
%% SENS corr
%imgRegrid_kb = imgRegrid_kb(:,:,1:44,:) .* conj(smaps(:,:,:,:));
imgRegrid_kb = imgRegrid_kb(:,:,:,:) .* conj(smaps(:,:,:,:));

%% res = SoS of coil data
res = sqrt(sum(abs(imgRegrid_kb).^2,4));
%%
slice = 25;
figure, imshow(imresize(abs(res(:,:,slice)),4),[]), title('gridding all coils at once');
figure, imshow(imresize(abs(z(:,:,slice)),4),[]), title('reference (CG)');

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

%% check forward gridding using solution z
z_pad = padarray(z,[0 0 10]);
%%

imwidth = E.imageDim(1);
osf = 2;
wg = 3;
sw = 8;
k = E.nufftStruct.om'./(2*pi);
w = ones(1,E.trajectory_length);
G3D = gridding3D(k,w,imwidth,osf,wg,sw,'deappo');

dataRadial = G3D*z_pad;
%% calculate density compensation
dc = sqrt(sum(abs(k').^2,2));
%dc = dc / max(dc);
dataRadial_dc = dataRadial .* dc;
%%
imgRegrid_dc = G3D'*dataRadial_dc;
imgRegrid = G3D'*dataRadial;
%%
slice = 25;
figure, imshow(imresize(abs(imgRegrid(:,:,slice)),4),[]), title('gridding');
figure, imshow(imresize(abs(imgRegrid_dc(:,:,slice)),4),[]), title('gridding dc');
figure, imshow(imresize(abs(z(:,:,slice)),4),[]), title('input z');

%show3DImageasArray([4 4],imgRegrid,'gridding','slice ');
%show3DImageasArray([4 4],imgRegrid_dc,'gridding dc','slice ');
