% MREG Test

clear all; close all; clc;
load ../../daten/MREG_data_Graz;

% test for adjoint property: looks good
z_pad = padarray(z,[0 0 10]);
x1 = randn(size(z_pad));
y2 = reshape(randn(size(data)),[11685 32]);

osf = 1.25;%1,1.25,1.5,1.75,2
wg = 3;%3-7
sw = 8;
imwidth = 64;
k = E.nufftStruct.om'./(2*pi);
w = ones(E.trajectory_length,1);
%%
G3D = GRIDDING3D(k,w,imwidth,osf,wg,sw);

x1 = repmat(x1,[1 1 1 32]);
%%
y1 = G3D * x1;
%%
x2 = G3D' * y2;

printf('Adjoint test result: %e\n', x1(:)'*x2(:) - y1(:)'*y2(:));

% Apply to data
%data1 = E * z;
%z1 = E' * data1;
