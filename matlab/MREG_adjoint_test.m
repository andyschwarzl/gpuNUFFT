% MREG Test

clear all; close all; clc;
load ../../daten/MREG_data_Graz;
addpath(genpath('GRIDDING3D'));
addpath(genpath('../bin'));
%%
% test for adjoint property: looks good
z_pad = padarray(z,[0 0 10]);
x1 = ones(size(z_pad));%randn(size(z_pad));
y2 = reshape(ones(size(data)),[11685 32]); %reshape(randn(size(data)),[11685 32]);

osf = 1.75;%1,1.25,1.5,1.75,2
wg = 3;%3-7
sw = 8;
imwidth = 64;
k = E.nufftStruct.om'./(2*pi);
w = ones(E.trajectory_length,1);
%%
G3D = GRIDDING3D(k,w,imwidth,osf,wg,sw,'false');
%%
x1_rep = repmat(x1,[1 1 1 32]);
%%
y1 = G3D * x1_rep;
y1_b = 1/sqrt(prod(([64 64 64]))) .* y1;
%%
x2 = G3D' * y2;
x2_b = 1/sqrt(prod(([64 64 64]))) .* x2;
test = sqrt(sum(x2.^2,4));
test_b = sqrt(sum(x2_b.^2,4));
%%
x1(:)'*test(:) - y1(:)'*y2(:)
x1(:)'*test_b(:) - y1_b(:)'*y2(:)
%printf('Adjoint test result: %e\n', x1(:)'*test(:) - y1_b(:)'*y2(:));

% Apply to data
%data1 = E * z;
%z1 = E' * data1;
