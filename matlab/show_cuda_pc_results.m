%%
%load ../../daten/results/2D_256_1_25_3_32
load ../../daten/results/2D_96_1_25_3_32
%%
figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gpuNUFFT dc sos');

%% cpu
load ../../daten/results/cpu_2D_96_1_25_3_32
%%
figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gpuNUFFT dc sos');
