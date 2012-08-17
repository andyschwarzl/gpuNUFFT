%%
load ../../daten/results/2D_256_1_25

%%
figure, imshow(imresize(((abs(recon_sos_dc(:,:)))),2),[]), title('gridding dc sos');
%%
load ../../daten/results/2D_224_1_50

%%
figure, imshow(imresize(((abs(recon_sos_dc(:,:)))),2),[]), title('gridding dc sos');