%%
load ../../daten/results/2D_256_2_5_32

%%
figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gridding dc sos');
%%
load ../../daten/results/2D_256_1_5_3_32

%%
figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gridding dc sos');