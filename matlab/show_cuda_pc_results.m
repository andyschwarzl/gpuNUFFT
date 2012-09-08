%%
load ../../daten/results/2D_256_1_25_3_32
%%

figure, imshow(imresize(((abs(recon_sos_res(:,:)))),2),[]), title('gridding dc sos');