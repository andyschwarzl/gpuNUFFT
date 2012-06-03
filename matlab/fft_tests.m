%% test fft
im_size = 16;
img = zeros([im_size im_size]);
img(9,9) = 1;
%img(2,15) = 1;
%img(4,4) = 1;

Xk = fft2(img);
%%
figure('name','spectrum');
imshow(imresize(real(Xk),4),[]);
%%
img_recon = ifft2(Xk);

figure,
imshow(imresize(abs(img_recon),4),[]);

