function [] = run_demo(v, FT, FTCPU, x,y,N, demoTitlePrefix)
%% Forward and adjoint transform
tic
test = FT*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb = (FT'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);

%% Forward and adjoint transform NUFFT
tic
test = FTCPU*(v);
timeFT = toc;
disp(['Time forward: ', num2str(timeFT), ' s']);
tic
img_comb_CPU = (FTCPU'*(test));
timeFTH = toc;
disp(['Time adjoint: ', num2str(timeFTH), ' s']);
%% fft reference
ref_k = fft2(v);
ref_img = (ifft2(reshape((ref_k),[N N])));

%% Compare results
figure;
h1=subplot(1,3,1); imshow(imag(img_comb),[]); title([demoTitlePrefix 'Regridding gpuNUFFT']);
h2=subplot(1,3,2); imshow(imag(img_comb_CPU),[]); title([demoTitlePrefix 'Regridding NUFFT']);
h3=subplot(1,3,3); imshow(imag(ref_img),[]);title('pure FFT Ref');
linkaxes([h1 h2 h3]);

figure;
subplot(3,1,1);plot(imag(img_comb(6,:)));title([demoTitlePrefix 'Line 6 - gpuNUFFT']);
subplot(3,1,2);plot(imag(img_comb_CPU(6,:)));title([demoTitlePrefix 'Line 6 - NUFFT']);
subplot(3,1,3);plot(imag(ref_img(6,:)));title('pure FFT Ref Line 6');

figure,surf(x,y,double(imag(img_comb)));title([demoTitlePrefix 'gpuNUFFT imag']);
figure,surf(x,y,double(imag(img_comb_CPU)));title([demoTitlePrefix 'NUFFT imag']);
