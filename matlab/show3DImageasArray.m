function show3DImageasArray(dim,p_img,f_title,prefix)

figure('name',f_title)
for k = 1:prod(dim),
    sl_tit = [prefix,int2str(k)];
    subplot(dim(1),dim(2),k);
    imshow(imresize(abs(p_img(:,:,k)),4),[]);
    title(sl_tit);
    axis off;
end