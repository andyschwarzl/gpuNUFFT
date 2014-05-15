function show3DImage(dim,p_img,f_title,prefix)

figure('name',f_title)
res = [];
for k = 1:dim(1),
    temp = [];
    for j = 1:dim(2)
        temp = [temp p_img(:,:,j+(k-1)*dim(2)) zeros(size(p_img,1),1)];
    end
    res = [res; zeros(1,size(temp,2)); temp];
end

imshow(abs(res),[]);
title(f_title);
axis off;

    