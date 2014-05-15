function save3DImage(dim,p_img,f_title,prefix)

filename = strrep(['results/' f_title, num2str(dim), '.png'],' ','')
res = [];
for k = 1:dim(1),
    temp = [];
    for j = 1:dim(2)
        temp = [temp p_img(:,:,j+(k-1)*dim(2)) zeros(size(p_img,1),1)];
    end
    res = [res; zeros(1,size(temp,2)); temp];
end

imwrite(abs(res)./max(abs(res(:))),filename);
    
