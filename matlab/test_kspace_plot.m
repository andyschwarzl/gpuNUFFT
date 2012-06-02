%close all;
im_w = 16;
%% test plot of k space
i_sl = 9;
x = test(1,:,i_sl);
y = test(2,:,i_sl);
v =abs(double(dataRadial(3072*(i_sl-1) + [1:3072])'));
size(x)

[xq,yq] = meshgrid(linspace(-0.5,0.5,im_w+1),linspace(-0.5,0.5,im_w+1));

%%
vq = griddata(x,y,v,xq,yq,'nearest');

figure, surf(xq,yq,vq);
title(['test slice ' num2str(i_sl)]);
colorbar;
%%
x_set = x(find(dataRadial(3072*(i_sl-1) + [1:3072])));
y_set = y(find(dataRadial(3072*(i_sl-1) + [1:3072])));

figure;
scatter(x_set,y_set);
title(['pos test slice ' num2str(i_sl)]);