%close all;
im_w = 32;
%% test plot of k space
x = k_traj(1,:,:);
y = k_traj(2,:,:);
v =abs(double(dataRadial'));
size(x)

[xq,yq] = meshgrid(linspace(-0.5,0.5,im_w+1),linspace(-0.5,0.5,im_w+1));

%%
vq = griddata(x,y,v,xq,yq,'nearest');

figure, surf(xq,yq,vq);
colorbar;
%%
x_set = x(find(dataRadial));
y_set = y(find(dataRadial));

figure;
scatter(x_set,y_set);
