%% 
close all; clear all; clc;

%% data values
data = [ 0.7, 1, 1, 1,0.5;
         1,   1, 1, 1,0.5];
    
%% data coords x,y,z in [-0.5,0.5] 
coords = [ -0.1, 0, 0.5, 0.3,-0.3;
              0, 0,   0, 0.3, 0.2;
              0, 0,   0, 0,     0];

[test_coords, test_sector_centers,sector_dim] = assign_sectors(10,5,coords);

[v i] = sort(test_coords);
v = v +1
sectors_test = zeros(1,sector_dim+1);
cnt = 0;
for b=1:sector_dim+1
    while (cnt < length(v) && b == v(cnt+1))
        cnt = cnt +1;
    end
    sectors_test(b)=cnt;
end
sectors_test
%% calculate indices of data elements in order to sort them
data_ind = i-1;
data_ind=[2*data_ind+1;2*data_ind+2];
data(data_ind)

%% calculate indices of coord elements in order to sort them
coord_ind = i-1;
coord_ind = [3*coord_ind+1;
             3*coord_ind+2;
             3*coord_ind+3];
coords(coord_ind)
%%
test_sector_centers = int32(reshape(test_sector_centers,[3,sector_dim]))

%% sector data indizes
sectors = [0,0,0,0,0,0,0,2,5];

%%
% sector_centers = [2, 7, 2, 7, 2, 7, 2, 7;
%                   2, 2, 7, 7, 2, 2, 7, 7;
%                   2, 2, 2, 2, 7, 7, 7, 7];

%% init params
params.im_width = uint32(10);
params.osr = single(1);
params.kernel_width = uint32(3);
params.sector_width = uint32(5);

%%
size(data)
size(coords)

%%
gdata = cuda_mex_kernel(single(data(data_ind)),single(coords(coord_ind)),int32(sectors_test),int32(test_sector_centers),params);

%%
test_gdata = gdata(:,:,:,6);
test2 = reshape(test_gdata (1,:,:),10,10);

%% print result
flipud(test2')