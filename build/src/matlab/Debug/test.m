%% 
close all; clear all; clc;

%% data values
data = [0.5, 0.7, 1, 1, 1;
        0.5, 1,   1, 1, 1];
    
%% data coords x,y,z in [-0.5,0.5] 
coords = [-0.3, -0.1, 0, 0.5, 0.3;
           0.2,    0, 0,   0, 0.3;
             0,    0, 0,   0, 0];

[test_coords, test_sector_centers] = assign_sectors(10,5,coords);
test_sector_centers
%% sector data indizes
sectors = [0,0,0,0,0,0,0,2,5];

%%
sector_centers = [2, 7, 2, 7, 2, 7, 2, 7;
                  2, 2, 7, 7, 2, 2, 7, 7;
                  2, 2, 2, 2, 7, 7, 7, 7];

%% init params
params.im_width = uint32(10);
params.osr = single(1);
params.kernel_width = uint32(3);
params.sector_width = uint32(5);

%%
size(data)
size(coords)

%%
gdata = cuda_mex_kernel(single(data(:,:)),single(coords(:,:)),uint32(sectors),uint32(sector_centers),params);

%%
test_gdata = gdata(:,:,:,6);
test2 = reshape(test_gdata (1,:,:),10,10);

%% print result
flipud(test2')