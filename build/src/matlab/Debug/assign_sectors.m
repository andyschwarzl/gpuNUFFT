function [data_sector,sector_centers,sector_dim] = assign_sectors(im_width,sector_width,data_coords)
%% params
%im_width = 16;
%sector_width = 8;

%% x, y, z count
sectors = (im_width / sector_width);
sector_count = (sectors).^3;
sector_dim = sector_count;
%% sector_splitting in one dimension
sector_ranges = linspace(-0.5,0.5,sectors+1);

%% test values 
test_data_x = data_coords(1,:);
test_data_y = data_coords(2,:);
test_data_z = data_coords(3,:);

%test_data_x = [-0.3, -0.1, 0, 0.5, 0.3];
%test_data_y = [ 0.2,    0, 0,   0, 0.3];
%test_data_z = [ 0.2,    0,   0,   0, 0];

r1 = [sector_ranges(1:end-1)] ;
r2 = [sector_ranges(2:end)] ;
r2(end)=r2(end)+0.01; %workaround damit wert bei 0.5 nicht weg fällt

q1 = bsxfun(@ge,test_data_x(:),r1(:).');
q2 = bsxfun(@lt,test_data_x(:),r2(:).');

Rx = q1 & q2;

q1 = bsxfun(@ge,test_data_y(:),r1(:).');
q2 = bsxfun(@lt,test_data_y(:),r2(:).');

Ry = q1 & q2;

q1 = bsxfun(@ge,test_data_z(:),r1(:).');
q2 = bsxfun(@lt,test_data_z(:),r2(:).');

Rz = q1 & q2;
%% dec2bin

data_sector = zeros(1,length(test_data_x));

base_x = (2.^(2*(log2(sectors))));
base_y = (2.^(1*(log2(sectors))));
base_z = (2.^(0*(log2(sectors))));

for n = 1:length(test_data_x)
   temp = base_x * (default_value(find(Rx(n,:))-1)) ...
                    + base_y * (default_value(find(Ry(n,:))-1)) ...
                    + base_z * (default_value(find(Rz(n,:))-1));
   data_sector(n) =   temp;
end

data_sector;

sector_centers = zeros(1,3*sector_count);

for x = 0: sectors-1
    for y = 0:sectors-1
        for z = 0:sectors-1
        sector_centers(3*(z+sectors*(y + sectors*x))+1) = (x)*sector_width +  floor(sector_width / 2);
        sector_centers(3*(z+sectors*(y + sectors*x))+2) = (y)*sector_width +  floor(sector_width / 2);
        sector_centers(3*(z+sectors*(y + sectors*x))+3) = (z)*sector_width +  floor(sector_width / 2);
        end
    end
end
dec2bin(data_sector);

