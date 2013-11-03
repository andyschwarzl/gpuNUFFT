function [data_sector_idx,sector_centers,sector_dim] = assign_sectors(im_width,sector_width,data_coords)
%function [data_sector_idx,sector_centers,sector_dim] = assign_sectors(im_width,sector_width,data_coords)
%assigns data points to sectors with dimension given by sector_dim.
%
%--------------------------------------------------------------------------
% INPUT
%--------------------------------------------------------------------------
% im_width       |
% sector_width   |
% data_coords    |   assumes data_coords scaled between -0.5 and 0.5 
%                |
%--------------------------------------------------------------------------
% OUTPUT         |     
%--------------------------------------------------------------------------
%                |
% data_sector_idx|   vector containing the assigned sector index according to input data_coords (1:n_sectors)
% sector_centers |   vector containing the sector centers in (x,y,z) in grid  units
% sector_dim     |   total count of sectors, calculated via im_width / sector_width 
%--------------------------------------------------------------------------

%% x, y, z count
if (mod(im_width,sector_width) ~= 0)
    error('ERROR: GRID width %d (image width * OSR) is no multiple of sector width %d',im_width,sector_width);
end 

sectors = (im_width / sector_width);

sector_count = (sectors).^3;
sector_dim = sector_count;

%% sector ranges in one dimension
sector_ranges = linspace(-0.5,0.5,sectors+1);

%% split coordinate values into extra vectors
x_coords = data_coords(1,:);
y_coords = data_coords(2,:);
z_coords = data_coords(3,:);

r1 = [sector_ranges(1:end-1)] ;
r2 = [sector_ranges(2:end)] ;
%delta is needed in order to assign correctly at sector range edges
%e.g.
%coords [-0.5,-0.4)[-0.4,0.3]...[0.4,0.5)
%sector      0          1    ...    n-1 
delta = 1 / im_width / 2; 
r2(end)=r2(end)+2*delta; %workaround damit wert bei 0.5 nicht weg fällt

q1 = bsxfun(@ge,x_coords(:)+delta,r1(:).');
q2 = bsxfun(@lt,x_coords(:)+delta,r2(:).');

Rx = q1 & q2;

q1 = bsxfun(@ge,y_coords(:)+delta,r1(:).');
q2 = bsxfun(@lt,y_coords(:)+delta,r2(:).');

Ry = q1 & q2;

q1 = bsxfun(@ge,z_coords(:)+delta,r1(:).');
q2 = bsxfun(@lt,z_coords(:)+delta,r2(:).');

Rz = q1 & q2;

data_sector_idx = zeros(1,length(x_coords));

base_x = (2.^(2*(log2(sectors))));
base_y = (2.^(1*(log2(sectors))));
base_z = (2.^(0*(log2(sectors))));

for n = 1:length(x_coords)
   temp = base_x * (default_value(find(Rx(n,:))-1)) ...
                    + base_y * (default_value(find(Ry(n,:))-1)) ...
                    + base_z * (default_value(find(Rz(n,:))-1));
   data_sector_idx(n) =   temp;
end

%calculate sector centers -> needed for computational purpose
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

