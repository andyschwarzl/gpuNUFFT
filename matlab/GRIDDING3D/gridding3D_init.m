function [res] = gridding3D_init(k,n,osf,sw)
coords = k;%[real(k(:))'; imag(k(:))';zeros(1,length(k(:)))];
[test_coords, test_sector_centers,sector_dim] = assign_sectors(osf*n,sw,coords);
res.coords = coords;
[v i] = sort(test_coords);
v = v +1;
sectors_test = zeros(1,sector_dim+1);
cnt = 0;
for b=1:sector_dim+1
    while (cnt < length(v) && b == int32(v(cnt+1)))
        cnt = cnt +1;
    end
    sectors_test(b)=cnt;
end
res.sectors_test = sectors_test;
%sectors_test
% calculate indices of data elements in order to sort them
data_ind = i-1;
res.data_ind=[2*data_ind+1;2*data_ind+2];

% calculate indices of coord elements in order to sort them
coord_ind = i-1;
res.coord_ind = [3*coord_ind+1;
             3*coord_ind+2;
             3*coord_ind+3];
%
res.test_sector_centers = int32(reshape(test_sector_centers,[3,sector_dim]));

