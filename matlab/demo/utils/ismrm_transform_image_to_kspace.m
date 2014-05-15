function [k] = ismrm_transform_image_to_kspace2(img, dim, k_shape)
%
%  [k] = ismrm_transform_image_to_kspace(img, dim)
%
%  Fourier transform from image space to k-space space along a given or all 
%  dimensions
%
%  INPUT:
%    - img     [x,y,..]      : image space data
%    - dim     vector        : Vector with dimensions to transform
%    - k_shape vector        : Set shape of output k-space matrix
%
%  OUPUT:
%    - k       [kx,ky,...]   : Data in k-space (along transformed dimensions)
%
%   Code made available for the ISMRM 2013 Sunrise Educational Course
% 
%   Michael S. Hansen (michael.hansen@nih.gov)
%   Philip J. Beatty (philip.beatty@sri.utoronto.ca)
%

if nargin < 2,
    dim = [];
end    
   
if nargin < 3,
    k_shape = [];
end
   
if isempty(dim),
    dim = ndims(img):-1:1;
end

if isempty(k_shape)
    k_shape = size(img);
end

k = img;
for d=1:length(dim),
    k = transform_one_dim(k, d, k_shape(d));   
end

return

function k = transform_one_dim(img, dim, k_extent)
    k_shape = size(img);
    k_shape(dim) = k_extent;
    
    k_indices = repmat({':'},1, ndims(img));
    k_indices{dim} = (1:k_extent)+bitshift(size(img, dim)-k_extent+1,-1);
        
    k = fftshift(fft(ifftshift(img, dim), [], dim), dim) ./ sqrt(size(img,dim));
    k = k(k_indices{:});
    
     
 return