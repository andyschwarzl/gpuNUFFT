function ismrm_imshow(image_matrix, scale, tile_shape, titles)
%
%  ismrm_imshow(image_matrix, [low high], [image_tile_rows
%  image_tile_columns], titles )
%
%  Displays a collection of images in a tiled figure
%
%  INPUT:
%    - image_matrix    : 2-D, displays single image
%                        3-D, third dimension indexes subimages
%    - [low high]      : low displays as black, high as white
%    - [image_tile_rows image_tile_columns] : specifies how to tile
%                                             subimages in display
%    - titles  : {'title1', 'title2',...}
%
%
%
%   Code made available for the ISMRM 2013 Sunrise Educational Course
% 
%   Michael S. Hansen (michael.hansen@nih.gov)
%   Philip J. Beatty (philip.beatty@sri.utoronto.ca)
%

%%
% Validate input

switch nargin
    case 1
        scale = [];
        tile_shape = [];
        titles = [];
    case 2
        tile_shape = [];
        titles = [];
    case 3
        titles = [];
    case 4
        
    otherwise
        error('valid number of arguments: 1-3');
end

assert( ndims(image_matrix) > 1 && ndims(image_matrix) < 4, 'image_matrix must have 2 or 3 dimensions')

if isempty(scale),
    scale = [min(image_matrix(:)), max(image_matrix(:))];
end

if isempty(tile_shape)
    tile_shape = [1 size(image_matrix,3)];
end
assert( prod(tile_shape) == size(image_matrix, 3), 'image tile rows x columns must equal the 3rd dim extent of image_matrix');

if ~isempty(titles),
    assert(numel(titles) == size(image_matrix,3), 'number of titles must equal 3rd dim extent of image_matrix');
end

%%
% Set figure shapes and strides
border = 2; % pixels
size_scale = 1.0;

num_rows = tile_shape(1);
num_cols = tile_shape(2);

im_shape = [size(image_matrix,1) size(image_matrix,2)] * size_scale;
im_stride = im_shape + border;
frame_shape = im_stride .* tile_shape - border;

% normalized shape and strides for subplot function
im_norm_shape = im_shape ./ frame_shape;
im_norm_stride = im_stride ./ frame_shape;

if isempty(titles),
    title_margin = 0;
else
    title_margin = num_rows*50;
end

figure_handle = figure;
set(figure_handle,'Units','Pixels');
set(figure_handle,'Position',[0 0 frame_shape(2) frame_shape(1)+title_margin]);
set(figure_handle,'Color',[1 1 1]);


%%
% Traverse & display images
curr_index = 1;
for row_index = 1:num_rows,
    for col_index = 1:num_cols,
        subplot('Position',[im_norm_stride(2) * (col_index-1) im_norm_stride(1) * (num_rows - row_index)  im_norm_shape(2) im_norm_shape(1)]);
        imshow(image_matrix(:,:,curr_index), scale);
        if ~isempty(titles),
            title(titles(curr_index));
        end
        curr_index = curr_index + 1;
    end
end

