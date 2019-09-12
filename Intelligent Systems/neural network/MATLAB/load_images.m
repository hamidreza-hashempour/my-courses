function [images] = load_images(file_name)
    file=fopen(file_name,'rb');
    assert(file ~= -1, ['Could not open ', file_name, '']);
    magic = fread(file, 1, 'int32', 0, 'ieee-be');
    number_of_Images = fread(file, 1, 'int32', 0, 'ieee-be');
    number_of_rows = fread(file, 1, 'int32', 0, 'ieee-be');
    number_of_cols = fread(file, 1, 'int32', 0, 'ieee-be');
    images = fread(file, inf, 'unsigned char');
    images = reshape(images, number_of_cols, number_of_rows, number_of_Images);
    images = permute(images,[2 1 3]);
    % Reshape to #pixels x #examples
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
    % Convert to double and rescale to [0,1]
    images = double(images) / 255;
end

