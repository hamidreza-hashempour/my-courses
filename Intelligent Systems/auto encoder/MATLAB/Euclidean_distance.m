function [ distance ] = Euclidean_distance( data, centroid)

[data_size, ~] = size(data);             % number of datas

test = ones(data_size, 1) * centroid;    % make it as same as size data so can do next line

distance = sqrt(sum((data-test).^2,2));  % calculate each distance from test point to all train point
                                         % so we need sum in each row ->
                                         % so use sum ( matrix, 2)                                 
end