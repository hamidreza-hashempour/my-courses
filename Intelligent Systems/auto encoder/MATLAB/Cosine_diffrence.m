function [ diffrence ] = Cosine_diffrence( data, centroid)

[data_size, ~] = size(data);
data_norm = zeros(data_size, 1);

for i=1:max(size(data))
    
    data_norm(i) = norm(data(i, :));    % calculate norm of data
end

center_norm = norm(centroid);           % calculate norm of center(test)

similiraty = data*centroid' ./ data_norm / center_norm;      % similitary with Cosien method
diffrence = 1 - similiraty;                                  % because of finding minimum

end