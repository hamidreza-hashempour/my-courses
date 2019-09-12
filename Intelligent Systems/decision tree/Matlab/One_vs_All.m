function [new_label] = One_vs_All(label, key)

new_label = zeros(max(size(label)), 1);

proper_Data = ismember(label, key); 
index = find(proper_Data);

new_label(index, :) = key;

end