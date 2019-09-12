function [ key ] = GINI( data, label)

feature_size = min(size(data));
out = zeros(feature_size, 1);

for i=1:feature_size
    
    unique_label = unique(data(:, i));           %give unique labels in label array
    
    gini = zeros(max(size(unique_label)), 1);
    num = zeros(max(size(unique_label)), 1);
    
    for k=1:max(size(unique_label))
        
        proper_Data = ismember(data(:, i), unique_label(k)); 
        index = find(proper_Data);
        new_label = label(index, 1);
        
        probibility = Probibility(new_label);
        gini(k) = Gini_index(probibility);
        num(k) = max(size(index));

    end
    
    out(i) = sum(num.*gini)/sum(num);

end

[~, key] = max(out);

end