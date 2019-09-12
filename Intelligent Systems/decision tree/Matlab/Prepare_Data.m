function [ new_data, new_label, new_id] = Prepare_Data(data, label, key, selected, id)

proper_Data = ismember(data(:, selected), key); 
index = find(proper_Data);

new_label = label(index, :);

new_data = data(index, :);
new_data(:,selected) = [];

new_id = id;
new_id(:, selected) = [];

end