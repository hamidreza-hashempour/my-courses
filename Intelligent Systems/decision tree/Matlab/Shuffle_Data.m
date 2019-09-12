function [ new_data, new_label, new_id] = Shuffle_Data(data, label, k, id)

new_data = cell(k, 1);
new_label = cell(k, 1);
new_id = cell(k, 1);

num_attr = randperm(min(size(data)));
num_data = randperm(max(size(data)));


for i=1:k
    
   new_data{i, 1} = data(num_data((i-1)*max(size(data))/k+1:i*max(size(data))/k), :);
   delete_num = id;
   select_attr = find(ismember(id, num_attr((i-1)*k+1:i*k)));
   delete_num(select_attr) = []; 
   new_data{i, 1}(:, delete_num) = [];
   
   new_label{i, 1} = label(num_data((i-1)*max(size(data))/k+1:i*max(size(data))/k));
   new_id{i, 1} = num_attr((i-1)*k+1:i*k);
   
end
end