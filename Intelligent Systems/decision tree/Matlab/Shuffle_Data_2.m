function [ new_data, new_label] = Shuffle_Data_2(data, label, k)

new_data = cell(k, 1);
new_label = cell(k, 1);

num_data = randperm(max(size(data)));


for i=1:k
    
   new_data{i, 1} = data(num_data((i-1)*max(size(data))/k+1:i*max(size(data))/k), :);
   new_label{i, 1} = label(num_data((i-1)*max(size(data))/k+1:i*max(size(data))/k));
   
end
end