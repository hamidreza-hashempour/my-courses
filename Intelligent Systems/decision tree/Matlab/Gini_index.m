function [ out ] = Gini_index( probibility )

data_size = size(probibility);
data_size = data_size(1);             %number of datas

out = 1;

for i=1:data_size
    
    out = out - probibility(i) .* probibility(i);
    
end
end

