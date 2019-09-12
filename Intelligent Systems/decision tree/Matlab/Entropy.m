function [ entropy ] = Entropy( probibility )

data_size = size(probibility);
data_size = data_size(1);             %number of datas

entropy = 0;

for i=1:data_size
    
    entropy = entropy - probibility(i) * log2(probibility(i));
    
end
end

