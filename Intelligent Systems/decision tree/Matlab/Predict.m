function [ out] = Predict(data, tree)

if(tree{1,1}.label ~= -1)
    
    out = tree{1,1}.label;
    return
    
elseif(tree{1,1}.attr ~= -1)
        
    index = tree{1,1}.attr;
    key = data(:, index);
    
    size_tree = size(tree);
    
    if(min(size_tree) == 1)
        size_tree = 1;
    else
        size_tree = max(size(tree));
    end
    
    num = zeros(size_tree, 1);
    
    for i=1:size_tree
        
        copy_tree = tree{i, 2};
        num(i) = copy_tree{1,1}.value;
        
        if(copy_tree{1,1}.value == key)
            
            tree = tree{i,2};
            [out] = Predict(data, tree);
            return
        end
    end
    if(i==size_tree)
            
            n_num = abs(num - key);
            [~, index_min] = min(n_num);
            tree = tree{index_min,2};
            [out] = Predict(data, tree);
            return
    end
end
end