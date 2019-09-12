function [ tree ] = ID3_GINI_Overfit( data, label, id, values, depth)

unique_label_ = unique(label);           %give unique labels in label array
Ncount = histc(label, unique_label_);    %the number of repeatation

[m, index] = max(Ncount);
tree{1,1}.value = values;
 
if m==sum(Ncount)
    
    tree{1,1}.label = unique_label_(index);
    tree{1,1}.attr = -1;
    return
    
end

if(min(size(id))==0)
    
    tree{1,1}.label = unique_label_(index);
    tree{1,1}.attr = -1;
    return
    
end

if(depth >= 6 )
    
    tree{1,1}.label = unique_label_(index);
    tree{1,1}.attr = -1;
    return

end

Information_Gain = GINI(data, label);
tree{1,1}.attr = id(Information_Gain);
tree{1,1}.label = -1;

unique_label = unique(data(:, Information_Gain));           %give unique labels in label array

for i=1:max(size(unique_label))
    
    [new_data, new_label, new_id] = Prepare_Data( data, label, unique_label(i), Information_Gain, id);
    [ tree{i,2} ] = ID3_GINI_Overfit(new_data, new_label, new_id, unique_label(i), depth+1);
    
end
end
