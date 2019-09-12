function [ out ] = Confusion_Matrix(data, label, tree)

prediction = zeros(max(size(label)), 1);

unique_label = unique(label);           %give unique labels in label array
out = zeros(max(size(unique_label)), max(size(unique_label)));

for i=1:max(size(label))
    
    prediction(i, 1) = Predict(data(i, :), tree);
    
    if(prediction(i, 1)==label(i,1))
        
        out(label(i,1), label(i,1)) = out(label(i,1), label(i,1)) + 1;
        
    else
        
        out(label(i,1), prediction(i, 1)) = out(label(i,1), prediction(i, 1)) + 1;
        
    end
    
end
end