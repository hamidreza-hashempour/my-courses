function [ out] = Majority_Voting(data, label, tree)

predict = cell(1,  max(size(tree)));
our_label = zeros(max(size(label)), 1);

for k=1:max(size(tree))
    
    prediction = zeros(max(size(label)), 1);
    
    for i=1:max(size(label))
        
        prediction(i, 1) = Predict(data(i, :), tree{1, k});
        
    end
    
    predict{1, k} = prediction;
    
end

for i=1:max(size(label))
    
    predict_label = zeros(1, max(size(tree)));
    
    for k=1:max(size(tree))
        
        predict_label(1, k) = predict{1, k}(i, 1);
        
    end
    
    unique_label = unique(predict_label);           %give unique labels in label array
    Ncount = histc(predict_label, unique_label);    %the number of repeatation
    
    [~, index] = max(Ncount);
    our_label(i, 1) = unique_label(index);
        
end

error = label - our_label;

out = 100 * ( 1 - max(size(find(error))) / max(size(label))) ;
