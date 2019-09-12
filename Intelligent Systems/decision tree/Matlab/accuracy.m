function [ out] = accuracy(data, label, tree)

prediction = zeros(max(size(label)), 1);

for i=1:max(size(label))
    
    prediction(i, 1) = Predict(data(i, :), tree);
    
end

error = label - prediction;

out = 100 * ( 1 - max(size(find(error))) / max(size(label))) ;
