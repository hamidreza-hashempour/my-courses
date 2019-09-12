function [ label ] = KNN_Cosine( data, Label, K, test)

[test_size, ~] = size(test);                         %number of datas

label = zeros(test_size, 1);

for i=1:test_size
    
    distance = Cosine_diffrence( data, test(i,:));	%calculate distance each test to all train_data
    label(i) = label_finder(distance, Label, K);	%calculate label of test with KNN
    
end
end