function [ error ] = KNN( train_data, train_label, test_data, test_label, K)

[~, lenght] = size(K);                                      %number of K for test
[test_size, ~] = size(test_label);                          %number of test datas
error = zeros(lenght,1);

for i=1:lenght
    
    test_predict = KNN_Euclidean(train_data, train_label, K(i), test_data);

    error(i) =  nnz(test_predict-test_label)/test_size*100;         % count which are not zero
    
    disp(['for K = ', num2str(K(i)), ' the error is = ', num2str(error(i)), ' %']);
    
end

[min_value, key] = min(error);      % find the best one in all K
disp(['the best answer is K = ', num2str(K(key)), ' with accuracy = ', num2str(100 - min_value), ' %']);

end