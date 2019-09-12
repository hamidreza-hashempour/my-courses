function [ weights, cost ] = Train_Network_SGD_Base( train_data, input_nodes, hidden_nodes, output_nodes, learning_rate, momentum_rate, number_epoch, batch_size)

tic;
delta{1} = zeros( input_nodes, hidden_nodes);
delta{2} = zeros( hidden_nodes, output_nodes);

weights{1} = (rand( input_nodes, hidden_nodes)*2 - 1)/4;       %random weights
weights{2} = (rand( hidden_nodes, output_nodes)*2 - 1)/4;

data_pre = ( train_data - min(min(train_data)) ) / ( max(max(train_data)) - min(min(train_data)) );              % change range to [0 - 1]

cost = zeros(number_epoch, 1);

for i = 1:number_epoch
    
    if i > 130 && learning_rate > 0.1
        
        learning_rate = learning_rate * 0.995;
        momentum_rate = momentum_rate * 1.003;
                
    end
    
    [data_size, ~] = size(train_data);
    num = randperm(data_size);                          % randomly change data position in array
    epoch_iteraion = data_size / batch_size;
    
    for k = 1:epoch_iteraion
        
        data = data_pre(num(batch_size*(k-1)+1:batch_size*k),:);        % select proper number of data
        [ predict, hidden_layer_output ] = Neural_Network( data, weights);
        [weights, delta] = Update_weights_momentum( data, data, predict, hidden_layer_output,weights, learning_rate, delta, momentum_rate);
        
    end
    
    [ predict, ~ ] = Neural_Network( data_pre, weights);
    cost(i) = sum(sum((predict - data_pre).^2, 2)) / max(size(data_pre));

    if i ~= 1
        disp(['in ', num2str(i), ' th iteraion Cost is ', num2str(cost(i)), ' and the cost has ', num2str(cost(i) - cost(i-1)), ' change']);
    else
        disp(['in ', num2str(i), ' th iteraion Cost is ', num2str(cost(i))]);
    end
end

time = toc;
disp(['Total time of process is   ', num2str(time)]);

end

