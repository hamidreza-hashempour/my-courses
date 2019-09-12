function[v,w]= neural_network_vectorized_train(v,w)
tic();
train_data = load('train_images.mat');
    train_data = double(train_data.train_images);
    train_label = load('train_lables.mat');
    train_label = double(train_label.train_lables);
    size(train_label)
    target = zeros(60000,4);
%   target = de2bi(train_label,4);
    for i=1:60000
        target(i,train_label(i)+1)=1;
    end
    epoch_count = 200;
    losses = zeros(epoch_count,1);
    test_losses = zeros(epoch_count,1);
    test_acc = zeros(epoch_count,1);
    train_acc = zeros(epoch_count,1);
    train_mean = mean(train_data);
    standard_daviation = std(train_data);
    tic()
    train_data = train_data';
    train_data = mapstd(train_data);
    train_data = train_data';
    %train_data = train_data./255;
    v1=0;
    v2=0;
    momentum=2;
    for epoch=1:epoch_count
        second_layer_input = train_data*v';
        hidden_layer_output = sigmoid_activation_function(second_layer_input);
%       hidden_layer_output = tanh(second_layer_input);
%       hidden_layer_output = second_layer_input;
        third_layer_input = hidden_layer_output * w';
        final_output = sigmoid_activation_function(third_layer_input);
%       final_output = tanh(third_layer_input);
        %final_output = third_layer_input;
        
% ---- one hot encoding        
        [~,max_index] = max(final_output,[],2);
        max_index = max_index';
        train_acc(epoch)=sum(max_index-1==train_label)/60000;
% ----  binary encoding
%       result = (bi2de(round(final_output)))';
%       train_acc(epoch)=sum(result==train_label)/60000;
        
        delta_outputs = final_output.*(1-final_output).*(target-final_output);
%       delta_outputs = (1-final_output.^2).*(target-final_output);
%       delta_outputs = (target-final_output);
        second_layer_weight_changes = delta_outputs'*hidden_layer_output;
        first_layer_weight_changes = (hidden_layer_output.*(1-hidden_layer_output).*(delta_outputs*w))'*train_data;
%       first_layer_weight_changes = ((1-hidden_layer_output.^2).*(delta_outputs*w))'*train_data;
%       first_layer_weight_changes = ((delta_outputs*w))'*train_data;
        v1 = momentum*v1 + (second_layer_weight_changes/60000);
        v2 = momentum*v2 + (first_layer_weight_changes/60000);
        w = w +(4)*v1;
        v = v + (4)*v2;
        losses(epoch)=(sum(sum((target-final_output).^2,2)/2))/60000;
        [test_losses(epoch), test_acc(epoch)]= neural_network_test(v,w);
    end
    test_acc(epoch)
    train_acc(epoch)
%     subplot(2,2,1);
%     plot(losses);
%     xlabel('epoch');
%     ylabel('train loss');
%     subplot(2,2,2);
%     plot(test_losses);
%     xlabel('epoch');
%     ylabel('test loss');
%     subplot(2,2,3);    
%     plot(test_acc);
%     xlabel('epoch');
%     ylabel('test acc');
%     subplot(2,2,4);
%     plot(train_acc);
%     xlabel('epoch');
%     ylabel('train acc');
    toc();
end
