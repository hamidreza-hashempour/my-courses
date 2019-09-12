function[v,w]= neural_network_train(v,w)
    train_data = load('train_images.mat');
    train_data = double(train_data.train_images);
    train_label = load('train_lables.mat');
    train_label = double(train_label.train_lables);
    epoch_count =6;
    losses=zeros(epoch_count);
    test_losses = zeros(epoch_count);
    test_acc = zeros(epoch_count);
    train_acc = zeros(epoch_count);
%     v1=0;
%     v2=0;
%     momentum=0.9;
    for epoch=1:epoch_count
        epoch_acc=0;
        tic();
%%      stochastic
        mean_loss=0;
         for image_index=1:60000
             image=train_data(image_index,:)/255;
             target = zeros(1,10);
             target(train_label(image_index)+1)=1;
%              target = de2bi(train_label(image_index),4);
             second_layer_input = perceptron_feedforward(image,v');
             hidden_layer_output = sigmoid_activation_function(second_layer_input);
%              hidden_layer_output = tanh(second_layer_input);
%              hidden_layer_output = second_layer_input;
             third_layer_input = perceptron_feedforward( hidden_layer_output,w');
             final_output = sigmoid_activation_function(third_layer_input);
%              final_output = tanh(third_layer_input);
%              final_output = third_layer_input;

             [~,max_index] = max(final_output);
             if(max_index-1==train_label(image_index))
                epoch_acc=epoch_acc+1;
                
%%            binary encoding
%               result = (bi2de(round(final_output)));
%               if(result==train_label(image_index))
%                   epoch_acc=epoch_acc+1;
              
             end
             [second_layer_weight_changes,delta_outputs] = calculate_second_layer_weight_change(hidden_layer_output,target,final_output);
             first_layer_weight_changes = calculate_first_layer_weight_change(image,hidden_layer_output,w,delta_outputs);
             w = w + (0.1)*(second_layer_weight_changes);
             v = v + (0.1)*first_layer_weight_changes;
             loss=sum((target-final_output).^2)/2;
             mean_loss = mean_loss + loss;
         end
         losses(epoch)=mean_loss/60000;
         train_acc(epoch)=epoch_acc/60000;
         [test_losses(epoch),test_acc(epoch)]=neural_network_test(v,w);
%%         batch  
%          first_layer_mean_change=zeros(64,784);
%          second_layer_mean_change=zeros(10,64);
%          epoch_loss = 0;
%          for image_index=1:60000
%             image=train_data(image_index,:)/255;
%             target = zeros(1,10);
%             target(train_label(image_index)+1)=1;
%             second_layer_input = perceptron_feedforward(image,v');
%             hidden_layer_output = sigmoid_activation_function(second_layer_input);
%             third_layer_input = perceptron_feedforward( hidden_layer_output,w');
%             final_output = sigmoid_activation_function(third_layer_input);
%             [second_layer_weight_changes,delta_outputs] = calculate_second_layer_weight_change(hidden_layer_output,target,final_output);
%             first_layer_weight_changes = calculate_first_layer_weight_change(image,hidden_layer_output,w,delta_outputs);
%             first_layer_mean_change=first_layer_mean_change+first_layer_weight_changes; 
%             second_layer_mean_change=second_layer_mean_change+second_layer_weight_changes;
%             loss=sum((target-final_output).^2)/2;
%             epoch_loss = epoch_loss + loss;
%          end
%          w = w +(0.3)*(second_layer_mean_change/60000);
%          v = v + (0.3)*(first_layer_mean_change/60000);
%          losses(epoch)=epoch_loss/60000;
    toc();
    end
    test_acc(epoch)
    train_acc(epoch)
%     subplot(2,2,1);
%     plot(losses);
%     xlabel('epoch');
%     ylabel('train loss');
%     subplot(2,2,2);
    plot(test_losses);
    xlabel('epoch');
    ylabel('test loss');
%     subplot(2,2,3);    
%     plot(test_acc);
%     xlabel('epoch');
%     ylabel('test acc');
%     subplot(2,2,4);
%     plot(train_acc);
%     xlabel('epoch');
%     ylabel('train acc');
end

