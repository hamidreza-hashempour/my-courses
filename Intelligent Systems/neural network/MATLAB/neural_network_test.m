function[test_loss, accuracy]= neural_network_test(v,w)
    test_data=load('test_images.mat');
    test_data=double(test_data.test_images);
    test_labels=load('test_labels');
    test_labels=double(test_labels.test_labels);
    correct_answers=0;
    mean_loss=0;
    confusion_matrix=zeros(10);
    for test_image_index=1:10000
        image = test_data(test_image_index,:);
%       target=de2bi(test_labels(test_image_index),4);
        target = zeros(1,10);
        target(test_labels(test_image_index)+1)=1;
        second_layer_input = perceptron_feedforward(image,v');
        hidden_layer_output = sigmoid_activation_function(second_layer_input);
%       hidden_layer_output = tanh(second_layer_input);
%       hidden_layer_output = second_layer_input;
        third_layer_input = perceptron_feedforward( hidden_layer_output,w');
        final_output = sigmoid_activation_function(third_layer_input);
%       final_output = tanh(third_layer_input);
%       final_output = third_layer_input;
        loss=sum((target-final_output).^2)/2;
        mean_loss=mean_loss+loss;
         
        [~,max_index] = max(final_output);
        if(max_index-1==test_labels(test_image_index))
            correct_answers=correct_answers+1;
        end
        confusion_matrix(max_index,test_labels(test_image_index)+1)=confusion_matrix(max_index,test_labels(test_image_index)+1)+1;
%         result = (bi2de(round(final_output)));
%             if(result==test_labels(test_image_index))
%                   correct_answers=correct_answers+1;
%             end
    
    end
    confusion_matrix
    accuracy = correct_answers/10000;
    test_loss = mean_loss/10000;
end

