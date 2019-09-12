function [ output, hidden_layer_output ] = Neural_Network( data, weights)

hidden_layer_output = Sigmoid( data * weights{1} );    

output = Sigmoid( hidden_layer_output * weights{2});

end