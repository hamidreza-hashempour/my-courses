function [ new_weights , new_Delta] = Update_weights_momentum( data, label, predict, hidden_layer_output,weights, alpha, prev_Delta, etta)

% Error = ( label - predict )^2;

delta_3 = -2 * ( predict - label ) .* Derive_function( predict ); 
delta_2 = (delta_3 * transpose( weights{2} )) .* Derive_function( hidden_layer_output);

delta_w2 = transpose( hidden_layer_output ) * delta_3;
delta_w1 = transpose( data ) * delta_2;

[data_size, ~] = size(data);

new_Delta{1} = delta_w1/data_size;
new_Delta{2} = delta_w2/data_size;

new_weights{1} = weights{1} + alpha * delta_w1/data_size + etta * prev_Delta{1};
new_weights{2} = weights{2} + alpha * delta_w2/data_size + etta * prev_Delta{2};

end