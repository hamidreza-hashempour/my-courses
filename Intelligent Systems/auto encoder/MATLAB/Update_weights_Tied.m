function [ new_weights , new_Delta] = Update_weights_Tied( data, label, predict, hidden_layer_output,weights, alpha, prev_Delta, etta)

% Error = ( label - predict )^2;

delta_3 = -2 * ( predict - label ) .* Derive_function( predict ); 
delta_2 = (delta_3 * transpose( weights{2} )) .* Derive_function( hidden_layer_output);

delta_w2 = transpose( hidden_layer_output ) * delta_3;
delta_w1 = transpose( data ) * delta_2;

delta_w = delta_w1 + transpose(delta_w2);

[data_size, ~] = size(data);

new_Delta = delta_w;

new_weights{1} = weights{1} + alpha * delta_w/data_size + etta * prev_Delta;
new_weights{2} = transpose(new_weights{1});

end