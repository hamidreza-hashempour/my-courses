function [output] = sigmoid_activation_function(input)
    output = 1./(1+exp(-input));
end

