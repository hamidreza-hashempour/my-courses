function [output] = relu_activation_function(input)
    if(input>0)
        output=input;
    else
        output=0;
    end
end

