function [weight_change,delta_outputs] = calculate_second_layer_weight_change(h,t,y)
    delta_outputs = y.*(1-y).*(t-y); %for sigmoid activation
%     delta_outputs = (1-y.^2).*(t-y); %for tanh
%     delta_outputs = t-y; %for linear
    weight_change=delta_outputs'*h;
end

