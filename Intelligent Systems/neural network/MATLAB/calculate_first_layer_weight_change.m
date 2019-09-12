function [weight_changes] = calculate_first_layer_weight_change(x,h,w,delta_output)
    sum_w_deltaout = delta_output*w;
    delta_hidden = h.*(1-h).*sum_w_deltaout; %for sigmoid
%     delta_hidden = (1-h.^2).*sum_w_deltaout; %for tanh
%     delta_hidden = sum_w_deltaout; % for linear
    weight_changes = delta_hidden'*x;
end

