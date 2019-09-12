function [ R ] = R_matrix( disk_number )

number_state = 3 ^ disk_number;
R = zeros(number_state, number_state);

for x = 1:number_state 
    for y = 1:number_state
        
        [x_prime, y_prime] = state(x, y, disk_number);
        R(x, y) = reward(x_prime, y_prime);
    end
end
end