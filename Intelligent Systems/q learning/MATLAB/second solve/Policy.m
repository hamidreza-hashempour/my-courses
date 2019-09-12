function [ output ] = Policy(Q, initial_state, goal_state)

total_state = max(size(Q));
disk_number = int8(log(total_state)/log(3));

current_state =  initial_state;

output = current_state;

while nnz(current_state - goal_state) ~= 0
    
    [~, index] = max(Q(position(current_state), :));
    current_state = state(index, index, disk_number);
    output = [output; current_state]; 
    
end
end