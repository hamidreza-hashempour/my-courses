function [ Q , Q_avg] = Q_matrix(R, alpha, gamma, epsilon, initial_state, number_iteration, goal_state)

total_state = max(size(R));
Q = zeros(total_state, total_state);
Q_avg = zeros(number_iteration, 1);

for i=1:number_iteration
    
    temp = 0;
    move = 0;
    
    if mod(i, 2) == 0
        index = randperm(total_state, 1);
        current_state =  state(index, index, int8(log(total_state)/log(3)));
    else
        current_state = initial_state;
    end
    
    while nnz(current_state - goal_state) ~= 0
        
        if rand >= epsilon

             possible_moves = find(R(position(current_state), :)>-inf);
            [~, index] = max(Q(position(current_state), possible_moves));
            index = possible_moves(index);
       
        else
            
            possible_move = find(R(position(current_state), :)>-inf);

            index = randperm(max(size(possible_move)), 1);
            index = possible_move(index);
            
        end

        Q_max = max(Q(index, :));
        Q(position(current_state), index) = Q(position(current_state), index) + alpha * ( R(position(current_state), index) + gamma * Q_max - Q(position(current_state), index) );
        temp = temp +  Q(position(current_state), index);
        move = move + 1;
        current_state = state(index, index, int8(log(total_state)/log(3)));
        

    end
    Q_avg(i, 1) = temp / move;
end
end