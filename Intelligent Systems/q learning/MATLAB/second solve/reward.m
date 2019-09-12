function [ output ] = reward(first, second)

if nnz( first-second ) ~= 1
    
    output = -inf;
    return
end

index = find(first - second);

for i=1:index-1
    if first(i) == first(index)
       output = -inf;
       return 
    end
    if second(i) == second(index)
        
       output = -inf;
       return 
    end
end

if unique(second) == 2
    output = 100;
else
    output = -0.01;
end
end