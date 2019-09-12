function [ i, j ] = state(x, y, disk_number)

i= [];
j= [];
x_p = x - 1;
y_p = y - 1;

for k = 1:disk_number - 1
    
    i = [mod(x_p,3), i];
    j = [mod(y_p,3), j];
    
    x_p = floor(x_p/3);
    y_p = floor(y_p/3);
        
    if k == disk_number - 1
      
        i = [x_p, i];
        j = [y_p, j];

    end
end
end
