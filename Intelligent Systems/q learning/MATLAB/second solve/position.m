function [ x ] = position(state)

disk_number = max(size(state));
x = 0;

for i=1:disk_number

    x = x + state(disk_number - i + 1) * power(3, i-1);
    
end

x = x + 1;

end