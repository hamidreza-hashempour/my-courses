function [ output ] = Sigmoid( data )

output = 1 ./ ( 1 + exp(-data));

end