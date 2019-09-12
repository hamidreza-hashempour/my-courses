function [ p ] = Probibility(data )

unique_label = unique(data);           %give unique labels in label array
Ncount = histc(data, unique_label);    %the number of repeatation
p = Ncount / sum(Ncount) ; 

end