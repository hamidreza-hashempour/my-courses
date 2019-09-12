function [ output ] = label_finder( Data, Label, k )

[data_size, ~] = size(Data);

if( data_size == 1 )     % check just has one data
    
    output = Label;
    return
    
end

label = zeros(k,1);
[Max,~] = max(Data);

for i=1:k
    
   [~, label(i)] = min(Data);           % find index of smallest distance
   Data(label(i)) = Max;                % replace it so next round again dont choose this index
   
end
    
label = Label(label);                   % find their Label with their index
unique_label = unique(label);           % give unique labels in label array
Ncount = histc(label, unique_label);    % the number of repeatation

[~, index] = max(Ncount);               %find best label for test data
output = unique_label(index);

end