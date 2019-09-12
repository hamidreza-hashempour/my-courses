function weights = initial(layers)


  weights = cell(1, 2);
    
  for i = 1:length(layers)-1
      Min_Val=-1;
      Max_Val=1;
    
    weights{i} = rand(layers(i), layers(i+1)) .* (Max_Val-Min_Val)+Min_Val;
  end
end