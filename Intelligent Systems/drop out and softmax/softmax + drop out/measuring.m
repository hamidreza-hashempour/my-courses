function [output, layer_output] = measuring(weights, input,Hidden_Func)


  layer_output = cell(1, length(weights)+2);
  Into_Last_Layer=zeros(1,10);
  SoftMax_Out=zeros(1,10);
  % Save input values into first cell.
  layer_output{1} = input(:)';

 
  for i = 1:length(weights)

    if Hidden_Func==1
    layer_output{i+1} = 1./(1+exp(-( layer_output{i}(:)' * weights{i} )));
    end
    if Hidden_Func==2
        layer_output{i+1} = tanh( layer_output{i}(:)' * weights{i} );
    end
  end
  Into_Last_Layer=layer_output{i}(:)' * weights{i};
  for j=1:length(Into_Last_Layer)
  SoftMax_Out(1,j)=(exp(Into_Last_Layer(j))/(sum(exp(Into_Last_Layer))));
  end
  layer_output{4}=SoftMax_Out;
  % Return output layer values.
  output = layer_output{end};
end