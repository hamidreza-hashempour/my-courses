function [output, output_of_layers] = measuring(weights, train_attr,Hidden_Func)


  output_of_layers = cell(1, length(weights)+2);
  Into_Last_Layer=zeros(1,10);
  SoftMax_Out=zeros(1,10);

  output_of_layers{1} = train_attr(:)';


  for i = 1:length(weights)

    if Hidden_Func==2
    output_of_layers{i+1} = tanh( output_of_layers{i}(:)' * weights{i} );
    end
    
    if Hidden_Func==1
    output_of_layers{i+1} = 1./(1+exp(-( output_of_layers{i}(:)' * weights{i} )));
    end
    
  end
  Into_Last_Layer=output_of_layers{i}(:)' * weights{i};
  for j=1:length(Into_Last_Layer)
  SoftMax_Out(1,j)=(exp(Into_Last_Layer(j))/(sum(exp(Into_Last_Layer))));
  end
  output_of_layers{4}=SoftMax_Out;

  output = output_of_layers{4};
end