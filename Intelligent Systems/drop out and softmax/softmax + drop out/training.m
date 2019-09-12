function [weights,Total_Error_Matrix,Error_On_Test,FinalAccuracy] = training(weights, repeat, train_attr, target,Mini_Batch_Num,Test_Attr,Test_Label,Learning_Rate,Hidden_Func)
  LEARNINGRATE = Learning_Rate;

  display('In Train...');
  for iter = 1:repeat
    wmwa=1;
   


    totalError = 0;
    W2_ = weights{2}; 
    W1_ = weights{1};
    sets = size(train_attr, 1);
    num_of_datas_in_minibatch=sets/Mini_Batch_Num;
    for batchset=1:Mini_Batch_Num
        
    delta2_W_Mean_softmax=zeros(size(W2_,1),size(W2_,2));
    delta1_W_Mean_softmax=zeros(size(W1_,1),size(W1_,2));

    for set = wmwa:1:(num_of_datas_in_minibatch+wmwa-1)
      t = target(set, :); % target output regarding current input

      [~, values] = measuring(weights, train_attr(set, :),Hidden_Func);
      totalError = totalError + sum(((-t.*log10(values{4})-(1-t).*log10(1-values{4}))))/Mini_Batch_Num;
      o_softmax = values{4};
      Dsoftmax=zeros(1,length(o_softmax));
      
      output1 = values{2}; % hidden layer values
      output  = values{1}; % input values
      for Leng_Soft=1:length(o_softmax)
      Dsoftmax(Leng_Soft)=(sum(o_softmax)-o_softmax(Leng_Soft)) * o_softmax(Leng_Soft);
      end
      Dsoftmax=diag(Dsoftmax);
      if Hidden_Func==1
      D1 = diag(output1 .* (1 - output1)); % matrix of derivatives for hidden layer
      end
      if Hidden_Func==2
        D1 = diag(1-output1.^2); % matrix of derivatives for hidden layer  
      end
      W2 = weights{2}; % weights from hidden to output
      W1 = weights{1}; % weights from input to hidden

      e_softmax = -(t./o_softmax) - ((t-1)./(1-o_softmax)) ;
      e_softmax= e_softmax(:);
      
      delta2_softmax = Dsoftmax * e_softmax;
      delta1_softmax = D1 * W2 * delta2_softmax;
      
      correctionsW2_softmax = -LEARNINGRATE * delta2_softmax * output1;
      correctionsW1_softmax = -LEARNINGRATE * delta1_softmax * output;
      
      delta2_W_Mean_softmax=delta2_W_Mean_softmax+correctionsW2_softmax';
      delta1_W_Mean_softmax=delta1_W_Mean_softmax+correctionsW1_softmax';
    end
      delta2_W_Mean_softmax=delta2_W_Mean_softmax/num_of_datas_in_minibatch;
      delta1_W_Mean_softmax=delta1_W_Mean_softmax/num_of_datas_in_minibatch;
      

      % Weight updates.
      weights{2} = W2 + delta2_W_Mean_softmax;
      weights{1} = W1 + delta1_W_Mean_softmax;
      wmwa=wmwa+num_of_datas_in_minibatch;
    end
    Total_Error_Matrix(iter)=totalError;
    Accuracy=testingdata(Test_Attr,weights,Test_Label,Hidden_Func);
    Error_On_Test(iter)=1-(Accuracy/size(Test_Attr,1));
  end
    FinalAccuracy=testingdata(Test_Attr,weights,Test_Label,Hidden_Func);
   

end