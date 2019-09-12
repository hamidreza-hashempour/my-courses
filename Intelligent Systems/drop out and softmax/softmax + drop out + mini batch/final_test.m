%%
%%

train_images=load('train_images.mat');
train_lables=load('train_lables.mat');
test_images=load('test_images.mat');
test_labels=load('test_labels.mat');
Train_Attr=train_images.train_images;
Train_Labels=train_lables.train_lables;
Train_Attr=double(Train_Attr);
Train_Labels=double(Train_Labels);
Test_Attr=double(test_images.test_images);
Test_Label=double(test_labels.test_labels);


%%
%%defining funcs and number if epochs and numbers of datas
    Num_Of_Epoches=input(' inter numbers of epochs you want \n ');
    Hidden_Func=input(' hidden func: for sigmoid func inter 1 and tanh inter 2 \n');
    Learning_Rate=input('inter learning rate \n');

%%
%%random weights
    
    Hidden_Layer_Num=input(' inter hidden layer number \n ');
    Mini_Batch_Num=input(' inter Mini_Batch_Num \n ');
    %Last_Layer_Neu_Num=input(' inter last layer number 4 or 10 \n ');
    My_Layers=[784 Hidden_Layer_Num 10];
    weights = initial(My_Layers);
    
    Out_Target_A_O=Target_Decoding(Train_Labels);
    
    
    [weights,Total_Error_Matrix,Error_On_Test,AccuracyFinal] = training(weights, Num_Of_Epoches, Train_Attr, Out_Target_A_O,Hidden_Layer_Num,Mini_Batch_Num,Test_Attr,Test_Label,Learning_Rate,Hidden_Func);
    
    confiusion_matrix=makingCmatrix(weights,Test_Attr,Test_Label,Hidden_Func);  
    plot(Total_Error_Matrix)
    
    
    
    
    
    
    
    
    
    
    
    