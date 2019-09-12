function [confiusion_matrix]=makingCmatrix(weights,Test_Attr,Test_Label,Hidden_Func)
    confiusion_matrix=zeros(10,10);
for i=1:size(Test_Attr,1)
        Each_Test_Data=Test_Attr(i,:);
    [output, values] = measuring(weights, Each_Test_Data,Hidden_Func);
    [Max_Value,Max_Index]=max(output);
    Neural_Decide_Label=Max_Index-1;
        confiusion_matrix(Neural_Decide_Label+1,Test_Label(i)+1)=confiusion_matrix(Neural_Decide_Label+1,Test_Label(i)+1)+1;
end
    
end