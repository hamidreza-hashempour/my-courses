function Accuracy=testingdata(Test_Attr,weights,Test_Label,Hidden_Func)
Accuracy=0;
for i=1:size(Test_Attr,1)
        Each_Test_Data=Test_Attr(i,:);
    [output, values] = measuring(weights, Each_Test_Data,Hidden_Func);
    [Max_Value,Max_Index]=max(output);
    Neural_Decide_Label=Max_Index-1;
        if (Neural_Decide_Label==Test_Label(i))
            Accuracy=Accuracy+1;
        end
end
end