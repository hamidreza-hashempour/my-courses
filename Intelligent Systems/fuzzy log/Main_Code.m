%%
%%
My_Datas=load('EMG_data.csv');
My_Datas(:,1)=[];
Num_Of_Clusters=input('Num of clusters \n ');
M_Fuzzier=input('M_Fuzzier \n ');
Max_Number_Of_Iteration=input('inter Max_Number_Of_iteration \n');
Epsilon=input('inter epsilon \n ');
options=[M_Fuzzier;Max_Number_Of_Iteration;Epsilon];
[Center_Matrix, U_Matrix, Object_Func] = Fuzzy_C_Means(My_Datas, Num_Of_Clusters, options);


Ok_Membership_Matrix=zeros(1,3);
Nok_Membership_Matrix=zeros(1,3);

for i=1:length(U_Matrix(1,:))
    [Max_Value,Max_Index]=max(U_Matrix(:,i));
    
    if ((Max_Index==1)&&(Max_Value>(U_Matrix(2,i)))&&(Max_Value>(U_Matrix(3,i))))
        if ((Max_Index==1)&&(Max_Value>(2*U_Matrix(2,i)))&&(Max_Value>(2*U_Matrix(3,i))))
             Ok_Membership_Matrix(1,Max_Index)=Ok_Membership_Matrix(1,Max_Index)+1;
             continue
        end
            Nok_Membership_Matrix(1,Max_Index)=Nok_Membership_Matrix(1,Max_Index)+1;
    end
    
    if ((Max_Index==2)&&(Max_Value>(U_Matrix(1,i)))&&(Max_Value>(U_Matrix(3,i))))
        if ((Max_Index==2)&&(Max_Value>(2*U_Matrix(1,i)))&&(Max_Value>(2*U_Matrix(3,i))))
             Ok_Membership_Matrix(1,Max_Index)=Ok_Membership_Matrix(1,Max_Index)+1;
             continue
        end
            Nok_Membership_Matrix(1,Max_Index)=Nok_Membership_Matrix(1,Max_Index)+1;
    end
    
    if ((Max_Index==3)&&(Max_Value>(U_Matrix(1,i)))&&(Max_Value>(U_Matrix(2,i))))
        if ((Max_Index==3)&&(Max_Value>(2*U_Matrix(1,i)))&&(Max_Value>(2*U_Matrix(2,i))))
             Ok_Membership_Matrix(1,Max_Index)=Ok_Membership_Matrix(1,Max_Index)+1;
             continue
        end
            Nok_Membership_Matrix(1,Max_Index)=Nok_Membership_Matrix(1,Max_Index)+1;
    end
end


X=My_Datas(:,1);
Y=My_Datas(:,2);
hold on


for i=1:length(My_Datas)
    if ((U_Matrix(1,i)>2*U_Matrix(2,i))&&(U_Matrix(1,i)>2*U_Matrix(3,i)))
        plot(X(i),Y(i),'o','MarkerFaceColor','red');
    end
    if ((U_Matrix(2,i)>2*U_Matrix(1,i))&&(U_Matrix(2,i)>2*U_Matrix(3,i)))
        plot(X(i),Y(i),'o','MarkerFaceColor','green');
    end
    if ((U_Matrix(3,i)>2*U_Matrix(1,i))&&(U_Matrix(3,i)>2*U_Matrix(2,i)))
        plot(X(i),Y(i),'o','MarkerFaceColor','blue');
    end
end
for i=1:length(Center_Matrix(:,1))
    plot(Center_Matrix(i,1),Center_Matrix(i,2),'o','MarkerFaceColor','black');
end

%plot(Object_Func)

