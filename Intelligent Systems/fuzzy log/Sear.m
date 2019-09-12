%%
%%

My_Datas=load('EMG_data.csv');
My_Datas(:,1)=[];

Length_Of_Data=length(My_Datas(:,1));
Num_Of_Features=length(My_Datas(1,:));


Num_Of_Clusters=input('Num of clusters \n ');
M_Fuzzier=input('M_Fuzzier \n ');
Uij=zeros(Length_Of_Data,Num_Of_Clusters);
Mean_Center_Matrix=zeros(Num_Of_Clusters,Num_Of_Features);
Distance_Matrix=zeros(Length_Of_Data,Num_Of_Clusters);
%%
%%initial Uij Matrix
for i=1:Length_Of_Data
     a=0.5;
     b=Num_Of_Clusters+0.5;
     Selected_Cluster=round(a+(b-a)*rand());
     
     Uij(i,Selected_Cluster)=1;
end
%%
%%

for z=1:7
    
Uij_Potencial_M=Uij.^M_Fuzzier;

%%
%% finding means of centers

for i=1:Num_Of_Features
    for j=1:Num_Of_Clusters
        
        Up_Of_Divide=transpose(My_Datas(:,i))*Uij_Potencial_M(:,j);
        Up_Of_Divide=Up_Of_Divide/1000;
        Down_Of_Divide=transpose(Uij_Potencial_M(:,j))*Uij_Potencial_M(:,j);
        Down_Of_Divide=Down_Of_Divide/1000;
        Mean_Center_Matrix(j,i)=Up_Of_Divide/Down_Of_Divide;
        
    end
end

%%
%%measuring distances

for i=1:Length_Of_Data
    for j=1:Num_Of_Clusters
        
       Distance_Matrix(i,j)=DIS(My_Datas(i,:),Mean_Center_Matrix(j,:),Num_Of_Features);
       
    end
end

%%
%%measuring new Uij matrix

for i=1:Length_Of_Data
    for j=1:Num_Of_Clusters
        
        Helping_Index_Sum_Of_Materials=0;
        for q=1:Num_Of_Clusters
            
            Helping_Index_Sum_Of_Materials=Helping_Index_Sum_Of_Materials+((Distance_Matrix(i,j))/(Distance_Matrix(i,q)))^(-2/(M_Fuzzier-1));
        end
     
        Uij(i,j)=Helping_Index_Sum_Of_Materials^(-1);
    end
end

end
        





















     
     
     
     
     
     
     