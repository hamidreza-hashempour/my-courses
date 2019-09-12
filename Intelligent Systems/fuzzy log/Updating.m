function [New_U_Matrix, Center_Matrix, Object_Func] = Updating(My_Data, U_Matrix, Num_Of_Clusters, M_Fuzzier)
U_Matrix_EXP_M = U_Matrix.^M_Fuzzier;    
Center_Matrix = U_Matrix_EXP_M*My_Data./((ones(size(My_Data, 2), 1)*sum(U_Matrix_EXP_M'))'); 
Distance = Distance_F(Center_Matrix, My_Data);     
Object_Func = sum(sum((Distance.^2).*U_Matrix_EXP_M)); 
Helping_Index = Distance.^(-2/(M_Fuzzier-1));    
New_U_Matrix = Helping_Index./(ones(Num_Of_Clusters, 1)*sum(Helping_Index));