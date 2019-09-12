function U = Initialization(Num_Of_Clusters, My_Data)
U = rand(Num_Of_Clusters, My_Data);
Sum_Of_Clmn = sum(U);
U = U./Sum_Of_Clmn(ones(Num_Of_Clusters, 1), :);