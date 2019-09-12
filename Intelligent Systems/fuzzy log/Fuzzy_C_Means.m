function [Cnt, U_Matrix, Object_Func] = Fuzzy_C_Means(My_Data, Num_Of_Clusters, options)
Data_Numbers = size(My_Data, 1);
M_Fuzzier = options(1);		
Max_Number_Of_Iteration = options(2);		
Epsilon = options(3);		
Object_Func = zeros(Max_Number_Of_Iteration, 1);	
U_Matrix = Initialization(Num_Of_Clusters, Data_Numbers);		
for i = 1:Max_Number_Of_Iteration,
	[U_Matrix, Cnt, Object_Func(i)] = Updating(My_Data, U_Matrix, Num_Of_Clusters, M_Fuzzier);
		fprintf('Iteration num = %d, object amount = %f\n', i, Object_Func(i));
	if i > 1,
		if abs(Object_Func(i) - Object_Func(i-1)) < Epsilon, break; end,
	end
end
iter_n = i;	
Object_Func(iter_n+1:Max_Number_Of_Iteration) = [];