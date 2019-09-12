function DisTance = Distance_F(Center_Matrix, data)
DisTance = zeros(size(Center_Matrix, 1), size(data, 1));
if size(Center_Matrix, 2) > 1,
    for k = 1:size(Center_Matrix, 1),
	DisTance(k, :) = sqrt(   sum(  (  (  data-ones(size(data, 1), 1)*Center_Matrix(k, :)).^2)'  )   );
    end 
else	
    for k = 1:size(Center_Matrix, 1),
	DisTance(k, :) = abs(Center_Matrix(k)-data)';
    end
end