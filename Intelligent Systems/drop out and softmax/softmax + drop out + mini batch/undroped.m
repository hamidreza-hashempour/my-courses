function [weights]=undroped(Dropedneuralnet,Mainneuralnet,Offlights)
for gamma=1:length(Dropedneuralnet)
    weights{gamma}=Dropedneuralnet{gamma};

end
Helping_W1_Matrix=weights{1};
Helping_W2_Matrix=weights{2};
HelpingMain_W1_Matrix=Mainneuralnet{1};
HelpingMain_W2_Matrix=Mainneuralnet{2};
for gamma=1:length(Offlights)
   if (Offlights(gamma)==0)
       Helping_W1_Matrix(:,gamma)=HelpingMain_W1_Matrix(:,gamma);
       Helping_W2_Matrix(gamma,:)=HelpingMain_W2_Matrix(gamma,:);
   end
    
end
weights{1}=Helping_W1_Matrix;
weights{2}=Helping_W2_Matrix;

end