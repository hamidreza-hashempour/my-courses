function [MainNeuralnet,DropedNeuralnet,r]=dropout(weights,HiddenLayerNum)


for beta=1:length(weights)
    MainNeuralnet{beta}=weights{beta};
    DropedNeuralnet{beta}=weights{beta};
end
DropedW1=DropedNeuralnet{1};
DropedW2=DropedNeuralnet{2};
r = randi([0,1],HiddenLayerNum,1);

for zeta=1:HiddenLayerNum
   if (r(zeta)==0)
       DropedW1(:,zeta)=0;
       DropedW2(zeta,:)=0;
   end
    
end
DropedNeuralnet{1}=DropedW1;
DropedNeuralnet{2}=DropedW2;

end