%%
%% designed by hamidreza hashempour
%% 4 June 2018
%% tehran university
%% solving honoi tower by q_learning algorithm

%%
%%describing reward matrix for N (smallest, ... ,biggest)
%% at first you inter  num of disks
%% it would creat for you a reward matrix sized by 3^n*3^n
%%by defining rules 1-just move one this in each movement 2-the big one
%%should not be top of small one
%%all of 3^n state wented to 3 base in order to make it easy to check
%%states by dec2base func


N=input(' inter num of discks \n ');
Reward_Matrix=zeros(3^N,3^N);
State_Matrix=zeros(3^N,N);

    for j=1:3^N
        help_index1=(dec2base(j-1,3)-'0');
        
        length_num=length(help_index1);
           State_Matrix(j,N-length_num+1:N)=(dec2base(j-1,3)-'0');
       
    end
 
  for i=1:3^N
      for j=1:3^N
          Common_Element=0;
          for q_Matrix=1:N
              if(State_Matrix(i,q_Matrix)==State_Matrix(j,q_Matrix))
                  Common_Element=Common_Element+1;
              end
              if(State_Matrix(i,q_Matrix)~=State_Matrix(j,q_Matrix))
                  Uncommon_Element_Clmn=q_Matrix;
              end
          end
              
         if (Common_Element==(N-1))
             helping_index2=0;
             for k=1:Uncommon_Element_Clmn
                 if(State_Matrix(i,k)==State_Matrix(i,Uncommon_Element_Clmn))
                 helping_index2=helping_index2+1;
                 end
             end
             if(helping_index2==1)
                 helping_index3=0;
             for k=1:Uncommon_Element_Clmn
                 if(State_Matrix(j,k)==State_Matrix(j,Uncommon_Element_Clmn))
                 helping_index3=helping_index3+1;
                 end
             end
                if(helping_index3==1)
                    Reward_Matrix(i,j)=-0.01;
                 Is_This_Final_State=isequal(State_Matrix(j,:),State_Matrix(3^N,:));
                if(Is_This_Final_State==1)
                    Reward_Matrix(i,j)=100;
                end
                    
                end
             end
         end
      end
  end
  
  for i=1:3^N
      for j=1:3^N
         if( Reward_Matrix(i,j)==0)
             Reward_Matrix(i,j)=-inf;
         end
      end
  end
          
          
   %%
   %%
   %%intering all parameters of question        
   discount_factor=input('inter discount factor \n') ;              
    learning_rate=input('inter learning rate \n');             
    epsilon=input('inter epsilon value for greedy algorithm \n'); 
    goalState=3^N;
%%
%%we will initiall q_matrix to zero at first
q_Matrix=zeros(size(Reward_Matrix));        
q1inf_Matrix=ones(size(Reward_Matrix))*inf;    
counter=0;                 
steps=0;                 
B=[];                    
Sum_Of_Rewards=0;             
exploit_Numbers=0;
explore_Numbers=0;
%%main for , for reaching to last state with compensately number in order
%%to optimaze movements
for episode=1:50000
    state=1;        
%% using while in order to reach last state in each steo of playing game
    while state~=goalState            
        x=find(Reward_Matrix(state,:)>=-60);        
        if size(x,1)>0,
            random_Var=rand; 
     if random_Var>=epsilon   
         [~,qbishin]=(max(q_Matrix(state,x(1:end)))); 
         x1 = x(qbishin); 
         if epsilon>=0.5
            epsilon=epsilon*0.99999; 
         else
             epsilon=epsilon*0.9999; 
         end
         Sum_Of_Rewards=Sum_Of_Rewards+q_Matrix(state,x1); 
         exploit_Numbers=exploit_Numbers+1;
     else       
             x1=x(1,ceil(rand*length(x)));   
             x1=x1(1);                  
             if epsilon>=0.5
                epsilon=epsilon*0.99999; 
             else
                epsilon=epsilon*0.9999;
             end
             Sum_Of_Rewards=Sum_Of_Rewards+q_Matrix(state,x1); 
             explore_Numbers=explore_Numbers+1;
     end
        x2 = find(Reward_Matrix(x1,:)>=-60);   
        qMax=(max(q_Matrix(x1,x2(1:end)))); 
        q_Matrix(state,x1)= q_Matrix(state,x1)+learning_rate*((Reward_Matrix(state,x1)+discount_factor*qMax)-q_Matrix(state,x1));   
        state=x1;    
        end
        if state~=goalState     
            steps=steps+1;
        else
            steps=steps+1;
            A(:,episode)=[episode; steps; Sum_Of_Rewards];   
            B=horzcat(B, A);     
        end
    end
%% events that cause breaking for its meaning you reach last state and num of movements are stable for a lot of states that cant change
    if sum(sum(abs(q1inf_Matrix-q_Matrix)))<0.0001 && sum(sum(q_Matrix >0)) && epsilon<0.15
        if counter>5,
            q1inf_Matrix=q_Matrix;
            break 
        else
            q1inf_Matrix=q_Matrix;
            counter=counter+1; 
        end

    else
        q1inf_Matrix=q_Matrix;
        counter=0; 
    end
    fprintf('Episode %i.required %i steps. reward gained is %i.\n', episode, steps, Sum_Of_Rewards);
    steps=0;   
    Sum_Of_Rewards=0;    

end
     
      
      for i=1:length(A(3,:))
                Sum_Q_In_All_States(i)=A(3,i)/A(2,i);
      end          
          
          
            
            