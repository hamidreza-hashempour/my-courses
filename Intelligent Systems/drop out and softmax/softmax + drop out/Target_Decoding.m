function [Decoded_Output]=Target_Decoding(Train_Labels)

    Out_Target_O_O=zeros(60000,4);
    Out_Target_A_O=zeros(60000,10);
    for i=1:60000
        if( Train_Labels(i)==0)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,1)=1;
        end
       if( Train_Labels(i)==1)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,2)=1;
       end
        if( Train_Labels(i)==2)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=1;
           Out_Target_O_O(i,4)=0;
           Out_Target_A_O(i,3)=1;
        end
       if( Train_Labels(i)==3)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=1;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,4)=1;
       end
       if( Train_Labels(i)==4)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=1;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=0;
           Out_Target_A_O(i,5)=1;
       end
       if( Train_Labels(i)==5)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=1;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,6)=1;
       end
       if( Train_Labels(i)==6)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=1;
           Out_Target_O_O(i,3)=1;
           Out_Target_O_O(i,4)=0;
           Out_Target_A_O(i,7)=1;
       end
       if( Train_Labels(i)==7)
           Out_Target_O_O(i,1)=0;
           Out_Target_O_O(i,2)=1;
           Out_Target_O_O(i,3)=1;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,8)=1;
       end
       if( Train_Labels(i)==8)
           Out_Target_O_O(i,1)=1;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=0;
           Out_Target_A_O(i,9)=1;
       end
       if( Train_Labels(i)==9)
           Out_Target_O_O(i,1)=1;
           Out_Target_O_O(i,2)=0;
           Out_Target_O_O(i,3)=0;
           Out_Target_O_O(i,4)=1;
           Out_Target_A_O(i,10)=1;
       end
    end   
    
    Decoded_Output=Out_Target_A_O;