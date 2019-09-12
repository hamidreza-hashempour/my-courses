function distance=DIS(x,y)
dist=0;
    for i=1:2
        dist=(x(i)-y(i))^2+dist;
    end

distance=dist^0.5;
end