close all;
clear vars;
p1 = "C:\Users\91993\Downloads\XOR- code+results\XOR- code+results\TISEAN_3.0.0-windows\Tisean_3.0.0\bin\point 1\";
S = dir(fullfile(p1,'*.csv'));
summ = [1:21];

for i = 1:length(S)
    data = csvread(p1+S(i).name);

    for k = 2:size(data,1)
    if (data(k,2) == data(k,3));
        summ(k,i) = data(k,2);
    elseif  ( data(k,3) == data(k,4)| data(k,2) == data(k,4));
        summ(k,i) = data(k,4);  
    end
    
    if (data(k,6) == data(k,7));
        summ(k,i+11) = data(k,6);  
    elseif (data(k,6) == data(k,8) | data(k,7) == data(k,8));
        summ(k,i+11) = data(k,8);  
    end
    
    end
end

csvwrite(p1+"summary.csv",summ);