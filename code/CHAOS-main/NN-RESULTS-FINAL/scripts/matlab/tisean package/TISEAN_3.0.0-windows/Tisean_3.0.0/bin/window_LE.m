
close all;
clear all;
                                %make folder from here onwards --> and
                                %the following string should be the same
                                %across all systems wherever it is
                                %supposed to be used


wpath = "C:\Users\91993\Desktop\chaos\NN-RESULTS-FINAL\datasets\initializations\diabetes\window_le";




fullFileName = "C:\Users\91993\Desktop\chaos\NN-RESULTS-FINAL\datasets\initializations\diabetes\weight_profiles\diabetes-2_weights.csv";


data = csvread(fullFileName);
samp = size(data,2);

LE = [];
  csvfilename = "\\window_LE_"+num2str(2)+".csv";
  for m = 1:samp
      timeseries = data(:,m);
      le = [];
      for i = 0:169
        x = timeseries(180*i+1:180*(i+1)+20.0);
        y = L_E(x);
    
        M = Slope(y);
        lslope = verify(M,i);
   
  
        le = [le,lslope];
                
      end
        LE = [LE;le];
  end
    csvwrite(wpath+csvfilename,LE);






function y = L_E(x)
    
    x=x';
    y = [];
    y=y';
    fid=fopen('logi1.dat','wt');
    fprintf(fid,'%f\n',x);
    fclose(fid);

    system('lyap_r -m2 -r20 -d1 -s20 -V0 -o logi1_lyap.dat logi1.dat');
    y=load('logi1_lyap.dat');

end

function M = Slope(y1)
    y_1 = interp1(y1(:,1),y1(:,2),[1 2.5],'linear');
    y_2 = interp1(y1(:,1),y1(:,2),[0.75 1],'linear');
    y_3= interp1(y1(:,1),y1(:,2),[0.5 0.6],'linear');
    y_4= interp1(y1(:,1),y1(:,2),[0.25 0.75],'linear');
    
    slope1 = (y_1(2)-y_1(1))/(2.5-1);
    slope2 = (y_2(2)-y_2(1))/(1-0.75);
    slope3 = (y_3(2)-y_3(1))/(0.6-0.5);
    slope4 = (y_4(2)-y_4(1))/(.75-.25);
    
    M = [slope1 , slope2,  slope3 , slope4];

end

function lslope = verify(M,i)
    M = round(M,6);
    if (M(2) == M(3))
        lslope = M(2);
    elseif  ( M(3) == M(4)|| M(2) == M(4))
        lslope = M(4);  
    else
        lslope = 0;  
        disp("Slope not matching at "+i);
    end
end



