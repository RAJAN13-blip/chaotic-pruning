close all;
clear all;  

dataset = "mnist";

window_size = 200;


purePath = "C:\Users\shiva\chaos dyn\Results\"+dataset+"\point 1\pure\";
wpath = "C:\Users\shiva\chaos dyn\Results\"+dataset+"\LE_window\";
pureFiles = dir(fullfile(purePath,'*.csv'));
    
overlap = 0.1;
index_val = window_size*(1-overlap);


for k = 1%:length(pureFiles)
    
  baseFileName = pureFiles(k).name;
  fullFileName = fullfile(purePath, baseFileName);
  fprintf(1, 'Now reading %s\n', baseFileName);

  data = csvread(fullFileName);
  samp = size(data,2);
  num_iterates = floor(size(data,1)/index_val);

  LE = zeros(num_iterates,samp);
  csvfilename = "window_LE_"+num2str(k)+".csv";
  count = 0;

  for m = 1:samp

      timeseries = data(:,m);
      le = zeros(num_iterates,1);
      for i = 0:num_iterates-1
        x = timeseries(index_val*i+1:index_val*(i+1)+window_size*overlap);
        y = L_E(x,m,i);
        
        if (size(y,1)>1)
            M = Slope(y);
            lslope = verify(M,i);
        else
            lslope = 0;
            count = count + 1;
        end 
        le(i+1) = lslope;
      end
        LE(:,m) = le;
        if (mod(m,50)==0)
         disp(m);
        end
  end

   csvwrite(wpath+csvfilename,LE);
   fprintf("No. of static windows : %d",count); 
        
end
    
  




function y = L_E(x,m,i)
    
 if (~any(ischange(x,'mean')) && max(x)-min(x)>=1e-5 )
        x=x';
        y = [];
        y=y';
        fid=fopen('logi1.dat','wt');
        fprintf(fid,'%f\n',x);
        fclose(fid);

        system('lyap_r -m2 -r20 -d1 -s3 -V0 -o logi1_lyap.dat logi1.dat');
        y=load('logi1_lyap.dat');
    else
        y =0;
        fprintf("Weights unvarying at : weight:%d, window:%d\n",m,i)
 end
    
       
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
