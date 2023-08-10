clear all
close all
k = 40;
N0 = 15;  %initial pop. size
L = 35;  % Length of time series

fid1=fopen('slope.dat','wt');

% %for r=2.0:.1:4;
% %for Noise = 0:0.1:.3;
% r=2.9;
% Noise=0;
% x = zeros(1,L);
% x(1) = N0;
% for i = 2:L;
%     x(i) = x(i-1)*exp((r+Noise*rand())*(1-x(i-1)/k));
%     %if (mod(x(i),1)~=0)
%     %    x(i) =  round(x(i));
%      %end
% end

x=[0.20001
0.6400239996
0.9215731181
0.2891044242
0.8220922245
0.5850263957
0.9710820481
0.1123268157
0.3988380087
0.959065006
1.57E-01
5.30E-01
9.97E-01
1.39E-02
5.48E-02
2.07E-01
6.57E-01
9.02E-01
3.54E-01
9.15E-01
3.11E-01
8.58E-01
4.88E-01
9.99E-01
2.17E-03
8.65E-03
3.43E-02
1.32E-01
4.60E-01
9.93E-01
2.59E-02
1.01E-01
3.63E-01
9.24E-01
2.80E-01
8.06E-01
6.26E-01
9.36E-01
2.39E-01
7.27E-01
7.95E-01
6.53E-01
9.07E-01
3.38E-01
8.95E-01
3.76E-01
9.38E-01
2.32E-01
7.13E-01
8.18E-01];
x=x';


y=[0.2
0.64
0.9216
0.28901376
0.8219392261
0.5854205387
0.9708133262
0.1133392473
0.4019738493
0.9615634951
1.48E-01
5.04E-01
1.00E+00
2.46E-04
9.85E-04
3.94E-03
1.57E-02
6.17E-02
2.32E-01
7.12E-01
8.20E-01
5.90E-01
9.67E-01
1.26E-01
4.42E-01
9.86E-01
5.37E-02
2.03E-01
6.48E-01
9.12E-01
3.20E-01
8.71E-01
4.50E-01
9.90E-01
4.00E-02
1.54E-01
5.20E-01
9.98E-01
6.31E-03
2.51E-02
9.79E-02
3.53E-01
9.14E-01
3.15E-01
8.63E-01
4.72E-01
9.97E-01
1.25E-02
4.96E-02
1.88E-01];
y=y';


fid=fopen('logi1.dat','wt');
fprintf(fid,'%f\n',y);
fclose(fid);

x1=load('logi1.dat');
%plot(x1);

system(['lyap_r -m2 -r20 -d1 -s20 -V0 -o logi1_lyap.dat logi1.dat']); %%% http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html


y1=load('logi1_lyap.dat');
%disp(size(y1))

if (length(y1(:,1))>5)
    pause(1);
plot(y1(:,1),y1(:,2));

[l,m]=ginput(2);
slope=(m(2)-m(1))/(l(2)-l(1));
disp(['the slope is : ',num2str(slope)])

fprintf(fid1,'%f \n',slope);

end 
%disp(Noise)
%end
%disp(r)
%end
