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

x=[0.2
0.64
0.9216
0.28901376
0.821939226
0.585420539
0.970813326
0.113339247
0.401973849
0.961563495
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
];
x=x';


y=[];
y=y';

fid=fopen('logi1.dat','wt');
fprintf(fid,'%f\n',x);
fclose(fid);

x1=load('logi1.dat');
%plot(x1);

%% If system does not work then run the following on linux terminal
%./lyap_r -m2 -r20 -d1 -s20 -V0 -o logi1_lyap.dat logi1.dat 
system('./lyap_r -m2 -r20 -d1 -s20 -V0 -o logi1_lyap.dat logi1.dat'); %%% http://www.mpipks-dresden.mpg.de/~tisean/Tisean_3.0.1/index.html

%%
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
%end'