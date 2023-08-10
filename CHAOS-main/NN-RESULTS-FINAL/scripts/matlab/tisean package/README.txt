Nithin Nagaraj
NIAS
June 28, 2022

How to run logi1_lyap.m:

feed the data in 'x' or 'y' variable inside logi1_lyap

after you run logi1_lyap.m, it prompts for user input to give two points to compute slope

the slope value is LYAPUNOV EXPONENT.

I am using MATLAB version '9.2.0.538062 (R2017a)'
but should work in any other version I think.

It is important that lyap_r.exe is in the same folder as logi1_lyap.m


Archana Mathur 15th July 2022

logi1_lyap_linux_update.m

INput is a csv file which has series, either the difference series or origianl series (iterates),
 of all the connections of the NN 
 
After selecting the csv file at run time, it asks for two input in the form of clicks onto the plots

The slopes are the lyapunov exponents of all the series of the csv file entered during runtime.

these are saved in slope.dat
