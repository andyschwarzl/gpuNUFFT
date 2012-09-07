#!/bin/sh
#testscript 
echo Start Testscript
for i in 1 # 2 # 3 4 5 6 7 8 9 10
do
	echo " running $i..." 
	matlab -nojvm -nosplash < test_gridding3D_Operator_2D_data_script.m > "script.out.$i" #& 
done
