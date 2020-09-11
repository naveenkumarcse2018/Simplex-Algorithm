# To run the program
python3 file_name.py

1.Please refer to the instructions displayed in the beginning of the program (while executing) and follow the input format as mentioned below

Observe the following example and enter the values of matrices

Maximize: 40X1 + 35X2

Constraints:

2x1 + 3x2 <=60 

4x1 + 3x2 <=96

x1>=0

x2>=0



In this case we have n=2 unknowns and m=4 constraints

Matrix A =[
		 [2,3],
		[4,3],
		[-1,0],
		[0,-1]
		]

Matrix B=[60,96,0,0]


Matrix C= [40,35]

**Note: We need to chane x1>=0 , x2>= to maximization representation as -x1<=0 and -x2<=0
*****************************************************************************************


2.Please enter the values correctly

3.Incase of degerating , as we choose random epsilon for constraints in-equality, we may get unboundness for choosing random epsilon. So if a problem has solution but
  program is giving ouput as "UNBOUND", then please re-run the program and enter the values (Randomness can't be unique always)
