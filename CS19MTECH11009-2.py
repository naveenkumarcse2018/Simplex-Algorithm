"""
	NAME: NAVEEN KUMAR KAMMARI
	ID: CS19MTECH110009

	Simplex-Algorithm: Degenerate-Case
"""


import numpy as np
from scipy.linalg import null_space
from scipy.optimize import linprog
import time
import math,random


time_inner=60
time_outer=360
# This function will check boundary region of the point with all constraints

def isInFeasibleRegian(A_dash, b_dash, del_x):
    i = 0
    for row in A_dash:
        result = np.dot(row, del_x.T)
        if not np.isclose(result, b_dash[i], atol=1e-05) and not result <= b_dash[i]:
            return False
        i += 1
    return True

# will return rows of A_Dash which satisfies == condition of b_dash
def findTightConstraintMatrix(del_x, A_dash, b_dash):
    i = 0
    A = []
    for row in A_dash:
        result = np.dot(row, del_x.T)
        if result == b_dash[i] or np.isclose(result, b_dash[i], atol=0.001):
            A.append(row)
        i += 1
    X = np.array(A)
    return X



#Function LPP 
#Using the concept explained in the class . Simplex Algorithm
def simplexAlgorithm(A_dash, b_dash,Z,del_x,m,n):
  optimal_found = False
  beta=0.001
  unbounded = False
  firstTime = False
  t_end = time.time() + time_outer
  #Algorithm starts here
  while True:
    del_x = np.round(del_x, 3)
    A_tight = findTightConstraintMatrix(del_x, A_dash, b_dash)
    size = A_tight.shape
    if size[0] >=n:  # If Tight matrix is  of size nxn
        inv_flag = False
        print("------------------------------------------------------------------")
        print("\nOur vertex point is ", del_x)
        print("Tight constraints matrix is ")
        print(A_tight)

        if size[0]>n: #Need to add epsilon and retry.
        	return False,del_x,False,False
        try:
            A_inverse = np.linalg.inv(A_tight)
        except:
            inv_flag = True

        if inv_flag == False:
            alpha_values = np.dot(Z.T, A_inverse)
        else:
            return False,del_x,False,False

        print("Alpha values are ", alpha_values)
        print("------------------------------------------------------------------")
        #checking for alpha values
        if alpha_values.size > 1:
            if all(x >= 0 for x in alpha_values):
                print("\nAll alpha values are positive")
                optimal_found = True
                break
            negative = np.where(alpha_values < 0)[0]
            negative_i = negative[0]
        else:
            if alpha_values >= 0:
                print("\nAll alpha values are positive ")
                optimal_found = True
                break
            else:
                negative_i = 0
        try:
            A_inverse = np.linalg.inv(A_tight)
            k = np.array(A_inverse[:, negative_i]*-1, dtype=float)
        except:
            print("Problem is unbounded/Infeasible")
            exit()

        del_x = np.add(del_x, beta*k)
        previous_point = del_x
        p_end = time.time()+time_inner

        #findin next vertex
        while(isInFeasibleRegian(A_dash, b_dash, del_x)):
            previous_point = del_x
            del_x = np.add(del_x, beta*k)
            if time.time() > p_end:
                unbounded = True
                break
        del_x = previous_point
        if unbounded:
            break

    else: #If tight constrains are not equal to n
        if A_tight.size == 0:
            if not firstTime:
                print("------------------------------------------------------------------")
                if isInFeasibleRegian(A_dash,b_dash,del_x):
                    print("The point that we founnd is feasible ,",del_x)
                else:
                    print("The point we founnd is non-feasible. So take some random point and move towards one of the vertex")
                firstTime = True
            else:

                k = del_x
                # print("Some random direction")
                previous_point = del_x
                s_end = time.time()+time_inner
                while(isInFeasibleRegian(A_dash, b_dash, del_x)):
                    previous_point = del_x
                    del_x = np.add(del_x, beta*k)
                    if time.time() > s_end:
                        unbounded = True
                        break
                del_x = previous_point
                if unbounded:
                    break

        else:

            ng = null_space(A_tight)
            try:
                if ng.T.shape[0] == 1:
                    ns = ng.T[0]
                else:
                    if ng.T.shape[0] == 0:
                        print("We are getting empty null space vector ")
                        print("Problem could be unbounded or infeasible. So move in some random direction")
                        ns=np.ones(n)
                    else:
                        ns = ng.T[0]
            except:
                print("Null space is not found")
                ns=np.ones(n)

            l = []
            for i in range(0, ns.size):
                l.append(ns[i])
            l= np.array(l)
            # print("Direction  ",l)
            previous_point = del_x
            del_x = np.add(del_x, beta*l)
            tight_matrix_equations=np.isclose(np.dot(A_dash,del_x),b_dash,atol=0.0001)
            if (tight_matrix_equations.sum()==n):
                continue

            previous_point = del_x
            p_end = time.time()+time_inner
            while(isInFeasibleRegian(A_dash, b_dash, del_x)):
                previous_point = del_x
                del_x = np.add(del_x, beta*l)
                if time.time() > p_end:
                    unbounded = True
                    break

            del_x = previous_point
            if unbounded:
                break
    if time.time() > t_end:
        unbounded = True
        break
    if unbounded:
        break
  return True,del_x,unbounded,optimal_found




#Reading Information 
print("*****************************************************************************************")
print("Observe the following example and enter the values of matrices\n")
print("Maximize: 40X1 + 35X2\n")
print("Constraints:\n")
print("2x1 + 3x2 <=60 \n4x1 + 3x2 <=96\nx1>=0\nx2>=0")
print("\nIn this case we have n=2 unknowns and m=4 constraints\n")
print("Matrix A =[[2,3],[4,3],[-1,0],[0,-1]]\nMatrix B=[60,96,0,0]\nMatrix C= [40,35]\n\n")
print("**Note: We need to chane x1>=0 , x2>= to maximization representation as -x1<=0 and -x2<=0")
print("Please enter valid input. Please observe the instructions before you enter the values")
print("*****************************************************************************************")

# #This line for reading inputs
print("\nPlease observe above example and enter the entries")
print("Enter no.of rows (m) in matrix A: - No.of constriants ") #no.of contrains
m=int(input())#no.of constrains
print("Enter no.of columns (n) in matrix A: No.of variables ")#no.of variables

n=int(input())#no.of variables 

A=np.ndarray(shape=(m,n),dtype=float)

#reading constraints array from the user
print("Enter elements of Matrix A: (Constriants)")
for i in range(0,m):
	print("\nContraint-"+str(i+1))
	for j in range(0,n):
		print("Coeffient of X"+str(j+1))
		A[i][j]=int(input())
l=[]
#reading inequalities of constraints
print("\nEnter Elements of Matrix B (Inequalities)")
for i in range(0,m):
	print("In-equality of constrint-",i+1)
	a=int(input())
	l.append(a)
B=np.array(l,dtype=float)
l=[]

#reading objective function
print("Enter elements of Matric C (Objective Function)")

for i in range(0,n):
	print("Coeffient of X",i+1)
	a=int(input())
	l.append(a)
C=np.array(l,dtype=float)



print()
print("------------------------------------------------------------------")
print("Matrix A is ")
print(A)
print("Matrix B is ")
print(B)
print("Matrix C is ")
print(C)
print("------------------------------------------------------------------")
print()

#Have to check for all B entries are +ve or not
m,n=A.shape
del_x=np.zeros(n)
flag_b=False
flag_b = any(x < 0 for x in B)

#If all entries of B are positive
if flag_b==False:
  print("All entries are positive")
  m,n=A.shape
  del_x=np.zeros(n)
  degerateCase=True
  gamma=0.001
  X=A
  Y=B
  Z=C
  while True:
  	degerateCase,point,unbounded,optimal=simplexAlgorithm(X,Y,Z,del_x,m,n)
  	del_x=np.round(point,3)
  	if degerateCase==False:
  		epsilon=np.array(np.random.uniform(0.00001,0.001,m))
  		Y=np.add(Y,epsilon)
  		# print(Y)
  	else:
  		break

#If one of the entries of B is negative 
else:
  print("Some of entries of B are negative")
  print("So we need to convert this problem into (n+1) dimensions")
  X=A
  Y=B
  Z=C
  m,n=X.shape
  #Making original problmen into (n+1) dimensions 
  row_to_be_added = []
  for i in range(0, n):
      row_to_be_added.append(0)
  X = np.vstack((X, row_to_be_added))
  column_to_added = []
  for i in range(0, m+1):
      column_to_added.append(0)
  X = np.column_stack((X, column_to_added))
  X[m][n] = -1
  col = -1*(min(Y))
  Y = np.append(Y, col)
  del_x=np.zeros(n+1)
  del_x[-1]=-1*col
  ind_min=np.where(Y==-1*col)[0]
  X[ind_min[0]][-1]=1
  Z=np.zeros(n+1)
  Z[-1]=1  
  #We have to get initial point for original problem by solving newly constructed (n+1) dimension problem
  m,n=X.shape
  print(del_x)
  print("Solving for new initial point")
  degerateCase=True
  while True:
  	degerateCase,init_point,unbounded,optimal=simplexAlgorithm(X,Y,Z,del_x,m,n)
  	del_x=np.round(init_point,3)
  	if degerateCase==False:
  		epsilon=np.array(np.random.uniform(0.00001,0.001,m))
  		Y=np.add(Y,epsilon)
  		# print(Y)
  	else:
  		break
  if init_point[-1]<0:
    if unbounded:
    	print("Problem is unbounded")
    	print("------------------------------------------------------------------")
    	exit()
    print("Found new initial point for problem")
    print("But the problem is Infeasible ")
    print("------------------------------------------------------------------")
    exit()
  else:
    #Taking input point 
    del_x=np.round(init_point[0:-1],3)
    print("------------------------Note---------------------------------------")
    print("Solving for our original problem")
    print("New initial point is ",del_x)
    m,n=A.shape
    degerateCase=True
    X=A
    Y=B
    Z=C
    while True:
    	degerateCase,point,unbounded,optimal=simplexAlgorithm(X,Y,Z,del_x,m,n)
    	del_x=np.round(point,3)
    	if degerateCase==False:
    		epsilon=np.array(np.random.uniform(0.00001,0.001,m))
    		Y=np.add(Y,epsilon)
    	else:
    		break
if optimal==True:
    result = np.dot(A, point)
    test = np.less_equal(result,B) 
    for i in range(0,m):
    	if test[i]==False:
    		if not np.isclose(result[i], B[i],atol=0.01,rtol=0.005):
		        print("But, Problem is infeasible\nThe point ",point," doesn't satisfy all the constraints")
		        print("------------------------------------------------------------------")
		        exit()
    cost = np.dot(C, point.T)
    print("The vetex at which we found optimal is ", np.round(point, 3))
    print("Optimal cost value is ", round(cost, 3))
    print("------------------------------------------------------------------")
elif unbounded==True:
    print("Problem is unbounded")
    print("------------------------------------------------------------------")
