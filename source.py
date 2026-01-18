# 1D FEM
# Last modified: August 25 2019
# Coded by Takuro TOKUNAGA

import numpy as np
import time
import scipy.linalg
from scipy.integrate import quad

start = time.time()

# functions for begin & finish
def begin():
    print ("begin")

def end():
    print ("finish")


# Boundary condition as function
def boundary_d(node, Dirichlet): # Dirichlet boundary

    bmat[:] -= Dirichlet*Amat[node,:]
    bmat[node] = Dirichlet
    Amat[node,:] = 0.0 # row zero fix
    Amat[:,node] = 0.0 # Column zero fix
    Amat[node,node] = 1.0 # diagonal one fix

def boundary_n(node, Neumann): # Neumann boundary
    bmat[node] += Neumann

def analytical(f,a,b,alpha,beta,argx):
    term1 = 0.5*f*np.power(argx,2.0) # x2
    term2 = (-f*b+beta)*argx         # x
    term3 = -0.5*f*np.power(a,2.0)
    term4 = (f*b-beta)*a+alpha
    
    fx = term1 + term2 + term3 + term4

    return fx


## parameters group
# FEM parameters
size = 2    # cannot change
element = 5 # can change
node_t = element + 1 # total node number, cannot change
node_d = 0 # Dirichlet boundary node number, can change
node_n = element # Neumann boundary node number, can change
Dirichlet = 1 # Dirichlet boundary condition
Neumann = -1 # Neumann boundary condition

b0 = 1   # can change

# function parameters
xmin = -1 # can change
xmax = 1 # can change
elength = (xmax-xmin)/element # no need to change
xvector = np.zeros(node_t, dtype='float64')
xvector[0] = xmin
for i in range(element):
    xvector[i+1] = xvector[i] + elength

#print(str(xvector))

# total matrix
Amat = np.zeros((node_t,node_t), dtype='float64')
bmat = np.zeros(node_t, dtype='float64')
umat = np.zeros(node_t, dtype='float64')

# element matrix
eAmat = np.zeros((size,size), dtype='float64')
ebmat = np.zeros(size, dtype='float64')

# element matrix initialization
for i in range(size):
    ebmat[i] = 1.0
    for j in range(size):
        if i==j:
            eAmat[i][j] = 1.0
        else:
            eAmat[i][j] = -1.0

#print(str(eAmat))
#print(str(ebmat))

# A & b matrix initialize
for i in range(element):
    for j in range(size):
        bmat[j+i] += ebmat[j]
        for k in range(size):
            Amat[j+i][k+i] += eAmat[j][k]

Amat = (1/elength)*Amat
bmat = -(b0*0.5*elength)*bmat # don't forget sign


#print(Amat)
#print(bmat)

# main start
begin()


# file open
f0 = open('results.txt', 'w')
f1 = open('analytical.txt', 'w')

# Boundary condition
boundary_d(node_d, Dirichlet)
boundary_n(node_n, Neumann)

#print(Amat)
#print(bmat)

# solving
umat = np.linalg.solve(Amat,bmat)
#print(umat)

exact_x = np.arange(xmin,xmax,0.01)
exact_y = analytical(b0,xmin,xmax,Dirichlet,Neumann,exact_x)


# output to file
for i in range(node_t):
    f0.write(str(xvector[i]))
    f0.write(str(' '))
    f0.write(str(umat[i]))
    f0.write(str('\n'))

# file close
f0.close()
f1.close()

# main end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
