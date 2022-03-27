# 1D FEM
# Last modified: August 25 2019
# Coded by Takuro TOKUNAGA

import numpy as np
import time
import scipy.linalg
from scipy.integrate import quad

# for graph
#import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

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

# integrands
def fx(argx):
    fx = b0

    return fx

def xfx(argx):
    xfx = argx*fx(argx)

    return xfx

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
    ebmat[0] = xvector[i+1]*quad(fx,xvector[i],xvector[i+1])[0] - quad(xfx,xvector[i],xvector[i+1])[0]
    ebmat[1] = -xvector[i]*quad(fx,xvector[i],xvector[i+1])[0] + quad(xfx,xvector[i],xvector[i+1])[0]
    for j in range(size):
        bmat[j+i] += ebmat[j]
        for k in range(size):
            Amat[j+i][k+i] += eAmat[j][k]

Amat = (1/elength)*Amat
bmat = -(1/elength)*bmat # don't forget sign


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

# graph display
csfont = {'fontname':'Times New Roman'} # define font
plt.figure

plt.grid() # grid display
plt.plot(exact_x, exact_y, 'blue', label="Analytical")
plt.plot(xvector, umat,'ro--',label="1D FEM")
## graph information
#plt.title('Distribution', **csfont) # graph title
plt.xlabel('x [-]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('f(x) [-]', fontdict=None, labelpad=None, **csfont)

# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='upper right', prop=font) # legend

# plot options
plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0], **csfont)
plt.yticks([-3.0, -2.0, -1.0, 1.0, 0.0, 1.0], **csfont)
# graph save & display
plt.savefig("solution.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)


# file close
f0.close()
f1.close()

# main end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
