# 4th order runge-kutta
# d2y/dx2 = -y
# Last modified: September 01 2019
# Coded by Takuro TOKUNAGA

import numpy as np
import time
import pandas as pd

# for graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

start = time.time()

# functions for begin & finish
def begin():
    print ("begin")

def end():
    print ("finish")


# dy1/dx = -y
def f1(argx,argy,argv):

    fx = -argy

    return fx

# dy2/dx = y1
def f2(argx,argy,argv):
    fx = argv

    return fx

def analytical(argx):
    fx = np.sin(argx)

    return fx

# main start
begin()

# parameters
nmax = 10000 # [-]
h = 0.001   # [-]

x0 = 0
y0 = 0
v0 = 1

xmax = x0+h*nmax
xmin = x0

# vectors
xv = np.zeros(nmax, dtype='float64')
yv = np.zeros(nmax, dtype='float64')

# file open
f0 = open('results.txt', 'w')
fem_result = pd.read_csv("../fem1d/fem.txt", sep=" ", header=None)
fem_result.columns = ["x", "fx"]
row, col = fem_result.shape # row & column of matorix
fem_x = np.zeros(row, dtype='float64')
fem_y = np.zeros(row, dtype='float64')

# input data into tables
for i in range(0, row):
    fem_x[i] = fem_result.iat[i,0] # x line
    fem_y[i] = fem_result.iat[i,1] # x line

print(str(fem_x))

for i in range(nmax):
    # for graphing
    xv[i] = x0
    yv[i] = y0

    k0 = h*f1(x0,y0,v0) # dp1
    l0 = h*f2(x0,y0,v0) # dy1

    k1 = h*f1(x0+0.5*h,y0+0.5*k0,v0+0.5*l0) # dp2
    l1 = h*f2(x0+0.5*h,y0+0.5*k0,v0+0.5*l0) # dy2

    k2 = h*f1(x0+0.5*h,y0+0.5*k1,v0+0.5*l1) # dp3
    l2 = h*f2(x0+0.5*h,y0+0.5*k1,v0+0.5*l1) # dy3

    k3 = h*f1(x0+0.5*h,y0+0.5*k2,v0+0.5*l2) # dp4
    l3 = h*f2(x0+0.5*h,y0+0.5*k2,v0+0.5*l2) # dy4

    # analytical
    solution = analytical(x0)

    x0 = x0 + h
    v0 = v0 + (k0 + 2*k1 + 2*k2 + k3)/6 # dy/dx
    y0 = y0 + (l0 + 2*l1 + 2*l2 + l3)/6 # y

    # output to file
    f0.write(str(x0))
    f0.write(str(' '))
    f0.write(str(y0))
    f0.write(str(' '))
    f0.write(str(v0))
    f0.write(str(' '))
    f0.write(str(solution))
    f0.write(str('\n'))

exact_x = np.arange(xmin,xmax,h)
exact_y = analytical(exact_x)

# graph display
csfont = {'fontname':'Times New Roman'} # define font
plt.figure

plt.grid() # grid display
plt.plot(exact_x, exact_y, 'blue', label="Analytical")
plt.plot(xv, yv,'r--',alpha=1, label="4th Runge-Kutta")
plt.plot(fem_x, fem_y, 'go',alpha=0.3, label="1D FEM")


## graph information
#plt.title('Distribution', **csfont) # graph title
plt.xlabel('x [-]', fontdict=None, labelpad=None, **csfont)
plt.ylabel('f(x), Solution [-]', fontdict=None, labelpad=None, **csfont)

# font for legend
font = font_manager.FontProperties(family='Times New Roman',
                                   weight='bold',
                                   style='normal', size=10)
plt.legend(loc='lower right', prop=font) # legend

# plot options
plt.xticks([0.0, 2.0, 4.0, 6.0, 8.0, 10.0], **csfont)
plt.yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], **csfont)
# graph save & display
plt.savefig("solution.png") # 1. file saving (1. should be before 2.)
plt.show()                  # 2. file showing (2. should be after 1.)


# file close
f0.close()

# main end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
