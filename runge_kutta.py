# 4th order runge-kutta
# Last modified: September 01 2019
# Coded by Takuro TOKUNAGA

import numpy as np
import time

# for graph
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

start = time.time()

# functions for begin & finish
def begin():
    print ("begin")

def end():
    print ("finish")


def f1(argt,argy,argv):

    fx = argv

    return fx

def f2(argt,argy,argv):
    g = 9.8
    fx = g

    return fx

def analytical(argx):
    radx = argx*(180/np.pi)
    fx = np.sin(radx)

    return fx

# main start
begin()

# parameters
nmax = 30 # [-]
h = 0.1   # [-]

t0 = 0
y0 = 0
v0 = 0

# file open
f0 = open('results.txt', 'w')

for i in range(nmax):
    k0 = h*f1(t0,y0,v0)
    l0 = h*f2(t0,y0,v0)

    k1 = h*f1(t0+0.5*h,y0+0.5*k0,v0+0.5*l0)
    l1 = h*f2(t0+0.5*h,y0+0.5*k0,v0+0.5*l0)

    k2 = h*f1(t0+0.5*h,y0+0.5*k1,v0+0.5*l1)
    l2 = h*f2(t0+0.5*h,y0+0.5*k1,v0+0.5*l1)

    k3 = h*f1(t0+0.5*h,y0+0.5*k2,v0+0.5*l2)
    l3 = h*f2(t0+0.5*h,y0+0.5*k2,v0+0.5*l2)

    t0 = t0 + h
    y0 = y0 + (k0 + 2*k1 + 2*k2 + k3)/6
    v0 = v0 + (l0 + 2*l1 + 2*l2 + l3)/6

    # output to file
    f0.write(str(t0))
    f0.write(str(' '))
    f0.write(str(y0))
    f0.write(str(' '))
    f0.write(str(v0))
    f0.write(str('\n'))

# file close
f0.close()

# main end
end()

# time display
elapsed_time = time.time()-start
print("elapsed_time:{:.2f}".format(elapsed_time) + "[sec]")
