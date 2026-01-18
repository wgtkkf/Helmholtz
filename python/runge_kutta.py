#!/usr/bin/env python3
# Implemented routines below 1. and 2.
# 1. 4th order runge-kutta metho
# 2. analytical solution

# Object-oriented style
# Created: xx xx, 2025
# Last update: January 6th, 2026

import numpy as np
import time

class Comments:
    def __init__(self, arg_num):
        self.arg_num = arg_num

    def switch(self):
        if self.arg_num == 0:
            print('### calculation start        ###')
        elif self.arg_num == 1:
            print('### calculation finish       ###')
        else:
            print('check the number')

class TimeCounter:
    def __init__(self, time_counter):
        self.time_counter = time_counter

    def time_return(self):
            return self.time_counter

class Derivatives:
    def __init__(self):
        pass

    # dy1/dx = -y
    def f1(self, argx, argy, argv):
        fx = -argy
        return fx

    # dy2/dx = y1
    def f2(self, argx, argy, argv):
        fx = argv
        return fx

class Analytical:
    def __init__(self):
        pass
    
    def f1(self, argx):
        fx = np.sin(argx)
        return fx

class RungeKutta:
    def __init__(self, nmax_value, dh_value, filename):        
        self.NMAX = nmax_value
        self.dH = dh_value
        self.filename = filename
        self.derivatives = Derivatives() # Derivative instances
        self.analytical = Analytical()    

    def loop_calculation(self):
        x0 = 0
        y0 = 0
        v0 = 1

        # empty vectors        
        xv = np.zeros(self.NMAX, dtype='float64')
        yv = np.zeros(self.NMAX, dtype='float64')    

        with open(self.filename, 'w') as f:
            for i in range(self.NMAX):
                # for graphing
                xv[i] = x0
                yv[i] = y0
                
                k0 = self.dH*self.derivatives.f1(x0,y0,v0) # dp1
                l0 = self.dH*self.derivatives.f2(x0,y0,v0) # dy1
                
                k1 = self.dH*self.derivatives.f1(x0 + 0.5*self.dH, y0 + 0.5*k0, v0+0.5*l0) # dp2
                l1 = self.dH*self.derivatives.f2(x0 + 0.5*self.dH, y0 + 0.5*k0, v0+0.5*l0) # dy2
                
                k2 = self.dH*self.derivatives.f1(x0 + 0.5*self.dH, y0 + 0.5*k1, v0+0.5*l1) # dp3
                l2 = self.dH*self.derivatives.f2(x0 + 0.5*self.dH, y0 + 0.5*k1, v0+0.5*l1) # dy3
                
                k3 = self.dH*self.derivatives.f1(x0 + 0.5*self.dH, y0 + 0.5*k2, v0+0.5*l2) # dp4
                l3 = self.dH*self.derivatives.f2(x0 + 0.5*self.dH, y0 + 0.5*k2, v0+0.5*l2) # dy4
                
                x0 = x0 + self.dH                
                v0 = v0 + (k0 + 2*k1 + 2*k2 + k3)/6 # dy/dx                
                y0 = y0 + (l0 + 2*l1 + 2*l2 + l3)/6 # y

                # analytical solution
                solution = self.analytical.f1(x0)
                f.write(f"{x0} {y0} {v0} {solution} {y0-solution} \n") # plot x0-y0 and x0-solution
    
# --- main routine ---
if __name__ == '__main__':
    start = TimeCounter(time.time())
    #
    Comments(0).switch()

    #
    rk = RungeKutta(10000,0.001, 'solution.txt') # nmax, dh, filename
    rk.loop_calculation()

    #
    Comments(1).switch()

    # time display
    end = TimeCounter(time.time())
    print(f"### elapsed_time: {end.time_return() - start.time_return():.2f} [sec] ###")