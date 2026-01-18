# One-dimensional finite element method for Helmholtz equation
# Coded by Takuro Tokunaga

# Update hisotry:
# September 02, 2019
# October   19, 2025
# January   12-13, 15, 2026

import numpy as np
import scipy.linalg
from scipy.integrate import quad
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

class FiniteElement:
    def __init__(self, arg_xmin, arg_xmax, arg_size, arg_element):
        self.arg_xmin = arg_xmin
        self.arg_xmax = arg_xmax
        self.arg_size = arg_size
        self.arg_element = arg_element            
        self.node_t = self.arg_element + 1 # total node number, cannot change        

        # total matrix zero initialization
        self.Amat = np.zeros((self.node_t,self.node_t), dtype='float64')
        self.A2mat = np.zeros((self.node_t, self.node_t), dtype='float64')
        self.bmat = np.zeros(self.node_t, dtype='float64')
        self.umat = np.zeros(self.node_t, dtype='float64')        

        # element matrix zero initialization
        self.eAmat = np.zeros((self.arg_size, self.arg_size), dtype='float64')
        self.eA2mat = np.zeros((self.arg_size, self.arg_size), dtype='float64')        
        self.xvector = np.zeros(self.node_t, dtype='float64')
    
    # 00, 01, 10, 11
    def basfx00(self, arg_1, arg_2): # basis function 1
        fx = (arg_2-arg_1)*(arg_2-arg_1)
        return fx
    
    def basfx01(self, arg_1, arg_2, arg_3): # basis function 1
        fx = -(arg_2-arg_3)*(arg_1-arg_3)
        return fx
    
    def basfx10(self, arg_1, arg_2, arg_3): # basis function 1
        fx = -(arg_2-arg_3)*(arg_1-arg_3)
        return fx

    def basfx11(self, arg_1, arg_2): # basis function 1
        fx = (arg_2-arg_1)*(arg_2-arg_1)
        return fx

    def matrix(self):
                        
        elength = (self.arg_xmax-self.arg_xmin)/self.arg_element # no need to change                
        self.xvector[0] = self.arg_xmin

        # discretization of x-axis
        for i in range(self.arg_element):
            self.xvector[i+1] = self.xvector[i] + elength

        # element matrix initialization
        for i in range(self.arg_size):            
            for j in range(self.arg_size):
                if i==j:
                    self.eAmat[i][j] = 1.0
                else:
                    self.eAmat[i][j] = -1.0

        # A & b matrix initialize
        for i in range(self.arg_element):    
            for j in range(self.arg_size):        
                for k in range(self.arg_size):                    

                    # Simpson’s Rule integration
                    dh = (self.xvector[i+1] - self.xvector[i])*0.1666666 # devided by 6
                    half = (self.xvector[i+1] + self.xvector[i])*0.5

                    #00
                    self.eA2mat[0][0] = dh*(self.basfx00(self.xvector[i], self.xvector[i+1]) + 4*self.basfx00(half, self.xvector[i+1]) + self.basfx00(self.xvector[i+1], self.xvector[i+1]))

                    #01
                    self.eA2mat[0][1] = dh*(self.basfx01(self.xvector[i], self.xvector[i+1], self.xvector[i]) \
                                            + 4*self.basfx01(self.xvector[i], self.xvector[i+1], (self.xvector[i+1] + self.xvector[i])*0.5) \
                                            + self.basfx01(self.xvector[i], self.xvector[i+1], self.xvector[i+1]))
                    
                    #10
                    self.eA2mat[1][0] = dh*(self.basfx10(self.xvector[i], self.xvector[i+1], self.xvector[i]) \
                                            + 4*self.basfx10(self.xvector[i], self.xvector[i+1], (self.xvector[i+1] + self.xvector[i])*0.5) \
                                            + self.basfx10(self.xvector[i], self.xvector[i+1], self.xvector[i+1]))
                                        

                    #11
                    self.eA2mat[1][1] = dh*(self.basfx11(self.xvector[i], self.xvector[i+1]) + 4*self.basfx11(half, self.xvector[i+1]) + self.basfx11(self.xvector[i+1], self.xvector[i+1]))                    
                    
                    # This works if you use quadpack
                    #self.eA2mat[0][0] = quad(basfx00,self.xvector[i], self.xvector[i+1])[0]
                    #self.eA2mat[0][1] = quad(basfx01,self.xvector[i], self.xvector[i+1])[0]
                    #self.eA2mat[1][0] = quad(basfx10,self.xvector[i], self.xvector[i+1])[0]
                    #self.eA2mat[1][1] = quad(basfx11,self.xvector[i], self.xvector[i+1])[0]

                    self.Amat[j+i][k+i] += self.eAmat[j][k]
                    self.A2mat[j+i][k+i] += self.eA2mat[j][k]

        self.Amat = (1/elength)*self.Amat
        self.A2mat = (1/np.power(elength,2.0))*self.A2mat
        self.Amat = self.Amat - self.A2mat # dont forget the minus sign

        return self.xvector, self.Amat

# Boundary condition as function
class Boundary(FiniteElement):
    def __init__(self, arg_node_d, arg_node_n, arg_dirichlet, arg_neumann, arg_xmin, arg_xmax, arg_size, arg_element):
        # 1. This runs the FiniteElement.__init__         
        super().__init__(arg_xmin, arg_xmax, arg_size, arg_element)

        # 2. Then you set your specific boundary data
        self.arg_node_d = arg_node_d
        self.arg_node_n = arg_node_n
        self.arg_dirichlet = arg_dirichlet
        self.arg_neumann = arg_neumann        

        return None
        
    def boundary_d(self, arg_Amat):                             # Dirichlet boundary
        temp_b = self.bmat.copy()
        temp_b[:] -= self.arg_dirichlet*self.Amat[self.arg_node_d,:]
        temp_b[self.arg_node_d] = self.arg_dirichlet    

        arg_Amat[self.arg_node_d,:] = 0.0                       # row zero fix
        arg_Amat[:,self.arg_node_d] = 0.0                       # Column zero fix
        arg_Amat[self.arg_node_d, self.arg_node_d] = 1.0        # diagonal one fix

        return temp_b, arg_Amat
        
    def boundary_n(self):                                       # Neumann boundary        
        temp_b = self.bmat.copy()
        temp_b[self.arg_node_n] += self.arg_neumann

        return temp_b

class BoundaryMerge:
    def __init__(self, matA, matB):
        self.matA = matA
        self.matB = matB
    
    def merge(self):                                            # Dirichlet boundary
        merged = self.matA + self.matB

        return merged

class FileOutput:
    def __init__(self):
        pass

    def write(self, arg_v1, arg_v2):
        # write to file
        with open("fem.txt", "w") as f:
            for x, u in zip(arg_v1, arg_v2):
                f.write(f"{x} {u}\n")

        return None

# --- main routine ---
if __name__ == '__main__':
    # main start
    start = TimeCounter(time.time())

    # Set parameters
    element = 50
    xmax = 10    

    # fem, Boundary condition, and fileoutput class    
    fem = FiniteElement(0, xmax, 2, element)                        # xmin, xmax, size, and element    
    bc = Boundary(0, element, 0, np.cos(xmax), 0, xmax, 2, element) # Dirichlet node, Neumann node, Dirichlet condition, Neumann condition
                                                                    # The last four numbers are arguments for inheritance
    writer = FileOutput()
    
    ### FEM routine from here ###
    # Generation of Amat
    Amat = fem.matrix()[1]

    #np.set_printoptions(linewidth=np.inf)
    #print(Amat)

    # Boundary condition
    diri = bc.boundary_d(Amat)[0]   # with overwriting of Amat
    neum = bc.boundary_n()            
    mrg = BoundaryMerge(diri, neum) # Boundary condition merge

    np.set_printoptions(linewidth=np.inf)
    #print(Amat)
    #print(mrg.merge())
        
    # Solver
    umat = scipy.linalg.solve(Amat, mrg.merge())
    ### FEM routine until here ###

    # File output
    writer.write(fem.matrix()[0], umat)
    
    # time display
    end = TimeCounter(time.time())
    print(f"### elapsed_time: {end.time_return() - start.time_return():.2f} [sec] ###")