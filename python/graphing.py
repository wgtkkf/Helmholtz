import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

# =============================================
# This code was generated supported by Gemini =
# =============================================
class Graph:
    # Modified to accept two sets of data: (x1, y1) and (x2, y2)
    def __init__(self, x1, y1, x2, y2, x3, y3):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.x3 = x3
        self.y3 = y3

    def display(self):
        csfont = {'fontname':'DejaVu Sans'} 
        plt.figure() 
        plt.grid(True, which="both", ls="-", alpha=0.5) # Enhanced grid for log scale

        # Plot 1: Analytical (Python)
        plt.plot(self.x1, self.y1, 'blue', linestyle='solid', label="4th order runge-kutta method") 
        
        # Plot 2: Numerical (C++) - Using a dashed red line to distinguish
        plt.plot(self.x2, self.y2, 'lime', linestyle='dashed', label="analytical solution") 

        # Plot 3: Finite Element Method (Python) - Using a dashed red line to distinguish
        plt.plot(self.x3, self.y3, 'red', linestyle='dotted', label="finite element method, python") 

        ## graph information    
        plt.xlabel(r'x [-]', **csfont)            
        plt.ylabel(r'y [-]', **csfont)

        # font for legend
        font = font_manager.FontProperties(family='DejaVu Sans',
                                           weight='bold',
                                           style='normal', size=10)
        plt.legend(loc='upper right', prop=font)         

        #
        plt.xlim(0, 10) # Sets x-axis from 0 to 100
        plt.ylim(-2, 2) # Useful for log scales

        # plot options
        plt.xticks([0, 5, 10], **csfont)
        plt.yticks([-2, -1, 0, 1, 2], **csfont)

        plt.tight_layout() # Ensures labels don't get cut off in the PNG        
        print("Saving graph to comparison.png...")

        # graph save & display
        plt.savefig("comparison.png", dpi=300) 
        plt.show()                  
        return 0

# ==========================================
# DATA LOADING LOGIC (EXTERNAL)
# ==========================================

# 1. Load C++ data (Dynamic row length)
cpp_data = np.loadtxt("solution.txt")
x_cpp = cpp_data[:, 0]
y_cpp = cpp_data[:, 1] # 4th order runge-kutta method

# 2. Load Python data (Dynamic row length)
py_data = np.loadtxt("solution.txt")
x_py = py_data[:, 0]
y_py = py_data[:, 3] # analytical solution

# 3. Load FEM data (Dynamic row length)
py_data = np.loadtxt("fem.txt")
x_fem_py = py_data[:, 0]
y_fem_py = py_data[:, 1] # finite element method

# 3. Create the instance with BOTH datasets
my_graph = Graph(x_py, y_py, x_cpp, y_cpp, x_fem_py, y_fem_py)

# 4. Generate the single combined plot
my_graph.display()