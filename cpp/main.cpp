#include<iostream>
#include "include/function.h"

int main()
{
    Message obj_m;    
    FEM obj_fe;
    Boundary obj_bc;
    Solver obj_solver;
    Display obj_disp;
    File obj_file;

    obj_m.start();
    FEM::FeedToSolver tensors = obj_fe.matrix(); /* define struct */

    obj_bc.boundary_d(&tensors.Amat, &tensors.bmat);
    obj_bc.boundary_n(&tensors.Amat, &tensors.bmat);    
    /* obj_disp.show(&tensors.Amat, &tensors.bmat); */

    obj_solver.pivlu(&tensors.Amat, &tensors.bmat, &tensors.uvector);
    obj_file.writer(&tensors.xvector, &tensors.uvector);

    obj_m.end();    

    return 0;
}

// See the below!
// Execute with this command: ./run.sh