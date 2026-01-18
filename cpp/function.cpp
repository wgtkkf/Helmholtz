#include <iostream>    // need this
#include "include/function.h"
#include <cmath>       // for mathemetical operator such as power function
#include <memory>      // This is for a smart pointer
#include <format>      // for printing out
#include <fstream>     // Required for file operations
#include <iomanip>     // For formatting
#include <Eigen/Dense> // For partialPivLu


/* Message methods */
void Message::start()
{
    std::cout << "### Calculation started.   ###\n";
}

void Message::end()
{
    std::cout << "### Calculation completed. ###\n";
}

/* FEM methods */
/* shape functions */
double FEM::shape_function_00(double x1, double x2)
{
    double fx;
    fx = (x2-x1)*(x2-x1);
    return fx;
}

double FEM::shape_function_01(double x1, double x2 , double x3)
{    
    double fx;
    fx = -(x2-x3)*(x1-x3);
    return fx;
}

double FEM::shape_function_10(double x1, double x2 , double x3)
{
    double fx;
    fx = -(x2-x3)*(x1-x3);
    return fx;
}

double FEM::shape_function_11(double x1, double x2)
{
    double fx;
    fx = (x2-x1)*(x2-x1);
    return fx;
}

/* matrix */
FEM::FeedToSolver FEM::matrix() /* FeedToSolver in FEM class */
{
    FeedToSolver tensors;

    // 1. Declare and size of Amat and xvector here inside of FeedToSolver    
    tensors.bmat = Eigen::VectorXd::Zero(NODE_TOTAL);
    tensors.Amat = Eigen::MatrixXd::Zero(NODE_TOTAL, NODE_TOTAL);    
    tensors.xvector = Eigen::VectorXd::Zero(NODE_TOTAL);
    tensors.uvector = Eigen::VectorXd::Zero(NODE_TOTAL);    
    
    int i, j, k;     /* parameters for ForLoop */    
    double dh, half;    

    tensors.xvector[0] = XMIN;    

    /* discretization of x-axis */
    for (int i=0 ; i<ELEMENT; ++i)
    {
        tensors.xvector[i+1] = tensors.xvector[i] + ELEMENT_LENGTH;        
    }

    /* element matrix initialization */
    for (i=0 ; i<SIZE; ++i)
    {       
        for (j=0 ; j<SIZE; ++j)
        {
            if (i==j)
            {
                eAmat(i,j) = 1.0;
            }else{
                eAmat(i,j) = -1.0;
            }
        }
    }

    /* A & b matrix initialization */
    for (i=0 ; i<ELEMENT; ++i)
    {        
        for (j=0 ; j<SIZE; ++j)
        {
            for (k=0 ; k<SIZE; ++k)
            {
                /* Simpson's rule integration */
                dh = (tensors.xvector[i+1] - tensors.xvector[i])*0.16666666; /* devided by 6 */
                half = (tensors.xvector[i+1] + tensors.xvector[i])*0.5;

                /* 00 */
                eA2mat(0,0) = dh*(shape_function_00(tensors.xvector[i], tensors.xvector[i+1]) 
                            + 4*shape_function_00(half, tensors.xvector[i+1]) 
                            + shape_function_00(tensors.xvector[i+1], tensors.xvector[i+1]));

                /* debug */
                /* std::cout << std::format("{:.2f}\n", eA2mat(0,0)); */

                /* 01 */
                eA2mat(0,1) = dh*(shape_function_01(tensors.xvector[i], tensors.xvector[i+1], tensors.xvector[i]) \
                            + 4*shape_function_01(tensors.xvector[i], tensors.xvector[i+1], (tensors.xvector[i+1] + tensors.xvector[i])*0.5) \
                            + shape_function_01(tensors.xvector[i], tensors.xvector[i+1], tensors.xvector[i+1]));

                /* 10 */
                eA2mat(1,0) = dh*(shape_function_10(tensors.xvector[i], tensors.xvector[i+1], tensors.xvector[i]) \
                            + 4*shape_function_10(tensors.xvector[i], tensors.xvector[i+1], (tensors.xvector[i+1] + tensors.xvector[i])*0.5) \
                            + shape_function_10(tensors.xvector[i], tensors.xvector[i+1], tensors.xvector[i+1]));

                /* 11 */
                eA2mat(1,1) = dh*(shape_function_11(tensors.xvector[i], tensors.xvector[i+1]) 
                            + 4*shape_function_11(half, tensors.xvector[i+1]) 
                            + shape_function_11(tensors.xvector[i+1], tensors.xvector[i+1]));

                tensors.Amat(j+i,k+i) += eAmat(j,k);
                A2mat(j+i,k+i) += eA2mat(j,k);
            }
        }
    }

    tensors.Amat = (1/ELEMENT_LENGTH)*tensors.Amat;
    A2mat = (1/std::pow(ELEMENT_LENGTH, 2))*A2mat;

    tensors.Amat = tensors.Amat - A2mat;
    
    return tensors;
}

void Boundary::boundary_d(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat)
{
    *bmat -= D_C * Amat->row(NODE_D).transpose();
    (*bmat)(NODE_D) = D_C;
    Amat->row(NODE_D).setZero();
    Amat->col(NODE_D).setZero();
    (*Amat)(NODE_D, NODE_D) = 1.0;    
}

void Boundary::boundary_n(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat)
{
    (*bmat)(NODE_N) = N_C;
}

void Solver::pivlu(Eigen::MatrixXd *Amat, Eigen::VectorXd *bmat, Eigen::VectorXd *umat)
{    

    if (Amat->rows() != bmat->size()) {
        std::cerr << "ERROR: Dimension mismatch! Cannot solve." << std::endl;        
    }
    // Debugging: Print dimensions before solving

    *umat = Amat->partialPivLu().solve(*bmat);    
}

void Display::show(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat)
{
    std::cout << "--- Matrix Amat ---" << std::endl;
    std::cout << *Amat << std::endl;
    std::cout << "-------------------" << std::endl;

    std::cout << "--- Matrix Bmat ---" << std::endl;
    std::cout << *bmat << std::endl;
    std::cout << "-------------------" << std::endl;
}

void File::writer(Eigen::VectorXd *xmat, Eigen::VectorXd *ymat)
{
    std::ofstream outFile("1dfem_cpp.txt");

    // Check if file is open
    if (outFile.is_open()){
        for (int i = 0; i < xmat->size(); ++i) {
            outFile << std::fixed << std::setprecision(6) 
            << (*xmat)(i) << " " << (*ymat)(i) << "\n";
    }
    outFile.close();}
}