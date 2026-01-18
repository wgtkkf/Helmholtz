#ifndef _FUNCTION_H_
#define _FUNCTION_H_
#include <numbers>
#include <memory>      /* This is for a smart pointer */
#include <Eigen/Dense> /* this is for 2x2 matrix */

class Message
{
    public:
        void start();
        void end();
};

class FEM /* Finite Element class */
{
    public:                        
        struct FeedToSolver {                        
            Eigen::VectorXd bmat{};
            Eigen::MatrixXd Amat{};    /* Just declare type, NODE_TOTAL is necessary but is defined in the class below */
            Eigen::VectorXd xvector{}; /* Just declare type, NODE_TOTAL is necessary but is defined in the class below */
            Eigen::VectorXd uvector{};
        };

        FeedToSolver matrix();       /* return FeedToSolver */ 

        /* constant parameters */
        static constexpr double XMIN = 0;
        static constexpr double XMAX = 10; 
        static constexpr int SIZE = 2;                 /* element matrix size */
        static constexpr int ELEMENT = 50;             /* number of element */
        static constexpr int NODE_TOTAL = ELEMENT + 1; /* number of total node */                                    
        
    private:
        /* element marix */
        Eigen::MatrixXd A2mat = Eigen::MatrixXd::Zero(NODE_TOTAL, NODE_TOTAL);
        Eigen::Matrix2d eAmat = Eigen::Matrix2d::Zero();
        Eigen::Matrix2d eA2mat = Eigen::Matrix2d::Zero();        

        double shape_function_00(double x1, double x2);        
        double shape_function_01(double x1, double x2, double x3); 
        double shape_function_10(double x1, double x2, double x3); 
        double shape_function_11(double x1, double x2);

        /* parameters for calculations  */
        static constexpr double ELEMENT_LENGTH = (XMAX-XMIN)/ELEMENT; /* number of total node */
    
};

class Boundary /* Finite Element class */
{
    public:                        
        void boundary_d(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat);
        void boundary_n(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat);        

        /* nodes for boundary and conditions */
        static constexpr int NODE_D = 0;
        static constexpr int NODE_N = FEM::ELEMENT;
        static constexpr double D_C = 0.0;            /* Dirichlet condition  */
        static constexpr double N_C = std::cos(FEM::XMAX); /* Neumann condition  */
    
};

class Solver{
    public:
        void pivlu(Eigen::MatrixXd *Amat, Eigen::VectorXd *bmat, Eigen::VectorXd *umat);
};

class Display /* Finite Element class */
{
    public:                        
        void show(Eigen::MatrixXd* Amat, Eigen::VectorXd* bmat);
};

class File{
    public:
        void writer(Eigen::VectorXd *xmat, Eigen::VectorXd *ymat);      
};

#endif // _FUNCTION_H_