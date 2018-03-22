#ifndef EULER
#define EULER

#include "mesh.h"
#include "Eigen/SparseQR"

class Euler
{
protected:
	int NEWTON_MAX = 10;
	double h = 0.01;

	Mesh* SM;

	SparseMatrix<double> grad_g;
	VectorXd g, f;
	CholmodSupernodalLLT<SparseMatrix<double>> llt_solver;

public:
	Euler(Mesh* M, double timestep){
        h = timestep;
        SM = M;

        SparseMatrix<double> P = SM->get_Pf();
        SparseMatrix<double> K = SM->get_Stiffness();
        SparseMatrix<double> J = SM->get_RSJacobian(); //3v x9T
        f.resize(P.cols());
        g.resize(J.cols());
        grad_g.resize(J.cols(), J.cols());
        SparseMatrix<double> CholeskyAnalyzeBlock = J.transpose()*K*J;
        llt_solver.analyzePattern(CholeskyAnalyzeBlock);
        // this->llt_solver
    }

    void getZeroOrder(double& totalE){
    	double Kinetic = 0.5*SM->get_v().transpose()*SM->get_Mass()*SM->get_v();

    	double StrainInternal = 0;

    	//Change Mgh to Mg(dh) for numerical stabilty reasons
    	double Gravity = 0;
    	


    	totalE = Kinetic + StrainInternal + Gravity; 
    }

    void getFirstOrder();

    void getSecondOrder();

    void rs_step()
    {
    	bool Nan = false;
    	int iter;
    	VectorXd x_k = SM->get_rsx();

    	VectorXd& x_old = SM->get_rsx();
    	VectorXd& v_old = SM->get_rsv();
    	SparseMatrix<double> P = SM->get_Pf();
        SparseMatrix<double> RegMass = SM->get_Mass();
        SparseMatrix<double> J = SM->get_RSJacobian();
        SparseMatrix<double> K = SM->get_Stiffness();
        SparseMatrix<double> JMJ = J.transpose()*RegMass*J;
        SparseQR<SparseMatrix<double>, COLAMDOrdering<int>> lu_decomp(J);
		auto rank = lu_decomp.rank();
		std::cout<<"Rank: "<<rank<<" Rows: "<<J.rows()<<" Cols: "<<J.cols()<<std::endl;
        for(int iter=0; iter<NEWTON_MAX; ++iter)
        {
        	g.setZero();
        	grad_g.setZero();
        	std::cout<<SM->get_x()<<"\n\n"<<std::endl;
        	double error = SM->rsx_to_x();

        	SM->setForces(f, SM->get_x());
        	SM->setStiffnessMatrix(K,SM->get_x());
            std::cout<<"Breaks here"<<std::endl;
            std::cout<<JMJ.rows()<<" "<<JMJ.cols()<<std::endl;
            std::cout<<f.rows()<<" "<<J.cols()<<std::endl;
        	g = JMJ*x_k - JMJ*x_old - JMJ*v_old*h - J.transpose()*f*h*h;
        	grad_g = JMJ;// - J.transpose()*K*J*h*h;
        	llt_solver.compute(grad_g);
            if(llt_solver.info() == Eigen::NumericalIssue){
                cout<<"Possibly using a non- pos def matrix in the LLT method"<<endl;
                exit(0);
            }
            VectorXd dx = -1* llt_solver.solve(g);
            x_k += P*dx;
            SM->set_rsx(x_k);

            if(x_k != x_k)
            {
                Nan = true;
                break;
            }
            if(g.squaredNorm()/g.size()< 1e-3)
            {
                break;
            }
        }
        if(Nan){
            cout<<"ERROR: Newton's method doesn't converge"<<endl;
            cout<<iter<<endl;
            exit(0);
        }
        if(iter== NEWTON_MAX){
            cout<<"ERROR: Newton max reached"<<endl;
            cout<<iter<<endl;
            exit(0);
        }
        v_old = (x_k - x_old)/h;
        x_old = x_k;

    }


};
#endif