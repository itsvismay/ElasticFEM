#ifndef EULER
#define EULER

#include "mesh.h"
#include "Eigen/SparseQR"
#include "optimization.h"
using namespace alglib;

class Euler
{
protected:
	int NEWTON_MAX = 10;
	double h = 0.01;

	Mesh* SM;

	SparseMatrix<double> grad_g;
	VectorXd g, f, xbfgs;
	CholmodSupernodalLLT<SparseMatrix<double>> llt_solver;

public:
	Euler(Mesh* M, double timestep, bool RSCoords=false){
        h = timestep;
        SM = M;

        SparseMatrix<double>& P = SM->get_Pf();//3v x 3red
        SparseMatrix<double>& K = SM->get_Stiffness();
        SparseMatrix<double> CholeskyAnalyzeBlock;
        f.resize(SM->get_x().rows());


        if(RSCoords){
            SparseMatrix<double> J = SM->get_RSJacobian(); //3v x9T
            CholeskyAnalyzeBlock = J.transpose()*K*J;
            g.resize(J.cols());
            grad_g.resize(J.cols(), J.cols()); 
        }else{
            g.resize(P.cols());
            grad_g.resize(P.cols(), P.cols());
            CholeskyAnalyzeBlock = P.transpose()*K*P;
        }
        llt_solver.analyzePattern(CholeskyAnalyzeBlock);
        // this->llt_solver
    }

    inline VectorXd& get_xbfgs(){return xbfgs; }
    inline VectorXd& get_g(){return g; }

    double getZeroOrder(VectorXd& x_k){
    	//this term combines 0.5*x_k.T*M*x_k - 0.5*x_k.T*M*x_o - 0.5*x_k.T*M*v_o*t + SE*t^2

        std::cout<<"ENERGIES"<<std::endl;
        VectorXd new_v = ((x_k - SM->get_x())/h) - SM->get_v();
        double KineticE = 0.5*new_v.transpose()*SM->get_Mass()*SM->get_v();
        std::cout<<"KE: "<<new_v.norm()<<std::endl;

        double StrainInternal = SM->getInternalEnergy(x_k);
        std::cout<<"SE: "<<StrainInternal<<std::endl;

        //Change Mgh to Mg(dh) for numerical stabilty reasons
        double Gravity = 0;
        #pragma omp parallel for 
        for(unsigned int i=0; i<x_k.size()/3; ++i){
            Gravity += SM->get_gravity()*-1*SM->get_Mass().coeffRef(3*i+1, 3*i+1)*(x_k(3*i+1) - SM->get_x()(3*i+1));
        }


        double PotentialE = (StrainInternal + Gravity)*h;//Maybe get rid of the h for numerical reasons
        std::cout<<PotentialE<<std::endl;
    	return KineticE + PotentialE;
    }

    void getFirstOrder(VectorXd& x_k, VectorXd& g){
        SparseMatrix<double>& P = SM->get_Pf();
        SM->setForces(f, x_k);
        g = P.transpose()*(
            SM->get_Mass()*x_k - 
            SM->get_Mass()*SM->get_x() - 
            SM->get_Mass()*SM->get_v()*h - 
            f*h*h);
    	
        return;
    }	

    void getSecondOrder(VectorXd& x_k, SparseMatrix<double>& grad_g){

        SparseMatrix<double>& P = SM->get_Pf();
        SparseMatrix<double>& K = SM->get_Stiffness();
        SM->setStiffnessMatrix(K, x_k);
        grad_g = P.transpose()*(SM->get_Mass() - h*h*K)*P;

        return;
    }

    void rs_step()
    {
    	bool Nan = false;
    	int iter;
    	VectorXd x_k = SM->get_rsx();

    	VectorXd& x_old = SM->get_rsx();
    	VectorXd& v_old = SM->get_rsv();
    	SparseMatrix<double>& P = SM->get_Pf();
        SparseMatrix<double>& RegMass = SM->get_Mass();
        SparseMatrix<double>& J = SM->get_RSJacobian();
        SparseMatrix<double>& K = SM->get_Stiffness();
        SparseMatrix<double> JMJ = J.transpose()*RegMass*J;
        
        //Trash this
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

    void step()
    {
        bool Nan = false;
        int iter;
        VectorXd x_k = SM->get_copy_x();
        VectorXd& x_old = SM->get_x();
        VectorXd& v_old = SM->get_v();

        SparseMatrix<double>& P = SM->get_Pf();
        VectorXd force;

        for(int iter=0; iter<NEWTON_MAX; ++iter)
        {
            g.setZero();
            grad_g.setZero();

            this->getFirstOrder(x_k, g);
            this->getSecondOrder(x_k, grad_g);
            
            std::cout<<force.squaredNorm()<<std::endl;
            llt_solver.compute(grad_g);
            if(llt_solver.info() == Eigen::NumericalIssue){
                cout<<"Possibly using a non- pos def matrix in the LLT method"<<endl;
                exit(0);
            }
            VectorXd dx = -1* llt_solver.solve(g);
            x_k += P*dx;
            
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

    void bfgs_step()
    {
        SparseMatrix<double>& P = SM->get_Pf();
        xbfgs = P.transpose()*SM->get_copy_x();
        // make sure xbfgs is fully sized when passed into getInternalEnergy

        VectorXd& v_old = SM->get_v();
        VectorXd& x_old = SM->get_x();

        auto rep_x = [](const real_1d_array &x, double func, void *ptr){};

        auto function_grad = [](const real_1d_array &x, double &func, real_1d_array &grad, void *ptr)
        {
            Euler* euler = (Euler*) ptr;
            VectorXd& xbfgs = euler->get_xbfgs();
            std::cout<<"func: "<<xbfgs.size()<<std::endl;
            for(unsigned int i=0; i<xbfgs.size(); ++i)
            {
                xbfgs(i) = x[i];
            }

            func = euler->getZeroOrder(xbfgs);
            std::cout<<"F: "<<func<<std::endl;
            VectorXd& g = euler->get_g();
            // std::cout<<g<<std::endl;
            euler->getFirstOrder(xbfgs, g);
            for(unsigned i=0; i<g.size(); ++i)
            {
                grad[i] = g(i);
            }
        };

        std::cout<<"func: "<<xbfgs.size()<<std::endl;
        real_1d_array y;
        double *positions= new double[xbfgs.size()];
        for(int i=0; i<xbfgs.size(); i++){
            positions[i] = xbfgs(i); // y = (x_k -x_o)/h - v // y = 0 - v
        }
        y.setcontent(SM->get_x().size(), positions);

        double epsg = sqrt(1e-11)/h;
        double epsf = 0;
        double epsx = 0;
        double stpmax = 0;
        ae_int_t maxits = 0;
        minlbfgsstate state;
        minlbfgsreport rep;
        minlbfgscreate(12, y, state);
        minlbfgssetcond(state, epsg, epsf, epsx, maxits);
        minlbfgssetxrep(state, true);
        minlbfgsoptimize(state, function_grad, rep_x, this);
        minlbfgsresults(state, y, rep);

        for(int i=0; i<xbfgs.size(); i++){
            xbfgs(i) = y[i] ; // y = (x_k -x_o)/h - v // y = 0 - v
        }
        v_old = (P*xbfgs - SM->get_x())/h;
        x_old = P*xbfgs;

    }


};
#endif