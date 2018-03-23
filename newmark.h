#ifndef NEWMARK
#define NEWMARK

#include "mesh.h"

class Newmark
{
protected:
    int NEWTON_MAX = 10;
    double beta = 0.25;
    double gamma = 0.5;
    double h = 0.01;

    Mesh* SM;

    VectorXd x_k;
    VectorXd v_k;
    VectorXd f_o;

    SparseMatrix<double> grad_g;
    VectorXd g;
    CholmodSupernodalLLT<SparseMatrix<double>> llt_solver;

public:
    Newmark(Mesh* M, double timestep){
        h = timestep;
        SM = M;
        double dofs = SM->get_x().rows();
        x_k.resize(dofs);
        v_k.resize(dofs);
        f_o.resize(dofs);

        SparseMatrix<double> P = SM->get_Pf();
        SparseMatrix<double> K = SM->get_Stiffness();
        g.resize(P.cols());
        grad_g.resize(P.cols(), P.cols());
        SparseMatrix<double> CholeskyAnalyzeBlock = P.transpose()*K*P;
        llt_solver.analyzePattern(CholeskyAnalyzeBlock);
        // this->llt_solver
    }

    void step(){
        bool Nan = false;
        int iter;
        x_k = SM->get_copy_x();
        v_k.setZero();
        SM->setForces(f_o, x_k);
        SparseMatrix<double>& P = SM->get_Pf();
        SparseMatrix<double>& RegMass = SM->get_Mass();
        SparseMatrix<double>& K = SM->get_Stiffness();
        VectorXd& v_old = SM->get_v();
        VectorXd& x_old = SM->get_x();
        VectorXd force = f_o;

        for(int iter=0; iter<NEWTON_MAX; ++iter)
        {
            g.setZero();
            grad_g.setZero();


            SM->setForces(force,x_k);
            SM->setStiffnessMatrix(K, x_k);

            g = P.transpose()*(RegMass * x_k - RegMass * x_old - h*RegMass * v_old - (h*h/2)*(1-2*beta) * f_o - (h*h*beta)*force);
            grad_g = P.transpose()* (RegMass - h*h*beta*K) * P;
            
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

};

#endif