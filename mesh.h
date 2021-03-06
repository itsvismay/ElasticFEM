#ifndef MESH 
#define MESH

#include "tetrahedron.h"
#include "dg_element.h"
#include <Eigen/CholmodSupport>

using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;

class Mesh
{

protected:
    MatrixXd V;
    MatrixXi T;

	double medianMass, m_gravity;
    std::vector<Tetrahedron> tets;
    
    //Used in the sim
    SparseMatrix<double> InvMass;
    SparseMatrix<double> RegMass;
    SparseMatrix<double> StiffnessMatrix;

    SparseMatrix<double> Pf_fixMatrix;
    SparseMatrix<double> Pm_moveMatrix;
    SparseMatrix<double> RSJacobian;
    VectorXd x, v, f, xr, vr;
    int handle;
    //end

    std::vector<int> fixVertsIndices;
    std::vector<int> movVertsIndices;

public:
    Mesh(MatrixXi& TT, MatrixXd& TV, double youngs, double poissons, double gravity){
        m_gravity = gravity;
        this->V = TV.transpose().eval();
        this->T = TT;
        
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        #pragma omp parallel for
        for(int i=0; i<TT.rows(); i++){
            //based on Tet indexes, get the vertices of the tet from TV
            Tetrahedron t(TT.row(i), mu, lambda, i);
            t.precompute(this->V);
            this->tets.push_back(t);
        }
    }

    void initializeMesh(){
        int vertsNum = this->V.cols();
        InvMass.resize(3*vertsNum, 3*vertsNum); RegMass.resize(3*vertsNum, 3*vertsNum);
        InvMass.setZero();RegMass.setZero();

        StiffnessMatrix.resize(3*vertsNum, 3*vertsNum);

        x.resize(3*vertsNum);v.resize(3*vertsNum);f.resize(3*vertsNum);
        x.setZero();v.setZero();f.setZero();
        #pragma omp parallel for
        for(unsigned int k=0; k<this->tets.size(); ++k)
        {
            Vector4i indices = this->tets[k].getIndices();
            x(3*indices(0))   = this->V.col(indices(0))(0);
            x(3*indices(0)+1) = this->V.col(indices(0))(1);
            x(3*indices(0)+2) = this->V.col(indices(0))(2);

            x(3*indices(1))   = this->V.col(indices(1))(0);
            x(3*indices(1)+1) = this->V.col(indices(1))(1);
            x(3*indices(1)+2) = this->V.col(indices(1))(2);

            x(3*indices(2))   = this->V.col(indices(2))(0);
            x(3*indices(2)+1) = this->V.col(indices(2))(1);
            x(3*indices(2)+2) = this->V.col(indices(2))(2);

            x(3*indices(3))   = this->V.col(indices(3))(0);
            x(3*indices(3)+1) = this->V.col(indices(3))(1);
            x(3*indices(3)+2) = this->V.col(indices(3))(2);
        }


        this->setLumpedMassMatrix();
        this->setStiffnessMatrix(this->StiffnessMatrix, this->x);
        this->setConstraints(this->fixVertsIndices, this->movVertsIndices, this->Pf_fixMatrix, this->Pm_moveMatrix);
    }

    void initializeRSCoords(){
        xr.resize(9*tets.size());
        vr.resize(9*tets.size());
        xr.setZero();
        vr.setZero();
        handle = 1;
        
        RSJacobian.resize(3*this->V.cols(), 9*tets.size());
        RSJacobian.setZero();
        setRSJacobian(RSJacobian, xr);



    }

    void setRSJacobian(SparseMatrix<double>& RSJac, VectorXd& xr)
    {
        VectorXd seen;
        seen.resize(this->V.cols());
        seen.setZero();

        vector<Trip> triplets;
        triplets.reserve(3*9*4*this->tets.size());

        for(unsigned int t =0; t<this->tets.size(); ++t)
        {
            Vector4i indices = this->tets[t].getIndices();
            for(unsigned int i=0; i<4; ++i)
            {
                if(seen(indices(i))< 1e-8){
                    Vector3d v = this->V.col(indices(i)) - this->V.col(handle); 
                    MatrixXd localJac = this->tets[t].getdxdxrJacobian(xr, v);
                    
                    for(unsigned r =0; r<9; ++r){
                        triplets.push_back(Trip(3*indices(i), 9*this->tets[i].getRSIndex()+r, localJac(0,r)));
                        
                    }
                    for(unsigned r =0; r<9; ++r){
                        triplets.push_back(Trip(3*indices(i)+1, 9*this->tets[i].getRSIndex()+r, localJac(1,r)));
                        
                    }
                    for(unsigned r =0; r<9; ++r){
                        triplets.push_back(Trip(3*indices(i)+2, 9*this->tets[i].getRSIndex()+r, localJac(2,r)));
                        
                    }
                    seen(indices(i)) +=1;
                }
            }
        }
        RSJacobian.setFromTriplets(triplets.begin(), triplets.end());
        // std::cout<<RSJacobian<<std::endl;
    }

    void setNewYoungsPoissons(double youngs, double poissons, int index){
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        cout<<"NEW** Mu and Lambda from Youngs and Poissons"<<endl;

        tets[index].setMu(mu);
        tets[index].setLambda(lambda);
    }
    
    void setStiffnessMatrix(SparseMatrix<double>& K, VectorXd& xv){
        this->StiffnessMatrix.setZero();

        vector<Trip> triplets1;
        triplets1.reserve(12*12*this->tets.size());
        #pragma omp parallel for collapse(2)
        for(unsigned int i=0; i<this->tets.size(); i++)
        {
            //Get P(dxn), dx = [1,0, 0...], then [0,1,0,....], and so on... for all 4 vert's x, y, z
            //P is the compute Force Differentials blackbox fxn

            Vector12d dx(12);
            dx.setZero();
            Vector4i indices = this->tets[i].getIndices();
            int kj;
            for(unsigned int j=0; j<12; j++)
            {
                dx(j) = 1;
                MatrixXd dForces = this->tets[i].computeForceDifferentials(xv, dx);
                kj = j%3;
                //row in order for dfxi/dxi ..dfxi/dzl
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[0], dForces(0,0)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[0]+1, dForces(1,0)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[0]+2, dForces(2,0)));

                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[1], dForces(0,1)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[1]+1, dForces(1,1)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[1]+2, dForces(2,1)));

                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[2], dForces(0,2)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[2]+1, dForces(1,2)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[2]+2, dForces(2,2)));

                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[3], dForces(0,3)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[3]+1, dForces(1,3)));
                triplets1.push_back(Trip(3*indices[j/3]+kj, 3*indices[3]+2, dForces(2,3)));
                dx(j) = 0; //ASK check is this efficient?
            }
        }
        K.setFromTriplets(triplets1.begin(), triplets1.end());
        return;
    }
    
    void setLumpedMassMatrix(){
        int vertsNum = this->V.cols();
        VectorXd massVector;
        massVector.resize(3*vertsNum);
        massVector.setZero();
        
        #pragma omp parallel for
        for(unsigned int i=0; i<this->tets.size(); i++){
            double vol = (this->tets[i].getUndeformedVolume()/4)*1e2; //density UNITS: kg/m^3
            Vector4i indices = this->tets[i].getIndices();

            massVector(3*indices(0)) += vol;
            massVector(3*indices(0)+1) += vol;
            massVector(3*indices(0)+2) += vol;

            massVector(3*indices(1)) += vol;
            massVector(3*indices(1)+1) += vol;
            massVector(3*indices(1)+2) += vol;

            massVector(3*indices(2)) += vol;
            massVector(3*indices(2)+1) += vol;
            massVector(3*indices(2)+2) += vol;

            massVector(3*indices(3)) += vol;
            massVector(3*indices(3)+1) += vol;
            massVector(3*indices(3)+2) += vol;
        }
        vector<double>tempForMedian;
        for(int i=0; i<3*vertsNum; i++){
            InvMass.coeffRef(i,i) = 1/massVector(i);
            RegMass.coeffRef(i,i) = massVector(i);
            tempForMedian.push_back(massVector(i));
        }
        sort(tempForMedian.begin(), tempForMedian.end());
        if(tempForMedian.size()%2 == 0){
            this->medianMass = 0.5*(tempForMedian[tempForMedian.size()/2-1]+tempForMedian[tempForMedian.size()/2]);
        }else{
            this->medianMass = tempForMedian[tempForMedian.size()/2];
        }
        cout<<"MEDIAN"<<endl;
        cout<<this->medianMass<<endl;
    }

    void setForces(VectorXd& f, VectorXd& xi){
        // //gravity
        f.setZero();
        for(unsigned int i=0; i<f.size()/3; i++){
            f(3*i+1) += this->RegMass.coeff(3*i+1, 3*i+1)*m_gravity;
        }

        //elastic
        #pragma omp parallel for
        for(unsigned int i=0; i<this->tets.size(); i++){
            this->tets[i].computeElasticForces(xi, f);
        }
        
        // //damping

        
        return;
    }

    void setConstraints(std::vector<int>& fix, std::vector<int> m, SparseMatrix<double>& Pf, SparseMatrix<double>& Pm){
        //fix max y
        int axis = 1;
        double tolr = 1e-4;
        double minx = this->V.rowwise().minCoeff()(axis);
        double maxx = this->V.rowwise().maxCoeff()(axis);
        
        for(int i=0; i<this->V.cols(); ++i)
        {
            if (this->V.col(i)(axis)> maxx-tolr ) 
            {
                // fix.push_back(i);
            }
        }


        //TODO: MAKE SURE fix vector is sorted
        if(fix.size()==0){
            this->Pf_fixMatrix.resize(3*this->V.cols(), 3*this->V.cols());
            this->Pf_fixMatrix.setIdentity();
            return;
        }
        
        Pf.resize(3*this->V.cols(), 3*this->V.cols() - 3*fix.size());
        Pf.setZero();

        int c =0;
        int j =0;
        
        for(int i=0; i<this->V.cols(); ++i)
        {
            if(i != fix[c])
            {
                Pf.coeffRef(3*i, 3*j) = 1.0;
                Pf.coeffRef(3*i+1, 3*j+1) =1.0;
                Pf.coeffRef(3*i+2, 3*j+2) =1.0;
                j+=1;
            }else
            {
                c+=1;
            }
        }

        this->Pf_fixMatrix = Pf;
    }


    MatrixXd getCurrentVerts(){
        Eigen::Map<Eigen::MatrixXd> newV(this->x.data(), this->V.rows(), this->V.cols());
        return newV.transpose();
    }

    double rsx_to_x(){

        double error = 0;
        VectorXd seen;
        seen.resize(this->V.cols());
        seen.setZero();

        for(auto& e: tets)
        {
            std::cout<<e.getRSF()<<std::endl;
            VectorXi indices = e.getIndices();
            for(unsigned int j =0; j<indices.size(); ++j)
            {
                Vector3d v = this->V.col(indices(j)) - this->V.col(handle);
                if(seen(indices(j)) > 1e-8){
                    error = (x.segment<3>(3*indices(j)) - e.getRSF()*v + this->V.col(handle)).squaredNorm();
                }else{
                    x.segment<3>(3*indices(j)) = e.getRSF()*v + this->V.col(handle);
                    seen(indices(j)) += 1;
                }
                
            }
        }

        std::cout<<"error: "<<error<<std::endl;
        return error;
    }

    void set_rsx(VectorXd& xr){
        for(auto& e: tets){
            e.setRSU(xr);
        }
    }

    double getInternalEnergy(VectorXd& xi){
        double E = 0;
        #pragma omp parallel for 
        for(auto& tet: this->tets){
            E += tet.getInternalEnergy(xi);
        }
        return E;
    }

    inline double get_gravity(){ return this->m_gravity; } 

    inline std::vector<Tetrahedron>& getTets(){ return this->tets; }

    inline MatrixXd& getColwiseV(){ return this->V; }
    
    inline MatrixXi& getT(){ return this->T; }

    inline VectorXd get_copy_x(){ return this->x; }

    inline VectorXd& get_x(){ return this->x; }

    inline VectorXd& get_v(){ return this->v;}

    inline VectorXd& get_rsx(){ return this->xr; }

    inline VectorXd& get_rsv(){ return this->vr;}

    inline SparseMatrix<double>& get_Mass(){ return this->RegMass;}

    inline SparseMatrix<double>& get_Stiffness(){ return this->StiffnessMatrix;}

    inline SparseMatrix<double>& get_Pf(){ return this->Pf_fixMatrix;}

    inline SparseMatrix<double>& get_RSJacobian(){ return this->RSJacobian; }
};

#endif