#include <stdio.h>

#include <igl/viewer/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <Eigen/LU>
#include <json.hpp>
#include <Eigen/CholmodSupport>

using json = nlohmann::json;

using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;
json j_input;

//############ TETRAHEDRON 

    class Tetrahedron
    {
    public:
    	Tetrahedron(VectorXi k, double mu, double lambda);
        void computeElasticForces(VectorXd& xi, VectorXd& f);
        void precompute(MatrixXd& TV);
        MatrixXd computeForceDifferentials(VectorXd& xi, Vector12d& dx);

        inline double getUndeformedVolume();
        inline VectorXi& getIndices();
        inline double getMu();
        inline double getLambda();
        inline Vector3d& getCentroid();

        void setMu(double mu);
        void setLambda(double lambda);
        void set_fibre_mag(double mag);
        void set_fibre_dir(Vector3d& v);


    protected:
    	Matrix3d DeformedShapeMatrix, ReferenceShapeMatrix, InvRefShapeMatrix;
        VectorXi verticesIndex;
        double undeformedVol, energy, currentVol;
        double mu, lambda, fibre_mag;
        Vector3d fibre_dir, tet_centroid;
    };

    Tetrahedron::Tetrahedron(VectorXi k, double mu, double lambda)
    {
        this->verticesIndex = k ;
        this->mu = mu;
        this->lambda = lambda;
        this->fibre_mag = 0;
        this->fibre_dir.setZero();
    }

    inline double Tetrahedron::getUndeformedVolume(){ return this->undeformedVol; } 
    
    inline VectorXi& Tetrahedron::getIndices(){ return this->verticesIndex; } 

    inline double Tetrahedron::getMu(){ return this->mu; } 

    inline double Tetrahedron::getLambda(){ return this->lambda; } 

    inline Vector3d& Tetrahedron::getCentroid() { return this->tet_centroid; }

    void Tetrahedron::setMu(double mu){ this->mu =mu; } 

    void Tetrahedron::setLambda(double lambda){ this->lambda=lambda; }

    void Tetrahedron::set_fibre_mag(double mag){ this->fibre_mag = mag; }

    void Tetrahedron::set_fibre_dir(Vector3d& v)
    {
        this->fibre_dir(0) = v(0);
        this->fibre_dir(1) = v(1);
        this->fibre_dir(2) = v(2);

    }
        
    void Tetrahedron::precompute(MatrixXd& TV)
    {
        Matrix3d Dm;
        for(int i=0; i<3; i++)
        {
            Dm.col(i) = TV.col(verticesIndex(i)) - TV.col(verticesIndex(3));
        }
        
        this->ReferenceShapeMatrix = Dm;
        this->InvRefShapeMatrix = Dm.inverse();
        this->undeformedVol = (1.0/6)*fabs(Dm.determinant());
        this->tet_centroid = (1.0/4)*(TV.col(verticesIndex(0)) + TV.col(verticesIndex(1)) + TV.col(verticesIndex(2)) + TV.col(verticesIndex(3)));
    }

    void Tetrahedron::computeElasticForces(VectorXd& xi, VectorXd& f)
    {

        Matrix3d Ds;
        for(int i=0; i<3; i++)
        {
            Ds.col(i) = xi.segment<3>(3*verticesIndex(i)) - xi.segment<3>(3*verticesIndex(3));
        }

        Matrix3d F = Ds*this->InvRefShapeMatrix;

        Matrix3d P;

        this->currentVol = (1.0/6)*fabs(Ds.determinant());
        
        //Neo
        double J = F.determinant();
        double I1 = (F.transpose()*F).trace();
        double powj = pow(J, -2.0/3.0);
        double I1bar = powj*I1;

        P = mu*(powj * F) +
        (- mu/3.0 * I1 * powj + lambda*(J-1.0)*J)*F.inverse().transpose();

        this->energy = this->undeformedVol*(mu/2.0 * (I1bar - 3) + lambda/2.0 * (J-1.0) * (J-1.0));

        
        if(F.determinant()<0)
        {
            this->energy = 1e40;
            cout<<"ERROR: F determinant is 0"<<endl;
            cout<<"Decrease timestep maybe - instantaneous force is too much with this timestep"<<endl;
            exit(0);
        }
        if(this->energy != this->energy)
        {
            //NANS
            cout<<"ENERGY nans"<<endl;
            exit(0);
        }

        Matrix3d H = -1*this->undeformedVol*P*((this->InvRefShapeMatrix).transpose());


        //##Muscle force, do the proper math later
        double v1 =this->fibre_dir[0];
        double v2 =this->fibre_dir[1];
        double v3 =this->fibre_dir[2];
        double a = this->fibre_mag;
        
        double o = this->InvRefShapeMatrix(0,0);
        double p = this->InvRefShapeMatrix(0,1);
        double q = this->InvRefShapeMatrix(0,2);
        double t = this->InvRefShapeMatrix(1,0);
        double r = this->InvRefShapeMatrix(1,1);
        double s = this->InvRefShapeMatrix(1,2);
        double u = this->InvRefShapeMatrix(2,0);
        double w = this->InvRefShapeMatrix(2,1);
        double e = this->InvRefShapeMatrix(2,2);

        double nx1_nx4 = Ds(0,0);
        double nx2_nx4 = Ds(0,1);
        double nx3_nx4 = Ds(0,2);
        double ny1_ny4 = Ds(1,0);
        double ny2_ny4 = Ds(1,1);
        double ny3_ny4 = Ds(1,2);
        double nz1_nz4 = Ds(2,0);
        double nz2_nz4 = Ds(2,1);
        double nz3_nz4 = Ds(2,2);

        //Symbolic derivative for P_muscle = dEnergy_muscle / dF, because I can't do tensor calc
        VectorXd force_muscle(12);
        force_muscle<<
        (-1 *a *(o *v1 + p *v2 + q *v3) *this->undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1* a* (o *v1 + p *v2 + q *v3) *this->undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* (o *v1 + p *v2 + q *v3) *this->undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1* a* (t* v1 + r* v2 + s* v3)*this->undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1* a* (t* v1 + r* v2 + s* v3) *this->undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* (t* v1 + r* v2 + s* v3) *this->undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1 *a * (u* v1 + e* v3 + v2* w) *this->undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1 *a * (u* v1 + e* v3 + v2* w) *this->undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1 *a * (u* v1 + e* v3 + v2* w) *this->undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1* a* this->undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((nx1_nx4)* o + (nx2_nx4)* t + (nx3_nx4)* u)* v1 + 
                (e *(nx3_nx4) + (nx1_nx4)* q + (nx2_nx4)* s)* v3 + 
                v2 *((nx1_nx4)* p + (nx2_nx4)* r + (nx3_nx4)* w))),

        (-1* a* this->undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u)* v1 + 
                (e *(ny3_ny4) + (ny1_ny4)* q + (ny2_ny4)* s)* v3 + 
                v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* this->undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u)* v1 + 
                (e *(nz3_nz4) + (nz1_nz4)* q + (nz2_nz4)* s)* v3 + 
                v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w)));

        //--------


        f.segment<3>(3*verticesIndex(0)) += H.col(0) + force_muscle.segment<3>(0);
        f.segment<3>(3*verticesIndex(1)) += H.col(1) + force_muscle.segment<3>(3);
        f.segment<3>(3*verticesIndex(2)) += H.col(2) + force_muscle.segment<3>(6);
        f.segment<3>(3*verticesIndex(3)) += -1*H.col(0) - H.col(1) - H.col(2) + force_muscle.segment<3>(9);
    }

    MatrixXd Tetrahedron::computeForceDifferentials(VectorXd& xi, Vector12d& dx)
    {

        Matrix3d Ds;
        for(int i=0; i<3; i++)
        {
            Ds.col(i) = xi.segment<3>(3*verticesIndex(i)) - xi.segment<3>(3*verticesIndex(3));
        }

        Matrix3d dDs;
        dDs <<  (dx(0) - dx(9)),   (dx(3) - dx(9)), (dx(6) - dx(9)),
                (dx(1) - dx(10)), (dx(4) - dx(10)), (dx(7) - dx(10)),
                (dx(2) - dx(11)), (dx(5) - dx(11)), (dx(8) - dx(11));

        Matrix3d F = Ds*this->InvRefShapeMatrix;
        Matrix3d dF = dDs*this->InvRefShapeMatrix;

        Matrix3d dP;
        
        //Neohookean
        double detF = F.determinant();
        double logdetF = log(detF);
        Matrix3d FInvTransp = (F.inverse()).transpose();
        dP = mu*dF + (mu - lambda*logdetF)*(FInvTransp)*dF.transpose()*(FInvTransp) + lambda*(F.inverse()*dF).trace()*(FInvTransp);
        

        Matrix3d dH = -1*this->undeformedVol*dP*((this->InvRefShapeMatrix).transpose());


        MatrixXd dForces(3,4);
        dForces.col(0) = dH.col(0);
        dForces.col(1) = dH.col(1);
        dForces.col(2) = dH.col(2);
        dForces.col(3) = -1*dH.col(0) - dH.col(1) - dH.col(2);

        return dForces;
    }

//############ END 

//############ SOLID_MESH 

    class SolidMesh
    {

    protected:
        MatrixXd V;
        MatrixXi T;

    	double medianMass;
        std::vector<Tetrahedron> tets;
        
        //Used in the sim
        SparseMatrix<double> InvMass;
        SparseMatrix<double> RegMass;
        SparseMatrix<double> StiffnessMatrix;

        SparseMatrix<double> Pf_fixMatrix;
        SparseMatrix<double> Pm_moveMatrix;
        VectorXd x, v, f;
        //end

        std::vector<int> fixVertsIndices;
        std::vector<int> movVertsIndices;

    public:
        SolidMesh(MatrixXi& TT, MatrixXd& TV, double youngs, double poissons);

        void initializeMesh();
        void setNewYoungsPoissons(double youngs, double poissons, int index);
        
        void setStiffnessMatrix(SparseMatrix<double>& K, VectorXd& xi);
        void setLumpedMassMatrix();
        void setForces(VectorXd& f, VectorXd& xi);

        void setConstraints(std::vector<int>& f, std::vector<int> m, SparseMatrix<double>& Pf, SparseMatrix<double>& Pm);


        inline std::vector<Tetrahedron>& getTets();
        inline MatrixXd& getColwiseV();
        inline MatrixXi& getT();
        inline VectorXd get_copy_x();
        inline VectorXd* get_px();
        inline VectorXd* get_pv();
        inline SparseMatrix<double>* get_pMass();
        inline SparseMatrix<double>* get_pStiffness();
        inline SparseMatrix<double>* get_pPf();
        MatrixXd getCurrentVerts();
    };

    SolidMesh::SolidMesh(MatrixXi& TT, MatrixXd& TV, double youngs, double poissons)
    {
        this->V = TV.transpose().eval();
        this->T = TT;
        
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        #pragma omp parallel for
        for(int i=0; i<TT.rows(); i++){
            //based on Tet indexes, get the vertices of the tet from TV
            Tetrahedron t(TT.row(i), mu, lambda);
            t.precompute(this->V);
            this->tets.push_back(t);
        }


    }
    MatrixXd SolidMesh::getCurrentVerts()
    {
        Eigen::Map<Eigen::MatrixXd> newV(this->x.data(), this->V.rows(), this->V.cols());
        return newV.transpose();
    }

    inline std::vector<Tetrahedron>& SolidMesh::getTets(){ return this->tets; }

    inline MatrixXd& SolidMesh::getColwiseV(){ return this->V; }
    
    inline MatrixXi& SolidMesh::getT(){ return this->T; }

    inline VectorXd SolidMesh::get_copy_x(){ return this->x; }

    inline VectorXd* SolidMesh::get_px(){ return &(this->x); }

    inline VectorXd* SolidMesh::get_pv(){ return &(this->v);}

    inline SparseMatrix<double>* SolidMesh::get_pMass(){ return &(this->RegMass);}

    inline SparseMatrix<double>* SolidMesh::get_pStiffness(){ return &(this->StiffnessMatrix);}

    inline SparseMatrix<double>* SolidMesh::get_pPf(){ return &(this->Pf_fixMatrix);}

    void SolidMesh::initializeMesh()
    {
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

    void SolidMesh::setLumpedMassMatrix()
    {
        int vertsNum = this->V.cols();
        VectorXd massVector;
        massVector.resize(3*vertsNum);
        massVector.setZero();
        
        #pragma omp parallel for
        for(unsigned int i=0; i<this->tets.size(); i++){
            double vol = (this->tets[i].getUndeformedVolume()/4)*1e2; //UNITS: kg/m^3
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

    void SolidMesh::setNewYoungsPoissons(double youngs, double poissons, int index)
    {
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        cout<<"NEW** Mu and Lambda from Youngs and Poissons"<<endl;

        tets[index].setMu(mu);
        tets[index].setLambda(lambda);
    }

    void SolidMesh::setStiffnessMatrix(SparseMatrix<double>& K, VectorXd& xi)
    {
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
                MatrixXd dForces = this->tets[i].computeForceDifferentials(xi, dx);
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
        this->StiffnessMatrix.setFromTriplets(triplets1.begin(), triplets1.end());
        return;
    }

    void SolidMesh::setForces(VectorXd& f, VectorXd& xi)
    {
        // //gravity
        f.setZero();
        double gravity = j_input["gravity"];
        for(unsigned int i=0; i<f.size()/3; i++){
            f(3*i+1) += this->RegMass.coeff(3*i+1, 3*i+1)*gravity;
        }

        //elastic
        #pragma omp parallel for
        for(unsigned int i=0; i<this->tets.size(); i++){
            this->tets[i].computeElasticForces(xi, f);
        }
        

        // //damping
        // f -= rayleighCoeff*forceGradient*(x_k - x_old)/h;
        return;
    }

    void SolidMesh::setConstraints(std::vector<int>& fix, std::vector<int> move, SparseMatrix<double>& Pf, SparseMatrix<double>& Pm)
    {
        //fix max y
        int axis = 1;
        double tolr = 1e-4;
        double minx = this->V.rowwise().minCoeff()(axis);
        double maxx = this->V.rowwise().maxCoeff()(axis);
        
        for(int i=0; i<this->V.cols(); ++i)
        {
            if (this->V.col(i)(axis)> maxx-tolr ) 
            {
                fix.push_back(i);
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

//############ END 

//############ INTEGRATOR 

    class Newmark
    {
    protected:
        int NEWTON_MAX = 10;
        double beta = 0.25;
        double gamma = 0.5;
        double h = 0.01;

        SolidMesh* SM;

        VectorXd x_k;
        VectorXd v_k;
        VectorXd f_o;

        SparseMatrix<double> grad_g;
        VectorXd g;
        CholmodSupernodalLLT<SparseMatrix<double>> llt_solver;

    public:
        Newmark(SolidMesh* M, double timestep);
        void step();

    };

    Newmark::Newmark(SolidMesh* M, double timestep)
    {
        this->h = timestep;
        this->SM = M;
        double dofs = (*this->SM->get_px()).rows();
        x_k.resize(dofs);
        v_k.resize(dofs);
        f_o.resize(dofs);

        SparseMatrix<double>* P = this->SM->get_pPf();
        SparseMatrix<double>* K = this->SM->get_pStiffness();
        this->g.resize((*P).cols());
        this->grad_g.resize((*P).cols(), (*P).cols());
        SparseMatrix<double> CholeskyAnalyzeBlock = (*P).transpose()*(*K)*(*P);
        llt_solver.analyzePattern(CholeskyAnalyzeBlock);
        // this->llt_solver

    }

    void Newmark::step()
    {   
        bool Nan = false;
        int iter;
        this->x_k = this->SM->get_copy_x();
        this->v_k.setZero();
        this->SM->setForces(this->f_o, this->x_k);
        SparseMatrix<double>* P = this->SM->get_pPf();
        SparseMatrix<double>* RegMass = this->SM->get_pMass();
        SparseMatrix<double>* K = this->SM->get_pStiffness();
        VectorXd *v_old = this->SM->get_pv();
        VectorXd *x_old = this->SM->get_px();
        VectorXd force = f_o;

        for(int iter=0; iter<this->NEWTON_MAX; ++iter)
        {
            this->g.setZero();
            this->grad_g.setZero();


            this->SM->setForces(force,this->x_k);
            this->SM->setStiffnessMatrix((*K), this->x_k);

            this->g = (*P).transpose()*((*RegMass) * this->x_k - (*RegMass) * (*x_old) - this->h*(*RegMass) * (*v_old) - (this->h*this->h/2)*(1-2*this->beta) * this->f_o - (this->h*this->h*this->beta)*force);
            this->grad_g = (*P).transpose()* ((*RegMass) - this->h*this->h*this->beta*(*K)) *(*P);
            
            // std::cout<<this->g.transpose()<<std::endl;
            llt_solver.compute(this->grad_g);
            if(llt_solver.info() == Eigen::NumericalIssue){
                cout<<"Possibly using a non- pos def matrix in the LLT method"<<endl;
                exit(0);
            }
            VectorXd dx = -1* llt_solver.solve(this->g);
            this->x_k += (*P)*dx;
            
            if(this->x_k != this->x_k)
            {
                Nan = true;
                break;
            }
            if(this->g.squaredNorm()/this->g.size()< 1e-3)
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
        (*v_old) = (x_k - (*x_old))/this->h;
        (*x_old) = x_k;

    }

//############ END




int main(int argc, char *argv[])
{
    std::cout<<"3d Neohookean Muscle\n";
    json j_config_parameters;
    std::ifstream i("../input/input.json");
    i >> j_input;

    double youngs_mod = j_input["youngs"];
    double poisson = j_input["poissons"];

	MatrixXd V;
	MatrixXi T;
	MatrixXi F;
	igl::readMESH(j_input["mesh_file"], V, T, F);


    SolidMesh* SM = new SolidMesh(T, V, youngs_mod, poisson);
    
    
    SM->initializeMesh();

    //Set Muscle stuff here
        Eigen::VectorXd Zc;
        Zc.resize(SM->getColwiseV().cols());
        Zc.setZero();
        double mag = j_input["fibre_mag"];
        int axis = j_input["axis"];
        Eigen::Vector3d maxs = SM->getColwiseV().rowwise().maxCoeff();
        Eigen::Vector3d mins = SM->getColwiseV().rowwise().minCoeff();
        double t = j_input["thresh"];
        Eigen::Vector3d thresh = mins + t*(maxs - mins);
        std::cout<<maxs.transpose()<<std::endl;
        std::cout<<mins.transpose()<<std::endl;
        std::cout<<thresh.transpose()<<"\naaa"<<std::endl;
        Eigen::Vector3d fibre_dir(j_input["fibre_dir"][0], j_input["fibre_dir"][1], j_input["fibre_dir"][2]);
        #pragma omp parallel for
        for(auto& tet: SM->getTets())
        {   
            int i0 = tet.getIndices()(0);
            int i1 = tet.getIndices()(1);
            int i2 = tet.getIndices()(2);
            int i3 = tet.getIndices()(3);

            if(SM->getColwiseV().col(i0)(axis)< thresh(axis))
            {
                tet.set_fibre_mag(mag);
                tet.set_fibre_dir(fibre_dir);
                Zc(i0/3) = mag;
                Zc(i1/3) = mag;
                Zc(i2/3) = mag;
                Zc(i3/3) = mag;
            }
        }
        Eigen::MatrixXd C;
        igl::jet(Zc, true, C);
        //Swap RGB colors because libigl background is blue
        C.col(2).swap(C.col(1));
    //---

    std::cout<< "STEPPING"<<std::endl;
    Newmark* nmrk = new Newmark(SM, j_input["timestep"]);
    nmrk->step();

	igl::viewer::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {   
        if(viewer.core.is_animating)
        {
            nmrk->step();
            MatrixXd newV = SM->getCurrentVerts();
            viewer.data.set_vertices(newV);
    	}
        return false;
    };

	viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;
    viewer.data.set_colors(C);

    viewer.launch();
    return EXIT_SUCCESS;

}

//TODO: SolidMesh shouldn't store V at all or even T. It should store vector x.
//Then pass x into the Tet class (indices would all multiply by 3)
//Then use the map function to convert
//Get rid of x To V function and V to x function
//Eigen::Map<Eigen::MatrixXd> dV(q1.data(), V1.cols(), V1.rows());
