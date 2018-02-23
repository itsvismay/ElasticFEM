#include <stdio.h>

#include <igl/viewer/Viewer.h>
#include <igl/writeOBJ.h>
#include <igl/barycenter.h>
#include <igl/readOFF.h>
#include <igl/readMESH.h>
#include <igl/readOBJ.h>
#include <igl/slice.h>
#include <json.hpp>

using json = nlohmann::json;

using namespace Eigen;
using namespace std;

typedef Eigen::Triplet<double> Trip;
typedef Matrix<double, 12, 1> Vector12d;

//############ TETRAHEDRON 

    class Tetrahedron{

    public:
    	Tetrahedron(VectorXi k, double mu, double lambda);
        void computeElasticForces(MatrixXd& TV, VectorXd& f);
        void precompute(MatrixXd& TV);
        MatrixXd computeForceDifferentials(MatrixXd& TV, Vector12d& dx);

        double getUndeformedVolume();
        VectorXi getIndices();
        double getMu();
        double getLambda();

        void setMu(double mu);
        void setLambda(double lambda);


    protected:
    	Matrix3d DeformedShapeMatrix, ReferenceShapeMatrix, InvRefShapeMatrix;
        VectorXi verticesIndex;
        double undeformedVol, energy, currentVol;
        double mu, lambda;
    };

    Tetrahedron::Tetrahedron(VectorXi k, double mu, double lambda){
        verticesIndex = k ;
        this->mu = mu;
        this->lambda = lambda;
    }

    double Tetrahedron::getUndeformedVolume(){  
       return this->undeformedVol;
    } 
    
    VectorXi Tetrahedron::getIndices(){  
       return this->verticesIndex;
    } 

    double Tetrahedron::getMu(){  
       return this->mu;
    } 

    double Tetrahedron::getLambda(){  
       return this->lambda;
    } 

    void Tetrahedron::setMu(double mu){  
       this->mu =mu;
    } 

    void Tetrahedron::setLambda(double lambda){  
       this->lambda=lambda;
    } 
        
    void Tetrahedron::precompute(MatrixXd& TV){
    	Matrix3d Dm;
        for(int i=0; i<3; i++){
            Dm.col(i) = TV.col(verticesIndex(i)) - TV.col(verticesIndex(3));
        }
    	
        this->ReferenceShapeMatrix = Dm;
        this->InvRefShapeMatrix = Dm.inverse();
        this->undeformedVol = (1.0/6)*fabs(Dm.determinant());
    }

    void Tetrahedron::computeElasticForces(MatrixXd &TV, VectorXd& f){

        Matrix3d Ds;
        for(int i=0; i<3; i++){
            Ds.col(i) = TV.col(verticesIndex(i)) - TV.col(verticesIndex(3));
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

        
        if(F.determinant()<0){
            this->energy = 1e40;
            cout<<"ERROR: F determinant is 0"<<endl;
            cout<<"Decrease timestep maybe - instantaneous force is too much with this timestep"<<endl;
            exit(0);
        }
        if(this->energy != this->energy){
            //NANS
            cout<<"ENERGY nans"<<endl;
            exit(0);
        }

        Matrix3d H = -1*this->undeformedVol*P*((this->InvRefShapeMatrix).transpose());

        f.segment<3>(3*verticesIndex(0)) += H.col(0);
        f.segment<3>(3*verticesIndex(1)) += H.col(1);
        f.segment<3>(3*verticesIndex(2)) += H.col(2);
        f.segment<3>(3*verticesIndex(3)) += -1*H.col(0) - H.col(1) - H.col(2);
    }

    MatrixXd Tetrahedron::computeForceDifferentials(MatrixXd& TV, Vector12d& dx){

        Matrix3d Ds;
        for(int i=0; i<3; i++){
            Ds.col(i) = TV.col(verticesIndex(i)) - TV.col(verticesIndex(3));
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

    class SolidMesh{

    protected:
        MatrixXd V;
        MatrixXi T;

    	double medianMass;
        std::vector<Tetrahedron> tets;
        
        //Used in the sim
        SparseMatrix<double> InvMass;
        SparseMatrix<double> RegMass;
        SparseMatrix<double> StiffnessMatrix;

        SparseMatrix<int> Pf_fixMatrix;
        SparseMatrix<int> Pm_moveMatrix;
        VectorXd x, v, f;
        //end

        std::vector<int> fixVertsIndices;
        std::vector<int> movVertsIndices;

    public:
        SolidMesh(MatrixXi& TT, MatrixXd& TV, double youngs, double poissons);

        void initializeMesh();
        void setNewYoungsPoissons(double youngs, double poissons, int index);
        
        void setStiffnessMatrix(SparseMatrix<double>& K);
        void setLumpedMassMatrix();
        void setForces(VectorXd& f);

        void xToV(VectorXd& x);
        void setConstraints(std::vector<int>& f, std::vector<int> m, SparseMatrix<int>& Pf, SparseMatrix<int>& Pm);


        std::vector<Tetrahedron> getTets();
        VectorXd* getx();
        VectorXd* getv();
        SparseMatrix<double>* getMass();
        SparseMatrix<double>* getStiffness();
        SparseMatrix<int>* getPf();
    };

    SolidMesh::SolidMesh(MatrixXi& TT, MatrixXd& TV, double youngs, double poissons){
        this->V = TV.transpose().eval();
        this->T = TT;
        
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        cout<<"Setting Mu and Lambda from Youngs and Poissons"<<endl;
        cout<<"SolidMesh init Youngs, poissons ="<<youngs<<", "<<poissons<<endl;
        cout<<"SolidMesh init Mu, Lambda ="<<mu<<", "<<lambda<<endl<<endl;
        for(int i=0; i<TT.rows(); i++){
            //based on Tet indexes, get the vertices of the tet from TV
            Tetrahedron t(TT.row(i), mu, lambda);
            t.precompute(this->V);
            this->tets.push_back(t);
        }


    }

    std::vector<Tetrahedron> SolidMesh::getTets(){
        return this->tets;
    }

    VectorXd* SolidMesh::getx(){
        return &(this->x);
    }

    VectorXd* SolidMesh::getv(){
        return &(this->v);
    }

    SparseMatrix<double>* SolidMesh::getMass(){
        return &(this->RegMass);
    }

    SparseMatrix<double>* SolidMesh::getStiffness(){
        return &(this->StiffnessMatrix);
    }

    SparseMatrix<int>* SolidMesh::getPf(){
        return &(this->Pf_fixMatrix);
    }

    void SolidMesh::initializeMesh(){
        int vertsNum = this->V.cols();
        InvMass.resize(3*vertsNum, 3*vertsNum); RegMass.resize(3*vertsNum, 3*vertsNum);
        InvMass.setZero();RegMass.setZero();

        StiffnessMatrix.resize(3*vertsNum, 3*vertsNum);

        x.resize(3*vertsNum);v.resize(3*vertsNum);f.resize(3*vertsNum);
        x.setZero();v.setZero();f.setZero();

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
        this->setStiffnessMatrix(this->StiffnessMatrix);
        this->setConstraints(this->fixVertsIndices, this->movVertsIndices, this->Pf_fixMatrix, this->Pm_moveMatrix);
    }

    void SolidMesh::setLumpedMassMatrix(){
        int vertsNum = this->V.cols();
        VectorXd massVector;
        massVector.resize(3*vertsNum);
        massVector.setZero();

        

        for(unsigned int i=0; i<this->tets.size(); i++){
            double vol = (this->tets[i].getUndeformedVolume()/4); //UNITS: kg/m^3
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

    void SolidMesh::setNewYoungsPoissons(double youngs, double poissons, int index){
        double mu = youngs/(2+ 2*poissons);
        double lambda = youngs*poissons/((1+poissons)*(1-2*poissons));
        cout<<"NEW** Mu and Lambda from Youngs and Poissons"<<endl;

        tets[index].setMu(mu);
        tets[index].setLambda(lambda);
    }

    void SolidMesh::setStiffnessMatrix(SparseMatrix<double>& K){
        this->StiffnessMatrix.setZero();

        vector<Trip> triplets1;
        triplets1.reserve(12*12*this->tets.size());
        for(unsigned int i=0; i<this->tets.size(); i++){
            //Get P(dxn), dx = [1,0, 0...], then [0,1,0,....], and so on... for all 4 vert's x, y, z
            //P is the compute Force Differentials blackbox fxn

            Vector12d dx(12);
            dx.setZero();
            Vector4i indices = this->tets[i].getIndices();
            int kj;
            for(unsigned int j=0; j<12; j++){
                dx(j) = 1;
                MatrixXd dForces = this->tets[i].computeForceDifferentials(this->V, dx);
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

    void SolidMesh::setForces(VectorXd& f){
        // //gravity
        f.setZero();

        // for(unsigned int i=0; i<f.size()/3; i++){
        //     f(3*i+1) += massVector(3*i+2)*gravity;
        // }

        //elastic
        for(unsigned int i=0; i<this->tets.size(); i++){
            this->tets[i].computeElasticForces(this->V, f);
        }
        

        // //damping
        // f -= rayleighCoeff*forceGradient*(x_k - x_old)/h;
        return;
    }

    void SolidMesh::setConstraints(std::vector<int>& fix, std::vector<int> move, SparseMatrix<int>& Pf, SparseMatrix<int>& Pm){
        //fix min x
        int axis = 0;
        double tolr = 1e-5;
        double minx = this->V.rowwise().minCoeff()(0);
        for(int i=0; i<this->V.cols(); ++i)
        {
            if (this->V.col(i)(axis)< minx+tolr ) 
            {
                fix.push_back(i);

                std::cout<<i<<std::endl;
            }
        }



        //TODO: MAKE SURE fix vector is sorted
        Pf.resize(3*this->V.cols(), 3*this->V.cols() - 3*fix.size());
        Pf.setZero();

        int c =0;
        int j =0;
        for(int i=0; i<this->V.cols(); ++i)
        {
            if(i != fix[c])
            {
                std::cout<<"not "<<i<<std::endl;
                Pf.coeffRef(3*i, 3*j) = 1.0;
                Pf.coeffRef(3*i+1, 3*j+1) =1.0;
                Pf.coeffRef(3*i+2, 3*j+2) =1.0;
                j+=1;
            }else
            {
                std::cout<<"Fixed "<<fix[c]<<std::endl;
                c+=1;
            }
        }

        this->Pf_fixMatrix = Pf;
    }

    //TODO: make this redundant
    void SolidMesh::xToV(VectorXd& q){
        this->V.setZero();
        for(unsigned int i=0; i<this->tets.size(); ++i)
        {
            Vector4i indices = this->tets[i].getIndices();
            this->V.col(indices(0)) = Vector3d(q(3*indices(0)), q(3*indices(0)+1),q(3*indices(0) +2));
            this->V.col(indices(1)) = Vector3d(q(3*indices(1)), q(3*indices(1)+1),q(3*indices(1) +2));
            this->V.col(indices(2)) = Vector3d(q(3*indices(2)), q(3*indices(2)+1),q(3*indices(2) +2));
            this->V.col(indices(3)) = Vector3d(q(3*indices(3)), q(3*indices(3)+1),q(3*indices(3) +2));
        }
        return;
    }

//############ END 

//############ INTEGRATOR 

    class Newmark{
    protected:
        int NEWTON_MAX = 100;
        SolidMesh* SM;

        VectorXd x_k;
        VectorXd v_k;
        VectorXd f_o;

        SparseMatrix<double> grad_g;
        VectorXd g;

    public:
        Newmark(SolidMesh* M);
        void step();

    };

    Newmark::Newmark(SolidMesh* M)
    {
        this->SM = M;
        double dofs = (*this->SM->getx()).rows();
        x_k.resize(dofs);
        v_k.resize(dofs);
        f_o.resize(dofs);


    }
    void Newmark::step()
    {
        x_k.setZero();
        v_k.setZero();
        this->SM->setForces(f_o);

        for(int i=0; i<this->NEWTON_MAX; ++i)
        {

            this->grad_g.setZero();
            this->g.setZero();

            
        }

    }

//############ END




int main(int argc, char *argv[])
{
	if(argc<2){
		std::cout<<"Oh no!"<<std::endl;
	}else{
		std::cout<<"No config file specified. Using defaults."<<std::endl;
		json j_config_parameters;
		//std::ifstream  config_json_file(std::string(argv[1]));
		//config_json_file >> j_config_parameters;
	}

    double youngs_mod = 1e6;
    double poisson = 0.45;
	MatrixXd V;
	MatrixXi T;
	MatrixXi F;
	igl::readMESH("/home/vismay/Scrapts/Beam.1.mesh", V, T, F);


    SolidMesh* SM = new SolidMesh(T, V, youngs_mod, poisson);
    
    
    SM->initializeMesh();

    std::cout<< "STEPPING"<<std::endl;
    Newmark* nmrk = new Newmark(SM);
    nmrk->step();

	igl::viewer::Viewer viewer;
    viewer.callback_pre_draw = [&](igl::viewer::Viewer & viewer)
    {
    	return false;
    };

	std::cout<<V<<std::endl;
	viewer.data.set_mesh(V,F);
    viewer.core.show_lines = true;
    viewer.core.invert_normals = true;
    viewer.core.is_animating = false;
    viewer.data.face_based = true;

    viewer.launch();
    return EXIT_SUCCESS;

}

//TODO: SolidMesh shouldn't store V at all or even T. It should store vector x.
//Then pass x into the Tet class (indices would all multiply by 3)
//Then use the map function to convert
//Get rid of x To V function and V to x function
//Eigen::Map<Eigen::MatrixXd> dV(q1.data(), V1.cols(), V1.rows());
