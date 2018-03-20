#ifndef TETRAHEDRON
#define TETRAHEDRON 

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>

using namespace Eigen;
using namespace std;
typedef Matrix<double, 12, 1> Vector12d;
class Tetrahedron
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Tetrahedron(VectorXi k, double mu, double lambda){
        m_verticesIndex = k ;
        m_mu = mu;
        m_lambda = lambda;
        m_fibre_mag = 0;
        m_fibre_dir.setZero();
    }
    void computeElasticForces(VectorXd& xi, VectorXd& f){
        Matrix3d Ds;
        for(int i=0; i<3; i++)
        {
            Ds.col(i) = xi.segment<3>(3*m_verticesIndex(i)) - xi.segment<3>(3*m_verticesIndex(3));
        }

        Matrix3d F = Ds*m_InvRefShapeMatrix;

        Matrix3d P;

        m_currentVol = (1.0/6)*fabs(Ds.determinant());
        
        //Neo
        double J = F.determinant();
        double I1 = (F.transpose()*F).trace();
        double powj = pow(J, -2.0/3.0);
        double I1bar = powj*I1;
        Vector3d Fv = F*m_fibre_dir;

        P = m_mu*(powj * F) +
        (- m_mu/3.0 * I1 * powj + m_lambda*(J-1.0)*J)*F.inverse().transpose();

        m_energy = m_undeformedVol*(m_mu/2.0 * (I1bar - 3) + m_lambda/2.0 * (J-1.0) * (J-1.0));

        
        if(F.determinant()<0)
        {
            m_energy = 1e40;
            cout<<"ERROR: F determinant is 0"<<endl;
            cout<<"Decrease timestep maybe - instantaneous force is too much with this timestep"<<endl;
            exit(0);
        }
        if(m_energy != m_energy)
        {
            //NANS
            cout<<"ENERGY nans"<<endl;
            exit(0);
        }

        Matrix3d H = -1*m_undeformedVol*P*((m_InvRefShapeMatrix).transpose());


        //##Muscle force, do the proper math later
        double v1 =m_fibre_dir[0];
        double v2 =m_fibre_dir[1];
        double v3 =m_fibre_dir[2];
        double a = m_fibre_mag;
        
        double o = m_InvRefShapeMatrix(0,0);
        double p = m_InvRefShapeMatrix(0,1);
        double q = m_InvRefShapeMatrix(0,2);
        double t = m_InvRefShapeMatrix(1,0);
        double r = m_InvRefShapeMatrix(1,1);
        double s = m_InvRefShapeMatrix(1,2);
        double u = m_InvRefShapeMatrix(2,0);
        double w = m_InvRefShapeMatrix(2,1);
        double e = m_InvRefShapeMatrix(2,2);

        double nx1_nx4 = Ds(0,0);
        double nx2_nx4 = Ds(0,1);
        double nx3_nx4 = Ds(0,2);
        double ny1_ny4 = Ds(1,0);
        double ny2_ny4 = Ds(1,1);
        double ny3_ny4 = Ds(1,2);
        double nz1_nz4 = Ds(2,0);
        double nz2_nz4 = Ds(2,1);
        double nz3_nz4 = Ds(2,2);

        //Symbolic derivative for f_muscle = dEnergy_muscle / dx, because I can't do tensor calc
        VectorXd force_muscle(12);
        force_muscle<<
        (-1 *a *(o *v1 + p *v2 + q *v3) *this->m_undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1* a* (o *v1 + p *v2 + q *v3) *this->m_undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* (o *v1 + p *v2 + q *v3) *this->m_undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1* a* (t* v1 + r* v2 + s* v3)*this->m_undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1* a* (t* v1 + r* v2 + s* v3) *this->m_undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* (t* v1 + r* v2 + s* v3) *this->m_undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1 *a * (u* v1 + e* v3 + v2* w) *this->m_undeformedVol*
            (((nx1_nx4)* o + (nx2_nx4) *t + (nx3_nx4) *u)* v1 
                + (e *(nx3_nx4) + (nx1_nx4) *q + (nx2_nx4)* s)* v3 
                + v2* ((nx1_nx4) *p + (nx2_nx4) *r + (nx3_nx4)* w))),
        
        (-1 *a * (u* v1 + e* v3 + v2* w) *this->m_undeformedVol*
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u) *v1 
                + (e* (ny3_ny4)* + (ny1_ny4)* q + (ny2_ny4)* s)* v3 
                + v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1 *a * (u* v1 + e* v3 + v2* w) *this->m_undeformedVol*
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u) *v1 
                + (e* (nz3_nz4)* + (nz1_nz4)* q + (nz2_nz4)* s)* v3 
                + v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w))),

        (-1* a* this->m_undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((nx1_nx4)* o + (nx2_nx4)* t + (nx3_nx4)* u)* v1 + 
                (e *(nx3_nx4) + (nx1_nx4)* q + (nx2_nx4)* s)* v3 + 
                v2 *((nx1_nx4)* p + (nx2_nx4)* r + (nx3_nx4)* w))),

        (-1* a* this->m_undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((ny1_ny4)* o + (ny2_ny4)* t + (ny3_ny4)* u)* v1 + 
                (e *(ny3_ny4) + (ny1_ny4)* q + (ny2_ny4)* s)* v3 + 
                v2 *((ny1_ny4)* p + (ny2_ny4)* r + (ny3_ny4)* w))),

        (-1* a* this->m_undeformedVol* ((-o - t - u) *v1 + (-e - q - s) *v3 + v2* (-p - r - w)) *
            (((nz1_nz4)* o + (nz2_nz4)* t + (nz3_nz4)* u)* v1 + 
                (e *(nz3_nz4) + (nz1_nz4)* q + (nz2_nz4)* s)* v3 + 
                v2 *((nz1_nz4)* p + (nz2_nz4)* r + (nz3_nz4)* w)));

        //--------


        f.segment<3>(3*m_verticesIndex(0)) += H.col(0) + force_muscle.segment<3>(0);
        f.segment<3>(3*m_verticesIndex(1)) += H.col(1) + force_muscle.segment<3>(3);
        f.segment<3>(3*m_verticesIndex(2)) += H.col(2) + force_muscle.segment<3>(6);
        f.segment<3>(3*m_verticesIndex(3)) += -1*H.col(0) - H.col(1) - H.col(2) + force_muscle.segment<3>(9);
    }
    void precompute(MatrixXd& TV){
        Matrix3d Dm;
        for(int i=0; i<3; i++)
        {
            Dm.col(i) = TV.col(m_verticesIndex(i)) - TV.col(m_verticesIndex(3));
        }
        
        m_ReferenceShapeMatrix = Dm;
        m_InvRefShapeMatrix = Dm.inverse();
        m_undeformedVol = (1.0/6)*fabs(Dm.determinant());
        m_tet_centroid = (1.0/4)*(TV.col(m_verticesIndex(0)) + TV.col(m_verticesIndex(1)) + TV.col(m_verticesIndex(2)) + TV.col(m_verticesIndex(3)));
    }
    
    MatrixXd computeForceDifferentials(VectorXd& xi, Vector12d& dx){
        Matrix3d Ds;
        for(int i=0; i<3; i++)
        {
            Ds.col(i) = xi.segment<3>(3*m_verticesIndex(i)) - xi.segment<3>(3*m_verticesIndex(3));
        }

        Matrix3d dDs;
        dDs <<  (dx(0) - dx(9)),   (dx(3) - dx(9)), (dx(6) - dx(9)),
                (dx(1) - dx(10)), (dx(4) - dx(10)), (dx(7) - dx(10)),
                (dx(2) - dx(11)), (dx(5) - dx(11)), (dx(8) - dx(11));

        Matrix3d F = Ds*m_InvRefShapeMatrix;
        Matrix3d dF = dDs*m_InvRefShapeMatrix;

        Matrix3d dP;
        
        //Neohookean
        double detF = F.determinant();
        double logdetF = log(detF);
        Matrix3d FInvTransp = (F.inverse()).transpose();
        dP = m_mu*dF + (m_mu - m_lambda*logdetF)*(FInvTransp)*dF.transpose()*(FInvTransp) + m_lambda*(F.inverse()*dF).trace()*(FInvTransp);
        

        Matrix3d dH = -1*m_undeformedVol*dP*((m_InvRefShapeMatrix).transpose());


        MatrixXd dForces(3,4);
        dForces.col(0) = dH.col(0);
        dForces.col(1) = dH.col(1);
        dForces.col(2) = dH.col(2);
        dForces.col(3) = -1*dH.col(0) - dH.col(1) - dH.col(2);

        return dForces;
    }

    inline double getUndeformedVolume(){
        return m_undeformedVol;
    }
    inline VectorXi& getIndices(){
        return m_verticesIndex;
    }
    inline double getMu(){
        return m_mu;
    }
    inline double getLambda(){
        return m_lambda;
    }
    inline Vector3d& getCentroid(){
        return m_tet_centroid;
    }

    void setMu(double mu){
        m_mu =mu;
    }
    void setLambda(double lambda){
        m_lambda=lambda;
    }
    void set_fibre_mag(double mag){
        m_fibre_mag = mag;
    }
    void set_fibre_dir(Vector3d& v){
        m_fibre_dir = v;
    }


protected:
	Matrix3d m_DeformedShapeMatrix, m_ReferenceShapeMatrix, m_InvRefShapeMatrix;
    VectorXi m_verticesIndex;
    double m_undeformedVol, m_energy, m_currentVol;
    double m_mu, m_lambda, m_fibre_mag;
    Vector3d m_fibre_dir, m_tet_centroid;
};

#endif