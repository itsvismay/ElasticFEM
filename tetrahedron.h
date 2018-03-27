#ifndef TETRAHEDRON
#define TETRAHEDRON 

#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;
using namespace std;
typedef Matrix<double, 12, 1> Vector12d;
inline double Cos(double a){ return std::cos(a);}
inline double Sin(double a){ return std::cos(a);}

class Tetrahedron
{
protected:
    Matrix3d m_DeformedShapeMatrix, m_ReferenceShapeMatrix, m_InvRefShapeMatrix;
    Matrix3d m_w, m_S, m_U;
    VectorXi m_verticesIndex;
    double m_undeformedVol, m_energy, m_currentVol;
    double m_mu, m_lambda, m_fibre_mag;
    Vector3d m_fibre_dir, m_tet_centroid, m_x0;
    int m_rsIndex;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	Tetrahedron(VectorXi k, double mu, double lambda, int t){
        m_rsIndex = t;
        m_verticesIndex = k ;
        m_mu = mu;
        m_lambda = lambda;
        m_fibre_mag = 0;
        m_fibre_dir.setZero();

        m_w.setZero();
        m_S.setZero();
        m_U.setIdentity();
    }


    Matrix3d getRSF(){
        return (m_w.exp())*(m_U*(m_S+Matrix3d::Identity())*m_U.transpose());
    }

    void setRSU(VectorXd& xr){
        Vector3d new_fibre_dir = getRSF()*m_fibre_dir;
        m_U = Quaterniond().setFromTwoVectors(new_fibre_dir, Vector3d::UnitX()).toRotationMatrix();
        
        double aX = xr(9*m_rsIndex + 0), aY = xr(9*m_rsIndex + 1), aZ = xr(9*m_rsIndex + 2), 
            as1 = xr(9*m_rsIndex + 3), as2 = xr(9*m_rsIndex + 4), as3 = xr(9*m_rsIndex + 5), 
            as4 = xr(9*m_rsIndex + 6), as5 = xr(9*m_rsIndex + 7), as6 = xr(9*m_rsIndex + 8);

        
        m_w =  AngleAxisd(aX, Vector3d::UnitZ())
                * AngleAxisd(aY,  Vector3d::UnitY())
                * AngleAxisd(aZ, Vector3d::UnitX());

        m_S(0,0) =  as1;
        m_S(1,1) =  as2;
        m_S(2,2) =  as3;

        m_S(0,1) =  as4;
        m_S(1,0) =  as4;

        m_S(0,2) =  as5;
        m_S(2,0) =  as5;

        m_S(1,2) =  as6;
        m_S(2,1) =  as6;

    }

    double getInternalEnergy(VectorXd& xi){
        Matrix3d Ds;
        for(int i=0; i<3; i++)
        {
            Ds.col(i) = xi.segment<3>(3*m_verticesIndex(i)) - xi.segment<3>(3*m_verticesIndex(3));
        }

        Matrix3d F = Ds*m_InvRefShapeMatrix;

        //Neo
        double J = F.determinant();
        double I1 = (F.transpose()*F).trace();
        double powj = pow(J, -2.0/3.0);
        double I1bar = powj*I1;
        Vector3d Fv = F*m_fibre_dir;

        double neo_energy = m_undeformedVol*(m_mu/2.0 * (I1bar - 3) + m_lambda/2.0 * (J-1.0) * (J-1.0));
        double muscle_energy = 0.5*m_fibre_mag*(m_fibre_dir.transpose()*F.transpose()*F*m_fibre_dir - 1);

        return neo_energy + muscle_energy;
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

    inline int getRSIndex(){
        return m_rsIndex;
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

    MatrixXd getdxdxrJacobian(VectorXd& xr, Vector3d v){
        double aX = xr(9*m_rsIndex + 0), aY = xr(9*m_rsIndex + 1), aZ = xr(9*m_rsIndex + 2), 
            as1 = xr(9*m_rsIndex + 3), as2 = xr(9*m_rsIndex + 4), as3 = xr(9*m_rsIndex + 5), 
            as4 = xr(9*m_rsIndex + 6), as5 = xr(9*m_rsIndex + 7), as6 = xr(9*m_rsIndex + 8);

        double v1 = v(0), v2 = v(1), v3 = v(2);
        double u1 = m_U(0,0), u2 = m_U(0,1), u3 = m_U(0,2), 
                u4 = m_U(1,0), u5 = m_U(1,1), u6 = m_U(1,2),
                u7 = m_U(2,0), u8 = m_U(2,1), u9 = m_U(2,2);

        MatrixXd m_Jacobian(3, 9);
        
        m_Jacobian << v1*(u3*(as5*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as6*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as3*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u1*(as1*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as4*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as5*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u2*(as4*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as2*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as6*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX)))) + 
            v2*(u6*(as5*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as6*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as3*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u4*(as1*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as4*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as5*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u5*(as4*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as2*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as6*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX)))) + 
            v3*(u9*(as5*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as6*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as3*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u7*(as1*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as4*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as5*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX))) + 
               u8*(as4*(-(u4*Cos(aX)*Cos(aY)) - u1*Cos(aY)*Sin(aX)) + as2*(-(u5*Cos(aX)*Cos(aY)) - u2*Cos(aY)*Sin(aX)) + as6*(-(u6*Cos(aX)*Cos(aY)) - u3*Cos(aY)*Sin(aX)))),
           v1*(u3*(as5*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as6*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as3*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u1*(as1*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as4*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as5*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u2*(as4*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as2*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as6*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY)))) + 
            v2*(u6*(as5*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as6*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as3*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u4*(as1*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as4*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as5*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u5*(as4*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as2*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as6*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY)))) + 
            v3*(u9*(as5*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as6*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as3*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u7*(as1*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as4*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as5*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY))) + 
               u8*(as4*(u7*Cos(aY) - u1*Cos(aX)*Sin(aY) + u4*Sin(aX)*Sin(aY)) + as2*(u8*Cos(aY) - u2*Cos(aX)*Sin(aY) + u5*Sin(aX)*Sin(aY)) + as6*(u9*Cos(aY) - u3*Cos(aX)*Sin(aY) + u6*Sin(aX)*Sin(aY)))),0,
           u1*v1*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u4*v2*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u7*v3*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)),
           u2*v1*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)) + u5*v2*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)) + u8*v3*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)),
           u3*v1*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY)) + u6*v2*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY)) + u9*v3*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY)),
           v1*(u2*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u1*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY))) + 
            v2*(u5*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u4*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY))) + 
            v3*(u8*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u7*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY))),
           v1*(u3*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u1*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))) + 
            v2*(u6*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u4*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))) + 
            v3*(u9*(u1*Cos(aX)*Cos(aY) - u4*Cos(aY)*Sin(aX) + u7*Sin(aY)) + u7*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))),
           v1*(u3*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)) + u2*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))) + 
            v2*(u6*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)) + u5*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))) + 
            v3*(u9*(u2*Cos(aX)*Cos(aY) - u5*Cos(aY)*Sin(aX) + u8*Sin(aY)) + u8*(u3*Cos(aX)*Cos(aY) - u6*Cos(aY)*Sin(aX) + u9*Sin(aY))),

            v1*(u3*(as5*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u1*(as1*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u2*(as4*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))) + 
            v2*(u6*(as5*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u4*(as1*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u5*(as4*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))) + 
            v3*(u9*(as5*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u7*(as1*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u8*(as4*(u4*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(u5*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(u6*(-(Cos(aZ)*Sin(aX)) - Cos(aX)*Sin(aY)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))),
           v1*(u3*(as5*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as6*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as3*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u1*(as1*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as4*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as5*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u2*(as4*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as2*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as6*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ)))) + 
            v2*(u6*(as5*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as6*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as3*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u4*(as1*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as4*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as5*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u5*(as4*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as2*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as6*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ)))) + 
            v3*(u9*(as5*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as6*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as3*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u7*(as1*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as4*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as5*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ))) + 
               u8*(as4*(u1*Cos(aX)*Cos(aY)*Sin(aZ) - u4*Cos(aY)*Sin(aX)*Sin(aZ) + u7*Sin(aY)*Sin(aZ)) + as2*(u2*Cos(aX)*Cos(aY)*Sin(aZ) - u5*Cos(aY)*Sin(aX)*Sin(aZ) + u8*Sin(aY)*Sin(aZ)) + 
                  as6*(u3*Cos(aX)*Cos(aY)*Sin(aZ) - u6*Cos(aY)*Sin(aX)*Sin(aZ) + u9*Sin(aY)*Sin(aZ)))),
           v1*(u3*(as5*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u1*(as1*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u2*(as4*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))) + 
            v2*(u6*(as5*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u4*(as1*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u5*(as4*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))) + 
            v3*(u9*(as5*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u7*(as1*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ)))) + 
               u8*(as4*(-(u7*Cos(aY)*Cos(aZ)) + u4*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u1*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Cos(aZ)) + u5*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u2*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Cos(aZ)) + u6*(-(Cos(aZ)*Sin(aX)*Sin(aY)) - Cos(aX)*Sin(aZ)) + u3*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))),
           u1*v1*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u4*v2*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u7*v3*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))),
           u2*v1*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u5*v2*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u8*v3*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))),
           u3*v1*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u6*v2*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
            u9*v3*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))),
           v1*(u2*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u1*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v2*(u5*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u4*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v3*(u8*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u7*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))),
           v1*(u3*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u1*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v2*(u6*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u4*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v3*(u9*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u7*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))),
           v1*(u3*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u2*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v2*(u6*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u5*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
            v3*(u9*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
               u8*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))),

            v1*(u3*(as5*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as3*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u1*(as1*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as5*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u2*(as4*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as6*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
               ) + v2*(u6*(as5*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as3*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u4*(as1*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as5*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u5*(as4*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as6*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
               ) + v3*(u9*(as5*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as6*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as3*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u7*(as1*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as4*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as5*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
                 + u8*(as4*(u1*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + 
                  as2*(u2*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))) + as6*(u3*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ)*Sin(aY) - Sin(aX)*Sin(aZ))))
               ),v1*(u3*(as5*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as6*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as3*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u1*(as1*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as4*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as5*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u2*(as4*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as2*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as6*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY)))) + 
            v2*(u6*(as5*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as6*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as3*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u4*(as1*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as4*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as5*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u5*(as4*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as2*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as6*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY)))) + 
            v3*(u9*(as5*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as6*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as3*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u7*(as1*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as4*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as5*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY))) + 
               u8*(as4*(-(u1*Cos(aX)*Cos(aY)*Cos(aZ)) + u4*Cos(aY)*Cos(aZ)*Sin(aX) - u7*Cos(aZ)*Sin(aY)) + as2*(-(u2*Cos(aX)*Cos(aY)*Cos(aZ)) + u5*Cos(aY)*Cos(aZ)*Sin(aX) - u8*Cos(aZ)*Sin(aY)) + 
                  as6*(-(u3*Cos(aX)*Cos(aY)*Cos(aZ)) + u6*Cos(aY)*Cos(aZ)*Sin(aX) - u9*Cos(aZ)*Sin(aY)))),
           v1*(u3*(as5*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u1*(as1*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u2*(as4*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))) + 
            v2*(u6*(as5*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u4*(as1*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u5*(as4*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))) + 
            v3*(u9*(as5*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as3*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u7*(as1*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as4*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as5*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ)))) + 
               u8*(as4*(-(u7*Cos(aY)*Sin(aZ)) + u1*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u4*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as2*(-(u8*Cos(aY)*Sin(aZ)) + u2*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u5*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))) + 
                  as6*(-(u9*Cos(aY)*Sin(aZ)) + u3*(Cos(aZ)*Sin(aX) + Cos(aX)*Sin(aY)*Sin(aZ)) + u6*(Cos(aX)*Cos(aZ) - Sin(aX)*Sin(aY)*Sin(aZ))))),
           u1*v1*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u4*v2*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u7*v3*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))),
           u2*v1*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u5*v2*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u8*v3*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))),
           u3*v1*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u6*v2*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
            u9*v3*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))),
           v1*(u2*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u1*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v2*(u5*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u4*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v3*(u8*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u7*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))),
           v1*(u3*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u1*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v2*(u6*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u4*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v3*(u9*(u7*Cos(aY)*Cos(aZ) + u4*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u1*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u7*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))),
           v1*(u3*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u2*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v2*(u6*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u5*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ)))) + 
            v3*(u9*(u8*Cos(aY)*Cos(aZ) + u5*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u2*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))) + 
               u8*(u9*Cos(aY)*Cos(aZ) + u6*(Cos(aZ)*Sin(aX)*Sin(aY) + Cos(aX)*Sin(aZ)) + u3*(-(Cos(aX)*Cos(aZ)*Sin(aY)) + Sin(aX)*Sin(aZ))));
        return m_Jacobian;
    }

};

#endif