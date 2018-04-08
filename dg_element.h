#ifndef TETRAHEDRON
#define TETRAHEDRON 
#include <stdio.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/LU>
#include <tetrahedron.h>

using namespace Eigen;
using namespace std;
typedef Matrix<double, 12, 1> Vector12d;

class DGElement
{
protected:
	Tetrahedron m_t1, m_t2;
	Vector3i face;
	Matrix3d m_vh;
	Vector3d m_handle;
	double nf;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	DGElement(Tetrahedron& t1, Tetrahedron& t2, VectorXi& xi, MatrixXd& V, int handle){
		m_t1 = t1;
		m_t2 = t2;
		VectorXi ind1 = t1.getIndices();
		VectorXi ind2 = t2.getIndices();

		face.setZero();
		if(!(ind1(0)==ind2(0) || ind1(0)==ind2(1) || ind1(0)==ind2(2) || ind1(0)==ind2(3)))
		{
			face(0) = ind1(1); face(1) = ind1(2); face(2) = ind1(3);
		}
		else if(!(ind1(1)==ind2(0) || ind1(1)==ind2(1) || ind1(1)==ind2(2) || ind1(1)==ind2(3)))
		{
			face(0) = ind1(0); face(1) = ind1(2); face(2) = ind1(3);
		}
		else if(!(ind1(2)==ind2(0) || ind1(2)==ind2(1) || ind1(2)==ind2(2) || ind1(2)==ind2(3)))
		{
			face(0) = ind1(0); face(1) = ind1(1); face(2) = ind1(3);
		}
		else
		{
			face(0) = ind1(0); face(1) = ind1(1); face(2) = ind1(2);
		}

		double face_area = 0.5*(xi.segment<3>(face(1)) - xi.segment<3>(face(0))).cross(xi.segment<3>(face(2)) - xi.segment<3>(face(0))).norm()
		double n = 100;
		this->nf = n*face_area*((1/t1.getUndeformedVolume()) + (1/t2.getUndeformedVolume()));
		
		m_handle = V.col(handle);
		m_vh.row(0) = V.col(face(0)) - V.col(handle);
		m_vh.row(1) = V.col(face(1)) - V.col(handle);
		m_vh.row(2) = V.col(face(2)) - V.col(handle);
	}



	double getDGEnergy(VectorXd& xi, MatrixXd& V){
		VectorXd u1(9); u1.setZero();
		VectorXd u2(9); u2.setZero();
		for(unsigned int j=0; j< face.size(); ++j){
			u1.segment<3>(3*j) = m_t1.getRSF()*m_vh.row(j) + m_handle;
			u2.segment<3>(3*j) = m_t2.getRSF()*m_vh.row(j) + m_handle;
		}
		return nf*(u1 - u2).squaredNorm();
	}

	void getForces(SparseMatrix<double>& J, VectorXd& force_rs){

		for(unsigned int j=0; j<face.size(); ++j){
			Vector3d u1 = m_t1.getRSF()*m_vh.row(j) + m_handle;
			Vector3d u2 = m_t2.getRSF()*m_vh.row(j) + m_handle;
			
			force_rs.segment<9>(9*m_t1.getRSIndex()) += J.block<3,9>().transpose()*;
			force_rs.segment<9>(9*m_t2.getRSIndex()) += ;
		}
	}

	inline Tetrahedron getT1(){ return m_t1; }
	inline Tetrahedron getT1(){ return m_t2; }
};

#endif