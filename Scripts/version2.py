import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})

#helpers
def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y):
	V = []
	for i in range(0,x):
		for j in range(0,y):
			V.append([i, j])
	return V, Delaunay(V).simplices

def torus_mesh(r1, r2):
	V = []
	T = []
	for theta in range(0, 12):
		angle = theta*np.pi/6.0
		V.append([r1*np.cos(angle), r1*np.sin(angle)])
		V.append([r2*np.cos(angle), r2*np.sin(angle)])
	for e in Delaunay(V).simplices:
		if get_area(V[e[0]], V[e[1]], V[e[2]])<5:
			T.append(list(e))
	return V, T	

class Mesh:
	#class vars

	def __init__(self, VT):
		#object vars
		self.V = VT[0]
		self.T = VT[1]
		self.P = None
		self.A = None
		self.x0 = np.ravel(self.V)
		self.g = np.ravel(self.V)

		self.q = np.zeros(len(self.T)*(1+2)) #theta, sx, sy

		#set initial strains
		for i in range(len(self.T)):
			self.q[3*i + 1] = 0.9
			self.q[3*i + 2] = 1

		#set initial rots
		for i in range(len(self.T)):
			self.q[3*i] = np.pi
			break
		print(self.q)

	def createBlockingMatrix(self, to_fix=[]):
		b = np.kron(np.delete(np.eye(len(self.V)), to_fix, axis =1), np.eye(2))
		return b

	def getP(self):
		if(self.P is None):
			P = np.zeros((6*len(self.T), 6*len(self.T)))
			sub_P = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [-1, 0, 1]]), np.eye(2))
			for i in range(len(self.T)):
				P[6*i:6*i+6, 6*i:6*i+6] = sub_P
			self.P = P

		return self.P

	def getA(self):
		if(self.A is None):
			A = np.zeros((6*len(self.T), 2*len(self.V)))
			for i in range(len(self.T)):
				e = self.T[i]
				for j in range(len(e)):
					v = e[j]
					A[6*i+2*j, 2*v] = 1
					A[6*i+2*j+1, 2*v+1] = 1
			self.A = A
		return self.A

	def getU(self, ind):
		if ind%2== 1:
			alpha = np.pi/3
		else:
			alpha = np.pi/4

		cU, sU = np.cos(alpha), np.sin(alpha)
		U = np.array(((cU,-sU), (sU, cU)))
		return U

	def getR(self, ind):
		theta = self.q[3*ind]
		c, s = np.cos(theta), np.sin(theta)
		R = np.array(((c,-s), (s, c)))
		# print("R",scipy.linalg.logm(R))
		return R

	def getS(self, ind):
		S = np.array([[self.q[3*ind+1], 0], [0, self.q[3*ind+2]]])  
		return S

	def getF(self, ind):
		U = self.getU(ind)
		S = self.getS(ind)
		R = self.getR(ind)
		F =  np.matmul(R, np.matmul(U, np.matmul(S, U.transpose())))
		return F

	def getGlobalF(self):
		print("q", self.q)
		GF = np.zeros((6*len(self.T), 6*len(self.T)))
		GR = np.zeros((6*len(self.T), 6*len(self.T)))
		GS = np.zeros((6*len(self.T), 6*len(self.T)))
		GU = np.zeros((6*len(self.T), 6*len(self.T)))
		for i in range(len(self.T)):
			F = self.getF(i)
			r = self.getR(i)
			s = self.getS(i)
			u = self.getU(i)
			F_e = np.kron(np.eye(3), F)
			r_e = np.kron(np.eye(3), r)
			s_e = np.kron(np.eye(3), s)
			u_e = np.kron(np.eye(3), u)
			GF[6*i:6*i+6, 6*i:6*i+6] = F_e
			GR[6*i:6*i+6, 6*i:6*i+6] = r_e
			GS[6*i:6*i+6, 6*i:6*i+6] = s_e
			GU[6*i:6*i+6, 6*i:6*i+6] = u_e
		# print(GF)
		# print(GR.dot(GU.dot(GS.dot(GU.T))))
		# exit()
		return GF, GR, GS, GU

	def getDiscontinuousVT(self):
		Ax = self.getA().dot(self.x0)
		RecV = np.zeros((3*len(self.T), 2))
		RecT = []
		for t in range(len(self.T)):
			tv0 = self.getF(t).dot(Ax[6*t:6*t+2])
			tv1 = self.getF(t).dot(Ax[6*t+2:6*t+4])
			tv2 = self.getF(t).dot(Ax[6*t+4:6*t+6])
			RecV[3*t,   0] = tv0[0]
			RecV[3*t,   1] = tv0[1]
			RecV[3*t+1, 0] = tv1[0]
			RecV[3*t+1, 1] = tv1[1]
			RecV[3*t+2, 0] = tv2[0]
			RecV[3*t+2, 1] = tv2[1]
			RecT.append([3*t, 3*t+1, 3*t+2])

		return RecV, RecT

	def getContinuousVT(self):
		RecV = np.zeros((2*len(V), 2))
		for i in range(len(self.g)/2):
			RecV[i, 0] = self.g[2*i]
			RecV[i, 1] = self.g[2*i+1]
		return RecV, self.T

class ARAP:

	def __init__(self, imesh):
		self.mesh = imesh
		self.BLOCK = self.mesh.createBlockingMatrix(to_fix = [0])
		A = self.mesh.getA()
		P = self.mesh.getP()
		self.PAx = P.dot(A.dot(self.mesh.x0))
		BtAtPtPAB = self.BLOCK.T.dot(A.T.dot(P.T.dot(P.dot(A.dot(self.BLOCK)))))
		self.pinvBtAtPtPAB = np.linalg.pinv(BtAtPtPAB)

		KKT_matrix1 = np.concatenate((np.eye(BtAtPtPAB.shape[0]), BtAtPtPAB), axis=0)
		KKT_matrix2 = np.concatenate((BtAtPtPAB.T, np.zeros(BtAtPtPAB.shape)), axis=0)
		KKT = np.concatenate((KKT_matrix1, KKT_matrix2), axis=1)
		self.CholFac, self.Lower = scipy.linalg.lu_factor(KKT)

	def energy(self):

	def itR(self):
		theta_list = []
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		for i in range(len(self.mesh.T)):
			PAx_e = self.PAx[6*i:6*i+6]
			PAg_e = PAg[6*i:6*i+6]
			Ue = np.kron(np.eye(3), self.mesh.getU(i))
			Se = np.kron(np.eye(3), self.mesh.getS(i))
			USUPAx = Ue.dot(Se.dot(Ue.T.dot(PAx_e.T)))
			m_PAg = np.zeros((3,2))
			m_PAg[0:1, 0:2] = PAg_e[0:2]
			m_PAg[1:2, 0:2] = PAg_e[2:4]
			m_PAg[2:3, 0:2] = PAg_e[4:6]

			m_USUPAx = np.zeros((3,2))
			m_USUPAx[0:1, 0:2] = USUPAx[0:2].T
			m_USUPAx[1:2, 0:2] = USUPAx[2:4].T
			m_USUPAx[2:3, 0:2] = USUPAx[4:6].T
			F = np.matmul(m_PAg.T,m_USUPAx)

			u, s, vt = np.linalg.svd(F, full_matrices=True)
			R = np.matmul(vt.T, u.T)
			veca = np.array([1,0])
			vecb = np.dot(R, veca)
			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
			theta_list.append(theta)
			self.mesh.q[3*i] = theta
		return theta_list
	
	def itT(self, useKKT=False):
		F = self.mesh.getGlobalF()[0]
		FPAx = F.dot(self.PAx)
		BtAtPtFPAx = self.BLOCK.T.dot(self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx)))
		if useKKT:
			ob = np.concatenate((np.zeros(len(BtAtPtFPAx)), BtAtPtFPAx))
			gu = scipy.linalg.lu_solve((self.CholFac, self.Lower), ob)
			self.mesh.g = BLOCK.dot(gu[0:len(BtAtPtFPAx)])
			
			return 1
		else:
			BtAtPtFPAx = self.BLOCK.T.dot(self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx)))
			self.mesh.g = self.BLOCK.dot(self.pinvBtAtPtPAB.dot(BtAtPtFPAx))

			return 1

	def iterate(self, useKKT = False):
		r = self.itR()
		g = self.itT(useKKT = useKKT)

def display():
	mesh = Mesh(rectangle_mesh(2,2))
	arap = ARAP(mesh)
	viewer = igl.viewer.Viewer()
	def key_down(viewer, b, c):
		RV, RT = mesh.getDiscontinuousVT()
		V2 = igl.eigen.MatrixXd(RV)
		T2 = igl.eigen.MatrixXi(RT)
		viewer.data.set_mesh(V2, T2)
		arap.iterate()

		return True
	key_down(viewer, "a", 1)
	viewer.core.is_animating = False
	viewer.callback_key_down = key_down
	viewer.launch()

display()