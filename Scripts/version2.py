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
	return np.array(V), np.array(T)	

def triangle_mesh():
	V = [[0,0], [1,0], [0,1]]
	T = [[0,1,2]]
	return V, T

class Mesh:
	#class vars

	def __init__(self, VT):
		#object vars
		self.fixed = []
		self.V = VT[0]
		self.T = VT[1]
		self.fixedOnElement = None
		self.P = None
		self.A = None
		self.C = None
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)+np.ravel(self.V)

		self.q = np.zeros(len(self.T)*(1+2)) #theta, sx, sy

		#set initial strains
		for i in range(len(self.T)):
			self.q[3*i + 1] = 1
			self.q[3*i + 2] = 1

		#set initial rots
		# for i in range(len(self.T)):
		# 	self.q[3*i] = np.pi/100
		# 	break
	def createBlockingMatrix(self, to_fix=[]):
		self.P = None #reset P because blocking verts will change P

		self.fixed = to_fix

		onVerts = np.zeros(len(self.V))
		onVerts[to_fix] = 1
		self.fixedOnElement = self.getA().dot(np.kron(onVerts, np.ones(2)))
		
		b = np.kron(np.delete(np.eye(len(self.V)), to_fix, axis =1), np.eye(2))

		ab = np.zeros(len(self.V))
		ab[to_fix] = 1
		to_reset = [i for i in range(len(ab)) if ab[i]==0]

		if (len(to_fix) == 0):
			return b, np.zeros((2*len(self.V), (2*len(self.V))))

		anti_b = np.kron(np.delete(np.eye(len(self.V)), to_reset, axis =1), np.eye(2))
		

		return b, anti_b


	def getP(self):
		if(self.P is None):
			P = np.zeros((6*len(self.T), 6*len(self.T)))
			sub_P = np.kron(np.matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.eye(2))
			# sub_P = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]]), np.eye(2))
			print(sub_P)
			for i in range(len(self.T)):
				P[6*i:6*i+6, 6*i:6*i+6] = sub_P
				# if(np.sum(self.fixedOnElement[6*i:6*i+6]) == 0):
				# 	P[6*i:6*i+6, 6*i:6*i+6] = sub_P

				# elif(np.sum(self.fixedOnElement[6*i:6*i+6]) == 2):
				# 	print("One rotation")
				# 	rotate_around = np.concatenate(
				# 		(np.diag(self.fixedOnElement[6*i:6*i+2]),
				# 			np.diag(self.fixedOnElement[6*i+2:6*i+4]),
				# 			np.diag(self.fixedOnElement[6*i+4:6*i+6])), axis =1)
				# 	r = np.eye(6) - np.concatenate((rotate_around, rotate_around, rotate_around))
				# 	print(r)
				# 	P[6*i:6*i+6, 6*i:6*i+6] = r

				# elif(np.sum(self.fixedOnElement[6*i:6*i+6]) >= 3):
				# 	continue
					#P[6*i:6*i+6, 6*i:6*i+6] = sub_P 
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

	def getC(self):
		if(self.C is None):
			self.C = np.kron(np.eye(len(self.T)), np.kron(np.ones((3,3))/3 , np.eye(2)))
		return self.C

	def getU(self, ind):
		if ind%2== 1:
			alpha = 0*np.pi/4
		else:
			alpha = 0*np.pi/4

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

	def getGlobalF(self, onlyget=0):
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

		return GF, GR, GS, GU

	def getDiscontinuousVT(self):
		CAg = self.getC().dot(self.getA().dot(self.g))
		Ax = self.getA().dot(self.x0) - CAg
		Fax = self.getGlobalF()[0].dot(Ax)
		new = Fax - self.getC().dot(Fax) + CAg
		RecV = np.zeros((3*len(self.T), 2))
		RecT = []
		for t in range(len(self.T)):
			RecV[3*t,   0] = new[6*t + 0]
			RecV[3*t,   1] = new[6*t + 1]
			RecV[3*t+1, 0] = new[6*t + 2]
			RecV[3*t+1, 1] = new[6*t + 3]
			RecV[3*t+2, 0] = new[6*t + 4]
			RecV[3*t+2, 1] = new[6*t + 5]
			RecT.append([3*t, 3*t+1, 3*t+2])

		return RecV, RecT

	def getContinuousVT(self):
		RecV = np.zeros((2*len(self.V), 2))
		for i in range(len(self.g)/2):
			RecV[i, 0] = self.g[2*i]
			RecV[i, 1] = self.g[2*i+1]
		return RecV, self.T


class ARAP:

	def __init__(self, imesh, ito_fix = []):
		self.mesh = imesh
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix(to_fix = ito_fix)
		#these are the fixed vertices which stay constant
		self.AbAbtg = self.ANTI_BLOCK.dot(self.ANTI_BLOCK.T.dot(self.mesh.g))

		A = self.mesh.getA()
		P = self.mesh.getP()
		self.PAx = P.dot(A.dot(self.mesh.x0))
		BtAtPtPAB = self.BLOCK.T.dot(A.T.dot(P.T.dot(P.dot(A.dot(self.BLOCK)))))
		print("not inv")
		print(BtAtPtPAB)

		#SVD inverse
		self.pinvBtAtPtPAB = np.linalg.pinv(BtAtPtPAB)
		
		#KKT Inverse
		KKT_matrix1 = np.concatenate((np.eye(BtAtPtPAB.shape[0]), BtAtPtPAB), axis=0)
		KKT_matrix2 = np.concatenate((BtAtPtPAB.T, np.zeros(BtAtPtPAB.shape)), axis=0)
		KKT = np.concatenate((KKT_matrix1, KKT_matrix2), axis=1)
		self.KKTCholFac, self.KKTLower = scipy.linalg.lu_factor(KKT)

		#LU inverse
		self.CholFac, self.Lower = scipy.linalg.lu_factor(BtAtPtPAB)

	def energy(self, _g, _R, _S, _U):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(_g))
		FPAx = _R.dot(_U.dot(_S.dot(_U.T.dot(self.PAx))))
		return 0.5*(np.dot(PAg - FPAx, PAg - FPAx))

	def Energy(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		F,R,S,U = self.mesh.getGlobalF()
		# print(R)
		# print(S)
		# print(U - np.eye(12))
		FPAx = R.dot(U.dot(S.dot(U.T.dot(self.PAx))))
		en = 0.5*(np.dot(PAg - FPAx, PAg - FPAx))
		print("en", en)
		print("q", self.mesh.q)
		return en

	def Jacobian(self):
		lhs_left = self.d_gradEgrdg()
		lhs_right = self.d_gradEgrdr()

		lhs = np.concatenate((lhs_left, lhs_right), axis =1)
		rhs = self.d_gradEgrds()

		SVDinv_Jac_s = np.matmul(np.linalg.pinv(lhs), rhs)

		dgds = SVDinv_Jac_s[0:lhs_left.shape[1],:]
		drds = SVDinv_Jac_s[lhs_left.shape[1]:,:]
		
		dEds = np.matmul(self.dEdg(),dgds) + np.matmul(self.dEdr()[1], drds) + self.dEds()[1]


		return dEds
		# KKTinv_Jac_s

	def dEdr(self):
		#TODO
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		dEdR = -1*np.multiply.outer(PAg, USUt.dot(self.PAx))
		_dRdr = self.dRdr()
		dEdr = np.tensordot(dEdR, _dRdr, axes = ([0,1],[0,1]))

		return dEdR, dEdr
	
	def dRdr(self):
		#	Iterate through each element, 
		#		set dRdrx and dRdry 
		#		then insert into a global dRdr matrix 
		#		then assemble them into tensor
		_dRdr = None
		for t in range(0, len(self.mesh.T)):
			c, s = np.cos(self.mesh.q[3*t]), np.sin(self.mesh.q[3*t])
			dRdr_e = np.kron(np.eye(3), np.array(((-s,-c), (c, -s))))

			gdRdr = np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T))) 
			gdRdr[6*t:6*t+6, 6*t:6*t+6] = dRdr_e


			if t == 0:
				_dRdr = np.dstack([gdRdr])
			else:
				_dRdr = np.dstack([_dRdr, gdRdr])

		return _dRdr

	def d2Rdr2(self):
		_d2Rdr2 = []
		for t in range(0,len(self.mesh.T)):
			ddRdrdrt = None
			for r in range(len(self.mesh.T)): 
				gddRdrdr = np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T)))

				if(t==r):
					c, s = np.cos(self.mesh.q[3*t]), np.sin(self.mesh.q[3*t])
					ddR_e = np.kron(np.eye(3), np.array(((-c,s), (-s, -c))))
					gddRdrdr[6*t:6*t+6, 6*t:6*t+6] = ddR_e 

				if r == 0:
					ddRdrdrt = np.dstack([gddRdrdr])
				else:
					ddRdrdrt = np.dstack([ddRdrdrt, gddRdrdr])
			_d2Rdr2.append(ddRdrdrt)	
		return np.array(_d2Rdr2)

	def dEdg(self):
		F = self.mesh.getGlobalF()[0]
		FPAx = F.dot(self.PAx)
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		AtPtPAg = self.mesh.getA().T.dot(self.mesh.getP().T.dot(PAg))
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		return AtPtPAg - AtPtFPAx

	def dEds(self):
		#TODO
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		gF, gR, gS, gU = self.mesh.getGlobalF()
		UtPAx = gU.T.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		RU = gR.dot(gU)
		dEdS =  np.multiply.outer(gS.dot(UtPAx), UtPAx) - np.multiply.outer(np.dot(RU.T, PAg), UtPAx)
		_dSds = self.dSds()#rank 3 tensor
		
		dEds = np.tensordot(dEdS, _dSds, axes = ([0,1], [0,1]))
		
		return dEdS, dEds
		
	def dSds(self):
		#	Iterate through each element, 
		#		set dSdsx and dSdsy 
		#		then insert into a global dSds matrix 
		#		assemble them into tensor
		_dSds = None
		for t in range(0, len(self.mesh.T)):
			dSdsx = np.array([[1,0],[0,0]])
			dSdsy = np.array([[0,0],[0,1]])
			
			dSdsx_e = np.kron(np.eye(3), dSdsx)
			dSdsy_e = np.kron(np.eye(3), dSdsy)

			gdSdsx = np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T))) 
			gdSdsy = np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T)))
			gdSdsx[6*t:6*t+6, 6*t:6*t+6] = dSdsx_e
			gdSdsy[6*t:6*t+6, 6*t:6*t+6] = dSdsy_e
				

			if t == 0:
				_dSds = np.dstack([gdSdsx, gdSdsy])
			else:
				_dSds = np.dstack([_dSds, gdSdsx, gdSdsy])
		return _dSds

	def d_gradEgrdg(self):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		USUtPAx = USUt.dot(self.PAx)

		d_gEgdg = self.mesh.getA().T.dot(self.mesh.getP().T.dot(self.mesh.getP().dot(self.mesh.getA())))
		_dRdr = self.dRdr()
		d_gErdg = np.tensordot(np.multiply.outer(-1*PA.T, USUtPAx.T), _dRdr, axes=([1,2], [0,1]))

		return np.concatenate((d_gEgdg, d_gErdg.T))

	def d_gradEgrdr(self):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		USUtPAx = USUt.dot(self.PAx)

		_dRdr = self.dRdr()
		d_gEgdR = np.multiply.outer(-1*PA.T, USUtPAx.T)
		d_gEgdr = np.tensordot(d_gEgdR, _dRdr, axes =([1,2],[0,1]))
		
		_ddRdrdr = self.d2Rdr2()
		negPAg_USUtPAx = np.multiply.outer( -1*PA.dot(self.mesh.g), USUtPAx)
		d_gErdr = np.tensordot(negPAg_USUtPAx, _ddRdrdr, axes = ([0,1],[1,2]))
		
		return np.concatenate((d_gEgdr, d_gErdr))

	def d_gradEgrds(self):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		UtPAx = gU.T.dot(PA.dot(self.mesh.x0))
		PAg = PA.dot(self.mesh.g)
		_dSds = self.dSds()
		_dRdr = self.dRdr()
		PAtRU = PA.T.dot(gR.dot(gU))

		d_gEgdS = np.multiply.outer(-1*PAtRU, UtPAx.T)

		d_gEgds = np.tensordot(d_gEgdS, _dSds, axes =([1,2],[0,1]))

		negPAg_U_UtPAx = np.multiply.outer(-1*PAg, np.multiply.outer(gU, UtPAx))
		negPAg_U_UtPAx_dRdr = np.tensordot(negPAg_U_UtPAx, _dRdr, axes=([0,1],[0,1]))
		d_gErds = np.tensordot(negPAg_U_UtPAx_dRdr, _dSds, axes=([0,1],[0,1]))

		return np.concatenate((d_gEgds, d_gErds))

	def itR(self):
		theta_list = []
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		Ax = self.mesh.getA().dot(self.mesh.x0)
		for i in range(len(self.mesh.T)):
			if(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) == 0):
				print("None")
				PAx_e = self.PAx[6*i:6*i+1]
			elif(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) == 2):
				print("One rotation")
				rotate_around = np.concatenate(
					(np.diag(self.mesh.fixedOnElement[6*i:6*i+2]),
						np.diag(self.mesh.fixedOnElement[6*i+2:6*i+4]),
						np.diag(self.mesh.fixedOnElement[6*i+4:6*i+6])), axis =1)
				r = np.eye(6) - np.concatenate((rotate_around, rotate_around, rotate_around))
				print(r)
				PAx_e = r.dot(Ax[6*i:6*i+6])

			elif(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) >= 3):
				print("No rotation")
				continue
				print("shit, fix this")

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
			print(theta)
			self.mesh.q[3*i] = theta

		return theta_list
	
	def itT(self, useKKT=False):
		F = self.mesh.getGlobalF()[0]
		B = self.BLOCK
		# print(B)
		ABBtx = self.mesh.getA().dot(B.dot(B.T.dot(self.mesh.x0)))
		AtPtFPABBtx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(F.dot(self.mesh.getP().dot(ABBtx))))
		BtAtPtFPAx = B.T.dot(AtPtFPABBtx)
		if useKKT:
			ob = np.concatenate((self.BLOCK.T.dot(self.mesh.x0), BtAtPtFPAx))
			gu = scipy.linalg.lu_solve((self.KKTCholFac, self.KKTLower), ob)
			self.mesh.g = self.BLOCK.dot(gu[0:len(BtAtPtFPAx)]) + self.AbAbtg
			print("Using KKT!!")
			return 1
		else:
			gu = scipy.linalg.lu_solve((self.CholFac, self.Lower), BtAtPtFPAx)
			self.mesh.g = self.BLOCK.dot(gu) + self.AbAbtg
			
			# self.mesh.g = self.BLOCK.dot(self.pinvBtAtPtPAB.dot(BtAtPtFPAx)) + self.AbAbtg
			print(self.mesh.g)
			# print(BtAtPtFPAx)
			# print(self.AbAbtg)

			return 1

	def iterate(self, its=1, useKKT = False):
		for i in range(its):
			self.Energy()
			r = self.itR()
			g = self.itT(useKKT = useKKT)

class LinearElastic:

	def __init__(self, imesh):
		pass

def FiniteDifferences():
	eps = 1e-4
	mesh = Mesh(rectangle_mesh(2,2))
	arap = ARAP(mesh)
	
	F0,R0,S0,U0 = mesh.getGlobalF()
	E0 = arap.energy(_g=mesh.g, _R =R0, _S=S0, _U=U0)
	print("Default Energy ", E0)
	
	def check_dEdg():
		dEdg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			dg[i] += eps
			Ei = arap.energy(_g=dg, _R =R0, _S=S0, _U=U0)
			dEdg.append((Ei - E0)/eps)
			dg[i] -= eps

		print(arap.dEdg() - np.array(dEdg))

	def check_dEds():
		realdEdS, realdEds = arap.dEds()
		dEds = []
		for i in range(len(mesh.T)):
			mesh.q[3*i+1] += eps
			F,R,S,U = mesh.getGlobalF()
			Ei = arap.energy(_g =mesh.g, _R=R0, _S=S, _U=U0)
			dEds.append((Ei - E0)/eps)
			mesh.q[3*i+1] -= eps

			mesh.q[3*i+2] += eps
			F,R,S,U = mesh.getGlobalF()
			Ei = arap.energy(_g =mesh.g, _R=R0, _S=S, _U=U0)
			dEds.append((Ei - E0)/eps)
			mesh.q[3*i+2] -=eps

		print(realdEds)
		print(np.array(dEds))

	def check_dEdS():
		dEdS_real, dEds_real = arap.dEds()
		F,R,S,U = mesh.getGlobalF()
		
		S[0,0] += eps
		Ei = arap.energy(_g=mesh.g, _R =R0, _S=S, _U=U0)
		print((Ei - E0)/eps - dEdS_real[0,0])

	def check_dEdr():
		realdEdR, realdEdr = arap.dEdr()
		dEdr = []
		for i in range(len(mesh.T)):
			mesh.q[3*i] += eps
			F,R,S,U = mesh.getGlobalF()
			Ei = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
			dEdr.append((Ei - E0)/eps)
			mesh.q[3*i] -= eps

		print(realdEdr)
		print(dEdr)

	def check_d_gradEgrdg():
		real = arap.d_gradEgrdg()
		fake = np.zeros(real.shape)
		
		dEdg = arap.dEdg()
		for i in range(len(mesh.g)):
			mesh.g[i] += eps
			fake[0:len(mesh.g), i] = (arap.dEdg() - dEdg)/eps
			mesh.g[i] -= eps

		dEdr = arap.dEdr()
		for i in range(len(mesh.g)):
			mesh.g[i] += eps
			fake[len(mesh.g):, i] = (arap.dEdr()[1] - dEdr[1])/eps
			mesh.g[i] -= eps


		# print(np.linalg.norm(fake - real))
		# print(real)
		return real

	def check_d_gradEgrdr():
		real = arap.d_gradEgrdr()
		fake = np.zeros(real.shape)

		dEdg = arap.dEdg()
		for i in range(len(mesh.T)):
			mesh.q[3*i] += eps
			fake[0:len(mesh.g), i] = (arap.dEdg() - dEdg)/eps
			mesh.q[3*i] -= eps

		dEdr = arap.dEdr()
		for i in range(len(mesh.T)):
			mesh.q[3*i] += eps
			fake[len(mesh.g): , i] = (arap.dEdr()[1] - dEdr[1])/eps
			mesh.q[3*i] -= eps

		# print(real)
		# print("fake")
		# print(fake)
		return real

	def check_d_gradEgrds():
		real = arap.d_gradEgrds()
		fake = np.zeros(real.shape)

		dEdg = arap.dEdg()
		for i in range(len(mesh.T)):
			for j in range(1,3):
				print(3*i+j, 2*i+(j-1))
				mesh.q[3*i+j] += eps
				fake[0:len(mesh.g),2*i+(j-1)] = (arap.dEdg() - dEdg)/eps
				mesh.q[3*i+j] -= eps

		dEdr = arap.dEdr()
		for i in range(len(mesh.T)):
			for j in range(1,3):
				print(3*i+j, 2*i+(j-1))
				mesh.q[3*i+j] += eps
				fake[len(mesh.g): ,2*i+(j-1)] = (arap.dEdr()[1] - dEdr[1])/eps
				mesh.q[3*i+j] -= eps

		print(real)
		print("fake")
		print(fake)
		print(fake - real)

	def check_d_gradEgdS():
		real = arap.d_gradEgrds()[0]
		F,R,S,U = mesh.getGlobalF()

		dEdg = arap.dEdg()

		P = mesh.getP()
		A = mesh.getA()
		PAg = P.dot(A.dot(mesh.g))
		AtPtPAg = A.T.dot(P.T.dot(PAg))
		print(U.T.dot(P.dot(A.dot(mesh.x0))) - real)
		return
		dEgdS = []
		for i in range(12):
			dEgdS.append([])
			for j in range(0,12):
				S[i, j] += eps
				GF = R.dot(U.dot(S.dot(U.T)))
				FPAx = GF.dot(P.dot(A.dot(mesh.x0)))
				AtPtFPAx = A.T.dot(P.T.dot(FPAx))
				dEdg_new = AtPtPAg - AtPtFPAx 
				S[i, j] -= eps
				dEgdS[i].append((dEdg_new - dEdg)/eps)
			

		dEgdS = np.array(dEgdS)
		_dSds = arap.dSds()
		print("diff")
		print(dEgdS[:,:,0])
		print(real[0])

		# print(np.sum(np.multiply(dEgdS[:,:,0], _dSds[:,:,0])))
		# print(np.sum(np.multiply(real[0,:,:], _dSds[:,:,0])))
	
	# check_d_gradEgdS()
	# check_d_gradEgrds()
	# right = check_d_gradEgrdr()
	# left = check_d_gradEgrdg()
	# check_dEdg()
	# check_dEdr()
	# check_dEds()
	# FLHS = np.concatenate((left, right), axis =1)
	# print(len(mesh.q)/3)
	# np.savetxt('full-left.csv', FLHS, delimiter=',')
# FiniteDifferences()


class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 0
		self.mesh = imesh
		self.arap = iarap 
		self.elastic = ielastic 
		self.adder = 0.1


	def set_strain(self):
		for i in range(len(self.mesh.T)):
			pass
			self.mesh.q[3*i+1] =1+ np.sin(self.time)
			# self.mesh.q[3*i +2] = 1+ np.sin(self.time)

	def set_random_strain(self):
		for i in range(len(self.mesh.T)):
			self.mesh.q[3*i+1] = 1 + np.random.uniform(0,1)
			# self.mesh.q[3*i + 2] = 1 + np.random.uniform(0,1)

	def iterate(self):
		# self.set_random_strain()
		self.set_strain()
		self.arap.iterate(its=1, useKKT = False)
		self.time += 0.1

	def solve(self):
		s0 = []
		for i in range(len(self.mesh.T)):
			s0.append(self.mesh.q[3*i + 1])
			s0.append(self.mesh.q[3*i + 2])

		print("initial s: ", s0)
		def energy(s):
			for i in range(len(self.mesh.T)):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			return self.arap.Energy()

		def jacobian(s):
			for i in range(len(self.mesh.T)):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			return self.arap.Jacobian()

		res = scipy.optimize.minimize(energy, np.array(s0), method='BFGS', jac=jacobian, options={'gtol': 1e-6, 'disp': True})
		print(res.x)

def display():
	mesh = Mesh(rectangle_mesh(2,2))
	arap = ARAP(imesh=mesh, ito_fix = [1,3])
	time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = None)
	# time_integrator.solve()
	viewer = igl.viewer.Viewer()

	def key_down(viewer, c, b):
		viewer.data.clear()
		time_integrator.iterate()
		DV, DT = mesh.getDiscontinuousVT()
		RV, RT = mesh.getContinuousVT()
		V2 = igl.eigen.MatrixXd(RV)
		T2 = igl.eigen.MatrixXi(RT)
		viewer.data.set_mesh(V2, T2)

		red = igl.eigen.MatrixXd([[1,0,0]])
		purple = igl.eigen.MatrixXd([[1,0,1]])
		green = igl.eigen.MatrixXd([[0,1,0]])
		black = igl.eigen.MatrixXd([[0,0,0]])

		paxes = []

		for e in DT:
			P = DV[e]
			DP = np.array([P[1], P[2], P[0]])
			viewer.data.add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)


		C = []
		for i in range(len(mesh.fixed)):
			C.append(mesh.V[mesh.fixed[i]])
		# print(C)
		viewer.data.add_points(igl.eigen.MatrixXd(C), red)

		CAg = mesh.getC().dot(mesh.getA().dot(mesh.g))
		cag = []
		for i in range(len(mesh.T)):
			cag.append(CAg[6*i:6*i+2])

		viewer.data.add_points(igl.eigen.MatrixXd(np.array(cag)), green)
		return True

	key_down(viewer, 'a', 1)
	viewer.callback_key_down = key_down
	viewer.core.is_animating = False
	viewer.launch()

display()