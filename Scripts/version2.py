import numpy as np
import math
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})

gravity = -9.81

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
	V = [[0,0], [1,0], [1,1]]
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
		self.Mass = None
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

	def getC(self):
		if(self.C is None):
			self.C = np.kron(np.eye(len(self.T)), np.kron(np.ones((3,3))/3 , np.eye(2)))
		return self.C

	def getU(self, ind):
		if ind%2== 1:
			alpha = np.pi/4
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

	def getMassMatrix(self):
		if(self.Mass is None):
			self.Mass = np.zeros((2*len(self.V), 2*len(self.V)))

			for i in range(len(self.T)):
				e = self.T[i]
				undef_area = 10*get_area(self.V[self.T[i][0]], self.V[self.T[i][1]], self.V[self.T[i][2]])

				self.Mass[2*e[0]+0, 2*e[0]+0] += undef_area/3.0
				self.Mass[2*e[0]+1, 2*e[0]+1] += undef_area/3.0
				
				self.Mass[2*e[1]+0, 2*e[1]+0] += undef_area/3.0
				self.Mass[2*e[1]+1, 2*e[1]+1] += undef_area/3.0
				
				self.Mass[2*e[2]+0, 2*e[2]+0] += undef_area/3.0
				self.Mass[2*e[2]+1, 2*e[2]+1] += undef_area/3.0

		return self.Mass

class ARAP:

	def __init__(self, imesh, ito_fix = []):
		self.mesh = imesh
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix(to_fix = ito_fix)
		#these are the fixed vertices which stay constant
		self.Abtg = self.ANTI_BLOCK.T.dot(self.mesh.g)

		A = self.mesh.getA()
		P = self.mesh.getP()
		B = self.BLOCK
		C = self.ANTI_BLOCK.T
		self.PAx = P.dot(A.dot(self.mesh.x0))
		AtPtPA = A.T.dot(P.T.dot(P.dot(A)))

		#LU inverse
		col1 = np.concatenate((AtPtPA, C), axis=0)
		col2 = np.concatenate((C.T, np.zeros((C.shape[0], C.shape[0]))), axis=0)
		KKT = np.concatenate((col1, col2), axis =1)
		self.CholFac, self.Lower = scipy.linalg.lu_factor(KKT)

	def energy(self, _g, _R, _S, _U):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(_g))
		FPAx = _R.dot(_U.dot(_S.dot(_U.T.dot(self.PAx))))
		return 0.5*(np.dot(PAg - FPAx, PAg - FPAx))

	def Energy(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		F,R,S,U = self.mesh.getGlobalF()
		FPAx = R.dot(U.dot(S.dot(U.T.dot(self.PAx))))
		en = 0.5*(np.dot(PAg - FPAx, PAg - FPAx))
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

		return dEds, dgds, drds
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
			PAx_e = self.PAx[6*i:6*i+6]
			# if(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) == 0):
			# 	PAx_e = self.PAx[6*i:6*i+6]
			# elif(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) == 2):
			# 	print("One rotation")
			# 	rotate_around = np.concatenate(
			# 		(np.diag(self.mesh.fixedOnElement[6*i:6*i+2]),
			# 			np.diag(self.mesh.fixedOnElement[6*i+2:6*i+4]),
			# 			np.diag(self.mesh.fixedOnElement[6*i+4:6*i+6])), axis =1)
			# 	r = np.eye(6) - np.concatenate((rotate_around, rotate_around, rotate_around))
			# 	# print(r)
			# 	PAx_e = r.dot(Ax[6*i:6*i+6])

			# elif(np.sum(self.mesh.fixedOnElement[6*i:6*i+6]) >= 3):
			# 	print("No rotation")
			# 	continue

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
			# print(theta)
			self.mesh.q[3*i] = theta

		return theta_list
	
	def itT(self):
		F = self.mesh.getGlobalF()[0]
		B = self.BLOCK

		FPAx = F.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		
		gd = np.concatenate((AtPtFPAx, self.Abtg))	
		gu = scipy.linalg.lu_solve((self.CholFac, self.Lower), gd)

		self.mesh.g = gu[0:AtPtFPAx.size]
		return 1

	def iterate(self, its=1):
		for i in range(its):
			r = self.itR()
			g = self.itT()

class NeohookeanElastic:

	def __init__(self, imesh, ito_fix = []):
		self.mesh = imesh
		self.f = np.zeros(2*len(self.mesh.V))
		self.v = np.zeros(2*len(self.mesh.V))
		self.M = self.mesh.getMassMatrix()
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix(to_fix = ito_fix)

		youngs = 1e2
		poissons = 0.3
		self.mu = youngs/(2+ 2*poissons)
		self.lambd = youngs*poissons/((1+poissons)*(1-2*poissons))
		self.dimensions = 2

	def ElementEnergy(self, xt, t):
		e = self.mesh.T[t]
		undef_area = get_area(self.mesh.V[e[0]], self.mesh.V[e[1]], self.mesh.	V[e[2]])

		c1 = xt[2*e[0]:2*e[0]+2] - xt[2*e[2]:2*e[2]+2]
		c2 = xt[2*e[1]:2*e[1]+2] - xt[2*e[2]:2*e[2]+2]
		Ds = np.column_stack((c1, c2))

		d1 = np.array(self.mesh.V[e[0]]) - np.array(self.mesh.V[e[2]])
		d2 = np.array(self.mesh.V[e[1]]) - np.array(self.mesh.V[e[2]])
		Dm = np.column_stack(( d1, d2))
		Dm = np.linalg.inv(Dm)

		F = np.matmul(Ds, Dm)
		# print(Dm)
		#Neo
		J = np.linalg.det(F)
		if(J<=0):
			print("F",F)
			print("J", J)
			print("Ds",Ds)
			print("Dm",Dm)
			print(xt, t)
			print("det(F) is 0 in ENERGY")
			return 1e40		
		I1 = np.trace(F.T.dot(F))
		powj = math.pow(J, -2.0/3)
		I1bar = powj*I1 

		#neo_e = undef_area*(self.mu*(I1bar - self.dimensions)/2.0 + (J-1.0)*(J-1.0)*self.lambd/2.0)
		neo_e = (self.mu/2.0)*(I1 - 2) - self.mu*math.log(J) + 0.5*self.lambd*math.log(J)*math.log(J)
		return neo_e

	def element_energy_GivenF(self, F, t):
		#Neo
		J = np.linalg.det(F)
		if(J<=0):
			print("F",F)
			print("J", J)
			print("Ds",Ds)
			print("Dm",Dm)
			print("xt, t",xt, t)
			print("det(F) is 0 in ENERGY")
			return 1e40		
		I1 = np.trace(F.T.dot(F))
		powj = math.pow(J, -2.0/3)
		I1bar = powj*I1 

		#neo_e = undef_area*(self.mu*(I1bar - self.dimensions)/2.0 + (J-1.0)*(J-1.0)*self.lambd/2.0)
		neo_e = (self.mu/2.0)*(I1 - 2) - self.mu*math.log(J) + 0.5*self.lambd*math.log(J)*math.log(J)
		return neo_e

	def ElementForce(self, f, xt, t):
		e = self.mesh.T[t]
		undef_area = get_area(self.mesh.V[e[0]], self.mesh.V[e[1]], self.mesh.	V[e[2]])

		c1 = xt[2*e[0]:2*e[0]+2] - xt[2*e[2]:2*e[2]+2]
		c2 = xt[2*e[1]:2*e[1]+2] - xt[2*e[2]:2*e[2]+2]
		Ds = np.column_stack((c1, c2))

		d1 = np.array(self.mesh.V[e[0]]) - np.array(self.mesh.V[e[2]])
		d2 = np.array(self.mesh.V[e[1]]) - np.array(self.mesh.V[e[2]])
		Dm = np.column_stack(( d1, d2))
		Dm = np.linalg.inv(Dm)

		F = np.matmul(Ds, Dm)
		
		#Neo
		J = np.linalg.det(F)
		if(J<=0):
			print(F, xt, t)
			print("det(F) is 0, instantaneous force too high")
			exit()

		I1 = np.trace(F.T.dot(F))
		powj = math.pow(J, -2.0/3)
		I1bar = powj*I1

		#P = self.mu*(powj*F)+((-1*self.mu*I1*powj/self.dimensions) + self.lambd*(J-1)*J)*np.linalg.inv(F).T
	
		P = self.mu*(F - np.transpose(np.linalg.inv(F))) + self.lambd*math.log(J)
		
		H = -1*undef_area*P.dot(Dm.T)

		f0 = H[:,0]
		f1 = H[:,1]
		f2 = -1*H[:,0] - H[:,1]
		f[2*e[0]:2*e[0]+2] += f0 
		f[2*e[1]:2*e[1]+2] += f1
		f[2*e[2]:2*e[2]+2] += f2
		return P

	def Energy(self, _x):
		En = 0.0
		for t in range(len(self.mesh.T)):
			En += self.ElementEnergy(_x, t)

		return En

	def Forces(self, _x):
		self.f.fill(0)
		for t in range(len(self.mesh.T)):
			self.ElementForce(self.f, _x, t)

		#gravity
		# for i in range(len(self.f)/2):
		# 	self.f[2*i+1] += self.M[2*i+1, 2*i+1]*gravity 

def FiniteDifferencesARAP():
	eps = 1e-5
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

		print(np.sum(arap.dEdg() - np.array(dEdg)))

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

		print(np.sum(np.array(dEds)-realdEds))

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

		print(np.sum(np.array(dEdr) - realdEdr))

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


		print(np.linalg.norm(fake - real))
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

		print(np.linalg.norm(real -fake))
		return real

	def check_d_gradEgrds():
		real = arap.d_gradEgrds()
		fake = np.zeros(real.shape)

		dEdg = arap.dEdg()
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.q[3*i+j] += eps
				fake[0:len(mesh.g),2*i+(j-1)] = (arap.dEdg() - dEdg)/eps
				mesh.q[3*i+j] -= eps

		dEdr = arap.dEdr()
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.q[3*i+j] += eps
				fake[len(mesh.g): ,2*i+(j-1)] = (arap.dEdr()[1] - dEdr[1])/eps
				mesh.q[3*i+j] -= eps

		print(np.linalg.norm(fake - real))

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
	
	check_dEdg()
	check_dEds()
	check_dEdr()
	left = check_d_gradEgrdg()
	right = check_d_gradEgrdr()
	check_d_gradEgrds()
# FiniteDifferencesARAP()

def FiniteDifferencesElasticity():
	eps = 1e-7
	mesh = Mesh(triangle_mesh())
	ne = NeohookeanElastic(imesh = mesh)
	arap = ARAP(imesh=mesh)
	E0 = ne.Energy(_x=mesh.g)

	def check_dEdx():
		ne.Forces(_x=mesh.g)
		real = -1*ne.f
		fake = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			dg[i] += eps 
			En = ne.Energy(_x=dg)
			print(En)
			fake.append((En - E0)/eps)
			dg[i] -= eps
		
		print(fake)
		print(real)

	def check_dEdF():
		realP = ne.ElementForce(f=ne.f, xt = mesh.g, t=0)
		print(realP)
		F = np.eye(2)

		e0 = ne.element_energy_GivenF(F=F,t=0)
		F[0,0] += eps
		e1 = ne.element_energy_GivenF(F=F, t=0)
		print((e1-e0)/eps)

	def check_dEds():

	check_dEdx()
	# check_dEdF()

FiniteDifferencesElasticity()

class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 0
		self.timestep = 0.001
		self.mesh = imesh
		self.arap = iarap 
		self.elastic = ielastic 
		self.adder = 0.001
		self.set_random_strain()


	def set_strain(self):
		for i in range(len(self.mesh.T)):
			pass
			# self.mesh.q[3*i+1] =1.01+ np.sin(self.time)
			# self.mesh.q[3*i +2] = 1.01 + np.sin(self.time)

	def set_random_strain(self):
		for i in range(len(self.mesh.T)):
			# self.mesh.q[3*i + 1] = 1.01 + np.random.uniform(0,2)
			self.mesh.q[3*i + 2] = 1.01 + np.random.uniform(0,2)

	def iterate(self):
		# self.set_strain()
		# self.arap.iterate(its=10)
		# print(self.mesh.g)
		self.mesh.g[2*self.mesh.fixed[1]] += 0.01
		self.time += self.timestep

	def only_solve_ARAP(self):
		self.arap.iterate(its=10)

		s0 = []
		for i in range(len(self.mesh.T)):
			s0.append(self.mesh.q[3*i + 1])
			s0.append(self.mesh.q[3*i + 2])

		print(s0, self.arap.Energy())

		def energy(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]
			
			# print(s)
			return self.arap.Energy()

		def jacobian(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			# print(s)
			return self.arap.Jacobian()[0]

		res = scipy.optimize.minimize(energy, s0, method='BFGS', jac=jacobian, options={'gtol': 1e-6, 'disp': True})
		
		for i in range(len(res.x)/2):
			self.mesh.q[3*i+1] = res.x[2*i]
			self.mesh.q[3*i+2] = res.x[2*i+1]

	def only_solve_Statics(self):
		self.iterate()
		
		AbAbtg = self.elastic.ANTI_BLOCK.dot(self.elastic.ANTI_BLOCK.T.dot(self.mesh.g))
		x0 = self.elastic.BLOCK.T.dot(self.mesh.g)
		

		def energy(_x):
			x = self.elastic.BLOCK.dot(_x) + AbAbtg

			StrainE = self.elastic.Energy(_x=x)

			E = StrainE
			return abs(E)

		def jacobian(_x):
			x = self.elastic.BLOCK.dot(_x) + AbAbtg
			
			self.elastic.Forces(_x=x)
			jac = self.elastic.BLOCK.T.dot(-1*self.elastic.f)

			return jac

		res = scipy.optimize.minimize(energy, x0, method='BFGS', jac=jacobian,  options={'gtol': 1e-6, 'disp': True, 'eps':1e-08})
		new_g = self.elastic.BLOCK.dot(res.x) + AbAbtg
		self.mesh.g = new_g
		print(self.mesh.g)

	def solve(self):
		self.arap.iterate(its=4)

		s0 = []
		for i in range(len(self.mesh.T)):
			s0.append(self.mesh.q[3*i +1])
			s0.append(self.mesh.q[3*i +2])

		def energy(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			E_arap = self.arap.Energy()
			E_elastic = self.elastic.Energy(_x=self.mesh.g)
			print("E", E_arap, E_elastic)
			return E_elastic

		def jacobian(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			J_arap, dgds, drds = self.arap.Jacobian()
			self.elastic.Forces(_x=self.mesh.g)
			J_elastic = -1*self.elastic.f.dot(dgds)
			print("f", self.elastic.f)
			print("Jac", J_elastic)
			return J_elastic
		
		res = scipy.optimize.minimize(energy, s0, method='Nelder-Mead',  options={'gtol': 1e-6, 'disp': True})
		# res = scipy.optimize.minimize(energy, s0, method='BFGS', jac=jacobian, options={'gtol': 1e-6, 'disp': True, 'eps':1e-08})
		print("s", res.x)

		for i in range(len(res.x)/2):
			self.mesh.q[3*i+1] = res.x[2*i]
			self.mesh.q[3*i+2] = res.x[2*i+1]

def display():
	mesh = Mesh(rectangle_mesh(2,2))
	neoh =NeohookeanElastic(imesh=mesh, ito_fix=[1,3])
	arap = ARAP(imesh=mesh, ito_fix = [1,3])
	time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)
	viewer = igl.viewer.Viewer()

	def key_down(viewer, a, bbbb):
		viewer.data.clear()
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
			C.append(mesh.g[2*mesh.fixed[i]:2*mesh.fixed[i]+2])

		viewer.data.add_points(igl.eigen.MatrixXd(np.array(C)), red)

		CAg = mesh.getC().dot(mesh.getA().dot(mesh.g))
		cag = []
		for i in range(len(mesh.T)):
			cag.append(CAg[6*i:6*i+2])

		viewer.data.add_points(igl.eigen.MatrixXd(np.array(cag)), green)
		# zero = 
		# viewer.data.add_points(, black)
		# time_integrator.only_solve_Statics()
		time_integrator.solve()
		return True

	key_down(viewer, 'a', 1)
	viewer.callback_key_down= key_down
	viewer.core.is_animating = False
	viewer.launch()

# display()