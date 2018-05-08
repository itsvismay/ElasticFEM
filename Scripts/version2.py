import numpy as np
import math
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import numdifftools as nd
import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.8f}'.format(x)})

#helpers
def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y):
	V = []
	for i in range(0,x):
		for j in range(0,y):
			V.append([0.1*i, 0.1*j])
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
		if(len(to_fix) == len(self.V)):
			return np.array([[]]), np.eye(2*len(self.V))
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
			sub_P = np.kron(np.matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.eye(2))/3.0
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
		F = self.getGlobalF()[0]
		C = self.getC()
		CAg = C.dot(self.getA().dot(self.g))
		Ax = self.getA().dot(self.x0)
		# print("CFPAx",)
		new = F.dot(self.getP().dot(Ax)) + CAg

		
		# CAg = self.getC().dot(self.getA().dot(self.g))
		# Ax = self.getA().dot(self.x0) - CAg
		# Fax = self.getGlobalF()[0].dot(Ax)
		# new = Fax - self.getC().dot(Fax) + CAg

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
		RecV = np.zeros((len(self.V), 2))
		for i in range(len(self.g)/2):
			RecV[i, 0] = self.g[2*i]
			RecV[i, 1] = self.g[2*i+1]
		
		return RecV, self.T

	def getMassMatrix(self):
		if(self.Mass is None):
			self.Mass = np.zeros((2*len(self.V), 2*len(self.V)))
			density = 1.0
			for i in range(len(self.T)):
				e = self.T[i]
				undef_area = density*get_area(self.V[e[0]], self.V[e[1]], self.V[e[2]])
				self.Mass[2*e[0]+0, 2*e[0]+0] += undef_area/3.0
				self.Mass[2*e[0]+1, 2*e[0]+1] += undef_area/3.0
				
				self.Mass[2*e[1]+0, 2*e[1]+0] += undef_area/3.0
				self.Mass[2*e[1]+1, 2*e[1]+1] += undef_area/3.0
				
				self.Mass[2*e[2]+0, 2*e[2]+0] += undef_area/3.0
				self.Mass[2*e[2]+1, 2*e[2]+1] += undef_area/3.0

		return self.Mass

class ARAP:
	#class vars

	def __init__(self, imesh, ito_fix = []):
		self.mesh = imesh
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix(to_fix = ito_fix)
		#these are the fixed vertices which stay constant

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

	def Jacobian(self, block = False, kkt= True):
		lhs_left = np.concatenate((self.Hess_Egg(block = block), self.Hess_Erg(block=block).T))
		lhs_right = np.concatenate((self.Hess_Erg(block = block),self.Hess_Err(block=block)))

		lhs = np.concatenate((lhs_left, lhs_right), axis =1)
		rhs = -1*np.concatenate((self.Hess_Egs(block=block), self.Hess_Ers(block=block)))

		#Constraining Rotation
		# R = np.eye(len(self.mesh.T))
		R = np.array([[] for i in self.mesh.T]).T
		# R = np.array([[0,1]])


		#NEW KKT SOLVE
		if kkt:
			C = self.ANTI_BLOCK.T
			g_size = C.shape[1]
			gb_size = C.shape[0]
			r_size = R.shape[1]
			rb_size = R.shape[0]
		
			
			col1 = np.concatenate((lhs_left, np.concatenate((C, np.zeros((rb_size, g_size))))))
			# print("col", col1.shape)

			col2 = np.concatenate((lhs_right, np.concatenate((np.zeros((gb_size, r_size)), R))))
			# print("col", col2.shape)

			col3 = np.concatenate((C.T, np.concatenate((np.zeros((r_size, gb_size)), np.concatenate((np.zeros((gb_size, gb_size)), np.zeros((rb_size, gb_size))))))) )
			# print("col", col3.shape)

			col4 = np.concatenate(( np.concatenate((np.zeros((g_size, rb_size)), R.T)), 
				np.concatenate((np.zeros((gb_size, rb_size)), np.zeros((rb_size, rb_size)))) ))
			# print("col", col4.shape)

			jacKKT = np.hstack((col1, col2, col3, col4))
			jacChol, jacLower = scipy.linalg.lu_factor(jacKKT)
			KKT_constrains = np.concatenate((rhs, np.zeros((gb_size+rb_size, rhs.shape[1]))))

			Jac_s = scipy.linalg.lu_solve((jacChol, jacLower), KKT_constrains)
			results = Jac_s[0:rhs.shape[0], :]


		dgds = results[0:lhs_left.shape[1],:]
		drds = results[lhs_left.shape[1]:,:]
		
		if block:
			dgds = self.BLOCK.dot(dgds)

		dEds = np.matmul(self.dEdg(),dgds) + np.matmul(self.dEdr()[1], drds) + self.dEds()[1]

		return dEds, dgds, drds
		# KKTinv_Jac_s

	def dEdr(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		term2 = gR.dot(np.multiply.outer(USUt.dot(self.PAx), USUt.dot(self.PAx)))
		dEdR = -1*np.multiply.outer(PAg, USUt.dot(self.PAx)) + term2
		_dRdr = self.dRdr()
		# print(np.tensordot(term2, _dRdr, axes = ([0,1],[0,1])))
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

	def Hess_Egg(self, block =False):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		USUtPAx = USUt.dot(self.PAx)

		Egg = self.mesh.getA().T.dot(self.mesh.getP().T.dot(self.mesh.getP().dot(self.mesh.getA())))
		if block:
			Egg = self.BLOCK.T.dot(Egg.dot(self.BLOCK))

		return Egg

	def Hess_Erg(self, block=False):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		USUtPAx = USUt.dot(self.PAx)

		_dRdr = self.dRdr()
		Erg = np.tensordot(np.multiply.outer(-1*PA.T, USUtPAx.T), _dRdr, axes=([1,2], [0,1]))
		if block:
			Erg = self.BLOCK.T.dot(Erg)

		return Erg

	def Hess_Err(self, block=False):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		USUt = gU.dot(gS.dot(gU.T))
		USUtPAx = USUt.dot(self.PAx)

		_ddRdrdr = self.d2Rdr2()
		negPAg_USUtPAx = np.multiply.outer( -1*PA.dot(self.mesh.g), USUtPAx)
		Err = np.tensordot(negPAg_USUtPAx, _ddRdrdr, axes = ([0,1],[1,2]))
		return Err

	def Hess_Egs(self, block=False):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		UtPAx = gU.T.dot(PA.dot(self.mesh.x0))
		PAg = PA.dot(self.mesh.g)
		_dSds = self.dSds()
		
		PAtRU = PA.T.dot(gR.dot(gU))

		d_gEgdS = np.multiply.outer(-1*PAtRU, UtPAx.T)
		d_gEgds = np.tensordot(d_gEgdS, _dSds, axes =([1,2],[0,1]))
		if block:
			d_gEgds = self.BLOCK.T.dot(d_gEgds)

		return d_gEgds

	def Hess_Ers(self, block=False):
		PA = self.mesh.getP().dot(self.mesh.getA())
		gF, gR, gS, gU = self.mesh.getGlobalF()
		UtPAx = gU.T.dot(PA.dot(self.mesh.x0))
		PAg = PA.dot(self.mesh.g)
		_dSds = self.dSds()
		_dRdr = self.dRdr()

		negPAg_U_UtPAx = np.multiply.outer(-1*PAg, np.multiply.outer(gU, UtPAx))
		negPAg_U_UtPAx_dRdr = np.tensordot(negPAg_U_UtPAx, _dRdr, axes=([0,1],[0,1]))
		d_gErds = np.tensordot(negPAg_U_UtPAx_dRdr, _dSds, axes=([0,1],[0,1]))

		return d_gErds

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
			# 	PAx_e = self.PAx[6*i:6*i+6]
			# 	# rotate_around = np.concatenate(
			# 	# 	(np.diag(self.mesh.fixedOnElement[6*i:6*i+2]),
			# 	# 		np.diag(self.mesh.fixedOnElement[6*i+2:6*i+4]),
			# 	# 		np.diag(self.mesh.fixedOnElement[6*i+4:6*i+6])), axis =1)
			# 	# r = np.eye(6) - np.concatenate((rotate_around, rotate_around, rotate_around))
			# 	# # print(r)
			# 	# PAx_e = r.dot(Ax[6*i:6*i+6])

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
			# print(R)
			veca = np.array([1,0])
			vecb = np.dot(R, veca)

			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
			# print("theta", theta)
			#check angle sign
			c, s = np.cos(theta), np.sin(theta)
			R_check = np.array(((c,-s), (s, c)))

			if(np.linalg.norm(R-R_check)<1e-8):
				theta *=-1

			theta_list.append(theta)
			self.mesh.q[3*i] = theta

		return theta_list
	
	def itT(self):
		F = self.mesh.getGlobalF()[0]
		B = self.BLOCK
		Abtg = self.ANTI_BLOCK.T.dot(self.mesh.g)
		
		FPAx = F.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		
		gd = np.concatenate((AtPtFPAx, Abtg))	
		gu = scipy.linalg.lu_solve((self.CholFac, self.Lower), gd)

		self.mesh.g = gu[0:AtPtFPAx.size]
		
		return 1

	def iterate(self, its=1):
		eps = 1e-5
		E0 = self.Energy()
		for i in range(100):
			g = self.itT()
			r = self.itR()
		# print("ARAP grad", np.linalg.norm(self.dEdg()) + np.linalg.norm(self.dEdr()[0]))		

class NeohookeanElastic:

	def __init__(self, imesh, ito_fix = []):
		self.mesh = imesh
		self.f = np.zeros(2*len(self.mesh.T))
		self.v = np.zeros(2*len(self.mesh.V))
		self.M = self.mesh.getMassMatrix()
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix(to_fix = ito_fix)

		youngs = 60000
		poissons = 0.49
		self.mu = youngs/(2+ 2*poissons)
		self.lambd = youngs*poissons/((1+poissons)*(1-2*poissons))
		self.dimensions = 2

		self.grav = np.array([0,-9.81])*10
		self.rho = 1

	def dSds(self):
		#2x2x2 version of dSds (for each element)
		dSdsx = np.array([[1,0],[0,0]])
		dSdsy = np.array([[0,0],[0,1]])
		
		return np.dstack((dSdsx, dSdsy))


	def GravityElementEnergy(self, rho, grav, cag, area, t):
		e = rho*area*grav.dot(cag)
		return e 

	def GravityEnergy(self, iarap = None):
		if(iarap is None):
			print("Why is arap null?")
			exit()

		Eg = 0
		
		Ax = self.mesh.getA().dot(self.mesh.x0)
		CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.g))

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			Eg += self.GravityElementEnergy(self.rho, self.grav, CAg[6*t:6*t+2], area, t)

		return Eg

	def GravityElementForce(self, rho, area, grav, cadgds, t):
		gt = rho*area*np.dot(grav, cadgds)
		return gt

	def GravityForce(self, iarap = None):
		if(iarap is None):
			print("Why is arap null?")
			exit()

		dgds = iarap.Jacobian()[1]

		fg = np.zeros(2*len(self.mesh.T))
		
		Ax = self.mesh.getA().dot(self.mesh.x0)
				
		CAdgds = self.mesh.getC().dot(self.mesh.getA().dot(dgds))
		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			fg -= self.GravityElementForce(self.rho, area, self.grav, CAdgds[6*t:6*t+2, :], t)

		return fg

	def PrinStretchElementEnergy(self, sx, sy):
		#from jernej's paper
		#neohookean energy
		if(sx*sy <0):
			return 1e40
		def f(x):
			return 0.5*self.mu*(x*x -1)
		def h(xy):
			# return abs(math.log((xy - 1)*(xy - 1)))
			return -1*self.mu*math.log(xy) + 0.5*self.lambd*math.log(xy)*math.log(xy)

		E =  f(sx) + f(sy) + h(sx*sy)

		if(E<0):
			print(sx, sy)
			print(f(sx), f(sy), h(sx*sy))
			print(self.mu, self.lambd)
			exit()
		return E

	def PrinStretchElementForce(self, sx, sy):
		if(sx*sy < 0):
			return np.array([1e40, 1e40])
		t1 = (self.lambd*math.log(sx*sy) + self.mu*(sx*sx - 1))/sx
		t2 = (self.lambd*math.log(sx*sy) + self.mu*(sy*sy - 1))/sy
		# t1 = self.mu*sx + 2*sy*(sx*sy -1)
		# t2 = self.mu*sy + 2*sx*(sx*sy -1)
		return np.array([t1, t2])

	def PrinStretchEnergy(self, _q):
		En = 0
		for t in range(len(self.mesh.T)):
			En += self.PrinStretchElementEnergy(_q[3*t + 1], _q[3*t + 2])
		return En

	def PrinStretchForce(self, _q):
		# print(_q)
		force = np.zeros(2*len(self.mesh.T))
		for t in range(len(self.mesh.T)):
			force[2*t:2*t +2] = self.PrinStretchElementForce(_q[3*t + 1], _q[3*t + 2])
		return force

	def Energy(self, iarap, iq):
		e1 =  self.GravityEnergy(iarap = iarap) 
		e2 = self.PrinStretchEnergy(_q=iq)
		return e1+e2

	def Forces(self, iarap, iq):
		f1 =  self.GravityForce(iarap = iarap)
		f2 = self.PrinStretchForce(_q=iq)
		return f1+f2

def FiniteDifferencesARAP():
	eps = 1e-5
	mesh = Mesh(rectangle_mesh(2,2))
	arap = ARAP(mesh, ito_fix=[1,3])
	
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

		print(arap.dEdg())
		print(dEdg)
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
			mesh.q[3*i] += 0.5*eps
			F,R,S,U = mesh.getGlobalF()
			Eleft = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
			mesh.q[3*i] -= 0.5*eps

			mesh.q[3*i] -= 0.5*eps
			F,R,S,U = mesh.getGlobalF()
			Eright = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
			mesh.q[3*i] += 0.5*eps

			dEdr.append((Eleft - Eright)/eps)
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

		print(np.linalg.norm(fake[8:10,:] - real[8:10,:]))
		return real

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
	
	def check_Hessian_dEdgdg():
		real = arap.Hess_Egg()

		Egg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Egg.append([])
			for j in range(len(mesh.g)):
				dg[i] += eps
				dg[j] += eps
				Eij = arap.energy(_g=dg, _R =R0, _S=S0, _U=U0)
				dg[i] -= eps
				dg[j] -= eps

				dg[i] += eps
				Ei = arap.energy(_g=dg, _R =R0, _S=S0, _U=U0)
				dg[i] -= eps

				dg[j] += eps
				Ej = arap.energy(_g=dg, _R =R0, _S=S0, _U=U0)
				dg[j] -= eps

				Egg[i].append((Eij - Ei - Ej + E0)/(eps*eps))
		

		print("Egg")
		print(np.sum(np.array(Egg) - real))

	def check_Hessian_dEdrdg():
		real = arap.Hess_Erg()

		Erg = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Erg.append([])
			for j in range(len(mesh.T)):
				dg[i] += eps
				mesh.q[3*j] += eps
				F,R,S,U = mesh.getGlobalF()
				Eij = arap.energy(_g =dg, _R=R, _S=S0, _U=U0)
				mesh.q[3*j] -= eps
				dg[i] -= eps

				dg[i] += eps
				Ei = arap.energy(_g=dg, _R =R0, _S=S0, _U=U0)
				dg[i] -= eps

				mesh.q[3*j] += eps
				F,R,S,U = mesh.getGlobalF()
				Ej = arap.energy(_g =dg, _R=R, _S=S0, _U=U0)
				mesh.q[3*j] -= eps


				Erg[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print("Erg")
		print(np.sum(np.array(Erg) - real))

	def check_Hessian_dEdrdr():
		real = arap.Hess_Err()

		Err = []
		for i in range(len(mesh.T)):
			Err.append([])
			for j in range(len(mesh.T)):
				mesh.q[3*i] += eps
				mesh.q[3*j] += eps
				F,R,S,U = mesh.getGlobalF()
				Eij = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
				mesh.q[3*j] -= eps
				mesh.q[3*i] -= eps

				mesh.q[3*j] += eps
				F,R,S,U = mesh.getGlobalF()
				Ej = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
				mesh.q[3*j] -= eps

				mesh.q[3*i] += eps
				F,R,S,U = mesh.getGlobalF()
				Ei = arap.energy(_g =mesh.g, _R=R, _S=S0, _U=U0)
				mesh.q[3*i] -= eps

				Err[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print("Err")
		print(np.sum(np.array(Err) - real))
	
	def check_Hessian_dEdrds():
		real = arap.Hess_Ers()

		Ers = []
		for i in range(len(mesh.T)):
			Ers.append([])
			for j in range(len(mesh.T)):
				for k in range(1, 3):
					mesh.q[3*i] += eps
					mesh.q[3*j + k] += eps
					F,R,S,U = mesh.getGlobalF()
					Eij = arap.energy(_g =mesh.g, _R=R, _S=S, _U=U0)
					mesh.q[3*j + k] -= eps
					mesh.q[3*i] -= eps

					mesh.q[3*i] += eps
					F,R,S,U = mesh.getGlobalF()
					Ei = arap.energy(_g =mesh.g, _R=R, _S=S, _U=U0)
					mesh.q[3*i] -= eps

					mesh.q[3*j + k] += eps
					F,R,S,U = mesh.getGlobalF()
					Ej = arap.energy(_g =mesh.g, _R=R, _S=S, _U=U0)
					mesh.q[3*j + k] -= eps

					Ers[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print("Ers")
		print(np.sum(np.array(Ers) - real))

	def check_Hessian_dEdgds():
		real = arap.Hess_Egs()

		Egs = []
		dg = np.zeros(len(mesh.g)) + mesh.g
		for i in range(len(mesh.g)):
			Egs.append([])
			for j in range(len(mesh.T)):
				for k in range(1,3):
					dg[i] += eps
					mesh.q[3*j+k] += eps
					F,R,S,U = mesh.getGlobalF()
					Eij = arap.energy(_g =dg, _R=R0, _S=S, _U=U0)
					mesh.q[3*j+k] -= eps
					dg[i] -= eps

					dg[i] += eps
					F,R,S,U = mesh.getGlobalF()
					Ei = arap.energy(_g=dg, _R =R0, _S=S, _U=U0)
					dg[i] -= eps

					mesh.q[3*j+k] += eps
					F,R,S,U = mesh.getGlobalF()
					Ej = arap.energy(_g =dg, _R=R, _S=S, _U=U0)
					mesh.q[3*j+k] -= eps

					Egs[i].append((Eij - Ei - Ej + E0)/(eps*eps))

		print("Egs")
		print(np.sum(np.array(Egs) - real))

	def check_dgds():
		Jac, real1, real2 = arap.Jacobian()

		dgds = []
		drds = []
		g0 = np.zeros(len(mesh.g)) + mesh.g
		r0 = np.array([mesh.q[3*ii] for ii in range(len(mesh.T))]) 
		q0 = np.zeros(len(mesh.q)) + mesh.q
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.g = np.zeros(len(mesh.g)) + g0

				mesh.q[3*i + j] += 0.5*eps 
				arap.iterate()
				r1 = np.array([mesh.q[3*ii] for ii in range(len(mesh.T))])
				# print("r1",r1)
				dgds_left =mesh.g + np.zeros(len(mesh.g))
				drds_left = r1
				# print("g", mesh.g)
				# print("CAg", mesh.getC().dot(mesh.getA().dot(mesh.g)))
				# print("disc.", mesh.getDiscontinuousVT()[0])
				mesh.q[3*i + j] -= 0.5*eps
				arap.iterate()

				mesh.q[3*i + j] -= 0.5*eps 
				arap.iterate()
				r2 = np.array([mesh.q[3*ii] for ii in range(len(mesh.T))])
				dgds_right =mesh.g + np.zeros(len(mesh.g))
				drds_right = r2
				mesh.q[3*i + j] += 0.5*eps
				arap.iterate()
				# print(r2)
				# print((drds_left - r0)/(0.5*eps))
				dgds.append((dgds_left - dgds_right)/(eps))
				drds.append((drds_left - drds_right)/(eps))
				# exit()
				

		print("FD")
		print(np.array(dgds).T)
		print(np.array(drds).T)
		print("")
		print("real")
		print(real1)
		print(real2)
		print("DIFF")
		print("dgds:", np.linalg.norm(real1 - np.array(dgds).T))
		print("drds:", np.linalg.norm(real2 - np.array(drds).T))

	def test():
		mesh.q[1] = 1.2
		arap.iterate(its=4)
		print(mesh.g)
		print(mesh.getDiscontinuousVT()[0])

	# check_dEdg()
	# check_dEds()
	# check_dEdr()
	# left = check_d_gradEgrdg()
	# right = check_d_gradEgrdr()
	# rhs = -1*check_d_gradEgrds()

	# check_Hessian_dEdgdg()
	# check_Hessian_dEdrdg()
	# check_Hessian_dEdrdr()
	# check_Hessian_dEdrds()
	# check_Hessian_dEdgds()
	check_dgds()
	# test()

# FiniteDifferencesARAP()

def FiniteDifferencesElasticity():
	eps = 1e-4
	mesh = Mesh(rectangle_mesh(2,2) )
	ne = NeohookeanElastic(imesh = mesh,ito_fix=[1,3])
	arap = ARAP(imesh=mesh, ito_fix=[1,3])
	

	def check_PrinStretchForce():
		e0 = ne.PrinStretchEnergy(_q = mesh.q)
		real = ne.PrinStretchForce(_q = mesh.q)
		print("e0", e0)
		dEds = []
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.q[3*i+j] += eps
				left = ne.PrinStretchEnergy(_q=mesh.q)
				mesh.q[3*i+j] -= eps

				mesh.q[3*i+j] -= eps
				right = ne.PrinStretchEnergy(_q=mesh.q)
				mesh.q[3*i+j] += eps

				dEds.append((left - right)/(2*eps))

		print("real", real)
		print("fake", dEds)
		print("Diff", np.sum(real - np.array(dEds)))

	def check_gravityForce():
		e0 = ne.GravityEnergy(iarap=arap)
		print("E0", e0)
		arap.iterate()
		
		real = -1*ne.GravityForce(iarap = arap)

		dEgds = []
		for i in range(len(mesh.T)):
			for j in range(1,3):
				mesh.g = np.zeros(len(mesh.g)) + mesh.x0
				mesh.q[3*i+j] += eps
				arap.iterate()
				e1 = ne.GravityEnergy(iarap=arap)
				dEgds.append((e1 - e0)/eps)
				mesh.q[3*i+j] -= eps
				arap.iterate()

		print("real", real)
		print("fake", dEgds)
		print("Diff", np.sum(real - np.array(dEgds)))

	def test():
		ne.Energy(iarap=arap, iq=mesh.q)
		ne.Forces(iarap=arap, iq=mesh.q)
	# check_PrinStretchForce()
	# check_gravityForce()
	# test()

# FiniteDifferencesElasticity()

class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 0
		self.timestep = 0.001
		self.mesh = imesh
		self.arap = iarap 
		self.elastic = ielastic 
		self.adder = 0.01
		# self.set_random_strain()


	def set_strain(self):
		for i in range(len(self.mesh.T)):
			pass
			# self.mesh.q[3*i+1] =1.01+ np.sin(self.time)
			# self.mesh.q[3*i +2] = 1.01 + np.sin(self.time)

	def set_random_strain(self):
		for i in range(len(self.mesh.T)):
			# self.mesh.q[3*i + 1] = 1.01 + np.random.uniform(0,2)
			self.mesh.q[3*i + 2] = 1.0 + np.random.uniform(0.001,0.1)

	def iterate(self):
		# self.set_strain()
		# self.arap.iterate(its=10)
		# print(self.mesh.g)
		if(self.time%5==0):
			self.adder *=-1
		self.mesh.g[2*self.mesh.fixed[0]+1] += self.adder
		self.mesh.g[2*self.mesh.fixed[1]+1] += self.adder
		self.mesh.g[2*self.mesh.fixed[2]+1] += self.adder
		# self.mesh.g[2*self.mesh.fixed[3]] += self.adder
		# self.mesh.g[2*self.mesh.fixed[4]] += self.adder

		# self.mesh.g[2*self.mesh.fixed[0]+1] += 3*self.adder
		# self.mesh.g[2*self.mesh.fixed[1]+1] += 2*self.adder
		# self.mesh.g[2*self.mesh.fixed[2]+1] += self.adder
		# self.mesh.g[2*self.mesh.fixed[3]+1] += self.adder
		# self.mesh.g[2*self.mesh.fixed[4]+1] += self.adder
		self.time += 1

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

	def solve(self):
		self.iterate()
		self.arap.iterate()
		s0 = []
		for i in range(len(self.mesh.T)):
			s0.append(self.mesh.q[3*i +1])
			s0.append(self.mesh.q[3*i +2])

		def energy(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			self.arap.iterate()
			E_arap = self.arap.Energy()
			E_elastic = self.elastic.Energy(iarap =self.arap, iq=self.mesh.q)
			print("E", 1e10*E_arap, E_elastic)
			return 1e10*E_arap + E_elastic

		def jacobian(s):
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]

			# self.arap.iterate()
			# print("g", self.mesh.g)

			J_arap, dgds, drds = self.arap.Jacobian()
			J_elastic = self.elastic.Forces(iarap=self.arap, iq = self.mesh.q)

			# print("Jac", J_arap, J_elastic)
			return 1e10*J_arap + J_elastic
		
		# res = scipy.optimize.minimize(energy, s0, method='Nelder-Mead',  options={'gtol': 1e-6, 'disp': True})
		res = scipy.optimize.minimize(energy, s0, method='BFGS', jac=jacobian, options={'gtol': 1e-6, 'disp': True, 'eps':1e-08})
		for i in range(len(res.x)/2):
			self.mesh.q[3*i+1] = res.x[2*i]
			self.mesh.q[3*i+2] = res.x[2*i+1]
		
		print("s", res.x)
		print("g", self.mesh.g)

def display():
	mesh = Mesh(rectangle_mesh(3,3))
	# to_fix = [0,1,2,3,4,20,21,22,23,24]
	to_fix = [0,3,6,2,5,8]
	# to_fix = [0,2,1,3]
	# to_fix = [10*i for i in range(10)]
	neoh =NeohookeanElastic(imesh=mesh, ito_fix=to_fix)
	arap = ARAP(imesh=mesh, ito_fix = to_fix)
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

		# print(cag)
		viewer.data.add_points(igl.eigen.MatrixXd(np.array(cag)), green)
		
		time_integrator.solve()
		return True

	key_down(viewer, 'a', 1)
	viewer.callback_key_down= key_down
	viewer.core.is_animating = False
	viewer.launch()

display()