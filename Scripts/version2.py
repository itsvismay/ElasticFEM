import numpy as np
import math
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from collections import defaultdict
from sets import Set
import datetime
import numdifftools as nd
import random
import sys, os
import cProfile
from scipy import sparse
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.8f}'.format(x)})

temp_png = os.path.join(os.getcwd(),"out.png")

#helpers
def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y, step=1):
	V = []
	for i in range(0,x+1):
		for j in range(0,y+1):
			V.append([step*i, step*j])
	return V, Delaunay(V).simplices, None

def torus_mesh(r1, r2, r3, step):
	V = []
	T = []
	for theta in range(0, 6):
		angle = theta*np.pi/3
		if(angle<=np.pi):
			V.append([step*r1*np.cos(angle), step*r1*np.sin(angle)])
			V.append([step*r2*np.cos(angle), step*r2*np.sin(angle)])
			# V.append([step*r3*np.cos(angle), step*r3*np.sin(angle)])
	
	V.append([0,0])
	for e in Delaunay(V).simplices:
		if e[0]!=len(V)-1 and e[1]!=len(V)-1 and e[2]!=len(V)-1 and get_area(V[e[0]], V[e[1]], V[e[2]])<5:
			T.append([e[0], e[1], e[2]])
			# T.append(list(e))
	
	return np.array(V[:len(V)-1]), np.array(T), None

def triangle_mesh():
	V = [[0,0], [1,0], [1,1]]
	T = [[0,1,2]]
	return V, T, [0]

def featherize(x, y, step=1):
	V,T,U = rectangle_mesh(x, y, step)
	# V,T, U = torus_mesh(5, 4, 3, step)

	half_x = step*(x)/2.0
	half_y = step*(y)/2.0
	u = []
	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[0]<half_x):
			u.append(3*np.pi/4)
		else:
			u.append(np.pi/4)

	return V, T, u

def get_min_max(iV,a):
	eps = 1e-5
	mov = []
	miny = np.amin(iV, axis=0)[a]
	maxy = np.amax(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - miny) < eps):
			mov.append(i)

		if(abs(iV[i][a] - maxy)<eps):
			mov.append(i)

	return mov

class Mesh:
	#class vars

	def __init__(self, iVTU, ito_fix=[]):
		#object vars
		self.fixed = ito_fix
		self.V = np.array(iVTU[0])
		self.T = iVTU[1]

		self.fixedOnElement = None
		self.P = None
		self.A = None
		self.C = None
		self.Mass = None
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)+np.ravel(self.V)
		self.u = iVTU[2] if iVTU[2] is not None else np.zeros(len(self.T))
		self.q = np.zeros(len(self.T)*(1+2)) #theta, sx, sy

		self.red_r = None
		self.r_element_cluster_map = None
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters()

		self.GF = np.zeros((6*len(self.T), 6*len(self.T)))
		self.GR = np.zeros((6*len(self.T), 6*len(self.T)))
		self.GS = np.zeros((6*len(self.T), 6*len(self.T)))
		self.GU = np.zeros((6*len(self.T), 6*len(self.T)))

		#set initial strains
		for i in range(len(self.T)):
			self.q[3*i + 1] = 1
			self.q[3*i + 2] = 1


		self.getGlobalF(updateR = True, updateS = True, updateU=True)

	def setupRotClusters(self):
		print("Setting up rotation clusters")
		# of rotation clusters
		t_set = Set([i for i in range(len(self.T))])
		nrc = 2
		self.red_r = np.zeros(nrc)
		self.r_element_cluster_map = np.zeros(len(self.T), dtype = int)
		
		centroids = self.getC().dot(self.getA().dot(self.x0))
		minx = np.amin(self.V, axis=0)[0]
		maxx = np.amax(self.V, axis=0)[0]
		for i in range(len(self.T)):

			if(centroids[6*i]<=(maxx + minx)/2.0):
				self.r_element_cluster_map[i] = 0
				self.r_cluster_element_map[0].append(i)
			else:
				self.r_element_cluster_map[i] = 1
				self.r_cluster_element_map[1].append(i)

		self.RotationBLOCK = []
		for i in range(len(self.red_r)):
			fixed = t_set.difference(Set(self.r_cluster_element_map[i]))
			
			b = sparse.kron(np.delete(np.eye(len(self.T)), list(fixed), axis=1), sparse.eye(6))
			self.RotationBLOCK.append(b)
		print("Done setting up rotation clusters \n")
		return

	def createBlockingMatrix(self):
		print("Creating Block and AB matrix")
		onVerts = np.zeros(len(self.V))
		onVerts[self.fixed] = 1
		self.fixedOnElement = self.getA().dot(np.kron(onVerts, np.ones(2)))
		if(len(self.fixed) == len(self.V)):
			return np.array([[]]), sparse.eye(2*len(self.V)).tocsc()
		b = np.kron(np.delete(np.eye(len(self.V)), self.fixed, axis =1), np.eye(2))

		ab = np.zeros(len(self.V))
		ab[self.fixed] = 1
		to_reset = [i for i in range(len(ab)) if ab[i]==0]

		if (len(self.fixed) == 0):
			return sparse.csc_matris(b), sparse.csc_matrix(np.zeros((2*len(self.V), (2*len(self.V)))))

		anti_b = np.kron(np.delete(np.eye(len(self.V)), to_reset, axis =1), np.eye(2))

		print("Done with Blocking matrix\n")
		return sparse.csc_matrix(b), sparse.csc_matrix(anti_b)

	def fixed_min_axis(self, a):
		eps = 1e-5
		mov = []
		miny = np.amin(self.V[self.fixed], axis=0)[a]
		
		for e in self.fixed:
			if(abs(self.V[e][a] - miny) < eps):
				mov.append(e)
		return mov

	def fixed_max_axis(self, a):
		eps = 1e-5
		mov = []
		miny = np.amax(self.V[self.fixed], axis=0)[a]
		
		for e in self.fixed:
			if(abs(self.V[e][a] - miny) < eps):
				mov.append(e)
		return mov

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
		alpha = self.u[ind]
		cU, sU = np.cos(alpha), np.sin(alpha)
		U = np.array(((cU,-sU), (sU, cU)))
		return U

	def getR(self, ind):
		# theta = self.q[3*ind]
		theta = self.red_r[self.r_element_cluster_map[ind]]
		c, s = np.cos(theta), np.sin(theta)
		R = np.array(((c,-s), (s, c)))
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

	def getGlobalF(self, updateR = True, updateS = True, updateU = False):
		if(updateR):
			self.GR.fill(0)
		if(updateS):
			self.GS.fill(0)
		for i in range(len(self.T)):
			if updateR:
				r = self.getR(i)
				self.GR[6*i:6*i+6, 6*i:6*i+6] = np.kron(np.eye(3), r)
			if updateS:
				s = self.getS(i)
				self.GS[6*i:6*i+6, 6*i:6*i+6] = np.kron(np.eye(3), s)
			if updateU:
				u = self.getU(i)
				self.GU[6*i:6*i+6, 6*i:6*i+6] = np.kron(np.eye(3), u)
		
		if(updateR or updateS or updateU):
			self.GF = self.GR.dot(self.GU.dot(self.GS.dot(self.GU.T)))

	def getDiscontinuousVT(self):
		C = self.getC()
		CAg = C.dot(self.getA().dot(self.g))
		Ax = self.getA().dot(self.x0)
		new = self.GF.dot(self.getP().dot(Ax)) + CAg

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

	def __init__(self, imesh):
		self.mesh = imesh
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix()
		#these are the fixed vertices which stay constant
		A = self.mesh.getA()
		P = self.mesh.getP()
		B = self.BLOCK
		C = self.ANTI_BLOCK.T
		self.PAx = P.dot(A.dot(self.mesh.x0))
		self.AtPtPA = A.T.dot(P.T.dot(P.dot(A)))
		

		self.DSDs = None

		#LU inverse
		col2 = sparse.vstack((C.T, sparse.csc_matrix((C.shape[0], C.shape[0]))))
		col1 = sparse.vstack((self.AtPtPA, C))
		KKT = sparse.hstack((col1, col2))

		self.CholFac = scipy.sparse.linalg.splu(KKT.tocsc())

	def energy(self, _g, _R, _S, _U):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(_g))
		FPAx = _R.dot(_U.dot(_S.dot(_U.T.dot(self.PAx))))
		return 0.5*(np.dot(PAg - FPAx, PAg - FPAx))

	def Energy(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		FPAx = self.mesh.GF.dot(self.PAx)
		en = 0.5*(np.dot(PAg - FPAx, PAg - FPAx))
		return en

	def Jacobian(self, block = False, kkt= True, useSparse=True):
		a = datetime.datetime.now()
		self.mesh.getGlobalF()
		b = datetime.datetime.now()
		Egg, Erg, Err, Egs, Ers = self.Hessians(useSparse=useSparse)
		c = datetime.datetime.now()
		lhs_left = np.concatenate((Egg, Erg.T))
		lhs_right = np.concatenate((Erg , Err))

		rhs = -1*np.concatenate((Egs, Ers))


		
		#Constraining Rotation
		# R = np.eye(len(self.mesh.T))
		R = np.array([[] for i in self.mesh.red_r]).T
		# R = np.array([[0,1]])
		d = datetime.datetime.now()
		#NEW KKT SOLVE
		if kkt:
			C = self.ANTI_BLOCK.T
			g_size = C.shape[1]
			gb_size = C.shape[0]
			r_size = R.shape[1]
			rb_size = R.shape[0]

			if useSparse:
				if rb_size>0:
					col1 = sparse.vstack((lhs_left, np.vstack((C.toarray(), np.zeros((rb_size, g_size))))))
					# print("col", col1.shape, col1.nnz)
					col2 = sparse.vstack((lhs_right, np.vstack((np.zeros((gb_size, r_size)), R))))
					# print("col", col2.shape, col2.nnz)
					col3 = sparse.vstack(( C.T, sparse.vstack((sparse.csc_matrix((r_size, gb_size)), np.vstack((np.zeros((gb_size, gb_size)), np.zeros((rb_size, gb_size))))))) )
					# print("col", col3.shape, col3.nnz)
					col4 = sparse.vstack(( np.vstack((np.zeros((g_size, rb_size)), R.T)), np.vstack((np.zeros((gb_size, rb_size)), np.zeros((rb_size, rb_size)))) ))
					jacKKT = sparse.hstack((col1, col2, col3, col4))
				else:
					col1 = sparse.vstack((lhs_left, C))
					# print("col", col1.shape, col1.nnz)
					col2 = sparse.vstack((lhs_right, sparse.csc_matrix((gb_size, r_size))))
					# print("col", col2.shape, col2.nnz)
					col3 = sparse.vstack(( C.T, 
						sparse.vstack((sparse.csc_matrix((r_size, gb_size)), 
									sparse.csc_matrix((gb_size, gb_size))))))
					# print("col", col3.shape, col3.nnz)
					jacKKT = sparse.hstack((col1, col2, col3))

				# print(jacKKT.toarray())
				# print("\n\n")
				KKT_constrains = np.vstack((rhs, np.zeros((gb_size+rb_size, rhs.shape[1]))))
				# print(KKT_constrains)
				# exit()
				# print(sparse.issparse(jacKKT), sparse.issparse(KKT_constrains), jacKKT.shape, jacKKT.nnz)
				jacChol = scipy.sparse.linalg.splu(jacKKT.tocsc())
				Jac_s = jacChol.solve(KKT_constrains)
			
			else:
				col1 = np.concatenate((lhs_left, np.concatenate((C.toarray(), np.zeros((rb_size, g_size))))))
				# print("col", col1.shape)
				col2 = np.concatenate((lhs_right, np.concatenate((np.zeros((gb_size, r_size)), R))))
				# print("col", col2.shape)
				col3 = np.concatenate((C.T.toarray(), np.concatenate((np.zeros((r_size, gb_size)), np.concatenate((np.zeros((gb_size, gb_size)), np.zeros((rb_size, gb_size))))))) )
				# print("col", col3.shape)
				col4 = np.concatenate(( np.concatenate((np.zeros((g_size, rb_size)), R.T)),
					np.concatenate((np.zeros((gb_size, rb_size)), np.zeros((rb_size, rb_size)))) ))
				# print("col", col4.shape)
				jacKKT = np.hstack((col1, col2, col3, col4))
				jacChol, jacLower = scipy.linalg.lu_factor(jacKKT)
				KKT_constrains = np.concatenate((rhs, np.zeros((gb_size+rb_size, rhs.shape[1]))))

				Jac_s = scipy.linalg.lu_solve((jacChol, jacLower), KKT_constrains)

			results = Jac_s[0:rhs.shape[0], :]
		

		e = datetime.datetime.now()
		dgds = results[0:lhs_left.shape[1],:]
		drds = results[lhs_left.shape[1]:,:]
		f = datetime.datetime.now()
		Eg,Er,Es = self.Gradients()
		aa = datetime.datetime.now()
		dEds = np.matmul(Eg, dgds) + np.matmul(Er, drds) + Es
		bb = datetime.datetime.now()
		print("Jac time: ", (b-a).microseconds, (c-b).microseconds, (d-c).microseconds, (e-d).microseconds, (f-e).microseconds, (aa-f).microseconds, (bb-aa).microseconds)
		return dEds, dgds, drds

	def Hessians(self, useSparse=True):		
		PA = self.mesh.getP().dot(self.mesh.getA())
		PAg = PA.dot(self.mesh.g)
		USUt = self.mesh.GU.dot(self.mesh.GS.dot(self.mesh.GU.T))
		USUtPAx = USUt.dot(self.PAx)
		UtPAx = self.mesh.GU.T.dot(PA.dot(self.mesh.x0))
		r_size = len(self.mesh.red_r)
		g_size = len(self.mesh.g)
		s_size = 2*len(self.mesh.T)
		DR = self.sparseDRdr()
		DS = self.sparseDSds()
		
		Egg = self.AtPtPA

		sample = np.multiply.outer(-1*PA.T, USUtPAx.T)
		if not useSparse:
			print("SPARSE IS FALSE")
			Erg = np.tensordot(sample, self.dRdr(), axes=([1,2], [0,1]))
		else:
			Erg = np.zeros((g_size, r_size))

			for j in range(Erg.shape[1]):
				sp = DR[j]
				for i in range(Erg.shape[0]):
					if sp.nnz>0:
						Erg[i,j] = sp.multiply(sample[i,:,:]).sum()

		
		negPAg_USUtPAx = np.multiply.outer( -1*PA.dot(self.mesh.g), USUtPAx)
		if not useSparse:
			Err = np.tensordot(negPAg_USUtPAx, self.d2Rdr2(), axes = ([0,1],[1,2]))
		else:
			DDR = self.sparseDDRdrdr()
			Err = np.zeros((r_size, r_size))
			for i in range(Err.shape[0]):
				for j in range(Err.shape[1]):
					spD = DDR[i][j]
					if spD.nnz>0:
						Err[i,j] = spD.multiply(negPAg_USUtPAx).sum()
		

		PAtRU = PA.T.dot(self.mesh.GR.dot(self.mesh.GU))
		d_gEgdS = np.multiply.outer(-1*PAtRU, UtPAx.T)
		
		if not useSparse:
			Egs = np.tensordot(d_gEgdS, self.dSds(), axes =([1,2],[0,1]))
		else:
			Egs = np.zeros((g_size, s_size))
			for j in range(Egs.shape[1]):
				sp = DS[j]
				for i in range(Egs.shape[0]):
					if sp.nnz>0:
						Egs[i,j] = sp.multiply(d_gEgdS[i,:,:]).sum()
		
	
		# print(self.mesh.GU)
		
		if not useSparse:
			#Original math with rank 4 tensor
			# first = np.multiply.outer(-PAg, np.multiply.outer(self.mesh.GU, UtPAx))
			# second = np.tensordot(first, self.dRdr(), axes=([0,1],[0,1]))
			# Ers = np.tensordot(second, self.dSds(), axes=([0,1],[0,1]))

			first = np.multiply.outer(-PAg, self.mesh.GU)
			mid = np.tensordot(first, self.dRdr(), axes=([0,1],[0,1]))
			second = np.multiply.outer(UtPAx, mid)
			Ers = np.tensordot(second, self.dSds(), axes=([0,1],[0,1]))

		else:
			Ers = np.zeros((r_size, s_size))
			first = np.multiply.outer(-PAg, self.mesh.GU)
			mid = np.zeros((first.shape[0], r_size))

			for j in range(mid.shape[1]):
				spR = DR[j]
				for i in range(mid.shape[0]):
					if spR.nnz>0:
						mid[i,j] = spR.multiply(first[:,:,i]).sum()

			second = np.multiply.outer(UtPAx, mid)
			for j in range(Ers.shape[1]):
				spS = DS[j]
				for i in range(Ers.shape[0]):
					if spS.nnz>0:
						Ers[i,j] = spS.multiply(second[:,:,i]).sum()

		
		return Egg, Erg, Err, Egs, Ers

	def Gradients(self):
		PA = self.mesh.getP().dot(self.mesh.getA())
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		USUt = self.mesh.GU.dot(self.mesh.GS.dot(self.mesh.GU.T))
		# _dRdr = self.dRdr()
		# _dSds = self.dSds()#rank 3 tensor
		DR = self.sparseDRdr()
		DS = self.sparseDSds()

		dEdR = -1*np.multiply.outer(PAg, USUt.dot(self.PAx))
		# dEdr = np.tensordot(dEdR, _dRdr, axes = ([0,1],[0,1]))
		dEdr = np.zeros(len(self.mesh.red_r))
		for i in range(len(dEdr)):
			dEdr[i] = DR[i].multiply(dEdR).sum()
	


		FPAx = self.mesh.GF.dot(self.PAx)
		AtPtPAg = self.AtPtPA.dot(self.mesh.g)
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		dEdg = AtPtPAg - AtPtFPAx

		UtPAx = self.mesh.GU.T.dot(self.PAx)
		RU = self.mesh.GR.dot(self.mesh.GU)
		dEdS =  np.multiply.outer(self.mesh.GS.dot(UtPAx), UtPAx) - np.multiply.outer(np.dot(RU.T, PAg), UtPAx)
		# dEds = np.tensordot(dEdS, _dSds, axes = ([0,1], [0,1]))
		dEds = np.zeros(2*len(self.mesh.T))
		for i in range(len(dEds)):
			dEds[i] = DS[i].multiply(dEdS).sum()

		
		return dEdg, dEdr, dEds

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

	def sparseDRdr(self):
		DR = []

		for t in range(len(self.mesh.red_r)):
			c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
			dRdr_e = np.array(((-s,-c), (c, -s)))
			B = self.mesh.RotationBLOCK[t]
			blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), dRdr_e)
	
			placeHolder = B.dot(blocked.dot(B.T))
			DR.append(placeHolder)

		return DR

	def sparseDDRdrdr(self):
		D2R = [[] for i in range(len(self.mesh.red_r))]

		for t in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[t]
			for r in range(len(self.mesh.red_r)):
				if(t==r):
					c, s = np.cos(self.mesh.red_r[t]), np.sin(self.mesh.red_r[t])
					ddR_e = np.array(((-c,s), (-s, -c)))
					blocked = sparse.kron(sparse.eye(3*len(self.mesh.r_cluster_element_map[t])), ddR_e)
					placeHolder = B.dot(blocked.dot(B.T))
				else:
					placeHolder = sparse.csc_matrix(np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T))))

				D2R[t].append(placeHolder)
		return D2R

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
		
	def sparseDSds(self):
		if self.DSDs == None:
			DS = []
			for t in range(len(self.mesh.T)):
				dSdsx = np.array([[1,0],[0,0]])
				dSdsy = np.array([[0,0],[0,1]])

				dSdsx_e = sparse.kron(sparse.eye(3), dSdsx)
				dSdsy_e = sparse.kron(sparse.eye(3), dSdsy)
				gdSdsx = sparse.csc_matrix(np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T)))) 
				gdSdsy = sparse.csc_matrix(np.zeros((6*len(self.mesh.T), 6*len(self.mesh.T))))
				gdSdsx[6*t:6*t+6, 6*t:6*t+6] = dSdsx_e
				gdSdsy[6*t:6*t+6, 6*t:6*t+6] = dSdsy_e

				DS.append(gdSdsx)
				DS.append(gdSdsy)
			self.DSDs = DS
		return self.DSDs

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

	def dEdr(self):
		g, r, s = self.Gradients()
		return None, r

	def dEdg(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		FPAx = self.mesh.GF.dot(self.PAx)
		AtPtPAg = self.AtPtPA.dot(self.mesh.g)
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		dEdg = AtPtPAg - AtPtFPAx
		return dEdg

	def dEds(self):
		g, r, s = self.Gradients()
		return None, s

	def Hess_Egg(self, block =False):
		return self.Hessians()[0]

	def Hess_Erg(self, block=False):
		return self.Hessians()[1]

	def Hess_Err(self, block=False):
		return self.Hessians()[2]

	def Hess_Egs(self, block=False):
		return self.Hessians()[3]

	def Hess_Ers(self, block=False):
		return self.Hessians()[4]

	def itR(self, GU, GS):
		theta_list = []
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		Ax = self.mesh.getA().dot(self.mesh.x0)

		for i in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[i]
			PAx_e = B.T.dot(self.PAx)
			
			PAg_e = B.T.dot(PAg)
			Ue = B.T.dot(GU.dot(B))
			Se = B.T.dot(GS.dot(B))
			

			USUPAx = Ue.dot(Se.dot(Ue.T.dot(PAx_e.T)))
			m_PAg = np.reshape(PAg_e, (len(PAg_e)/2,2))
			m_USUPAx = np.reshape(USUPAx, (len(PAg_e)/2,2))
			

			F = np.matmul(m_PAg.T,m_USUPAx)

			u, s, vt = np.linalg.svd(F, full_matrices=True)
			R = np.matmul(vt.T, u.T)
			
			veca = np.array([1,0])
			vecb = np.dot(R, veca)

			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))

			#check angle sign
			c, s = np.cos(theta), np.sin(theta)
			R_check = np.array(((c,-s), (s, c)))

			if(np.linalg.norm(R-R_check)<1e-8):
				theta *=-1

			theta_list.append(theta)
			self.mesh.red_r[i] = theta
		return theta_list
	
	def itT(self):
		Abtg = self.ANTI_BLOCK.T.dot(self.mesh.g)
		
		FPAx = self.mesh.GF.dot(self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.x0)))
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		
		gd = np.concatenate((AtPtFPAx, Abtg))
		gu = self.CholFac.solve(gd)
		self.mesh.g = gu[0:AtPtFPAx.size]

		return 1

	def iterate(self, its=100):
		Eg0 = self.dEdg()
		GU = sparse.csc_matrix(self.mesh.GU)
		GS = sparse.csc_matrix(self.mesh.GS)
		for i in range(its):
			g = self.itT()
			r = self.itR(GU, GS)
			self.mesh.getGlobalF(updateR=True, updateS=False, updateU=False)
			Eg = self.dEdg()
			# print("i", i,np.linalg.norm(Eg-Eg0))
			if(1e-10 > np.linalg.norm(Eg-Eg0)):
				print("ARAP converged", np.linalg.norm(Eg))	
				return
			Eg0 = Eg

class NeohookeanElastic:

	def __init__(self, imesh):
		self.mesh = imesh
		self.f = np.zeros(2*len(self.mesh.T))
		self.v = np.zeros(2*len(self.mesh.V))
		#self.M = self.mesh.getMassMatrix()
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix()

		youngs = 60000
		poissons = 0.49
		self.mu = youngs/(2+ 2*poissons)
		self.lambd = youngs*poissons/((1+poissons)*(1-2*poissons))
		self.dimensions = 2

		self.grav = np.array([0,-9.81])
		self.rho = 10

	def dSds(self):
		#2x2x2 version of dSds (for each element)
		dSdsx = np.array([[1,0],[0,0]])
		dSdsy = np.array([[0,0],[0,1]])
		
		return np.dstack((dSdsx, dSdsy))

	def GravityElementEnergy(self, rho, grav, cag, area, t):
		e = rho*area*grav.dot(cag)
		return e 

	def GravityEnergy(self):
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

	def GravityForce(self, dgds):
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

	def Energy(self, iq):
		e2 = self.PrinStretchEnergy(_q=iq)
		e1 = -1*self.GravityEnergy() 
		return e2 + e1

	def Forces(self, iq, idgds):
		f2 = self.PrinStretchForce(_q=iq)
		if idgds is None:
			return f2
		f1 =  self.GravityForce(idgds)
		return f2 + f1

class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 0
		self.timestep = 0.001
		self.mesh = imesh
		self.arap = iarap 
		self.elastic = ielastic 
		self.adder = 3e-2
		# self.set_random_strain()
		self.mov = np.array(self.mesh.fixed_min_axis(1))

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

		if(self.time%20 == 0):
			self.adder *= -1

		self.mesh.g[2*self.mov+1] -= self.adder


		self.time += 1

	def solve(self):
		self.iterate()
		s0 = []
		for i in range(len(self.mesh.T)):
			s0.append(self.mesh.q[3*i +1])
			s0.append(self.mesh.q[3*i +2])

		def energy(s):
			# a = datetime.datetime.now()
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]
			# b = datetime.datetime.now()
			self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)
			# c = datetime.datetime.now()
			self.arap.iterate()
			# d = datetime.datetime.now()
			E_arap =  1e1*self.arap.Energy()
			# e = datetime.datetime.now()
			E_elastic =  self.elastic.Energy(iq=self.mesh.q)
			# f = datetime.datetime.now()
			print("E", E_arap, E_elastic)
			# print("Solve En time: ", (b-a).microseconds, (c-b).microseconds, (d-c).microseconds, (e-d).microseconds, (f-e).microseconds)

			return E_arap + E_elastic

		def jacobian(s):
			# a = datetime.datetime.now()
			for i in range(len(s)/2):
				self.mesh.q[3*i + 1] = s[2*i]
				self.mesh.q[3*i + 2] = s[2*i +1]
			# b = datetime.datetime.now()
			# c = datetime.datetime.now()
			dgds = None
			J_arap, dgds, drds = self.arap.Jacobian()

			# d = datetime.datetime.now()
			J_elastic = self.elastic.Forces(iq = self.mesh.q, idgds=dgds)
			# e = datetime.datetime.now()
			# print("Solve Jac time: ", (b-a).microseconds, (c-b).microseconds, (d-c).microseconds, (e-d).microseconds)
			return  J_elastic + 1e1*J_arap
		
		res = scipy.optimize.minimize(energy, s0, method='BFGS', jac=jacobian, options={'gtol': 1e-6, 'disp': True, 'eps':1e-08})
		for i in range(len(res.x)/2):
			self.mesh.q[3*i+1] = res.x[2*i]
			self.mesh.q[3*i+2] = res.x[2*i+1]
		
		print("r", self.mesh.red_r)
		print("s", res.x)
		print("g", self.mesh.g)

def display():
	iV, iT, iU = featherize(1,3,.1)
	to_fix = get_min_max(iV,1)
	
	mesh = Mesh((iV,iT, iU),ito_fix=to_fix)

	neoh =NeohookeanElastic(imesh=mesh )
	arap = ARAP(imesh=mesh)
	time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)

	viewer = igl.viewer.Viewer()

	tempR = igl.eigen.MatrixXuc(1280, 800)
	tempG = igl.eigen.MatrixXuc(1280, 800)
	tempB = igl.eigen.MatrixXuc(1280, 800)
	tempA = igl.eigen.MatrixXuc(1280, 800)

	def key_down(viewer, aaa, bbb):
		viewer.data.clear()
		# if(time_integrator.time>30):
		# 	exit()
		time_integrator.iterate()
		arap.iterate()
		# print(mesh.red_r)
		# time_integrator.solve()

		
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


		FIXED = []
		for i in range(len(mesh.fixed)):
			FIXED.append(mesh.g[2*mesh.fixed[i]:2*mesh.fixed[i]+2])

		viewer.data.add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)

		#centroids
		CAg = mesh.getC().dot(mesh.getA().dot(mesh.g))
		cag = []
		for i in range(len(mesh.T)):
			cag.append(CAg[6*i:6*i+2])
			C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
			U = 0.01*mesh.getU(i).transpose()+C
			viewer.data.add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), red)

		# print(cag)
		viewer.data.add_points(igl.eigen.MatrixXd(np.array(cag)), green)
		
		if (time_integrator.time>1):
			viewer.core.draw_buffer(viewer.data, viewer.opengl, False, tempR, tempG, tempB, tempA)
			igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(time_integrator.time)+".png")
			# pass
		return True

	# for clicks in range(40):
	key_down(viewer, 'a', 123)
	viewer.callback_key_down = key_down
	viewer.core.is_animating = False
	viewer.launch()
# display()

def headless():
	iV, iT, iU = rectangle_mesh(9,9,1)
	# iV, iT, iU = triangle_mesh()
	to_fix = get_min_max(iV,1)
	
	mesh = Mesh((iV,iT, iU),ito_fix=to_fix)
	mesh.q[0] = 0
	neoh =NeohookeanElastic(imesh=mesh )
	arap = ARAP(imesh=mesh)
	a = datetime.datetime.now()
	# print(arap.dRdr()[:,:,0])
	# print(arap.sparseDRdr()[0])
	# print(arap.d2Rdr2()[1,:,:,1])
	# print(arap.sparseDDRdrdr()[1][1])

	# arap.Hessians()
	b = datetime.datetime.now()
	print("Time", (b-a).microseconds)
# headless()