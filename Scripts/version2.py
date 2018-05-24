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
def generate_bbw_matrix():
	# List of boundary indices (aka fixed value indices into VV)
	b = igl.eigen.MatrixXi()
	# List of boundary conditions of each weight function
	bc = igl.eigen.MatrixXd()

	igl.boundary_conditions(V, T, C, igl.eigen.MatrixXi(), BE, igl.eigen.MatrixXi(), b, bc)

	bbw_data = igl.BBWData()
	# only a few iterations for sake of demo
	bbw_data.active_set_params.max_iter = 8
	bbw_data.verbosity = 2
	if not igl.bbw(V, T, b, bc, bbw_data, W):
		exit(-1)

	# Normalize weights to sum to one
	igl.normalize_row_sums(W, W)
	# precompute linear blend skinning matrix
	igl.lbs_matrix(V, W, M)

def generate_euclidean_weights(CAx, handles, others):
	W = np.zeros((len(handles) + len(others), len(handles)))

	for i in range(len(handles)):
		caxi = CAx[6*handles[i]:6*handles[i]+2]
		W[handles[i], i] = 1
		for j in range(len(others)):
			caxj = CAx[6*others[j]:6*others[j]+2 ]
			d = np.linalg.norm(caxi - caxj)

			W[others[j], i] = 1.0/d 
	for j in range(len(others)):
		W[others[j],:] /= np.sum(W[others[j],:])

	# print(W)
	# print(np.kron(W, np.eye(2)))
	# exit()
	return np.kron(W, np.eye(3))

def get_area(p1, p2, p3):
	return np.linalg.norm(np.cross((np.array(p1) - np.array(p2)), (np.array(p1) - np.array(p3))))*0.5

def get_centroid(p1, p2, p3):
	return (np.array(p1)+np.array(p2)+np.array(p3))/3.0

def rectangle_mesh(x, y, angle=0, step=1):
	V = []
	for i in range(0,x+1):
		for j in range(0,y+1):
			V.append([step*i, step*j])

	T = Delaunay(V).simplices
	
	return V, T, np.array([angle for i in range(len(T))])

def torus_mesh(r1, r2, r3, step):
	V = []
	T = []
	for theta in range(0, 80):
		angle = theta*np.pi/69
		if(angle<=np.pi):
			V.append([step*r1*np.cos(angle), step*r1*np.sin(angle)])
			V.append([step*r2*np.cos(angle), step*r2*np.sin(angle)])
			V.append([step*r3*np.cos(angle), step*r3*np.sin(angle)])
	
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
			u.append((np.pi/2) - 0.1)
		else:
			u.append((np.pi/2) + 0.1)

	return V, T, u

def get_min_max(iV,a):
	eps = 1e-3
	mov = []
	miny = np.amin(iV, axis=0)[a]
	maxy = np.amax(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - miny) < eps):
			mov.append(i)

		if(abs(iV[i][a] - maxy)< eps):
			mov.append(i)

	return mov

def get_min(iV, a):
	eps = 1e-1
	mov = []
	miny = np.amin(iV, axis=0)[a]
	for i in range(len(iV)):
		if(abs(iV[i][a] - miny) < eps):
			mov.append(i)

	return mov

def get_max(iV, a):
	eps = 2e-1
	mov = []
	maxy = np.amax(iV, axis=0)[a]
	for i in range(len(iV)):

		if(abs(iV[i][a] - maxy)< eps):
			mov.append(i)

	return mov

def feather_muscle1_test_setup(x = 3, y = 2):
	step = 0.1
	V,T,U = rectangle_mesh(x, y, step=step)
	# V,T, U = torus_mesh(5, 4, 3, step)

	half_x = step*(x)/2.0
	half_y = step*(y)/2.0
	u = []
	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[1]<half_y):
			u.append(-0.15)
		else:
			u.append(0.15)

	to_fix =[]
	for i in get_min_max(V,1):
		if(V[i][0]>half_x):
			to_fix.append(i)

	return (V, T, u), to_fix

def feather_muscle2_test_setup(r1 =1, r2=2, r3=3, r4 = 4, p1 = 10, p2 = 5):
	step = 0.1
	V = []
	T = []
	u = []
	V.append([(r4+1)*step, (r4+1)*step])
	V.append([(r4+1)*step + 1.5*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step - 1.5*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step + 1.75*step*r1, (r4+1)*step ])
	V.append([(r4+1)*step - 1.75*step*r1, (r4+1)*step ])
	for theta in range(0, p1):
		angle = theta*np.pi/p2
		# if(angle<=np.pi):
		V.append([2*step*r1*np.cos(angle) + (r4+1)*step, step*r1*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r2*np.cos(angle) + (r4+1)*step, step*r2*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r3*np.cos(angle) + (r4+1)*step, step*r3*np.sin(angle)+ (r4+1)*step])
		V.append([2*step*r4*np.cos(angle) + (r4+1)*step, step*r4*np.sin(angle)+ (r4+1)*step])

	T = Delaunay(V).simplices
	
	for i in range(len(T)):
		e = T[i]
		c = get_centroid(V[e[0]], V[e[1]], V[e[2]])
		if(c[1]< (step*(r4+1))):
			u.append(-0.15)
		else:
			u.append(0.15)


	to_fix =get_max(V,0)
	print(to_fix)
	return (V, T, u), to_fix

class Mesh:
	#class vars

	def __init__(self, iVTU, ito_fix=[]):
		#object vars
		self.fixed = ito_fix
		self.V = np.array(iVTU[0])
		self.T = iVTU[1]

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.Mass = None
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)+np.ravel(self.V)
		self.u = iVTU[2] if iVTU[2] is not None else np.zeros(len(self.T))
		self.q = np.zeros(len(self.T)*(1+2)) #theta, sx, sy

		#Rotation clusterings
		self.red_r = None
		self.r_element_cluster_map = None
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters()
		# self.readInRotClusters()

		#S skinnings
		self.red_s = None
		self.sW = None
		self.setupStrainSkinnings()

		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GU = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		#set initial strains
		for i in range(len(self.T)):
			self.q[3*i + 1] = 1
			self.q[3*i + 2] = 1


		self.getGlobalF(updateR = True, updateS = True, updateU=True)

	def setupStrainSkinnings(self):
		print("Setting up skinnings")
		if len(self.T) == 1:
			self.s_handles_ind = [0]
			self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))
			self.sW = np.eye(3)
			return

		t_set = Set([i for i in range(len(self.T))])

		# self.s_handles_ind =[i for i in range(len(self.T)) if i%50==0]
		self.s_handles_ind = [1]
		self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))

		centroids = self.getC().dot(self.getA().dot(self.x0))
		
		#generate weights by euclidean dist
		self.sW = generate_euclidean_weights(centroids, self.s_handles_ind, list(t_set.difference(Set(self.s_handles_ind))))
		# print(self.T)
		# print(self.s_handles_ind)
	
		print("Done setting up skinnings")
		return

	def setupRotClusters(self):
		print("Setting up rotation clusters")
		# of rotation clusters
		t_set = Set([i for i in range(len(self.T))])
		nrc = 4 #len(self.T)
		self.red_r = np.zeros(nrc)
		self.r_element_cluster_map = np.zeros(len(self.T), dtype = int)
		
		centroids = self.getC().dot(self.getA().dot(self.x0))
		minx = np.amin(self.V, axis=0)[0]
		maxx = np.amax(self.V, axis=0)[0]
		miny = np.amin(self.V, axis=0)[1]
		maxy = np.amax(self.V, axis=0)[1]
		for i in range(len(self.T)):
				# self.r_element_cluster_map[i] = i
				# self.r_cluster_element_map[i].append(i)
			if(centroids[6*i]<=(maxx + minx)/2.0):
				if(centroids[6*i+1]<=(maxy + miny)/2.0):
					self.r_element_cluster_map[i] = 0
					self.r_cluster_element_map[0].append(i)
				else:
					self.r_element_cluster_map[i] = 1
					self.r_cluster_element_map[1].append(i)
			else:
				if(centroids[6*i+1]<=(maxy + miny)/2.0):
					self.r_element_cluster_map[i] = 2
					self.r_cluster_element_map[2].append(i)
				else:
					self.r_element_cluster_map[i] = 3
					self.r_cluster_element_map[3].append(i)
		
		self.RotationBLOCK = []
		for i in range(len(self.red_r)):
			fixed = t_set.difference(Set(self.r_cluster_element_map[i]))
			
			b = sparse.kron(np.delete(np.eye(len(self.T)), list(fixed), axis=1), sparse.eye(6))
			self.RotationBLOCK.append(b.tocsc())
		print("Done setting up rotation clusters \n")
		return

	def readInRotClusters(self):
		print("Read in rotation clusters")
		t_set = Set([i for i in range(len(self.T))])
		nrc = 5
		self.red_r = np.zeros(nrc)
		self.r_element_cluster_map = np.zeros(len(self.T), dtype=int)
		
		values = [[16,0,1,12], [9,10,11,13],[14,15,5,4],[6,7]]
		for i in range(len(values)):
			for j in range(len(values[i])):
				self.r_element_cluster_map[values[i][j]] = i+1	
		
		for i in range(len(self.r_element_cluster_map)):
			self.r_cluster_element_map[self.r_element_cluster_map[i]].append(i)

		
		self.RotationBLOCK = []
		for i in range(len(self.red_r)):
			fixed = t_set.difference(Set(self.r_cluster_element_map[i]))
			
			b = sparse.kron(np.delete(np.eye(len(self.T)), list(fixed), axis=1), sparse.eye(6))
			self.RotationBLOCK.append(b)

		print("done reading rot clusters\n")
		return
	
	def createBlockingMatrix(self):
		print("Creating Block and AB matrix")
		onVerts = np.zeros(len(self.V))
		onVerts[self.fixed] = 1
		self.number_of_verts_fixed_on_element = self.getA().dot(np.kron(onVerts, np.ones(2)))
		if(len(self.fixed) == len(self.V)):
			return np.array([[]]), sparse.eye(2*len(self.V)).tocsc()
		b = sparse.kron(np.delete(np.eye(len(self.V)), self.fixed, axis =1), np.eye(2))

		ab = np.zeros(len(self.V))
		ab[self.fixed] = 1
		to_reset = [i for i in range(len(ab)) if ab[i]==0]

		if (len(self.fixed) == 0):
			return sparse.csc_matrix(b), sparse.csc_matrix((2*len(self.V), (2*len(self.V))))

		anti_b = sparse.kron(np.delete(np.eye(len(self.V)), to_reset, axis =1), np.eye(2))

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
			sub_P = np.kron(np.matrix([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]), np.eye(2))/3.0
			# sub_P = np.kron(np.matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]]), np.eye(2))
			p_block = []
			for i in range(len(self.T)):
				p_block.append(sub_P)

			self.P = sparse.block_diag(p_block)

		return self.P

	def getA(self):
		if(self.A is None):
			A = sparse.csc_matrix((6*len(self.T), 2*len(self.V)))
			for i in range(len(self.T)):
				e = self.T[i]
				for j in range(len(e)):
					v = e[j]
					A[6*i+2*j, 2*v] = 1
					A[6*i+2*j+1, 2*v+1] = 1
			A.eliminate_zeros()
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
		# sx = self.q[3*ind+1]
		# sy = self.q[3*ind+2]

		sx = self.sW[3*ind,:].dot(self.red_s)
		sy = self.sW[3*ind+1,:].dot(self.red_s)
		z = self.sW[3*ind +2,:].dot(self.red_s)
		S = np.array([[sx, z], [z, sy]])  
		return S

	def getF(self, ind):
		U = self.getU(ind)
		S = self.getS(ind)
		R = self.getR(ind)
		F =  np.matmul(R, np.matmul(U, np.matmul(S, U.transpose())))
		return F

	def getGlobalF(self, updateR = True, updateS = True, updateU = False):
		U_block_diag = []
		R_matrix = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		S_block_diag = []

		#cluster-wise update of R
		if updateR:
			for t in range(len(self.red_r)):
				c, s = np.cos(self.red_r[t]), np.sin(self.red_r[t])
				r_block = np.array(((c,-s), (s, c)))
				B = self.RotationBLOCK[t]
				GR_block = sparse.kron(sparse.eye(3*len(self.r_cluster_element_map[t])), r_block).tocsc()
		
				R_matrix += B.dot(GR_block.dot(B.T))

			self.GR = R_matrix

		for i in range(len(self.T)):
			if updateS:
				S_block_diag.append(sparse.kron(sparse.eye(3), self.getS(i)))
			if updateU:
				U_block_diag.append(sparse.kron(sparse.eye(3),self.getU(i)))
		
		if updateS:
			self.GS = sparse.block_diag(S_block_diag)

		if updateU:
			self.GU = sparse.block_diag(U_block_diag)
		
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
		self.sparseDSds()

		#LU inverse
		col2 = sparse.vstack((C.T, sparse.csc_matrix((C.shape[0], C.shape[0]))))
		col1 = sparse.vstack((self.AtPtPA, C))
		KKT = sparse.hstack((col1, col2))

		self.CholFac = scipy.sparse.linalg.splu(KKT.tocsc())


		r_size = len(self.mesh.red_r)
		g_size = len(self.mesh.g)
		s_size = len(self.mesh.red_s)

		self.Erg = np.zeros((g_size, r_size))
		self.Err = np.zeros((r_size, r_size))
		self.Egs = np.zeros((g_size, s_size))
		self.Ers = np.zeros((r_size, s_size))
		self.Ers_mid = np.zeros(( 6*len(self.mesh.T), r_size ))	

		self.USUtPAx_E = []


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
		print("		Jacobian")
		a = datetime.datetime.now()
		self.mesh.getGlobalF(updateR=True, updateS=True, updateU=False)
		b = datetime.datetime.now()
		Egg, Erg, Err, Egs, Ers = self.Hessians(useSparse=useSparse)

		c = datetime.datetime.now()
		lhs_left = sparse.vstack((Egg, Erg.T))
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
				col1 = np.concatenate((lhs_left.toarray(), np.concatenate((C.toarray(), np.zeros((rb_size, g_size))))))
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
		# print("Jac time: ", (b-a).microseconds, (c-b).microseconds, (d-c).microseconds, (e-d).microseconds, (f-e).microseconds, (aa-f).microseconds, (bb-aa).microseconds)
		print("		Jacobian")
		return dEds, dgds, drds

	def Hessians(self, useSparse=True):	
		print("			Hessians")
		PA = self.mesh.getP().dot(self.mesh.getA())
		PAg = PA.dot(self.mesh.g)
		USUt = self.mesh.GU.dot(self.mesh.GS.dot(self.mesh.GU.T))
		USUtPAx = USUt.dot(self.PAx)
		UtPAx = self.mesh.GU.T.dot(self.PAx)
		r_size = len(self.mesh.red_r)
		g_size = len(self.mesh.g)
		s_size = len(self.mesh.red_s)
		DR = self.sparseDRdr()
		DS = self.sparseDSds()
		

		a1 = datetime.datetime.now()
		###############ERG
		# if not useSparse:
		# 	print("Erg NOT SPARSE")
		# 	sample = np.multiply.outer(-1*PA.T.toarray(), USUtPAx.T)
		# 	for j in range(Erg.shape[1]):
		# 		sp = DR[j]
		# 		for i in range(Erg.shape[0]):
		# 			if sp.nnz>0:
		# 				Erg[i,j] = sp.multiply(sample[i,:,:]).sum()

		# else:
		# 	sample = self.sparseErg_first(-1*PA, USUtPAx)
		# 	for i in range(self.Erg.shape[0]):
		# 		for j in range(self.Erg.shape[1]):
		# 			self.Erg[i,j] = sample[i].multiply(DR[j]).sum()

		a2 = datetime.datetime.now()
		###############ERR
		# negPAg_USUtPAx = np.multiply.outer( -1*PAg, USUtPAx)
		# DDR = self.sparseDDRdrdr()
		# for i in range(self.Err.shape[0]):
		# 	for j in range(self.Err.shape[1]):
		# 		spD = DDR[i][j]
		# 		if spD.nnz>0:
		# 			self.Err[i,j] = spD.multiply(negPAg_USUtPAx).sum()
		
		a3 = datetime.datetime.now()
		###############EGS
		# PAtRU = PA.T.dot(self.mesh.GR.dot(self.mesh.GU))
		# if not useSparse:
		# 	print("Egs NOT SPARSE")
		# 	d_gEgdS = np.multiply.outer(-1*PAtRU.toarray(), UtPAx.T)
		# 	for j in range(self.Egs.shape[1]):
		# 		sp = DS[j]
		# 		for i in range(self.Egs.shape[0]):
		# 			if sp.nnz>0:
		# 				self.Egs[i,j] = sp.multiply(d_gEgdS[i,:,:]).sum()
		# else:
		# 	d_gEgdS = self.sparseEgs_first(-1*PAtRU, UtPAx)
		# 	for i in range(self.Egs.shape[0]):
		# 		for j in range(self.Egs.shape[1]):
		# 			self.Egs[i,j] = d_gEgdS[i].multiply(DS[j]).sum()
		
		a4 = datetime.datetime.now()
		###############ERS
		if not useSparse:
			print("Ers NOT SPARSE")
			first = np.multiply.outer(-PAg, self.mesh.GU.toarray())
			for j in range(self.Ers_mid.shape[1]):
				spR = DR[j]
				for i in range(self.Ers_mid.shape[0]):
					self.Ers_mid[i,j] = spR.multiply(first).sum()

			second = np.multiply.outer(UtPAx, self.Ers_mid)
			for j in range(self.Ers.shape[1]):
				spS = DS[j]
				for i in range(self.Ers.shape[0]):
					self.Ers[i,j] = spS.multiply(second[:,:,i]).sum()
		else:
			# first = self.Ers_first(PAg)
			# for i in range(self.Ers_mid.shape[0]):
			# 	for j in range(self.Ers_mid.shape[1]):
			# 		self.Ers_mid[i,j] = DR[j].multiply(first[i]).sum()

			odd, even = self.sparseErs_first(PAg)
			for i in range(self.Ers_mid.shape[1]):
				spR = DR[i]
				odd_spR = spR.multiply(odd).sum(axis = 0) #6T x 6T (each 2 cols get summed to Ers_mid i,j)
				even_spR = spR.multiply(even).sum(axis = 0)#6T x 6T (each 2 cols get summed to Ers_mid i+1,j)
				odd_i = odd_spR.reshape((odd_spR.shape[1]/2, 2)).sum(axis=1)
				even_i = even_spR.reshape((even_spR.shape[1]/2, 2)).sum(axis=1)
				col_i = np.zeros((len(PAg),1))
				col_i[2*np.arange(self.Ers_mid.shape[0]/2)] = odd_i
				col_i[2*np.arange(self.Ers_mid.shape[0]/2)+1] = even_i
				self.Ers_mid[:, i]   = col_i[:,0]

			second = self.sparseErs_second(UtPAx, self.Ers_mid)
			for i in range(self.Ers.shape[0]):
				if second[i].nnz>0:
					for j in range(self.Ers.shape[1]):
						self.Ers[i,j] = second[i].multiply(DS[j]).sum()

			print(self.Ers)
		a5 = datetime.datetime.now()
		
		# print("TIME: ", (a2-a1).microseconds, (a3-a2).microseconds, (a4-a3).microseconds, (a5-a4).microseconds)
		print("			Hessians")
		return self.AtPtPA, self.Erg, self.Err, self.Egs, self.Ers

	def sparseErg_first(self, nPAT, USUtPAx):
		first = []
		spUSUtPAx = sparse.csc_matrix(USUtPAx.T)
		
		for c in range(nPAT.shape[1]):
			PATc = nPAT.getcol(c)
			sp = PATc.dot(spUSUtPAx)
			first.append(sp)


		return first

	def sparseEgs_first(self, PAtRU, UtPAx):
		first = []
		spUtPAx = sparse.csc_matrix(UtPAx.T)
		PAtRU_T = PAtRU.tocsr()

		for r in range(PAtRU_T.shape[0]):
			mat_r = PAtRU.getrow(r)
			sp = mat_r.T.dot(spUtPAx)
			first.append(sp)

		return first

	def sparseErs_first(self, nPAg):

		odd_blocks = []
		even_blocks = []
		for t in range(len(self.mesh.T)):
			u = self.mesh.getU(t)
			odd_blocks.append(np.multiply.outer(u[:,0], nPAg[6*t + 0:6*t + 2]))
			odd_blocks.append(np.multiply.outer(u[:,0], nPAg[6*t + 2:6*t + 4]))
			odd_blocks.append(np.multiply.outer(u[:,0], nPAg[6*t + 4:6*t + 6]))
			
			even_blocks.append(np.multiply.outer(u[:,1], nPAg[6*t + 0:6*t + 2]))
			even_blocks.append(np.multiply.outer(u[:,1], nPAg[6*t + 2:6*t + 4]))
			even_blocks.append(np.multiply.outer(u[:,1], nPAg[6*t + 4:6*t + 6]))

		odd = sparse.block_diag(odd_blocks).tocsc()
		even = sparse.block_diag(even_blocks).tocsc()

		return odd, even

	def sparseErs_second(self, UtPAx, mid):
		second = []
		spUtPAx = sparse.csc_matrix(UtPAx.T)
		spmid = sparse.csc_matrix(mid)
		for c in range(spmid.shape[1]):
			midc = spmid.getcol(c)
			sp = midc.dot(spUtPAx)
			second.append(sp)


		return second

	def Ers_first(self, nPAg):
		first = []
		spPAg = sparse.csc_matrix(nPAg.T)
		for c in range(self.mesh.GU.shape[1]):
			GUc = self.mesh.GU.getcol(c)
			sp = GUc.dot(spPAg)
			first.append(sp)
		return first

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
		dEdS =  np.multiply.outer(self.mesh.GS.dot(UtPAx), UtPAx) - np.multiply.outer(RU.T.dot(PAg), UtPAx)
		# dEds = np.tensordot(dEdS, _dSds, axes = ([0,1], [0,1]))
		dEds = np.zeros(len(self.mesh.red_s))
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
			s = self.mesh.sW.dot(self.mesh.red_s)
	
			for t in range(len(self.mesh.red_s)/3):
				sWx = self.mesh.sW[:,3*t]
				sWy = self.mesh.sW[:,3*t+1]
				sWo = self.mesh.sW[:,3*t+2]
				x_block = []
				y_block = []
				off_diagonal_block =[]
				for i in range(s.shape[0]/3):
					dSdsx = np.array([[sWx[3*i],0],[0,0]])
					dSdsy = np.array([[0,0],[0,sWy[3*i+1]]])
					dSdso = np.array([[0, sWo[3*i+2]],[sWo[3*i+2],0]])

					x_block.append(sparse.kron(sparse.eye(3), dSdsx))
					y_block.append(sparse.kron(sparse.eye(3), dSdsy))
					off_diagonal_block.append(sparse.kron(sparse.eye(3), dSdso))
				gdSdsx = sparse.block_diag(x_block)
				gdSdsy = sparse.block_diag(y_block)
				gdSdso = sparse.block_diag(off_diagonal_block)
			

				DS.append(gdSdsx)
				DS.append(gdSdsy)
				DS.append(gdSdso)

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

	def itR(self):
		PAg = self.mesh.getP().dot(self.mesh.getA().dot(self.mesh.g))
		Ax = self.mesh.getA().dot(self.mesh.x0)

		for i in range(len(self.mesh.red_r)):
			# a1 = datetime.datetime.now()
			one = self.mesh.RotationBLOCK[i].T.dot(PAg)
			two = self.USUtPAx_E[i]

			m_PAg = np.reshape(one, (len(one)/2,2))
			m_USUPAx = np.reshape(two, (len(two)/2,2))
			F = np.matmul(m_PAg.T,m_USUPAx)
			# a2 = datetime.datetime.now()
			# f_11 = one[2*np.arange(len(one)/2)].dot(two[2*np.arange(len(two)/2)])
			# f_12 = one[2*np.arange(len(one)/2)].dot(two[2*np.arange(len(two)/2)+1])
			# f_21 = one[2*np.arange(len(one)/2)+1].dot(two[2*np.arange(len(two)/2)])
			# f_22 = one[2*np.arange(len(one)/2)+1].dot(two[2*np.arange(len(two)/2)+1])
			# F = np.array([[f_11, f_12], [f_21, f_22]])

			u, s, vt = np.linalg.svd(F, full_matrices=True)
			R = np.matmul(vt.T, u.T)

			veca = np.array([1,0])
			vecb = np.dot(R, veca)


			#check angle sign
			theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
			c, s = np.cos(theta), np.sin(theta)
			R_check = np.array(((c,-s), (s, c)))
			if(np.linalg.norm(R-R_check)<1e-8):
				theta *=-1

			self.mesh.red_r[i] = theta
			# a3 = datetime.datetime.now()
			# print("Time: ", (a2-a1).microseconds, (a3-a2).microseconds)
		# print("\n")
		return 1
	
	def itT(self):
		Abtg = self.ANTI_BLOCK.T.dot(self.mesh.g)
		
		FPAx = self.mesh.GF.dot(self.PAx)
		AtPtFPAx = self.mesh.getA().T.dot(self.mesh.getP().T.dot(FPAx))
		
		gd = np.concatenate((AtPtFPAx, Abtg))
		gu = self.CholFac.solve(gd)
		self.mesh.g = gu[0:AtPtFPAx.size]

		return 1

	def iterate(self, its=100):
		print("	+ARAP iterate")
		self.USUtPAx_E = []
		for i in range(len(self.mesh.red_r)):
			B = self.mesh.RotationBLOCK[i]
			PAx_e = B.T.dot(self.PAx)
			
			Ue = B.T.dot(self.mesh.GU.dot(B))
			Se = B.T.dot(self.mesh.GS.dot(B))

			USUPAx = Ue.dot(Se.dot(Ue.T.dot(PAx_e.T)))

			self.USUtPAx_E.append(USUPAx)

		Eg0 = self.dEdg()
		for i in range(its):
			# at = datetime.datetime.now()
			g = self.itT()
			# bt = datetime.datetime.now()
			r = self.itR()
			# ct = datetime.datetime.now()
			# print("TIMES: ", (bt-at).microseconds, (ct-bt).microseconds)
			self.mesh.getGlobalF(updateR=True, updateS=False, updateU=False)
			Eg = self.dEdg()
			# print("i", i,np.linalg.norm(Eg-Eg0))
			if(1e-8 > np.linalg.norm(Eg-Eg0)):
				print("	-ARAP iterate ")
				# print("ARAP converged", np.linalg.norm(Eg))	
				return
			Eg0 = Eg

class NeohookeanElastic:

	def __init__(self, imesh):
		self.mesh = imesh
		self.f = np.zeros(2*len(self.mesh.T))
		self.v = np.zeros(2*len(self.mesh.V))
		# self.M = self.mesh.getMassMatrix()
		self.BLOCK, self.ANTI_BLOCK = self.mesh.createBlockingMatrix()

		self.youngs = 60000
		self.poissons = 0.49
		self.mu = self.youngs/(2+ 2*self.poissons)
		self.lambd = self.youngs*self.poissons/((1+self.poissons)*(1-2*self.poissons))
		self.dimensions = 2

		self.grav = np.array([0,-9.81])
		self.rho = 10000

		self.muscle_fiber_mag_target = 1000
		self.muscle_fibre_mag = 100

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
		gt = -rho*area*np.dot(grav, cadgds)
		return gt

	def GravityForce(self, dgds):
		fg = np.zeros(len(self.mesh.red_s))		
		Ax = self.mesh.getA().dot(self.mesh.x0)
				
		CAdgds = self.mesh.getC().dot(self.mesh.getA().dot(dgds))

		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			gv = self.mesh.getU(t).dot(self.grav)
			fg += self.GravityElementForce(self.rho, area, gv, CAdgds[6*t:6*t+2, :], t)

		return fg

	def WikipediaPrinStretchElementForce(self, a, rs, wx, wy, wo):
		md = 0.5*(self.youngs*self.poissons)/((1.0+self.poissons)*(1.0-2.0*self.poissons))
		mc = 0.5*self.youngs/(2.0*(1.0+self.poissons))

		t_0 = np.dot(wo, rs)
		t_1 = np.dot(wx, rs)
		t_2 = np.dot(wy, rs)

		if(t_1<=0 or t_2<=0 or t_1*t_2-t_0*t_0<=0):
			return 1e40

		t_3 = ((t_1 * t_2) - (t_0 * t_0))
		t_4 = np.dot(rs, wo)
		t_5 = (2.0 / 6)
		t_6 = np.dot(rs, wx)
		t_7 = np.dot(rs, wy)
		t_8 = ((t_6 * t_7) - (t_4 * t_4))
		t_9 = ((mc * (t_8 ** -(1 + t_5))) * (t_6 + t_7))
		t_10 = (((2 * a) * t_9) / 6.0)
		t_11 = -t_5
		t_12 = ((a * mc) * (t_8 ** t_11))
		t_13 = (1.0 / 2)
		t_14 = ((a * md) * ((t_8 ** (t_13 - 1)) * ((t_8 ** t_13) - 1)))
		functionValue = (a * ((mc * (((t_3 ** t_11) * (t_1 + t_2)) - 2)) + (md * (((t_3 ** t_13) - 1) ** 2))))
		gradient = (((((((((((4 * a) * t_9) / 6.0) * t_4) * wo) - (((t_10 * t_7) * wx) + ((t_10 * t_6) * wy))) + (t_12 * wx)) + (t_12 * wy)) + ((t_14 * t_7) * wx)) + ((t_14 * t_6) * wy)) - ((((4 * t_14) / 2.0) * t_4) * wo))
		
		return -gradient

	def WikipediaForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		Ax = self.mesh.getA().dot(self.mesh.x0)
		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			force += self.WikipediaPrinStretchElementForce(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:] )

		return force

	def WikipediaPrinStretchElementEnergy(self, area, rs, wx, wy, wo):
		
		m_D = 0.5*(self.youngs*self.poissons)/((1.0+self.poissons)*(1.0-2.0*self.poissons))
		m_C = 0.5*self.youngs/(2.0*(1.0+self.poissons))
		
		#MATH version
		# c = [[sx, so],[so, sy]]
		# I1 = np.trace(c)
		# I3 = np.linalg.det(c)
		# J = math.sqrt(I3)
		# I1_b = math.pow(J, -2.0/3)*(I1)
		# v1 = area*(m_C*(I1_b -2) + m_D*(J-1)*(J-1))

		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40
		term1 = m_C*(math.pow((sx*sy - so*so), -2.0/6)*(sx + sy) - 2)
		term2 = m_D*math.pow((math.pow(sx*sy -so*so, 1/2.0) - 1), 2)
		v2 = area*(term1 + term2)

		return v2

	def WikipediaEnergy(self,_rs):
		E = 0
		
		Ax = self.mesh.getA().dot(self.mesh.x0)
	
		for t in range(len(self.mesh.T)):
			area = get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6])
			E += self.WikipediaPrinStretchElementEnergy(area, _rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:],self.mesh.sW[3*t+2,:])

		return E

	def MuscleElementEnergy(self, rs,wx, wy, wo, u):
		sx = wx.dot(rs)
		sy = wy.dot(rs)
		so = wo.dot(rs)
		if(sx<=0 or sy<=0 or sx*sy-so*so<=0):
			return 1e40

		c = np.array([[sx, so],[so, sy]])

		return 0.5*self.muscle_fibre_mag*(u.dot(c.dot(c.T.dot(u.T))))

	def MuscleEnergy(self, _rs):
		En = 0
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
		
			En += self.MuscleElementEnergy(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u)
		
		return En 

	def MuscleElementForce(self, rs, wx, wy, wo, u1, u2):

		t0 = self.muscle_fibre_mag *u1*u1 
		t1 = t0*(rs.dot(wx)*wx + rs.dot(wo)*wo)


		t2 = 0.5*self.muscle_fibre_mag*u1*u2
		t3 = t2*rs.dot(wo)
		t4 = t3*wx + t2*rs.dot(wx)*wo + t3*wy + t2*rs.dot(wy)*wo

		t5 = self.muscle_fibre_mag *u2*u2 
		t6 = t5*(rs.dot(wy)*wy + rs.dot(wo)*wo)
			
		return -1*(t1 + 2*t4 + t6)

	def MuscleForce(self, _rs):
		force = np.zeros(len(self.mesh.red_s))
		for t in range(len(self.mesh.T)):
			alpha = self.mesh.u[t]
			c, s = np.cos(alpha), np.sin(alpha)
			u = np.array([c,s]).dot(np.array([[c,-s],[s, c]]))
			force += self.MuscleElementForce(_rs, self.mesh.sW[3*t,:], self.mesh.sW[3*t+1,:], self.mesh.sW[3*t+2,:],u[0], u[1])
		
		return force 

	def Energy(self, irs):
		e2 = self.WikipediaEnergy(_rs=irs)
		e1 = -1*self.GravityEnergy() 
		e3 = self.MuscleEnergy(_rs=irs)
		return e1 + e2 + e3

	def Forces(self, irs, idgds):
		f2 = self.WikipediaForce(_rs=irs)
		f1 =  self.GravityForce(idgds)
		if idgds is None:
			return f2 + f3
		f3 = self.MuscleForce(_rs=irs)
		return f1 + f2 + f3

class TimeIntegrator:

	def __init__(self, imesh, iarap, ielastic = None):
		self.time = 0
		self.timestep = 0.001
		self.mesh = imesh
		self.arap = iarap 
		self.elastic = ielastic 
		self.adder = 5e-2
		# self.set_random_strain()
		self.mov = np.array(self.mesh.fixed_min_axis(1))
		self.bnds = [(1e-5, None) for i in range(len(self.mesh.red_s))]

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

		# if(self.time%10 == 0):
		# 	self.adder *= -1

		# self.mesh.g[2*self.mov+1] -= self.adder
		self.elastic.muscle_fibre_mag = 100
		print(self.elastic.muscle_fibre_mag)
		self.time += 1

	def move_g(self):
		if(self.time%10 == 0):
			self.adder *= -1

		self.mesh.g[2*self.mov +1] -= self.adder

	def solve(self):
		pass

	def static_solve(self):
		print("Static Solve")
		self.iterate()
		s0 = self.mesh.red_s + np.zeros(len(self.mesh.red_s))
		
		alpha1 =1e5
		alpha2 =1

		def energy(s):
			for i in range(len(s)):
				self.mesh.red_s[i] = s[i]

			# print("guess ", self.mesh.red_s)
			self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)
			
			self.arap.iterate()

			E_arap = self.arap.Energy()
			E_elastic =  self.elastic.Energy(irs=self.mesh.red_s)
			# print("s", self.mesh.red_s)
			print("E", E_arap, E_elastic)
	

			return alpha1*E_arap + alpha2*E_elastic

		def jacobian(s):
			for i in range(len(s)):
				self.mesh.red_s[i] = s[i]

			# print("guess ",self.mesh.red_s)
			dgds = None
			self.arap.iterate()
			J_arap, dgds, drds = self.arap.Jacobian()

			J_elastic = -1*self.elastic.Forces(irs = self.mesh.red_s, idgds=dgds)

	
			return  alpha2*J_elastic + alpha1*J_arap
		
		res = scipy.optimize.minimize(energy, s0, method='L-BFGS-B', bounds=self.bnds,  jac=jacobian, options={'gtol': 1e-6, 'ftol':1e-4, 'disp': False, 'eps':1e-8})
		

		for i in range(len(res.x)):
			self.mesh.red_s[i] = res.x[i]
		
		self.mesh.getGlobalF(updateR=False, updateS=True, updateU=False)
		print("r1", self.mesh.red_r)
		print("s1", res.x)
		print("g1", self.mesh.g)
		print(res)
		print("static solve")
		

def display():
	# iV, iT, iU = rectangle_mesh(3,3,.1)
	# # iV, iT, iU = torus_mesh(5, 4, 3, .1)
	# to_fix = get_min(iV,1)
	VTU, to_fix = feather_muscle2_test_setup(p1 = 100, p2 = 50)
	mesh = Mesh(VTU,ito_fix=to_fix)

	neoh =NeohookeanElastic(imesh=mesh )
	arap = ARAP(imesh=mesh)
	time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)

	viewer = igl.viewer.Viewer()

	tempR = igl.eigen.MatrixXuc(1280, 800)
	tempG = igl.eigen.MatrixXuc(1280, 800)
	tempB = igl.eigen.MatrixXuc(1280, 800)
	tempA = igl.eigen.MatrixXuc(1280, 800)

	# sy_inds = [i for i in range(len(mesh.red_s)) if i%2==0]
	# mesh.red_s[0] = 1.17
	# mesh.red_s[1] = 0.8
	# mesh.red_s[2] = 0.00001
	# print(mesh.red_s)
	# mesh.getGlobalF()
	# arap.iterate()
	# print(mesh.red_s)
	# mesh.red_s[sy_inds] = 1
	# print("En", neoh.Energy(irs=mesh.red_s))
	# mesh.g[2*time_integrator.mov+1] += 0.2
	# arap.iterate()
	def mouse_down(viewer, aaa, bbb):
		bc = igl.eigen.MatrixXd()
		RV, RT = mesh.getContinuousVT()
		# Cast a ray in the view direction starting from the mouse position
		fid = igl.eigen.MatrixXi(np.array([-1]))
		coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
		hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
		viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(RV), igl.eigen.MatrixXi(RT), fid, bc)
		if hit:
			# paint hit red
			print(fid)
			return True

		return False

	def key_down(viewer):
		viewer.data.clear()
		if(time_integrator.time>5):
			exit()
		# print(mesh.u)
		# print(mesh.red_s)
		# if(aaa==65):
			# time_integrator.iterate()
			# arap.iterate()
			# print("Earap", arap.Energy())
			# print("En", neoh.Energy(irs=mesh.red_s))

			# J_arap, dgds, drds = arap.Jacobian()
			# print("Grad arap ", J_arap)
			# J_elastic = neoh.Forces(irs = mesh.red_s, idgds=dgds)
			# print("Grad n ", J_elastic)
		if(viewer.core.is_animating):
			time_integrator.static_solve()
			# time_integrator.move_g()
			# arap.iterate()
			# arap.Hessians()
			# print("TETS: ", len(mesh.T))
			# print(neoh.WikipediaForce(_rs=mesh.red_s))
			# print(neoh.MuscleForce(_rs=mesh.red_s))

		
		DV, DT = mesh.getDiscontinuousVT()
		RV, RT = mesh.getContinuousVT()
		V2 = igl.eigen.MatrixXd(RV)
		T2 = igl.eigen.MatrixXi(RT)
		viewer.data.set_mesh(V2, T2)
	
		red = igl.eigen.MatrixXd([[1,0,0]])
		purple = igl.eigen.MatrixXd([[1,0,1]])
		green = igl.eigen.MatrixXd([[0,1,0]])
		black = igl.eigen.MatrixXd([[0,0,0]])
		

		for e in DT:
			P = DV[e]
			DP = np.array([P[1], P[2], P[0]])
			viewer.data.add_edges(igl.eigen.MatrixXd(P), igl.eigen.MatrixXd(DP), purple)


		FIXED = []
		for i in range(len(mesh.fixed)):
			FIXED.append(mesh.g[2*mesh.fixed[i]:2*mesh.fixed[i]+2])

		viewer.data.add_points(igl.eigen.MatrixXd(np.array(FIXED)), red)

			
		CAg = mesh.getC().dot(mesh.getA().dot(mesh.g))
		#Skinning handles
		# for i in range(len(mesh.s_handles_ind)):
		# 	C = np.matrix([CAg[6*mesh.s_handles_ind[i]:6*mesh.s_handles_ind[i]+2],CAg[6*mesh.s_handles_ind[i]:6*mesh.s_handles_ind[i]+2]])
		# 	U = 0.02*mesh.getU(i).transpose()+C
		# 	viewer.data.add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
			# viewer.data.add_points(igl.eigen.MatrixXd(np.array([CAg[6*mesh.s_handles_ind[i]:6*mesh.s_handles_ind[i]+2]])), black)
		
		#centroids and rotation clusters
		for i in range(len(mesh.T)):
			S = mesh.getS(i)
			C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
			U = 0.01*mesh.getU(i).transpose()+C
			if(np.linalg.norm(mesh.sW[2*i,:])>=1):
				viewer.data.add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
				viewer.data.add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), green)
			else:	
				viewer.data.add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
				viewer.data.add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), red)
			viewer.data.add_points(igl.eigen.MatrixXd(np.array([CAg[6*i:6*i+2]])), igl.eigen.MatrixXd([[0,.2*mesh.r_element_cluster_map[i],1-0.2*mesh.r_element_cluster_map[i]]]))
			# print(np.array([CAg[6*i:6*i+2]]))
		# print(np.array(cag))
		# print(igl.eigen.MatrixXd(np.array(cag)))
		
		#Write image
		if(viewer.core.is_animating):
			viewer.core.draw_buffer(viewer.data, viewer.opengl, False, tempR, tempG, tempB, tempA)
			igl.png.writePNG(tempR, tempG, tempB, tempA, "frames/"+str(time_integrator.time)+".png")
		# pass


		# Clustered Rotations
		# colors  = []
		# e_ind = 0
		# tot_e = len(mesh.T)
		# for e in mesh.r_element_cluster_map:
		# 	colors.append([1, 1, mesh.r_element_cluster_map[e_ind]])
		# 	e_ind += 1
		# Colors = igl.eigen.MatrixXd(colors)
		# viewer.data.set_colors(Colors);

		return True

	# for clicks in range(40):
	# key_down(viewer, 'b', 123)
	viewer.callback_post_draw = key_down
	viewer.callback_mouse_down = mouse_down
	viewer.core.is_animating = False
	viewer.launch()

# display()


import re
import cProfile, pstats, StringIO
def headless():
	VTU, to_fix = feather_muscle2_test_setup(p1 = 60, p2 = 30)
	# VTU, to_fix = feather_muscle1_test_setup(x = 2, y = 2)
	print(len(VTU[0]), len(VTU[1]))
	mesh = Mesh(VTU,ito_fix=to_fix)

	neoh =NeohookeanElastic(imesh=mesh )
	arap = ARAP(imesh=mesh)
	time_integrator = TimeIntegrator(imesh = mesh, iarap = arap, ielastic = neoh)
	time_integrator.move_g()

	pr = cProfile.Profile()
	pr.enable()
	
	# print(len(mesh.T))
	print(len(mesh.T))
	arap.Hessians()

	pr.disable()
	s = StringIO.StringIO()
	sortby = 'cumulative'
	ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
	ps.print_stats()
	print s.getvalue()

headless()