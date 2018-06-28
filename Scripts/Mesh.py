import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse

import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.5f}'.format(x)})
from iglhelpers import *
temp_png = os.path.join(os.getcwd(),"out.png")

from Helpers import *

class Mesh:

	def __init__(self, iVTU, ito_fix=[], ito_mov=[], setup=False, red_g= True):
		#object vars
		self.reduced_g = red_g
		self.youngs = 60000
		self.poissons = 0.45
		self.fixed = ito_fix
		self.V = np.array(iVTU[0])
		self.T = iVTU[1]
		self.mov = ito_mov

		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)#+np.ravel(self.V)
		self.u = iVTU[2] if iVTU[2] is not None else np.zeros(len(self.T))

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.N = None
		self.BLOCK = None
		self.ANTI_BLOCK = None
		self.Mass = None

		self.G = None
		self.Eigvals = None
		self.z = None
		self.z0 = None
		
		t_size = len(self.T)
		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tolil()
		self.GU = sparse.diags([np.ones(6*t_size)],[0]).tolil()

		if(setup==False):

			# Rotation clusterings
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
	
			
			print("\n+ Setup GF")
			self.getGlobalF(updateR = True, updateS = True, updateU=True)
			print("- Done with GF")

	def setupStrainSkinnings(self):
		print("Setting up skinnings")
		if len(self.T) == 1:
			self.s_handles_ind = [0]
			self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))
			self.sW = np.eye(3)
			return

		t_set = Set([i for i in range(len(self.T))])

		# self.s_handles_ind =[i for i in range(len(self.T)) if i%1==0]
		self.s_handles_ind = [1,3]
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
		nrc =  2#len(self.T)
		self.red_r = np.zeros(nrc)
		self.r_element_cluster_map = np.zeros(len(self.T), dtype = int)
		centroids = self.getC().dot(self.getA().dot(self.x0))
		mins = np.amin(self.V, axis=0)
		maxs = np.amax(self.V, axis=0)
		minx = mins[0]
		maxx = maxs[0]
		miny = mins[1]
		maxy = maxs[1]

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
					self.r_element_cluster_map[i] = 0
					self.r_cluster_element_map[0].append(i)
				else:
					self.r_element_cluster_map[i] = 1
					self.r_cluster_element_map[1].append(i)

		self.RotationBLOCK = []
		for i in range(len(self.red_r)):
			notfixed = Set(self.r_cluster_element_map[i])
			# fixed = t_set.difference(notfixed)
			# b = sparse.kron(np.delete(np.eye(len(self.T)), list(fixed), axis=1), sparse.eye(6))
			
			nf = np.array(list(notfixed))
			bo = sparse.eye(len(self.T)).tocsc()

			b = sparse.kron(bo[:,nf], sparse.eye(6))
			self.RotationBLOCK.append(b.tocsc())

		print("Done setting up rotation clusters \n")
		return

	def setupModalAnalysis(self, modes):
		# not_moving = set(self.fixed) - set(self.mov)
		B, AB = self.createBlockingMatrix()
		M = self.getMassMatrix()
		K = self.getStiffnessMatrix()
		M = B.T.dot(M.dot(B))
		K = B.T.dot(K.dot(B))
		eig, ev = general_eig_solve(A=K, B =M, modes=modes)
		ev *= np.logical_or(1e-10>ev , ev<-1e-10)
		self.G = sparse.csc_matrix(B.dot(ev))
		self.G.eliminate_zeros()

		self.Eigvals = eig
		print("Done Modal Analysis")

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

	def createBlockingMatrix(self, fix=None):
		if(self.BLOCK == None or self.ANTI_BLOCK==None or fix != None):
			print("Creating Block and AB matrix")
			if(fix==None):
				fix = self.fixed

			if(len(fix) == len(self.V)):
				self.BLOCK =  sparse.csc_matrix(np.array([[]]))
				self.ANTI_BLOCK = sparse.eye(2*len(self.V)).tocsc()
				return self.BLOCK, self.ANTI_BLOCK
			if (len(fix) == 0):
				self.BLOCK = sparse.eye(2*len(self.V)).tocsc()
				self.ANTI_BLOCK= sparse.csc_matrix((2*len(self.V), (2*len(self.V))))
				return self.BLOCK, self.ANTI_BLOCK

			onVerts = np.zeros(len(self.V))
			onVerts[fix] = 1
			self.number_of_verts_fixed_on_element = self.getA().dot(np.kron(onVerts, np.ones(2)))

			Id = sparse.eye(len(self.V)).tocsc()
			anti_b = sparse.kron(Id[:,fix], np.eye(2))
			
			ab = np.zeros(len(self.V))
			ab[fix] = 1
			notfix = [i for i in range(0, len(ab)) if ab[i] == 0]
			b = sparse.kron(Id[:,notfix], np.eye(2))

			print("Done with Blocking matrix\n")
			self.BLOCK = b.tocsc() 
			self.ANTI_BLOCK = anti_b.tocsc()

		return self.BLOCK, self.ANTI_BLOCK

	def fixed_min_axis(self, a):
		if(self.fixed==[]):
			return []
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
			self.P = sparse.kron(sparse.eye(len(self.T)), sub_P).tocsc()

		return self.P

	def getA(self):
		if(self.A is None):
			A = sparse.lil_matrix((6*len(self.T), 2*len(self.V)))
			for i in range(len(self.T)):
				e = self.T[i]
				for j in range(len(e)):
					v = e[j]
					A[6*i+2*j, 2*v] = 1
					A[6*i+2*j+1, 2*v+1] = 1

			self.A = A.tocsc()
			self.A.eliminate_zeros()
		return self.A

	def getC(self):
		if(self.C is None):
			self.C = sparse.kron(sparse.eye(len(self.T)), sparse.kron(np.ones((3,3))/3 , np.eye(2)))
		return self.C

	def getN(self):
		if self.N == None:
			B, AB = self.createBlockingMatrix()

			not_moving = np.array(list(set(self.fixed) - set(self.mov)))
			mov = np.array(self.mov )
			col1 = np.zeros((B.shape[0],2))
			col2 = np.zeros((B.shape[0], 2))
			col1[2*not_moving, 0] = 1
			col1[2*not_moving+1, 1] = 1
			col2[2*mov, 0] = 1
			col2[2*mov+1, 1] = 1
			self.N = (sparse.hstack((B,col1, col2)), sparse.hstack((B, 0.25*col1, 0.25*col2)))

		return self.N

	def getU(self, ind):
		alpha = self.u[ind]
		cU, sU = np.cos(alpha), np.sin(alpha)
		U = np.array(((cU,-sU), (sU, cU)))
		return U

	def getR(self, ind):
		theta = self.red_r[self.r_element_cluster_map[ind]]
		c, s = np.cos(theta), np.sin(theta)
		R = np.array(((c,-s), (s, c)))
		return R

	def getS(self, ind):
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
				GR_block = sparse.kron(sparse.eye(3*len(self.r_cluster_element_map[t])), r_block)

				R_matrix += B.dot(GR_block.dot(B.T))

			self.GR = R_matrix

		if updateS:
			s = self.sW.dot(self.red_s)
			sx = s[3*np.arange(len(self.T))]
			sy = s[3*np.arange(len(self.T))+1]
			so = s[3*np.arange(len(self.T))+2]
			diag_x = np.kron(sx, np.array([1,0,1,0,1,0]))
			diag_y = np.kron(sy, np.array([0,1,0,1,0,1]))
			diag_o = np.kron(so, np.array([1,0,1,0,1,0]))
			diag_xy = diag_x+ diag_y
			self.GS = sparse.diags([diag_o[:-1], diag_xy, diag_o[:-1]],[-1,0,1]).tocsc()
					
	
		if updateU:
			for i in range(len(self.T)):
				u = self.getU(i)
				self.GU[6*i+0:6*i+2, 6*i+0:6*i+2] = u
				self.GU[6*i+2:6*i+4, 6*i+2:6*i+4] = u
				self.GU[6*i+4:6*i+6, 6*i+4:6*i+6] = u
			self.GU.tocsc()

		if(updateR or updateS or updateU):
			self.GF = self.GR.dot(self.GU.dot(self.GS.dot(self.GU.T)))
		return

	def getDiscontinuousVT(self):
		C = self.getC()
		# print(self.z)
		# print(self.G.dot(self.z))
		CAg = C.dot(self.getA().dot(self.getg()))
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

	def getContinuousVT(self, g=[]):
		if len(g) ==0:
			x = self.getg()
		else:
			x = g
		RecV = np.zeros((len(self.V), 2))
		for i in range(len(x)/2):
			RecV[i, 0] = x[2*i]
			RecV[i, 1] = x[2*i+1]

		return RecV, self.T

	def getMassMatrix(self):
		if(self.Mass is None):
			print("Creating Mass Matrix")
			mass_diag = np.zeros(2*len(self.V))
			density = 1.0
			for i in range(len(self.T)):
				e = self.T[i]
				undef_area = density*get_area(self.V[e[0]], self.V[e[1]], self.V[e[2]])
				mass_diag[2*e[0]+0] += undef_area/3.0
				mass_diag[2*e[0]+1] += undef_area/3.0

				mass_diag[2*e[1]+0] += undef_area/3.0
				mass_diag[2*e[1]+1] += undef_area/3.0

				mass_diag[2*e[2]+0] += undef_area/3.0
				mass_diag[2*e[2]+1] += undef_area/3.0
			print("Done with Mass matrix")
			self.Mass = sparse.diags(mass_diag)
		return self.Mass

	def getStiffnessMatrix(self):
		print("Getting stiffness matrix")
		K = np.zeros((2*len(self.V), 2*len(self.V)))


		D = np.array([[1-self.poissons, self.poissons, 0],
                        [ self.poissons, 1-self.poissons, 0],
                        [ 0, 0, 0.5-self.poissons]])*(self.youngs/((1+self.poissons)*(1-2*self.poissons)))

		for e in self.T:
			B = self.Be(e)
			local_K = B.T.dot(D.dot(B))*1*get_area(self.g[2*e[0]:2*e[0]+2], self.g[2*e[1]:2*e[1]+2], self.g[2*e[2]:2*e[2]+2])
			j = 0
			for r in local_K:
				kj = j%2
				for s in range(r.shape[0]/2):
					dfxrdxs = r.item(2*s)
					dfxrdys = r.item(2*s+1)

					K[2*e[j/2]+kj, 2*e[s]] += dfxrdxs
					K[2*e[j/2]+kj, 2*e[s]+1] += dfxrdys

				j+=1

		return sparse.csc_matrix(K)

	def Be(self, e):
		d1 = np.array(self.V[e[0]]) - np.array(self.V[e[2]])
		d2 = np.array(self.V[e[1]]) - np.array(self.V[e[2]])
		Dm = np.column_stack(( d1, d2))
		Dm = np.linalg.inv(Dm)

		b0 = Dm[:, 0]
		b1 = Dm[:, 1]
		b2 = -Dm[:, 0] - Dm[:, 1]

		Be = np.array([ [b0[0], 0, b1[0], 0 , b2[0], 0],
						[0, b0[1], 0, b1[1], 0 , b2[1]],
						[b0[1], b0[0], b1[1], b1[0], b2[1], b2[0]]])
		return Be

	def getg(self):
		gn = self.G.dot(self.z)
		return gn + self.x0
