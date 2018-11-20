import numpy as np
import math
from collections import defaultdict
from sets import Set
import json
import scipy
from scipy.optimize import minimize
from scipy.spatial import Delaunay
from scipy import sparse
from scipy.cluster.vq import vq, kmeans, whiten
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

	def __init__(self, iVTU=None, ito_fix=[], ito_mov=[], modes_used=None, read_in=False, muscle=True):
		if read_in:
			return
		#Get Variables setup
		if muscle:
			self.elem_youngs = [600000 for i in range(len(iVTU[1]))] #g/cm*s^2
			self.elem_poissons = [0.45 for i in range(len(iVTU[1]))]
			self.poissons = 0.45
			self.fixed = list(set(ito_fix))#.union(set(ito_mov)))
			nrc = 5
			nsh = 5
		else:
			self.elem_youngs = [600000 for i in range(len(T))] #g/cm*s^2
			self.elem_poissons = [0.45 for i in range(len(T))]
			self.fixed = list(set(ito_fix).union(set(ito_mov)))
			nrc = 1
			nsh = 1
		self.shandle_muscle = []
		self.V = np.array(iVTU[0])
		self.T = iVTU[1]
		print("MeshSize:")
		print(self.V.shape, self.T.shape)
		self.mov = list(set(ito_mov))

		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)#+np.ravel(self.V)
		self.u = iVTU[2] if iVTU[2] is not None else np.zeros(len(self.T))
		self.u_clusters_element_map = None 
		self.u_toggle = np.ones(len(self.T))

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.N = None
		self.BLOCK = None
		self.ANTI_BLOCK = None
		self.Mass = None

		self.G = None
		self.Q = None
		self.Eigvals = None
		self.z = None
		
		t_size = len(self.T)
		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tolil()
		self.GU = sparse.diags([np.ones(6*t_size)],[0]).tolil()
		self.GUSUt = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tocsc()


		# Modal analysis
		if modes_used is None:
			self.Q = None 
			self.G = np.eye(2*len(self.V))
		else:
			self.Q =self.setupModes(modes_used=modes_used)
			self.G = self.Q[:,:]
		self.z = np.zeros(self.G.shape[1])

		# Rotation clusterings
		self.red_r = None
		self.r_element_cluster_map = None
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters(rclusters=False, nrc=nrc)

		#S skinnings
		self.red_s = None
		self.s_handles_ind = None
		self.sW = None
		self.setupStrainSkinnings(shandles=False, nsh=nsh)
		self.red_s_dot = np.zeros(len(self.red_s))

		print("\n+ Setup GF")
		self.getGlobalF(updateR = True, updateS = True, updateU=True)
		print("- Done with GF")

		Ax = self.getA().dot(self.x0)
		self.areas = []
		for t in range(len(self.T)):
			self.areas.append(get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6]))
	
	def init_from_file(self, V=None, T=None, u=None, Q=None, fix=None, mov=None, r_element_cluster_map=None, s_handles_ind=None, u_clusters_element_map=None, modes_used=None):
		self.elem_youngs = [600000 for i in range(len(T))]
		self.elem_poissons = [0.45 for i in range(len(T))]
		self.V = V
		self.T = T
		self.shandle_muscle = []
		# print("MeshSize:")
		# print(self.V.shape, self.T.shape)
		# self.fixed = np.hstack((fix[0,:],mov[0,:]))
		self.fixed = fix[0,:]
		self.mov = mov[0,:]
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)#+np.ravel(self.V)
		self.u = u[0,:]
		self.u_toggle = np.ones(len(self.T))

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.N = None
		self.BLOCK = None
		self.ANTI_BLOCK = None
		self.Mass = None

		self.Eigvals = None
		self.z = None
		
		t_size = len(self.T)
		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tolil()
		self.GU = sparse.diags([np.ones(6*t_size)],[0]).tolil()
		
		# U clusters
		self.u_clusters_element_map = u_clusters_element_map

		# Modal analysis
		self.Q = Q
		if modes_used is not None and len(Q) != 0:
			self.G = Q[:, :modes_used]
		else:
			self.G = np.eye(2*len(self.V))

		self.z = np.zeros(self.G.shape[1])
		
		# Rotation clusterings
		self.red_r = None
		self.r_element_cluster_map = r_element_cluster_map[:,0]
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters(rclusters=True, nrc=2)


		# self.readInRotClusters()
		#S skinnings
		self.red_s = None
		self.s_handles_ind = s_handles_ind[0,:]
		self.sW = None
		self.setupStrainSkinnings(shandles = True)
		self.red_s_dot = np.zeros(len(self.red_s))

		
		print("\n+ Setup GF")
		self.getGlobalF(updateR = True, updateS = True, updateU=True)
		print("- Done with GF")
		Ax = self.getA().dot(self.x0)
		self.areas = []
		for t in range(len(self.T)):
			self.areas.append(get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6]))

	def init_muscle_bone(self, V, T, u, s_ind, r_ind, sW, emat, fix, mov, shandle_muscle, modes_used):
		self.elem_youngs = np.array([600000 if e<0.5 else 6e5 for e in emat])
		self.elem_poissons = np.array([0.45 if e<0.5 else 0.45 for e in emat])
		self.u_toggle = emat
		self.shandle_muscle = shandle_muscle

		self.V = V
		self.T = T

		self.fixed = fix
		self.mov = mov
		
		self.x0 = np.ravel(self.V)
		self.g = np.zeros(len(self.V)*2)
		self.u = u

		self.number_of_verts_fixed_on_element = None
		self.P = None
		self.A = None
		self.C = None
		self.N = None
		self.BLOCK = None
		self.ANTI_BLOCK = None
		self.Mass = None

		self.Eigvals = None
		self.z = None
		
		t_size = len(self.T)
		self.GF = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GR = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		self.GS = sparse.diags([np.zeros(6*t_size-1), np.ones(6*t_size), np.zeros(6*t_size-1)],[-1,0,1]).tolil()
		self.GU = sparse.diags([np.ones(6*t_size)],[0]).tolil()
		
		# U clusters
		# self.u_clusters_element_map = u_clusters_element_map
		Q = []
		# Modal analysis
		if modes_used is not None and len(Q) != 0:
			self.G = Q[:, :modes_used]
		elif modes_used is not None:
			self.Q =self.setupModes(modes_used=modes_used)
			self.G = self.Q[:,:]
		else:
			self.G = np.eye(2*len(self.V))

		self.z = np.zeros(self.G.shape[1])
		
		# Rotation clusterings
		self.red_r = None
		self.r_element_cluster_map = r_ind
		self.r_cluster_element_map = defaultdict(list)
		self.RotationBLOCK = None
		self.setupRotClusters(rclusters=True, nrc=1)


		#S skinnings
		self.s_handles_ind = s_ind
		self.sW = sW
		self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))
		self.red_s_dot = np.zeros(len(self.red_s))

		
		print("\n+ Setup GF")
		self.getGlobalF(updateR = True, updateS = True, updateU=True)
		print("- Done with GF")
	
		Ax = self.getA().dot(self.x0)
		self.areas = []
		for t in range(len(self.T)):
			self.areas.append(get_area(Ax[6*t+0:6*t+2], Ax[6*t+2:6*t+4], Ax[6*t+4:6*t+6]))

	def setupModes(self, modes_used=None):
		A = self.getA()
		P = self.getP()
		B, AB = self.createBlockingMatrix()
		C = AB.T
		M = sparse.diags(self.getVertexWiseMassDiags())
		K = A.T.dot(P.T.dot(P.dot(A)))
		eig, ev = general_eig_solve(A=K, B = M, modes=modes_used)

		ev *= np.logical_or(1e-10<ev , ev<-1e-10)
		eig = eig[2:]
		ev = ev[:,2:]
		ev = np.divide(ev, eig*eig)
		ev = sparse.csc_matrix(ev)
		############handle modes KKT solve#####
		col1 = sparse.vstack((K, C))
		col2 = sparse.vstack((C.T, sparse.csc_matrix((C.shape[0], C.shape[0]))))
		KKT = sparse.hstack((col1, col2))
		eHconstrains = sparse.vstack((sparse.csc_matrix((K.shape[0], C.shape[0])), sparse.eye(C.shape[0])))
		eH = sparse.linalg.spsolve(KKT.tocsc(), eHconstrains.tocsc())[0:K.shape[0]]
		# eH *= np.logical_or(1e-10<eH , eH<-1e-10)
		# eHFac =  scipy.sparse.linalg.splu(KKT.tocsc())
		# eH = eHFac.solve(eHconstrains.toarray())[0:K.shape[0]]
		#######################################
		###############QR get orth basis#######
		eHeV = sparse.hstack((eH, ev))
		# print(eH.shape, ev.shape)
		# exit()
		# eVN = np.append(eVeH, np.zeros((len(self.self.x0),1)),1)
		# eVN[:,-1] = self.self.x0
		Q1, QR1 = np.linalg.qr(eHeV.toarray(), mode="reduced")
		# Q = eHeV
		return Q1

	def setupStrainSkinnings(self, shandles = False, nsh=1):
		print("Setting up skinnings")
		if len(self.T) == 1:
			self.s_handles_ind = [0]
			self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))
			self.sW = np.eye(3)
			return

		t_set = Set([i for i in range(len(self.T))])

		if shandles is False:
			print("No reduced skinning handles")
			# self.s_handles_ind =[i for i in range(len(self.T)) if i%1==0]
			# self.s_handles_ind = [0]
			skinning_r_cluster_element_map = defaultdict(list)
			skinning_r_element_cluster_map = self.kmeans_rotationclustering(clusters=nsh)
			
			for i in range(len(self.T)):			
				skinning_r_cluster_element_map[skinning_r_element_cluster_map[i]].append(i)
		
			self.s_handles_ind = []
			CAx0 = self.getC().dot(self.getA().dot(self.x0))
			for k in range(len(skinning_r_cluster_element_map.keys())):
				els = np.array(skinning_r_cluster_element_map[k], dtype='int32')
				centx = CAx0[6*els]
				centy = CAx0[6*els+1]
				avc = np.array([np.sum(centx)/len(els), np.sum(centy)/len(els)]) 
				minind = els[0]
				mindist = np.linalg.norm(avc-np.array([centx[0], centy[0]]))
				for i in range(1,len(els)):
					dist = np.linalg.norm(avc-np.array([centx[i], centy[i]]))
					if dist<=mindist:
						mindist = dist 
						minind = els[i]

				self.s_handles_ind.append(minind)


		self.red_s = np.kron(np.ones(len(self.s_handles_ind)), np.array([1,1,0]))


		#generate weights by euclidean dist
		self.sW = self.bbw_strain_skinning_matrix(handles = self.s_handles_ind)

		print("Done setting up skinnings")
		return

	def bbw_strain_skinning_matrix(self, handles=[0]):
		vertex_handles = self.T[handles]
		unique_vert_handles = np.unique(vertex_handles)
		helper = np.add(np.zeros(unique_vert_handles[-1]+1), -1)

		for i in range(len(unique_vert_handles)):
			helper[unique_vert_handles[i]] = i 

		vert_to_tet = np.zeros((len(handles), 3), dtype="int32")
		for i in range(vertex_handles.shape[0]):
			vert_to_tet[i,:] = helper[vertex_handles[i]]

		C = self.V[unique_vert_handles]
		P = np.array([np.arange(len(C))], dtype="int32").T

		V = igl.eigen.MatrixXd(self.V)
		T = igl.eigen.MatrixXi(self.T)
		M = igl.eigen.MatrixXd()
		W = igl.eigen.MatrixXd()
		C = igl.eigen.MatrixXd(C)
		P = igl.eigen.MatrixXi(P)
		# List of boundary indices (aka fixed value indices into VV)
		b = igl.eigen.MatrixXi()
		# List of boundary conditions of each weight function
		bc = igl.eigen.MatrixXd()
		# print(unique_vert_handles)
		# print(self.V[np.array([989, 1450, 1610])])
		# print(C)
		# exit()
		igl.boundary_conditions(V, T, C, P, igl.eigen.MatrixXi(), igl.eigen.MatrixXi(), b, bc)	

		
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
		
		vW = e2p(W) #v x verts of handles

		tW = np.zeros((len(self.T), len(handles))) #T x handles
		#get average of vertices for each triangle
		for i in range(len(self.T)):
			e = self.T[i]
			for h in range(len(handles)):
				if i== handles[h]:
					tW[i,:] *= 0
					tW[i,h] = 1

					break
				p0 = vW[e[0],vert_to_tet[h,:]].sum()
				p1 = vW[e[1],vert_to_tet[h,:]].sum()
				p2 = vW[e[2],vert_to_tet[h,:]].sum()
				tW[i,h] = (p0+p1+p2)/3.

		tW /= np.sum(tW, axis =1)[:, np.newaxis] #normalize rows to sum to 1
		return np.kron(tW, np.eye(3))

	def setupRotClusters(self, rclusters=False, nrc=1):
		print("Setting up rotation clusters")
		# of rotation clusters
		t_set = Set([i for i in range(len(self.T))])
		if rclusters is False:
			if nrc == len(self.T):
				self.r_element_cluster_map = np.arange(nrc)
			else:
				self.r_element_cluster_map = self.kmeans_rotationclustering(clusters=nrc)
		
		for i in range(len(self.T)):			
			self.r_cluster_element_map[self.r_element_cluster_map[i]].append(i)
		
		if rclusters is True:
			nrc = len(self.r_cluster_element_map.keys())
		
		self.red_r = np.zeros(nrc)
	
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

	def kmeans_rotationclustering(self, clusters = 5):
		A = self.getA()
		C = self.getC()
		G = np.add(self.G.T, self.x0)
		#all modes at once
		CAG = C.dot(A.dot(G.T))#scipy wants data in format: observations(elem) x features (modes)

		Data = np.zeros((len(self.T), 2*self.G.shape[1]))
		# print(CAG.shape, Data.shape)
		for i in range(len(self.T)):
			point = CAG[6*i:6*i+2, :]
			Data[i,:] = np.ravel(point) #triangle by x1,y1,x2,y2, x3,y3....

		print(clusters, Data.shape)
		centroids,_ = kmeans(Data, clusters)
		idx,_ = vq(Data,centroids)
		return idx

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
			# self.P = sparse.kron(sparse.eye(len(self.T)), sub_P).tocsc()
			self.P = sparse.kron(sparse.diags((self.u_toggle-1)*10 -1)*-1, sub_P).tocsc()
			# print(((self.u_toggle-1)*10 -1)*-1)
			# exit()
			# sparse.diags(self.u_toggle*10+1)
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
		# R_matrix = sparse.csc_matrix((6*len(self.T), 6*len(self.T)))
		S_block_diag = []

		#cluster-wise update of R
		if updateR:
			diag = np.zeros(6*len(self.T))
			offdiag = np.zeros(6*len(self.T))
			for t in range(len(self.red_r)):
				c1, c2 = np.cos(self.red_r[t]), np.sin(self.red_r[t])
				B = self.RotationBLOCK[t]
				diag += B.dot(c1*np.ones(6*len(self.r_cluster_element_map[t])))
				offdiag += B.dot(c2*np.kron(np.ones(3*len(self.r_cluster_element_map[t])), np.array([1,0])))

			self.GR = sparse.diags([offdiag[:-1], diag, -1*offdiag[:-1]],[-1,0,1]).tocsc()

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

		if (updateR or updateS or updateU):
			if (updateS or updateU):
				self.GUSUt = self.GU.dot(self.GS.dot(self.GU.T))
		self.GF = self.GR.dot(self.GUSUt)
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
			mass_diag = np.zeros(6*len(self.T))
			density = 1000.0
			for i in range(len(self.T)):
				e = self.T[i]
				undef_area = density*get_area(self.V[e[0]], self.V[e[1]], self.V[e[2]])
				mass_diag[6*i+0] += undef_area/3.0
				mass_diag[6*i+1] += undef_area/3.0

				mass_diag[6*i+2] += undef_area/3.0
				mass_diag[6*i+3] += undef_area/3.0

				mass_diag[6*i+4] += undef_area/3.0
				mass_diag[6*i+5] += undef_area/3.0
			
			print("Done with Mass matrix")
			self.Mass = sparse.diags(mass_diag)
		return self.Mass
	def getVertexWiseMassDiags(self):
		mass_diag = np.zeros(2*len(self.V))
		density = 1000
		for i in range(len(self.T)):
			e = self.T[i]
			undef_area = density*get_area(self.V[e[0]], self.V[e[1]], self.V[e[2]])
			mass_diag[2*e[0]+0] += undef_area/3.0
			mass_diag[2*e[0]+1] += undef_area/3.0

			mass_diag[2*e[1]+0] += undef_area/3.0
			mass_diag[2*e[1]+1] += undef_area/3.0

			mass_diag[2*e[2]+0] += undef_area/3.0
			mass_diag[2*e[2]+1] += undef_area/3.0

		return mass_diag

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
