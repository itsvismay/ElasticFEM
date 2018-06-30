#Setting up the mesh
#1. Create V, T (mesh shape)
#2. Figure out muscle fiber directions U
#	- harmonic function gradient
#	- heat
#3. Get modes
#5. Rotation clusters using K-means clustering on modes
#6. BBW for skinning meshes. One handle per rotation cluster.

import numpy as np
import math
from collections import defaultdict
from sets import Set
import datetime
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
from Helpers import *
from Mesh import Mesh
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *


def rectangle_mesh(x, y, step=1):
	V = []
	for i in range(0,x+1):
		for j in range(0,y+1):
			V.append([step*i, step*j])
	
	T = Delaunay(V).simplices
	return V, T

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

def feather_muscle2_test_setup(r1 =1, r2=2, r3=3, r4 = 4, p1 = 200, p2 = 100):
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
	return (V, T, u), to_fix

def modal_analysis(mesh):
	A = mesh.getA()
	P = mesh.getP()
	B, AB = mesh.createBlockingMatrix()
	C = AB.T
	M = mesh.getMassMatrix()
	K = A.T.dot(P.T.dot(P.dot(A)))
	if K.shape[0]-3<500:
		num_modes = K.shape[0]-3
	else:
		num_modes = 500

	eig, ev = general_eig_solve(A=K, B = M, modes=num_modes+2)

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
	eVeH = sparse.hstack((ev, eH))
	# eVN = np.append(eVeH, np.zeros((len(self.mesh.x0),1)),1)
	# eVN[:,-1] = self.mesh.x0
	# Q, QR1 = np.linalg.qr(eVeH, mode="reduced")
	Q = eVeH
	return Q

def k_means_rclustering(mesh, clusters = 5):
	A = mesh.getA()
	C = mesh.getC()
	G = np.add(mesh.G.toarray().T, mesh.x0)
	#all modes at once
	CAG = C.dot(A.dot(G.T))#scipy wants data in format: observations(elem) x features (modes)

	Data = np.zeros((len(mesh.T), 2*mesh.G.shape[1]))
	# print(CAG.shape, Data.shape)
	for i in range(len(mesh.T)):
		point = CAG[6*i:6*i+2, :]
		Data[i,:] = np.ravel(point) #triangle by x1,y1,x2,y2, x3,y3....


	centroids,_ = kmeans(Data, clusters)
	idx,_ = vq(Data,centroids)

	return idx

def bbw_strain_skinning_matrix(mesh, handles=[0,1]):
	vertex_handles = mesh.T[handles]
	unique_vert_handles = np.unique(vertex_handles)
	helper = np.add(np.zeros(unique_vert_handles[-1]+1), -1)

	for i in range(len(unique_vert_handles)):
		helper[unique_vert_handles[i]] = i 

	vert_to_tet = np.zeros((len(handles), 3), dtype="int32")
	for i in range(vertex_handles.shape[0]):
		vert_to_tet[i,:] = helper[vertex_handles[i]]

	C = mesh.V[unique_vert_handles]
	P = np.array([np.arange(len(C))], dtype="int32").T

	V = igl.eigen.MatrixXd(mesh.V)
	T = igl.eigen.MatrixXi(mesh.T)
	M = igl.eigen.MatrixXd()
	W = igl.eigen.MatrixXd()
	C = igl.eigen.MatrixXd(C)
	P = igl.eigen.MatrixXi(P)
	# List of boundary indices (aka fixed value indices into VV)
	b = igl.eigen.MatrixXi()
	# List of boundary conditions of each weight function
	bc = igl.eigen.MatrixXd()

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

	tW = np.zeros((len(mesh.T), len(handles))) #T x handles
	#get average of vertices for each triangle
	for i in range(len(mesh.T)):
		e = mesh.T[i]
		for h in range(len(handles)):
			if i== handles[h]:
				tW[i,:] *= 0
				tW[i,h] = 1

				break
			p0 = vW[e[0],vert_to_tet[h,:]].sum()
			p1 = vW[e[1],vert_to_tet[h,:]].sum()
			p2 = vW[e[2],vert_to_tet[h,:]].sum()
			tW[i,h] = (p0+p1+p2)/3.

	tW /= np.linalg.norm(tW, axis =1)[:, np.newaxis] #normalize rows to sum to 1
	

def heat_method(mesh):
	t = 1e-1
	eLc = igl.eigen.SparseMatrixd()
	igl.cotmatrix(igl.eigen.MatrixXd(mesh.V), igl.eigen.MatrixXi(mesh.T), eLc)
	Lc = e2p(eLc)

	M = mesh.getMassMatrix()
	Mdiag = M.diagonal()[2*np.arange(Lc.shape[0])]
	Mc = sparse.diags(Mdiag)


	#Au = b st. Cu = Cu0
	u0 = np.zeros(len(mesh.V))
	fixed = list(set(mesh.fixed) - set(mesh.mov))
	u0[fixed] = 2
	u0[mesh.mov] = -2

	Id = sparse.eye(len(mesh.V)).tocsc()
	fixedverts = [i for i in range(len(u0)) if u0[i]!=0]
	C = Id[:,fixedverts]

	A = (Mc - t*Lc)
	# u = sparse.linalg.spsolve(A.tocsc(), u0)
	col1 = sparse.vstack((A, C.T))
	col2 = sparse.vstack((C, sparse.csc_matrix((C.shape[1], C.shape[1]))))
	KKT = sparse.hstack((col1, col2))
	lhs = np.concatenate((u0, C.T.dot(u0)))
	u = sparse.linalg.spsolve(KKT.tocsc(), lhs)[0:len(u0)]
	
	eG = igl.eigen.SparseMatrixd()
	nV = np.concatenate((mesh.V, u[:,np.newaxis]), axis=1)
	igl.grad(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(mesh.T), eG)
	eu = igl.eigen.MatrixXd(u)
	eGu = (eG*eu).MapMatrix(len(mesh.T), 3)
	Gu = e2p(eGu)

	gradu = np.zeros(len(mesh.T))
	uvecs = np.zeros((len(mesh.T),2))
	for i in range(len(mesh.T)):
		e = mesh.T[i]
		
		# area2 = 2*get_area(mesh.V[e[0]], mesh.V[e[1]], mesh.V[e[2]])
		# p0 = np.concatenate((mesh.V[e[0]], [0]))
		# p1 = np.concatenate((mesh.V[e[1]], [0]))
		# p2 = np.concatenate((mesh.V[e[2]], [0]))
		# normal = get_unit_normal(p0, p1, p2)
		# s1 = u[e[0]]*np.cross(normal, mesh.V[e[1]] - mesh.V[e[2]])[0:2]
		# s2 = u[e[1]]*np.cross(normal, mesh.V[e[2]] - mesh.V[e[0]])[0:2]
		# s3 = u[e[2]]*np.cross(normal, mesh.V[e[0]] - mesh.V[e[1]])[0:2]
		# uvec = (1/area2)*(s1+s2+s3)
		uvec = Gu[i,0:2]
		uvecs[i,:] = uvec
		veca = uvec
		vecb = np.array([1,0])
		# theta = np.arccos(np.dot(veca,vecb)/(np.linalg.norm(veca)*np.linalg.norm(vecb)))
		x1 = np.cross(veca, vecb).dot(np.array([0,0,1]))
		x2 = np.linalg.norm(veca)*np.linalg.norm(vecb) + veca.dot(vecb)
		theta = 2*np.arctan2(x1, x2)[2]
		# print(theta)
		gradu[i] = theta

	return gradu, u, eGu, uvecs

class Preprocessing:
	def __init__(self, _VT):
		self.last_mouse = None
		self.V = _VT[0]
		self.T = _VT[1]
		self.U = np.zeros(len(self.T))
		self.Fix = get_max(self.V, a=1, eps=1e-2)		
		self.Mov = get_min(self.V, a=1, eps=1e-2)
		self.rClusters = []
		self.gi = 0
		self.mesh = None
		self.uvec = None
		self.eGu = None
		self.UVECS = None

	def save_mesh_setup(self, name=None):
		#SAVE: V, T, Fixed points, Moving points,Eigs, EigV 
		# Muscle clusters, Rotation clusters, Skinning Handles, 
		# Maybe ARAP pre-processing info

		#AND json file with basic info
		# - mesh folder name (so ARAP pre-processing can potentially be saved)
		# - sizes, YM, poisson, muscle strengths, density
		if name==None:
			name = str(datetime.datetime.now())
		folder = "./MeshSetups/"+name+"/"

	def read_mesh_setup(self, name=None):
		if name==None:
			print("Name can't be none.")
			exit()

	def createMesh(self):
		to_fix = self.Fix+self.Mov 
		to_mov = self.Mov
		
		self.mesh = Mesh([self.V, self.T, self.U], ito_fix = to_fix, ito_mov=to_mov, setup= True, red_g=True)
		# Q = modal_analysis(self.mesh)
		# self.mesh.G = Q[:, :15]
		# self.mesh.z = np.zeros(15)
		# self.rClusters = k_means_rclustering(self.mesh)
		# sW = bbw_strain_skinning_matrix(self.mesh)

		self.mesh.u, self.uvec, self.eGu, self.UVECS = heat_method(self.mesh)
	
		
	def display(self):
		red = igl.eigen.MatrixXd([[1,0,0]])
		purple = igl.eigen.MatrixXd([[1,0,1]])
		green = igl.eigen.MatrixXd([[0,1,0]])
		black = igl.eigen.MatrixXd([[0,0,0]])
		blue = igl.eigen.MatrixXd([[0,0,1]])
		white = igl.eigen.MatrixXd([[1,1,1]])

		randc = [[random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)] for i in range(10)]

		viewer = igl.glfw.Viewer()
		def mouse_up(viewer, btn, bbb):
			print("up")

		def mouse_down(viewer, btn, bbb):
			print("down")
			# Cast a ray in the view direction starting from the mouse position
			bc = igl.eigen.MatrixXd()
			fid = igl.eigen.MatrixXi(np.array([-1]))
			coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
			hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(self.V), igl.eigen.MatrixXi(self.T), fid, bc)
			if hit and btn==0:
				# paint hit red
				print("fix", fid)
				ind = e2p(fid)[0][0]
				self.Fix.append(self.T[ind][0])
				self.Fix.append(self.T[ind][1])
				self.Fix.append(self.T[ind][2])
				fixed_pts = []
				for i in range(len(self.Fix)):
					fixed_pts.append(self.V[self.Fix[i]])
				viewer.data().add_points(igl.eigen.MatrixXd(np.array(fixed_pts)), red)
				return True
			if hit and btn==2:
				# paint hit red
				print("mov", fid)
				ind = e2p(fid)[0][0]
				self.Mov.append(self.T[ind][0])
				self.Mov.append(self.T[ind][1])
				self.Mov.append(self.T[ind][2])
				mov_pts = []
				for i in range(len(self.Mov)):
					mov_pts.append(self.V[self.Mov[i]])
				viewer.data().add_points(igl.eigen.MatrixXd(np.array(mov_pts)), green)
				return True
			if hit:
				print("Element", fid)
				ind = e2p(fid)[0][0]
				print(self.T[ind])
				return True
			return False

		def key_down(viewer,aaa, bbb):
			if(aaa == 65):
				self.createMesh()

			viewer.data().clear()
			if self.uvec is None:
				nV = self.V
			else:
				nV = self.V
				# print(self.mesh.V.shape, self.uvec.shape)
				# nV = np.concatenate((self.mesh.V, self.uvec[:,np.newaxis]), axis=1)
				# BC = igl.eigen.MatrixXd()
				# igl.barycenter(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(self.T), BC)

				# GU_mag = self.eGu.rowwiseNorm()
				# max_size = igl.avg_edge_length(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(self.T)) / GU_mag.mean()
				
				# viewer.data().add_edges(BC, BC + max_size*self.eGu, black)

			viewer.data().set_mesh(igl.eigen.MatrixXd(nV), igl.eigen.MatrixXi(self.T))

			centroids = []
			for i in range(len(self.T)):
				p1 = self.V[self.T[i][0]]
				p2 = self.V[self.T[i][1]]
				p3 = self.V[self.T[i][2]]
				c = get_centroid(p1, p2, p3) 
				color = black
				if (self.rClusters != []):
					color = igl.eigen.MatrixXd(np.array([randc[self.rClusters[i]]]))

				# viewer.data().add_points(igl.eigen.MatrixXd(np.array([c])),  color)
			
			if not self.uvec is None:
				CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.x0))
				for i in range(len(self.T)):
					C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
					U = np.multiply(self.mesh.getU(i), np.array([[0.03],[0.03]])) + C
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)
					# viewer.data().add_edges(igl.eigen.MatrixXd(C[1,:]), igl.eigen.MatrixXd(U[1,:]), red)

			fixed_pts = []
			for i in range(len(self.Fix)):
				fixed_pts.append(self.V[self.Fix[i]])
			viewer.data().add_points(igl.eigen.MatrixXd(np.array(fixed_pts)), red)
			mov_pts = []
			for i in range(len(self.Mov)):
				mov_pts.append(self.V[self.Mov[i]])
			viewer.data().add_points(igl.eigen.MatrixXd(np.array(mov_pts)), green)


		key_down(viewer, "b", 123)
		viewer.callback_mouse_down = mouse_down
		viewer.callback_key_down = key_down
		viewer.callback_mouse_up = mouse_up
		viewer.core.is_animating = False
		viewer.launch()


