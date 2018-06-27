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
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *

from Helpers import *
from Mesh import Mesh

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
	print(ev.shape)

	ev *= np.logical_or(1e-10<ev , ev<-1e-10)
	eig = eig[2:]
	ev = sparse.csc_matrix(ev[:,2:])
	############handle modes KKT solve#####
	col1 = sparse.vstack((K, C))
	col2 = sparse.vstack((C.T, sparse.csc_matrix((C.shape[0], C.shape[0]))))
	KKT = sparse.hstack((col1, col2))
	eHconstrains = sparse.vstack((sparse.csc_matrix((K.shape[0], C.shape[0])), sparse.eye(C.shape[0])))
	eH = sparse.linalg.spsolve(KKT.tocsc(), eHconstrains.tocsc())[0:K.shape[0]]
	# eH *= np.logical_or(1e-10>eH , eH<-1e-10)
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

def k_means_rclustering(mesh, mode=0):
	A = mesh.getA()
	C = mesh.getC()
	mesh.z = np.zeros(len(mesh.z))
	mesh.z[mode] = 1
	CAG = C.dot(A.dot(mesh.getg()))#scipy wants data in format: observations(elem) x features (modes)
	Data = np.zeros((len(mesh.T), 2))
	# print(CAG.shape, Data.shape)
	for i in range(len(mesh.T)):
		point = CAG[6*i:6*i+2]
		Data[i,:] = np.ravel(point)


	centroids,_ = kmeans(Data, 4)
	idx,_ = vq(Data,centroids)
	print(idx.shape, Data.shape, centroids.shape)
	print(len(mesh.T))
	return idx



class Preprocessing:
	def __init__(self, _VT):
		self.last_mouse = None
		self.V = _VT[0]
		self.T = _VT[1]
		self.U = np.zeros(len(self.T))
		self.Fix = get_max(self.V,a=1, eps=1e-2)		
		self.Mov = get_min(self.V, a=1, eps=1e-2)
		self.rClusters = []
		self.gi = 0
		self.mesh = None

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
		Q = modal_analysis(self.mesh)
		self.mesh.G = Q[:, :15]
		self.mesh.z = np.zeros(15)
		self.rClusters = k_means_rclustering(self.mesh)
		print(self.rClusters)

	def display(self):
		red = igl.eigen.MatrixXd([[1,0,0]])
		purple = igl.eigen.MatrixXd([[1,0,1]])
		green = igl.eigen.MatrixXd([[0,1,0]])
		black = igl.eigen.MatrixXd([[0,0,0]])
		blue = igl.eigen.MatrixXd([[0,0,1]])
		white = igl.eigen.MatrixXd([[1,1,1]])
		
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
			return False

		def key_down(viewer,aaa, bbb):
			if(aaa == 65):
				self.createMesh()
			if(aaa == 67):
				#new rot cluster mode
				self.rClusters = k_means_rclustering(self.mesh, mode=self.gi)
				self.gi+=1

			viewer.data().clear()
			viewer.data().set_mesh(igl.eigen.MatrixXd(self.V), 
									igl.eigen.MatrixXi(self.T))

			centroids = []
			for i in range(len(self.T)):
				p1 = self.V[self.T[i][0]]
				p2 = self.V[self.T[i][1]]
				p3 = self.V[self.T[i][2]]
				c = get_centroid(p1, p2, p3) 
				color = black
				if (self.rClusters != []):
					if (self.rClusters[i]==0):
						color = purple
					if (self.rClusters[i]==1):
						color = blue
					if (self.rClusters[i]==2):
						color = white

				viewer.data().add_points(igl.eigen.MatrixXd(np.array([c])),  color)
			
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


