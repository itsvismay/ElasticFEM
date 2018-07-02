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

import random
import sys, os
import cProfile
sys.path.insert(0, os.getcwd()+"/../../libigl/python/")
import pyigl as igl
from Helpers import *
from Mesh import Mesh
np.set_printoptions(threshold="nan", linewidth=190, precision=8, formatter={'all': lambda x:'{:2.3f}'.format(x)})
from iglhelpers import *


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
		if self.mesh is not None:
			igl.writeDMAT(folder+"V.dmat", igl.eigen.MatrixXd(np.array(self.mesh.V)), True)
			# igl.writeDMAT(folder+"F.dmat", igl.eigen.MatrixXd(self.mesh.T), True)
			# igl.writeDMAT(folder+"FixV.dmat", igl.eigen.MatrixXd(np.array(self.mesh.fixed)), True)
			# igl.writeDMAT(folder+"MovV.dmat", igl.eigen.MatrixXd(np.array(self.mesh.mov)), True)
			
			# if self.mesh.Q is not None:
			# 	igl.writeDMAT(folder+"Modes.dmat", igl.eigen.MatrixXd(self.mesh.Q), True)
			# 	# igl.writeDMAT(folder+"Eigs.dmat", )
			# if self.mesh.r_element_cluster_map is not None:
			# 	igl.writeDMAT(folder+"Rclusters.dmat", igl.eigen.MatrixXi(self.mesh.r_element_cluster_map), True)
			# if self.mesh.s_handles_ind is not None:
			# 	igl.writeDMAT(folder+"SHandles.dmat", igl.eigen.MatrixXi(np.array(self.s_handles_ind, dtype='int32')), True)
		
	def read_mesh_setup(self, name=None):
		if name==None:
			print("Name can't be none.")
			exit()

	def createMesh(self):
		to_fix = self.Fix+self.Mov 
		to_mov = self.Mov
		
		self.mesh = Mesh([self.V, self.T, self.U], ito_fix = to_fix, ito_mov=to_mov, setup= True, red_g=True)
		self.mesh.u, self.uvec, self.eGu, self.UVECS = heat_method(self.mesh)
	
	def getMesh(self):
		if self.mesh is not None:
			return self.mesh 
		else:
			print("Mesh is None, sending default mesh.")
			self.createMesh()
			return self.mesh

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
			if(aaa == 83):
				self.save_mesh_setup(name="test")

			viewer.data().clear()
			if self.uvec is None:
				nV = self.V
			else:
				#3d Heat Gradient
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
				if (self.mesh is not None):
					color = igl.eigen.MatrixXd(np.array([randc[self.mesh.r_element_cluster_map[i]]]))

				viewer.data().add_points(igl.eigen.MatrixXd(np.array([c])),  color)
			
			if not self.mesh is None:
				CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.x0))
				for i in range(len(self.T)):
					C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
					U = np.multiply(self.mesh.getU(i), np.array([[0.03],[0.03]])) + C
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)

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


