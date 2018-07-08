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
import json


class Preprocessing:

	def __init__(self, _VT=None):
		self.middle_button_down = False
		if _VT is not None:
			self.V = _VT[0]
			self.T = _VT[1]
			self.U = np.zeros(len(self.T))
			self.Fix = get_max(self.V, a=1, eps=1e-2)		
			self.Mov = get_min(self.V, a=1, eps=1e-2)
			self.gi = 0
			self.mesh = None
			self.uvec = None
			self.eGu = None
			self.uClusters = []
			self.uClusterNum = -1
		self.UVECS = None

	def save_mesh_setup(self, name=None):
		#SAVE: V, T, Fixed points, Moving points,Eigs, EigV 
		# Muscle clusters, Rotation clusters, Skinning Handles, 
		# Maybe ARAP pre-processing info

		#AND json file with basic info
		# - mesh folder name (so ARAP pre-processing can potentially be saved)
		# - sizes, YM, poisson, muscle strengths, density
		if name is None:
			name = str(datetime.datetime.now())
			os.makedirs("./MeshSetups/"+name)
		folder = "./MeshSetups/"+name+"/"
		print("writing DMATS to "+folder)
		if self.mesh is not None:
			igl.writeDMAT(folder+"V.dmat", igl.eigen.MatrixXd(np.array(self.mesh.V)), True)
			igl.writeDMAT(folder+"F.dmat", igl.eigen.MatrixXi(self.mesh.T), True)
			igl.writeDMAT(folder+"FixV.dmat", igl.eigen.MatrixXi(np.array([self.mesh.fixed], dtype='int32')), True)
			igl.writeDMAT(folder+"MovV.dmat", igl.eigen.MatrixXi(np.array([self.mesh.mov], dtype='int32')), True)
			igl.writeDMAT(folder+"Uvec.dmat", igl.eigen.MatrixXd(np.array([self.mesh.u])), True)

			if self.mesh.Q is not None:
				igl.writeDMAT(folder+"Modes.dmat", igl.eigen.MatrixXd(self.mesh.Q.toarray()), True)
				# igl.writeDMAT(folder+"Eigs.dmat")
			if self.mesh.r_element_cluster_map is not None:
				igl.writeDMAT(folder+"Rclusters.dmat", igl.eigen.MatrixXi(self.mesh.r_element_cluster_map), True)
			if self.mesh.s_handles_ind is not None:
				igl.writeDMAT(folder+"SHandles.dmat", igl.eigen.MatrixXi(np.array([self.mesh.s_handles_ind], dtype='int32')), True)
			if self.mesh.u_clusters_element_map is not None:
				for i in range(len(self.mesh.u_clusters_element_map)):
					igl.writeDMAT(folder+"uClusters"+str(i)+".dmat", igl.eigen.MatrixXi(np.array([self.mesh.u_clusters_element_map[i]])), True)

			data = {"uClusters": len(self.mesh.u_clusters_element_map)}
			with open(folder+"params.json", 'w') as outfile:
				json.dump(data, outfile)

		print("Done writing DMAT")

	def read_mesh_setup(self, name=None):
		if name==None:
			print("Name can't be none.")
			exit()
		else:
			folder = "./MeshSetups/"+name+"/"
			jdata = json.load(open(folder+"params.json"))
			len_uClusters = jdata['uClusters']
			print("READING DMATs from "+folder)
			eV = igl.eigen.MatrixXd()
			eT = igl.eigen.MatrixXi()
			eu = igl.eigen.MatrixXd()
			eQ = igl.eigen.MatrixXd()
			efix = igl.eigen.MatrixXi()
			emov = igl.eigen.MatrixXi()
			es_ind = igl.eigen.MatrixXi()
			er_ind = igl.eigen.MatrixXi()
			u_ind = []
			eu_ind = igl.eigen.MatrixXi()

			igl.readDMAT(folder+"V.dmat", eV)
			igl.readDMAT(folder+"F.dmat", eT)
			igl.readDMAT(folder+"Uvec.dmat", eu)
			igl.readDMAT(folder+"Modes.dmat", eQ)
			igl.readDMAT(folder+"FixV.dmat", efix)
			igl.readDMAT(folder+"MovV.dmat", emov)
			igl.readDMAT(folder+"Rclusters.dmat", er_ind)
			igl.readDMAT(folder+"SHandles.dmat", es_ind)
			for i in range(len_uClusters):
				igl.readDMAT(folder+"uClusters"+str(i)+".dmat", eu_ind)
				u_ind.append(e2p(eu_ind)[0,:])

			self.mesh = Mesh(read_in = True)
			self.mesh.init_from_file(V=e2p(eV), 
								T=e2p(eT), 
								u=e2p(eu), 
								Q=e2p(eQ), 
								fix=e2p(efix), 
								mov=e2p(emov), 
								r_element_cluster_map=e2p(er_ind), 
								s_handles_ind=e2p(es_ind), 
								u_clusters_element_map= u_ind,
								modes_used=None)
			
			print("Done reading DMAT")

	def createMesh(self, modes=None):
		to_fix = self.Fix
		to_mov = self.Mov
		
		self.mesh = Mesh([self.V, self.T, self.U], ito_fix = to_fix, ito_mov=to_mov, read_in= False, modes_used=modes)
		self.mesh.u, self.uvec, self.eGu, self.UVECS = heat_method(self.mesh)
		CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.x0))
		self.uClusters = [[t for t in range(len(self.T)) if CAg[6*t+1]<=0.1],
							[t for t in range(len(self.T)) if CAg[6*t+1]>=0.9]]

		self.mesh.u_clusters_element_map = [np.array(list(e), dtype="int32") for e in self.uClusters]
		self.mesh.getGlobalF(updateU=True)

	def getMesh(self, name=None, modes_used=None):
		if name is not None:
			self.read_mesh_setup(name = name)
		else:
			self.createMesh(modes=modes_used)
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
			if btn==1:
				self.middle_button_down = False


		def mouse_down(viewer, btn, bbb):
			# Cast a ray in the view direction starting from the mouse position
			bc = igl.eigen.MatrixXd()
			fid = igl.eigen.MatrixXi(np.array([-1]))
			coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
			hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
			viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(self.V), igl.eigen.MatrixXi(self.T), fid, bc)
			ind = e2p(fid)[0][0]

			if hit and btn==0:
				# paint hit red
				self.Fix.append(self.T[ind][np.argmax(bc)])
				print("fix",self.T[ind][np.argmax(bc)])
				return True
			
			if hit and btn==2:
				# paint hit red
				self.Mov.append(self.T[ind][np.argmax(bc)])
				print("mov",self.T[ind][np.argmax(bc)])
				return True

			if hit and btn==1:
				self.middle_button_down = True
				self.uClusters.append(set())
				self.uClusterNum += 1
				return True
			
			return False

		def mouse_move(viewer, mx, my):
			if self.middle_button_down:
				# Cast a ray in the view direction starting from the mouse position
				bc = igl.eigen.MatrixXd()
				fid = igl.eigen.MatrixXi(np.array([-1]))
				coord = igl.eigen.MatrixXd([viewer.current_mouse_x, viewer.core.viewport[3] - viewer.current_mouse_y])
				hit = igl.unproject_onto_mesh(coord, viewer.core.view * viewer.core.model,
				viewer.core.proj, viewer.core.viewport, igl.eigen.MatrixXd(self.V), igl.eigen.MatrixXi(self.T), fid, bc)
				ind = e2p(fid)[0][0]
				self.uClusters[self.uClusterNum].add(ind)
				return True


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

			if self.mesh is not None:
				Colors = np.ones(self.mesh.T.shape)
				if (aaa==82):
					for i in range(len(self.mesh.T)): 
						color = black
						Colors[i,:] = randc[self.mesh.r_element_cluster_map[i]]
				elif(aaa==67):
					for i in range(len(self.mesh.u_clusters_element_map)):
						for j in range(len(self.mesh.u_clusters_element_map[i])):
							k = self.mesh.u_clusters_element_map[i][j]
							Colors[k,:] = randc[i]
				
				viewer.data().set_colors(igl.eigen.MatrixXd(np.array(Colors)))

			if not self.mesh is None:
				CAg = self.mesh.getC().dot(self.mesh.getA().dot(self.mesh.x0))
				for i in range(len(self.T)):
					C = np.matrix([CAg[6*i:6*i+2],CAg[6*i:6*i+2]])
					U = np.multiply(self.mesh.getU(i), np.array([[0.03],[0.03]])) + C
					viewer.data().add_edges(igl.eigen.MatrixXd(C[0,:]), igl.eigen.MatrixXd(U[0,:]), black)

		def pre_draw(viewer):
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
		viewer.callback_mouse_move = mouse_move
		viewer.callback_pre_draw = pre_draw
		viewer.core.is_animating = False
		viewer.launch()


